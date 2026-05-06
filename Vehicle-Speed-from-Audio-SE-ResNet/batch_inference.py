#!/usr/bin/env python3
"""
Standalone cross-dataset ensemble inference (Combined RMSE and MAE).
Evaluates each dataset's audio against each checkpoint family and prints 3x3 grids
per vehicle, plus summary averages. On the MixedData_model checkpoint column: RealData and
SimulatedData rows use that row's audio only; the ExtendedSimulatedData row uses
RealData + ExtendedSimulatedData combined (MixedData).
Other columns use the row's data source only.

Filenames must end with _{speed}.wav (speed is the last integer before .wav). By default, vehicles are
listed using union across datasets (--vehicles union). Use --vehicles intersection only if you want
vehicles present in every dataset folder.
Results go to the terminal and to a text file.

Place checkpoint folders next to the data roots (default layout):
  <repo>/RealData_model/fold_{1..10}_best.keras
  <repo>/SimulatedData_model/fold_{1..10}_best.keras
  <repo>/MixedData_model/fold_{1..10}_best.keras

Run:
  python batch_inference.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import re
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _model_package_has_src(root: str) -> bool:
    return os.path.isfile(os.path.join(root, "src", "config.py"))


def _find_model_package_root(explicit: Optional[str]) -> str:
    """Locate Vehicle-Speed-from-Audio-SE-ResNet (the folder that contains src/)."""
    candidates: List[str] = []
    if explicit:
        candidates.append(os.path.abspath(explicit))
    for env_key in ("SE_RESNET_ROOT", "VEHICLE_SPEED_SE_RESNET_ROOT"):
        env = os.environ.get(env_key)
        if env:
            candidates.append(os.path.abspath(env.strip()))
    candidates.extend(
        [
            os.path.join(_SCRIPT_DIR, "Vehicle-Speed-from-Audio-SE-ResNet"),
            os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "Vehicle-Speed-from-Audio-SE-ResNet")),
            os.path.normpath(
                os.path.join(
                    _SCRIPT_DIR,
                    "..",
                    "DopplerNet_validation_real",
                    "Vehicle-Speed-from-Audio-SE-ResNet",
                )
            ),
        ]
    )
    seen: set[str] = set()
    for c in candidates:
        p = os.path.normpath(os.path.abspath(c))
        if p in seen:
            continue
        seen.add(p)
        if _model_package_has_src(p):
            return p
    tried = "\n".join(f"  - {c}" for c in candidates)
    raise ImportError(
        "Could not import `src` — the SE-ResNet code package is missing.\n"
        "Copy the folder `Vehicle-Speed-from-Audio-SE-ResNet/` (with `src/` inside) next to this script,\n"
        "or set SE_RESNET_ROOT to that folder's path, or run:\n"
        "  python grid.py --model_root /path/to/Vehicle-Speed-from-Audio-SE-ResNet\n\n"
        "Tried:\n"
        f"{tried}"
    )


# Parse model_root before importing src (works with full argparse later via parse_known_args).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--model_root", type=str, default=None)
_pre_args, _ = _pre.parse_known_args()

_MODEL_ROOT = _find_model_package_root(_pre_args.model_root)
if _MODEL_ROOT not in sys.path:
    sys.path.insert(0, _MODEL_ROOT)

_REPO_ROOT = _SCRIPT_DIR

# Before importing TF: hide C++ INFO/WARN; after import: hide retracing / absl noise during predict loops.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=".*(tf\\.function retracing|tf.function retracing|retracing is expensive).*",
)

from src.config import Config  # noqa: E402
from src.data_loader import get_tf_dataset  # noqa: E402
from src.models import build_se_resnet  # noqa: E402
from src.utils import calculate_global_stats  # noqa: E402

# Ordered dataset keys (grid rows).
DATASET_KEYS = ("RealData", "SimulatedData", "MixedData")

# Ordered checkpoint keys (grid columns), parallel to DATASET_KEYS.
CKPT_KEYS = ("RealData_model", "SimulatedData_model", "MixedData_model")

# Backward-compat alias: grid_mae.py imports DATA_KEYS from this module.
DATA_KEYS = DATASET_KEYS


def _alnum_lower(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _canonical_vehicle_name(name: str) -> str:
    """
    Map known folder-name variants to a single canonical id so Real/Simulated/ExtendedSimulated line up.
    Peugeot 307 and 3008: handles legacy "Peuguot" typo (case-insensitive on input).
    Nissan Qashqai: folder variants (e.g. NissanQashQai vs NissanQashqai) merge to one id.
    """
    k = name.strip()
    low = k.lower()
    if low in ("peuguot307", "peugeot307"):
        return "Peugeot307"
    if low in ("peuguot3008", "peugeot3008"):
        return "Peugeot3008"
    if _alnum_lower(k) == "nissanqashqai":
        return "NissanQashqai"
    return k


def _speed_int_from_wav_basename(basename: str) -> Optional[int]:
    """
    Parse trailing _{speed}.wav with integer or decimal speed (SimulatedData uses e.g. _100.0.wav).
    Returns rounded km/h as int for label consistency with Real/ExtendedSimulated integer filenames.
    """
    m = re.search(r"_(\d+(?:\.\d+)?)\.wav$", basename, re.IGNORECASE)
    if not m:
        return None
    return int(round(float(m.group(1))))


def get_all_audio_paths_and_labels_relaxed(data_root: str) -> Tuple[List[str], np.ndarray]:
    """
    VS13-style layout with permissive filenames: basename ending in _{speed}.wav
    (e.g. Mercedes_AMG550_50.wav, or SimulatedData-style KiaSportage_100.0.wav).
    """
    all_paths: List[str] = []
    all_speeds: List[int] = []

    try:
        vehicle_folders = [
            d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
        ]
    except OSError:
        return [], np.array([])

    for vehicle_folder in vehicle_folders:
        vehicle_path = os.path.join(data_root, vehicle_folder)
        split_file_path = os.path.join(vehicle_path, "Train_valid_split.txt")
        if not os.path.exists(split_file_path):
            continue

        with open(split_file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    base_name = parts[0]
                    for ext in (".wav", ".WAV"):
                        wav_file = os.path.join(vehicle_path, base_name + ext)
                        if os.path.exists(wav_file):
                            break
                    else:
                        continue
                    bn = os.path.basename(wav_file)
                    spd = _speed_int_from_wav_basename(bn)
                    if spd is not None:
                        all_paths.append(wav_file)
                        all_speeds.append(spd)

    return all_paths, np.array(all_speeds, dtype=np.int64)


def collect_vehicle_names(
    data_roots: Dict[str, str], mode: str = "union"
) -> Tuple[List[str], Dict[str, Set[str]]]:
    """
    mode='union': include every vehicle folder that exists in any dataset (default).
    mode='intersection': only vehicles present in all three roots (often very small).
    Returns (sorted names, per-key vehicle sets for diagnostics).
    """
    per_key: Dict[str, Set[str]] = {}
    for key in DATASET_KEYS:
        root = data_roots[key]
        if not os.path.isdir(root):
            per_key[key] = set()
            continue
        local: Set[str] = set()
        for d in os.listdir(root):
            vp = os.path.join(root, d)
            if os.path.isdir(vp) and os.path.isfile(os.path.join(vp, "Train_valid_split.txt")):
                local.add(d)
        per_key[key] = local

    if mode == "intersection":
        # Intersect on canonical names so Peugeot 307/3008 variants count as the same vehicle
        sets_loaded: List[Set[str]] = []
        for k in DATASET_KEYS:
            if not os.path.isdir(data_roots[k]):
                continue
            sets_loaded.append({_canonical_vehicle_name(d) for d in per_key[k]})
        if not sets_loaded:
            return [], per_key
        names = sets_loaded[0].copy()
        for s in sets_loaded[1:]:
            names &= s
        return sorted(names), per_key

    u: Set[str] = set()
    for s in per_key.values():
        for d in s:
            u.add(_canonical_vehicle_name(d))
    return sorted(u), per_key


def _resolve_mixed_model_ckpt_dir(root: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    a = os.path.join(root, "MixedData_model")
    # Tolerate a common typo variant
    b = os.path.join(root, "MixedDataModel")
    if os.path.isdir(a):
        return a
    if os.path.isdir(b):
        return b
    return a


def _vehicle_label(path: str) -> str:
    return _canonical_vehicle_name(os.path.basename(os.path.dirname(path)))


def ensemble_predict_metrics(
    paths: np.ndarray,
    speeds: np.ndarray,
    stats: dict,
    weights_dir: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Mean ensemble prediction RMSE and MAE (km/h); (None, None) if no samples or no usable weights."""
    if len(paths) == 0:
        return None, None

    val_ds = get_tf_dataset(paths, speeds, stats, is_training=False)

    n_frames = int(np.ceil(Config.AUDIO_LENGTH_SAMPLES / Config.HOP_LENGTH))
    input_shape = (Config.N_MELS, n_frames, 1)

    fold_predictions: List[np.ndarray] = []

    for fold in range(1, Config.N_FOLDS + 1):
        weight_path = os.path.join(weights_dir, f"fold_{fold}_best.keras")
        if not os.path.isfile(weight_path):
            continue
        model = build_se_resnet(input_shape)
        model.load_weights(weight_path)
        preds = model.predict(val_ds, verbose=0)
        fold_predictions.append(preds.flatten())
        del model
        tf.keras.backend.clear_session()

    if not fold_predictions:
        return None, None

    stacked = np.array(fold_predictions)
    ensemble = np.mean(stacked, axis=0)
    
    diffs = speeds.astype(np.float64) - ensemble
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    mae = float(np.mean(np.abs(diffs)))
    
    return rmse, mae


def vehicle_sample_counts(
    vehicle: str,
    dataset_paths: Dict[str, List[str]],
    dataset_speeds: Dict[str, np.ndarray],
) -> Tuple[int, int, int, int]:
    """Returns (n_real, n_simulated, n_extended_sim, n_mixed) clip counts for canonical vehicle."""
    n_real = 0
    n_simulated = 0
    n_extended_sim = 0
    if "RealData" in dataset_paths:
        pv, _ = mask_vehicle(dataset_paths["RealData"], dataset_speeds["RealData"], vehicle)
        n_real = int(len(pv))
    if "SimulatedData" in dataset_paths:
        pv, _ = mask_vehicle(dataset_paths["SimulatedData"], dataset_speeds["SimulatedData"], vehicle)
        n_simulated = int(len(pv))
    if "MixedData" in dataset_paths:
        pv, _ = mask_vehicle(dataset_paths["MixedData"], dataset_speeds["MixedData"], vehicle)
        n_extended_sim = int(len(pv))
    rs_paths, _, rs_stats = real_sim_paths_and_stats_for_vehicle(
        dataset_paths, dataset_speeds, vehicle, quiet_stats=True
    )
    n_mixed = int(len(rs_paths)) if rs_stats is not None else 0
    return n_real, n_simulated, n_extended_sim, n_mixed


def mask_vehicle(
    paths: List[str], speeds: np.ndarray, vehicle: str
) -> Tuple[np.ndarray, np.ndarray]:
    want = _canonical_vehicle_name(vehicle)
    idx = [i for i, p in enumerate(paths) if _vehicle_label(p) == want]
    if not idx:
        return np.array([]), np.array([])
    paths_np = np.array([paths[i] for i in idx])
    sp = speeds[np.array(idx, dtype=np.int64)]
    return paths_np, sp


def real_sim_paths_and_stats_for_vehicle(
    dataset_paths: Dict[str, List[str]],
    dataset_speeds: Dict[str, np.ndarray],
    vehicle: str,
    quiet_stats: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """Merge RealData + ExtendedSimulatedData samples for one vehicle (MixedData); stats fit that union."""
    parts_p: List[np.ndarray] = []
    parts_s: List[np.ndarray] = []
    for key in ("RealData", "ExtendedSimulatedData"):
        if key not in dataset_paths:
            continue
        pv, sv = mask_vehicle(dataset_paths[key], dataset_speeds[key], vehicle)
        if len(pv) > 0:
            parts_p.append(pv)
            parts_s.append(sv)
    if not parts_p:
        return np.array([]), np.array([]), None
    paths_np = np.concatenate(parts_p) if len(parts_p) > 1 else parts_p[0]
    speeds_np = np.concatenate(parts_s) if len(parts_s) > 1 else parts_s[0]
    plist = paths_np.tolist() if isinstance(paths_np, np.ndarray) else list(paths_np)
    buf = io.StringIO()
    if quiet_stats:
        with contextlib.redirect_stdout(buf):
            stats = calculate_global_stats(plist)
    else:
        stats = calculate_global_stats(plist)
    return paths_np, speeds_np, stats


def format_grid_cell(val: Optional[float]) -> str:
    if val is None:
        return "   N/A   "
    return f"{val:9.4f}"


def run_evaluation(
    data_roots: Dict[str, str],
    ckpt_roots: Dict[str, str],
    quiet_stats: bool,
    output_path: str,
    vehicle_mode: str = "union",
) -> None:
    lines: List[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 72)
    emit("Cross-dataset inference — Combined RMSE and MAE Reports")
    emit(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    emit("Rows: data source. Columns: checkpoint family.")
    emit("=" * 72)

    # Load full datasets & stats once per data source
    dataset_stats: Dict[str, dict] = {}
    dataset_paths: Dict[str, List[str]] = {}
    dataset_speeds: Dict[str, np.ndarray] = {}

    for key in DATASET_KEYS:
        root = data_roots[key]
        if not os.path.isdir(root):
            emit(f"[WARN] Missing data directory: {root}")
            continue
        emit(f"\nLoading index: {key} -> {root}")
        paths, speeds = get_all_audio_paths_and_labels_relaxed(root)
        if len(paths) == 0:
            emit(f"[WARN] No audio samples under {root}")
            continue
        dataset_paths[key] = paths
        dataset_speeds[key] = speeds

        buf = io.StringIO()
        if quiet_stats:
            with contextlib.redirect_stdout(buf):
                stats = calculate_global_stats(paths)
        else:
            stats = calculate_global_stats(paths)
        dataset_stats[key] = stats

    vehicles, per_key_vehicles = collect_vehicle_names(data_roots, mode=vehicle_mode)
    
    emit("\n" + "=" * 72)
    emit("Per-vehicle sample counts")
    emit("=" * 72)
    hdr = f"{'Vehicle':<22} {'Real':>8} {'Simulated':>12} {'ExtSimulated':>14} {'Mixed':>8}"
    emit(hdr)
    emit("-" * 72)
    for vehicle in vehicles:
        nr, ns, nes, nm = vehicle_sample_counts(vehicle, dataset_paths, dataset_speeds)
        emit(f"{vehicle:<22} {nr:>8} {ns:>12} {nes:>14} {nm:>8}")

    col_headers = list(CKPT_KEYS)
    row_labels = list(DATASET_KEYS)

    # Store all grid results: vehicle -> metric -> grid[3][3]
    results: Dict[str, Dict[str, List[List[Optional[float]]]]] = {}
    
    emit("\nRunning inference...")
    for vehicle in vehicles:
        rs_paths, rs_speeds, rs_stats = real_sim_paths_and_stats_for_vehicle(
            dataset_paths, dataset_speeds, vehicle, quiet_stats
        )

        rmse_grid: List[List[Optional[float]]] = []
        mae_grid: List[List[Optional[float]]] = []
        
        for r, row_key in enumerate(row_labels):
            rmse_row: List[Optional[float]] = []
            mae_row: List[Optional[float]] = []
            
            if row_key not in dataset_paths:
                rmse_row = [None] * 3
                mae_row = [None] * 3
            elif row_key == "MixedData":
                if rs_stats is None:
                    rmse_row = [None] * 3
                    mae_row = [None] * 3
                else:
                    for col_key in CKPT_KEYS:
                        wdir = ckpt_roots[col_key]
                        if not os.path.isdir(wdir):
                            rmse_row.append(None)
                            mae_row.append(None)
                        else:
                            rmse, mae = ensemble_predict_metrics(rs_paths, rs_speeds, rs_stats, wdir)
                            rmse_row.append(rmse)
                            mae_row.append(mae)
            else:
                p_full = dataset_paths[row_key]
                s_full = dataset_speeds[row_key]
                paths_v, speeds_v = mask_vehicle(p_full, s_full, vehicle)
                for col_key in CKPT_KEYS:
                    wdir = ckpt_roots[col_key]
                    if not os.path.isdir(wdir):
                        rmse_row.append(None)
                        mae_row.append(None)
                    else:
                        rmse, mae = ensemble_predict_metrics(paths_v, speeds_v, dataset_stats[row_key], wdir)
                        rmse_row.append(rmse)
                        mae_row.append(mae)
            
            rmse_grid.append(rmse_row)
            mae_grid.append(mae_row)
        
        results[vehicle] = {"RMSE": rmse_grid, "MAE": mae_grid}

    # Helper to print metric block
    def emit_metric_block(metric_name: str):
        emit("\n" + "=" * 72)
        emit(f"GRID RESULTS: {metric_name} (km/h)")
        emit("=" * 72)
        
        all_vals: List[float] = []
        cell_samples: List[List[List[float]]] = [[[] for _ in range(3)] for _ in range(3)]
        per_vehicle_means: List[float] = []

        for vehicle in vehicles:
            emit("\n" + "-" * 72)
            emit(f"Vehicle: {vehicle} ({metric_name})")
            emit("-" * 72)
            
            grid = results[vehicle][metric_name]
            header = f"{'': <14}" + "".join(f"{h: >20}" for h in col_headers)
            emit(header)
            for r, row_key in enumerate(row_labels):
                label = f"{row_key: <14}"
                cells = "".join(format_grid_cell(grid[r][c]) for c in range(3))
                emit(label + cells)
                
                for c in range(3):
                    v = grid[r][c]
                    if v is not None:
                        cell_samples[r][c].append(v)
                        all_vals.append(v)

            flat = [x for row in grid for x in row if x is not None]
            if flat:
                vm = float(np.mean(flat))
                per_vehicle_means.append(vm)
                emit(f"\n  Mean {metric_name} (this vehicle, valid cells): {vm:.4f} km/h")
            else:
                emit(f"\n  Mean {metric_name} (this vehicle): N/A")

        emit("\n" + "=" * 72)
        emit(f"Summary — mean {metric_name} per grid cell (over vehicles)")
        emit("=" * 72)
        emit(
            f"Each cell is the arithmetic mean of that cell's {metric_name} across all vehicles "
            "that had a numeric value there (N/A if no vehicle contributed)."
        )
        
        cell_means: List[List[Optional[float]]] = []
        for r in range(3):
            row_m: List[Optional[float]] = []
            for c in range(3):
                xs = cell_samples[r][c]
                row_m.append(float(np.mean(xs)) if xs else None)
            cell_means.append(row_m)

        emit(header)
        for r, row_key in enumerate(row_labels):
            label = f"{row_key: <14}"
            cells = "".join(format_grid_cell(cell_means[r][c]) for c in range(3))
            emit(label + cells)

        flat_cell_means = [x for row in cell_means for x in row if x is not None]
        if flat_cell_means:
            emit(f"\n  Final average (mean of cell means): {float(np.mean(flat_cell_means)):.4f} km/h")
        if per_vehicle_means:
            emit(f"  Average of per-vehicle mean {metric_name}: {np.mean(per_vehicle_means):.4f} km/h")
        if all_vals:
            emit(f"  Global mean {metric_name} (all valid entries): {np.mean(all_vals):.4f} km/h")

    # Output RMSE then MAE
    emit_metric_block("RMSE")
    emit_metric_block("MAE")

    emit("\n" + "=" * 72)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"Saved report to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3×3 cross-dataset RMSE and MAE grids per vehicle (ensemble inference)."
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=_REPO_ROOT,
        help="Repository root (contains RealData, SimulatedData, ExtendedSimulatedData).",
    )
    parser.add_argument(
        "--model_root",
        type=str,
        default=None,
        help="Path to Vehicle-Speed-from-Audio-SE-ResNet (folder containing src/). "
        "Also settable via SE_RESNET_ROOT. Required if that folder is not next to this script.",
    )
    # Data source overrides
    parser.add_argument("--real_data", type=str, default=None,
                        help="Override path to RealData directory.")
    parser.add_argument("--simulated_data", type=str, default=None,
                        help="Override path to SimulatedData directory.")
    parser.add_argument("--extended_simulated_data", type=str, default=None,
                        help="Override path to ExtendedSimulatedData directory.")
    # Checkpoint dir overrides
    parser.add_argument("--real_model_ckpt", type=str, default=None,
                        help="Override path to RealData_model checkpoint directory.")
    parser.add_argument("--simulated_model_ckpt", type=str, default=None,
                        help="Override path to SimulatedData_model checkpoint directory.")
    parser.add_argument("--mixed_model_ckpt", type=str, default=None,
                        help="Override path to MixedData_model checkpoint directory.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .txt path (default: cross_inference_report_<timestamp>.txt in repo root).",
    )
    parser.add_argument(
        "--verbose_stats",
        action="store_true",
        help="Print per-file stats progress from calculate_global_stats (noisy).",
    )
    parser.add_argument(
        "--vehicles",
        type=str,
        choices=("union", "intersection"),
        default="union",
        help="union: one grid per vehicle that appears in any dataset (N/A where no clips). "
        "intersection: only vehicles that exist in RealData, SimulatedData, and ExtendedSimulatedData (often only a few).",
    )
    args = parser.parse_args()
    root = os.path.abspath(args.repo_root)

    data_roots = {
        "RealData": args.real_data or os.path.normpath(os.path.join(root, "..", "Datasets", "RealData")),
        "SimulatedData": args.simulated_data or os.path.normpath(os.path.join(root, "..", "Datasets", "SimulatedData")),
        "MixedData": args.extended_simulated_data or os.path.normpath(os.path.join(root, "..", "Datasets", "ExtendedSimulatedData")),
    }
    ckpt_roots = {
        "RealData_model": args.real_model_ckpt or os.path.join(root, "checkpoints", "RealData_model"),
        "SimulatedData_model": args.simulated_model_ckpt or os.path.join(root, "checkpoints", "SimulatedData_model"),
        "MixedData_model": _resolve_mixed_model_ckpt_dir(os.path.join(root, "checkpoints"), args.mixed_model_ckpt),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_inference_results_dir = os.path.join(root, "results", "batch_inference_results")
    os.makedirs(batch_inference_results_dir, exist_ok=True)
    out = args.output or os.path.join(batch_inference_results_dir, f"cross_inference_report_{ts}.txt")

    run_evaluation(
        data_roots,
        ckpt_roots,
        quiet_stats=not args.verbose_stats,
        output_path=out,
        vehicle_mode=args.vehicles,
    )


if __name__ == "__main__":
    main()
