#!/usr/bin/env python3
"""
Strict leakage-free speed estimation evaluation (offline / rebuttal).

Protocols
---------
1) legacy_ensemble : full-set mean of all 10 folds (matches released batch_inference;
                     NOT leakage-free for Mixed→Real). Reported only for comparison.
2) oof_retrain     : fresh 10-fold CV; each clip scored ONLY by its held-out fold.
                     Mel stats fit on that fold's train split only.
3) lovo            : leave-one-vehicle-out on real recordings. For held-out vehicle v,
                     train models with ZERO real clips of v (and ZERO ExtendedSim of v
                     for Mixed). Mel stats from train only. Hard assert no path overlap.

Does not modify hosted checkpoints used by the submission; writes local results only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split

from src.config import Config
from src.models import build_se_resnet

try:
    import librosa
except ImportError as e:
    raise SystemExit(f"librosa required: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alnum_lower(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def canonical_vehicle(name: str) -> str:
    k = name.strip()
    low = k.lower()
    if low in ("peuguot307", "peugeot307"):
        return "Peugeot307"
    if low in ("peuguot3008", "peugeot3008"):
        return "Peugeot3008"
    if _alnum_lower(k) == "nissanqashqai":
        return "NissanQashqai"
    return k


def speed_from_basename(basename: str) -> Optional[int]:
    m = re.search(r"_(\d+(?:\.\d+)?)\.wav$", basename, re.IGNORECASE)
    if not m:
        return None
    return int(round(float(m.group(1))))


def list_clips(data_root: str) -> List[dict]:
    """Load all clips referenced by Train_valid_split.txt (both train and valid tags)."""
    clips: List[dict] = []
    if not os.path.isdir(data_root):
        return clips
    for vehicle_folder in sorted(os.listdir(data_root)):
        vehicle_path = os.path.join(data_root, vehicle_folder)
        if not os.path.isdir(vehicle_path):
            continue
        split_path = os.path.join(vehicle_path, "Train_valid_split.txt")
        if not os.path.isfile(split_path):
            continue
        veh = canonical_vehicle(vehicle_folder)
        with open(split_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                base = parts[0]
                wav = None
                for ext in (".wav", ".WAV"):
                    cand = os.path.join(vehicle_path, base + ext)
                    if os.path.isfile(cand):
                        wav = cand
                        break
                if wav is None:
                    continue
                spd = speed_from_basename(os.path.basename(wav))
                if spd is None:
                    continue
                clips.append(
                    {
                        "path": os.path.normpath(wav),
                        "speed": float(spd),
                        "vehicle": veh,
                        "stem": base,
                    }
                )
    # Stable order for reproducibility
    clips.sort(key=lambda c: (c["vehicle"], c["speed"], c["path"]))
    return clips


def assert_disjoint(train_paths: Sequence[str], test_paths: Sequence[str], label: str) -> None:
    tr = {os.path.normcase(os.path.abspath(p)) for p in train_paths}
    te = {os.path.normcase(os.path.abspath(p)) for p in test_paths}
    overlap = tr & te
    if overlap:
        examples = list(overlap)[:5]
        raise RuntimeError(f"LEAKAGE DETECTED [{label}]: {len(overlap)} overlapping paths e.g. {examples}")


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def mel_shape() -> Tuple[int, int, int]:
    n_frames = int(np.ceil(Config.AUDIO_LENGTH_SAMPLES / Config.HOP_LENGTH))
    return (Config.N_MELS, n_frames, 1)


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)
    if len(audio) > Config.AUDIO_LENGTH_SAMPLES:
        audio = audio[: Config.AUDIO_LENGTH_SAMPLES]
    else:
        audio = np.pad(audio, (0, Config.AUDIO_LENGTH_SAMPLES - len(audio)), "constant")
    return audio.astype(np.float32)


def audio_to_mel_db(audio: np.ndarray) -> np.ndarray:
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def augment_audio(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = audio.copy()
    if rng.random() > Config.AUGMENT_PROB:
        return x
    gain_db = float(rng.uniform(*Config.GAIN_DB))
    x *= 10.0 ** (gain_db / 20.0)
    snr_db = float(rng.uniform(*Config.NOISE_SNR_DB))
    power = float(np.sum(x ** 2) / len(x))
    if power > 1e-6:
        noise_power = power / (10 ** (snr_db / 10))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=len(x)).astype(np.float32)
        x = x + noise
    return x


@dataclass
class MelStats:
    mean: np.ndarray  # (n_mels, 1)
    std: np.ndarray

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def fit_stats_from_mels(mels: Sequence[np.ndarray]) -> MelStats:
    """Fit z-score stats on train mels only (no augmentation)."""
    sums = np.zeros((Config.N_MELS, 1), dtype=np.float64)
    sumsq = np.zeros((Config.N_MELS, 1), dtype=np.float64)
    frames = 0
    for mel in mels:
        sums += np.sum(mel, axis=1, keepdims=True)
        sumsq += np.sum(mel.astype(np.float64) ** 2, axis=1, keepdims=True)
        frames += mel.shape[1]
    mean = (sums / max(frames, 1)).astype(np.float32)
    var = sumsq / max(frames, 1) - mean.astype(np.float64) ** 2
    std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    std[std < 1e-8] = 1e-8
    return MelStats(mean=mean, std=std)


def normalize_mel(mel: np.ndarray, stats: MelStats) -> np.ndarray:
    return ((mel - stats.mean) / stats.std).astype(np.float32)[..., np.newaxis]


class AudioCache:
    """Cache waveforms keyed by (abspath, sample_rate)."""

    def __init__(self) -> None:
        self._wav: Dict[Tuple[str, int], np.ndarray] = {}
        self._mel_clean: Dict[Tuple[str, int], np.ndarray] = {}

    def get_audio(self, path: str) -> np.ndarray:
        key = (os.path.normcase(os.path.abspath(path)), Config.SAMPLE_RATE)
        if key not in self._wav:
            self._wav[key] = load_audio(path)
        return self._wav[key]

    def get_clean_mel(self, path: str) -> np.ndarray:
        key = (os.path.normcase(os.path.abspath(path)), Config.SAMPLE_RATE)
        if key not in self._mel_clean:
            self._mel_clean[key] = audio_to_mel_db(self.get_audio(path))
        return self._mel_clean[key]


def build_xy(
    paths: Sequence[str],
    speeds: Sequence[float],
    cache: AudioCache,
    stats: MelStats,
    training: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = []
    for p in paths:
        if training:
            audio = augment_audio(cache.get_audio(p), rng)
            mel = audio_to_mel_db(audio)
        else:
            mel = cache.get_clean_mel(p)
        xs.append(normalize_mel(mel, stats))
    x = np.stack(xs, axis=0)
    y = np.asarray(speeds, dtype=np.float32)
    return x, y


def compile_model(n_train: int) -> tf.keras.Model:
    model = build_se_resnet(mel_shape())
    steps = max(n_train // Config.BATCH_SIZE, 1)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=Config.INIT_LR,
        decay_steps=steps * Config.EPOCHS,
        alpha=0.0,
    )
    opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=Config.WEIGHT_DECAY)
    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def train_model(
    train_paths: List[str],
    train_speeds: np.ndarray,
    cache: AudioCache,
    seed: int,
    val_frac: float = 0.15,
) -> Tuple[tf.keras.Model, MelStats, dict]:
    """Train on train_* only; internal val carved from train only; stats train-only."""
    assert len(train_paths) >= 4, "Need enough train clips"
    idx = np.arange(len(train_paths))
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_frac, random_state=seed, shuffle=True
    )
    # Ensure at least 1 val
    if len(va_idx) == 0:
        va_idx = tr_idx[-1:]
        tr_idx = tr_idx[:-1]

    tr_paths = [train_paths[i] for i in tr_idx]
    va_paths = [train_paths[i] for i in va_idx]
    tr_speeds = train_speeds[tr_idx]
    va_speeds = train_speeds[va_idx]

    assert_disjoint(tr_paths, va_paths, "internal_train_val")

    # Stats from TRAIN split only (no val, no test)
    train_mels = [cache.get_clean_mel(p) for p in tr_paths]
    stats = fit_stats_from_mels(train_mels)

    x_tr, y_tr = build_xy(tr_paths, tr_speeds, cache, stats, training=True, seed=seed)
    x_va, y_va = build_xy(va_paths, va_speeds, cache, stats, training=False, seed=seed)

    tf.keras.backend.clear_session()
    set_global_seeds(seed)
    model = compile_model(len(tr_paths))
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=Config.PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
    ]
    hist = model.fit(
        x_tr,
        y_tr,
        validation_data=(x_va, y_va),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=cb,
        verbose=2,
    )
    meta = {
        "n_train": int(len(tr_paths)),
        "n_val": int(len(va_paths)),
        "epochs_ran": int(len(hist.history["loss"])),
        "best_val_rmse": float(np.min(hist.history.get("val_rmse", [np.nan]))),
    }
    return model, stats, meta


def predict_rmse(
    model: tf.keras.Model,
    paths: List[str],
    speeds: np.ndarray,
    cache: AudioCache,
    stats: MelStats,
) -> Tuple[float, float, np.ndarray]:
    x, y = build_xy(paths, speeds, cache, stats, training=False, seed=0)
    pred = model.predict(x, batch_size=Config.BATCH_SIZE, verbose=0).reshape(-1)
    diff = y.astype(np.float64) - pred.astype(np.float64)
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    return rmse, mae, pred


def rmse_mae(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    diff = y.astype(np.float64) - pred.astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2))), float(np.mean(np.abs(diff)))


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

def set_sr(sample_rate: int, label: str) -> None:
    Config.SAMPLE_RATE = int(sample_rate)
    Config.AUDIO_LENGTH_SAMPLES = Config.SAMPLE_RATE * Config.DURATION_SECONDS
    print(f"[INFO] SR={Config.SAMPLE_RATE} Hz ({label})")


def detect_checkpoint_input_frames(weights_dir: str) -> Optional[int]:
    """Read spatial width from first available fold .keras (None, H, W, C)."""
    for fold in range(1, Config.N_FOLDS + 1):
        wp = os.path.join(weights_dir, f"fold_{fold}_best.keras")
        if not os.path.isfile(wp):
            continue
        model = tf.keras.models.load_model(wp, compile=False)
        shape = model.input_shape
        del model
        tf.keras.backend.clear_session()
        if shape and len(shape) >= 3 and shape[2] is not None:
            return int(shape[2])
    return None


def sr_for_n_frames(n_frames: int) -> int:
    """Invert n_frames ~= ceil(sr * duration / hop). Prefer standard rates."""
    for sr in (16000, 22050, 44100):
        if int(np.ceil(sr * Config.DURATION_SECONDS / Config.HOP_LENGTH)) == n_frames:
            return sr
    # fallback approximate
    return int(round(n_frames * Config.HOP_LENGTH / Config.DURATION_SECONDS))


def run_legacy_ensemble(
    clips: List[dict],
    weights_dir: str,
    cache: AudioCache,
    sample_rate: int,
    label: str,
) -> dict:
    """Reproduce full-set 10-fold ensemble (NOT leakage-free when train includes eval)."""
    set_sr(sample_rate, label)
    paths = [c["path"] for c in clips]
    speeds = np.array([c["speed"] for c in clips], dtype=np.float64)
    expect = int(np.ceil(Config.AUDIO_LENGTH_SAMPLES / Config.HOP_LENGTH))

    mels = [cache.get_clean_mel(p) for p in paths]
    stats = fit_stats_from_mels(mels)
    x, _ = build_xy(paths, speeds, cache, stats, training=False, seed=0)

    fold_preds = []
    used = []
    for fold in range(1, Config.N_FOLDS + 1):
        wp = os.path.join(weights_dir, f"fold_{fold}_best.keras")
        if not os.path.isfile(wp):
            continue
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(wp, compile=False)
        in_frames = int(model.input_shape[2])
        if in_frames != expect:
            raise RuntimeError(
                f"Checkpoint frame width {in_frames} != Config mel width {expect} "
                f"(SR={Config.SAMPLE_RATE}). Refusing silent mismatch."
            )
        pred = model.predict(x, batch_size=Config.BATCH_SIZE, verbose=0).reshape(-1)
        fold_preds.append(pred)
        used.append(fold)
        del model
    if not fold_preds:
        raise FileNotFoundError(f"No fold weights in {weights_dir}")

    ens = np.mean(np.stack(fold_preds, axis=0), axis=0)
    rmse, mae = rmse_mae(speeds, ens)
    vehicles = [c["vehicle"] for c in clips]
    per_v = []
    for v in sorted(set(vehicles)):
        m = np.array([vv == v for vv in vehicles])
        r, a = rmse_mae(speeds[m], ens[m])
        per_v.append({"vehicle": v, "n": int(m.sum()), "rmse": r, "mae": a})
    macro_rmse = float(np.mean([p["rmse"] for p in per_v]))
    return {
        "protocol": "legacy_ensemble_fullset",
        "leakage_free": False,
        "n": len(paths),
        "sample_rate": sample_rate,
        "folds_used": used,
        "rmse_pooled": rmse,
        "mae_pooled": mae,
        "rmse_macro_vehicle": macro_rmse,
        "per_vehicle": per_v,
        "note": "Full-set ensemble; Mixed->Real is NOT held-out.",
    }


def run_oof_retrain(
    clips: List[dict],
    cache: AudioCache,
    dataset_tag: str,
    seed: int,
    sample_rate: int,
) -> dict:
    """Fresh KFold; OOF prediction only; stats per fold on train split only."""
    set_sr(sample_rate, dataset_tag)
    paths = [c["path"] for c in clips]
    speeds = np.asarray([c["speed"] for c in clips], dtype=np.float64)
    vehicles = [c["vehicle"] for c in clips]
    oof = np.full(len(paths), np.nan, dtype=np.float64)
    fold_metrics = []

    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=seed)
    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(paths), start=1):
        tr_paths = [paths[i] for i in tr_idx]
        te_paths = [paths[i] for i in te_idx]
        assert_disjoint(tr_paths, te_paths, f"oof_fold_{fold_i}")

        model, stats, meta = train_model(
            tr_paths, speeds[tr_idx], cache, seed=seed + fold_i
        )
        rmse, mae, pred = predict_rmse(model, te_paths, speeds[te_idx], cache, stats)
        oof[te_idx] = pred
        fold_metrics.append(
            {
                "fold": fold_i,
                "n_test": int(len(te_idx)),
                "fold_rmse": rmse,
                "fold_mae": mae,
                **meta,
            }
        )
        del model
        tf.keras.backend.clear_session()
        print(f"[OOF {dataset_tag}] fold {fold_i}/{Config.N_FOLDS} RMSE={rmse:.4f}")

    assert not np.isnan(oof).any(), "Incomplete OOF coverage"
    rmse, mae = rmse_mae(speeds, oof)
    per_vehicle = {}
    for v in sorted(set(vehicles)):
        m = np.array([veh == v for veh in vehicles])
        r, a = rmse_mae(speeds[m], oof[m])
        per_vehicle[v] = {"n": int(m.sum()), "rmse": r, "mae": a}
    return {
        "protocol": "oof_retrain",
        "leakage_free": True,
        "dataset": dataset_tag,
        "n": len(paths),
        "rmse": rmse,
        "mae": mae,
        "mean_fold_val_rmse": float(np.mean([f["fold_rmse"] for f in fold_metrics])),
        "folds": fold_metrics,
        "per_vehicle": per_vehicle,
    }


def run_lovo(
    real_clips: List[dict],
    ext_clips: List[dict],
    cache: AudioCache,
    seed: int,
    sample_rate: int,
    strict_mixed: bool = True,
    state_path: Optional[str] = None,
) -> dict:
    """
    Leave-one-vehicle-out on REAL test clips.
    Real model: train on real of other vehicles only.
    Mixed model: train on real(other) + ExtendedSim; if strict_mixed, drop ExtSim of v too.
    Both use the same sample_rate (matched to submitted Mixed/Real checkpoints = 16 kHz).
    """
    vehicles = sorted({c["vehicle"] for c in real_clips})
    results_real: List[dict] = []
    results_mixed: List[dict] = []

    if state_path and os.path.isfile(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        results_real = list(prev.get("results_real", []))
        results_mixed = list(prev.get("results_mixed", []))
        print(
            f"[INFO] Resuming LOVO from {state_path}: "
            f"real_done={len(results_real)} mixed_done={len(results_mixed)}",
            flush=True,
        )

    done_real = {r["holdout_vehicle"] for r in results_real}
    done_mixed = {r["holdout_vehicle"] for r in results_mixed}
    all_real_paths = {c["path"] for c in real_clips}

    def _save_state() -> None:
        if not state_path:
            return
        tmp = {
            "results_real": results_real,
            "results_mixed": results_mixed,
            "sample_rate": sample_rate,
            "strict_mixed": strict_mixed,
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(tmp, f, indent=2)

    for v in vehicles:
        test_real = [c for c in real_clips if c["vehicle"] == v]
        train_real = [c for c in real_clips if c["vehicle"] != v]
        test_paths = [c["path"] for c in test_real]
        test_speeds = np.asarray([c["speed"] for c in test_real], dtype=np.float64)
        v_seed = seed + int(hashlib.md5(v.encode()).hexdigest()[:6], 16) % 10000

        # ---- Real-only ----
        if v not in done_real:
            set_sr(sample_rate, f"lovo_real_{v}")
            tr_paths = [c["path"] for c in train_real]
            tr_speeds = np.asarray([c["speed"] for c in train_real], dtype=np.float64)
            assert_disjoint(tr_paths, test_paths, f"lovo_real_{v}")
            assert all(c["vehicle"] != v for c in train_real)

            print(
                f"\n===== LOVO RealData_Model | holdout={v} | "
                f"train={len(tr_paths)} test={len(test_paths)} =====",
                flush=True,
            )
            model, stats, meta = train_model(tr_paths, tr_speeds, cache, seed=v_seed)
            rmse, mae, pred = predict_rmse(model, test_paths, test_speeds, cache, stats)
            results_real.append(
                {
                    "holdout_vehicle": v,
                    "n_train": meta["n_train"] + meta["n_val"],
                    "n_test": len(test_paths),
                    "rmse": rmse,
                    "mae": mae,
                    **meta,
                }
            )
            print(f"[LOVO Real] {v}: RMSE={rmse:.4f} MAE={mae:.4f}", flush=True)
            del model
            tf.keras.backend.clear_session()
            _save_state()
        else:
            print(f"[INFO] skip LOVO Real {v} (cached)", flush=True)

        # ---- Mixed ----
        if v not in done_mixed:
            set_sr(sample_rate, f"lovo_mixed_{v}")
            if strict_mixed:
                train_ext = [c for c in ext_clips if c["vehicle"] != v]
            else:
                train_ext = list(ext_clips)
            mixed_train = train_real + train_ext
            m_paths = [c["path"] for c in mixed_train]
            m_speeds = np.asarray([c["speed"] for c in mixed_train], dtype=np.float64)
            assert_disjoint(m_paths, test_paths, f"lovo_mixed_{v}")
            for c in mixed_train:
                if c["path"] in all_real_paths and c["vehicle"] == v:
                    raise RuntimeError(f"LEAKAGE: real holdout vehicle in mixed train: {c['path']}")
            if strict_mixed:
                assert all(c["vehicle"] != v for c in mixed_train)

            print(
                f"===== LOVO MixedData_Model | holdout={v} | train={len(m_paths)} "
                f"(real={len(train_real)} ext={len(train_ext)}) test={len(test_paths)} =====",
                flush=True,
            )
            model, stats, meta = train_model(m_paths, m_speeds, cache, seed=v_seed + 17)
            rmse, mae, pred = predict_rmse(model, test_paths, test_speeds, cache, stats)
            results_mixed.append(
                {
                    "holdout_vehicle": v,
                    "n_train": meta["n_train"] + meta["n_val"],
                    "n_test": len(test_paths),
                    "n_train_real": len(train_real),
                    "n_train_ext": len(train_ext),
                    "rmse": rmse,
                    "mae": mae,
                    **meta,
                }
            )
            print(f"[LOVO Mixed] {v}: RMSE={rmse:.4f} MAE={mae:.4f}", flush=True)
            del model
            tf.keras.backend.clear_session()
            # Drop mel cache to limit RAM growth across vehicles
            cache._mel_clean.clear()
            _save_state()
        else:
            print(f"[INFO] skip LOVO Mixed {v} (cached)", flush=True)

    def aggregate(rows: List[dict]) -> dict:
        tot_n = sum(r["n_test"] for r in rows)
        mse = sum((r["rmse"] ** 2) * r["n_test"] for r in rows) / max(tot_n, 1)
        mae = sum(r["mae"] * r["n_test"] for r in rows) / max(tot_n, 1)
        return {
            "n_test_total": tot_n,
            "rmse_weighted": float(np.sqrt(mse)),
            "mae_weighted": float(mae),
            "rmse_macro_vehicle": float(np.mean([r["rmse"] for r in rows])),
            "mae_macro_vehicle": float(np.mean([r["mae"] for r in rows])),
            "per_vehicle": rows,
        }

    return {
        "protocol": "leave_one_vehicle_out",
        "leakage_free": True,
        "sample_rate": sample_rate,
        "strict_mixed_excludes_extsim_of_holdout_vehicle": strict_mixed,
        "RealData_Model": aggregate(results_real),
        "MixedData_Model": aggregate(results_mixed),
    }


def manifest_hash(clips: List[dict]) -> str:
    blob = json.dumps(
        [{"p": c["path"], "s": c["speed"], "v": c["vehicle"]} for c in clips],
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root",
        default=_ROOT,
        help="Vehicle-Speed-from-Audio-SE-ResNet root",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Datasets root containing RealData, ExtendedSimulatedData",
    )
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument(
        "--protocols",
        default="legacy,lovo,oof",
        help="Comma list: legacy,lovo,oof",
    )
    parser.add_argument(
        "--skip_strict_mixed",
        action="store_true",
        help="If set, Mixed LOVO keeps ExtendedSim of holdout vehicle",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for JSON reports",
    )
    args = parser.parse_args()

    repo = os.path.abspath(args.repo_root)
    data_root = args.data_root or os.path.normpath(os.path.join(repo, "..", "Datasets"))
    out_dir = args.out_dir or os.path.join(
        repo, "results", "leakage_free", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    set_global_seeds(args.seed)
    protocols = {p.strip().lower() for p in args.protocols.split(",") if p.strip()}

    real_dir = os.path.join(data_root, "RealData")
    ext_dir = os.path.join(data_root, "ExtendedSimulatedData")
    real_clips = list_clips(real_dir)
    ext_clips = list_clips(ext_dir)
    if not real_clips:
        raise SystemExit(f"No RealData clips under {real_dir}")

    print(f"[INFO] Real clips: {len(real_clips)} vehicles={sorted({c['vehicle'] for c in real_clips})}")
    print(f"[INFO] ExtendedSim clips: {len(ext_clips)}")
    print(f"[INFO] Output: {out_dir}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "real_n": len(real_clips),
        "ext_n": len(ext_clips),
        "real_hash": manifest_hash(real_clips),
        "ext_hash": manifest_hash(ext_clips),
        "real_vehicles": sorted({c["vehicle"] for c in real_clips}),
        "config": {
            "epochs": Config.EPOCHS,
            "patience": Config.PATIENCE,
            "batch_size": Config.BATCH_SIZE,
            "n_folds": Config.N_FOLDS,
            "init_lr": Config.INIT_LR,
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    cache = AudioCache()
    report_path = os.path.join(out_dir, "strict_leakage_free_report.json")
    if os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        report.setdefault("manifest", manifest)
        report.setdefault("results", {})
        print(f"[INFO] Loaded existing report with keys={list(report['results'].keys())}", flush=True)
    else:
        report = {"manifest": manifest, "results": {}}

    # Keep latest manifest fields
    report["manifest"] = {**report.get("manifest", {}), **manifest}

    t0 = time.time()

    train_sr = 16000
    if "legacy" in protocols:
        print("\n##### PROTOCOL: legacy ensemble (not leakage-free) #####", flush=True)
        ckpt = os.path.join(repo, "checkpoints")
        real_frames = detect_checkpoint_input_frames(os.path.join(ckpt, "RealData_model"))
        mixed_frames = detect_checkpoint_input_frames(os.path.join(ckpt, "MixedData_model"))
        real_sr = sr_for_n_frames(real_frames) if real_frames else 16000
        mixed_sr = sr_for_n_frames(mixed_frames) if mixed_frames else 16000
        print(f"[INFO] Detected RealData_model frames={real_frames} -> SR={real_sr}", flush=True)
        print(f"[INFO] Detected MixedData_model frames={mixed_frames} -> SR={mixed_sr}", flush=True)
        manifest["detected_checkpoint_sr"] = {
            "RealData_model": real_sr,
            "MixedData_model": mixed_sr,
            "real_frames": real_frames,
            "mixed_frames": mixed_frames,
        }
        train_sr = mixed_sr
        manifest["leakage_free_train_sr"] = train_sr
        report["manifest"] = {**report.get("manifest", {}), **manifest}

        report["results"]["legacy_RealData_Model_on_Real"] = run_legacy_ensemble(
            real_clips,
            os.path.join(ckpt, "RealData_model"),
            cache,
            real_sr,
            "legacy_RealData_Model",
        )
        report["results"]["legacy_MixedData_Model_on_Real"] = run_legacy_ensemble(
            real_clips,
            os.path.join(ckpt, "MixedData_model"),
            cache,
            mixed_sr,
            "legacy_MixedData_Model",
        )
        print("Legacy Real->Real:", report["results"]["legacy_RealData_Model_on_Real"], flush=True)
        print("Legacy Mixed->Real:", report["results"]["legacy_MixedData_Model_on_Real"], flush=True)
        _dump(out_dir, report)
    else:
        prev_sr = report.get("manifest", {}).get("leakage_free_train_sr")
        train_sr = int(prev_sr) if prev_sr else 16000
        manifest["leakage_free_train_sr"] = train_sr
        report["manifest"] = {**report.get("manifest", {}), **manifest}
    if "lovo" in protocols:
        print("\n##### PROTOCOL: LOVO (leakage-free, primary) #####", flush=True)
        state_path = os.path.join(out_dir, "lovo_state.json")
        report["results"]["lovo"] = run_lovo(
            real_clips,
            ext_clips,
            cache,
            seed=args.seed,
            sample_rate=train_sr,
            strict_mixed=not args.skip_strict_mixed,
            state_path=state_path,
        )
        print(json.dumps(report["results"]["lovo"], indent=2)[:2000], flush=True)
        _dump(out_dir, report)

    if "oof" in protocols:
        print("\n##### PROTOCOL: OOF retrain RealData (leakage-free) #####")
        report["results"]["oof_real"] = run_oof_retrain(
            real_clips,
            cache,
            dataset_tag="RealData",
            seed=args.seed,
            sample_rate=train_sr,
        )
        _dump(out_dir, report)

        print("\n##### PROTOCOL: OOF Mixed pool -> metrics on real clips only #####")
        report["results"]["oof_mixed_on_real_slice"] = _oof_mixed_real_slice(
            real_clips, ext_clips, cache, args.seed, sample_rate=train_sr
        )
        _dump(out_dir, report)

    report["elapsed_sec"] = time.time() - t0
    _dump(out_dir, report)
    print(f"\n[DONE] wrote {out_dir} in {report['elapsed_sec']:.1f}s")
    _print_summary(report)
    return 0


def _oof_mixed_real_slice(
    real_clips: List[dict],
    ext_clips: List[dict],
    cache: AudioCache,
    seed: int,
    sample_rate: int,
) -> dict:
    """OOF on Mixed pool; report metrics only on real clips (each from its holdout fold)."""
    set_sr(sample_rate, "oof_mixed_real_slice")
    mixed = sorted(real_clips + ext_clips, key=lambda c: (c["vehicle"], c["speed"], c["path"]))
    paths = [c["path"] for c in mixed]
    speeds = np.asarray([c["speed"] for c in mixed], dtype=np.float64)
    is_real = np.array(
        [
            os.path.normcase(os.path.abspath(c["path"]))
            in {os.path.normcase(os.path.abspath(r["path"])) for r in real_clips}
            for c in mixed
        ]
    )
    oof = np.full(len(paths), np.nan)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=seed)
    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(paths), start=1):
        tr_paths = [paths[i] for i in tr_idx]
        te_paths = [paths[i] for i in te_idx]
        assert_disjoint(tr_paths, te_paths, f"mixed_oof_{fold_i}")
        model, stats, meta = train_model(tr_paths, speeds[tr_idx], cache, seed=seed + 100 + fold_i)
        _, _, pred = predict_rmse(model, te_paths, speeds[te_idx], cache, stats)
        oof[te_idx] = pred
        del model
        tf.keras.backend.clear_session()
        n_real_te = int(is_real[te_idx].sum())
        print(f"[OOF Mixed] fold {fold_i} done (real in test fold: {n_real_te})")

    real_mask = is_real
    assert not np.isnan(oof[real_mask]).any()
    rmse, mae = rmse_mae(speeds[real_mask], oof[real_mask])
    return {
        "protocol": "oof_mixed_pool_metrics_on_real_clips_only",
        "leakage_free": True,
        "sample_rate": sample_rate,
        "n_mixed": len(paths),
        "n_real_eval": int(real_mask.sum()),
        "rmse": rmse,
        "mae": mae,
        "note": "Each real clip predicted only by the Mixed fold that held it out of training.",
    }


def _dump(out_dir: str, report: dict) -> None:
    path = os.path.join(out_dir, "strict_leakage_free_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] saved {path}")


def _print_summary(report: dict) -> None:
    print("\n========== SUMMARY ==========")
    r = report.get("results", {})
    if "legacy_MixedData_Model_on_Real" in r:
        x = r["legacy_MixedData_Model_on_Real"]
        print(
            f"LEGACY Mixed->Real pooled={x.get('rmse_pooled', x.get('rmse')):.4f} "
            f"macro={x.get('rmse_macro_vehicle', float('nan')):.4f} (NOT leakage-free)"
        )
    if "legacy_RealData_Model_on_Real" in r:
        x = r["legacy_RealData_Model_on_Real"]
        print(
            f"LEGACY Real->Real pooled={x.get('rmse_pooled', x.get('rmse')):.4f} "
            f"macro={x.get('rmse_macro_vehicle', float('nan')):.4f} (NOT leakage-free)"
        )
    if "lovo" in r:
        L = r["lovo"]
        print(
            f"LOVO RealData_Model RMSE={L['RealData_Model']['rmse_weighted']:.4f} "
            f"(macro={L['RealData_Model']['rmse_macro_vehicle']:.4f})"
        )
        print(
            f"LOVO MixedData_Model RMSE={L['MixedData_Model']['rmse_weighted']:.4f} "
            f"(macro={L['MixedData_Model']['rmse_macro_vehicle']:.4f})"
        )
    if "oof_real" in r:
        print(f"OOF RealData RMSE={r['oof_real']['rmse']:.4f}")
    if "oof_mixed_on_real_slice" in r:
        print(f"OOF Mixed->Real (real clips) RMSE={r['oof_mixed_on_real_slice']['rmse']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
