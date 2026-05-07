import os
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d

from visualization.plot_utils import compute_path_points


def compute_road_boundaries(all_paths, lane_width, include_opposite=False, road_y_center=0.0, road_angle=0.0):
    """
    Compute road boundaries (upper, lower, and centerline) based on all vehicle paths.
    """
    if not all_paths:
        return None, None, None, None

    # Find x range
    x_min = min(np.min(x) for x, y in all_paths)
    x_max = max(np.max(x) for x, y in all_paths)

    # Common x grid
    x_common = np.linspace(x_min, x_max, 300)

    # If we have road_angle, the road is rotated. 
    # For now, we compute boundaries in the road-aligned frame or infer from paths.
    # The safest way is to use the center of mass of the paths as the baseline centerline.
    
    # Interpolate all paths onto common x-grid
    y_interpolated = []
    for x, y in all_paths:
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        y_interp = np.interp(x_common, x_sorted, y_sorted,
                            left=y_sorted[0], right=y_sorted[-1])
        y_interpolated.append(y_interp)

    # Calculate centerline
    avg_y_per_path = [np.mean(y) for y in y_interpolated]

    if len(y_interpolated) > 1:
        sorted_indices = np.argsort(avg_y_per_path)
        num_cars = len(sorted_indices)
        split_idx = num_cars // 2

        lower_indices = sorted_indices[:split_idx]
        upper_indices = sorted_indices[split_idx:]

        y_lower_max = np.max([y_interpolated[i] for i in lower_indices], axis=0)
        y_upper_min = np.min([y_interpolated[i] for i in upper_indices], axis=0)

        y_centerline = (y_lower_max + y_upper_min) / 2
    else:
        y_centerline = np.mean(y_interpolated, axis=0)

    # Smooth centerline
    y_centerline_smooth = gaussian_filter1d(y_centerline, sigma=10)

    # Road boundaries
    # The total road width is 2 * lane_width (fwd + opp)
    y_upper = y_centerline_smooth + lane_width
    y_lower = y_centerline_smooth - lane_width

    return x_common, y_upper, y_lower, y_centerline_smooth


def check_path_violations(path_x, path_y, x_common, y_upper, y_lower, y_centerline,
                         tolerance=0.5):
    """
    Check if a single path violates road boundaries or crosses the median.

    Args:
        path_x, path_y: Vehicle path coordinates
        x_common: Common x-coordinates for boundaries
        y_upper, y_lower: Road boundaries
        y_centerline: Center line (median)
        tolerance: Allowed tolerance for violations (in meters)

    Returns:
        dict with violation information
    """
    # Interpolate boundaries to path's x-coordinates
    y_upper_interp = np.interp(path_x, x_common, y_upper)
    y_lower_interp = np.interp(path_x, x_common, y_lower)
    y_center_interp = np.interp(path_x, x_common, y_centerline)

    violations = {
        'has_violation': False,
        'upper_boundary_violations': [],
        'lower_boundary_violations': [],
        'median_crossings': [],
        'summary': ''
    }

    # Check upper boundary violations
    upper_violations = path_y > (y_upper_interp + tolerance)
    if np.any(upper_violations):
        violation_indices = np.where(upper_violations)[0]
        max_violation = np.max(path_y[upper_violations] - y_upper_interp[upper_violations])
        violations['has_violation'] = True
        violations['upper_boundary_violations'] = [
            {
                'index': int(idx),
                'x': float(path_x[idx]),
                'y': float(path_y[idx]),
                'boundary_y': float(y_upper_interp[idx]),
                'violation_distance': float(path_y[idx] - y_upper_interp[idx])
            }
            for idx in violation_indices[::10]  # Sample every 10th point
        ]
        violations['summary'] += f"Upper boundary violated by {max_violation:.2f}m. "

    # Check lower boundary violations
    lower_violations = path_y < (y_lower_interp - tolerance)
    if np.any(lower_violations):
        violation_indices = np.where(lower_violations)[0]
        max_violation = np.max(y_lower_interp[lower_violations] - path_y[lower_violations])
        violations['has_violation'] = True
        violations['lower_boundary_violations'] = [
            {
                'index': int(idx),
                'x': float(path_x[idx]),
                'y': float(path_y[idx]),
                'boundary_y': float(y_lower_interp[idx]),
                'violation_distance': float(y_lower_interp[idx] - path_y[idx])
            }
            for idx in violation_indices[::10]
        ]
        violations['summary'] += f"Lower boundary violated by {max_violation:.2f}m. "

    # Check median crossings (detect when path crosses from one side to other)
    y_relative_to_center = path_y - y_center_interp
    sign_changes = np.diff(np.sign(y_relative_to_center))
    crossings = np.where(sign_changes != 0)[0]

    if len(crossings) > 0:
        violations['has_violation'] = True
        violations['median_crossings'] = [
            {
                'index': int(idx),
                'x': float(path_x[idx]),
                'y_before': float(path_y[idx]),
                'y_after': float(path_y[idx + 1]),
                'centerline_y': float(y_center_interp[idx])
            }
            for idx in crossings
        ]
        violations['summary'] += f"Median crossed {len(crossings)} time(s). "

    if not violations['has_violation']:
        violations['summary'] = "No violations detected."

    return violations


def validate_scene_paths(scenes_data, lane_width=4.0, include_opposite=False,
                        tolerance=0.5, y_shift=0.0, road_y_center=0.0, road_angle=0.0):
    """
    Validate all vehicle paths in a scene for boundary and median violations.

    Args:
        scenes_data: List of (path_type, params, vehicle_name) tuples
        lane_width: Total road width
        include_opposite: Whether road has opposite traffic
        tolerance: Allowed tolerance for violations (meters)
        y_shift: Y-axis transformation applied to paths (from plotting code)

    Returns:
        dict with validation results for all vehicles
    """
    # Compute all vehicle paths
    all_paths = []
    for path_type, params, vehicle_name in scenes_data:
        x, y, _ = compute_path_points(path_type, params, n_points=200, 
                                     road_y_center=road_y_center, 
                                     road_angle=road_angle)
        # Apply the same y_shift transformation as in plotting (usually 0 if absolute)
        y_shifted = y + y_shift
        all_paths.append((x, y_shifted))

    # Compute road boundaries
    x_common, y_upper, y_lower, y_centerline = compute_road_boundaries(
        all_paths, lane_width, include_opposite, road_y_center, road_angle
    )

    # Validate each vehicle path
    results = {
        'scene_valid': True,
        'total_vehicles': len(scenes_data),
        'vehicles_with_violations': 0,
        'vehicle_results': []
    }

    for i, ((x, y), (path_type, params, vehicle_name)) in enumerate(zip(all_paths, scenes_data)):
        violations = check_path_violations(x, y, x_common, y_upper, y_lower,
                                          y_centerline, tolerance)

        vehicle_result = {
            'vehicle_id': i + 1,
            'vehicle_name': vehicle_name,
            'path_type': path_type,
            'violations': violations
        }

        if violations['has_violation']:
            results['scene_valid'] = False
            results['vehicles_with_violations'] += 1

        results['vehicle_results'].append(vehicle_result)

    return results


def save_validation_report(validation_results, output_dir, scene_id):
    """Save validation results to JSON and text files."""
    # Save JSON
    validation_file = os.path.join(output_dir, f"validation.json")
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)

    # Save human-readable report
    report_file = os.path.join(output_dir, f"validation.txt")
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"PATH VALIDATION REPORT - Scene {scene_id}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total Vehicles: {validation_results['total_vehicles']}\n")
        f.write(f"Scene Valid: {'YES' if validation_results['scene_valid'] else 'NO'}\n")
        f.write(f"Vehicles with Violations: {validation_results['vehicles_with_violations']}\n")
        f.write("=" * 70 + "\n\n")

        for vehicle_result in validation_results['vehicle_results']:
            vid = vehicle_result['vehicle_id']
            vname = vehicle_result['vehicle_name']
            vpath = vehicle_result['path_type']
            violations = vehicle_result['violations']

            status = "VALID" if not violations['has_violation'] else "INVALID"
            f.write(f"Vehicle {vid}: {vname} ({vpath}) - {status}\n")

            if violations['has_violation']:
                f.write(f"  Summary: {violations['summary']}\n")

                if violations['upper_boundary_violations']:
                    f.write(f"  - Upper boundary violations: "
                          f"{len(violations['upper_boundary_violations'])} points\n")

                if violations['lower_boundary_violations']:
                    f.write(f"  - Lower boundary violations: "
                          f"{len(violations['lower_boundary_violations'])} points\n")

                if violations['median_crossings']:
                    f.write(f"  - Median crossings: {len(violations['median_crossings'])}\n")

            f.write("\n")

    return validation_file, report_file
