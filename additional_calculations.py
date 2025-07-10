# This file contains additional calculations for the project.
# Hausdorf dimension
import numpy as np
from typing import Dict, Any

# function for single 2d curve
def calc_hausdorff_dimension_for_single_signal(time: np.ndarray, signal: np.ndarray, n_boxes=10):
    """
    Calculate the box-counting Hausdorff dimension for a singel 2D curve (time vs signal).
    """
    if len(time) < 2 or len(signal) < 2:
        return np.nan

    # Normalize data to fit inside unit square [0,1] x [0,1]
    x = (time - np.min(time)) / (np.max(time) - np.min(time))
    y = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    # Stack into points
    points = np.vstack((x, y)).T

    # Box sizes
    epsilons = np.logspace(-1, 0, n_boxes, base=10.0)  # From 0.1 to 1.0

    N = []  # number of boxes needed

    for eps in epsilons:
        # Create grid
        bins = np.ceil(1 / eps).astype(int)
        grid = set()
        for px, py in points:
            ix = int(px / eps)
            iy = int(py / eps)
            grid.add((ix, iy))
        N.append(len(grid))

    log_eps = -np.log10(epsilons)
    log_N = np.log10(N)

    # Linear fit to estimate slope = Hausdorff dimension
    slope, _ = np.polyfit(log_eps, log_N, 1)
    return round(slope, 4)


# called function to calcuate Hausdorff dimensions for all axes
def compute_hausdorff_dimensions_all_axes(file_data: Dict[str, Any], force_columns=None) -> None:
    """
    Computes Hausdorff dimension for each contact interval across multiple force axes.
    Stores results under 'hausdorff_<force>' in each interval dictionary.
    """
    if force_columns is None:
        # Default axes based on your system
        force_columns = ["Fx", "Fy", "Fz", "Mz", "Fres_yz", "Fres_xyz"]

    for side in ["G1R", "G2L"]:
        intervals = file_data[side].get("intervals", {})
        for label, stats in intervals.items():
            if not isinstance(stats, dict) or "interval_data" not in stats:
                continue
            interval_data = stats["interval_data"]
            time = np.array(interval_data.get("Time [s]", []))

            for force in force_columns:
                # Try exact match first
                signal_list = interval_data.get(force)

                # If not found, try fuzzy match
                if signal_list is None:
                    matching_keys = [k for k in interval_data.keys() if force in k and "[" in k]
                    if matching_keys:
                        signal_list = interval_data[matching_keys[0]]

                signal = np.array(signal_list if signal_list is not None else [])
                if len(time) > 5 and len(signal) == len(time):
                    hd = calc_hausdorff_dimension_for_single_signal(time, signal)
                else:
                    hd = None
                stats[f"hausdorff_{force}"] = hd
        print(f"[{side}] Hausdorff dimensions computed for axes: {', '.join(force_columns)}.")