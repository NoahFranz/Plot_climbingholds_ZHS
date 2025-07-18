# This file contains additional calculations for  hausdorff dimensions and plotting.
# Hausdorf dimension
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import config
import os

# function for single 2d curve
def calc_hausdorff_dimension_for_single_signal(time: np.ndarray, signal: np.ndarray, n_boxes=15,curr_file_data=Dict[str, Any], current_side="G1R", current_force="Fx", current_interval="") -> float:
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
            ix = int(np.floor(px / eps))
            iy = int(np.floor(py / eps))
            grid.add((ix, iy))
        N.append(len(grid))

    log_eps = -np.log10(epsilons)
    log_N = np.log10(N)
    if config.create_hausdorff_plots:
        for current_force in config.force_to_plot:
            plt.figure()
            plt.plot(log_eps, log_N, 'o-', label="log-log data")
            slope, intercept = np.polyfit(log_eps, log_N, 1)
            plt.plot(log_eps, slope * log_eps + intercept, '--', label=f"slope = {slope:.3f}")
            curr_filennumber = config.file_number
            saveFolder = config.save_folder or "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/plotsWithoutSaveFolder"

            force_plot_dir = os.path.join(saveFolder, curr_filennumber, curr_filennumber + "-" + current_force + "-debug", current_side, current_force)

            plt.xlabel("log(1/ε)")
            plt.ylabel("log N(ε)")
            manual_title = f"{curr_filennumber}_{current_side}_{current_interval}_{current_force}-HD Log-Log Fit".replace(" ", "_")
            plt.title(manual_title)
            plt.legend()
            plt.grid(True)
            #plt.show()
            
            os.makedirs(force_plot_dir, exist_ok=True)
            filename = f"{curr_filennumber}_{current_side}_{current_interval}_{current_force}_HD-debug.png".replace(" ", "_")
            plt.savefig(os.path.join(force_plot_dir, filename))
            plt.close()
            print(f"Debug plot saved to: {os.path.join(force_plot_dir, filename)}")
        else:
            slope, _ = np.polyfit(log_eps, log_N, 1)
    else:
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


        # plot hasudrof plots
def plot_hausdorff_intervals(file_data: Dict[str, Any], folder_path: str, fname: str) -> None:
    """
    Plots force-time curves for each interval and force axis with Hausdorff values annotated.
    Saves plots as PNG files in a 'plots' subfolder.
    """
    plot_dir = os.path.join(folder_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for side in ["G1R", "G2L"]:
        intervals = file_data.get(side, {}).get("intervals", {})
        for int_key, stats in intervals.items():
            if int_key == "Mean-Metrics":
                continue
            interval_data = stats.get("interval_data", {})
            time = interval_data.get("Time [s]", [])
            if not time:
                continue
            for force in ["Fx", "Fy", "Fz", "Mz", "Fres_yz", "Fres_xyz"]:
                # Find the correct key containing the force and a bracket (like "[N]")
                matching_keys = [k for k in interval_data if force in k]
                force_series = interval_data.get(matching_keys[0], []) if matching_keys else []
                hausdorff = stats.get(force, {}).get("hausdorff", None)
                if force_series and hausdorff is not None:
                    file_number = config.file_number if config.file_number else "000"
                    plt.figure()
                    plt.plot(time, force_series, label=f"{force}")
                    plt.plot(time, force_series, 'x', label='data points')
                    plt.xlabel("Time [s]")
                    plt.ylabel(force)
                    plt.title(f" | {file_number} | {int_key} | {force} | HD = {hausdorff:.3f}")
                    plt.grid(True)
                    force_plot_dir = os.path.join(plot_dir, file_number, side, force)
                    os.makedirs(force_plot_dir, exist_ok=True)
                    filename = f"{file_number}_{side}_{int_key}_{force}_HD.png".replace(" ", "_")
                    plt.savefig(os.path.join(force_plot_dir, filename))
                    plt.close()