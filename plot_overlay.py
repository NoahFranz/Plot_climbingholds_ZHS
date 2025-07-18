import os
import json
import pandas as pd
#mport pingouin as pg

from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib import colormaps
from config import COLOR_MAPPING

"""
Module for plotting force-time curves from summary_metadata.json, including per-interval overlays and per-file mean±STD.
Provides functions: plot_interval_for_single_test and plot_compare_and_combine_test_Retest.
"""

def load_data(json_path: str, exclude_interval: str = "") -> Dict[str, Any]:
    data = json.load(open(json_path))
    if "068" in os.path.basename(json_path):
        # Always delete I1 if the file contains 068
        interval_key = "I1"
        for file_data in data.values():
            for side in ["G1R", "G2L"]:
                if side in file_data and "intervals" in file_data[side]:
                    file_data[side]["intervals"].pop(interval_key, None)
    elif exclude_interval:
        # Apply GUI-based deletion only if not overridden by 068
        interval_key = f"I{exclude_interval.strip()}"
        for file_data in data.values():
            for side in ["G1R", "G2L"]:
                if side in file_data and "intervals" in file_data[side]:
                    file_data[side]["intervals"].pop(interval_key, None)
    return data

def interpolate_curves(curves: List[Tuple[np.ndarray, np.ndarray]], t_common: np.ndarray) -> List[np.ndarray]:
    interpolated = []
    for t, f in curves:
        if len(t) > 1 and len(f) > 1:
            interp_f = interp1d(t, f, kind='linear', bounds_error=False, fill_value=np.nan)(t_common)
            interpolated.append(interp_f)
    return interpolated

def extract_file_identity(file_key: str) -> str:
    return file_key.split("-")[0]

def construct_force_key(force_component: str, side_key: str) -> str:
    """Return the JSON key for force_component and side_key, mapping side to suffix and component to units."""
    side_suffix = "1" if side_key == "G1R" else "2"
    unit_map = {
        "Fy": " [%]",
        "Fz": " [%]",
        "Fx": " [%]",
        "Mz": " [Nm]",
        "FgR_sum": " [%]",
        "Fres_yz": "",
        "Fres_xyz": ""
    }
    unit = unit_map.get(force_component, "")
    return f"{force_component}_{side_suffix}{unit}"

def plot_mean_and_std(
    t_common: np.ndarray,
    all_f_interp: List[np.ndarray],
    show_std: bool,
    force_component: str = None,
    side_label: str = None,
    mean_linewidth: float = 1.0,
    mean_linestyle: str = "-"
) -> None:
    """Plot the mean curve and optional ±1 STD shading for interpolated force data."""
    mean_curve = np.nanmean(all_f_interp, axis=0)
    label_str = f"Mean-{force_component}-{side_label[-1]}"
    plt.plot(
        t_common,
        mean_curve,
        color="black",
        linewidth=mean_linewidth,
        linestyle=mean_linestyle,
        label=label_str
    )
    if show_std:
        std_curve = np.nanstd(all_f_interp, axis=0)
        plt.fill_between(
            t_common,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color="black",
            alpha=0.2,
            label="±1 STD"
        )

def plot_interval_for_single_test(
    json_path: str,
    force_component: str = "Fy",
    normalize_time: bool = False,
    show_std: bool = False,
    save_flag: bool = False,
    exclude_interval: str = "",
    show_points: bool = False,
    mean_linewidth: float = 1.0,
    mean_linestyle: str = "-",
    filename_suffix: str = "",
    show_trial_mean: bool = False,
    show_trial_std: bool = False,
    show_global_mean=True,
) -> None:
    """
    Plot force-time curves for each side, with options to normalize time, show std, and save figures.
    
    Parameters:
        json_path (str): Path to metadata JSON.
        force_component (str): 'Fy', 'Fz', or 'Fx'.
        normalize_time (bool): If True, time axis rescaled to [0,1].
        show_std (bool): If True, plot shaded STD region.
        save_flag (bool): If True, save figures as PNG.
    """
    data = load_data(json_path, exclude_interval)

    sides = {"G1R": [], "G2L": []}
    cmap = colormaps["tab10"]

    # Collect intervals
    for file_idx, (file_key, file_content) in enumerate(data.items()):
        for side_key in ["G1R", "G2L"]:
            if side_key not in file_content:
                continue
            intervals = file_content[side_key].get("intervals", {})
            for int_idx, (int_key, interval_data) in enumerate(intervals.items()):
                if int_key == "Mean-Metrics":
                    continue
                series = interval_data.get("interval_data", {})
                t = np.array(series.get("Time [s]", []))
                force_key = construct_force_key(force_component, side_key)
                f = np.array(series.get(force_key, []))
                t = np.asarray(t, dtype=float)
                f = np.asarray(f, dtype=float)
                if len(t) < 2 or len(f) < 2:
                    continue
                sort_idx = np.argsort(t)
                t, f = t[sort_idx], f[sort_idx]
                t = t - t[0]
                if normalize_time:
                    t = t / t[-1] if t[-1] != 0 else t

                sides[side_key].append({
                    "t": t,
                    "f": f,
                    "file": file_key,
                    "color": cmap(file_idx % 10),
                    "alpha": 0.3 + 0.7 * (1 - int_idx / max(1, len(intervals)-1))
                })

    # --- Add mean/std per trial if requested ---
    if show_trial_mean:
        for side_key in sides:
            file_grouped = {}
            for item in sides[side_key]:
                file_grouped.setdefault(item["file"], []).append((item["t"], item["f"]))

            new_sides = {side_key: []}
            for file_key, curves in file_grouped.items():
                if len(curves) < 1:
                    continue
                t_min = max(curve[0][0] for curve in curves)
                t_max = min(curve[0][-1] for curve in curves)
                if t_max <= t_min:
                    continue
                t_common = np.linspace(t_min, t_max, 200)
                interpolated = interpolate_curves(curves, t_common)
                if not interpolated:
                    continue
                mean = np.nanmean(interpolated, axis=0)
                trial_curve = {
                    "t": t_common,
                    "f": mean,
                    "file": file_key,
                    "color": cmap(list(file_grouped.keys()).index(file_key) % 10),
                    "alpha": 1.0
                }
                new_sides[side_key].append(trial_curve)
                if show_trial_std:
                    std = np.nanstd(interpolated, axis=0)
                    trial_curve["std"] = std
            sides[side_key] = new_sides[side_key]

    def plot_side(intervals: List[Dict[str, Any]], side_label: str) -> None:
        plt.figure(figsize=(10, 5))
        all_f_interp = []
        files_seen = {}

        t_min = min((curve["t"][0] for curve in intervals if len(curve["t"]) > 0), default=0)
        t_max = max((curve["t"][-1] for curve in intervals if len(curve["t"]) > 0), default=1)
        t_common = np.linspace(t_min, t_max, 200)

        for interval in intervals:
            t = interval["t"]
            f = interval["f"]
            color = interval["color"]
            alpha = interval["alpha"]
            file = interval["file"]
            side_number = side_label[-1]  # Extract "R" or "L"
            file_identity = extract_file_identity(file)
            label = f"{file_identity}-{force_component}-{side_number}" if file not in files_seen else f"_{file_identity}-{force_component}-{side_number}"
            files_seen[file] = True

            if not show_std:
                plt.plot(t, f, color=color, alpha=alpha, label=label)
            try:
                interp_f = interp1d(t, f, kind='linear', bounds_error=False, fill_value=np.nan)(t_common)
                all_f_interp.append(interp_f)
            except Exception:
                continue
            # Move the show_points block here so "x" markers are drawn for each interval
            if show_points:
                plt.plot(t, f, "x", color=color, alpha=alpha * 0.9)
            # Show STD band if present in interval (for trial mean+std)
            if show_std and "std" in interval:
                std = interval["std"]
                plt.fill_between(t, f - std, f + std, color=color, alpha=0.3, label="±1 STD" if file not in files_seen else None)

        if show_global_mean and all_f_interp:
            plot_mean_and_std(
                t_common,
                all_f_interp,
                show_std,
                force_component=force_component,
                side_label=side_label,
                mean_linewidth=mean_linewidth,
                mean_linestyle=mean_linestyle
            )

        plt.xlabel("Time [s]" if not normalize_time else "Normalized Time")
        plt.ylabel("F [%]")
        y_min = min((min(curve["f"]) for curve in intervals if len(curve["f"]) > 0), default=0)
        y_max = max((max(curve["f"]) for curve in intervals if len(curve["f"]) > 0), default=80)
        #plt.ylim(y_min - 30, 80)
        plt.title(f"{force_component} – {side_label}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        base_dir = os.path.dirname(json_path)
        # Create subfolder for this force-component and side
        subfolder = os.path.join(base_dir, f"{force_component}_{side_label[-1]}")
        os.makedirs(subfolder, exist_ok=True)
        # Use file_identity from the first interval if available, else empty string
        file_identity = extract_file_identity(intervals[0]["file"]) if intervals and "file" in intervals[0] else ""
        filename = f"{file_identity}_interval_overlay_{force_component}-{side_label[-1]}"
        if normalize_time:
            filename += "_t_norm"
        if show_std:
            filename += "_std"
        if show_points:
            filename += "_data-marked"
        if show_trial_mean:
            filename += "_trialmean"
        if show_trial_std:
            filename += "_std"
        if filename_suffix:
            filename += f"_{filename_suffix}"
        filename += ".png"
        filepath = os.path.join(subfolder, filename)
        if save_flag:
            print(f"Saving plot to {filepath}")
            plt.savefig(filepath, dpi=700)
        plt.show()

    plot_side(sides["G1R"], "G1R")
    plot_side(sides["G2L"], "G2L")



def plot_compare_and_combine_test_Retest(
    json_path: str,
    force_component: str = "Fy",
    filename_suffix: str = "",
    normalize_time: bool = False,
    save_flag: bool = False,
    only_combined: bool = False,
    mean_linewidth: float = 1.0,
    mean_linestyle: str = "-"
) -> None:
    """
    Plot mean ± STD per file and combined for each side, with options to normalize time and save figures.
    
    Parameters:
        json_path (str): Path to metadata JSON.
        force_component (str): 'Fy', 'Fz', or 'Fx'.
        normalize_time (bool): If True, time axis rescaled to [0,1].
        save_flag (bool): If True, save figures as PNG.
    """
    data = load_data(json_path)

    cmap = colormaps["tab10"]
    

    for side_key in ["G1R", "G2L"]:
        all_interpolated = []
        file_curves = {}
        t_min, t_max = float("inf"), float("-inf")
        if not only_combined:
            for file_idx, (file_key, file_content) in enumerate(data.items()):
                if side_key not in file_content:
                    continue
                intervals = file_content[side_key].get("intervals", {})
                file_curves[file_key] = []
                for int_key, interval_data in intervals.items():
                    if int_key == "Mean-Metrics":
                        continue
                    series = interval_data.get("interval_data", {})
                    t = np.array(series.get("Time [s]", []), dtype=float)
                    force_key = construct_force_key(force_component, side_key)
                    f = np.array(series.get(force_key, []), dtype=float)
                    if len(t) < 2 or len(f) < 2:
                        continue
                    sort_idx = np.argsort(t)
                    t, f = t[sort_idx], f[sort_idx]
                    t = t - t[0]
                    if normalize_time:
                        t = t / t[-1] if t[-1] != 0 else t
                    t_min = min(t_min, t[0])
                    t_max = max(t_max, t[-1])
                    file_curves[file_key].append((t, f))
                    all_interpolated.append((t, f))
        else:
            # Still collect all_interpolated data even when skipping per-file means
            for file_key, file_content in data.items():
                if side_key not in file_content:
                    continue
                intervals = file_content[side_key].get("intervals", {})
                for int_key, interval_data in intervals.items():
                    if int_key == "Mean-Metrics":
                        continue
                    series = interval_data.get("interval_data", {})
                    t = np.array(series.get("Time [s]", []), dtype=float)
                    force_key = construct_force_key(force_component, side_key)
                    f = np.array(series.get(force_key, []), dtype=float)
                    if len(t) < 2 or len(f) < 2:
                        continue
                    sort_idx = np.argsort(t)
                    t, f = t[sort_idx], f[sort_idx]
                    t = t - t[0]
                    if normalize_time:
                        t = t / t[-1] if t[-1] != 0 else t
                    t_min = min(t_min, t[0])
                    t_max = max(t_max, t[-1])
                    all_interpolated.append((t, f))

        if not all_interpolated:
            continue

        t_common = np.linspace(t_min, t_max, 200)
        plt.figure(figsize=(10, 5))

        for file_idx, (file_key, curves) in enumerate(file_curves.items()):
            interpolated = interpolate_curves(curves, t_common)
            if interpolated:
                mean = np.nanmean(interpolated, axis=0)
                std = np.nanstd(interpolated, axis=0)
                file_identity = extract_file_identity(file_key)
                label_name = f"{file_identity}-{force_component}-{side_key[-1]}"
                plt.plot(t_common, mean, label=label_name, color=cmap(file_idx % 10))
                plt.fill_between(t_common, mean - std, mean + std, color=cmap(file_idx % 10), alpha=0.2)

        combined_interp = interpolate_curves(all_interpolated, t_common)
        if combined_interp:
            combined_mean = np.nanmean(combined_interp, axis=0)
            combined_std = np.nanstd(combined_interp, axis=0)

            # Ensure subfolder exists
            base_dir = os.path.dirname(json_path)
            subfolder = os.path.join(base_dir, f"{force_component}_{side_key[-1]}")
            os.makedirs(subfolder, exist_ok=True)
            # Export combined mean and STD to Excel
            import pandas as pd
            df_export = pd.DataFrame({
                "Time": t_common,
                "Mean": combined_mean,
                "STD": combined_std
            })
            excel_path = os.path.join(subfolder, f"combined_mean_STD_{force_component}_{side_key}.xlsx")
            df_export.to_excel(excel_path, index=False)
            print(f"Saved combined mean/STD to: {excel_path}")

            plt.plot(
                t_common,
                combined_mean,
                color="black",
                linewidth=mean_linewidth,
                linestyle=mean_linestyle,
                label=f"Combined-{force_component}-{side_key[-1]}"
            )
            plt.fill_between(
                t_common,
                combined_mean - combined_std,
                combined_mean + combined_std,
                color="black",
                alpha=0.2
            )

        all_interp_flat = np.concatenate([arr for arr in combined_interp if arr is not None])
        if len(all_interp_flat) > 0:
            y_min = np.nanmin(all_interp_flat)
            y_max = np.nanmax(all_interp_flat)
            plt.ylim(y_min - 5, y_max + 5)
        else:
            plt.ylim(-10, 80)

        plt.xlabel("Normalized Time" if normalize_time else "Time [s]")
        plt.ylabel("F [%]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        base_dir = os.path.dirname(json_path)
        # Create subfolder for this force-component and side
        subfolder = os.path.join(base_dir, f"{force_component}_{side_key[-1]}")
        os.makedirs(subfolder, exist_ok=True)
        file_ids = "_".join(sorted(set(extract_file_identity(k) for k in file_curves.keys())))
        filename = f"mean_std_all_files_{force_component}-{side_key[-1]}_{file_ids}"
        if only_combined:
            filename = f"mean_std_combined_{force_component}-{side_key[-1]}_{file_ids}"
        if normalize_time:
            filename += "_t_norm"
        filename += ".png"
        filepath = os.path.join(subfolder, filename)
        if save_flag:
            print(f"Saving plot to {filepath}")
            plt.savefig(filepath, dpi=700)
        plt.title(f"{force_component} – Mean ± STD per File and Combined – {side_key}")
        plt.show()
def calculate_reliability_metric(json_path: str, re_metric: str, force_component: str,) -> None:
    """
    Extracts force metric data from JSON for test-retest reliability evaluation and exports to Excel.

    Parameters:
        json_path (str): Path to the summary metadata JSON file.
        re_metric (str): Metric to extract; one of "mean", "max", "duration_s".
    """
    import pandas as pd

    assert re_metric in ["mean", "max", "duration_s"], "Invalid metric selected."

    # Load the JSON summary metadata
    data = load_data(json_path)

    # Define the sides of interest and valid metric names
    sides = ["G1R", "G2L"]

    # Set up output Excel path
    base_dir = os.path.dirname(json_path)
    out_path = os.path.join(base_dir, f"Reliability_{re_metric}_{force_component}.xlsx")

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for side in sides:
            records = {"Test": [], "Retest": []}
            interval_labels = []
            table_name = f"{re_metric}_{force_component}_{side}"

            # Loop over all files and filter by "Test" or "Retest" using substrings
            for file_key, file_content in data.items():
                label = "Test" if "020" in file_key else "Retest" if "072" in file_key else None
                if label is None or side not in file_content:
                    continue

                # Navigate to the interval metrics for the given side
                intervals = file_content[side].get("intervals", {})
                metric_values = []
                for int_key, interval_data in intervals.items():
                    if int_key == "Mean-Metrics":
                        continue
                    if re_metric == "duration_s":
                        interval_value = interval_data.get("duration_s", None)
                    else:
                        interval_value = interval_data.get(force_component, {}).get(re_metric, None)

                    # Extract the force component dictionary (e.g., Fy, Fz)
                    if interval_value is not None:
                        metric_values.append(interval_value)
                        # Store up to 3 unique interval labels, regardless of label
                        if int_key not in interval_labels:
                            if len(interval_labels) < 3:
                                interval_labels.append(int_key)
                        # Continue checking all intervals regardless

                # Store values per session label
                records[label].append(metric_values)
                print("records", records)

            # Prepare the DataFrame with header row and interval labels
            df = pd.DataFrame(columns=[""] + interval_labels + ["Mean"])
            
            df.at[0, ""] = "Test"
            df.at[1, ""] = "Retest"

            # Fill in per-interval values for Test and Retest
            for i, label in enumerate(interval_labels):
                test_val = records["Test"][0][i] if records["Test"] and len(records["Test"][0]) > i else None
                retest_val = records["Retest"][0][i] if records["Retest"] and len(records["Retest"][0]) > i else None
                df.loc[0, label] = test_val
                df.loc[1, label] = retest_val

            # Add mean values from Mean-Metrics
            for file_key, file_content in data.items():
                label = "Test" if "020" in file_key else "Retest" if "072" in file_key else None
                if label is None or side not in file_content:
                    continue
                mean_metrics = file_content[side].get("intervals", {}).get("Mean-Metrics", {})
                if re_metric == "duration_s":
                    mean_val = mean_metrics.get("Contacttime", {}).get("mean", None)
                else:
                    mean_val = mean_metrics.get(force_component, {}).get(re_metric, None)
                if mean_val is not None:
                    df.loc[0 if label == "Test" else 1, "Mean"] = mean_val

            # Export to Excel starting at row 3 (so table name can be written manually in row 1)
            df.to_excel(writer, sheet_name=side, startrow=2, index=False)
            worksheet = writer.sheets[side]
            worksheet.cell(row=1, column=1, value=table_name)

        print(f"Reliability metric exported to: {out_path}")
def append_reliability_stats_to_excel(file_path: str) -> None:
    """Load an Excel file with Test/Retest rows, compute SEM, MDC, CoV, and write results back to the same file."""
    import pandas as pd
    import numpy as np
    from openpyxl import load_workbook

    df = pd.read_excel(file_path, sheet_name=None)
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        for sheet_name, table in df.items():
            if table.empty or table.shape[0] < 2:
                continue

            try:
                # Dynamically find rows labeled 'Test' and 'Retest' in column 0
                test_row = table[table.iloc[:, 0].astype(str).str.lower() == "test"]
                retest_row = table[table.iloc[:, 0].astype(str).str.lower() == "retest"]

                if test_row.empty or retest_row.empty:
                    print(f"Skipping sheet {sheet_name}: 'Test' or 'Retest' row not found.")
                    continue

                test_vals = test_row.iloc[0, 1:-1].dropna().astype(float).values
                retest_vals = retest_row.iloc[0, 1:-1].dropna().astype(float).values

                if len(test_vals) != len(retest_vals):
                    continue

                diffs = test_vals - retest_vals
                sem = np.std(diffs, ddof=1) / np.sqrt(2)
                mdc = 1.96 * np.sqrt(2) * sem

                test_mean = np.mean(test_vals)
                test_std = np.std(test_vals, ddof=1)
                cov_test = (test_std / test_mean) * 100 if test_mean != 0 else np.nan

                retest_mean = np.mean(retest_vals)
                retest_std = np.std(retest_vals, ddof=1)
                cov_retest = (retest_std / retest_mean) * 100 if retest_mean != 0 else np.nan

                summary_df = pd.DataFrame({
                    "Metric": ["SEM", "MDC", "CoV_Test [%]", "CoV_Retest [%]"],
                    "Value": [round(sem, 2), round(mdc, 2), round(cov_test, 2), round(cov_retest, 2)]
                })

                start_row = table.shape[0] + 4
                summary_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)

            except Exception as e:
                print(f"Skipping sheet {sheet_name} due to error: {e}")
def export_STD_CoV_for_trials(json_path: str, force_component: str) -> None:
    import pandas as pd
    data = load_data(json_path)
    sides = ["G1R", "G2L"]

    for file_key, file_content in data.items():
        file_id = extract_file_identity(file_key)
        excel_path = os.path.join(
            os.path.dirname(json_path),
            f"{file_id}_trial_STD_CoV_{force_component}.xlsx"
        )

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for side in sides:
                intervals = file_content.get(side, {}).get("intervals", {})
                curves = []

                for int_key, interval_data in intervals.items():
                    if int_key == "Mean-Metrics":
                        continue
                    series = interval_data.get("interval_data", {})
                    t = np.array(series.get("Time [s]", []), dtype=float)
                    force_key = construct_force_key(force_component, side)
                    f = np.array(series.get(force_key, []), dtype=float)
                    if len(t) < 2 or len(f) < 2:
                        continue
                    sort_idx = np.argsort(t)
                    t, f = t[sort_idx], f[sort_idx]
                    t = t - t[0]
                    t = t / t[-1] if t[-1] != 0 else t
                    curves.append((t, f))

                if not curves:
                    continue

                t_common = np.linspace(0, 1, 200)
                interpolated = interpolate_curves(curves, t_common)
                interpolated = np.array(interpolated)

                mean = np.nanmean(interpolated, axis=0)
                std = np.nanstd(interpolated, axis=0)
                cov = np.divide(std, mean, out=np.full_like(std, np.nan), where=mean!=0) * 100

                df = pd.DataFrame({
                    "Mean": mean,
                    "STD": std,
                    "CoV [%]": cov
                })

                summary = df[["STD", "CoV [%]"]].agg(["min", "max", "mean"])
                df.loc["min"] = [""] + list(summary.loc["min"])
                df.loc["max"] = [""] + list(summary.loc["max"])
                df.loc["mean"] = [""] + list(summary.loc["mean"])

                df.to_excel(writer, sheet_name=side, index=False)
        print(f"Saved trial STD/CoV data to: {excel_path}")


# --- Utility function: ICC(3,1) manual calculation ---
def calculate_icc_3_1(data_matrix: np.ndarray) -> float:
    """
    Calculate ICC(3,1) manually from a 2D numpy array.
    Each row = subject, each column = repeated measures (e.g., Test and Retest).

    Parameters:
        data_matrix (np.ndarray): shape (n_subjects, n_sessions)

    Returns:
        float: ICC(3,1) value
    """
    n, k = data_matrix.shape  # n = subjects, k = sessions

    # Mean calculations
    grand_mean = np.mean(data_matrix)
    subject_means = np.mean(data_matrix, axis=1)
    session_means = np.mean(data_matrix, axis=0)

    # Sum of Squares
    ss_total = np.sum((data_matrix - grand_mean) ** 2)
    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    ss_sessions = n * np.sum((session_means - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_sessions

    # Degrees of freedom
    df_subjects = n - 1
    df_error = (n - 1) * (k - 1)

    # Mean Squares
    ms_subjects = ss_subjects / df_subjects
    ms_error = ss_error / df_error

    # ICC(3,1) formula
    icc = (ms_subjects - ms_error) / (ms_subjects + (k - 1) * ms_error)
    
    return icc

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import ttk

    # --- Build Settings Window (with scrollable frame) ---
    root = tk.Tk()
    root.title("Plot Settings")

    # --- Create scrollable frame ---
    canvas = tk.Canvas(root, borderwidth=0, highlightthickness=0)
    vscrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscrollbar.set)
    vscrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    scrollable_frame = tk.Frame(canvas)
    # Attach the frame to the canvas
    scrollable_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    # Variables bound to each control
    show_trial_mean_per_file = tk.BooleanVar(value=False)
    show_trial_std_per_file = tk.BooleanVar(value=False)
    tk.Checkbutton(scrollable_frame, text="     Option: Show Mean per Trial", variable=show_trial_mean_per_file).pack(anchor="w", padx=10, pady=2)
    tk.Checkbutton(scrollable_frame, text="     Option: Show STD per Trial", variable=show_trial_std_per_file).pack(anchor="w", padx=10, pady=2)
    show_global_mean = tk.BooleanVar(value=True)
    tk.Checkbutton(scrollable_frame, text="     Option: Show Global Mean", variable=show_global_mean).pack(anchor="w", padx=10, pady=2) 
    # --- Mean Line Width and Style Controls ---
    GUI_mean_linewidth = tk.DoubleVar(value=1.5)
    GUI_mean_linestyle = tk.StringVar(value="-")

    # Mean Line Width
    tk.Label(scrollable_frame, text="Mean Line Width:").pack(anchor="w", padx=10, pady=(10, 0))
    tk.Entry(scrollable_frame, textvariable=GUI_mean_linewidth).pack(anchor="w", padx=20)
    # Mean Line Style
    tk.Label(scrollable_frame, text="Mean Line Style:").pack(anchor="w", padx=10, pady=(10, 0))
    ttk.OptionMenu(scrollable_frame, GUI_mean_linestyle, "-", "-", "--", "-.", ":").pack(anchor="w", padx=20)
    canceled = tk.BooleanVar(value=False)
    exec_plot1 = tk.BooleanVar(value=False)
    exec_plot2 = tk.BooleanVar(value=False)
    GUI_norm = tk.BooleanVar(value=False)
    GUI_showstd = tk.BooleanVar(value=False)
    GUI_force_comp = tk.StringVar(value="Fy")
    GUI_save = tk.BooleanVar(value=False)
    GUI_only_combined = tk.BooleanVar(value=False)
    rel_frame = tk.LabelFrame(scrollable_frame, text="Reliability Metric", padx=10, pady=5)
    rel_frame.pack(fill="both", expand="yes", padx=10, pady=(10, 0))
    GUI_show_raw_points = tk.BooleanVar(value=False)
    GUI_filename_suffix = tk.StringVar(value="")

    do_export_std_cov_trials = tk.BooleanVar(value=False)

    # New BooleanVar for loop all enabled
    loop_forces_enabled = tk.BooleanVar(value=False)

    GUI_exclude_interval = tk.StringVar(value="")
    tk.Label(scrollable_frame, text="Exclude Interval #:").pack(anchor="w", padx=10, pady=(10, 0))
    tk.Entry(scrollable_frame, textvariable=GUI_exclude_interval).pack(anchor="w", padx=20)

    # Checkboxes
    tk.Checkbutton(scrollable_frame, text="Function per trial:    Plot Force Intervals / or mean of single trial",     variable=exec_plot1).pack(anchor="w", padx=10, pady=2)
    tk.Checkbutton(scrollable_frame, text="Plot Test-Retest comparrison Mean & STD",variable=exec_plot2).pack(anchor="w", padx=10, pady=2)
    tk.Checkbutton(scrollable_frame, text="     Option: Show Only Combined Mean ± STD", variable=GUI_only_combined).pack(anchor="w", padx=10, pady=2)
    tk.Checkbutton(scrollable_frame, text="     Option: Normalize Time",           variable=GUI_norm).pack(anchor="w", padx=10, pady=2)
    tk.Checkbutton(scrollable_frame, text="     Option Show STD Band",            variable=GUI_showstd).pack(anchor="w", padx=10, pady=2)
    tk.Checkbutton(scrollable_frame, text="Save Figures as PNG",      variable=GUI_save).pack(anchor="w", padx=10, pady=2)
    tk.Label(scrollable_frame, text="Filename Suffix:").pack(anchor="w", padx=10, pady=(10, 0))
    tk.Entry(scrollable_frame, textvariable=GUI_filename_suffix).pack(anchor="w", padx=20)
    tk.Checkbutton(rel_frame, text="Export STD and CoV for trials", variable=do_export_std_cov_trials).pack(anchor="w")

    tk.Checkbutton(scrollable_frame, text="     Option: Show Raw Data Points", variable=GUI_show_raw_points).pack(anchor="w", padx=10, pady=2)


    # Dropdown for force-component
    tk.Label(scrollable_frame, text="Force Component:").pack(anchor="w", padx=10, pady=(10,0))
    ttk.OptionMenu(scrollable_frame, GUI_force_comp, GUI_force_comp.get(), "Fy", "Fz", "Fx").pack(anchor="w", padx=20)

    # Frame for Looping combinations
    loop_frame = tk.LabelFrame(scrollable_frame, text="Looping", padx=10, pady=5)
    loop_frame.pack(fill="both", expand="yes", padx=10, pady=(10, 0))

    # Add a checkbox to enable looping
    tk.Checkbutton(loop_frame, text="Enable Looping", variable=loop_forces_enabled).pack(anchor="w", padx=20)

    loop_forces = {
        "Fy": tk.BooleanVar(value=False),
        "Fz": tk.BooleanVar(value=False),
        "Fx": tk.BooleanVar(value=False),
        "Mz": tk.BooleanVar(value=False),
    }

    loop_metrics = {
        "mean": tk.BooleanVar(value=False),
        "max": tk.BooleanVar(value=False),
        "duration_s": tk.BooleanVar(value=False),
    }

    tk.Label(loop_frame, text="Forces:").pack(anchor="w", padx=10)
    for name, var in loop_forces.items():
        tk.Checkbutton(loop_frame, text=name, variable=var).pack(anchor="w", padx=20)

    tk.Label(loop_frame, text="Metrics:").pack(anchor="w", padx=10)
    for name, var in loop_metrics.items():
        tk.Checkbutton(loop_frame, text=name, variable=var).pack(anchor="w", padx=20)

    calc_rel_metric = tk.BooleanVar(value=False)
    tk.Checkbutton(rel_frame, text="Calculate Reliability Metrics", variable=calc_rel_metric).pack(anchor="w")
    append_stats = tk.BooleanVar(value=False)
    tk.Checkbutton(rel_frame, text="Append SEM/MDC/CoV to Excel", variable=append_stats).pack(anchor="w")
    tk.Label(rel_frame, text="Metric:").pack(anchor="w", padx=10, pady=(5, 0))
    re_metric_selection = tk.StringVar(value="mean")
    ttk.OptionMenu(rel_frame, re_metric_selection, "mean", "mean", "max", "duration_s").pack(anchor="w", padx=20)

    # OK button to close dialog
    # Cancel button to terminate without action
    def cancel_and_quit():
        canceled.set(True)
        root.destroy()

    tk.Button(scrollable_frame, text="OK", command=root.destroy).pack(pady=(15,10))
    tk.Button(scrollable_frame, text="Cancel", command=cancel_and_quit).pack(pady=(0,10))

    # Configure scrolling: update scrollregion when the frame changes
    def _on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    scrollable_frame.bind("<Configure>", _on_frame_configure)
    # Allow mousewheel scrolling (Windows/Mac/Linux)
    def _on_mousewheel(event):
        if event.num == 5 or event.delta == -120:
            canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta == 120:
            canvas.yview_scroll(-1, "units")
    # Bind both standard and Mac events
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_mousewheel)
    canvas.bind_all("<Button-5>", _on_mousewheel)

    # Center window on the screen
    root.update_idletasks()
    win_width = root.winfo_width()
    win_height = root.winfo_height()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - win_width) // 2
    y = (screen_height - win_height) // 2
    root.geometry(f"{win_width}x{win_height}+{x}+{y}")
    root.mainloop()
    GUI_show_points = GUI_show_raw_points.get()
    mean_linewidth_val = GUI_mean_linewidth.get()
    mean_linestyle_val = GUI_mean_linestyle.get()
    filename_suffix_val = GUI_filename_suffix.get()
    GUI_show_trial_mean = show_trial_mean_per_file.get()
    GUI_show_trial_std = show_trial_std_per_file.get()

    # --- Read settings back into variables ---
    selected_loop_forces = [name for name, var in loop_forces.items() if var.get()]
    selected_loop_metrics = [name for name, var in loop_metrics.items() if var.get()]
    # Extract value for loop_forces_enabled (renamed to loop_all)
    loop_all = loop_forces_enabled.get()
    if canceled.get():
        import sys
        sys.exit()
    calc_reliability_metrics                  = calc_rel_metric.get()
    selected_re_metric                     = re_metric_selection.get()
    exec_plot_interval_for_single_test_function = exec_plot1.get()
    exec_plot_compare_and_combine_test_Retest_function    = exec_plot2.get()
    GUI_time_normalization              = GUI_norm.get()
    GUI_show_std                        = GUI_showstd.get()
    GUI_force_component                 = GUI_force_comp.get()
    GUI_save_flag                       = GUI_save.get()
    do_append_stats                    = append_stats.get()
    GUI_combined_only = GUI_only_combined.get()
    export_std_cov_trials = do_export_std_cov_trials.get()
    GUI_show_global_mean = show_global_mean.get()
   # calculate_icc_flag = calculate_icc.get()

    # Path to your JSONs (adjust as needed)
    import glob
    json_dir ="/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Griffe"
    #json_dir = "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Reliability/071_endurace"
    json_paths = glob.glob(os.path.join(json_dir, "*summary.json"))
    selected_loop_forces = [name for name, var in loop_forces.items() if var.get()]
    for force_component in selected_loop_forces:
        GUI_force_component = force_component
        # Conditionally call plotting functions
        if exec_plot_interval_for_single_test_function:
            for json_path in json_paths:
                plot_interval_for_single_test(
                    json_path=json_path,
                    force_component=GUI_force_component,
                    normalize_time=GUI_time_normalization,
                    show_std=GUI_show_std,
                    save_flag=GUI_save_flag,
                    exclude_interval=GUI_exclude_interval.get(),
                    show_points=GUI_show_points,
                    mean_linewidth=mean_linewidth_val,
                    mean_linestyle=mean_linestyle_val,
                    filename_suffix = filename_suffix_val,
                    show_trial_mean=GUI_show_trial_mean,
                    show_trial_std=GUI_show_trial_std,
                    show_global_mean=GUI_show_global_mean,
                )

        if exec_plot_compare_and_combine_test_Retest_function:
            for json_path in json_paths:
                plot_compare_and_combine_test_Retest(
                    json_path=json_path,
                    force_component=GUI_force_component,
                    normalize_time=GUI_time_normalization,
                    save_flag=GUI_save_flag,
                    only_combined=GUI_combined_only,
                    filename_suffix=filename_suffix_val,
                    mean_linewidth=mean_linewidth_val,
                    mean_linestyle=mean_linestyle_val
                )

        if calc_reliability_metrics:
            if selected_loop_forces and selected_loop_metrics:
                for f in selected_loop_forces:
                    for m in selected_loop_metrics:
                        for json_path in json_paths:
                            calculate_reliability_metric(
                                json_path=json_path,
                                re_metric=m,
                                force_component=f
                            )
            else:
                for json_path in json_paths:
                    calculate_reliability_metric(
                        json_path=json_path,
                        re_metric=selected_re_metric,
                        force_component=GUI_force_component
                    )

        # Optionally run post-processing on Excel output
        if do_append_stats:
            # Update with actual output file if dynamic
            # Keep the loop over output_dir as is
            output_dir = json_dir
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.startswith("Reliability_") and file.endswith(".xlsx"):
                        append_reliability_stats_to_excel(os.path.join(root, file))
            print("SEM, MDC, and CoV appended to reliability Excel files.")

        if export_std_cov_trials:
            for json_path in json_paths:
                export_STD_CoV_for_trials(
                    json_path=json_path,
                    force_component=GUI_force_component
                )


    

