import os
import json
import pandas as pd
#mport pingouin as pg

from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib import colormaps

from config import COLOR_MAPPING
from config import file_acronyms_map
from plotUITLS import pretty_component


def finalize_legend_last_std(ax):
    """Ensure legend shows once and any entry containing 'STD' appears last.
    Preserves insertion order for other labels and removes duplicates.
    """
    handles, labels = ax.get_legend_handles_labels()
    # Keep only visible, labeled entries
    pairs = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
    # De-duplicate while preserving first occurrence
    seen = set()
    uniq = []
    for h, l in pairs:
        if l in seen:
            continue
        seen.add(l)
        uniq.append((h, l))
    # Partition so that any label containing 'STD' is moved to the end
    std_pairs = [(h, l) for h, l in uniq if "STD" in l]
    nonstd_pairs = [(h, l) for h, l in uniq if "STD" not in l]
    if uniq:
        ax.legend([h for h, _ in nonstd_pairs + std_pairs], [l for _, l in nonstd_pairs + std_pairs])
    else:
        ax.legend()

# --- Utility: detect if axes actually contains plotted data (lines, collections, patches, images)
def _axes_has_content(ax):
    return bool(ax.lines or ax.collections or ax.patches or ax.images)



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
    """Return the JSON key for force_component and side_key, handling units and when to omit side suffix."""
    side_suffix = "1" if side_key == "G1R" else "2"

    # Components that are already aggregated (no per-sensor suffix in JSON)
    no_suffix_components = {"FgR_sum", "Fres_yz", "Fres_xyz", "Fres",
                            "Fy_sum", "Fz_sum", "Fx_sum"}

    unit_map = {
        "Fy": " [%]",
        "Fz": " [%]",
        "Fx": " [%]",
        "Mz": " [Nm]",
        "FgR": " [%]",
        "FgR_sum": " [%]",
        "Fy_sum": " [%]",
        "Fz_sum": " [%]",
        "Fx_sum": " [%]",
        "Fres_yz": "",
        "Fres_xyz": "",
        "Fres": " [%]"
    }
    unit = unit_map.get(force_component, "")

    # Special handling for FgR: use side-specific grip force
    if force_component == "FgR":
        return f"FgR_{side_suffix}{unit}"
    
    # Components that are already aggregated (no per-sensor suffix in JSON)
    if force_component in no_suffix_components:
        # e.g., "FgR_sum [%]" or "Fres_yz"
        return f"{force_component}{unit}"

    # Default: per-sensor forces/moment use side suffix, e.g., "Fy_1 [%]"
    return f"{force_component}_{side_suffix}{unit}"

# --- Utility: choose y-label based on filename and force/moment ---
from typing import Optional
def _ylabel_by_units(json_path: str, force_component: Optional[str], dual_mode: bool = False) -> str:
    """
    Decide y-axis label based on whether the JSON filename indicates normalization by body weight.
    If the filename contains '_NBW', use percent body weight units; otherwise use SI units.
    - Forces: F [\%BW]  or F [N]
    - Moments: M [\%BW·m] or M [Nm]
    In dual_mode (Fy/Fz together), always treat as forces.
    """
    bw_norm = "_NBW" in os.path.basename(json_path)
    is_moment = (force_component is not None) and str(force_component).startswith("M")
    if dual_mode:
        return "F [%BW]" if bw_norm else "F [N]"
    if is_moment:
        return "M [%BW·m]" if bw_norm else "M [Nm]"
    else:
        return "F [%BW]" if bw_norm else "F [N]"

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
    label_str = f"Mean–{pretty_component(force_component)}"
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
    second_force_component: Optional[str] = None,
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


    dual_mode = second_force_component is not None
    if dual_mode:
        # Keep the implementation simple: only raw interval overlays in dual mode
        show_trial_mean = False
        show_trial_std = False
        show_global_mean = False

    cmap = colormaps["tab10"]

    def _collect_intervals_for_component(component: str) -> Dict[str, list]:
        sides_local = {"G1R": [], "G2L": []}
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
                    force_key = construct_force_key(component, side_key)
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
                    sides_local[side_key].append({
                        "t": t,
                        "f": f,
                        "file": file_key,
                        "color": cmap(file_idx % 10),
                        "alpha": 0.3 + 0.7 * (1 - int_idx / max(1, len(intervals)-1)),
                        "interval_index": int_idx,
                    })
        return sides_local

    sides = _collect_intervals_for_component(force_component)
    sides_second = _collect_intervals_for_component(second_force_component) if dual_mode else {"G1R": [], "G2L": []}

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
                print(f"[DEBUG] Processing file: {file_key} with {len(curves)} curves")
                print(f"[DEBUG] {file_key}:")
                for i, (t_vals, _) in enumerate(curves):
                    print(f"  Curve {i}: t[0]={t_vals[0]:.3f}, t[-1]={t_vals[-1]:.3f}, len={len(t_vals)}")
                t_min = min(curve[0][0] for curve in curves)
                t_max = max(curve[0][-1] for curve in curves)
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

    def plot_side(intervals: List[Dict[str, Any]],
                  intervals_second: List[Dict[str, Any]],
                  side_label: str) -> None:
        plt.figure(figsize=(10, 5))
        all_f_interp = []
        files_seen = {}

        # Style cycle for up to 4 intervals: 1st solid, 2nd dashed, 3rd dotted, 4th dash-dot
        style_cycle = ['-', '--', ':', '-.']
        # Enforce component-specific colors in dual mode
        color_primary = COLOR_MAPPING.get(force_component)
        color_secondary = COLOR_MAPPING.get(second_force_component) if dual_mode else None
        # Sort intervals by original interval index for consistent styling across components
        def _sorted_by_interval_index(items):
            return sorted(items, key=lambda d: d.get('interval_index', 0))
        intervals = _sorted_by_interval_index(intervals)
        intervals_second = _sorted_by_interval_index(intervals_second)

        t_min = min((curve["t"][0] for curve in intervals if len(curve["t"]) > 0), default=0)
        t_max = max((curve["t"][-1] for curve in intervals if len(curve["t"]) > 0), default=1)
        t_common = np.linspace(t_min, t_max, 200)

        # Add flag to track if the STD legend label has been shown
        showed_std_legend = False

        for interval in intervals:
            t = interval["t"]
            f = interval["f"]
            # Override color in dual mode to make Fy/Fz visually distinct
            color = color_primary if (dual_mode and color_primary) else interval["color"]
            alpha = interval["alpha"]
            file = interval["file"]
            side_number = side_label[-1]  # Extract "R" or "L"
            file_identity = extract_file_identity(file)
            from config import file_acronyms_map
            base_label = file_acronyms_map.get(file_identity, file_identity)
            comp_label = pretty_component(force_component)
            base_with_comp = f"{base_label}–{comp_label}"
            label = base_with_comp if file not in files_seen else f"_{base_with_comp}"
            files_seen[file] = True

            idx = interval.get('interval_index', 0)
            linestyle = style_cycle[min(idx, len(style_cycle)-1)] if dual_mode else '-'
            plt.plot(t, f, color=color, alpha=alpha, linestyle=linestyle, label=label)
            try:
                interp_f = interp1d(t, f, kind='linear', bounds_error=False, fill_value=np.nan)(t_common)
                all_f_interp.append(interp_f)
            except Exception:
                continue
            # Move the show_points block here so "x" markers are drawn for each interval
            if show_points:
                plt.plot(t, f, ".", color=color, alpha=alpha * 0.9)
            # Show STD band if present in interval (for trial mean+std)
            if show_std and "std" in interval:
                std = interval["std"]
                std_label = "±1 STD" if not showed_std_legend else None
                plt.fill_between(t, f - std, f + std, color=color, alpha=0.3, label=std_label)
                showed_std_legend = True


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
        if dual_mode and intervals_second:
            # Plot the second force component with component color and consistent style
            for interval in intervals_second:
                t = interval["t"]
                f = interval["f"]
                color = color_secondary if color_secondary else interval["color"]
                alpha = interval["alpha"]
                file = interval["file"]
                file_identity = extract_file_identity(file)
                from config import file_acronyms_map
                base_label = file_acronyms_map.get(file_identity, file_identity)
                label = f"{base_label}–{pretty_component(second_force_component)}"
                idx = interval.get('interval_index', 0)
                linestyle = style_cycle[min(idx, len(style_cycle)-1)]
                plt.plot(t, f, linestyle=linestyle, color=color, alpha=alpha, label=label)

        plt.xlabel("Time [s]" if not normalize_time else "Normalized Time")
        # Choose y-label based on filename units and component
        ylabel = _ylabel_by_units(json_path, force_component, dual_mode=dual_mode)
        plt.ylabel(ylabel)
        y_min = min((min(curve["f"]) for curve in intervals if len(curve["f"]) > 0), default=0)
        y_max = max((max(curve["f"]) for curve in intervals if len(curve["f"]) > 0), default=80)
        #plt.ylim(y_min - 30, 80)
        title_str = f"{pretty_component(force_component)} – {side_label}"
        plt.title(title_str)
        plt.grid(True)
        ax = plt.gca()
        # If nothing meaningful was drawn, do not save or show
        if not _axes_has_content(ax):
            print("No meaningful data to plot for", title_str)
            plt.close(plt.gcf())
            return

        finalize_legend_last_std(ax)
        plt.tight_layout()
        base_dir = os.path.dirname(json_path)
        # Create subfolder for plots under .../plots/G1R or .../plots/G2L
        subfolder = os.path.join(base_dir, "plots", side_label)
        os.makedirs(subfolder, exist_ok=True)
        # Use file_identity from the first interval if available, else empty string
        file_identity = extract_file_identity(intervals[0]["file"]) if intervals and "file" in intervals[0] else ""
        # Adjust x-limit based on specific file identities
        x_limit_ids = {"055", "056", "057", "058", "059", "060", "061", "062", "063"}
        if file_identity in x_limit_ids:
            plt.xlim(-0.02, 0.55)
        filename = f"{file_identity}_interval_overlay_{force_component}-{side_label[-1]}"
        # If both components are present, reflect this in the filename
        if dual_mode and second_force_component:
            filename += f"_{force_component}+{second_force_component}"
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
        # Append _NBW to exported filename if the source JSON indicates normalization
        if "_NBW" in os.path.basename(json_path):
            filename += "_NBW"
        filename += ".png"
        filepath = os.path.join(subfolder, filename)
        if save_flag:
            print(f"Saving plot to {filepath}")
            title_obj = plt.gca().title
            title_text = title_obj.get_text()
            title_obj.set_text("")  # Remove title before saving
            plt.savefig(filepath, dpi=700)
            title_obj.set_text(title_text)  # Restore title for display
        plt.show()
    
    plot_side(sides["G1R"], sides_second["G1R"], "G1R")
    plot_side(sides["G2L"], sides_second["G2L"], "G2L")



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
                from config import file_acronyms_map
                label_name = file_acronyms_map.get(file_identity, file_identity)
                label_name = f"{label_name}–{pretty_component(force_component)}"
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
            # Append _NBW to export name if the source JSON indicates normalization
            nbw_tag = "_NBW" if "_NBW" in os.path.basename(json_path) else ""
            excel_path = os.path.join(subfolder, f"combined_mean_STD_{force_component}_{side_key}{nbw_tag}.xlsx")
            df_export.to_excel(excel_path, index=False)
            print(f"Saved combined mean/STD to: {excel_path}")

            plt.plot(
                t_common,
                combined_mean,
                color="black",
                linewidth=mean_linewidth,
                linestyle=mean_linestyle,
                label=f"Combined–{pretty_component(force_component)}"
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
        ylabel = _ylabel_by_units(json_path, force_component, dual_mode=False)
        plt.ylabel(ylabel)
        plt.grid(True)
        ax = plt.gca()
        # Skip saving/showing if figure has no plotted content
        if not _axes_has_content(ax):
            plt.close(plt.gcf())
            continue

        finalize_legend_last_std(ax)
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
        # Append _NBW to exported filename if the source JSON indicates normalization
        if "_NBW" in os.path.basename(json_path):
            filename += "_NBW"
        filename += ".png"
        filepath = os.path.join(subfolder, filename)
        if save_flag:
            print(f"Saving plot to {filepath}")
            plt.savefig(filepath, dpi=700)
        plt.title(f"{pretty_component(force_component)} – Mean ± STD per File and Combined – {side_key}")
        plt.show()
def calculate_reliability_metric(json_path: str, re_metric: str, force_component: str,) -> None:
    """
    Extracts force metric data from JSON for test-retest reliability evaluation and exports to Excel.

    Parameters:
        json_path (str): Path to the summary metadata JSON file.
        re_metric (str): Metric to extract; one of "mean", "max", "duration_s".
    """
    import pandas as pd
    
    # Debug: Print available data sources in the JSON
    print(f"\n=== DEBUG: Available data sources for {force_component} ===")
    data = load_data(json_path)
    for file_key, file_content in data.items():
        if "G1R" in file_content:
            g1r_intervals = file_content["G1R"].get("intervals", {})
            print(f"\nFile {file_key} - G1R intervals:")
            for int_key, int_data in g1r_intervals.items():
                if int_key != "Mean-Metrics":
                    print(f"  {int_key}: Available keys: {list(int_data.keys())}")
                    if "Fres_xyz" in int_data:
                        print(f"    Fres_xyz: {list(int_data['Fres_xyz'].keys())}")
                    if "FgR_1" in int_data:
                        print(f"    FgR_1: {list(int_data['FgR_1'].keys())}")
                    if "FgR_sum" in int_data:
                        print(f"    FgR_sum: {list(int_data['FgR_sum'].keys())}")
            
            mean_metrics = g1r_intervals.get("Mean-Metrics", {})
            print(f"  Mean-Metrics: Available keys: {list(mean_metrics.keys())}")
            if "Fres_xyz" in mean_metrics:
                print(f"    Fres_xyz: {list(mean_metrics['Fres_xyz'].keys())}")
            if "FgR_1" in mean_metrics:
                print(f"    FgR_1: {list(mean_metrics['FgR_1'].keys())}")
            if "FgR_sum" in mean_metrics:
                print(f"    FgR_sum: {list(mean_metrics['FgR_sum'].keys())}")
    
    print("=== END DEBUG ===\n")

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
                        # Try to get the force component data
                        interval_value = interval_data.get(force_component, {}).get(re_metric, None)
                        
                        # Special fallback for Fres: if not found, try Fres_xyz
                        if interval_value is None and force_component == "Fres":
                            interval_value = interval_data.get("Fres_xyz", {}).get(re_metric, None)
                            if interval_value is not None:
                                print(f"Fres: Using Fres_xyz fallback for {side}, interval {int_key}, value: {interval_value}")
                        
                        # Special handling for FgR: use side-specific grip force
                        if interval_value is None and force_component == "FgR":
                            # For FgR, use side-specific data: FgR_1 for G1R, FgR_2 for G2L
                            side_suffix = "1" if side == "G1R" else "2"
                            fgr_key = f"FgR_{side_suffix}"
                            interval_value = interval_data.get(fgr_key, {}).get(re_metric, None)
                            if interval_value is not None:
                                print(f"FgR: Using {fgr_key} for {side}, interval {int_key}, value: {interval_value}")
                        
                        # Debug: Show which data source is being used
                        if interval_value is not None:
                            print(f"Data source for {force_component} on {side}, interval {int_key}: {interval_value}")
                            # Check if this value exists in other force components
                            if force_component == "Fres":
                                fgr_check = interval_data.get("FgR_1" if side == "G1R" else "FgR_2", {}).get(re_metric, None)
                                if fgr_check is not None:
                                    print(f"  -> Same value found in FgR: {fgr_check}")
                            elif force_component == "FgR":
                                fres_check = interval_data.get("Fres_xyz", {}).get(re_metric, None)
                                if fres_check is not None:
                                    print(f"  -> Same value found in Fres: {fres_check}")

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
                print(f"records for {label}: {metric_values}")
                
                # Debug: Compare values between Fres and FgR if both are being processed
                if force_component in ["Fres", "FgR"]:
                    print(f"DEBUG: {force_component} values for {label}: {metric_values}")
                    # Check if we can access the other component's data for comparison
                    if force_component == "Fres":
                        # Try to get FgR values for comparison
                        fgr_values = []
                        for int_key, interval_data in intervals.items():
                            if int_key != "Mean-Metrics":
                                side_suffix = "1" if side == "G1R" else "2"
                                fgr_key = f"FgR_{side_suffix}"
                                fgr_val = interval_data.get(fgr_key, {}).get(re_metric, None)
                                if fgr_val is not None:
                                    fgr_values.append(fgr_val)
                        if fgr_values:
                            print(f"DEBUG: FgR values for comparison: {fgr_values}")
                    elif force_component == "FgR":
                        # Try to get Fres values for comparison
                        fres_values = []
                        for int_key, interval_data in intervals.items():
                            if int_key != "Mean-Metrics":
                                fres_val = interval_data.get("Fres_xyz", {}).get(re_metric, None)
                                if fres_val is not None:
                                    fres_values.append(fres_val)
                        if fres_values:
                            print(f"DEBUG: Fres values for comparison: {fres_values}")

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
                    # Try to get the force component data
                    mean_val = mean_metrics.get(force_component, {}).get(re_metric, None)
                    
                    # Special fallback for Fres: if not found, try Fres_xyz
                    if mean_val is None and force_component == "Fres":
                        mean_val = mean_metrics.get("Fres_xyz", {}).get(re_metric, None)
                        if mean_val is not None:
                            print(f"Fres Mean: Using Fres_xyz fallback for {side}, value: {mean_val}")
                    
                    # Special handling for FgR: use side-specific grip force
                    if mean_val is None and force_component == "FgR":
                        # For FgR, use side-specific data: FgR_1 for G1R, FgR_2 for G2L
                        side_suffix = "1" if side == "G1R" else "2"
                        fgr_key = f"FgR_{side_suffix}"
                        mean_val = mean_metrics.get(fgr_key, {}).get(re_metric, None)
                        if mean_val is not None:
                            print(f"FgR Mean: Using {fgr_key} for {side}, value: {mean_val}")
                    
                    # Debug: Show which mean data source is being used
                    if mean_val is not None:
                        print(f"Mean data source for {force_component} on {side}: {mean_val}")
                        # Check if this value exists in other force components
                        if force_component == "Fres":
                            fgr_mean_check = mean_metrics.get("FgR_1" if side == "G1R" else "FgR_2", {}).get(re_metric, None)
                            if fgr_mean_check is not None:
                                print(f"  -> Same mean value found in FgR: {fgr_mean_check}")
                        elif force_component == "FgR":
                            fres_mean_check = mean_metrics.get("Fres_xyz", {}).get(re_metric, None)
                            if fres_mean_check is not None:
                                print(f"  -> Same mean value found in Fres: {fres_mean_check}")
                if mean_val is not None:
                    df.loc[0 if label == "Test" else 1, "Mean"] = mean_val

            # Export to Excel starting at row 3 (so table name can be written manually in row 1)
            df.to_excel(writer, sheet_name=side, startrow=2, index=False)
            worksheet = writer.sheets[side]
            worksheet.cell(row=1, column=1, value=table_name)

        print(f"Reliability metric exported to: {out_path}")
        
        # Final debug summary
        print(f"\n=== FINAL DEBUG SUMMARY for {force_component} ===")
        print(f"Total Test records: {len(records.get('Test', []))}")
        print(f"Total Retest records: {len(records.get('Retest', []))}")
        if 'Test' in records and records['Test']:
            print(f"Test values: {records['Test']}")
        if 'Retest' in records and records['Retest']:
            print(f"Retest values: {records['Retest']}")
        print("=== END FINAL DEBUG ===\n")
def append_reliability_stats_to_excel(file_path: str) -> None:
    """Load an Excel file with Test/Retest rows, compute SEM, MDC, CoV, and write results back to the same file."""
    import pandas as pd
    import numpy as np
    from openpyxl import load_workbook

    # First, read the existing file to get the data
    df = pd.read_excel(file_path, sheet_name=None)
    
    # Close the file to ensure it's not locked
    import time
    time.sleep(0.1)  # Small delay to ensure file is closed
    
    # Now append the statistics
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        for sheet_name, table in df.items():
            if table.empty or table.shape[0] < 2:
                print(f"Skipping sheet {sheet_name}: Empty or insufficient data (shape: {table.shape})")
                continue
            
            # Debug: Print table structure
            print(f"Processing sheet {sheet_name}: Shape {table.shape}, Columns: {list(table.columns)}")
            print(f"First few rows of {sheet_name}:")
            print(table.head())

            try:
                # Dynamically find rows labeled 'Test' and 'Retest' in column 0
                test_row = table[table.iloc[:, 0].astype(str).str.lower() == "test"]
                retest_row = table[table.iloc[:, 0].astype(str).str.lower() == "retest"]

                if test_row.empty or retest_row.empty:
                    print(f"Skipping sheet {sheet_name}: 'Test' or 'Retest' row not found.")
                    continue

                # Get the actual row indices for test and retest
                test_idx = test_row.index[0]
                retest_idx = retest_row.index[0]
                
                # Get all columns except the first (label) and last (mean) columns
                # This gives us the interval columns (I1, I2, I3, etc.)
                interval_columns = table.columns[1:-1] if len(table.columns) > 2 else table.columns[1:]
                
                # Extract values for the interval columns only
                test_vals = table.loc[test_idx, interval_columns].dropna().astype(float).values
                retest_vals = table.loc[retest_idx, interval_columns].dropna().astype(float).values

                if len(test_vals) == 0 or len(retest_vals) == 0:
                    print(f"Skipping sheet {sheet_name}: No valid interval data found.")
                    continue
                    
                if len(test_vals) != len(retest_vals):
                    print(f"Skipping sheet {sheet_name}: Test and Retest have different numbers of intervals ({len(test_vals)} vs {len(retest_vals)}).")
                    continue
                
                # Debug information
                print(f"Processing sheet {sheet_name}: Test values: {test_vals}, Retest values: {retest_vals}")

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
                print(f"Error type: {type(e).__name__}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
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

                # Build DataFrame with an extra 'Description' column
                df = pd.DataFrame({
                    "Description": [""] * len(mean),
                    "Mean": mean,
                    "STD": std,
                    "CoV [%]": cov
                })

                # Compute summary stats for STD and CoV only
                summary = df[["STD", "CoV [%]"]].agg(["min", "max", "mean"])

                # Append labeled summary rows
                summary_rows = pd.DataFrame([
                    {
                        "Description": "min",
                        "Mean": "",
                        "STD": summary.loc["min", "STD"],
                        "CoV [%]": summary.loc["min", "CoV [%]"],
                    },
                    {
                        "Description": "max",
                        "Mean": "",
                        "STD": summary.loc["max", "STD"],
                        "CoV [%]": summary.loc["max", "CoV [%]"],
                    },
                    {
                        "Description": "mean",
                        "Mean": "",
                        "STD": summary.loc["mean", "STD"],
                        "CoV [%]": summary.loc["mean", "CoV [%]"],
                    },
                ])
                df = pd.concat([df, summary_rows], ignore_index=True)

                df.to_excel(writer, sheet_name=side, index=False)
        print(f"Saved trial STD/CoV data to: {excel_path}")


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

    # Welcome and explanation section
    welcome_frame = tk.LabelFrame(scrollable_frame, text="📊 Plot Overlay Tool - Overview", padx=10, pady=10)
    welcome_frame.pack(fill="x", padx=10, pady=(10, 15))
    
    tk.Label(welcome_frame, text="This tool processes climbing force data from JSON files and creates:", 
             font=("Arial", 9, "bold")).pack(anchor="w")
    
    # Workflow explanation
    workflow_frame = tk.Frame(welcome_frame)
    workflow_frame.pack(fill="x", padx=10, pady=(5, 0))
    
    tk.Label(workflow_frame, text="1. 📈 Force interval plots (single trials)", font=("Arial", 8)).pack(anchor="w")
    tk.Label(workflow_frame, text="2. 🔄 Test-Retest comparisons", font=("Arial", 8)).pack(anchor="w")
    tk.Label(workflow_frame, text="3. 📊 Reliability metrics (SEM, MDC, CoV)", font=("Arial", 8)).pack(anchor="w")
    tk.Label(workflow_frame, text="4. 📋 Statistical summaries (STD, CoV per trial)", font=("Arial", 8)).pack(anchor="w")
    
    tk.Label(welcome_frame, text="Select your options below and click OK to run the analysis.", 
             font=("Arial", 8), fg="blue").pack(anchor="w", pady=(5, 0))

    # Variables bound to each control
    show_trial_mean_per_file = tk.BooleanVar(value=True)
    tk.Checkbutton(scrollable_frame, text="     Option: Show Mean per Trial", variable=show_trial_mean_per_file).pack(anchor="w", padx=10, pady=2)
    # Unified STD checkbox
    show_std_flag = tk.BooleanVar(value=False)
    tk.Checkbutton(scrollable_frame, text="     Option: Show and Compute STD (per trial)", variable=show_std_flag).pack(anchor="w", padx=10, pady=2)
    show_global_mean = tk.BooleanVar(value=False)
    tk.Checkbutton(scrollable_frame, text="     Option: Show Global Mean", variable=show_global_mean).pack(anchor="w", padx=10, pady=2)
    GUI_plot_fy_fz_together = tk.BooleanVar(value=False)
    tk.Checkbutton(scrollable_frame, text="     Option: Plot Fy and Fz together (same figure)", variable=GUI_plot_fy_fz_together).pack(anchor="w", padx=10, pady=2)
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
    plot_interval_for_single_var = tk.BooleanVar(value=True)
    exec_test_retest_var = tk.BooleanVar(value=False)
    GUI_norm = tk.BooleanVar(value=False)
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
    tk.Checkbutton(scrollable_frame, text="Function per trial:    Plot Force Intervals / or mean of single trial",     variable=plot_interval_for_single_var).pack(anchor="w", padx=10, pady=2)
    tk.Checkbutton(scrollable_frame, text="Plot Test-Retest comparrison Mean & STD",variable=exec_test_retest_var).pack(anchor="w", padx=10, pady=2)
    tk.Checkbutton(scrollable_frame, text="     Option: Show Only Combined Mean ± STD", variable=GUI_only_combined).pack(anchor="w", padx=10, pady=2)
    tk.Checkbutton(scrollable_frame, text="     Option: Normalize Time",           variable=GUI_norm).pack(anchor="w", padx=10, pady=2)
    tk.Checkbutton(scrollable_frame, text="Save Figures as PNG",      variable=GUI_save).pack(anchor="w", padx=10, pady=2)
    tk.Label(scrollable_frame, text="Filename Suffix:").pack(anchor="w", padx=10, pady=(10, 0))
    tk.Entry(scrollable_frame, textvariable=GUI_filename_suffix).pack(anchor="w", padx=20)
    tk.Checkbutton(rel_frame, text="Export STD and CoV for trials", variable=do_export_std_cov_trials).pack(anchor="w")

    tk.Checkbutton(scrollable_frame, text="     Option: Show Raw Data Points", variable=GUI_show_raw_points).pack(anchor="w", padx=10, pady=2)


    # Dropdown for force-component
    tk.Label(scrollable_frame, text="Force Component:").pack(anchor="w", padx=10, pady=(10,0))
    ttk.OptionMenu(
        scrollable_frame, GUI_force_comp, GUI_force_comp.get(),
        "Fy", "Fz", "Fx", "Fres", "FgR", "FgR_sum", "Fy_sum", "Fz_sum", "Fx_sum"
    ).pack(anchor="w", padx=20)
    
    # Force component explanations
    force_explanations = tk.Frame(scrollable_frame)
    force_explanations.pack(fill="x", padx=20, pady=(0, 10))
    
    tk.Label(force_explanations, text="Force Component Explanations:", font=("Arial", 9, "bold")).pack(anchor="w")
    
    # Create a frame for force explanations with better layout
    explanations_frame = tk.Frame(force_explanations)
    explanations_frame.pack(fill="x", padx=10)
    
    # Left column
    left_col = tk.Frame(explanations_frame)
    left_col.pack(side="left", fill="both", expand=True)
    
    tk.Label(left_col, text="• Fy: Vertical force (up/down)", font=("Arial", 8)).pack(anchor="w")
    tk.Label(left_col, text="• Fz: Forward/backward force", font=("Arial", 8)).pack(anchor="w")
    tk.Label(left_col, text="• Fx: Side-to-side force", font=("Arial", 8)).pack(anchor="w")
    tk.Label(left_col, text="• Fres: Resultant force (√Fy²+Fz²+Fx²)", font=("Arial", 8)).pack(anchor="w")
    tk.Label(left_col, text="• FgR: Side-specific grip force (FgR_1 for G1R, FgR_2 for G2L)", font=("Arial", 8)).pack(anchor="w")
    
    # Right column
    right_col = tk.Frame(explanations_frame)
    right_col.pack(side="right", fill="both", expand=True)
    
    tk.Label(right_col, text="• FgR_sum: Grip force sum", font=("Arial", 8)).pack(anchor="w")
    tk.Label(right_col, text="• Fy_sum: Sum of Y forces", font=("Arial", 8)).pack(anchor="w")
    tk.Label(right_col, text="• Fz_sum: Sum of Z forces", font=("Arial", 8)).pack(anchor="w")
    tk.Label(right_col, text="• Fx_sum: Sum of X forces", font=("Arial", 8)).pack(anchor="w")
    
    # Special notes about force components
    tk.Label(force_explanations, text="💡 Note: Fres automatically uses Fres_xyz data if available", 
             font=("Arial", 8), fg="green").pack(anchor="w", pady=(5, 0))
    tk.Label(force_explanations, text="💡 Note: FgR uses side-specific data (FgR_1 for G1R, FgR_2 for G2L)", 
             font=("Arial", 8), fg="green").pack(anchor="w", pady=(2, 0))

    # Frame for Looping combinations
    loop_frame = tk.LabelFrame(scrollable_frame, text="Looping - Process Multiple Forces & Metrics", padx=10, pady=5)
    loop_frame.pack(fill="both", expand="yes", padx=10, pady=(10, 0))

    # Add a checkbox to enable looping
    tk.Checkbutton(loop_frame, text="Enable Looping", variable=loop_forces_enabled).pack(anchor="w", padx=20)
    
    # Looping explanation
    tk.Label(loop_frame, text="Looping allows you to process multiple force components and metrics automatically.", 
             font=("Arial", 8), fg="blue").pack(anchor="w", padx=20, pady=(5, 0))
    tk.Label(loop_frame, text="When enabled, the script will run for each selected force and metric combination.", 
             font=("Arial", 8), fg="blue").pack(anchor="w", padx=20)
    
    # Looping tip
    tk.Label(loop_frame, text="💡 Tip: Use looping to batch-process multiple analyses at once!", 
             font=("Arial", 8), fg="green").pack(anchor="w", padx=20, pady=(5, 0))

    loop_forces = {
        "Fy": tk.BooleanVar(value=False),
        "Fz": tk.BooleanVar(value=False),
        "Fx": tk.BooleanVar(value=False),
        "Mz": tk.BooleanVar(value=False),
        "Fres": tk.BooleanVar(value=False),
        "FgR": tk.BooleanVar(value=False),
        "FgR_sum": tk.BooleanVar(value=False),
        "Fy_sum": tk.BooleanVar(value=False),
        "Fz_sum": tk.BooleanVar(value=False),
        "Fx_sum": tk.BooleanVar(value=False),
    }

    loop_metrics = {
        "mean": tk.BooleanVar(value=False),
        "max": tk.BooleanVar(value=False),
        "duration_s": tk.BooleanVar(value=False),
    }

    tk.Label(loop_frame, text="Forces (select which force components to process):", font=("Arial", 9, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
    
    # Create a frame for force checkboxes with hints
    forces_frame = tk.Frame(loop_frame)
    forces_frame.pack(fill="x", padx=20)
    
    # Left column for forces
    forces_left = tk.Frame(forces_frame)
    forces_left.pack(side="left", fill="both", expand=True)
    
    # Right column for force hints
    forces_right = tk.Frame(forces_frame)
    forces_right.pack(side="right", fill="both", expand=True)
    
    # Add force checkboxes with hints
    force_hints = {
        "Fy": "Vertical force",
        "Fz": "Forward/backward force", 
        "Fx": "Side-to-side force",
        "Mz": "Moment around Z-axis",
        "Fres": "Resultant force magnitude",
        "FgR": "Side-specific grip force (FgR_1/FgR_2)",
        "FgR_sum": "Grip force sum",
        "Fy_sum": "Sum of Y forces",
        "Fz_sum": "Sum of Z forces",
        "Fx_sum": "Sum of X forces"
    }
    
    for name, var in loop_forces.items():
        # Create a frame for each force checkbox and hint
        force_row = tk.Frame(forces_frame)
        force_row.pack(fill="x", pady=1)
        
        # Checkbox on the left
        tk.Checkbutton(force_row, text=name, variable=var).pack(side="left")
        
        # Hint on the right
        hint_text = force_hints.get(name, "")
        tk.Label(force_row, text=f"({hint_text})", font=("Arial", 7), fg="gray").pack(side="right", padx=(10, 0))

    tk.Label(loop_frame, text="Metrics (select which statistical measures to calculate):", font=("Arial", 9, "bold")).pack(anchor="w", padx=10, pady=(15, 5))
    
    # Create a frame for metric checkboxes with hints
    metrics_frame = tk.Frame(loop_frame)
    metrics_frame.pack(fill="x", padx=20)
    
    # Add metric checkboxes with hints
    metric_hints = {
        "mean": "Average value across intervals",
        "max": "Peak/maximum value",
        "duration_s": "Contact time duration"
    }
    
    for name, var in loop_metrics.items():
        # Create a frame for each metric checkbox and hint
        metric_row = tk.Frame(metrics_frame)
        metric_row.pack(fill="x", pady=1)
        
        # Checkbox on the left
        tk.Checkbutton(metric_row, text=name, variable=var).pack(side="left")
        
        # Hint on the right
        hint_text = metric_hints.get(name, "")
        tk.Label(metric_row, text=f"({hint_text})", font=("Arial", 7), fg="gray").pack(side="right", padx=(10, 0))

    # Reliability metrics explanation
    tk.Label(rel_frame, text="Reliability Analysis - Test-Retest Consistency:", font=("Arial", 9, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
    tk.Label(rel_frame, text="Calculates SEM (Standard Error of Measurement), MDC (Minimal Detectable Change),", 
             font=("Arial", 8), fg="blue").pack(anchor="w", padx=20)
    tk.Label(rel_frame, text="and CoV (Coefficient of Variation) for test-retest reliability assessment.", 
             font=("Arial", 8), fg="blue").pack(anchor="w", padx=20)
    
    calc_rel_metric = tk.BooleanVar(value=False)
    tk.Checkbutton(rel_frame, text="Calculate Reliability Metrics", variable=calc_rel_metric).pack(anchor="w")
    
    append_stats = tk.BooleanVar(value=False)
    tk.Checkbutton(rel_frame, text="Append SEM/MDC/CoV to Excel", variable=append_stats).pack(anchor="w")
    
    tk.Label(rel_frame, text="Reliability Metric Type:", font=("Arial", 9, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
    re_metric_selection = tk.StringVar(value="mean")
    ttk.OptionMenu(rel_frame, re_metric_selection, "mean", "mean", "max", "duration_s").pack(anchor="w", padx=20)
    
    # Metric type explanations
    metric_explanations = tk.Frame(rel_frame)
    metric_explanations.pack(fill="x", padx=20, pady=(5, 10))
    
    tk.Label(metric_explanations, text="• mean: Average force across intervals (most common)", font=("Arial", 7), fg="gray").pack(anchor="w")
    tk.Label(metric_explanations, text="• max: Peak force values (for strength assessment)", font=("Arial", 7), fg="gray").pack(anchor="w")
    tk.Label(metric_explanations, text="• duration_s: Contact time duration (for timing analysis)", font=("Arial", 7), fg="gray").pack(anchor="w")

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
    root.geometry("800x1200")
    root.mainloop()
    GUI_show_points = GUI_show_raw_points.get()
    mean_linewidth_val = GUI_mean_linewidth.get()
    mean_linestyle_val = GUI_mean_linestyle.get()
    filename_suffix_val = GUI_filename_suffix.get()
    GUI_show_trial_mean = show_trial_mean_per_file.get()
    GUI_show_trial_std = show_std_flag.get()
    GUI_dual_fy_fz = GUI_plot_fy_fz_together.get()

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
    exec_plot_interval_for_single_test_function = plot_interval_for_single_var.get()
    exec_plot_compare_and_combine_test_Retest_function    = exec_test_retest_var.get()
    GUI_time_normalization              = GUI_norm.get()
    GUI_show_std                        = show_std_flag.get()
    GUI_force_component                 = GUI_force_comp.get()
    GUI_save_flag                       = GUI_save.get()
    do_append_stats                    = append_stats.get()
    GUI_combined_only = GUI_only_combined.get()
    export_std_cov_trials = do_export_std_cov_trials.get()
    GUI_show_global_mean = show_global_mean.get()
   # calculate_icc_flag = calculate_icc.get()

    # Path to your JSONs (adjust as needed)
    import glob
    json_dirs = [
       # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/worst-black/front"
       # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/medium-yellow/front"
        "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Technik/Rightside_data"
       # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Reliability/Test-Rest/further metrics"
        #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/best-grey/cross",
        #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/medium-yellow/cross",
        #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/worst-black/cross",
#        "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/best-grey/front",
#"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/worst-black/front",
#"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/medium-yellow/front",
#"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/sorted_by_shoes/front/Trail",
#"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/sorted_by_shoes/front/Perf",
#"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/sorted_by_shoes/front/HighEnd",
#"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/sorted_by_shoes/front/Basic",
        #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/best-grey/cross"
       # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/best-grey/cross+front-combined",
        #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/best-grey/front"
       # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Technik/Rightside_data",
        #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Technik/Rightside_data/hide"
       # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/ForceDevRatio"
        #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Clipping"
       # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Clipping/low",
       # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Clipping/high",
       # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/shakeout",
         #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/shakeout/GL_2vs1FH",
 # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/shakeout/GR_2vs1FH",
 # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/shakeout/Switch",
 # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/shakeout/Switch/single",
  #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/shakeout/Switch/single_I2"
 # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/shakeout/pure 2vs1FH"
 #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/shakeout/GL_2vs1FH/Frontal_vs_hipIn"
 #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/shakeout/GL_2vs1FH/only_front"
 #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/shakeout/GL_2vs1FH"
 #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/shakeout/Switch/GL"
 #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/ForceDevRatio"
# "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Moment"
# "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/shakeout"

        #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Griffe"
        #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Reliability/071_endurace"
    ]
    json_paths = []
    for dir_path in json_dirs:
        json_paths.extend(glob.glob(os.path.join(dir_path, "*summary.json")))
    json_paths = list(set(json_paths))  # Optional: remove duplicates
    selected_loop_forces = [name for name, var in loop_forces.items() if var.get()]
    for force_component in selected_loop_forces:
        GUI_force_component = force_component
        # Conditionally call plotting functions
        if exec_plot_interval_for_single_test_function:
            for json_path in json_paths:
                second_fc = None
                if GUI_dual_fy_fz and GUI_force_component in ("Fy", "Fz"):
                    # Always pair Fy with Fz regardless of selection
                    second_fc = "Fz" if GUI_force_component == "Fy" else "Fy"
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
                    second_force_component=second_fc,
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

        # Always append SEM/MDC/CoV when reliability metrics are calculated
        if calc_reliability_metrics:
            print("Appending SEM, MDC, and CoV to reliability Excel files...")
            # Process each directory separately since json_dirs is a list
            for output_dir in json_dirs:
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if file.startswith("Reliability_") and file.endswith(".xlsx"):
                            append_reliability_stats_to_excel(os.path.join(root, file))
            print("SEM, MDC, and CoV appended to reliability Excel files.")
        # Also run post-processing if explicitly requested via checkbox
        elif do_append_stats:
            print("Appending SEM, MDC, and CoV to reliability Excel files (manual request)...")
            # Process each directory separately since json_dirs is a list
            for output_dir in json_dirs:
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


    

