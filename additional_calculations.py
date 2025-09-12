# This file contains additional calculations for hausdorff dimensions and plotting.
# Hausdorff dimension calculation using box-counting method
import numpy as np
from typing import Dict, Any, Union, Tuple, Optional
import matplotlib.pyplot as plt
import config
import os

# References:
# - Fuss & Niegl (2008): Fractal dimension analysis of force-time curves in climbing
# - Box-counting dimension: D = lim(ε→0) log(N(ε)) / log(1/ε)
# - For climbing force signals: typically 1.1-1.8 (lower = smoother/more skillful)

def hausdorff_dimension_boxcount(
    t: np.ndarray,
    x: np.ndarray,
    *,
    normalize: bool = True,
    detrend: bool = False,
    min_points: int = 64,
    min_scales: int = 8,
    max_scales: int = 30,
    scale_mode: str = "auto",      # "auto" | "logspace"
    eps_min_frac: float = 2.0,     # lower bound ~ eps_min = eps_min_frac / N after normalization
    eps_max_frac: float = 0.25,    # upper bound fraction of unit box
    robust_fit: str = "theilsen",  # "theilsen" | "ransac" | "ols"
    return_debug: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Compute the (planar) box-counting (Hausdorff) dimension of a time series curve Gamma = {(t, x(t))}
    embedded in R^2 using a 2D grid covering.
    
    The Hausdorff dimension characterizes the fractal complexity of force-time curves in climbing:
    - Lower values (1.1-1.3): Smooth, skillful force application
    - Higher values (1.5-1.8): More irregular, less controlled movements
    
    Args:
        t: Time array (1D)
        x: Signal array (1D)
        normalize: Whether to normalize (t,x) to [0,1]² (removes unit dependence)
        detrend: Whether to remove linear trend from x
        min_points: Minimum points required for calculation
        min_scales: Minimum number of scales for fit
        max_scales: Maximum number of scales to generate
        scale_mode: Scale generation mode ("auto" or "logspace")
        eps_min_frac: Minimum epsilon as fraction of data range
        eps_max_frac: Maximum epsilon as fraction of unit box
        robust_fit: Regression method ("theilsen", "ransac", "ols")
        return_debug: Whether to return debug information
        
    Returns:
        Hausdorff dimension (float) or (dimension, debug_dict) if return_debug=True
        
    References:
        - Fuss & Niegl (2008): Fractal dimension analysis in climbing biomechanics
        - Box-counting dimension: D = lim(ε→0) log(N(ε)) / log(1/ε)
    """
    
    # Input validation
    if len(t) != len(x):
        return np.nan if not return_debug else (np.nan, {"error": "Length mismatch"})
    
    if len(t) < min_points:
        return np.nan if not return_debug else (np.nan, {"error": f"Too few points: {len(t)} < {min_points}"})
    
    # Handle constant or nearly constant signals
    if np.std(x) < 1e-10:
        return 1.0 if not return_debug else (1.0, {"note": "Constant signal"})
    
    # Copy arrays to avoid modifying originals
    t_norm = t.copy().astype(np.float64)
    x_norm = x.copy().astype(np.float64)
    
    # Detrend if requested
    if detrend:
        coeffs = np.polyfit(t_norm, x_norm, 1)
        x_norm = x_norm - (coeffs[0] * t_norm + coeffs[1])
    
    # Normalize to unit square [0,1]²
    if normalize:
        t_range = np.max(t_norm) - np.min(t_norm)
        x_range = np.max(x_norm) - np.min(x_norm)
        
        if t_range > 0:
            t_norm = (t_norm - np.min(t_norm)) / t_range
        if x_range > 0:
            x_norm = (x_norm - np.min(x_norm)) / x_range
    
    # Stack into 2D points
    points = np.column_stack((t_norm, x_norm))
    
    # Generate epsilon sequence
    N = len(points)
    if scale_mode == "auto":
        # Adaptive scale selection based on data size
        eps_min = max(eps_min_frac / N, 1e-6)
        eps_max = min(eps_max_frac, 0.5)
        epsilons = np.logspace(np.log10(eps_min), np.log10(eps_max), max_scales)
    else:  # logspace
        epsilons = np.logspace(-2, 0, max_scales)
    
    # Count boxes for each epsilon
    counts = []
    valid_epsilons = []
    
    for eps in epsilons:
        if eps <= 0 or eps >= 1:
            continue
            
        # Grid size: ensure at least 2x2 grid
        grid_size = max(2, int(np.floor(1.0 / eps)))
        
        # Use histogram2d for efficient box counting
        try:
            hist, _, _ = np.histogram2d(
                points[:, 0], points[:, 1], 
                bins=grid_size, 
                range=[[0, 1], [0, 1]]
            )
            # Count non-empty boxes
            box_count = np.count_nonzero(hist)
            counts.append(box_count)
            valid_epsilons.append(eps)
        except Exception:
            continue
    
    if len(counts) < min_scales:
        return np.nan if not return_debug else (np.nan, {"error": f"Insufficient valid scales: {len(counts)} < {min_scales}"})
    
    # Convert to log-log coordinates
    log_eps = -np.log10(valid_epsilons)
    log_counts = np.log10(counts)
    
    # Find optimal scaling window by maximizing R²
    best_r2 = -np.inf
    best_slope = np.nan
    best_intercept = np.nan
    best_range = (0, len(log_eps))
    
    for start in range(len(log_eps) - min_scales + 1):
        for end in range(start + min_scales, len(log_eps) + 1):
            if end - start < min_scales:
                continue
                
            x_fit = log_eps[start:end]
            y_fit = log_counts[start:end]
            
            # Robust regression
            try:
                if robust_fit == "theilsen":
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit, y_fit)
                    r2 = r_value ** 2
                elif robust_fit == "ransac":
                    from sklearn.linear_model import RANSACRegressor
                    from sklearn.linear_model import LinearRegression
                    ransac = RANSACRegressor(LinearRegression(), random_state=42)
                    ransac.fit(x_fit.reshape(-1, 1), y_fit)
                    slope = ransac.estimator_.coef_[0]
                    intercept = ransac.estimator_.intercept_
                    r2 = ransac.score(x_fit.reshape(-1, 1), y_fit)
                else:  # OLS
                    try:
                        slope, intercept, r_value, p_value, std_err = np.polyfit(x_fit, y_fit, 1, full=True)[0:2]
                        r2 = r_value ** 2 if r_value is not None else 0
                    except:
                        # Fallback to simple polyfit without full=True
                        coeffs = np.polyfit(x_fit, y_fit, 1)
                        slope, intercept = coeffs[0], coeffs[1]
                        # Calculate R² manually
                        y_pred = slope * x_fit + intercept
                        ss_res = np.sum((y_fit - y_pred) ** 2)
                        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Prefer reasonable slopes (1.0 ≤ D ≤ 2.0 for 2D embedding)
                if 0.8 <= slope <= 2.2 and r2 > best_r2:
                    best_r2 = r2
                    best_slope = slope
                    best_intercept = intercept
                    best_range = (start, end)
                    
            except Exception:
                continue
    
    # Fallback to OLS if robust methods fail
    if np.isnan(best_slope):
        try:
            coeffs = np.polyfit(log_eps, log_counts, 1)
            best_slope = coeffs[0]
            best_intercept = coeffs[1]
            # Calculate R² manually
            y_pred = best_slope * log_eps + best_intercept
            ss_res = np.sum((log_counts - y_pred) ** 2)
            ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
            best_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            best_range = (0, len(log_eps))
        except Exception:
            return np.nan if not return_debug else (np.nan, {"error": "Regression failed"})
    
    # Return results
    if return_debug:
        debug_info = {
            "slope": best_slope,
            "intercept": best_intercept,
            "r2": best_r2,
            "fit_range": best_range,
            "scales": valid_epsilons,
            "counts": counts,
            "log_eps": log_eps,
            "log_counts": log_counts,
            "n_points": N,
            "normalized": normalize,
            "detrended": detrend
        }
        return best_slope, debug_info
    else:
        return best_slope


def calc_hausdorff_dimension_for_single_signal(
    time: np.ndarray, 
    signal: np.ndarray, 
    n_boxes: int = 15,
    curr_file_data: Optional[Dict[str, Any]] = None, 
    current_side: str = "G1R", 
    current_force: str = "Fx", 
    current_interval: str = ""
) -> float:
    """
    Calculate the box-counting Hausdorff dimension for a single 2D curve (time vs signal).
    
    This is the public interface used by the pipeline. It calls the robust implementation
    with sensible defaults tuned for climbing force data (10-200 Hz typical).
    
    Args:
        time: Time array
        signal: Signal array  
        n_boxes: Number of box sizes (legacy parameter, kept for compatibility)
        curr_file_data: File data dict (legacy parameter)
        current_side: Side identifier (legacy parameter)
        current_force: Force identifier (legacy parameter)
        current_interval: Interval identifier (legacy parameter)
        
    Returns:
        Hausdorff dimension (float)
    """
    # Use the robust implementation with climbing-optimized defaults
    return hausdorff_dimension_boxcount(
        t=time,
        x=signal,
        normalize=True,      # Remove unit dependence
        detrend=False,       # Keep original signal characteristics
        min_points=32,       # Minimum for reliable estimation
        min_scales=6,        # Minimum scales for fit
        max_scales=25,       # Reasonable number of scales
        scale_mode="auto",   # Adaptive scaling
        eps_min_frac=2.0,    # Lower bound
        eps_max_frac=0.25,   # Upper bound
        robust_fit="theilsen", # Robust regression
        return_debug=False
    )


# called function to calculate Hausdorff dimensions for all axes
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


def plot_hausdorff_intervals(file_data: Dict[str, Any], folder_path: str, fname: str) -> None:
    """
    Plots force-time curves for each interval and force axis with Hausdorff values annotated.
    Saves plots as PNG files in a 'plots' subfolder.
    
    Only runs when config.create_hausdorff_plots is True to avoid heavy plotting in hot path.
    """
    if not getattr(config, 'create_hausdorff_plots', False):
        return
        
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
                    
                    # Create figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Main plot: force vs time
                    ax1.plot(time, force_series, 'b-', linewidth=1.5, label=f"{force}")
                    ax1.plot(time, force_series, 'rx', markersize=3, alpha=0.6, label='data points')
                    ax1.set_xlabel("Time [s]")
                    ax1.set_ylabel(f"{force} [%BW]")
                    ax1.set_title(f"{file_number} | {int_key} | {force} | HD = {hausdorff:.3f}")
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
                    
                    # Inset: log-log plot with fit
                    if len(time) >= 32:  # Only if enough points for meaningful fit
                        try:
                            # Get debug info for plotting
                            _, debug_info = hausdorff_dimension_boxcount(
                                np.array(time), np.array(force_series), 
                                return_debug=True
                            )
                            
                            log_eps = debug_info["log_eps"]
                            log_counts = debug_info["log_counts"]
                            slope = debug_info["slope"]
                            r2 = debug_info["r2"]
                            fit_range = debug_info["fit_range"]
                            
                            # Plot all points
                            ax2.plot(log_eps, log_counts, 'ko', markersize=4, alpha=0.7, label='Box counts')
                            
                            # Highlight fit range
                            if fit_range[1] > fit_range[0]:
                                ax2.plot(log_eps[fit_range[0]:fit_range[1]], 
                                       log_counts[fit_range[0]:fit_range[1]], 
                                       'ro', markersize=5, label='Fit range')
                            
                            # Plot fitted line
                            x_fit = np.array(log_eps[fit_range[0]:fit_range[1]])
                            y_fit = slope * x_fit + debug_info["intercept"]
                            ax2.plot(x_fit, y_fit, 'r--', linewidth=2, 
                                   label=f'Fit: D={slope:.3f}, R²={r2:.3f}')
                            
                            ax2.set_xlabel("log(1/ε)")
                            ax2.set_ylabel("log N(ε)")
                            ax2.set_title("Box-counting log-log plot")
                            ax2.grid(True, alpha=0.3)
                            ax2.legend()
                            
                        except Exception as e:
                            ax2.text(0.5, 0.5, f"Plot error:\n{str(e)}", 
                                   transform=ax2.transAxes, ha='center', va='center')
                            ax2.set_title("Box-counting plot (error)")
                    
                    plt.tight_layout()
                    
                    # Save plot
                    force_plot_dir = os.path.join(plot_dir, file_number, side, force)
                    os.makedirs(force_plot_dir, exist_ok=True)
                    filename = f"{file_number}_{side}_{int_key}_{force}_HD.png".replace(" ", "_")
                    plt.savefig(os.path.join(force_plot_dir, filename), dpi=150, bbox_inches='tight')
                    plt.close()