#!/usr/bin/env python3
"""
Video + LVM data synchronisation and live plotting
-------------------------------------------------
Loads two .mp4 videos (assumed to start at the same time) and one National Instruments
LabVIEW Measurement (.lvm) file, then plays both videos while plotting a selected 
column from the .lvm file live, synchronised by time.

Usage examples
--------------
python data_video_syn.py \
  --vid1 /mnt/data/042_backV_redflat_GX010858.mp4 \
  --vid2 /mnt/data/042_sideV_Gr_orange_incut-GX010251.mp4 \
  --lvm  /mnt/data/042-Hold-04_Orange_Incut_73kg_Noah_lvl-ADV_25-05-14_1047.lvm \
  --col Fy --time-col Time

Optional: 
  --offset 0.0            # seconds to shift the LVM data (+ delays the curve)
  --downsample 1          # plot every Nth LVM sample
  --title "D3_T1_Fy_Fz demo"

Notes
-----
- If a `loadData.py` is present in the project with a function `load_lvm(path)` or
  `read_lvm(path)`, it will be used to load the LVM robustly. Otherwise, a lightweight
  internal parser is used as a fallback.
- The two videos are assumed to be hardware-synchronised at t=0. The script drives
  playback by the shorter of the two videos. If needed, supply a `--offset` for LVM
  or per‑video offsets `--v1-offset`, `--v2-offset`.
- Keyboard controls while the OpenCV window is focused:
    Space  : Pause/Resume
    q/ESC  : Quit
    +/-    : Change playback speed (0.25x .. 2x)

Author: Noah MA – ICHs tools
"""
from __future__ import annotations
import argparse
import sys
import os
from typing import Optional, Tuple, List
import uuid

# numpy, pandas, cv2, matplotlib
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob
import re
from scipy.signal import savgol_filter
import config

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -----------------------------
# LVM LOADING
# -----------------------------

def load_lvm(path: str, time_col: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Load a National Instruments .lvm file.

    Tries project-specific loader from loadData.py first, then falls back to a
    simple, resilient parser. Returns (DataFrame, time_column_name).
    """
    # Check if file exists and has content
    if not os.path.exists(path):
        raise FileNotFoundError(f"LVM file not found: {path}")
    
    file_size = os.path.getsize(path)
    if file_size == 0:
        raise ValueError(f"LVM file is empty: {path}")
    
    print(f"[load_lvm] Loading file: {path} (size: {file_size} bytes)")
    # Try loadData.py if present in project
    try:
        import importlib
        if os.path.exists(os.path.join(os.path.dirname(__file__), 'loadData.py')):
            ld = importlib.import_module('loadData')
            if hasattr(ld, 'load_lvm'):
                df = ld.load_lvm(path)  # expected to return a DataFrame
            elif hasattr(ld, 'read_lvm'):
                df = ld.read_lvm(path)
            else:
                df = None
            if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
                # Pick time column
                tcol = _pick_time_col(df, user_time_col=time_col)
                return df.reset_index(drop=True), tcol
            else:
                print(f"[load_lvm] loadData.py returned invalid DataFrame: {type(df)}")
    except pd.errors.EmptyDataError as e:
        print(f"[load_lvm] loadData.py: EmptyDataError - {e}")
        # Continue to fallback parser
    except Exception as e:
        print(f"[load_lvm] loadData.py route failed: {e}")
        import traceback
        traceback.print_exc()

    # Fast-path: known LabVIEW .lvm format where header with column names is on line 22 (0-indexed => header=21)
    try:
        df_fast = pd.read_csv(path, sep='\t', decimal=',', header=21, engine='python', on_bad_lines='skip')
        df_fast.columns = [str(c).strip() for c in df_fast.columns]
        tcol_fast = _pick_time_col(df_fast, user_time_col=time_col)
        return df_fast.reset_index(drop=True), tcol_fast
    except Exception:
        pass  # fall back to heuristic parser below

    # Fallback parser: detect header length and delimiter, then read numeric block
    with open(path, 'r', errors='ignore') as f:
        lines = f.readlines()

    # Check if file is empty
    if not lines:
        raise ValueError("LVM file is empty or contains no readable content")

    # Find the first line that looks like a header with column names
    header_idx = None
    for i, line in enumerate(lines[:200]):
        # Candidates: lines containing multiple tokens separated by tab/space/semicolon
        if ('\t' in line or ';' in line or ',' in line or ' ' in line) and any(c.isalpha() for c in line):
            # Heuristic: later confirm by trying to parse a few rows below as floats
            header_idx = i
            break
    if header_idx is None:
        # Find first line where most tokens are numeric — then we create generic headers
        for i, line in enumerate(lines):
            toks = _smart_split(line)
            if _mostly_numeric(toks):
                header_idx = i
                break
    
    # If still no header found, use the first line
    if header_idx is None:
        header_idx = 0

    # Build a temporary csv text from header_idx onward
    try:
        text = ''.join(lines[header_idx:])
        sep = _guess_sep(lines[header_idx])
        from io import StringIO
        _buf = StringIO(text)
        try:
            # Use the python engine for flexible parsing and skip malformed lines
            df = pd.read_csv(_buf, sep=(sep if sep is not None else None), engine='python', on_bad_lines='skip')
        except Exception:
            # Fallback: let pandas auto-detect the separator with the python engine
            _buf.seek(0)
            df = pd.read_csv(_buf, sep=None, engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"[load_lvm] Error in fallback parser: {e}")
        print(f"[load_lvm] header_idx: {header_idx}, lines length: {len(lines)}")
        raise

    # Clean up: drop all-empty columns, coerce to numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='ignore')
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.astype(str).str.contains('^Unnamed')]
    
    # Check if DataFrame is still valid after cleanup
    if df.empty or len(df.columns) == 0:
        raise ValueError("DataFrame is empty after parsing and cleanup")

    # Guess time column
    tcol = _pick_time_col(df, user_time_col=time_col)
    # Ensure monotonic numeric time in seconds
    df[tcol] = pd.to_numeric(df[tcol], errors='coerce')
    # If time appears in milliseconds, normalise to seconds (simple heuristic)
    non_nan = df[tcol].dropna()
    if len(non_nan) > 1:
        dt_med = np.median(np.diff(non_nan.values))
        if dt_med > 10:  # probably in ms
            df[tcol] = df[tcol] / 1000.0
    return df.reset_index(drop=True), tcol


def _pick_time_col(df: pd.DataFrame, user_time_col: Optional[str]) -> str:
    if df is None or df.empty or len(df.columns) == 0:
        raise ValueError("DataFrame is empty or has no columns")
    
    if user_time_col and user_time_col in df.columns:
        return user_time_col
    # common time column names
    candidates = [
        'Time', 'time', 't', 'Zeit', 'Timestamp', 'Elapsed Time',
        'Absolute Time', 'Seconds', 'Sample Time'
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # otherwise: first column if numeric & monotonic increasing
    for c in df.columns:
        series = pd.to_numeric(df[c], errors='coerce')
        if series.isna().mean() < 0.2:
            vals = series.dropna().values
            if len(vals) > 3 and np.all(np.diff(vals[: min(50, len(vals))]) >= 0):
                return c
    # fallback - check again that we have columns
    if len(df.columns) == 0:
        raise ValueError("DataFrame has no columns")
    return str(df.columns[0])  # Ensure we return a string


def _guess_sep(line: str) -> str:
    if '\t' in line:
        return '\t'
    if ';' in line:
        return ';'
    if ',' in line and line.count(',') >= line.count(' '):
        return ','
    return None  # let pandas auto-detect whitespace


def _smart_split(line: str) -> List[str]:
    if '\t' in line:
        return line.strip().split('\t')
    if ';' in line:
        return line.strip().split(';')
    if ',' in line:
        return line.strip().split(',')
    return line.strip().split()


def _mostly_numeric(tokens: List[str]) -> bool:
    if not tokens:
        return False
    cnt = 0
    for tok in tokens:
        try:
            float(tok)
            cnt += 1
        except Exception:
            pass
    return cnt / max(1, len(tokens)) > 0.6


# -----------------------------
# TITLE/NAME HELPERS
# -----------------------------
def _file_acronym_from_path(path: str) -> str:
    """Return a short plot title derived from config.file_acronyms_map using the
    leading 3-digit file number in the LVM filename (e.g., '039', '042').
    Fallbacks to the 3-digit number or basename if no mapping found.
    """
    try:
        base = os.path.basename(path)
        m = re.search(r"(\d{3})", base)
        if m:
            num = m.group(1)
            return config.file_acronyms_map.get(num, num)
        return os.path.splitext(base)[0]
    except Exception:
        return os.path.basename(path)


# -----------------------------
# LEGEND + COLOR HELPERS
# -----------------------------
def _base_key_for_color(label: str) -> str:
    """Return base token matching config.COLOR_MAPPING keys.
    Examples:
      'Fy_1' -> 'Fy'
      'Fy_sum [%]' -> 'Fy_sum'
      'Fres_xyz_2 [N]' -> 'Fres_xyz'
    """
    name = str(label)
    # Strip unit suffix
    if '[' in name:
        name = name.split('[')[0].strip()
    name = name.replace('  ', ' ')
    # Keep *_sum variants intact
    if '_sum' in name:
        # Normalise things like 'Fy_sum ' -> 'Fy_sum'
        return name.split()[0]
    # Side-specific suffix _1/_2 => drop
    name = name.replace('_1', '').replace('_2', '')
    # For resultant tokens keep full prefix
    if name.startswith('Fres_xyz'):
        return 'Fres_xyz'
    if name.startswith('Fres_yz'):
        return 'Fres_yz'
    # Basic forces Fx/Fy/Fz/Mz
    for base in ('Fx', 'Fy', 'Fz', 'Mz', 'FgR'):
        if name.startswith(base):
            return base
    return name


from typing import Optional


def color_for_label(label: str) -> Optional[str]:
    """Pick a matplotlib color hex from config.COLOR_MAPPING for a given label."""
    try:
        key = _base_key_for_color(label)
        return config.COLOR_MAPPING.get(key)
    except Exception:
        return None


def format_legend(label: str) -> str:
    """Format legend so that force axis and side are both in the subscript (index).

    Examples (Matplotlib mathtext):
      - 'Fy_2' -> '$F_{L,y}$'
      - 'Fx_1' -> '$F_{R,x}$'
      - 'Fy_sum' -> '$F_{y,sum}$'
      - 'Fres_xyz_2' -> '$F_{L,xyz}$'
      - 'Mz_1' -> '$M_{R,z}$'
    """
    name = str(label)
    # Strip unit suffix
    if '[' in name:
        name = name.split('[')[0].strip()
    # Determine base symbol (F or M)
    base_symbol = 'F'
    if name.startswith('M'):
        base_symbol = 'M'
    # Side token
    side_token = ''
    if '_1' in name:
        side_token = 'R'
    elif '_2' in name:
        side_token = 'L'
    # Sum token
    is_sum = '_sum' in name
    # Axis token
    if name.startswith('Fres_xyz') or name.startswith('Mres_xyz'):
        axis = 'xyz'
    elif name.startswith('Fres_yz') or name.startswith('Mres_yz'):
        axis = 'yz'
    elif name.startswith(('Fx', 'Mx')):
        axis = 'x'
    elif name.startswith(('Fy', 'My')):
        axis = 'y'
    elif name.startswith(('Fz', 'Mz')):
        axis = 'z'
    else:
        axis = ''
    # Compose label: base with side and axis (and optional sum) in subscript
    parts = []
    if side_token:
        parts.append(side_token)
    if axis:
        parts.append(axis)
    if is_sum:
        parts.append('sum')
    if parts:
        return f"${base_symbol}_{{{','.join(parts)}}}$"
    return f"${base_symbol}$"


# -----------------------------
# SYNC FIRST PEAK WITH VIDEO
# -----------------------------
def _detect_side_video_marker_time(video_path: str) -> Optional[float]:
    """Detect the start time (s) of the green marker in the top-right region of the side video.

    Returns None if not found or the video cannot be opened.
    """
    print(f"[MATCH] _detect_side_video_marker_time called: video={os.path.basename(video_path)}")
    global _last_green_ratio  # Store the green ratio for debug prints
    _last_green_ratio = None
    vpath = video_path
    base = os.path.basename(video_path)
    if 'sideV' not in base:
        try:
            folder = os.path.dirname(video_path)
            cand = [p for p in glob.glob(os.path.join(folder, '*.mp4')) if 'sideV' in os.path.basename(p)]
            if cand:
                vpath = cand[-1]
        except Exception:
            pass
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # ROI: top-right quadrant
    x0 = int(width * 0.75); x1 = width
    y0 = 0; y1 = int(height * 0.25)
    # Detection thresholds
    green_low = np.array([35, 50, 50], dtype=np.uint8)
    green_high = np.array([85, 255, 255], dtype=np.uint8)
    ratio_thresh = 0.02  # 2% pixels green
    consec = 3
    consec_hits = 0
    frame_idx = 0
    t_video_marker = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            frame_idx += 1
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, green_low, green_high)
        frac = float(np.count_nonzero(mask)) / float(mask.size)
        if frac >= ratio_thresh:
            consec_hits += 1
        else:
            consec_hits = 0
        if consec_hits >= consec:
            # Use the exact frame time when the detection condition is reached
            t_video_marker = frame_idx / float(fps)
            _last_green_ratio = frac  # Store the green pixel ratio globally
            break
        frame_idx += 1
    cap.release()
    return t_video_marker


def _detect_motion_time_in_video(video_path: str,
                                      roi_rel: Tuple[float, float, float, float] = (0.6, 0.2, 0.35, 0.6),
                                      warmup_frames: int = 20,
                                      diff_thresh: int = 25,
                                      area_ratio_thresh: float = 0.01,
                                      consec: int = 4) -> Optional[float]:
    print(f"[MATCH] _detect_motion_time_in_video called: roi={roi_rel}, consec={consec}")
    """Detect the time when noticeable motion first occurs inside a ROI of the selected video.

    Args:
        video_path: Path to the video file for motion detection.
        roi_rel: Relative ROI (x, y, w, h) in fractions of frame size.
        warmup_frames: Number of initial frames to build background model.
        diff_thresh: Pixel intensity difference threshold (on 8-bit gray) to consider as motion.
        area_ratio_thresh: Fraction of ROI pixels that must exceed threshold to count as motion.
        consec: Number of consecutive frames that must satisfy the condition.

    Returns:
        Time in seconds of motion onset, or None if not detected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x_rel, y_rel, w_rel, h_rel = roi_rel
    x0 = int(max(0, min(1, x_rel)) * width)
    y0 = int(max(0, min(1, y_rel)) * height)
    x1 = int(min(width, x0 + w_rel * width))
    y1 = int(min(height, y0 + h_rel * height))

    bg = None
    frame_idx = 0
    hit = 0
    t_motion = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if bg is None:
            bg = gray.astype('float32')
        if frame_idx < warmup_frames:
            cv2.accumulateWeighted(gray, bg, 0.1)
        else:
            diff = cv2.absdiff(gray, cv2.convertScaleAbs(bg))
            _, th = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
            ratio = float(np.count_nonzero(th)) / float(th.size)
            if ratio >= area_ratio_thresh:
                hit += 1
            else:
                hit = 0
            if hit >= consec:
                t_motion = frame_idx / float(fps)
                motion_value = hit  # Store the motion value
                break
            # slowly update background to adapt lighting but preserve static background
            cv2.accumulateWeighted(gray, bg, 0.01)
        frame_idx += 1
    cap.release()
    try:
        if t_motion is not None:
            print(f"[sync] Detected back-video motion onset at t={t_motion:.3f}s (motion={motion_value}, roi={tuple(round(v,3) for v in roi_rel)})")
        else:
            print("[sync] No motion onset detected in back video ROI")
    except Exception:
        pass
    return t_motion


def _compute_data_match_time(data: pd.DataFrame, hold_side: str = 'G2L', match_force: str = 'Fy', 
                           metric: str = 'peakForce', threshold: float = 20.0) -> Optional[float]:
    """Compute the match time based on selected metric for the selected force/side.

    Metrics:
      - peakForce: time of first local maximum above threshold
      - maxROFD_rising: time of maximum positive slope (ROFD = Rate of Force Development)
      - maxROFD_falling: time of maximum negative slope
      - threshold_rising: first upward crossing of user threshold
      - threshold_falling: first downward crossing of user threshold
    """
    print(f"[MATCH] _compute_data_match_time called: hold_side={hold_side}, match_force={match_force}, metric={metric}, threshold={threshold}")
    print(f"[DEBUG] _compute_data_match_time FUNCTION START")
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return None
    tcol = _pick_time_col(data, user_time_col=None)
    t = pd.to_numeric(data[tcol], errors='coerce').to_numpy(dtype=float)
    side_suffix = '2' if str(hold_side).upper() == 'G2L' else '1'
    ftoken = str(match_force).strip()
    if ftoken not in ('Fy', 'Fz', 'Fx', 'Fres_xyz', 'FgR'):
        ftoken = 'Fy'
    col_prefix = f"{ftoken}_{side_suffix} "
    force_col = next((c for c in data.columns if c.startswith(col_prefix)), None)
    if force_col is None:
        force_col = next((c for c in data.columns if c.startswith(f"{ftoken}_")), None)
    if force_col is None:
        return None
    y = pd.to_numeric(data[force_col], errors='coerce').to_numpy(dtype=float)
    
    # Validate data before processing
    if len(y) == 0 or not np.any(np.isfinite(y)):
        print(f"[sync] Warning: No valid data in column {force_col}")
        return None
    
    print(f"[sync] Data validation: len={len(y)}, finite={np.sum(np.isfinite(y))}, range=[{np.nanmin(y):.1f}, {np.nanmax(y):.1f}]")
    
    y = _savgol_safe(y, window=11, poly=5)
    unit_pct = ('[%]' in force_col)
    
    # Determine threshold based on metric and unit
    if metric.startswith('threshold') or metric == 'early_dent':
        if unit_pct:
            thr = threshold
        else:
            # For Newton units, try to compute threshold from body weight
            thr = None  # Will handle in specific metric logic
    elif metric == 'peakForce':
        if unit_pct:
            thr = threshold
        else:
            thr = None
    else:  # maxROFD metrics don't need amplitude threshold
        thr = None
    
    t_match = None
    
    if metric == 'peakForce':
        print(f"[peakForce] Searching for peak >= {thr:.1f} in range [0, {len(y)-1}]")
        # Find first local maximum above threshold
        for i in range(1, len(y) - 1):
            if not np.isfinite(y[i]):
                continue
            if thr is None or y[i] >= thr:
                if y[i] > y[i - 1] and y[i] >= y[i + 1]:
                    t_match = float(t[i])
                    peak_idx = i  # Store the peak index
                    print(f"[peakForce] Found peak at index {i} with value {y[i]:.1f} at t={t[i]:.3f}s")
                    break
        if t_match is None:
            print(f"[peakForce] No peak >= {thr:.1f} found, trying fallback method")
            # Fallback: find first threshold crossing then peak in next 1s
            if thr is not None:
                cs = np.where((y[:-1] < thr) & (y[1:] >= thr))[0]
                if cs.size > 0:
                    print(f"[peakForce] Found {cs.size} threshold crossings, using first at index {cs[0]}")
                    i0 = int(cs[0] + 1)
                    valid = np.isfinite(t)
                    dt = np.median(np.diff(t[valid])) if np.sum(valid) > 3 else 0.01
                    w = max(3, int(round(1.0 / max(1e-6, dt))))
                    j1 = min(len(y), i0 + w)
                    seg = y[i0:j1]
                    if seg.size > 0:
                        rel = int(np.nanargmax(seg))
                        t_match = float(t[i0 + rel])
                        print(f"[peakForce] Fallback: found peak at index {i0 + rel} with value {seg[rel]:.1f} at t={t_match:.3f}s")
                else:
                    print(f"[peakForce] No threshold crossings found for {thr:.1f}")
            else:
                print(f"[peakForce] No threshold available for fallback")
    
    elif metric == 'maxROFD_rising':
        # Find maximum positive slope
        slope = np.gradient(y, t)
        if np.any(np.isfinite(slope)):
            idx = int(np.nanargmax(slope))
            t_match = float(t[idx])
    
    elif metric == 'maxROFD_falling':
        # Find maximum negative slope (most negative)
        slope = np.gradient(y, t)
        if np.any(np.isfinite(slope)):
            idx = int(np.nanargmin(slope))
            t_match = float(t[idx])
    
    elif metric == 'threshold_rising':
        # Find first upward crossing of threshold
        if thr is not None:
            cs = np.where((y[:-1] < thr) & (y[1:] >= thr))[0]
            if cs.size > 0:
                idx = int(cs[0] + 1)
                t_match = float(t[idx])
    
    elif metric == 'threshold_falling':
        # Find first downward crossing of threshold
        if thr is not None:
            cs = np.where((y[:-1] > thr) & (y[1:] <= thr))[0]
            if cs.size > 0:
                idx = int(cs[0] + 1)
                t_match = float(t[idx])
    
    elif metric == 'early_dent':
        # Find the first significant dent (local minimum) before the main force development
        # Look for a local minimum that goes below threshold, but only in the first part of the data
        print(f"[early_dent] Starting early_dent detection with threshold={thr}")
        print(f"[early_dent] Data range: {len(y)} points, time range: {t[0]:.3f}s to {t[-1]:.3f}s")
        print(f"[early_dent] Force range: {np.nanmin(y):.2f} to {np.nanmax(y):.2f}")
        
        if thr is not None:
            # Find the first major peak to define the "early" region
            # Use a higher threshold to find the main force development
            main_threshold = max(10.0, thr * 2)  # At least 10% BW or 2x the dent threshold
            print(f"[early_dent] Main threshold for early region: {main_threshold}")
            
            # Find first crossing of main threshold
            main_crossings = np.where((y[:-1] < main_threshold) & (y[1:] >= main_threshold))[0]
            print(f"[early_dent] Main threshold crossings found: {len(main_crossings)}")
            if main_crossings.size > 0:
                early_end = int(main_crossings[0])  # End of "early" region
                print(f"[early_dent] Early region ends at index {early_end} (t={t[early_end]:.3f}s)")
            else:
                early_end = len(y) // 3  # Use first third if no main peak found
                print(f"[early_dent] No main threshold crossing found, using first third: index {early_end}")
            
            # Look for local minima below threshold in the early region
            early_data = y[:early_end]
            early_t = t[:early_end]
            print(f"[early_dent] Early data range: {len(early_data)} points, force range: {np.nanmin(early_data):.2f} to {np.nanmax(early_data):.2f}")
            
            # Find local minima
            local_mins = []
            for i in range(1, len(early_data) - 1):
                if (early_data[i] < early_data[i-1] and 
                    early_data[i] <= early_data[i+1] and 
                    early_data[i] <= thr):
                    local_mins.append(i)
            
            print(f"[early_dent] Local minima below threshold {thr}: {len(local_mins)} found")
            if local_mins:
                print(f"[early_dent] Local minima indices: {local_mins[:5]}...")  # Show first 5
                for i, idx in enumerate(local_mins[:3]):  # Show details of first 3
                    print(f"[early_dent]   Min {i+1}: index={idx}, t={early_t[idx]:.3f}s, force={early_data[idx]:.2f}")
            
            if local_mins:
                # Take the first significant local minimum
                t_match = float(early_t[local_mins[0]])
                print(f"[early_dent] Selected first local minimum at t={t_match:.3f}s")
            else:
                # Fallback: find first crossing below threshold in early region
                print(f"[early_dent] No local minima found, trying threshold crossing fallback")
                cs = np.where((early_data[:-1] > thr) & (early_data[1:] <= thr))[0]
                print(f"[early_dent] Threshold crossings in early region: {len(cs)}")
                if cs.size > 0:
                    idx = int(cs[0] + 1)
                    t_match = float(early_t[idx])
                    print(f"[early_dent] Fallback: threshold crossing at t={t_match:.3f}s")
                else:
                    t_match = None
                    print(f"[early_dent] No detection found - no local minima or threshold crossings")
        else:
            t_match = None
            print(f"[early_dent] No threshold provided, cannot detect early dent")
        
        print(f"[early_dent] Final result: t_match = {t_match}")
    
    try:
        utag = '%' if unit_pct else 'N'
        if metric.startswith('threshold'):
            # Get the value at the detected time
            if t_match is not None and len(t) > 0 and len(y) > 0:
                # Find the closest time index
                time_idx = np.argmin(np.abs(t - t_match))
                if 0 <= time_idx < len(y):
                    detected_value = y[time_idx] if np.isfinite(y[time_idx]) else float('nan')
                    print(f"[sync] Detected LVM {metric} at t={t_match:.3f}s (value={detected_value:.1f}{utag}, thr={threshold}{utag})")
                else:
                    print(f"[sync] Detected LVM {metric} at t={t_match:.3f}s (invalid index {time_idx}, len={len(y)})")
        else:
            # Get the value at the detected time
            if t_match is not None and len(t) > 0 and len(y) > 0:
                # Use the stored peak index if available, otherwise find closest time
                if 'peak_idx' in locals() and peak_idx is not None:
                    time_idx = peak_idx
                    print(f"[DEBUG] Using stored peak_idx={peak_idx}")
                else:
                    # Find the closest time index
                    time_idx = np.argmin(np.abs(t - t_match))
                    print(f"[DEBUG] Using closest time_idx={time_idx}")
                
                if 0 <= time_idx < len(y):
                    detected_value = y[time_idx] if np.isfinite(y[time_idx]) else float('nan')
                    print(f"[DEBUG] time_idx={time_idx}, t[time_idx]={t[time_idx]:.3f}, t_match={t_match:.3f}, y[time_idx]={y[time_idx]}, detected_value={detected_value}")
                    print(f"[sync] Detected LVM {metric} at t={t_match:.3f}s (value={detected_value:.1f}{utag})")
                else:
                    print(f"[sync] Detected LVM {metric} at t={t_match:.3f}s (invalid index {time_idx}, len={len(y)})")
    except Exception as e:
        print(f"[sync] Error in debug print: {e}")
        pass
    
    return t_match
    
    return t_match


def sync_first_peak_with_video(data: pd.DataFrame, video_path: str, hold_side: str = 'G2L', match_force: str = 'Fy') -> float:
    """Estimate time offset between data and video by aligning the first force peak
    to the appearance of a green marker box in the top-right corner of the side video.

    Steps:
      1) Data: use the selected matching force of the selected hold side
         (G1R -> <F>_1, G2L -> <F>_2). Supported: Fy, Fz, Fx, Fres_xyz, FgR.
         Only accept peaks >= 20% body weight. If units are [%], threshold is 20.
         If units are [N], parse body weight from filename (e.g., '_73kg') to get 0.2*BW*9.81.
         Then find the point of maximal falling (most negative slope) within ~5s after that peak.
      2) Video: open the side view video (filename contains 'sideV') and detect the time when a
         green rectangle appears in the top-right region. Use the exact frame time when detected.
      3) Return delta = t_video_marker - t_data_max_fall (seconds). If either detection fails, return 0.0.

    Args:
        data: LVM dataframe with a time column and force columns.
        video_path: Path to a video file. If it does not contain 'sideV', search for a sibling
                    video file in the same directory whose name contains 'sideV'.

    Returns:
        float: Time offset in seconds (positive delays data to match the video marker),
               or 0.0 on failure.
    """
    print(f"[MATCH] sync_first_peak_with_video called: hold_side={hold_side}, match_force={match_force}")
    print(f"[DEBUG] sync_first_peak_with_video FUNCTION START")
    try:
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            return 0.0
        # Pick time column
        tcol = _pick_time_col(data, user_time_col=None)
        t = pd.to_numeric(data[tcol], errors='coerce').to_numpy(dtype=float)
        # Pick force column of selected side
        side_suffix = '2' if str(hold_side).upper() == 'G2L' else '1'
        ftoken = str(match_force).strip()
        # Map token to column prefix
        if ftoken not in ('Fy', 'Fz', 'Fx', 'Fres_xyz', 'FgR'):
            ftoken = 'Fy'
        col_prefix = f"{ftoken}_{side_suffix} "
        force_col = next((c for c in data.columns if c.startswith(col_prefix)), None)
        if force_col is None:
            # fallback: try any side for this token
            force_col = next((c for c in data.columns if c.startswith(f"{ftoken}_")), None)
        if force_col is None:
            return 0.0
        y = pd.to_numeric(data[force_col], errors='coerce').to_numpy(dtype=float)
        unit_pct = ('[%]' in force_col)

        # Filter for stability
        y = _savgol_safe(y, window=11, poly=5)

        # Threshold in same units as y (20% BW)
        if unit_pct:
            thresh = 10.0
        else:
            bw_kg = _extract_body_weight_from_path(video_path)
            climberforce = float(bw_kg * 9.81) if bw_kg else None
            if not climberforce:
                return 0.0
            thresh = 0.2 * climberforce

        # Find first local maximum above threshold
        peak_idx = None
        print(f"[sync] Searching for peak >= {thresh:.1f} in range [0, {len(y)-1}]")
        for i in range(1, len(y) - 1):
            if not np.isfinite(y[i]):
                continue
            if y[i] >= thresh and y[i] > y[i - 1] and y[i] >= y[i + 1]:
                peak_idx = i
                print(f"[sync] Found peak at index {i} with value {y[i]:.1f}")
                break
        if peak_idx is None:
            print(f"[sync] No peak >= {thresh:.1f} found, trying fallback method")
            # As fallback, find first threshold crossing then peak in next 1s
            crossings = np.where((y[:-1] < thresh) & (y[1:] >= thresh))[0]
            if crossings.size == 0:
                print(f"[sync] No threshold crossing found for {thresh:.1f} in LVM data")
                return 0.0
            i0 = int(crossings[0] + 1)
            # Window ~ 1s (approx by median dt)
            valid = np.isfinite(t)
            dt = np.median(np.diff(t[valid])) if np.sum(valid) > 3 else 0.01
            w = max(3, int(round(1.0 / max(1e-6, dt))))
            j1 = min(len(y), i0 + w)
            seg = y[i0:j1]
            if seg.size == 0:
                print("[sync] Empty segment after threshold crossing for peak detection")
                return 0.0
            peak_idx = i0 + int(np.nanargmax(seg))
        t_data_peak = float(t[peak_idx]) if np.isfinite(t[peak_idx]) else 0.0
        try:
            unit_tag = '%' if unit_pct else 'N'
            # Get the actual value at the detected time
            if t_data_peak is not None:
                # Find the closest time index
                time_idx = np.argmin(np.abs(t - t_data_peak))
                if 0 <= time_idx < len(y):
                    peak_value = y[time_idx] if np.isfinite(y[time_idx]) else float('nan')
                    print(f"[sync] Detected LVM peak ({ftoken}_{side_suffix}) at t={t_data_peak:.3f}s (value={peak_value:.1f}{unit_tag}, threshold={thresh:.1f}{unit_tag})")
                else:
                    print(f"[sync] Detected LVM peak ({ftoken}_{side_suffix}) at t={t_data_peak:.3f}s (invalid index {time_idx}, len={len(y)})")
        except Exception as e:
            print(f"[sync] Error in peak debug print: {e}")
            pass

        # Find point of maximal falling (most negative slope) within ~5s after peak
        valid = np.isfinite(t)
        dt = np.median(np.diff(t[valid])) if np.sum(valid) > 3 else 0.01
        w = max(5, int(round(5.0 / max(1e-6, dt))))
        end_idx = min(len(y) - 1, peak_idx + w)
        # Compute slope using gradient on the smoothed curve
        slope = np.gradient(y, t)
        fall_idx = None
        if np.any(np.isfinite(slope[peak_idx+1:end_idx])):
            rel = int(np.nanargmin(slope[peak_idx+1:end_idx]))
            fall_idx = peak_idx + 1 + rel
        t_data_fall = float(t[fall_idx]) if (fall_idx is not None and np.isfinite(t[fall_idx])) else t_data_peak
        try:
            sval = slope[fall_idx] if fall_idx is not None and np.isfinite(slope[fall_idx]) else float('nan')
            print(f"[sync] Detected LVM maximal falling at t={t_data_fall:.3f}s (slope={sval:.3f} units/s)")
        except Exception:
            pass

        t_video_marker = _detect_side_video_marker_time(video_path)
        if t_video_marker is None:
            print("[sync] No green marker detected in side video")
            return 0.0
        try:
            green_ratio_info = f", green_ratio={_last_green_ratio:.3f}" if _last_green_ratio is not None else ""
            print(f"[sync] Detected side-video green marker at t={t_video_marker:.3f}s{green_ratio_info}")
        except Exception:
            pass
        delta = float(t_video_marker - t_data_fall)
        return delta
    except Exception:
        return 0.0

# -----------------------------
# VIDEO HELPERS
# -----------------------------

def open_video(path: str) -> Tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, float(fps), frame_count, width, height


def read_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok:
        return None
    return frame

# -----------------------------
# SYNC MAPPER
# -----------------------------

def make_time_indexer(times_s: np.ndarray, values: np.ndarray, mode: str = 'linear'):
    """Create a fast indexer f(t)->value.
    mode='linear' (default) does linear interpolation between samples.
    mode='nearest' snaps to the closest sample (old behavior).
    """
    times_s = np.asarray(times_s, dtype=float)
    values = np.asarray(values)
    if len(times_s) != len(values):
        raise ValueError("times and values length mismatch")
    if len(times_s) == 0:
        raise ValueError("empty time series")

    if mode == 'nearest':
        def idx_fn(t: float):
            i = np.searchsorted(times_s, t)
            if i <= 0:
                return values[0]
            if i >= len(values):
                return values[-1]
            # pick closer of neighbours
            if abs(times_s[i] - t) < abs(t - times_s[i - 1]):
                return values[i]
            return values[i - 1]
        return idx_fn

    # default: linear interpolation
    def idx_fn(t: float):
        i = np.searchsorted(times_s, t)
        if i <= 0:
            return values[0]
        if i >= len(values):
            return values[-1]
        t0 = times_s[i - 1]; t1 = times_s[i]
        v0 = values[i - 1];  v1 = values[i]
        # Guard against identical timestamps
        if t1 == t0:
            return v1
        alpha = (t - t0) / (t1 - t0)
        return (1.0 - alpha) * v0 + alpha * v1
    return idx_fn

# -----------------------------
# DATAFRAME HELPERS
# -----------------------------
def numeric_columns(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().mean() > 0.9:
            cols.append(c)
    return cols


# Helper: pick columns by token (case-insensitive, prefer side-specific over *_sum)
def _pick_columns_by_token(df: pd.DataFrame, token: str) -> list:
    """Return columns whose name contains the token (case-insensitive).
    If side-specific channels exist (e.g., *_1, *_2), prefer those and
    exclude aggregate *_sum when possible.
    """
    if not token:
        return []
    token_l = str(token).lower()
    cols = [c for c in df.columns if token_l in str(c).lower()]
    if not cols:
        return []
    side_cols = [c for c in cols if ("_1" in str(c) or "_2" in str(c))]
    if side_cols:
        cols = [c for c in side_cols if "_sum" not in str(c).lower()]
    return cols

# --- Weight extraction and filtering helpers ---
def _extract_body_weight_from_path(path: str) -> Optional[float]:
    """Extract body weight in kg from filename if it contains patterns like '_73kg'."""
    try:
        fname = os.path.basename(path)
        m = re.search(r"_(\d{2,3})kg", fname, re.IGNORECASE)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None

def _savgol_safe(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Apply Savitzky–Golay filter with safeguards (odd window, <= len(y), poly < window)."""
    n = int(max(3, window))
    # window must be odd and <= len(y)
    if n % 2 == 0:
        n += 1
    if n > len(y):
        n = len(y) if len(y) % 2 == 1 else max(3, len(y) - 1)
    p = int(max(1, min(poly, n - 1)))
    try:
        return savgol_filter(y, window_length=n, polyorder=p, mode='interp')
    except Exception:
        return y  # fallback: unfiltered if params invalid

# -----------------------------
# MAIN PLAYER
# -----------------------------

def play_two_videos_with_live_plot(
    vid1_path: str,
    vid2_path: str,
    lvm_path: str,
    lvm_col: str,
    time_col: Optional[str] = None,
    lvm_offset: float = 0.0,
    v1_offset: float = 0.0,
    v2_offset: float = 0.0,
    downsample: int = 1,
    bw_kg: Optional[float] = None,
    sg_window: int = 11,
    sg_poly: int = 5,
    interp_mode: str = 'linear',
    title: Optional[str] = None,
    start_at_video_time: Optional[float] = None,
    export_video: bool = False,
    export_path: Optional[str] = None,
):
    # Load LVM
    df, tcol = load_lvm(lvm_path, time_col)
    # Resolve which columns to plot: support up to 3 comma-separated tokens
    tokens = [tok.strip() for tok in (lvm_col or '').split(',') if tok.strip()]
    if not tokens:
        raise ValueError("Please provide at least one column token to plot.")
    plot_cols = []
    for tok in tokens[:3]:
        cols = _pick_columns_by_token(df, tok)
        if not cols and tok in df.columns:
            cols = [tok]
        if cols:
            plot_cols.extend(cols)
    # De-duplicate while preserving order
    seen = set()
    plot_cols = [c for c in plot_cols if not (c in seen or seen.add(c))]
    if not plot_cols:
        raise ValueError(f"None of the requested tokens were found: {tokens}")

    # Base time in seconds
    t_s_base = pd.to_numeric(df[tcol], errors='coerce').to_numpy(dtype=float)

    # --- Normalization setup (% body weight) ---
    auto_bw = _extract_body_weight_from_path(lvm_path)
    if bw_kg is None:
        bw_kg = auto_bw
    bw_newton = float(bw_kg * 9.81) if bw_kg else None
    normalize_to_bw = bw_newton is not None

    # Prepare an indexer per column
    y_at_map = {}
    for c in plot_cols:
        y_c = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float)
        mask_c = ~np.isnan(t_s_base) & ~np.isnan(y_c)
        t_c = t_s_base[mask_c]
        y_c = y_c[mask_c]
        if y_c.size == 0:
            continue
        # --- Filtering ---
        y_c = _savgol_safe(y_c, sg_window, sg_poly)
        # --- Normalize to %BW if column appears to be in Newton (no '%' in name) and bw available
        #     Never normalize FgR_* (already in % by definition)
        cname = str(c)
        if normalize_to_bw and ('%' not in cname) and (not cname.startswith('FgR')):
            y_c = (y_c / bw_newton) * 100.0
        if downsample > 1:
            t_c = t_c[::downsample]
            y_c = y_c[::downsample]
        # Apply global LVM offset
        t_c = t_c + float(lvm_offset)
        y_at_map[c] = make_time_indexer(t_c, y_c, mode=interp_mode)

    # Open videos
    cap1, fps1, n1, w1, h1 = open_video(vid1_path)
    cap2, fps2, n2, w2, h2 = open_video(vid2_path)

    # Playback configuration
    max_frames = min(n1, n2)
    # We step using wall time derived from frame index / fps per stream + its offset
    # To avoid drift, we map each stream to its own t and average their t, 
    # which is robust when fps differ slightly
    paused = False
    speed = 1.0

    # Matplotlib live figure
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 3))
    default_name = _file_acronym_from_path(lvm_path)
    fig.canvas.manager.set_window_title(title or f"{default_name} – {lvm_col}")
    ax.set_xlabel('Time [s]')
    # Always show Y axis in percent of body weight
    ax.set_ylabel("F [%BW]")
    # keep a scrolling window of last N seconds
    window_s = 10.0
    lines = {}
    # Dynamic force value text objects
    force_texts = {}
    for c in plot_cols:
        line_color = color_for_label(c) or None
        lobj, = ax.plot([], [], lw=2, label=format_legend(str(c)), color=line_color)
        lines[c] = lobj
        # Create text object for current force value
        text_obj = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                          fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        force_texts[c] = text_obj
    if plot_cols:
        # Smart legend positioning to avoid overlap
        if len(plot_cols) <= 2:
            ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        else:
            # For 3+ plots, use upper left to avoid overlap with force values
            ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
    ax.grid(True, linestyle=':')

    # Buffers for the last window, per column
    buffers = {c: {"t": [], "y": []} for c in plot_cols}

    # Prepare OpenCV canvas (side-by-side for display, 2-row for export)
    out_h = max(h1, h2)
    out_w = w1 + w2
    win_name = "ICHs: Dual video + live plot (focus this window for keys)"
    
    # Export canvas dimensions (2-row layout: videos on top, plot on bottom)
    plot_height = int(out_h * 0.5)  # Plot is 50% of video height
    export_h = out_h + plot_height
    export_w = max(w1 + w2, 1200)  # Ensure minimum width for plot
    
    # Video export setup
    video_writer = None
    if export_video and export_path:
        # Ensure export path has .mp4 extension
        if not export_path.lower().endswith('.mp4'):
            export_path = export_path + '.mp4'
        
        # Create directory if it doesn't exist
        export_dir = os.path.dirname(export_path)
        if export_dir and not os.path.exists(export_dir):
            try:
                os.makedirs(export_dir, exist_ok=True)
                print(f"[export] Created directory: {export_dir}")
            except Exception as e:
                print(f"[export] Warning: Could not create directory {export_dir}: {e}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps_out = min(fps1, fps2)  # Use the lower fps for export
        video_writer = cv2.VideoWriter(export_path, fourcc, fps_out, (export_w, export_h))
        
        # Check if video writer was created successfully
        if not video_writer.isOpened():
            print(f"[export] Error: Could not create video writer for {export_path}")
            video_writer = None
        else:
            print(f"[export] Starting video export to: {export_path}")
            print(f"[export] Output resolution: {export_w}x{export_h}, FPS: {fps_out}")
            print(f"[export] Layout: Videos ({w1}x{h1} + {w2}x{h2}) + Live plot ({plot_height}px height)")

    # Optional: start both videos at a specific wall-clock time (seconds)
    start_t = float(start_at_video_time) if (start_at_video_time is not None) else 0.0
    if start_t > 0.0:
        try:
            start_frame1 = int(max(0, round((start_t - float(v1_offset)) * fps1)))
            start_frame2 = int(max(0, round((start_t - float(v2_offset)) * fps2)))
            cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame1)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame2)
        except Exception:
            pass

    frame_idx = 0
    while frame_idx < max_frames:
        if not paused:
            f1 = read_frame(cap1)
            f2 = read_frame(cap2)
            if f1 is None or f2 is None:
                break
            # Compute current time from each stream
            t1 = start_t + frame_idx / fps1 + float(v1_offset)
            t2 = start_t + frame_idx / fps2 + float(v2_offset)
            t_now = 0.5 * (t1 + t2)

            # Update per-series buffers
            t_min = t_now - window_s
            for cname, idx_fn in y_at_map.items():
                current_val = float(idx_fn(t_now))
                buffers[cname]["t"].append(t_now)
                buffers[cname]["y"].append(current_val)
                # Trim
                bt = buffers[cname]["t"]
                by = buffers[cname]["y"]
                while bt and bt[0] < t_min:
                    bt.pop(0); by.pop(0)
                # Update line data
                lines[cname].set_data(bt, by)
                # Update dynamic force value text
                if cname in force_texts:
                    text_obj = force_texts[cname]
                    text_obj.set_text(f"{format_legend(str(cname))}: {current_val:.1f}%")
                    # Position text objects vertically stacked, avoid legend overlap
                    text_idx = list(force_texts.keys()).index(cname)
                    if len(plot_cols) <= 2:
                        # For 1-2 plots: use top-left (legend is top-right)
                        text_obj.set_position((0.02, 0.98 - text_idx * 0.08))
                    else:
                        # For 3+ plots: use top-right (legend is top-left)
                        text_obj.set_position((0.98, 0.98 - text_idx * 0.08))

            ax.set_xlim(max(0.0, t_min), max(window_s, t_now))
            # Auto-scale y with margins across all series
            all_vals = []
            for cname in plot_cols:
                all_vals.extend(buffers[cname]["y"])
            if len(all_vals) > 1:
                yv = np.array(all_vals, dtype=float)
                ymin, ymax = float(np.nanmin(yv)), float(np.nanmax(yv))
                if np.isfinite(ymin) and np.isfinite(ymax):
                    if ymin == ymax:
                        ymin -= 1.0; ymax += 1.0
                    pad = 0.05 * (ymax - ymin)
                    ax.set_ylim(ymin - pad, ymax + pad)
            fig.canvas.draw(); fig.canvas.flush_events()

            # Compose video frame (side-by-side), resize if heights differ
            if h1 != out_h:
                f1 = cv2.resize(f1, (w1, out_h))
            if h2 != out_h:
                f2 = cv2.resize(f2, (w2, out_h))
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            canvas[:, :w1, :] = f1
            canvas[:, w1:w1 + w2, :] = f2

            # HUD overlay
            unit_tag = "%BW" if normalize_to_bw else ""
            first_label = str(plot_cols[0]) if plot_cols else str(lvm_col)
            first_val = buffers[plot_cols[0]]["y"][-1] if plot_cols and buffers[plot_cols[0]]["y"] else float('nan')
            
            hud = (
                f"t={t_now:6.2f}s  {first_label}={first_val:.2f}{('%' if unit_tag else '')}  "
                f"(series={len(plot_cols)})  fps1={fps1:.2f} fps2={fps2:.2f}  x{speed:.2f}"
            )
            cv2.putText(canvas, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add keyboard controls hint (show for first 5 seconds or when paused)
            show_hint = (t_now < 5.0) or paused
            if show_hint:
                hint_lines = [
                    "Keyboard Controls:",
                    "SPACE: Pause/Resume",
                    "Q/ESC: Quit",
                    "+/-: Speed up/down",
                    "A/D or ←/→: Jump -2s/+2s",
                    "S/F: Jump -10s/+10s"
                ]
                
                # Create semi-transparent background for hints
                hint_bg = np.zeros((len(hint_lines) * 25 + 20, 300, 3), dtype=np.uint8)
                hint_bg[:, :, :] = (0, 0, 0)  # Black background
                
                # Add hints to background
                for i, line in enumerate(hint_lines):
                    y_pos = 20 + i * 25
                    color = (255, 255, 255) if i == 0 else (200, 200, 200)  # White for title, gray for controls
                    cv2.putText(hint_bg, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                
                # Overlay hints on canvas
                hint_h, hint_w = hint_bg.shape[:2]
                canvas_h, canvas_w = canvas.shape[:2]
                
                # Position hints in top-right corner
                x_offset = canvas_w - hint_w - 20
                y_offset = 50
                
                # Blend hints with canvas
                alpha = 0.8
                canvas[y_offset:y_offset+hint_h, x_offset:x_offset+hint_w] = (
                    alpha * hint_bg + (1 - alpha) * canvas[y_offset:y_offset+hint_h, x_offset:x_offset+hint_w]
                )
            


            cv2.imshow(win_name, canvas)
            
            # Write frame to video if exporting
            if video_writer is not None and not paused:
                # Create export canvas with videos on top and plot on bottom
                export_canvas = np.zeros((export_h, export_w, 3), dtype=np.uint8)
                
                # Place videos on top row
                if h1 != out_h:
                    f1_resized = cv2.resize(f1, (w1, out_h))
                else:
                    f1_resized = f1
                if h2 != out_h:
                    f2_resized = cv2.resize(f2, (w2, out_h))
                else:
                    f2_resized = f2
                
                # Place videos side by side in top row
                export_canvas[:out_h, :w1] = f1_resized
                export_canvas[:out_h, w1:w1+w2] = f2_resized
                
                # Capture matplotlib plot and add to bottom row
                try:
                    # Convert matplotlib figure to image
                    fig.canvas.draw()
                    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    
                    # Calculate plot dimensions maintaining aspect ratio
                    plot_h = plot_height
                    plot_w = export_w
                    
                    # Get original plot dimensions
                    orig_h, orig_w = plot_img.shape[:2]
                    
                    # Calculate scaling to fit in allocated space while maintaining aspect ratio
                    scale_h = plot_h / orig_h
                    scale_w = plot_w / orig_w
                    scale = min(scale_h, scale_w)  # Use smaller scale to maintain aspect ratio
                    
                    # Calculate new dimensions
                    new_w = int(orig_w * scale)
                    new_h = int(orig_h * scale)
                    
                    # Resize plot maintaining aspect ratio
                    plot_img_resized = cv2.resize(plot_img, (new_w, new_h))
                    
                    # Center the plot in the allocated space
                    start_x = (plot_w - new_w) // 2
                    start_y = out_h + (plot_h - new_h) // 2
                    
                    # Convert RGB to BGR for OpenCV
                    plot_img_bgr = cv2.cvtColor(plot_img_resized, cv2.COLOR_RGB2BGR)
                    
                    # Place plot in bottom row, centered
                    export_canvas[start_y:start_y+new_h, start_x:start_x+new_w] = plot_img_bgr
                    
                except Exception as e:
                    print(f"[export] Warning: Could not capture plot: {e}")
                    # If plot capture fails, just use videos
                    export_canvas = canvas
                
                video_writer.write(export_canvas)
            
            # Advance according to speed (skip frames)
            if not paused:
                step = 1 if speed >= 1.0 else 1  # keep step 1; slow speed is achieved by delay
                frame_idx += step

        # --- Jump helper ---
        def _jump_seconds(delta_s: float):
            nonlocal frame_idx
            step_frames = int(round(delta_s * max(fps1, fps2)))
            frame_idx = int(np.clip(frame_idx + step_frames, 0, max_frames - 1))
            cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Keyboard - always process keys, even when paused
        wait_time = int(max(1, (1000 / max(fps1, fps2)) / max(0.25, speed))) if not paused else 30  # Longer wait when paused
        key = cv2.waitKey(wait_time) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (ord('+'), ord('=')):
            speed = min(2.0, speed * 1.25)
        elif key in (ord('-'), ord('_')):
            speed = max(0.25, speed / 1.25)
        # --- Jump controls ---
        elif key in (81, ord('a'), ord('A')):  # Left arrow or 'a' -> -2s
            _jump_seconds(-2.0)
        elif key in (83, ord('d'), ord('D')):  # Right arrow or 'd' -> +2s
            _jump_seconds(+2.0)
        elif key in (ord('s'), ord('S')):      # 's' -> +5s
            _jump_seconds(+10.0)
        elif key in (ord('f'), ord('F')):      # 'f' -> -5s
            _jump_seconds(-10.0)

    cap1.release(); cap2.release(); cv2.destroyAllWindows()
    
    # Clean up video writer
    if video_writer is not None:
        video_writer.release()
        print(f"[export] Video export completed: {export_path}")
    
    plt.ioff(); plt.show(block=False)

# -----------------------------
# GUIq
# -----------------------------
def launch_gui():
    """
    Simple Tkinter GUI to select inputs and run the synced video+plot player.
    """
    root = tk.Tk()
    root.title("ICHs – Dual video + LVM live plot")
    root.geometry("1400x800")

    # Auto-detect defaults from a folder (env var ICH_VIDEO_DIR can override)
    default_dir = os.environ.get(
        "ICH_VIDEO_DIR",
        "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/video_sync"
    )

    def _auto_pick_files(base_dir: str):
        try:
            mp4s = glob.glob(os.path.join(base_dir, "*.mp4"))
            mp4s = sorted(mp4s, key=lambda p: os.path.getmtime(p))
            lvms = glob.glob(os.path.join(base_dir, "*.lvm"))
            lvms = sorted(lvms, key=lambda p: os.path.getmtime(p))
        except Exception:
            mp4s, lvms = [], []
        # pick two most-recent mp4s and the most-recent lvm
        vid2 = mp4s[-1] if len(mp4s) >= 1 else ""
        vid1 = mp4s[-2] if len(mp4s) >= 2 else ""
        lvm  = lvms[-1] if len(lvms) >= 1 else ""
        return vid1, vid2, lvm

    _def_vid1, _def_vid2, _def_lvm = _auto_pick_files(default_dir)

    # Variables
    vid1_var = tk.StringVar(value=_def_vid1)
    vid2_var = tk.StringVar(value=_def_vid2)
    lvm_var  = tk.StringVar(value=_def_lvm)
    time_col_var = tk.StringVar(value="")
    data_col_var = tk.StringVar(value="")
    hold_side_var = tk.StringVar(value="G2L")  # Detecting hold side for sync (G2L default)
    match_force_var = tk.StringVar(value="Fy")  # Force used for matching (default Fy)
    offset_var = tk.DoubleVar(value=0.0)
    v1_off_var = tk.DoubleVar(value=0.0)
    v2_off_var = tk.DoubleVar(value=0.0)
    down_var = tk.IntVar(value=1)
    title_var = tk.StringVar(value="ICHs – Live synchronised plot")
    auto_sync_var = tk.BooleanVar(value=True)
    sync_method_var = tk.StringVar(value="motion")  # 'marker' or 'motion'
    motion_video_var = tk.StringVar(value="backV")  # 'backV' or 'sideV' for motion detection
    motion_metric_var = tk.StringVar(value="threshold_rising")  # 'peakForce', 'maxROFD_rising', 'maxROFD_falling', 'threshold_rising', 'threshold_falling', 'early_dent'
    motion_threshold_var = tk.DoubleVar(value=3.0)  # User input threshold for threshold-based metrics
    # Motion ROI (relative x,y,w,h). Default ROI; can be overridden via picker
    motion_roi_rel = [0.6, 0.2, 0.35, 0.6]
    # Video export options
    export_video_var = tk.BooleanVar(value=False)
    export_path_var = tk.StringVar(value="")
    guid_var = tk.StringVar(value=str(uuid.uuid4()))
    status_var = tk.StringVar(value=f"GUID: {guid_var.get()}")

    def browse_file(var: tk.StringVar, types):
        path = filedialog.askopenfilename(filetypes=types)
        if path:
            var.set(path)

    def load_lvm_and_fill():
        path = lvm_var.get().strip()
        if not os.path.exists(path):
            messagebox.showerror("LVM file", f"File not found:\n{path}")
            return
        try:
            df, tcol = load_lvm(path, None)
            print(f"[DEBUG] Loaded DataFrame shape: {df.shape if df is not None else 'None'}")
            print(f"[DEBUG] DataFrame columns: {list(df.columns) if df is not None else 'None'}")
            print(f"[DEBUG] Time column: {tcol}")
            
            if df is None or df.empty:
                messagebox.showerror("LVM parse error", "Failed to load LVM file or file is empty")
                return
            if not tcol or tcol not in df.columns:
                messagebox.showerror("LVM parse error", f"Could not find valid time column. Found columns: {list(df.columns)}")
                return
            t_candidates = [tcol] + [c for c in df.columns if c != tcol]
            time_col_var.set(t_candidates[0] if t_candidates else "")
            time_cb['values'] = t_candidates
            ncols = numeric_columns(df)
            if tcol in ncols:
                ncols = [c for c in ncols if c != tcol]
            data_cb['values'] = ncols
            if ncols:
                # Set default to Fy_2, Fz_2 if available, otherwise first column
                default_cols = []
                for col in ["Fy_2", "Fz_2"]:
                    if col in ncols:
                        default_cols.append(col)
                if default_cols:
                    data_col_var.set(", ".join(default_cols))
                else:
                    data_col_var.set(ncols[0])
            # Update default window title preview using acronym map
            try:
                acr = _file_acronym_from_path(path)
                status_var.set(f"LVM loaded. Detected time column: {tcol}. Title: {acr}")
                root.title(f"ICHs – {acr} live plot")
            except Exception:
                status_var.set(f"LVM loaded. Detected time column: {tcol}.")
        except IndexError as e:
            messagebox.showerror("LVM parse error", f"Index error while parsing LVM file: {str(e)}\nThis usually means the file format is not recognized.")
        except Exception as e:
            messagebox.showerror("LVM parse error", f"Error loading LVM file: {str(e)}")

    def _pick_motion_roi():
        """Open the first frame of selected video and let user draw an ROI; store as relative coords."""
        motion_video = motion_video_var.get()
        if motion_video == "backV":
            path = vid1_var.get().strip()
            required_string = "backV"
        else:  # sideV
            path = vid2_var.get().strip()
            required_string = "sideV"
            
        if not os.path.exists(path):
            messagebox.showerror(f"{motion_video} video", f"File not found:\n{path}")
            return
        if required_string not in os.path.basename(path):
            messagebox.showerror(f"{motion_video} video", f"Video must contain '{required_string}' for ROI selection.")
            return
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read(); cap.release()
        if not ok or frame is None:
            messagebox.showerror("Back video", "Could not read first frame for ROI selection.")
            return
        h, w = frame.shape[:2]
        try:
            rect = cv2.selectROI(f"Select motion ROI ({motion_video} video)", frame, showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows()
        except Exception as e:
            cv2.destroyAllWindows()
            messagebox.showerror("ROI selection", str(e))
            return
        x, y, rw, rh = rect
        if rw <= 0 or rh <= 0:
            messagebox.showwarning("ROI selection", "No ROI selected.")
            return
        motion_roi_rel[:] = [x / float(w), y / float(h), rw / float(w), rh / float(h)]
        status_var.set(f"Motion ROI set to rel={tuple(round(v,3) for v in motion_roi_rel)} ({motion_video})")
        
        # Show the ROI frame with the selected region highlighted
        _show_roi_frame(path, motion_roi_rel)

    def _show_roi_frame_for_motion():
        """Display the first frame of the selected motion video with the ROI highlighted."""
        motion_video = motion_video_var.get()
        if motion_video == "backV":
            path = vid1_var.get().strip()
        else:  # sideV
            path = vid2_var.get().strip()
        _show_roi_frame(path, motion_roi_rel)

    def _show_roi_frame(video_path: str, roi_rel: List[float]):
        """Display the first frame of the video with the ROI highlighted."""
        try:
            cap = cv2.VideoCapture(video_path)
            ok, frame = cap.read()
            cap.release()
            
            if not ok or frame is None:
                messagebox.showerror("ROI Display", "Could not read frame for ROI display.")
                return
            
            h, w = frame.shape[:2]
            x, y, rw, rh = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(roi_rel)]
            
            # Draw ROI rectangle
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
            
            # Add text label
            cv2.putText(frame, "Motion ROI", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow("Motion ROI Frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            messagebox.showerror("ROI Display", f"Error displaying ROI frame: {str(e)}")

    def start_run():
        # Validate inputs
        for p in (vid1_var.get(), vid2_var.get(), lvm_var.get()):
            if not os.path.exists(p):
                messagebox.showerror("Missing file", f"File not found:\n{p}")
                return
        # Enforce naming convention: backV and sideV
        if 'backV' not in os.path.basename(vid1_var.get()):
            messagebox.showerror("Video 1", "The first video must contain 'backV' in its filename.")
            return
        if 'sideV' not in os.path.basename(vid2_var.get()):
            messagebox.showerror("Video 2", "The second video must contain 'sideV' in its filename (green marker).")
            return
        if not data_col_var.get():
            messagebox.showerror("Column", "Please choose a data column to plot.")
            return
        # Fire the player (blocks UI until finished)
        root.withdraw()
        try:
            # Auto-sync (optional)
            lvm_df, _tcol = load_lvm(lvm_var.get(), None)
            delta = 0.0
            if auto_sync_var.get():
                sync_method = sync_method_var.get()
                print(f"[DEBUG] sync_method_var.get() = '{sync_method}'")
                if sync_method == 'marker':
                    print(f"[DEBUG] sync_first_peak_with_video FUNCTION START")
                    delta = sync_first_peak_with_video(lvm_df, vid2_var.get(), hold_side=hold_side_var.get(), match_force=match_force_var.get())
                    t_sync_video = _detect_side_video_marker_time(vid2_var.get())
                else:
                    # Motion-based: use selected video ROI motion onset
                    motion_video = motion_video_var.get()
                    if motion_video == "backV":
                        video_path = vid1_var.get()
                    else:  # sideV
                        video_path = vid2_var.get()
                    
                    print(f"[sync] Using {motion_video} video for motion detection: {os.path.basename(video_path)}")
                    t_motion = _detect_motion_time_in_video(video_path, roi_rel=tuple(motion_roi_rel))
                    if t_motion is None:
                        delta = 0.0
                        t_sync_video = None
                    else:
                        # Align data using selected metric
                        user_threshold = motion_threshold_var.get()
                        print(f"[sync] User threshold from GUI: {user_threshold}")
                        t_data = _compute_data_match_time(lvm_df, hold_side=hold_side_var.get(), match_force=match_force_var.get(), 
                                                        metric=motion_metric_var.get(), threshold=user_threshold)
                        if t_data is None:
                            delta = 0.0
                        else:
                            delta = float(t_motion - t_data)
                        t_sync_video = t_motion
                start_time = float(max(0.0, (t_sync_video or 0.0) - 5.0))
            play_two_videos_with_live_plot(
                vid1_path=vid1_var.get(),
                vid2_path=vid2_var.get(),
                lvm_path=lvm_var.get(),
                lvm_col=data_col_var.get(),
                time_col=(time_col_var.get() or None),
                lvm_offset=float(offset_var.get()) + float(delta),
                v1_offset=float(v1_off_var.get()),
                v2_offset=float(v2_off_var.get()),
                downsample=max(1, int(down_var.get())),
                title=title_var.get(),
                start_at_video_time=(start_time if auto_sync_var.get() else None),
                export_video=export_video_var.get(),
                export_path=export_path_var.get() if export_video_var.get() else None,
            )
        except Exception as e:
            messagebox.showerror("Run failed", str(e))
        finally:
            # new GUID for next run
            guid_var.set(str(uuid.uuid4()))
            status_var.set(f"GUID: {guid_var.get()}")
            root.deiconify()

    # Layout helpers
    def row(parent, r, label, widget, button=None):
        ttk.Label(parent, text=label, width=18).grid(row=r, column=0, sticky="e", padx=6, pady=4)
        widget.grid(row=r, column=1, sticky="we", padx=6, pady=4)
        if button:
            button.grid(row=r, column=2, sticky="w", padx=6, pady=4)

    # Auto-load LVM file and detect columns if files are found
    def auto_load_files():
        if _def_lvm and os.path.exists(_def_lvm):
            lvm_var.set(_def_lvm)
            load_lvm_and_fill()
            status_var.set(f"Auto-loaded: {os.path.basename(_def_lvm)}")
        else:
            status_var.set("No LVM file found in default directory")

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(1, weight=1)

    # File selectors
    v1_entry = ttk.Entry(frm, textvariable=vid1_var)
    v1_btn = ttk.Button(frm, text="Browse…", command=lambda: browse_file(vid1_var, [("Video", "*.mp4 *.MP4")]))
    row(frm, 0, "Video 1 (backV) (MP4):", v1_entry, v1_btn)

    v2_entry = ttk.Entry(frm, textvariable=vid2_var)
    v2_btn = ttk.Button(frm, text="Browse…", command=lambda: browse_file(vid2_var, [("Video", "*.mp4 *.MP4")]))
    row(frm, 1, "Video 2 (sideV) (MP4):", v2_entry, v2_btn)

    lvm_entry = ttk.Entry(frm, textvariable=lvm_var)
    lvm_btn = ttk.Button(frm, text="Browse…", command=lambda: browse_file(lvm_var, [("LVM", "*.lvm *.LVM"), ("All files", "*.*")]))
    row(frm, 2, "LVM File:", lvm_entry, lvm_btn)

    # Column selectors
    time_cb = ttk.Combobox(frm, textvariable=time_col_var, values=[], state="readonly")
    # Allow typing multiple tokens (comma-separated) in the data column field
    data_cb = ttk.Combobox(frm, textvariable=data_col_var, values=[], state="normal")
    load_btn = ttk.Button(frm, text="Detect Columns", command=load_lvm_and_fill)
    row(frm, 3, "Time column:", time_cb, load_btn)
    row(frm, 4, "Data column:", data_cb, None)
    # Hint for multi-column selection (own row for more vertical space)
    ttk.Label(
        frm,
        text="Tip: You can enter up to 3 tokens, comma-separated (e.g., Fy, Fz, Fres_xyz)",
        wraplength=750,
        justify="left"
    ).grid(row=5, column=1, columnspan=2, sticky="w", padx=6, pady=(0, 8))

    # Detecting hold selector
    hold_cb = ttk.Combobox(frm, textvariable=hold_side_var, values=["G2L", "G1R"], state="readonly")
    row(frm, 6, "Detecting hold:", hold_cb, None)

    # Matching force selector
    mf_cb = ttk.Combobox(frm, textvariable=match_force_var, values=["Fy", "Fz", "Fx", "Fres_xyz", "FgR"], state="readonly")
    row(frm, 7, "Matching force:", mf_cb, None)

    # Numeric params
    off_entry = ttk.Entry(frm, textvariable=offset_var)
    row(frm, 8, "LVM offset [s]:", off_entry, None)
    # Explanation for LVM offset
    ttk.Label(
        frm,
        text="LVM offset: + delays LVM curve (shifts right), - advances LVM curve (shifts left). Adds to auto-sync delta.",
        wraplength=750,
        justify="left",
        foreground="#666"
    ).grid(row=9, column=1, columnspan=2, sticky="w", padx=6, pady=(0, 8))

    v1off_entry = ttk.Entry(frm, textvariable=v1_off_var)
    row(frm, 10, "Video 1 offset [s]:", v1off_entry, None)

    v2off_entry = ttk.Entry(frm, textvariable=v2_off_var)
    row(frm, 11, "Video 2 offset [s]:", v2off_entry, None)

    down_entry = ttk.Entry(frm, textvariable=down_var)
    row(frm, 12, "Downsample (N):", down_entry, None)

    title_entry = ttk.Entry(frm, textvariable=title_var)
    row(frm, 13, "Plot title:", title_entry, None)

    # Status + actions
    status = ttk.Label(frm, textvariable=status_var, foreground="#555")
    status.grid(row=14, column=0, columnspan=3, sticky="w", padx=6, pady=(12, 4))

    # Auto-sync toggle
    sync_row = ttk.Frame(frm)
    sync_row.grid(row=15, column=0, columnspan=3, sticky="we", padx=6, pady=(8, 4))
    ttk.Checkbutton(sync_row, text="Auto sync", variable=auto_sync_var).pack(side="left", padx=(0,8))
    ttk.Label(sync_row, text="Method:").pack(side="left")
    ttk.Combobox(sync_row, values=["marker", "motion"], textvariable=sync_method_var, state="readonly", width=10).pack(side="left", padx=(6,0))
    ttk.Label(sync_row, text="Video:").pack(side="left", padx=(8,0))
    ttk.Combobox(sync_row, values=["backV", "sideV"], textvariable=motion_video_var, state="readonly", width=8).pack(side="left", padx=(6,0))
    ttk.Label(sync_row, text="Metric:").pack(side="left", padx=(8,0))
    ttk.Combobox(sync_row, values=["peakForce", "maxROFD_rising", "maxROFD_falling", "threshold_rising", "threshold_falling", "early_dent"], 
                textvariable=motion_metric_var, state="readonly", width=12).pack(side="left", padx=(6,0))
    ttk.Label(sync_row, text="Thresh:").pack(side="left", padx=(8,0))
    ttk.Entry(sync_row, textvariable=motion_threshold_var, width=6).pack(side="left", padx=(6,0))
    ttk.Button(sync_row, text="Pick ROI", command=_pick_motion_roi).pack(side="left", padx=(8,0))
    ttk.Button(sync_row, text="Show ROI", command=lambda: _show_roi_frame_for_motion()).pack(side="left", padx=(8,0))

    # Video export controls
    export_row = ttk.Frame(frm)
    export_row.grid(row=16, column=0, columnspan=3, sticky="we", padx=6, pady=(8, 4))
    ttk.Checkbutton(export_row, text="Export video", variable=export_video_var).pack(side="left", padx=(0,8))
    export_entry = ttk.Entry(export_row, textvariable=export_path_var, width=50)
    export_entry.pack(side="left", padx=(0,8))
    ttk.Button(export_row, text="Browse...", command=lambda: browse_file(export_path_var, [("MP4", "*.mp4")])).pack(side="left", padx=(0,8))
    ttk.Label(export_row, text="(Optional: export combined video with live plot)").pack(side="left", padx=(8,0))

    btns = ttk.Frame(frm)
    btns.grid(row=17, column=0, columnspan=3, sticky="e", padx=6, pady=8)
    ttk.Button(btns, text="Start", command=start_run).pack(side="right", padx=6)
    ttk.Button(btns, text="Quit", command=root.destroy).pack(side="right", padx=6)

    # Auto-load files when GUI starts
    root.after(100, auto_load_files)  # Small delay to ensure GUI is fully loaded

    root.mainloop()

# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Play two videos with live LVM plot (synced)")
    p.add_argument('--vid1', type=str, required=False, default='/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/video_sync/039_backV_Gr_red-standard_GX010853.mp4', help='Path to first MP4')
    p.add_argument('--vid2', type=str, required=False, default='/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/video_sync/039_sideV_Gr_red-standard_GX010247.mp4', help='Path to second MP4')
    p.add_argument('--lvm',  type=str, required=False, default='/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/video_sync/039-Hold-01_Red_Standard_73kg_Noah_lvl-ADV_25-05-14_1030.lvm', help='Path to .lvm file')
    p.add_argument('--col', type=str, required=False, default=None,
                   help='Column token to plot (e.g., Fy, Fz, Fx, Mz). Token matches multiple columns (e.g., Fy_1 and Fy_2). Optional when using --gui.')
    p.add_argument('--time-col', type=str, default=None, help='Name of time column in LVM (auto-detected if omitted)')
    p.add_argument('--offset', type=float, default=0.0, help='Global offset for LVM curve in seconds (positive delays curve)')
    p.add_argument('--v1-offset', type=float, default=0.0, help='Optional offset for video 1 time origin [s]')
    p.add_argument('--v2-offset', type=float, default=0.0, help='Optional offset for video 2 time origin [s]')
    p.add_argument('--downsample', type=int, default=1, help='Plot every Nth LVM sample to lighten plotting load')
    p.add_argument('--title', type=str, default='ICHs – Live synchronised plot', help='Matplotlib window title')
    p.add_argument('--guid', type=str, default=None, help='Unique identifier for this run (GUID). If not provided, one will be generated automatically.')
    p.add_argument('--gui', action='store_true', help='Launch a GUI to select inputs instead of using CLI args')
    p.add_argument('--bw', type=float, default=None, help='Body weight in kg (if omitted, try to parse from filename like _73kg). Used to normalize forces to %BW.')
    p.add_argument('--sg-window', type=int, default=21, help='Savitzky–Golay window length (odd).')
    p.add_argument('--sg-poly', type=int, default=3, help='Savitzky–Golay polynomial order (< window).')
    p.add_argument('--interp', type=str, choices=['linear', 'nearest'], default='linear',
                   help='Interpolation for mapping time to data values (linear looks smoother; nearest shows steps).')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.gui:
        # Launch GUI mode and exit afterwards
        launch_gui()
        return 0
    if args.guid is None:
        args.guid = str(uuid.uuid4())
    print(f"[info] Run GUID: {args.guid}")
    if not os.path.exists(args.vid1):
        print(f"[warn] vid1 not found: {args.vid1}")
    if not os.path.exists(args.vid2):
        print(f"[warn] vid2 not found: {args.vid2}")
    if not os.path.exists(args.lvm):
        print(f"[warn] lvm not found: {args.lvm}")
    try:
        play_two_videos_with_live_plot(
            vid1_path=args.vid1,
            vid2_path=args.vid2,
            lvm_path=args.lvm,
            lvm_col=args.col,
            time_col=args.time_col,
            lvm_offset=args.offset,
            v1_offset=args.v1_offset,
            v2_offset=args.v2_offset,
            downsample=max(1, int(args.downsample)),
            bw_kg=args.bw,
            sg_window=args.sg_window,
            sg_poly=args.sg_poly,
            interp_mode=args.interp,
            title=args.title,
        )
        return 0
    except Exception as e:
        print(f"[error] {e}")
        return 1


if __name__ == '__main__':
    import sys as _sys
    # If launched without CLI args (e.g., Run button), open GUI by default
    if len(_sys.argv) <= 1:
        raise SystemExit(main(['--gui']))
    raise SystemExit(main())
