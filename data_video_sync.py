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
    except Exception as e:
        print(f"[load_lvm] loadData.py route failed: {e}")

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

    # Build a temporary csv text from header_idx onward
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

    # Clean up: drop all-empty columns, coerce to numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='ignore')
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.astype(str).str.contains('^Unnamed')]

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
    # fallback
    return df.columns[0]


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
    sg_window: int = 21,
    sg_poly: int = 3,
    interp_mode: str = 'linear',
    title: Optional[str] = None,
):
    # Load LVM
    df, tcol = load_lvm(lvm_path, time_col)
    # Resolve which columns to plot: token match (e.g., 'Fy' -> ['Fy_1','Fy_2'])
    plot_cols = _pick_columns_by_token(df, lvm_col) if lvm_col else []
    if not plot_cols and lvm_col and lvm_col in df.columns:
        plot_cols = [lvm_col]
    if not plot_cols:
        raise ValueError(
            f"Column token '{lvm_col}' not found in LVM. Available: {list(df.columns)}"
        )

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
        # --- Normalize to %BW if column appears to be in Newton (no '%' in name) and bw available ---
        if normalize_to_bw and ('%' not in str(c)):
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
    fig.canvas.manager.set_window_title(title or f"LVM live: {lvm_col}")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(f"{lvm_col} (SG filtered" + (", %BW" if normalize_to_bw else "") + ")")
    # keep a scrolling window of last N seconds
    window_s = 10.0
    lines = {}
    for c in plot_cols:
        lobj, = ax.plot([], [], lw=2, label=str(c))
        lines[c] = lobj
    if plot_cols:
        ax.legend(loc='best')
    ax.grid(True, linestyle=':')

    # Buffers for the last window, per column
    buffers = {c: {"t": [], "y": []} for c in plot_cols}

    # Prepare OpenCV canvas (side-by-side)
    out_h = max(h1, h2)
    out_w = w1 + w2
    win_name = "ICHs: Dual video + live plot (focus this window for keys)"

    frame_idx = 0
    while frame_idx < max_frames:
        if not paused:
            f1 = read_frame(cap1)
            f2 = read_frame(cap2)
            if f1 is None or f2 is None:
                break
            # Compute current time from each stream
            t1 = frame_idx / fps1 + float(v1_offset)
            t2 = frame_idx / fps2 + float(v2_offset)
            t_now = 0.5 * (t1 + t2)

            # Update per-series buffers
            t_min = t_now - window_s
            for cname, idx_fn in y_at_map.items():
                buffers[cname]["t"].append(t_now)
                buffers[cname]["y"].append(float(idx_fn(t_now)))
                # Trim
                bt = buffers[cname]["t"]
                by = buffers[cname]["y"]
                while bt and bt[0] < t_min:
                    bt.pop(0); by.pop(0)
                # Update line data
                lines[cname].set_data(bt, by)

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

            cv2.imshow(win_name, canvas)
            # Advance according to speed (skip frames)
            step = 1 if speed >= 1.0 else 1  # keep step 1; slow speed is achieved by delay
            frame_idx += step

        # --- Jump helper ---
        def _jump_seconds(delta_s: float):
            nonlocal frame_idx
            step_frames = int(round(delta_s * max(fps1, fps2)))
            frame_idx = int(np.clip(frame_idx + step_frames, 0, max_frames - 1))
            cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Keyboard
        key = cv2.waitKey(int(max(1, (1000 / max(fps1, fps2)) / max(0.25, speed)))) & 0xFF
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
            _jump_seconds(+5.0)
        elif key in (ord('f'), ord('F')):      # 'f' -> -5s
            _jump_seconds(-5.0)

    cap1.release(); cap2.release(); cv2.destroyAllWindows()
    plt.ioff(); plt.show(block=False)

# -----------------------------
# GUI
# -----------------------------
def launch_gui():
    """
    Simple Tkinter GUI to select inputs and run the synced video+plot player.
    """
    root = tk.Tk()
    root.title("ICHs – Dual video + LVM live plot")
    root.geometry("750x420")

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
    offset_var = tk.DoubleVar(value=0.0)
    v1_off_var = tk.DoubleVar(value=0.0)
    v2_off_var = tk.DoubleVar(value=0.0)
    down_var = tk.IntVar(value=1)
    title_var = tk.StringVar(value="ICHs – Live synchronised plot")
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
            t_candidates = [tcol] + [c for c in df.columns if c != tcol]
            time_col_var.set(t_candidates[0] if t_candidates else "")
            time_cb['values'] = t_candidates
            ncols = numeric_columns(df)
            if tcol in ncols:
                ncols = [c for c in ncols if c != tcol]
            data_cb['values'] = ncols
            if ncols:
                data_col_var.set(ncols[0])
            status_var.set(f"LVM loaded. Detected time column: {tcol}.")
        except Exception as e:
            messagebox.showerror("LVM parse error", str(e))

    def start_run():
        # Validate inputs
        for p in (vid1_var.get(), vid2_var.get(), lvm_var.get()):
            if not os.path.exists(p):
                messagebox.showerror("Missing file", f"File not found:\n{p}")
                return
        if not data_col_var.get():
            messagebox.showerror("Column", "Please choose a data column to plot.")
            return
        # Fire the player (blocks UI until finished)
        root.withdraw()
        try:
            play_two_videos_with_live_plot(
                vid1_path=vid1_var.get(),
                vid2_path=vid2_var.get(),
                lvm_path=lvm_var.get(),
                lvm_col=data_col_var.get(),
                time_col=(time_col_var.get() or None),
                lvm_offset=float(offset_var.get()),
                v1_offset=float(v1_off_var.get()),
                v2_offset=float(v2_off_var.get()),
                downsample=max(1, int(down_var.get())),
                title=title_var.get(),
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

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(1, weight=1)

    # File selectors
    v1_entry = ttk.Entry(frm, textvariable=vid1_var)
    v1_btn = ttk.Button(frm, text="Browse…", command=lambda: browse_file(vid1_var, [("Video", "*.mp4 *.MP4")]))
    row(frm, 0, "Video 1 (MP4):", v1_entry, v1_btn)

    v2_entry = ttk.Entry(frm, textvariable=vid2_var)
    v2_btn = ttk.Button(frm, text="Browse…", command=lambda: browse_file(vid2_var, [("Video", "*.mp4 *.MP4")]))
    row(frm, 1, "Video 2 (MP4):", v2_entry, v2_btn)

    lvm_entry = ttk.Entry(frm, textvariable=lvm_var)
    lvm_btn = ttk.Button(frm, text="Browse…", command=lambda: browse_file(lvm_var, [("LVM", "*.lvm *.LVM"), ("All files", "*.*")]))
    row(frm, 2, "LVM File:", lvm_entry, lvm_btn)

    # Column selectors
    time_cb = ttk.Combobox(frm, textvariable=time_col_var, values=[], state="readonly")
    data_cb = ttk.Combobox(frm, textvariable=data_col_var, values=[], state="readonly")
    load_btn = ttk.Button(frm, text="Detect Columns", command=load_lvm_and_fill)
    row(frm, 3, "Time column:", time_cb, load_btn)
    row(frm, 4, "Data column:", data_cb, None)

    # Numeric params
    off_entry = ttk.Entry(frm, textvariable=offset_var)
    row(frm, 5, "LVM offset [s]:", off_entry, None)

    v1off_entry = ttk.Entry(frm, textvariable=v1_off_var)
    row(frm, 6, "Video 1 offset [s]:", v1off_entry, None)

    v2off_entry = ttk.Entry(frm, textvariable=v2_off_var)
    row(frm, 7, "Video 2 offset [s]:", v2off_entry, None)

    down_entry = ttk.Entry(frm, textvariable=down_var)
    row(frm, 8, "Downsample (N):", down_entry, None)

    title_entry = ttk.Entry(frm, textvariable=title_var)
    row(frm, 9, "Plot title:", title_entry, None)

    # Status + actions
    status = ttk.Label(frm, textvariable=status_var, foreground="#555")
    status.grid(row=10, column=0, columnspan=3, sticky="w", padx=6, pady=(12, 4))

    btns = ttk.Frame(frm)
    btns.grid(row=11, column=0, columnspan=3, sticky="e", padx=6, pady=8)
    ttk.Button(btns, text="Start", command=start_run).pack(side="right", padx=6)
    ttk.Button(btns, text="Quit", command=root.destroy).pack(side="right", padx=6)

    root.mainloop()

# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Play two videos with live LVM plot (synced)")
    p.add_argument('--vid1', type=str, required=False, default='/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/video_sync/039_backV_Gr_red-standard_GX010853.mp44', help='Path to first MP4')
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
    raise SystemExit(main())
