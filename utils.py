import tkinter as tk
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
def clean_data(df):
    """
    Entfernt alle Spalten, die ein 'U' im Namen oder 'Comment' enthalten.
    """
    columns_to_drop = [col for col in df.columns if "U" in col or "Comment" in col or "X_Value" in col]
    df_clean = df.drop(columns=columns_to_drop)
    # Entferne doppelte Leerzeichen aus Spaltennamen
    df_clean.columns = df_clean.columns.str.replace(r'\s+', ' ', regex=True)
    return df_clean


def get_min_max_values(df):
    """
    Bestimmt den  minimalen und maximalen Wert eines datasets (außer 'Time [s]') aus dem DataFrame.
    """
    numeric_cols = [col for col in df.columns if col != "Time [s]"]
    min_val = df[numeric_cols].min().min()
    max_val = df[numeric_cols].max().max()
    return min_val, max_val

def split_data(df):
    """
    Teilt den DataFrame in zwei Teile:
      - griff_left: Enthält 'Time [s]' und alle Spalten, die eine '2' im Namen haben
      - griff_right: Enthält 'Time [s]' und alle Spalten, die eine '1' im Namen haben
    """
    griff_left = df[["Time [s]"] + [col for col in df.columns if "2" in col]]
    griff_right = df[["Time [s]"] + [col for col in df.columns if "1" in col]]
    return griff_left, griff_right

def get_min_max_values_per_column(df):
    """
    Gibt für jede Spalte (außer 'Time [s]') das Minimum und Maximum als Dictionary zurück.
    """
    return {
        col: {
            "min": df[col].min(),
            "max": df[col].max()
        }
        for col in df.columns if col != "Time [s]"
    }

def compute_global_ylimits_for_plots(plot_dict, forces_g1, forces_g2, margin=1.2):
    """
    Berechnet die globalen y-Achsen-Grenzen für beide Griffe (G2L und G1R)
    und fügt diese als 'global_y_limits' zum plot_dict hinzu.
    Es wird das extremste y_max und das kleinste y_min beider Griffe verwendet.
    """
    global_y_limits = {}

    # Initialisiere max/min für G1 und G2 mit extremen Werten
    cols_g2 = [col for col in plot_dict["G2L"]["data"].columns if any(force in col for force in forces_g2)]
    min_g2 = plot_dict["G2L"]["data"][cols_g2].min().min()
    max_g2 = plot_dict["G2L"]["data"][cols_g2].max().max()

    cols_g1 = [col for col in plot_dict["G1R"]["data"].columns if any(force in col for force in forces_g1)]
    min_g1 = plot_dict["G1R"]["data"][cols_g1].min().min()
    max_g1 = plot_dict["G1R"]["data"][cols_g1].max().max()

    y_min = min(min_g1, min_g2)
    y_max = max(max_g1, max_g2)

    # Berechne das globale y_max und y_min (mit Sicherheitsmarge)
    global_y_max = y_max
    global_y_min = y_min

    # Speichern der globalen Limits in global_y_limits
    global_y_limits["global_y_min"] = global_y_min
    global_y_limits["global_y_max"] = global_y_max

    # Füge die globalen Limits zum plot_dict hinzu
    plot_dict["G1R"]["global_y_limits"] = global_y_limits
    plot_dict["G2L"]["global_y_limits"] = global_y_limits

    # Rückgabe des aktualisierten plot_dict
    return plot_dict

def get_force_suffix(forces_to_plot):
    """
    Erstellt einen Suffix-String basierend auf den ausgewählten Kräften.
    Gibt '_all_' zurück, wenn alle Kräfte aktiv sind, sonst z.B. '_only_Fx_Fy'.
    """
    all_forces = {"Fx", "Fy", "Fz", "Mz", "FgR", "FgR_calc"}

    selected_forces = set()
    for side in ["G1", "G2"]:
        for force, active in forces_to_plot.get(side, {}).items():
            if force != "all" and active:
                selected_forces.add(force)

    if selected_forces == all_forces:
        return "_all_"
    else:
        sorted_forces = sorted(selected_forces)
        return "_only_" + "_".join(sorted_forces)

def trim_dataframe_by_time(df, start_seconds, end_seconds):
    """
    Schneidet einen DataFrame anhand der Zeitspalte 'Time [s]' vorne und hinten zu.
    
    Parameters:
        df : pandas.DataFrame
            Der zu trimmende DataFrame.
        start_seconds : int
            Anzahl der Sekunden, die vom Anfang abgeschnitten werden.
        end_seconds : int
            Anzahl der Sekunden, die vom Ende abgeschnitten werden.
    
    Returns:
        df : pandas.DataFrame
            Der getrimmte DataFrame.
    """
    if "Time [s]" not in df.columns:
        return df  # keine Zeitspalte vorhanden

    t_min = df["Time [s]"].min() + start_seconds
    t_max = df["Time [s]"].max() - end_seconds

    return df[(df["Time [s]"] >= t_min) & (df["Time [s]"] <= t_max)].reset_index(drop=True)

def prepare_data(current_dict, forces_to_plot, cutoff):
    if current_dict is None:
        print("Keine .lvm-Dateien gefunden.")
        return None, [], []
    
    if cutoff["start"] > 0 or cutoff["end"] > 0:
        for filename, file_data in current_dict.items():
            for hold in ["G1R", "G2L"]:
                if hold in file_data:
                    df = file_data[hold]["data"]
                    trimmed_df = trim_dataframe_by_time(df, start_seconds=cutoff["start"], end_seconds=cutoff["end"])
                    current_dict[filename][hold]["data"] = trimmed_df

    forces_g1 = [k for k, v in forces_to_plot["G1"].items() if k != "all" and v]
    forces_g2 = [k for k, v in forces_to_plot["G2"].items() if k != "all" and v]
    return current_dict, forces_g1, forces_g2

def print_current_dict_summary(current_dict):
    for key in current_dict:
        print("\nCurrent_dict of File")
        print(f"{key} →")
        for hold, content in current_dict[key].items():
            if not isinstance(content, dict):
                continue  # Skip non-dict entries like 'climberforce'
            print(f"  {hold}: {list(content.keys())}")
            print(f"    Spalten: {content['data'].columns.tolist()}")
            print(f"    stats {hold}:")
            for k, v in content["stats"].items():
                min_v = f"{v['min']:.1f}"
                max_v = f"{v['max']:.1f}"
                print(f"      {k}: min= {min_v}, max= {max_v}")
            if "global_y_limits" in content:
                gl = content["global_y_limits"]
                print(f"    global_y_limits: min= {gl['global_y_min']:.2f}, max= {gl['global_y_max']:.2f}")
            else:
                print("    global_y_limits: Nicht gesetzt")


def _find_active_segment(time: pd.Series, mask: pd.Series, min_duration: float) -> Optional[Tuple[float, float]]:
    """
    Hilfsfunktion: Findet das erste Segment, in dem mask=True für mindestens min_duration Sekunden.
    Gibt (t_start, t_end) zurück oder None.
    """
    # Kennzeichne Gruppen zusammenhängender True/False-Werte
    grp = (mask != mask.shift()).cumsum()
    # Erstelle DataFrame mit explizit benannter Mask-Spalte
    df_seg = pd.concat([
        time.rename(time.name),
        mask.rename("mask"),
        grp.rename("grp")
    ], axis=1)
    # Suche das erste True-Segment mit ausreichender Dauer
    for _, g in df_seg.groupby("grp"):
        if g["mask"].iloc[0]:
            duration = g[time.name].iloc[-1] - g[time.name].iloc[0]
            if duration >= min_duration:
                return (g[time.name].iloc[0], g[time.name].iloc[-1])
    return None
def _find_last_active_segment(time: pd.Series,
                              mask: pd.Series,
                              min_duration: float) -> Optional[Tuple[float, float]]:
    """
    Findet das letzte Segment, in dem mask==True für mindestens min_duration Sekunden.
    """
    # Gruppen für zusammenhängende True/False
    grp = (mask != mask.shift()).cumsum()
    last_valid = None
    # iteriere über Gruppen
    df = pd.concat([time, mask.rename("m"), grp.rename("grp")], axis=1)
    for _, g in df.groupby("grp"):
        if g["m"].iloc[0]:
            dur = g[time.name].iloc[-1] - g[time.name].iloc[0]
            if dur >= min_duration:
                last_valid = (g[time.name].iloc[0], g[time.name].iloc[-1])
    return last_valid

def get_force_contact_times(
    df: pd.DataFrame,
    forces: List[str],
    # Optional single absolute threshold for both start and end
    threshold: Optional[float] = None,
    start_frac: float = 0.05,
    start_dur: float = 0.1,
    end_frac: float = 0.05,
    end_dur: float = 0.01,
    time_col: str = "Time [s]",
    start_threshold: Optional[float] = None,
    end_threshold: Optional[float] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Bestimmt Kontaktzeiten anhand von 'Fz' im DataFrame.
    Nur die Kraft 'Fz' wird zur Detektion der Kontaktzeiten verwendet.
    Für alle anderen Kräfte werden exakt dieselben Zeitabschnitte (Kontaktzeiten) verwendet wie für 'Fz'.
    
    Für jede Kontaktzeit gilt zusätzlich:
      - Das Intervall beginnt, wenn 'Fz' für mindestens start_dur Sekunden oberhalb des Startschwellwertes liegt.
      - Das Intervall endet, wenn 'Fz' für mindestens end_dur Sekunden unterhalb des Endschwellwertes bleibt.
      - Innerhalb des Intervalls muss 'Fz' mindestens einmal ≥ 5 betragen.

    Rückgabe:
      { 'Fz': [(t0, t1), ...], 'Fx': [(t0, t1), ...], ... }
    """
    # Default hardcoded thresholds & force set (for quick tests)
    start_threshold = 1.3
    end_threshold = 1.1
    forces = {"Fz"}
    contact_time = {}
    time = df[time_col]
    # If a single threshold is given, override only the start threshold
    if threshold is not None:
        start_threshold = threshold

    for force in forces:
        # max-Wert über alle Spalten dieser Kraft
        cols = [c for c in df.columns if force in c]
        if not cols:
            continue
        series = df[cols].max(axis=1)
        # Determine start/end thresholds: use absolute threshold if provided, else fraction of max
        threshold_start = (
            start_threshold if start_threshold is not None
            else series.max() * start_frac
        )
        threshold_end = (
            end_threshold if end_threshold is not None
            else series.max() * end_frac
        )

        # Build masks: start = rising above start_threshold; end = falling to/below end_threshold
        mask_start = series > threshold_start
        mask_end   = series <= threshold_end

        # Find all rising-edge indices where an interval could start
        start_edges = mask_start & ~mask_start.shift(fill_value=False)
        start_idxs = time.index[start_edges]

        force_intervals = []
        for start_idx in start_idxs:
            t0 = time.loc[start_idx]
            # Only consider end after t0
            mask_end_after = mask_end & (time >= t0)
            if mask_end_after.any():
                end_idx = mask_end_after[mask_end_after].index[0]
                t1 = time.loc[end_idx]
            else:
                t1 = time.iloc[-1]
            # Accept only intervals where duration ≥ end_dur
            if (t1 - t0) >= end_dur:
                # --- Zusatzbedingung: min. einmal Fz >= 5 innerhalb des Intervalls ---
 
                # Suche die richtige Fz-Spalte im DataFrame - pro Griff-Dict gibt es nur eine
                Fz_cols = [col for col in df.columns if "Fz" in col]
                mask_interval = (df[time_col] >= t0) & (df[time_col] <= t1)
                # Prüfe, ob irgendein Wert >= 5 ist
                if Fz_cols and (df.loc[mask_interval, Fz_cols] >= 5).any().any():
                    force_intervals.append((t0, t1))

        contact_time[force] = force_intervals

    return contact_time


def compute_impulses(
    df: pd.DataFrame,
    contact_time: Dict,
    forces: List,
    time_col: str = "Time [s]"
) -> Dict:
    """
    Berechnet den Impuls (Integral über Kraft·Zeit) für jede Kraft im gegebenen Intervall.
    
    Parameters:
      df         : DataFrame mit Time- und Kraftspalten
      contact_time  : dict mapping Kraftname → (t_start, t_end)
      forces     : Liste der Kraftstichworte (z.B. ["Fy","Fx","Fz","Mz"])
      time_col   : Name der Zeitspalte (Default "Time [s]")
    
    Returns:
      impulses : dict mapping Kraftname → Impuls [N·s] bzw. [%·s] (je nach Normierung)
    """
    print("calculating impulses")
    impulses = {}
    for force in forces:
        if force not in contact_time:
            continue
        t0, t1 = contact_time[force]
        # Subset des DataFrames auf das Intervall
        mask = (df[time_col] >= t0) & (df[time_col] <= t1)
        t_vals = df.loc[mask, time_col].values
        # Summiere ggf. mehrere Spalten (Sensorkanäle) auf Zeilenebene
        force_cols = [c for c in df.columns if force in c]
        if not force_cols or len(t_vals) < 2:
            impulses[force] = 0.0
            continue
        f_vals = df.loc[mask, force_cols].sum(axis=1).values
        imp = float(np.trapz(f_vals, t_vals))
        impulses[force] = imp
        # Debug: print each force's impulse
        print(f"[compute_impulses] {force}: Impuls = {imp:.2f}")
    return impulses

def compute_impulses_per_contact(
    df: pd.DataFrame,
    contact_time: List,
    force: str,
    time_col: str = "Time [s]"
) -> List:
    """
    Berechnet den Impuls (Integral über Kraft·Zeit) für jedes Intervall in `contact_time`.
    """
    impulses = []
    for t0, t1 in contact_time:
        mask = (df[time_col] >= t0) & (df[time_col] <= t1)
        t_vals = df.loc[mask, time_col].values
        cols = [c for c in df.columns if force in c]
        if not cols or len(t_vals) < 2:
            impulses.append(0.0)
            continue
        f_vals = df.loc[mask, cols].sum(axis=1).values
        imp = float(np.trapz(f_vals, t_vals))
        imp = round(imp, 1)
        impulses.append(imp)
    return impulses


# -------------------------------------------------------------
# Hilfsfunktion: Eindeutige, ausgewählte Kräfte extrahieren
def get_unique_selected_forces(forces_to_plot: Dict[str, Dict[str, bool]]) -> List[str]:
    """
    Gibt eine Liste eindeutiger Kraftnamen zurück, die in forces_to_plot
    auf True gesetzt sind. Ignoriert den Schlüssel 'all'.

    Args:
        forces_to_plot (Dict[str, Dict[str, bool]]): Dictionary mit Griffen 'G1' und 'G2',
            die jeweils eine Dict von Kraftnamen zu bools enthalten.

    Returns:
        List[str]: Eine Liste eindeutiger Kraftnamen, für die der Wert True ist.
    """
    unique_forces: List[str] = []
    for grip_dict in forces_to_plot.values():
        for force, selected in grip_dict.items():
            if force != "all" and selected and force not in unique_forces:
                unique_forces.append(force)
    return unique_forces

def print_nested_keys(d, indent=0):
    """
    Gibt alle Keys einer verschachtelten Dictionary-Struktur aus.
    """
    for key, value in d.items():
        print("  " * indent + f"- {key}")
        if isinstance(value, dict):
            print_nested_keys(value, indent + 1)


def trim_low_force_periods(df, force_cols: List[str] = ["Fy"], threshold=10, min_duration=3.0, buffer=3.0):
    """
    Entfernt Abschnitte, in denen alle angegebenen Kräfte über längere Zeit unterhalb eines Schwellenwerts liegen.
    Behalte jedoch jeweils einen Puffer von 'buffer' Sekunden davor und danach.
    """
    time = df["Time [s]"]
    mask = np.zeros(len(df), dtype=bool)

    for col in force_cols:
        if col not in df.columns:
            continue
        abs_low = df[col].abs() < threshold
        mask = mask | abs_low  # Nur wenn ALLE Kräfte niedrig, könnte man `&=` verwenden

    low_force_df = df[mask]
    intervals = []
    start_idx = None
    for i, is_low in enumerate(mask):
        if is_low and start_idx is None:
            start_idx = i
        elif not is_low and start_idx is not None:
            end_idx = i
            duration = time.iloc[end_idx-1] - time.iloc[start_idx]
            if duration >= min_duration:
                t_start = max(0, time.iloc[start_idx] - buffer)
                t_end = time.iloc[end_idx-1] + buffer
                intervals.append((t_start, t_end))
            start_idx = None
    # Falls am Ende noch ein Intervall offen ist
    if start_idx is not None:
        duration = time.iloc[-1] - time.iloc[start_idx]
        if duration >= min_duration:
            t_start = max(0, time.iloc[start_idx] - buffer)
            t_end = time.iloc[-1] + buffer
            intervals.append((t_start, t_end))

    # Jetzt löschen wir alle Zeitpunkte innerhalb dieser "Inaktivitätsintervalle"
    drop_mask = np.zeros(len(df), dtype=bool)
    for t0, t1 in intervals:
        drop_mask |= (df["Time [s]"] >= t0) & (df["Time [s]"] <= t1)

    return df[~drop_mask].reset_index(drop=True)