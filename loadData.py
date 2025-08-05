from typing import List, Dict, Any
import glob
import os
import re
import json
import config
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from utils import clean_data, get_min_max_values_per_column
from utils import get_force_contact_times, compute_impulses_per_contact, trim_low_force_periods
from additional_calculations import compute_hausdorff_dimensions_all_axes, calc_hausdorff_dimension_for_single_signal, plot_hausdorff_intervals


# --- Neue Hilfsfunktion: Zeitbereich aus Kontaktzeiten bestimmen ---
def get_trim_range_from_contact_times(g1r: pd.DataFrame, g2l: pd.DataFrame, autotrim: bool) -> tuple:
    """
    Ermittelt den globalen Start- und Endzeitpunkt basierend auf den Kontaktzeiten
    beider Griffseiten. Nur relevant, wenn autotrim aktiviert ist.
    """
    if not autotrim:
        global_start = g1r["Time [s]"].iloc[0]
        global_end = g1r["Time [s]"].iloc[-1]
        return global_start, global_end

    temp_data = {
        "G1R": {"data": g1r.copy()},
        "G2L": {"data": g2l.copy()}
    }
    force_keys = ["Fy", "Fx", "Fz", "Mz", "Fres_xyz", "Fres_yz"]
    compute_contact_times(temp_data, force_keys)

    all_times = []
    for side in ["G1R", "G2L"]:
        contact_times = temp_data[side].get("contact_time", {})
        all_times.extend(t for force in contact_times.values() for (t0, t1) in force for t in (t0, t1))

    if all_times:
        global_start = max(0, min(all_times) - 3)
        global_end = max(all_times) + 3
    else:
        global_start = min(g1r["Time [s]"].iloc[0], g2l["Time [s]"].iloc[0])
        global_end = max(g1r["Time [s]"].iloc[-1], g2l["Time [s]"].iloc[-1])

    return global_start, global_end


# --- Hilfsfunktionen ---

def correct_time_jumps(time_series, threshold=2.0):
    corrected = time_series.copy()
    offset = 0.0
    for i in range(1, len(time_series)):
        dt = time_series.iloc[i] - time_series.iloc[i - 1]
        if dt > threshold:
            sprung = dt
            corrected.iloc[i:] = corrected.iloc[i:] - sprung
    return corrected

def prepare_time_column(df, autotrim=True):
    df["Time [s]"] = correct_time_jumps(df["Time [s]"])
    if autotrim:
        df["Time [s]"] -= df["Time [s]"].iloc[0]
    return df

def parse_metadata_from_filename(file_name):
    result = {
        "athlete": "Unbekannt",
        "weight": 100,
        "identity": "Unknown",
        "file_number": "000",
        "level": "unknown",
    }
    if match := re.search(r"_(\d+)kg", file_name):
        result["weight"] = int(match.group(1))
    if match := re.search(r"(?<=_)([^_]+)(?=_lvl-)", file_name):
        result["athlete"] = match.group(1)
    # Extract climber's level
    if match := re.search(r"_lvl-([^_]+)", file_name):
        result["level"] = match.group(1)
    if match := re.search(r"^(.+?)_\d+kg", file_name):
        result["identity"] = match.group(1)
    # Extract 3-digit file number from the start of the identity
    if match := re.match(r"(\d{3})", result["identity"]):
        result["file_number"] = match.group(1)
        config.file_number = result["file_number"]
        config.processed_files_list.append(result["file_number"])
    return result

def split_grip_sides(df):
    g1r = df[["Time [s]"] + ["FgR_sum [%]"] + [col for col in df.columns if "1" in col]]
    g2l = df[["Time [s]"] + ["FgR_sum [%]"]+ [col for col in df.columns if "2" in col]]
    return g1r, g2l


# --- Datenberechnung ---

def compute_contact_times(file_data: Dict, force_keys: List[str]) -> None:
    for side in ["G1R", "G2L"]:
        contact_times = get_force_contact_times(file_data[side]["data"], force_keys)

        # Entferne Intervalle, die kürzer als 0.5 Sekunden sind
        for force, intervals in contact_times.items():
            filtered = [(t0, t1) for (t0, t1) in intervals if (t1 - t0) >= 0.5]
            # --- Additional force threshold filter ---
            df = file_data[side]["data"]
            extended_valid_intervals = []
            for t0, t1 in filtered:
                mask = (df["Time [s]"] >= t0) & (df["Time [s]"] <= t1)
                subdf = df.loc[mask]
                force_cols = [col for col in subdf.columns if force in col and "[" in col]
                # Check if any relevant force column exceeds 150 N or 20% of body weight at any point in interval
                force_valid = any(
                    (subdf[col] > 150).any() or (subdf[col] > 20).any()
                    for col in force_cols
                )
                if force_valid:
                    extended_valid_intervals.append((t0, t1))
            contact_times[force] = extended_valid_intervals

        file_data[side]["contact_time"] = contact_times
        # Ensure all forces in force_keys share the same contact intervals
        primary_force = "Fz" if "Fz" in contact_times else next(iter(contact_times), None)
        if primary_force:
            shared_times = contact_times[primary_force]
            for force in force_keys:
                if force not in contact_times:
                    contact_times[force] = shared_times
        #print(f"{side} contact times:", {k: v for k, v in contact_times.items()})

def compute_impulses(file_data: Dict, force_keys: List[str]) -> None:
    """
    Berechnet die Impulse für jede Kraft pro Griffseite und speichert sie im file_data-Dictionary.
    """
    for side in ["G1R", "G2L"]:
        impulses = {}
        for force in force_keys:
            ctimes = file_data[side]["contact_time"].get(force, [])
            if force in ["Fx", "Mz"]:
                impulses[force] = compute_impulses_per_contact(
                    file_data[side]["data"], ctimes, force, use_abs=True
                )
            else:
                impulses[force] = compute_impulses_per_contact(
                    file_data[side]["data"], ctimes, force
                )

        # Compute total resultant force impulse over Fz intervals
        res_impulses = []
        df = file_data[side]["data"]
        intervals = file_data[side]["contact_time"].get("Fz", [])

        if "Fres_xyz" in df.columns:
            for t0, t1 in intervals:
                mask = (df["Time [s]"] >= t0) & (df["Time [s]"] <= t1)
                fres = df.loc[mask, "Fres_xyz"]
                time = df.loc[mask, "Time [s]"]
                impulse = np.trapz(fres, x=time)
                res_impulses.append(impulse)

        impulses["F_total"] = res_impulses

        # Store a clean copy under "all" without including "all" itself recursively
        impulses_filtered = {k: v for k, v in impulses.items() if k != "all"}
        impulses["all"] = impulses_filtered
        #print(f"{side} all impulses: {impulses['all']}")
        file_data[side]["impulses"] = impulses



# --- Neue Funktion: Kraftstatistiken pro Kontaktintervall ---
def compute_interval_force_stats(file_data: Dict) -> None:
    """
    Berechnet Min, Max, Mittelwert und Impuls, hausdorff für Fx, Fy, Fz innerhalb jedes Kontaktintervalls.
    Speichert die Ergebnisse unter file_data[side]["intervals"]["I1"], ["I2"], ...
    Vor der Berechnung werden ungültige Intervalle entfernt (basierend auf config.invalid_intervals_list).
    Nach der Berechnung werden die Zeitbereiche der ungültigen Intervalle auch aus dem DataFrame entfernt.
    """
    from utils import compute_impulses_per_contact  # ensure this import is present at top of file
    for side in ["G1R", "G2L"]:
        df = file_data[side]["data"]
        intervals = file_data[side]["contact_time"].get("Fz", [])
        # --- Store segments of invalid intervals ---
        # (new logic: store in file_data["invalid_intervals"][side][...])
        if "invalid_intervals" not in file_data:
            file_data["invalid_intervals"] = {}
        if side not in file_data["invalid_intervals"]:
            file_data["invalid_intervals"][side] = {}
        # Mapping for force to impulse name
        force_map = {
            "Fx": "Px",
            "Fy": "Py",
            "Fz": "Pz",
            "Mz": "PMz",
            "Fres_xyz": "Pxyz",
            "Fres_yz": "Pyz",
            "FgR_sum": "Psum",
            "FgR_1": "Pgr1",
            "FgR_2": "Pgr2"
        }
        interval_stats = {}
        file_number = file_data.get("file_number", "000")
        # Loop over all intervals and store invalid segments before continue
        for i, (t0, t1) in enumerate(intervals):
            if (i + 1) in config.invalid_intervals_list:
                mask = (df["Time [s]"] >= t0) & (df["Time [s]"] <= t1)
                segment = df.loc[mask]
                file_data["invalid_intervals"][side][f"I{i+1}"] = segment.to_dict(orient="list")
                continue
            # For valid intervals, define mask/segment here
            mask = (df["Time [s]"] >= t0) & (df["Time [s]"] <= t1)
            segment = df.loc[mask]
            current_interval = f"I{i+1}"
            stats_entry = {
                "interval_timing": (round(t0, 3), round(t1, 3)),
                "duration_s": round(t1 - t0, 3)
            }
            for force in force_map:
                # Only select columns that have the force 
                if force == "FgR_sum":
                    force_cols = [col for col in df.columns if col.startswith("FgR_sum")]
                else:
                    force_cols = [col for col in df.columns if force in col]
                if force_cols:
                    force_cols = [force_cols[0]]
                    series = segment[force_cols[0]]
                    if force in ["Fx", "Mz"]:
                        impulse = float(np.trapz(np.abs(series), x=segment["Time [s]"]))
                    else:
                        impulse = float(np.trapz(series, x=segment["Time [s]"]))
                    time = segment["Time [s]"].values
                    if len(time) >= 2:
                        slope = np.gradient(series, time)
                        maxROFD = np.max(slope)
                    else:
                        maxROFD = None
                    force_dict = {
                        "min": round(series.min(), 1),
                        "max": round(series.max(), 1),
                        "mean": round(series.mean(), 1),
                        "impuls": round(impulse, 1),
                        "maxROFD": round(maxROFD, 2) if maxROFD is not None else None,
                        "hausdorff": calc_hausdorff_dimension_for_single_signal(
                            time, series, current_side=side, current_force=force, current_interval=current_interval,
                        )
                    }
                    stats_entry[force] = force_dict
                    stats_entry[force_map[force]] = round(impulse, 1)
            interval_stats[f"I{i+1}"] = stats_entry
            stats_entry["interval_data"] = segment.to_dict(orient="list")
        # Mittelwerte über alle Intervalle hinweg berechnen
        mean_metrics = {}
        all_force_data = {}
        for entry in interval_stats.values():
            for force, metrics in entry.items():
                if force in ["interval_timing"]:
                    continue
                if isinstance(metrics, dict) and any(k in metrics for k in ["maxROFD", "hausdorff"]):
                    if force not in all_force_data:
                        all_force_data[force] = {"min": [], "max": [], "mean": [], "impuls": [], "maxROFD": [], "hausdorff": []}
                    for key in ["min", "max", "mean", "impuls", "maxROFD", "hausdorff"]:
                        val = metrics.get(key)
                        if val is not None:
                            all_force_data[force][key].append(val)
        # Ensure FgR_sum is included if present
        # (already handled above; this ensures it is included in stats and mean)
        for force, metric_lists in all_force_data.items():
            mean_entry = {}
            for key, vals in metric_lists.items():
                if vals:
                    mean_entry[key] = round(np.mean(vals), 2)
            mean_metrics[force] = mean_entry
        durations = [entry["duration_s"] for entry in interval_stats.values() if "duration_s" in entry]
        if durations:
            mean_metrics["Contacttime"] = {"mean": round(np.mean(durations), 2)}
        interval_stats["Mean-Metrics"] = mean_metrics
        file_data[side]["intervals"] = interval_stats
        # Remove invalid intervals' time ranges from df
        for i, (t0, t1) in enumerate(intervals):
            if (i + 1) in config.invalid_intervals_list:
                df.drop(df[(df["Time [s]"] >= t0) & (df["Time [s]"] <= t1)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

def calc_FgR(current_dict):
    """
    Fügt jeder G1R- und G2L-Datenstruktur eine neue Spalte 'FgR_calc' hinzu.
    Formel:
    FgR_calc = (cos(40°)*Fy + sin(40°)*Ff) / (73 * 9.81) * 100
    """

    angle_rad = np.deg2rad(40)
    cos_40 = np.cos(angle_rad)
    sin_40 = np.sin(angle_rad)

    for file_data in current_dict.values():
        for side in ["G1R", "G2L"]:
            df = file_data[side]["data"]
            fy_cols = [col for col in df.columns if "Fy" in col]
            ff_cols = [col for col in df.columns if "Fz" in col]  # Ff ≈ Fz

            if fy_cols and ff_cols:
                fy = df[fy_cols[0]]
                fz = df[ff_cols[0]]
                fgr_calc = (cos_40 * fy + sin_40 * fz) / (73 * 9.81) * 100
                df = file_data[side]["data"]
                df.loc[:, "FgR_calc"] = fgr_calc

def calc_resultant_fy_fz(current_dict):
    """
    Fügt den DataFrames 'Fres_xyz','Fres_yz' und 'phiyz' hinzu:
      - 'Fres_xyz' ist die resultierende Kraft aus Fy, Fz und Fx.
      - 'Fres_yz' ist die resultierende Kraft aus Fy und Fz.
      - 'phiyz' ist der Winkel von Fres_yz bezogen auf die Senkrechte (Erdbeschleunigung), korrigiert um 40°.
    """
    for file_data in current_dict.values():
        for side in ["G1R", "G2L"]:
            df = file_data[side]["data"]
            fy_cols = [col for col in df.columns if "Fy" in col]
            fz_cols = [col for col in df.columns if "Fz" in col]
            fx_cols = [col for col in df.columns if "Fx" in col]

            if fy_cols and fz_cols:
                fy = df[fy_cols[0]]
                fz = df[fz_cols[0]]
                Fres_yz = np.sqrt(fy**2 + fz**2)
                if fx_cols:
                    fx = df[fx_cols[0]]
                    Fres_xyz = np.sqrt(fy**2 + fz**2 + fx**2)
                else:
                    Fres_xyz = None
                    print(f"Keine Fx-Spalte in {side} gefunden. Fres_xyz wird nicht berechnet.")
                angle = np.rad2deg(np.arctan2(fz, fy))  # Winkel in Grad
                phiyz = angle - 40  # Bezug zur Senkrechten (Wandwinkel)

                df.loc[:, "Fres_yz"] = Fres_yz
                df.loc[:, "Fres_xyz"] = Fres_xyz
                df.loc[:, "φ_yz"] = phiyz


# --- Datenfilterung & Verarbeitung ---

def apply_filter(df, windowlength, polyorder, mode="interp"):
    df_filtered = df.copy()
    for col in df.columns:
        if col != "Time [s]":
            df_filtered[col] = savgol_filter(df[col], window_length=windowlength, polyorder=polyorder, mode=mode)
    return df_filtered


# --- IO & Export ---

def export_impulse_data(file_data, fname, folder_path):
    date_match = re.search(r"(\d{2})-(\d{2})-(\d{2})", fname)
    if date_match:
        yy, mm, dd = date_match.groups()
        date_str = f"{dd}_{mm}_{yy}_impulses"
    else:
        date_str = "unknown_date_impulses"
    impulse_dir = os.path.join(folder_path, date_str)
    os.makedirs(impulse_dir, exist_ok=True)
    txt_path = os.path.join(impulse_dir, f"{fname}_impulses.txt")
    try:
        with open(txt_path, "w") as f:
            f.write(f"Impulsdaten für Datei '{fname}'\n")
            for side in ["G1R", "G2L"]:
                f.write(f"\n{side}:\n")
                contact_times_side = file_data[side].get("contact_time", {})
                for force_name, ctimes in contact_times_side.items():
                    if ctimes:
                        ctimes_str = ", ".join(f"({t0:.2f}-{t1:.2f}s)" for t0, t1 in ctimes)
                    else:
                        ctimes_str = "keine Kontaktzeiten"
                    f.write(f"  {force_name} Kontaktzeiten: {ctimes_str}\n")
                    impulses_side = file_data[side]["impulses"].get(force_name, {})
                    for comp, vals in impulses_side.items():
                        label = comp.replace("F", "P", 1) if comp.startswith("F") else comp
                        vals_str = ", ".join(f"{v:.1f}" for v in vals)
                        f.write(f"    {label}: [{vals_str}]\n")
        print(f"Impulsdaten gespeichert in: {txt_path}")
    except Exception as e:
        print(f"Fehler beim Schreiben der Impulsdatei '{txt_path}': {e}")



def export_data_to_excel(file_data, fname, folder_path):
    excel_folder = os.path.join(folder_path, "excel")
    os.makedirs(excel_folder, exist_ok=True)
    excel_path = os.path.join(excel_folder, f"{config.file_number}{config.optional_suffix}_Interval_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Metadaten
        meta_df = pd.DataFrame({
            "athletename": [file_data.get("athletename", "")],
            "climberforce": [file_data.get("climberforce", "")],
            "file_identity": [file_data.get("file_identity", "")],
            "level": [file_data.get("level", "")],
            "invalid_intervals": [", ".join(f"I{x}" for x in config.invalid_intervals_list)]
        })
        meta_df.to_excel(writer, sheet_name="Metadata", index=False)

        # Store skipped intervals with placeholder content
        skipped = {}
        for i in config.invalid_intervals_list:
            for side in ["G1R", "G2L"]:
                contact = file_data[side].get("contact_time", {}).get("Fz", [])
                if i - 1 < len(contact):
                    t0, t1 = contact[i - 1]
                    df = file_data[side]["data"]
                    segment = df[(df["Time [s]"] >= t0) & (df["Time [s]"] <= t1)]
                    skipped_key = f"I{i}_{side}"
                    skipped[skipped_key] = segment.to_dict(orient="list")

        # Combine all skipped intervals into one DataFrame and write to a single sheet named "Invalid_Intervals"
        all_skipped_rows = []
        for key, content in skipped.items():
            df = pd.DataFrame(content)
            df.insert(0, "interval_id", key)
            all_skipped_rows.append(df)
        if all_skipped_rows:
            pd.concat(all_skipped_rows).to_excel(writer, sheet_name="Invalid_Intervals", index=False)

        for side in ["G1R", "G2L"]:
            # Kontaktzeiten
            contact = file_data[side].get("contact_time", {})
            if contact:
                contact_df = pd.DataFrame({k: pd.Series([v]) for k, v in contact.items()})
                contact_df.to_excel(writer, sheet_name=f"{side}_Contact", index=False)

            # Impulse
            impulses = file_data[side].get("impulses", {})
            impulses_df = pd.DataFrame({k: pd.Series(v) for k, v in impulses.items() if k != "all"})
            impulses_df.to_excel(writer, sheet_name=f"{side}_Impulses", index=False)

            # Intervalle
            intervals = file_data[side].get("intervals", {})
            if intervals:
                int_rows = []
                for int_key, stats in intervals.items():
                    base_info = {"intervall_id": int_key}
                    if "interval_timing" in stats:
                        base_info["interval_timing"] = stats["interval_timing"]
                    if "duration_s" in stats:
                        base_info["duration_s"] = stats["duration_s"]
                    for force, values in stats.items():
                        if force in ["interval_timing", "duration_s"]:
                            continue
                        row = {"intervall_id": int_key, "force": force}
                        # Ensure hausdorff is included for each metric row
                        if isinstance(values, dict):
                            row.update(values)
                            # If hausdorff is not present, add it as None
                            if "hausdorff" not in row:
                                row["hausdorff"] = None
                        else:
                            row["impuls"] = values  # für einfache Impulse wie Px, Py etc.
                            row["hausdorff"] = None
                        row.update(base_info)
                        int_rows.append(row)
                    # Add a DataFrame-compatible empty row after each interval
                    int_rows.append({col: None for col in ["intervall_id", "interval_timing", "duration_s", "force", "max", "min", "mean", "impuls", "hausdorff"]})

                int_rows = [row for row in int_rows if row]  # Remove empty dicts
                int_df = pd.DataFrame(int_rows)
                cols = ["intervall_id", "interval_timing", "duration_s", "force", "max", "min", "mean", "impuls", "hausdorff"]
                existing_cols = [col for col in cols if col in int_df.columns]
                # Ensure 'hausdorff' is included in export columns
                if "hausdorff" not in existing_cols:
                    existing_cols.append("hausdorff")
                int_df = int_df[existing_cols]
                int_df.to_excel(writer, sheet_name=f"{side}_Intervals", index=False)
                worksheet = writer.sheets[f"{side}_Intervals"]
                from openpyxl.styles import Font
                bold_font = Font(bold=True)
                for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
                    for cell in row:
                        if cell.column_letter in ['D', 'E', 'H']:  # force, max, impuls
                            cell.font = bold_font

    print(f"Excel-Export abgeschlossen: {excel_path}")


# --- Hauptfunktion ---

def process_single_lvm_file(file_path, settings):
    """
    Liest und verarbeitet eine einzelne .lvm-Datei. Gibt Dateiname, file_data und clean_df_trimmed zurück.
    Die Funktion ist in logisch kommentierte Abschnitte gegliedert:
      1. Metadaten einlesen
      2. Rohdatenverarbeitung
      3. Griffseitentrennung und Trimmung
      4. Optional: Filterung
      5. Strukturieren und Normieren
      6. Resultierende berechnen
    """
    if config.deebug_mode:
        print(f"Calculating file: {file_path}")
    # === 1. Settings einlesen ===
    SVGwindowlength = settings.get("SVGwindowlength")
    SVGpolyorder = settings.get("SVGpolyorder")
    usefilter = settings.get("use_filter", False)
    normalizeByweight = settings.get("normalizeByweight", False)
    autotrim = settings.get("autotrim", True)

    df = pd.read_csv(file_path, sep="\t", decimal=",", skiprows=0, header=21)
    df.columns = df.columns.astype(str)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = prepare_time_column(df, autotrim=autotrim)

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    # --- Auto-skip interval 2 for specific files ---
    if file_name.startswith("017-Shoes-3_1_FH-best_Sh-trail_TK-front"):
        if 2 not in config.invalid_intervals_list:
            config.invalid_intervals_list.append(2)
    metadata = parse_metadata_from_filename(file_name)
    athlete_name = metadata["athlete"]
    kgclimber = metadata["weight"]
    climberforce = kgclimber * 9.81
    file_identity = metadata["identity"]
    
    # Append excluded intervals from config if key is contained in file_name
    for key in config.excluded_intervals_dict:
        if key in file_name:
            excluded = config.excluded_intervals_dict[key]
            config.invalid_intervals_list.extend([i for i in excluded if i not in config.invalid_intervals_list])

    # === 2. Rohdatenverarbeitung ===
    clean_df = clean_data(df) # remove X_Value, comments and U_data
    clean_df = trim_low_force_periods(clean_df, threshold=10, min_duration=3, buffer=2)
    clean_df = calc_fgr_sum(clean_df)

    # === 3. Griffseitentrennung und Trimmung ===
    g1r, g2l = split_grip_sides(clean_df)
    global_start, global_end = get_trim_range_from_contact_times(g1r, g2l, autotrim)
    g1r = g1r[(g1r["Time [s]"] >= global_start) & (g1r["Time [s]"] <= global_end)].reset_index(drop=True)
    g2l = g2l[(g2l["Time [s]"] >= global_start) & (g2l["Time [s]"] <= global_end)].reset_index(drop=True)
    min_len = min(len(g1r), len(g2l))
    g1r = g1r.iloc[:min_len].reset_index(drop=True)
    g2l = g2l.iloc[:min_len].reset_index(drop=True)

    # === 4. Optional: Filterung ===
    if usefilter:
        g1r_filtered = apply_filter(g1r, SVGwindowlength, SVGpolyorder, mode='interp')
        g2l_filtered = apply_filter(g2l, SVGwindowlength, SVGpolyorder, mode='interp')
        clean_df = apply_filter(clean_df, SVGwindowlength, SVGpolyorder, mode='interp')
        file_data = {
            "G1R": {"data": g1r_filtered, "stats": get_min_max_values_per_column(g1r_filtered)},
            "G2L": {"data": g2l_filtered, "stats": get_min_max_values_per_column(g2l_filtered)},
        }
    else:
        file_data = {
            "G1R": {"data": g1r, "stats": get_min_max_values_per_column(g1r)},
            "G2L": {"data": g2l, "stats": get_min_max_values_per_column(g2l)},
        }
    clean_df_trimmed = clean_df[(clean_df["Time [s]"] >= global_start) & (clean_df["Time [s]"] <= global_end)].reset_index(drop=True)

    # === 5. Strukturieren und Normieren ===
    # Metadaten speichern
    file_data["climberforce"] = climberforce
    file_data["athletename"] = athlete_name
    file_data["file_identity"] = file_identity
    file_data["level"] = metadata.get("level", "")
   # file_data["file_number"] = file_identity[:3]
    file_data["total_df"] = clean_df_trimmed
    

    # Normierung und weitere Berechnungen pro Griffseite
    for side in ["G1R", "G2L"]:
        side_df = file_data[side]["data"]
        if normalizeByweight and climberforce is not None and isinstance(climberforce, (int, float)):
            normalize_forces_by_weight(side_df, climberforce)
        file_data[side]["data"] = side_df

    # === 6. Resultierende berechnen ===
    # Berechne Fres_yz und φ_yz für beide Seiten
    temp_dict = {file_name: file_data}
    calc_resultant_fy_fz(temp_dict)
    file_data = temp_dict[file_name]

    return file_name, file_data, clean_df_trimmed


def finalize_file_export(file_data, fname, folder_path, save_plot):
    """
    Führt die Export- und Berechnungslogik pro Datei durch.
    """
    force_keys = ["Fy", "Fx", "Fz", "Mz", "Fres_xyz", "Fres_yz"]
    if config.deebug_mode:
        print(f"in finalize_file_export_config-file_number pre setting: {config.file_number}")
    config.file_number = fname[:3]  # Set file_number from filename
    if config.deebug_mode:
        print(f"in finalize_file_export_config-file_number AFTER setting: {config.file_number}")
    compute_contact_times(file_data, force_keys)
    compute_interval_force_stats(file_data)
    if config.create_hausdorff_plots:
        plot_hausdorff_intervals(file_data, folder_path, fname)
    compute_impulses(file_data, force_keys)
    if config.plot_settings["export_data"]:
        export_data_to_excel(file_data, fname, folder_path)
    # if save_plot:
    #     export_impulse_data(file_data, fname, folder_path)


def load_lvm_data(folder_path, *, settings, export=False) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Lädt .lvm-Dateien aus dem angegebenen Verzeichnis und bereitet sie für die spätere Analyse auf.
    """
    save_plot = settings.get("save_plot", False)
    data_dict = {}

    file_paths = sorted([fp for fp in glob.glob(os.path.join(folder_path, "*.lvm")) if "MAX" not in os.path.basename(fp)])
    for file_path in file_paths:
        print("loading: ",file_path)
        
        file_name, file_data, clean_df_trimmed = process_single_lvm_file(file_path, settings)
      #  config.file_number = file_name[:3]  # Set file_number from filename
        print("in load_lvm_data_config.file_number = ",config.file_number)
        data_dict[file_name] = file_data

    # Export und finale Berechnungen pro Datei
    for fname, file_data in data_dict.items():
         finalize_file_export(file_data, fname, folder_path, save_plot)

    # Kopie mit zusätzlichem 'data'-Eintrag
    exportable_dict = {}
    for fname, content in data_dict.items():
        exportable_dict[fname] = {}
        for key, val in content.items():
            if key in ["G1R", "G2L"]:
                exportable_dict[fname][key] = {
                    **{k: v for k, v in val.items() if k != "data"},
                    "data": val["data"].to_dict(orient="list")
                }
            elif key != "total_df":
                exportable_dict[fname][key] = val
        # Adjust logic for invalid_intervals: copy if present in content
        if "invalid_intervals" in content:
            exportable_dict[fname]["invalid_intervals"] = content["invalid_intervals"]
    if export:
        filename = "_".join(config.processed_files_list) + config.optional_suffix+"_summary.json"
        output_path = os.path.join(folder_path, filename)
        try:
            with open(output_path, "w") as f:
                json.dump(exportable_dict, f, indent=2)
            print(f"Zusammenfassung gespeichert unter: {output_path}")
        except Exception as e:
            print(f"Fehler beim Speichern der JSON-Datei: {e}")
        print("\n++++++++++++ files in JSON ++++++++++++++++++++++")
        for filename in data_dict:
            print(f"\n--- {filename} ---")
        # print(list(data_dict[filename].keys()))

    return data_dict if data_dict else None


# --- Neue Funktion zur Normierung ---
def normalize_forces_by_weight(df, climberforce):
    """
    Skaliert die Kräfte Fy, Fz, Fx, Mz in Prozent des Körpergewichts.
    """
# Exportiere Original und Normalisiert nebeneinander in eine Excel-Datei
    try:
        original_df = df.copy()
        for force_type in ["Fy", "Fz", "Fx", "Mz"]:
            force_cols = [col for col in df.columns if force_type in col]
            for col in force_cols:
                df[col] = (df[col] / climberforce) * 100
                new_col = col.replace("[N]", "[%]")
                df.rename(columns={col: new_col}, inplace=True)
       # export_path = "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/pipeline/original_vs_normalized_debug.xlsx"
       # with pd.ExcelWriter(export_path) as writer:
        #    original_df.to_excel(writer, sheet_name="Original", index=False)
         #   df.to_excel(writer, sheet_name="Normalized", index=False)
        #print(f"[normalize_forces_by_weight] Debug-Export erfolgreich: {export_path}")
    except Exception as e:
        print(f"[normalize_forces_by_weight] Fehler beim Export: {e}")


    
    

# --- Neue Funktion: Summenspalte FgR_sum [%] ---
def calc_fgr_sum(df):
    """
    Fügt dem übergebenen DataFrame eine neue Spalte 'FgR_sum [%]' hinzu.
    Diese Spalte enthält die Zeilensumme aus 'FgR_1 [%]' und 'FgR_2 [%]', sofern beide vorhanden sind.
    """
    fgr1_col = next((col for col in df.columns if "FgR_1" in col), None)
    fgr2_col = next((col for col in df.columns if "FgR_2" in col), None)

    if fgr1_col and fgr2_col:
        df["FgR_sum [%]"] = df[fgr1_col] + df[fgr2_col]
    return df


