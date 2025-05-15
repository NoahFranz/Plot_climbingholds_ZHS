from typing import List, Dict, Any
import glob
import os
import re
import json

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from utils import clean_data, get_min_max_values_per_column
from utils import get_force_contact_times, compute_impulses_per_contact, trim_low_force_periods


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
        "identity": "Unknown"
    }
    if match := re.search(r"_(\d+)kg", file_name):
        result["weight"] = int(match.group(1))
    if match := re.search(r"(?<=_)([^_]+)(?=_lvl-)", file_name):
        result["athlete"] = match.group(1)
    if match := re.search(r"^(.+?)_\d+kg", file_name):
        result["identity"] = match.group(1)
    return result

def get_flank_time_range(df, schwellwert=10, offset_sec=4, autotrim=True):
    """
    Gibt das globale Zeitintervall zurück, in dem gültige Daten liegen,
    basierend auf Datenflankenerkennung in allen Fy-Spalten.
    Berücksichtigt Flanken über alle Spalten hinweg.
    Wenn autotrim=False, werden der erste und letzte Index des DataFrames verwendet.
    """
    if not autotrim:
        # Wenn autotrim deaktiviert ist, verwende den gesamten Bereich des DataFrames
        t_start = max(0, df["Time [s]"].iloc[0])
        t_end = df["Time [s]"].iloc[-1]
        return (t_start, t_end)

    fy_cols = [col for col in df.columns if "Fy" in col]
    if not fy_cols:
        return (df["Time [s]"].iloc[0], df["Time [s]"].iloc[-1])
    
    global_start, global_end = None, None

    for col in fy_cols:
        diffs = np.abs(df[col].diff())
        diffs = diffs.rolling(window=5, min_periods=1).mean()
        # Debug: max diff vs. threshold in get_flank_time_range
        max_diff = diffs.max()
        print(f"[get_flank_time_range] Spalte '{col}': max diff = {max_diff:.3f}, schwellwert = {schwellwert}")
        start_idx = diffs[diffs > schwellwert].first_valid_index()
        end_idx = diffs[diffs > schwellwert].last_valid_index()
        
        if start_idx is not None:
            curr_start_time = df.loc[start_idx, "Time [s]"]
            global_start = curr_start_time if global_start is None else min(global_start, curr_start_time)
            print(f"[get_flank_time_range] Spalte '{col}': global_start = {global_start:.3f}")
        
        if end_idx is not None:
            curr_end_time = df.loc[end_idx, "Time [s]"]
            global_end = curr_end_time if global_end is None else max(global_end, curr_end_time)
            print(f"[get_flank_time_range] Spalte '{col}': global_end = {global_end:.3f}")

    # Prüfen, ob keine Flanken gefunden wurden
    if global_start is None or global_end is None:
        return (df["Time [s]"].iloc[0], df["Time [s]"].iloc[-1])
    
    # Offset hinzufügen
    t_start = max(0, global_start - offset_sec)
    t_end = global_end + offset_sec
    print("[t_start, t_end] = ", (t_start, t_end))
    return (t_start, t_end)

def split_grip_sides(df):
    g1r = df[["Time [s]"] + [col for col in df.columns if "1" in col]]
    g2l = df[["Time [s]"] + [col for col in df.columns if "2" in col]]
    return g1r, g2l


# --- Datenberechnung ---

def compute_contact_times(file_data: Dict, force_keys: List[str]) -> None:
    for side in ["G1R", "G2L"]:
        contact_times = get_force_contact_times(file_data[side]["data"], force_keys)
        
        file_data[side]["contact_time"] = contact_times
        # Ensure all forces in force_keys share the same contact intervals
        primary_force = "Fz" if "Fz" in contact_times else next(iter(contact_times), None)
        if primary_force:
            shared_times = contact_times[primary_force]
            for force in force_keys:
                if force not in contact_times:
                    contact_times[force] = shared_times
        print(f"{side} contact times:", {k: v for k, v in contact_times.items()})

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
        print(f"{side} all impulses: {impulses['all']}")
        file_data[side]["impulses"] = impulses



# --- Neue Funktion: Kraftstatistiken pro Kontaktintervall ---
def compute_interval_force_stats(file_data: Dict) -> None:
    """
    Berechnet Min, Max, Mittelwert und Impuls für Fx, Fy, Fz innerhalb jedes Kontaktintervalls.
    Speichert die Ergebnisse unter file_data[side]["intervals"]["I1"], ["I2"], ...
    """
    from utils import compute_impulses_per_contact  # ensure this import is present at top of file
    for side in ["G1R", "G2L"]:
        df = file_data[side]["data"]
        intervals = file_data[side]["contact_time"].get("Fz", [])
        interval_stats = {}

        # Mapping for force to impulse name
        force_map = {
            "Fx": "Px",
            "Fy": "Py",
            "Fz": "Pz",
            "Mz": "PMz",
            "Fres_xyz": "Pxyz",
            "Fres_yz": "Pyz"
        }

        for i, (t0, t1) in enumerate(intervals):
            mask = (df["Time [s]"] >= t0) & (df["Time [s]"] <= t1)
            segment = df.loc[mask]

            stats_entry = {
                "interval_timing": (round(t0, 3), round(t1, 3)),
                "duration_s": round(t1 - t0, 3)
            }

            for force in force_map:
                force_cols = [col for col in df.columns if force in col]
                if force_cols:
                    series = segment[force_cols].sum(axis=1)
                    impulse = float(np.trapz(np.abs(series) if force in ["Fx", "Mz"] else series, x=segment["Time [s]"]))
                    stats_entry[force] = {
                        "min": round(series.min(), 1),
                        "max": round(series.max(), 1),
                        "mean": round(series.mean(), 1),
                        "impuls": round(impulse, 1)
                    }
                    # Add impulse under new name as well
                    stats_entry[force_map[force]] = round(impulse, 1)

            interval_stats[f"I{i+1}"] = stats_entry

        # Mittelwerte über alle Intervalle hinweg berechnen
        mean_metrics = {}
        all_force_data = {}

        # Only include forces, not the extra impulse keys
        for entry in interval_stats.values():
            for force, metrics in entry.items():
                if force in ["interval_timing"]:
                    continue
                # Only aggregate for force stats, not the separate impulse keys
                if isinstance(metrics, dict):
                    if force not in all_force_data:
                        all_force_data[force] = {"min": [], "max": [], "mean": [], "impuls": []}
                    for key in ["min", "max", "mean", "impuls"]:
                        val = metrics.get(key)
                        if val is not None:
                            all_force_data[force][key].append(val)

        for force, metric_lists in all_force_data.items():
            mean_metrics[force] = {
                key: round(np.mean(vals), 2) if vals else None
                for key, vals in metric_lists.items()
            }

        interval_stats["Mean-Metrics"] = mean_metrics
        file_data[side]["intervals"] = interval_stats

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


# --- Neue Funktion: Excel-Export ---
def export_data_to_excel(file_data, fname, folder_path):
    excel_folder = os.path.join(folder_path, "excel")
    os.makedirs(excel_folder, exist_ok=True)
    excel_path = os.path.join(excel_folder, f"{fname}_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Metadaten
        meta_df = pd.DataFrame({
            "athletename": [file_data.get("athletename", "")],
            "climberforce": [file_data.get("climberforce", "")],
            "file_identity": [file_data.get("file_identity", "")]
        })
        meta_df.to_excel(writer, sheet_name="Metadata", index=False)

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
                        if isinstance(values, dict):
                            row.update(values)
                        else:
                            row["impuls"] = values  # für einfache Impulse wie Px, Py etc.
                        row.update(base_info)
                        int_rows.append(row)
                    # Add a DataFrame-compatible empty row after each interval
                    int_rows.append({col: None for col in ["intervall_id", "interval_timing", "duration_s", "force", "max", "min", "mean", "impuls"]})

                int_rows = [row for row in int_rows if row]  # Remove empty dicts
                int_df = pd.DataFrame(int_rows)
                cols = ["intervall_id", "interval_timing", "duration_s", "force", "max", "min", "mean", "impuls"]
                existing_cols = [col for col in cols if col in int_df.columns]
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

def load_lvm_data(folder_path, *, settings) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
Lädt .lvm-Dateien aus dem angegebenen Verzeichnis und bereitet sie für die spätere Analyse auf.

Abhängig vom Parameter 'usefilter' wird entweder das gefilterte oder das ungefilterte Dictionary erzeugt.
Die Filterung erfolgt mit dem Savitzky-Golay-Filter (Fensterlänge, Polynomgrad einstellbar).

Für jede .lvm-Datei wird ein Eintrag im Rückgabe-Dictionary erzeugt.
Struktur des Rückgabewerts:
  {
      "Dateiname": {
          "G1R": {
              "data": DataFrame der rechten Seite (Spalten mit '1'),
              "stats": Dictionary mit min/max Werten pro Spalte
          },
          "G2L": {
              "data": DataFrame der linken Seite (Spalten mit '2'),
              "stats": Dictionary mit min/max Werten pro Spalte
          }
      },
      ...
  }

Zusätzlich werden folgende Spalten ergänzt:
  - 'FgR_calc': Berechnete relative Griffkraft
  - 'Fres_yz': Resultierende aus Fy und Fz
  - 'φ_yz': Winkel zwischen Fres_yz und der Senkrechten (korrigiert um 40°)

Rückgabe:
  dict[str, dict[str, dict[str, Any]]]
"""
    SVGwindowlength = settings.get("SVGwindowlength")
    SVGpolyorder = settings.get("SVGpolyorder")
    usefilter = settings.get("use_filter", False)
    normalizeByweight = settings.get("normalizeByweight", False)
    save_plot = settings.get("save_plot", False)
    autotrim = settings.get("autotrim", True)

    data_dict = {}

    for file_path in [fp for fp in glob.glob(os.path.join(folder_path, "*.lvm")) if "MAX" not in os.path.basename(fp)]:
        print(file_path)

        # Einlesen und Grunddatenaufbereitung
        df = pd.read_csv(file_path, sep="\t", decimal=",", skiprows=0, header=21)
        df.columns = df.columns.astype(str)
        df = df.apply(pd.to_numeric, errors='coerce')

        df = prepare_time_column(df, autotrim=autotrim)

        # Metadaten extrahieren
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        metadata = parse_metadata_from_filename(file_name)
        athlete_name = metadata["athlete"]
        kgclimber = metadata["weight"]
        climberforce = kgclimber * 9.81
        file_identity = metadata["identity"]

        # Datenbereinigung und Filterung
        clean_df = clean_data(df)
        clean_df = trim_low_force_periods(clean_df, threshold=10, min_duration=3, buffer=2)

        # Aufteilung in Griffseiten
        g1r, g2l = split_grip_sides(clean_df)

        # Gemeinsames Zeitintervall für beide Griffe bestimmen
        start_g1r, end_g1r = get_flank_time_range(g1r)
        print("start_g1r, end_g1r", start_g1r, end_g1r)
        start_g2l, end_g2l = get_flank_time_range(g2l)
        print("start_g2l, end_g2l", start_g2l, end_g2l)
        global_start = min(start_g1r, start_g2l)
        print("global_start", global_start)
        global_end = max(end_g1r, end_g2l)
        print("global_end", global_end)

        # Trimme die Datenframes auf den gemeinsamen Zeitbereich
        g1r = g1r[(g1r["Time [s]"] >= global_start) & (g1r["Time [s]"] <= global_end)].reset_index(drop=True)
        g2l = g2l[(g2l["Time [s]"] >= global_start) & (g2l["Time [s]"] <= global_end)].reset_index(drop=True)

        # Angleichung der Längen beider Seiten nach dem Trimmen
        min_len = min(len(g1r), len(g2l))
        g1r = g1r.iloc[:min_len].reset_index(drop=True)
        g2l = g2l.iloc[:min_len].reset_index(drop=True)

        # Filterung falls gewünscht
        if usefilter:
            g1r_filtered = apply_filter(g1r, SVGwindowlength, SVGpolyorder, mode='interp')
            g2l_filtered = apply_filter(g2l, SVGwindowlength, SVGpolyorder, mode='interp')
            data_dict[file_name] = {
                "G1R": {"data": g1r_filtered, "stats": get_min_max_values_per_column(g1r_filtered)},
                "G2L": {"data": g2l_filtered, "stats": get_min_max_values_per_column(g2l_filtered)},
            }
        else:
            data_dict[file_name] = {
                "G1R": {"data": g1r, "stats": get_min_max_values_per_column(g1r)},
                "G2L": {"data": g2l, "stats": get_min_max_values_per_column(g2l)},
            }

        # Metadaten speichern
        data_dict[file_name]["climberforce"] = climberforce
        data_dict[file_name]["athletename"] = athlete_name
        data_dict[file_name]["file_identity"] = file_identity

        # Normierung und weitere Berechnungen pro Griffseite
        for side in ["G1R", "G2L"]:
            df = data_dict[file_name][side]["data"]

            # Normierung auf Körpergewicht
            if normalizeByweight and climberforce is not None and isinstance(climberforce, (int, float)):
                normalize_forces_by_weight(df, climberforce)

        # Berechne Fres_yz und φ_yz für beide Seiten
        calc_resultant_fy_fz(data_dict)

    # Berechne und speichere die Aktivitäts-Kontaktzeiten und Impulse für jede Kraft pro Griff
    force_keys = ["Fy", "Fx", "Fz", "Mz", "Fres_xyz", "Fres_yz"]
    for fname, file_data in data_dict.items():
        compute_contact_times(file_data, force_keys)
        compute_interval_force_stats(file_data)
        compute_impulses(file_data, force_keys)
        export_data_to_excel(file_data, fname, folder_path)
      #  if save_plot:
          #  export_impulse_data(file_data, fname, folder_path)
           
    

    # Kopie ohne "data"-Einträge
    exportable_dict = {}
    for fname, content in data_dict.items():
        exportable_dict[fname] = {}
        for key, val in content.items():
            if key in ["G1R", "G2L"]:
                exportable_dict[fname][key] = {k: v for k, v in val.items() if k != "data"}
            else:
                exportable_dict[fname][key] = val

    # Speicherpfad definieren
    output_path = os.path.join(folder_path, "summary_metadata.json")
    try:
        with open(output_path, "w") as f:
            json.dump(exportable_dict, f, indent=2)
        print(f"Zusammenfassung gespeichert unter: {output_path}")
    except Exception as e:
        print(f"Fehler beim Speichern der JSON-Datei: {e}")

    return data_dict if data_dict else None


# --- Neue Funktion zur Normierung ---
def normalize_forces_by_weight(df, climberforce):
    """
    Skaliert die Kräfte Fy, Fz, Fx, Mz in Prozent des Körpergewichts.
    """
    for force_type in ["Fy", "Fz", "Fx", "Mz"]:
        force_cols = [col for col in df.columns if force_type in col]
        for col in force_cols:
            df[col] = df[col] / climberforce * 100
