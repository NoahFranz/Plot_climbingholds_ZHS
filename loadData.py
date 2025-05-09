
def prepare_time_column(df, autotrim=True):
    df["Time [s]"] = correct_time_jumps(df["Time [s]"])
    if autotrim:
        df["Time [s]"] -= df["Time [s]"].iloc[0]
    return df

import pandas as pd
import os
import glob
from utils import clean_data, get_min_max_values_per_column
from utils import get_force_contact_times, compute_impulses_per_contact, trim_low_force_periods
from scipy.signal import savgol_filter
import numpy as np
import re



def split_grip_sides(df):
    g1r = df[["Time [s]"] + [col for col in df.columns if "1" in col]]
    g2l = df[["Time [s]"] + [col for col in df.columns if "2" in col]]
    return g1r, g2l

def load_lvm_data(folder_path, *, settings):
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
  - 'Fres': Resultierende aus Fy und Fz
  - 'φ_yz': Winkel zwischen Fres und der Senkrechten (korrigiert um 40°)

Rückgabe:
  dict[str, dict[str, dict[str, Any]]]
"""
    SVGwindowlength = settings.get("SVGwindowlength")
    SVGpolyorder = settings.get("SVGpolyorder")
    usefilter = settings.get("usefilter", False)
    normalizeByweight = settings.get("normalizeByweight", False)
    save_plot = settings.get("save_plot", False)
    autotrim = settings.get("autotrim", True)

    data_dict = {}

    for file_path in [fp for fp in glob.glob(os.path.join(folder_path, "*.lvm")) if "MAX" not in os.path.basename(fp)]:
        print(file_path)
        df = pd.read_csv(file_path, sep="\t", decimal=",", skiprows=0, header=21)
        df.columns = df.columns.astype(str)
        df = df.apply(pd.to_numeric, errors='coerce')

        df = prepare_time_column(df, autotrim=autotrim)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        metadata = parse_metadata_from_filename(file_name)
        athlete_name = metadata["athlete"]
        kgclimber = metadata["weight"]
        climberforce = kgclimber * 9.81
        file_identity = metadata["identity"]
        #remove U and "comment" colums
        clean_df = clean_data(df)
        # print(clean_df)
        clean_df = trim_low_force_periods(clean_df, threshold=10, min_duration=3, buffer=2)

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
        # Gewicht als Integer im Dict speichern
        data_dict[file_name]["climberforce"] = climberforce

        # save athlete name in dict
        data_dict[file_name]["athletename"] = athlete_name

        # save file identity suffix in dict
        data_dict[file_name]["file_identity"] = file_identity

        # Falls gewünscht: Normiere alle Kräfte (Fy, Fz, Fx) auf das Kletterergewicht (climberforce)
        if normalizeByweight:
            if climberforce is not None and isinstance(climberforce, (int, float)):
                for side in ["G1R", "G2L"]:
                    df = data_dict[file_name][side]["data"]
                    for force_type in ["Fy", "Fz", "Fx", "Mz"]:
                        force_cols = [col for col in df.columns if force_type in col]
                        for col in force_cols:
                            df[col] = df[col] / climberforce
                            df[col] = df[col] * 100
            else:
                print(f"⚠️ Kein gültiges Gewicht für Datei '{file_name}', Normierung wird übersprungen.")

        # calc_FgR(data_dict)
        calc_resultant_fy_fz(data_dict)

    # Berechne und speichere die Aktivitäts-Kontaktzeiten für jede Kraft pro Griff
    force_keys = ["Fy", "Fx", "Fz", "Mz"]
    for fname, file_data in data_dict.items():
        contact_times_g1 = get_force_contact_times(file_data["G1R"]["data"], force_keys)
      #  print(f"[Kontaktzeiten] Datei '{fname}' G1R:")
        for force, ctimes in contact_times_g1.items():
            if not ctimes:
                print(f"  {force}: keine Kontaktzeiten gefunden")
            else:
                continue
                for idx, (t0, t1) in enumerate(ctimes):
                    
                    print(f"  {force} [{idx}]: Start = {t0:.2f}s, Ende = {t1:.2f}s")
        file_data["G1R"]["contact_time"] = contact_times_g1
        stats_g1 = compute_contact_time_stats_per_force(contact_times_g1)
        file_data["G1R"]["contact_time_stats"] = stats_g1

        contact_times_g2 = get_force_contact_times(file_data["G2L"]["data"], force_keys)
     #   print(f"[Kontaktzeiten] Datei '{fname}' G2L:")
        for force, ctimes in contact_times_g2.items():
            if not ctimes:
                print(f"  {force}: keine Kontaktzeiten gefunden")
            else:
                for idx, (t0, t1) in enumerate(ctimes):
                   # print(f"  {force} [{idx}]: Start = {t0:.2f}s, Ende = {t1:.2f}s")
                   continue
        file_data["G2L"]["contact_time"] = contact_times_g2
        stats_g2 = compute_contact_time_stats_per_force(contact_times_g2)
        file_data["G2L"]["contact_time_stats"] = stats_g2

        # Impuls-Berechnung pro Griff für jede erkannte Kontaktzeit
        # G1R: compute impulses per contact_time per force
        impulses_dict_g1 = {}
        for force in force_keys:
            ctimes = contact_times_g1.get(force, [])
            impulses_dict_g1[force] = compute_impulses_per_contact(
                file_data["G1R"]["data"], ctimes, force
            )
        file_data["G1R"]["impulses"] = impulses_dict_g1

        # G2L: compute impulses per contact_time per force
        impulses_dict_g2 = {}
        for force in force_keys:
            ctimes = contact_times_g2.get(force, [])
            impulses_dict_g2[force] = compute_impulses_per_contact(
                file_data["G2L"]["data"], ctimes, force
            )
        file_data["G2L"]["impulses"] = impulses_dict_g2

        if save_plot:
            # --- Export impulse data to text file ---
            # Extrahiere Datum aus dem Dateinamen (Format: YY-MM-DD)
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
                        # Kontaktzeiten ausgeben
                        contact_times_side = file_data[side].get("contact_time", {})
                        for force_name, ctimes in contact_times_side.items():
                            if ctimes:
                                ctimes_str = ", ".join(f"({t0:.2f}-{t1:.2f}s)" for t0, t1 in ctimes)
                            else:
                                ctimes_str = "keine Kontaktzeiten"
                            f.write(f"  {force_name} Kontaktzeiten: {ctimes_str}\n")
                            # Impulse pro Komponente ausgeben
                            impulses_side = file_data[side]["impulses"].get(force_name, {})
                            for comp, vals in impulses_side.items():
                                vals_str = ", ".join(f"{v:.1f}" for v in vals)
                                f.write(f"    {comp}: [{vals_str}]\n")
                print(f"Impulsdaten gespeichert in: {txt_path}")
            except Exception as e:
                print(f"Fehler beim Schreiben der Impulsdatei '{txt_path}': {e}")

    return data_dict if data_dict else None

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
    Fügt den DataFrames 'Fres' und 'phiyz' hinzu:
      - 'Fres' ist die resultierende Kraft aus Fy und Fz.
      - 'phiyz' ist der Winkel von Fres bezogen auf die Senkrechte (Erdbeschleunigung), korrigiert um 40°.
    """
    for file_data in current_dict.values():
        for side in ["G1R", "G2L"]:
            df = file_data[side]["data"]
            fy_cols = [col for col in df.columns if "Fy" in col]
            fz_cols = [col for col in df.columns if "Fz" in col]

            if fy_cols and fz_cols:
                fy = df[fy_cols[0]]
                fz = df[fz_cols[0]]
                fres = np.sqrt(fy**2 + fz**2)
                angle = np.rad2deg(np.arctan2(fz, fy))  # Winkel in Grad
                phiyz = angle - 40  # Bezug zur Senkrechten (Wandwinkel)

                df.loc[:, "Fres"] = fres
                df.loc[:, "φ_yz"] = phiyz


def apply_filter(df, windowlength, polyorder, mode="interp"):
    df_filtered = df.copy()
    for col in df.columns:
        if col != "Time [s]":
            df_filtered[col] = savgol_filter(df[col], window_length=windowlength, polyorder=polyorder, mode=mode)
    return df_filtered

def compute_contact_time_stats_per_force(contact_time_dict):
    """
    Erwartet contact_time_dict = {'Fz': [(t0, t1), ...], ...}
    Gibt ein Dict zurück: {'Fz': {'min':..., 'max':..., 'mean':...}, ...}
    """
    stats = {}
    for force, intervals in contact_time_dict.items():
        if not intervals:
            stats[force] = {'min': 0, 'max': 0, 'mean': 0}
            continue
        durations = [round(t1-t0, 1) for t0, t1 in intervals if t1 > t0]
        if durations:
            stats[force] = {
                'min': round(min(durations), 1),
                'max': round(max(durations), 1),
                'mean': round(sum(durations) / len(durations), 1)
            }
        else:
            stats[force] = {'min': 0, 'max': 0, 'mean': 0}
    return stats


        # --- Korrigiere große Zeitsprünge in der Time-Spalte ---
def correct_time_jumps(time_series, threshold=2.0):
    corrected = time_series.copy()
    offset = 0.0
    for i in range(1, len(time_series)):
        dt = time_series.iloc[i] - time_series.iloc[i - 1]
        if dt > threshold:
            sprung = dt
            corrected.iloc[i:] = corrected.iloc[i:] - sprung
    return corrected

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