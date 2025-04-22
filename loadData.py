import pandas as pd
import os
import glob
from utils import clean_data, get_min_max_values_per_column
from scipy.signal import savgol_filter
import numpy as np


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
def load_lvm_data(folder_path, SVGwindowlength, SVGpolyorder, usefilter):
    data_dict = {}
    filtered_data_dict = {}

    for file_path in [fp for fp in glob.glob(os.path.join(folder_path, "*.lvm")) if "MAX" not in os.path.basename(fp)]:
        print(file_path)
        df = pd.read_csv(file_path, sep="\t", decimal=",", skiprows=0, header=21)
        df.columns = df.columns.astype(str)
        df = df.apply(pd.to_numeric, errors='coerce')
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        clean_df = clean_data(df)
        g1r = clean_df[["Time [s]"] + [col for col in clean_df.columns if "1" in col]]
        g2l = clean_df[["Time [s]"] + [col for col in clean_df.columns if "2" in col]]
        # Trimme Daten an der Datenflanke
        g1r = trim_by_dataflanke(g1r)
        g2l = trim_by_dataflanke(g2l)
        
        def apply_filter(df):
            df_filtered = df.copy()
            for col in df.columns:
                if col != "Time [s]":
                    df_filtered[col] = savgol_filter(df[col], window_length=SVGwindowlength, polyorder=SVGpolyorder, mode="interp")
            return df_filtered

        if usefilter:
            g1r_filtered = apply_filter(g1r)
            g2l_filtered = apply_filter(g2l)
            data_dict[file_name] = {
                "G1R": {"data": g1r_filtered, "stats": get_min_max_values_per_column(g1r_filtered)},
                "G2L": {"data": g2l_filtered, "stats": get_min_max_values_per_column(g2l_filtered)},
            }
        else:
            data_dict[file_name] = {
                "G1R": {"data": g1r, "stats": get_min_max_values_per_column(g1r)},
                "G2L": {"data": g2l, "stats": get_min_max_values_per_column(g2l)},
            }

    # Nur das tatsächlich genutzte Dictionary berechnen/vervollständigen
    
        calc_FgR(data_dict)
        calc_resultant_fy_fz(data_dict)
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


# --- Hilfsfunktion: trim_by_dataflanke ---
def trim_by_dataflanke(df, schwellwert=10, offset_sec=3):
    """
    Schneidet alle Daten vor der ersten und nach der letzten erkannten Datenflanke ab.
    Die Datenflanke ist definiert als ein signifikanter Anstieg z.B. in Fy.
    Es bleiben nur Daten im Intervall (t_start - offset_sec) bis (t_ende + offset_sec) erhalten.
    Falls mehrere Fy-Spalten vorhanden sind, wird der früheste Startzeitpunkt und späteste Endzeitpunkt berücksichtigt.

    Parameter:
        df : DataFrame mit 'Time [s]' und mindestens einer Kraftspalte
        schwellwert : float – minimale Änderung, um eine Flanke zu erkennen
        offset_sec : float – Zeit in Sekunden als Puffer vor und nach den Flanken

    Rückgabe:
        getrimmter DataFrame
    """
    fy_cols = [col for col in df.columns if "Fy" in col]
    if not fy_cols:
        return df  # keine Fy-Spalte vorhanden

    flank_starts = []
    flank_ends = []

    for col in fy_cols:
        diffs = np.abs(df[col].diff())
        diffs = diffs.rolling(window=5, min_periods=1).mean()

        start_idx = diffs[diffs > schwellwert].first_valid_index()
        end_idx = diffs[diffs > schwellwert].last_valid_index()

        if start_idx is not None:
            flank_starts.append(df.loc[start_idx, "Time [s]"])
        if end_idx is not None:
            flank_ends.append(df.loc[end_idx, "Time [s]"])

    if not flank_starts or not flank_ends:
        return df  # keine gültige Flanke gefunden

    t_start = max(0, min(flank_starts) - offset_sec)
    t_end = max(flank_ends) + offset_sec

    return df[(df["Time [s]"] >= t_start) & (df["Time [s]"] <= t_end)].reset_index(drop=True)