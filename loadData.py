import pandas as pd
import os
import glob
from utils import clean_data, get_min_max_values_per_column
from scipy.signal import savgol_filter

def load_lvm_data(folder_path, SVGwindowlength, SVGpolyorder):
    """
    Lädt alle .lvm-Dateien im angegebenen Ordner.

    Für jede Datei wird ein Eintrag im Rückgabe-Dictionary erstellt.
    Der Schlüssel ist der Teil des Dateinamens vor dem ersten Unterstrich.

    Jeder Eintrag enthält:
      - "G2L": nur Spalten mit "2" im Namen (inkl. "Time [s]") → linker Griff
      - "G1R": nur Spalten mit "1" im Namen (inkl. "Time [s]") → rechter Griff

    Beide Untereinträge enthalten:
      - "data": den gefilterten DataFrame
      - "stats": ein Dictionary mit min/max für jede Spalte (außer "Time [s]")

    Rückgabe: dict[str, dict[str, dict[str, Any]]]
    """
    data_dict = {}
    filtered_data_dict = {}
    for file_path in glob.glob(os.path.join(folder_path, "*.lvm")):
        print(file_path)
        # Lese .lvm-Datei als DataFrame (Header-Zeile 21, deutsche Kommas)
        df = pd.read_csv(file_path, sep="\t", decimal=",", skiprows=0, header=21)
        df.columns = df.columns.astype(str)
        df = df.apply(pd.to_numeric, errors='coerce')
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        clean_df = clean_data(df)
        g1r = clean_df[["Time [s]"] + [col for col in clean_df.columns if "1" in col]]
        g2l = clean_df[["Time [s]"] + [col for col in clean_df.columns if "2" in col]]
        
        data_dict[file_name] = {
            "G1R": {
                "data": g1r,
                "stats": get_min_max_values_per_column(g1r)
            },
            "G2L": {
                "data": g2l,
                "stats": get_min_max_values_per_column(g2l)
            }

        }
        
        # Filtere alle Spalten außer Time [s] mit Savitzky-Golay
        def apply_filter(df):
            df_filtered = df.copy()
            for col in df.columns:
                if col != "Time [s]":
                    df_filtered[col] = savgol_filter(df[col], window_length=SVGwindowlength, polyorder=SVGpolyorder, mode="interp")
            return df_filtered

        g1r_filtered = apply_filter(g1r)
        g2l_filtered = apply_filter(g2l)

        filtered_data_dict[file_name] = {
            "G1R": {
                "data": g1r_filtered,
                "stats": get_min_max_values_per_column(g1r_filtered)
            },
            "G2L": {
                "data": g2l_filtered,
                "stats": get_min_max_values_per_column(g2l_filtered)
            }
        }
    return data_dict, filtered_data_dict if data_dict else (None, None)
