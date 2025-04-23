import tkinter as tk
def clean_data(df):
    """
    Entfernt alle Spalten, die ein 'U' im Namen oder 'Comment' enthalten.
    """
    columns_to_drop = [col for col in df.columns if "U" in col or "Comment" in col or "X_Value" in col]
    df_clean = df.drop(columns=columns_to_drop)
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
    und fügt diese als 'global_limits' zum plot_dict hinzu.
    Es wird das extremste y_max und das kleinste y_min beider Griffe verwendet.
    """
    global_limits = {}

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

    # Speichern der globalen Limits in global_limits
    global_limits["global_y_min"] = global_y_min
    global_limits["global_y_max"] = global_y_max

    # Füge die globalen Limits zum plot_dict hinzu
    plot_dict["G1R"]["global_limits"] = global_limits
    plot_dict["G2L"]["global_limits"] = global_limits

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
            print(f"  {hold}: {list(content.keys())}")
            print(f"    Spalten: {content['data'].columns.tolist()}")
            print(f"    stats {hold}:")
            for k, v in content["stats"].items():
                min_v = f"{v['min']:.1f}"
                max_v = f"{v['max']:.1f}"
                print(f"      {k}: min= {min_v}, max= {max_v}")
            if "global_limits" in content:
                gl = content["global_limits"]
                print(f"    global_limits: min= {gl['global_y_min']:.2f}, max= {gl['global_y_max']:.2f}")
            else:
                print("    global_limits: Nicht gesetzt")