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
    Bestimmt den globalen minimalen und maximalen Wert (außer 'Time [s]') aus dem DataFrame.
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