import matplotlib.pyplot as plt
import pandas as pd
import math

DEFAULT_FIGSIZE = (6.3, 8)

def clean_label(label):
    """
    Entfernt standardisierte Suffixe und Einheiten aus einem Spaltennamen.
    """
    return label.replace("_1", "").replace("_2", "").replace("[N]", "").replace("[Nm]", "")

def compute_ylimits(data_subset, margin=1.2, fallback=(-100, 800)):
    """
    Berechnet die y-Achsen-Grenzen mit Sicherheitsmarge. Fällt auf Standardwerte zurück, wenn ungültig.
    """
    if data_subset.empty:
        print("using fallback values for y-limits")
        return fallback
    y_min = data_subset.min().min()
    y_max = data_subset.max().max()
    if pd.isna(y_min) or pd.isna(y_max) or math.isinf(y_min) or math.isinf(y_max):
        return fallback
    if abs(y_min) < 7 and abs(y_max) < 7:
        return (-7, 7)
    return (y_min * margin if y_min < 0 else y_min / margin,
            y_max * margin if y_max > 0 else y_max / margin)

def apply_default_plot_style(fig):
    """
    Wendet Standardwerte für Schriftgröße, Schriftart und andere Stiloptionen auf die gesamte Figure an.
    """
    for ax in fig.get_axes():
        ax.tick_params(labelsize=10)
        ax.title.set_fontsize(12)
        ax.xaxis.label.set_fontsize(11)
        ax.yaxis.label.set_fontsize(11)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Arial")
            label.set_fontsize(10)
    
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend:
            legend.prop.set_size(3)
            legend._ncol = 4
            legend.set_frame_on(True)

def save_figure_with_title(fig, filename, grip_label, save_plot=False, figstyle="", save_folder="."):
    """
    Setzt den Figure-Titel auf einen eindeutigen Namen, der aus 'filename' und 'grip_label'
    zusammengesetzt wird. Dieser Name wird immer angezeigt.
    Wenn save_plot=True, wird die Figure als PNG gespeichert.
    """
    safe_name = f"{filename}_{grip_label}_{figstyle}_plot.png"
    if save_plot:
        print("saving plots as png")
        # Einzelne Subplot-Titel und Suptitle entfernen, falls gespeichert wird
        for ax in fig.get_axes():
            ax.set_title("")
        fig._suptitle = None
        plt.tight_layout()
        import os
        full_path = os.path.join(save_folder, safe_name)
        fig.savefig(full_path)
        print(f"Plot gespeichert unter: {full_path}")
    else:
        fig.suptitle(safe_name, fontsize=14)
        plt.tight_layout()

def plot_normal_forces(ax, hold_data, forces, color_mapping):
    """
    Plottet alle normalen Kräfte (außer Mz) auf der übergebenen Achse.
    Gibt ein Tupel (y_min, y_max) der beobachteten y-Grenzen zurück.
    """
    time_data = hold_data["Time [s]"]
    y_min = float('inf')
    y_max = float('-inf')
    for force in forces:
        if force == "Mz":  # Ignoriere Mz in dieser Funktion
            continue
        # Suche Spalten, deren Name den Kraftnamen enthält
        cols = [col for col in hold_data.columns if force in col]
        for col in cols:
            label = clean_label(col)
            ax.plot(time_data, hold_data[col], label=label, color=color_mapping.get(force, None))
            local_min = hold_data[col].min()
            local_max = hold_data[col].max()
            y_min = min(y_min, local_min)
            y_max = max(y_max, local_max)
    return y_min, y_max

def plot_mz_on_secondary_axis(ax, time, data, mz_cols):
    # Bestimme die minimalen und maximalen Werte für Mz aus den angegebenen Spalten
    mz_raw_min = data[mz_cols].min().min()
    mz_raw_max = data[mz_cols].max().max()
    
    # Berechne negative und positive Anteile
    mz_neg = abs(mz_raw_min) if mz_raw_min < 0 else 0
    mz_pos = mz_raw_max if mz_raw_max > 0 else 0
    
    # Bestimme den Mz-Bereich mit einer Sicherheitsmarge von 10%
    mz_range = max(mz_neg, mz_pos, 5) * 1.1
    
    # Plotte jede Mz-Spalte in einem hellen Farbton
    for col in mz_cols:
        ax.plot(time, data[col], label=clean_label(col), color="thistle")
    
    # Setze Beschriftung und y-Achsenlimits für Mz
    ax.set_ylabel("Mz [Nm]")
    ax.set_ylim([-mz_range, mz_range])

def combine_legends(ax, secondary_ax=None, loc="upper left", ncol=4):
    """
    Kombiniert die Legenden von ax und, falls vorhanden, secondary_ax.
    """
    handles, labels = ax.get_legend_handles_labels()
    if secondary_ax is not None:
        sec_handles, sec_labels = secondary_ax.get_legend_handles_labels()
        handles += sec_handles
        labels += sec_labels
    if handles:
        ax.legend(handles, labels, loc=loc, ncol=ncol)

def plot_mz_data(ax, hold_data):
    """
    Plottet die Mz-Daten auf der übergebenen Achse und gibt die Legendenhandles zurück.
    """
    time_data = hold_data["Time [s]"]
    mz_cols = [col for col in hold_data.columns if "Mz" in col]
    plot_mz_on_secondary_axis(ax, time_data, hold_data, mz_cols)
    return ax.get_legend_handles_labels()