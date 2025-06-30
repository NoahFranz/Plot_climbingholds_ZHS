from scipy.io import savemat
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import numpy as np

DEFAULT_FIGSIZE = (6.3, 8)

def apply_latex_style():
    """
    Applies LaTeX styling to matplotlib plots for consistent, publication-quality output.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 100,
        "axes.linewidth": 1,
        "lines.linewidth": 1,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": True,
        "legend.edgecolor": "black",
        "grid.alpha": 0.3,
    })

def clean_label(label):
    """
    Entfernt standardisierte Suffixe und Einheiten aus einem Spaltennamen.
    """
    return label.replace("_1", "").replace("_2", "").replace("[N]", "").replace("[Nm]", "").replace("[%]", "")

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

def apply_default_plot_style(fig, normalizebyweight=False):
    """
    Wendet Standardwerte für Schriftgröße, Schriftart und andere Stiloptionen auf die gesamte Figure an.
    Entfernt doppelte Schleifen und hält y-Label-Logik zentral.
    """
   # apply_latex_style()

    for ax in fig.get_axes():
        # Grundstil
        ax.tick_params(labelsize=10)
        ax.title.set_fontsize(12)
        ax.xaxis.label.set_fontsize(11)
        ax.yaxis.label.set_fontsize(11)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Serif")
            label.set_fontsize(10)

        # Dynamisches y-Label
        set_dynamic_ylabel(ax, normalizebyweight=normalizebyweight)
        # Erhöhe den Abstand des y-Achsentitels, damit er nicht abgeschnitten wird
        ax.yaxis.labelpad = 1

        # Legendenstil
        legend = ax.get_legend()
        if legend:
            legend.prop.set_size(3)
            legend._ncol = 5
            legend.set_frame_on(True)

        # FgR-Spezialfall: Limits oder Sekundärachse
        labels = [line.get_label() for line in ax.get_lines()]
        if labels and all("FgR" in label for label in labels):
            ax.set_ylim([-5, 90])
        elif any("FgR" in label for label in labels) and not all("FgR" in label for label in labels):
            sec_ax = ax.twinx()
            for line in ax.get_lines():
                if "FgR" in line.get_label():
                    sec_ax.plot(line.get_xdata(), line.get_ydata(),
                                label=line.get_label(),
                                color=line.get_color(),
                                linestyle=line.get_linestyle())
                    line.set_visible(False)
            # Dynamisches y-Label für Sekundärachse
            set_dynamic_ylabel(sec_ax, normalizebyweight=normalizebyweight)
            sec_ax.set_ylim([-5, 90])
            combine_legends(ax, sec_ax, loc="upper left", ncol=5)

    # Passe das Layout an, damit Achsentitel und Suptitle nicht abgeschnitten werden
    fig.tight_layout()
    # Optional: Erweitere den linken Rand bei Bedarf
    fig.subplots_adjust(left=0.09)

def save_figure_with_title(fig, filename, grip_label, save_plot=False, figstyle="", save_folder="."):
    """
    Setzt den Figure-Titel auf einen eindeutigen Namen, der aus 'filename' und 'grip_label'
    zusammengesetzt wird. Dieser Name wird immer angezeigt.
    Wenn save_plot=True, wird die Figure als PNG gespeichert.
    """
    safe_name = f"{filename}_{grip_label}_{figstyle}_plot.png"
    # Ersetze doppelte Unterstriche durch einfache
    safe_name = safe_name.replace("__", "_")
    if save_plot:
        print("saving plots as png")
        # Einzelne Subplot-Titel und Suptitle entfernen, falls gespeichert wird
        #for ax in fig.get_axes():
        #    ax.set_title("")
        full_path = os.path.join(save_folder, safe_name)
        fig.savefig(full_path)
        svg_name = safe_name.replace(".png", ".svg")
        full_path_svg = os.path.join(save_folder, svg_name)
        fig.savefig(full_path_svg)
        print(f"Plot gespeichert unter: {full_path}")
        print(f"Plot gespeichert unter: {full_path_svg}")
    else:
        fig.suptitle(safe_name, fontsize=14)
        fig._suptitle = safe_name

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

def combine_legends(ax, secondary_ax=None, loc="upper left", ncol=5):
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

def set_dynamic_ylabel(ax, normalizebyweight=False):
    """
    Setzt den y-Achsentitel je nach Dateninhalt.
    Wenn alle gezeigten Linien nur 'FgR' oder 'FgR_calc' enthalten,
    wird 'F [%]' gesetzt, bei 'φ_yz' wird 'Winkel [°]' gesetzt.
    Wenn normalizebyweight=True, wird unabhängig von den anderen Kräften 'F [%]' gesetzt.
    Sonst 'F [N]'.
    """
    labels = [line.get_label() for line in ax.get_lines()]
    if all("FgR" in label for label in labels):
        ax.set_ylabel("F [%]")
    elif all("φ_yz" in label for label in labels):
        ax.set_ylabel("Winkel [°]")
    elif normalizebyweight:
        ax.set_ylabel("F [%]")
    else:
        ax.set_ylabel("F [N]")



def find_max_impulse_interval_length(impulses, intervals):
    """
    Findet zu einer Liste von Impulsen und Kontaktintervallen das Zeitintervall des maximalen Impulses
    und berechnet dessen Dauer (auf 1 Nachkommastelle gerundet).

    Args:
        impulses: Liste von Impulswerten (float)
        intervals: Liste von (start, end)-Tupeln, in Sekunden

    Returns:
        (t0, t1, length): Start- und Endzeit des Intervalls und die Dauer in Sekunden (float, float, float)
        Falls Eingaben leer oder inkonsistent: (None, None, None)
    """
    if not impulses or not intervals or len(impulses) != len(intervals):
        return None, None, None
    max_idx = int(np.argmax(impulses))
    t0, t1 = intervals[max_idx]
    interval_length = round(t1 - t0, 1)
    return interval_length


def apply_barplot_style(ax, margin=1.2):
    """
    Setzt Standard-Layout für Barplots.
    Erhöht y-Limit automatisch mit 'margin'.
    """
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    new_ymax = ymax + (y_range * (margin - 1))
    ax.set_ylim([ymin, new_ymax])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()


def map_force_to_axis(force: str) -> str:
    """
    Wandelt einen Kraftnamen in den zugehörigen Achsenbuchstaben um.
    Beispiele:
      - "Fz" → "z"
      - "Fy" → "y"
      - "Fx" → "x"
      - "Mz" → "mz"
    Ist der Kraftname nicht bekannt, wird ein leerer String zurückgegeben.
    """
    mapping = {
        "Fz": "z",
        "Fy": "y",
        "Fx": "x",
        "Mz": "mz"
    }
    # Normalisiere Eingabe auf Groß-/Kleinschreibung
    key = force.strip().capitalize()
    return mapping.get(key, "")
