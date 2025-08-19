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


# --------- New helper functions for pretty component/legend labels ---------
def pretty_component(name: str) -> str:
    """
    Map a raw component name (e.g., 'Fy', 'Fz', 'Fx', 'Mz', 'FgR', 'Fres')
    to a MathText-formatted string for legend labels (e.g., '$F_y$', '$M_z$').
    Falls back to the original name if no mapping exists.
    """
    mapping = {
        "Fy": r"$F_y$",
        "Fz": r"$F_z$",
        "Fx": r"$F_x$",
        "Mz": r"$M_z$",
        "FgR": r"$F_{\mathrm{gR}}$",
        "FgR_sum": r"$F_{\mathrm{gR,sum}}$",
        "Fres_xyz": r"$F_{\mathrm{res}}$",
        "Pres_xyz": r"$P_{\mathrm{res}}$",
        "tcontact": r"$t_{\mathrm{contact}}$",
        "Py": r"$P_y$",
        "Pz": r"$P_z$",
        "Px": r"$P_x$",
        "Pz": r"$P_z$",
        "Pyz": r"$P_{\mathrm{yz}}",
        "Fy_sum": r"$F_{y,\mathrm{sum}}$",
        "Fz_sum": r"$F_{z,\mathrm{sum}}$",
        "Fx_sum": r"$F_{x,\mathrm{sum}}$",
        "Fres_xyz_sum": r"$F_{\mathrm{res,sum}}$",
        "Pres_xyz_sum": r"$P_{\mathrm{res,sum}}$",
    }
    if not isinstance(name, str):
        return name
    key = name.strip()
    return mapping.get(key, key)

def build_legend_label(base_label: str, component: str, include_units: bool = False) -> str:
    """
    Compose a consistent legend label combining a base label (e.g., file/grip)
    and a pretty-printed component. Optionally appends units for moments.

    Example:
        build_legend_label("TrialA", "Mz", include_units=True)
        -> 'TrialA–$M_z$ [\%BW·m]'
    """
    comp = pretty_component(component)
    if include_units and isinstance(component, str) and component.strip().startswith("M"):
        return f"{base_label}–{comp} [%BW·m]"
    return f"{base_label}–{comp}"

def compute_ylimits(data_subset, margin=1.2, fallback=(-100, 800)):
    """
    Berechnet die y-Achsen-Grenzen mit Sicherheitsmarge.
    Unterstützt sowohl Pandas DataFrames als auch NumPy-Arrays.
    """
    if isinstance(data_subset, pd.DataFrame):
        if data_subset.empty:
            print("Using fallback y-limits (empty DataFrame)")
            return fallback
        y_min = data_subset.min().min()
        y_max = data_subset.max().max()
    elif isinstance(data_subset, np.ndarray):
        if data_subset.size == 0:
            print("Using fallback y-limits (empty array)")
            return fallback
        y_min = np.nanmin(data_subset)
        y_max = np.nanmax(data_subset)
    else:
        raise TypeError("data_subset must be a DataFrame or ndarray")

    if pd.isna(y_min) or pd.isna(y_max) or math.isinf(y_min) or math.isinf(y_max):
        return fallback
    if abs(y_min) < 7 and abs(y_max) < 7:
        return (-7, 7)
    return (
        y_min * margin if y_min < 0 else y_min / margin,
        y_max * margin if y_max > 0 else y_max / margin,
    )

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
        fig.savefig(full_path, dpi=700)
        print(f"Plot gespeichert unter: {full_path}")
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
    # ensure only one visible legend entry per base force label (e.g., FgR)
    seen_labels = set()
    for force in forces:
        if force == "Mz":  # Ignoriere Mz in dieser Funktion
            continue
        # Suche Spalten, deren Name den Kraftnamen enthält
        cols = [col for col in hold_data.columns if force in col]
        for col in cols:
            label = clean_label(col).strip()
            # normalize FgR_1 / FgR_2 -> FgR for legend
            if label.startswith("FgR_"):
                label = "FgR"
            # only first occurrence gets visible label
            if label and label not in seen_labels:
                legend_label = label
                seen_labels.add(label)
            else:
                legend_label = f"_{label}" if label else None
            color_key = "FgR" if force.startswith("FgR") else force
            ax.plot(time_data, hold_data[col], label=legend_label, color=color_mapping.get(color_key, None))


    

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
    Entfernt doppelte Labels (bevorzugt erste Vorkommen) und ignoriert
    technisch versteckte Labels (die mit '_' beginnen).
    """
    handles, labels = ax.get_legend_handles_labels()
    if secondary_ax is not None:
        sec_handles, sec_labels = secondary_ax.get_legend_handles_labels()
        handles += sec_handles
        labels += sec_labels
    # Filter + de-duplicate
    seen = set()
    f_handles, f_labels = [], []
    for h, l in zip(handles, labels):
        # skip empty labels, intentionally hidden labels, or invisible artists
        if not l or l.startswith("_"):
            continue
        if hasattr(h, "get_visible") and not h.get_visible():
            continue
        if l in seen:
            continue  # drop duplicates
        seen.add(l)
        f_handles.append(h)
        f_labels.append(l)
    if f_handles:
        ax.legend(f_handles, f_labels, loc=loc, ncol=ncol)
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

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
        ax.set_ylabel("F [%BW]")
    elif all("φ_yz" in label for label in labels):
        ax.set_ylabel("Winkel [°]")
    elif normalizebyweight:
        ax.set_ylabel("F [%BW]")
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
