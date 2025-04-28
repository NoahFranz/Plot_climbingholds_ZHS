import matplotlib.pyplot as plt
import pandas as pd
import math
from plotUITLS import*
from matplotlib.colors import to_rgb
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np


import logging

logger = logging.getLogger(__name__)

PLOT_CONFIG = {
    "default_figsize": (6.3, 8),
    "legend_ncol": 5
}

def adjust_color(color: str, shift: float = 0.2) -> Tuple[float, float, float]:
    r, g, b = to_rgb(color)
    r = max(0, min(1, r - shift))
    g = max(0, min(1, g + shift / 2))
    b = max(0, min(1, b + shift))
    return (r, g, b)

def get_color_for(force: str, hold: str) -> Union[str, Tuple[float, float, float]]:
    """
    Wählt eine Farbe für 'force' und 'hold' basierend auf COLOR_MAPPING.
    Für G2L wird die Basisfarbe leicht angepasst.
    """
    # Bestimme Basisschlüssel aus COLOR_MAPPING
    base_key = next((f for f in COLOR_MAPPING if f in force), None)
    base_color = COLOR_MAPPING.get(base_key, "black")
    # Für den linken Griff (G2L) leicht ins Spektrum verschieben
    if hold == "G2L":
        return adjust_color(base_color, shift=0.15)
    return base_color

def _plot_force_lines(ax: plt.Axes, df: pd.DataFrame, forces: List[str],
                      color_map: Dict[str, str], alpha: float = 1.0,
                      suffix: str = "") -> None:
    """
    Plottet alle Spalten aus df, deren Name einen Eintrag in forces enthält.
    """
    print("executing: _plo_data_force_lines")
    time = df["Time [s]"]
    for force in forces:
        cols = [col for col in df.columns if force in col]
        for col in cols:
            ax.plot(time, df[col], label=clean_label(force) + suffix,
                    color=color_map.get(force, "black"), alpha=alpha)

COLOR_MAPPING = {
    "Fy": "blue",
    "Fx": "green",
    "Fz": "orange",
    "Mz": "#8B1A1A",  # Kaminrot
    "FgR": "#9ACD32",
    "FgR_calc": "#32CD32",
    "Fres": "#4B0082",     # Indigo
    "φ_yz": "#800080"      # Lila
}

def plot_single_hold_splitview(
    hold_data: pd.DataFrame,
    forces: List[str],
    filename: str = "",
    grip_label: str = "",
    save_plot: bool = False,
    margin: float = 1.25,
    save_folder: str = ".",
    cutoff: Optional[Dict[str, float]] = None,
    normalizebyweight: bool = False
) -> None:
    """
    Erstellt eine Figure für einen einzelnen Griff, in der:
      - Im oberen Subplot die Normalkräfte (alle Spalten außer "Time [s]" und solchen, die "Mz" enthalten) geplottet werden.
      - Im unteren Subplot der Moment (alle Spalten, die "Mz" enthalten) geplottet wird.
    
    Parameters:
      hold_data : pandas.DataFrame
          DataFrame mit Zeit- und Kraftdaten.
      forces : list
          Liste der auszuwertenden Kräfte.
      filename : str
          Dateiname für die Speicherung des Plots.x
      grip_label : str
          Label des Griffs (z. B. G1R/G2L).
      save_plot : bool
          Wenn True, wird der Plot gespeichert.
      margin : float
          Multiplikator zur Erweiterung der y-Achsen-Grenzen.
      save_folder : str
          Pfad zum Speicherordner.
      cutoff : dict or None
          Dictionary mit "start" und "end" in Sekunden für die Trimmung.
    
    Die Funktion erstellt eine Figure mit zwei Subplots (obere Zeile für Normalkräfte, untere Zeile für Moment),
    passt die Achsenbeschriftung und fügt jeweils eine Legende hinzu.
    """
    try:
        figstyle = "1G_split"

        # Extrahiere die Zeitdaten aus der "Time [s]" Spalte
        time = hold_data["Time [s]"]
        
        # Filtere Kraftspalten basierend auf GUI-Auswahl
        selected_forces = [f for f in forces if f != "Mz"]
        normal_cols = [col for col in hold_data.columns if any(f in col for f in selected_forces)]
        moment_cols = [col for col in hold_data.columns if "Mz" in col and "Mz" in forces]

        # Erstelle eine Figure mit 2 Zeilen und 1 Spalte; 
        # die Breite ist fix (6.3 Zoll, passend für LaTeX) und Höhe auf 8 Zoll gewählt.
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=PLOT_CONFIG["default_figsize"], sharex=True)
        
        # --- Oberer Subplot: Normalkräfte ---
        for col in normal_cols:
            curret_forcen = next((f for f in COLOR_MAPPING if f in col), None)
            ax_top.plot(time, hold_data[col], label=clean_label(col), color=COLOR_MAPPING.get(curret_forcen))
        ax_top.set_title(f"Kräfte – {grip_label}")
        if normal_cols:
            data_subset = hold_data[normal_cols].dropna()
            ax_top.set_ylim(compute_ylimits(data_subset, margin=margin, fallback=(-100, 1000)))
        # Füge eine Legende hinzu (innerhalb des Plots)
        ax_top.legend(loc="upper right", ncol=PLOT_CONFIG["legend_ncol"])
        
        # --- Unterer Subplot: Moment ---
        for col in moment_cols:
            ax_bottom.plot(time, hold_data[col], label=clean_label(col), color=COLOR_MAPPING["Mz"])
        ax_bottom.set_title(f"Moment – {grip_label}")
        ax_bottom.set_xlabel("Time [s]")
        ax_bottom.set_ylabel("Mz [Nm]")
        if moment_cols:
            data_subset = hold_data[moment_cols].dropna()
            y_min, y_max = compute_ylimits(data_subset, margin=margin, fallback=(-10, 10))
            # Wenn beide Werte innerhalb ±7 liegen, setze feste Grenzen
            ax_bottom.set_ylim([y_min, y_max])
        ax_bottom.legend(loc="upper right", ncol=PLOT_CONFIG["legend_ncol"])
        
        # Setze den eindeutigen Titel und speichere optional
        apply_default_plot_style(fig, normalizebyweight=normalizebyweight)
        if save_plot:
            save_figure_with_title(fig, filename, grip_label, save_plot=save_plot, figstyle=figstyle, save_folder=save_folder)
    except Exception as e:
        logger.exception(f"plot_single_hold_splitview failed for {filename} / {grip_label}: {e}")

# =================================== plot_data_per_hold ====================================

def plot_data_per_hold(
    plot_dict: Dict[str, Dict[str, Any]],
    forces_g1: List[str],
    forces_g2: List[str],
    filename: str,
    save_plot: bool = False,
    margin: float = 1.2,
    save_folder: str = ".",
    cutoff: Optional[Dict[str, float]] = None,
    normalizebyweight: bool = False,
    show_contact_time: bool = False
) -> None:
    """
    Erstellt eine Figure mit separaten Subplots für den linken (G2L) und rechten Griff (G1R).
    
    Für jeden Griff wird unterschieden:
      - Falls ausschließlich "Mz" ausgewählt ist, werden nur die Mz-Daten geplottet.
      - Andernfalls werden die normalen Kräfte (außer Mz) geplottet,
        und falls Mz zusätzlich aktiv ist, auf einer Sekundärachse ergänzt.
    
    Anschließend werden die Legenden der primären und sekundären Achsen kombiniert.

    Parameters:
        plot_dict : dict
            Dictionary mit Daten und Metadaten für G1R und G2L.
        forces_g1 : list
            Liste der auszuwertenden Kräfte für G1R.
        forces_g2 : list
            Liste der auszuwertenden Kräfte für G2L.
        filename : str
            Dateiname für die Speicherung des Plots.
        save_plot : bool
            Wenn True, wird der Plot gespeichert.
        margin : float
            Multiplikator zur Erweiterung der y-Achsen-Grenzen.
        save_folder : str
            Pfad zum Speicherordner.
        cutoff : dict or None
            Dictionary mit "start" und "end" in Sekunden für die Trimmung.
    """
    print("executing: plo_data_per_hold")
    try:
        figstyle = "2G"
        grip_label = "OL_UR"

        # Farbmapping für die Kräfte: Zuordnung von Kraftnamen zu Farben
        color_mapping = COLOR_MAPPING
        
        # Prüfe, ob ausschließlich der Momentenwert "Mz" ausgewählt wurde für jeden Griff
        only_mz_g1 = forces_g1 == ["Mz"]
        only_mz_g2 = forces_g2 == ["Mz"]


        # Überprüfe, ob für einen Griff überhaupt Kräfte vorhanden sind (um Achsen zu bestimmen)
        has_g1 = bool(forces_g1)
        has_g2 = bool(forces_g2)
        num_axes = has_g1 + has_g2

        # Erstelle die Figure mit passender Höhe basierend auf der Anzahl der Plots
        fig_height = 4 * num_axes
        fig, axes = plt.subplots(num_axes, 1, figsize=(6.3, fig_height), sharex=True)
        axes = [axes] if num_axes == 1 else axes


        # ----- Linker Griff (G2L) -----
        if only_mz_g2:
            # Falls nur Mz ausgewählt ist, plotte nur Mz auf der primären Achse
            mz_cols = [col for col in plot_dict["G2L"]["data"].columns if "Mz" in col]
            ax_left = axes[0]
            time_left = plot_dict["G2L"]["data"]["Time [s]"]
            plot_mz_on_secondary_axis(ax_left, time_left, plot_dict["G2L"]["data"], mz_cols)
            ax_left.set_ylabel("Mz [Nm]")
            # Hole und setze die Legende
            combine_legends(ax_left, None, loc="upper left", ncol=PLOT_CONFIG["legend_ncol"])
        else:
            # Plotte die normalen Kräfte (außer Mz) auf dem oberen Plot
            ax_left = axes[0]
            time_left = plot_dict["G2L"]["data"]["Time [s]"]
            plot_normal_forces(ax_left, plot_dict["G2L"]["data"], forces_g2, color_mapping)
            ax_left.set_title("GL")
            # Nach Plotten der Normalkräfte: lokale y-Limits berechnen
            data_left = plot_dict["G2L"]["data"][[col for col in plot_dict["G2L"]["data"].columns if any(f in col for f in forces_g2)]]
            y_min_left, y_max_left = compute_ylimits(data_left, margin=margin)
            ax_left.set_ylim([y_min_left, y_max_left])
            # Falls Mz ebenfalls aktiv ist, erstelle eine Sekundäxe und plotte Mz
            mz_cols = [col for col in plot_dict["G2L"]["data"].columns if "Mz" in col]
            sec_ax_left = None
            if mz_cols and "Mz" in forces_g2:
                sec_ax_left = ax_left.twinx()
                # Plot Mz-Daten auf Sekundärachse
                plot_mz_on_secondary_axis(sec_ax_left, time_left, plot_dict["G2L"]["data"], mz_cols)
                # Eigene y-Limits für den Moment basierend auf compute_ylimits
                mz_df_left = plot_dict["G2L"]["data"][mz_cols]
                y_min_mz_left, y_max_mz_left = compute_ylimits(mz_df_left, margin=margin)
                sec_ax_left.set_ylim([y_min_mz_left, y_max_mz_left])
            # Optional: contact_time shading (support multiple contact_time per force)
            if show_contact_time:
                contact_time = plot_dict["G2L"].get("contact_time", {})
                for force in forces_g2:
                    ivals = contact_time.get(force, [])
                    for (t0, t1) in ivals:
                        ax_left.axvspan(t0, t1,
                                        color=get_color_for(force, "G2L"),
                                        alpha=0.15)
            # Kombiniere Legenden von primärer und sekundärer Achse
            combine_legends(ax_left, sec_ax_left, loc="upper left", ncol=PLOT_CONFIG["legend_ncol"])
        
        # ----- Rechter Griff (G1R) -----
        if has_g1:
            if only_mz_g1:
                # Falls für den rechten Griff ausschließlich Mz ausgewählt wurde, plotte nur Mz
                mz_cols = [col for col in plot_dict["G1R"]["data"].columns if "Mz" in col]
                ax_right = axes[1]
                time_right = plot_dict["G1R"]["data"]["Time [s]"]
                plot_mz_on_secondary_axis(ax_right, time_right, plot_dict["G1R"]["data"], mz_cols)
                ax_right.set_ylabel("Mz [Nm]")
                combine_legends(ax_right, None, loc="upper left", ncol=PLOT_CONFIG["legend_ncol"])
            else:
                # Plotte normale Kräfte (außer Mz) für den rechten Griff
                ax_right = axes[1]
                time_right = plot_dict["G1R"]["data"]["Time [s]"]
                plot_normal_forces(ax_right, plot_dict["G1R"]["data"], forces_g1, color_mapping)
                ax_right.set_title("GR")
                ax_right.set_xlabel("Time [s]")
                # Nach Plotten der Normalkräfte: lokale y-Limits berechnen
                data_right = plot_dict["G1R"]["data"][[col for col in plot_dict["G1R"]["data"].columns if any(f in col for f in forces_g1)]]
                y_min_right, y_max_right = compute_ylimits(data_right, margin=margin)
                ax_right.set_ylim([y_min_right, y_max_right])
                # Falls Mz aktiv ist, plotte zusätzlich Mz auf einer Sekundärachse
                mz_cols = [col for col in plot_dict["G1R"]["data"].columns if "Mz" in col]
                sec_ax_right = None
                if mz_cols and "Mz" in forces_g1:
                    sec_ax_right = ax_right.twinx()
                    # Plot Mz-Daten auf Sekundärachse
                    plot_mz_on_secondary_axis(sec_ax_right, time_right, plot_dict["G1R"]["data"], mz_cols)
                    # Eigene y-Limits für den Moment basierend auf compute_ylimits
                    mz_df_right = plot_dict["G1R"]["data"][mz_cols]
                    y_min_mz_right, y_max_mz_right = compute_ylimits(mz_df_right, margin=margin)
                    sec_ax_right.set_ylim([y_min_mz_right, y_max_mz_right])
                # Optional: contact_time shading (support multiple contact_time per force)
                if show_contact_time:
                    contact_time = plot_dict["G1R"].get("contact_time", {})
                    for force in forces_g1:
                        ivals = contact_time.get(force, [])
                        for (t0, t1) in ivals:
                            ax_right.axvspan(t0, t1,
                                             color=get_color_for(force, "G1R"),
                                             alpha=0.15)
                combine_legends(ax_right, sec_ax_right, loc="upper left", ncol=PLOT_CONFIG["legend_ncol"])
        
        # Dynamische Berechnung der Zeitachsen-Grenzen mit 10% Puffer
        time_min = min(time_left.min(), time_right.min()) if has_g1 else time_left.min()
        time_max = max(time_left.max(), time_right.max()) if has_g1 else time_left.max()
        time_range = time_max - time_min
        axes[0].set_xlim([time_min - 0.01 * time_range, time_max + 0.05 * time_range])
        if has_g1:
            axes[1].set_xlim([time_min - 0.01 * time_range, time_max + 0.05 * time_range])
        
        plt.tight_layout()
        apply_default_plot_style(fig, normalizebyweight=normalizebyweight)
        if save_plot:
            save_figure_with_title(fig, filename, grip_label, save_plot=save_plot, figstyle=figstyle, save_folder=save_folder)
    except Exception as e:
        logger.exception(f"plot_data_per_hold failed for {filename}: {e}")

def plot_selected_forces_comparison(
    file_dict: Dict[str, Dict[str, Any]],
    forces_g1: List[str],
    forces_g2: List[str],
    filename: str,
    save_folder: str = ".",
    save_plot: bool = False,
    margin: float = 1.2,
    cutoff: Optional[Dict[str, float]] = None,
    normalizebyweight: bool = False,
    show_contact_time: bool = False
) -> None:
    """
    Erstellt Vergleichsplots für die ausgewählten Kräfte der beiden Griffe.
    
    Parameter:
      file_dict : dict
          Dictionary, das die Daten für G1R und G2L enthält.
          G1R /G2L
          G1R: ['data', 'stats', 'global_limits']
            Spalten: ['Time [s]', 'Fy_1  [N]', 'Fx_1 [N]', 'Fz_1 [N]', 'Mz_1 [Nm]', 'FgR_1 [%]']
          Analog für G2L
      forces_g1 : list
          Liste der auszuwertenden Kräfte für G1R.
      forces_g2 : list
          Liste der auszuwertenden Kräfte für G2L.
      filename : str
          Der Name der Datei für die Speicherung des Plots.
      save_folder : str
          Der Ordner, in dem die Plots gespeichert werden sollen.
      save_plot : bool
          Wenn True, wird die erstellte Figure als PNG abgespeichert.
      margin : float
          Multiplikator zur Erweiterung der y-Achsen-Grenzen.
      cutoff : dict or None
          Dictionary mit "start" und "end" in Sekunden für die Trimmung.
    """
    print("executing: plo_selected_forces_comparison")
    try:
        time = file_dict["G1R"]["data"]["Time [s]"]
        holdNameList = {"G1R", "G2L"}

        # Liste mit den darzustellenden Kräften
        allForcesList = []
        for f in forces_g1 + forces_g2:
            if f not in allForcesList:
                allForcesList.append(f)

        fig, axes = plt.subplots(2, 1, figsize=PLOT_CONFIG["default_figsize"])
        axes_list = list(axes)
        logger.debug(f"allForcesList in plot_selected_forces_comparison: {allForcesList}")
        for idx, current_force in enumerate(allForcesList[:2]):
            ax = axes_list[idx]
            for current_hold in ["G1R", "G2L"]:
                suffix = {"G1R": "_R", "G2L": "_L"}[current_hold]
                alpha = 1 if current_hold == "G1R" else 0.5
                curr_color = get_color_for(current_force, current_hold)
                cols = [c for c in file_dict[current_hold]["data"].columns if current_force in c]
                if not cols:
                    continue
                df = file_dict[current_hold]["data"]
                _plot_force_lines(ax, df, [current_force], COLOR_MAPPING, alpha=alpha, suffix=suffix)

        # Lokale y-Limits für Vergleichsplots berechnen
        for idx, current_force in enumerate(allForcesList[:2]):
            ax = axes_list[idx]
            # Sammle Daten beider Griffe für diese Kraft
            cols_g1 = [c for c in file_dict["G1R"]["data"].columns if current_force in c]
            cols_g2 = [c for c in file_dict["G2L"]["data"].columns if current_force in c]
            df_cmp = pd.concat([
                file_dict["G1R"]["data"][cols_g1],
                file_dict["G2L"]["data"][cols_g2]
            ], axis=1)
            y_min_cmp, y_max_cmp = compute_ylimits(df_cmp, margin=margin)
            ax.set_ylim([y_min_cmp, y_max_cmp])
        # Optional: contact_time shading
        if show_contact_time:
            contact_time_g1 = file_dict["G1R"].get("contact_time", {})
            contact_time_g2 = file_dict["G2L"].get("contact_time", {})
            # Top axis corresponds to first force
            force_top = allForcesList[0]
            if force_top in contact_time_g1:
                t0, t1 = contact_time_g1[force_top]
                axes[0].axvspan(t0, t1, color=get_color_for(force_top, "G1R"), alpha=0.15)
            # Bottom axis corresponds to second force
            if len(allForcesList) > 1:
                force_bot = allForcesList[1]
                if force_bot in contact_time_g2:
                    t0, t1 = contact_time_g2[force_bot]
                    axes[1].axvspan(t0, t1, color=get_color_for(force_bot, "G2L"), alpha=0.15)

        # Top plot axis
        axes[0].set_title(f"Vergleich: {filename}")
        axes[0].set_xlabel("Time [s]")
        axes[0].legend(ncol=PLOT_CONFIG["legend_ncol"])

        # Bottom plot Axis
        axes[1].set_xlabel("Time [s]")
        axes[1].legend(ncol=PLOT_CONFIG["legend_ncol"])
        
        apply_default_plot_style(fig=fig, normalizebyweight=normalizebyweight)
        if save_plot:
            plt.savefig(f"{save_folder}/{filename}.png")
        plt.show()
    except Exception as e:
        logger.exception(f"plot_selected_forces_comparison failed for {filename}: {e}")

def plot_impulses_bar(
    all_lvm_data_dict,
    force="Fz",  # Oder "Fy" etc.
    split_grips=False,  # True: oben/unten, False: gruppiert nebeneinander
    show_values=True,
    figsize=(8, 5),
    title="Impulsvergleich"
):
    """
    Erstellt ein Balkendiagramm der Impulse für mehrere Dateien (Athleten).

    Args:
        all_lvm_data_dict: dict mit Datei → {'G1R': {'impulses': ...}, 'G2L': ...}
        force: Name der betrachteten Kraft, z.B. "Fz"
        split_grips: True = zwei Subplots, False = beide Griffe nebeneinander in einem Plot
        show_values: True = Impulswerte über Balken anzeigen
    """
    # Sammle Impulse
    print("\n +++++++++ in plot_impulse_bar ++++++++++")
    athlete_names = []
    g1r_impulses, g2l_impulses = [], []
    contact_times_g1 = []
    contact_times_g2 = []
   # athlete_idx = 1
    for fname, dct in all_lvm_data_dict.items():
        #get athletename of file
        curr_ath_name = all_lvm_data_dict[fname]["athletename"]

        #Impuls listen
        g1_imp_list = dct.get("G1R", {}).get("impulses", {}).get(force, [])[force]
        g2_imp_list = dct.get("G2L", {}).get("impulses", {}).get(force, [])[force]

         # Kontaktzeiten als Liste von (start, end)
        g1_intervals = dct.get("G1R", {}).get("contact_time", {}).get(force, [])
        g2_intervals = dct.get("G2L", {}).get("contact_time", {}).get(force, [])

        # Falls mehrere contact_time: nimm nur den größten
        g1_imp = max(g1_imp_list, key = abs) if g1_imp_list else 0
        g2_imp = max(g2_imp_list, key = abs) if g2_imp_list else 0
        
        # Kontaktzeit des maximalen Impulses
        g1_ct = find_max_impulse_interval_length(g1_imp_list, g1_intervals)
        g2_ct = find_max_impulse_interval_length(g2_imp_list, g2_intervals)
        contact_times_g1.append(g1_ct)
        contact_times_g2.append(g2_ct)

        print(f"[DEBUG] Datei: {fname}")
      #  print(f"        | G1R Impuls-dict: {g1_imp_dict} | G2L Impuls-dict: {g2_imp_dict}")
        print(f"        | G1R g1_imp: {g1_imp} | G2L g2_imp: {g2_imp}")



        athlete_names.append(curr_ath_name)
        #athlete_idx += 1
        g1r_impulses.append(g1_imp)
        g2l_impulses.append(g2_imp)

    print("\n IMPULS die geplottert werden:")
    print("     g1r_impulses:", g1r_impulses)
    print("     g2l_impulses:", g2l_impulses)
    indices = np.arange(len(athlete_names))

    if split_grips:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        ax1.bar(indices, g1r_impulses, color="deepskyblue", label="G1R")
        ax1.set_ylabel("Impuls G1R ['%'s]")
        ax1.set_title(title + " – G1R")
        if show_values:
            for i, val in enumerate(g1r_impulses):
                ax1.text(i, val, f"{val:.1f}", ha="center", va="bottom")
        ax2.bar(indices, g2l_impulses, color="coral", label="G2L")
        ax2.set_ylabel("Impuls G2L ['%'s]")
        ax2.set_title(title + " – G2L")
        if show_values:
            for i, val in enumerate(g2l_impulses):
                ax2.text(i, val, f"{val:.1f}", ha="center", va="bottom")
        plt.xticks(indices, athlete_names, rotation=25, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        width = 0.35
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(indices - width/2, g1r_impulses, width, label="G1R", color="deepskyblue")
        ax.bar(indices + width/2, g2l_impulses, width, label="G2L", color="coral")
        ax.set_xticks(indices)
        ax.set_xticklabels(athlete_names, rotation=25, ha="right")
        ax.set_ylabel("Impuls ['%'s]")
        ax.set_title(title)
        ax.legend()
        if show_values:
          for i, (val, ct) in enumerate(zip(g1r_impulses, contact_times_g1)):
              txt = f"{val:.1f}\n({ct}s)" if ct is not None else f"{val:.1f}"
              ax.text(i - width/2, val, txt, ha="center", va="bottom")
          for i, (val, ct) in enumerate(zip(g2l_impulses, contact_times_g2)):
              txt = f"{val:.1f}\n({ct}s)" if ct is not None else f"{val:.1f}"
              ax.text(i + width/2, val, txt, ha="center", va="bottom")
        plt.tight_layout()
        plt.show()
        