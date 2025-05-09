
import matplotlib.pyplot as plt
import pandas as pd
from plotUITLS import*
from matplotlib.colors import to_rgb
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import os  # für Dateisystemoperationen
from utils import print_nested_keys


# plotdata.py: Sammlung von Funktionen zum Erstellen von Diagrammen für Kraft- und Impulsdaten
# Diese Datei enthält:
# - Funktion zum Anpassen von Farben für verschiedene Griffe
# - Funktionen zum Erstellen von Linien- und Balkendiagrammen für Daten aus G1R und G2L
# - Unterstützung für Kontaktzeit-Markierungen und Impuls-Balkendiagramme


import logging

logger = logging.getLogger(__name__)

# Basiskonfiguration für alle Plots: Standardgröße und Legenden-Spaltenzahl
PLOT_CONFIG = {
    "default_figsize": (6.3, 8),
    "legend_ncol": 5
}

def adjust_color(color: str, shift: float = 0.2) -> Tuple[float, float, float]:
    # Verschiebt eine Farbe leicht, um Variante für den linken Griff zu erzeugen
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
    # wählt je nach Griff (G2L vs. G1R) die passende Farbe aus
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
    # Zeichnet alle ausgewählten Kräfte als Linien in der aktuellen Achse
    print("executing: _plo_data_force_lines")
    time = df["Time [s]"]
    for force in forces:
        cols = [col for col in df.columns if force in col]
        for col in cols:
            ax.plot(time, df[col], label=clean_label(force) + suffix,
                    color=color_map.get(force, "black"), alpha=alpha)

COLOR_MAPPING = {
    "Fy": "orange",
    "Fx": "green",
    "Fz": "blue",
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

        # Zeitachse aus den Rohdaten extrahieren
        time = hold_data["Time [s]"]
        
        # Filtere Kraftspalten basierend auf GUI-Auswahl
        selected_forces = [f for f in forces if f != "Mz"]
        normal_cols = [col for col in hold_data.columns if any(f in col for f in selected_forces)]
        moment_cols = [col for col in hold_data.columns if "Mz" in col and "Mz" in forces]

        # Erstelle eine Figure mit 2 Zeilen und 1 Spalte; 
        # die Breite ist fix (6.3 Zoll, passend für LaTeX) und Höhe auf 8 Zoll gewählt.
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=PLOT_CONFIG["default_figsize"], sharex=True)
        fig.suptitle(filename, fontsize=12)
        
        # Untere bzw. obere Achsen einrichten und Daten plotten
        for col in normal_cols:
            curret_forcen = next((f for f in COLOR_MAPPING if f in col), None)
            ax_top.plot(time, hold_data[col], label=clean_label(col), color=COLOR_MAPPING.get(curret_forcen))
        ax_top.set_title(f"Kräfte – {grip_label}")
        if normal_cols:
            data_subset = hold_data[normal_cols].dropna()
            ax_top.set_ylim(compute_ylimits(data_subset, margin=margin, fallback=(-100, 1000)))
        # Füge eine Legende hinzu (innerhalb des Plots)
        ax_top.legend(loc="upper right", ncol=PLOT_CONFIG["legend_ncol"])
        
        for col in moment_cols:
            ax_bottom.plot(time, hold_data[col], label=clean_label(col), color=COLOR_MAPPING["Mz"])
        ax_bottom.set_title(f"Moment – {grip_label}")
        ax_bottom.set_xlabel("Time [s]")
        ax_bottom.set_ylabel("Mz [Nm]")
        if moment_cols:
            data_subset = hold_data[moment_cols].dropna()
            y_min, y_max = compute_ylimits(data_subset, margin=margin, fallback=(-10, 10))
            ax_bottom.set_ylim([y_min, y_max])
        ax_bottom.legend(loc="upper right", ncol=PLOT_CONFIG["legend_ncol"])
        
        # Beschriftungen und Achsenbegrenzungen setzen
        apply_default_plot_style(fig, normalizebyweight=normalizebyweight)
        if save_plot:
            save_figure_with_title(fig, filename, grip_label, save_plot=save_plot, figstyle=figstyle, save_folder=save_folder)
    except Exception as e:
        # Ausnahmebehandlung: Bei Fehlern in der Plot-Erstellung ausgeben, aber Programm nicht abbrechen
        logger.exception(f"plot_single_hold_splitview failed for {filename} / {grip_label}: {e}")

# ======================================================
# Funktion: plot_data_per_hold
# Erstellt ein zweizeiliges Diagramm mit G2L (links) und G1R (rechts)
# Unterstützt normale Kräfte und Momente, optionale Kontaktzeit-Schattierungen
# ======================================================

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
    show_contact_time: bool = False,
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
    print("     executing: plo_data_per_hold")
    print("     current file_dict:")
   # print_nested_keys(plot_dict, 1)
    
    try:
        figstyle = "2G"
        grip_label = "OL_UR"

        # Farbzuordnungen für die verschiedenen Krafttypen (inkl. neu hinzugefügter)
        color_mapping = COLOR_MAPPING
        
        # Prüfe, ob ausschließlich der Momentenwert "Mz" ausgewählt wurde für jeden Griff
        only_mz_g1 = forces_g1 == ["Mz"]
        only_mz_g2 = forces_g2 == ["Mz"]

        # Aufbau der Figure und Achsen basierend auf Anzahl der dargestellten Griffe
        has_g1 = bool(forces_g1)
        has_g2 = bool(forces_g2)
        num_axes = has_g1 + has_g2
        fig_height = 4 * num_axes
        fig, axes = plt.subplots(num_axes, 1, figsize=(6.3, fig_height), sharex=True)
        axes = [axes] if num_axes == 1 else axes
        # === Einfügen des Dateinamens/Identität als Suptitle ===
        file_identity = plot_dict.get("file_identity", filename)
        fig.suptitle(file_identity, fontsize=12)

        # --- Plot für linken Griff (G2L) ---
        if only_mz_g2:
            mz_cols = [col for col in plot_dict["G2L"]["data"].columns if "Mz" in col]
            ax_left = axes[0]
            time_left = plot_dict["G2L"]["data"]["Time [s]"]
            plot_mz_on_secondary_axis(ax_left, time_left, plot_dict["G2L"]["data"], mz_cols)
            ax_left.set_ylabel("Mz [Nm]")
            combine_legends(ax_left, None, loc="upper left", ncol=PLOT_CONFIG["legend_ncol"])
        else:
            ax_left = axes[0]
            time_left = plot_dict["G2L"]["data"]["Time [s]"]
            # Falls nur Moment gewünscht: nur Mz darstellen
            plot_normal_forces(ax_left, plot_dict["G2L"]["data"], forces_g2, color_mapping)
            ax_left.set_title("GL")
            # Normalkräfte plotten und y-Grenzen berechnen
            data_left = plot_dict["G2L"]["data"][[col for col in plot_dict["G2L"]["data"].columns if any(f in col for f in forces_g2)]]
            y_min_left = plot_dict["G1R"]["global_y_limits"]["global_y_min"]*1.2
            y_max_left = plot_dict["G1R"]["global_y_limits"]["global_y_max"]*1.2
            ax_left.set_ylim([y_min_left, y_max_left])
            # Optional: Momente auf Sekundärachse anzeigen
            mz_cols = [col for col in plot_dict["G2L"]["data"].columns if "Mz" in col]
            sec_ax_left = None
            if mz_cols and "Mz" in forces_g2:
                sec_ax_left = ax_left.twinx()
                plot_mz_on_secondary_axis(sec_ax_left, time_left, plot_dict["G2L"]["data"], mz_cols)
                mz_df_left = plot_dict["G2L"]["data"][mz_cols]
                y_min_mz_left, y_max_mz_left = compute_ylimits(mz_df_left, margin=margin)
                sec_ax_left.set_ylim([y_min_mz_left, y_max_mz_left])
           
            # Optional: Kontaktzeiten als halbtransparente Flächen markieren und alle Impulse pro Intervall anzeigen
            if show_contact_time:
                # Kontaktzeiten als halbtransparente Flächen markieren
                contact_time = plot_dict["G2L"].get("contact_time", {})
                for force in forces_g2:
                    ivals = contact_time.get(force, [])
                    for (t0, t1) in ivals:
                        ax_left.axvspan(t0, t1,
                                        color=get_color_for(force, "G2L"),
                                        alpha=0.15)

                # Alle Impulse pro Intervall anzeigen
                impulses_dict = plot_dict["G2L"].get("impulses", {})
                contact_time = plot_dict["G2L"].get("contact_time", {})
                for force in forces_g2:
                    imp_list = impulses_dict.get(force, [])
                    ivals = contact_time.get(force, [])
                    for (imp, (t0, t1)) in zip(imp_list, ivals):
                        try:
                            imp_val = float(imp)
                            if abs(imp_val) > 0 and t1 > t0:
                                # Place label at right end of interval, 3/4 up the y-axis
                                x_pos = t1+0.5
                                ylims = ax_left.get_ylim()
                                y_pos = ylims[0] + 0.75 * (ylims[1] - ylims[0])
                                ct = t1 - t0
                                txt = f"{imp_val:.1f}\n({ct:.2f}s)"
                                ax_left.text(x_pos, y_pos, txt, color="grey", ha="left", va="center", fontsize=6)
                        except Exception as e:
                            logger.warning(f"Fehler beim Anzeigen von Impuls {force} für G2L: {e}")

            # Zusammenführen von Primär- und Sekundär-Legenden
            combine_legends(ax_left, sec_ax_left, loc="upper left", ncol=PLOT_CONFIG["legend_ncol"])
        
        # --- Plot für rechten Griff (G1R) ---
        if has_g1:
            if only_mz_g1:
                mz_cols = [col for col in plot_dict["G1R"]["data"].columns if "Mz" in col]
                ax_right = axes[1]
                time_right = plot_dict["G1R"]["data"]["Time [s]"]
                plot_mz_on_secondary_axis(ax_right, time_right, plot_dict["G1R"]["data"], mz_cols)
                ax_right.set_ylabel("Mz [Nm]")
                combine_legends(ax_right, None, loc="upper left", ncol=PLOT_CONFIG["legend_ncol"])
            else:
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
                    plot_mz_on_secondary_axis(sec_ax_right, time_right, plot_dict["G1R"]["data"], mz_cols)
                    mz_df_right = plot_dict["G1R"]["data"][mz_cols]
                    y_min_mz_right, y_max_mz_right = compute_ylimits(mz_df_right, margin=margin)
                    sec_ax_right.set_ylim([y_min_mz_right, y_max_mz_right])
                # Optional: Kontaktzeiten als halbtransparente Flächen markieren
                if show_contact_time:
                    contact_time = plot_dict["G1R"].get("contact_time", {})
                    for force in forces_g1:
                        ivals = contact_time.get(force, [])
                        for (t0, t1) in ivals:
                            ax_right.axvspan(t0, t1,
                                             color=get_color_for(force, "G1R"),
                                             alpha=0.15)
                
                
                # Alle Impulse pro Intervall anzeigen
                impulses_dict = plot_dict["G1R"].get("impulses", {})
                contact_time = plot_dict["G1R"].get("contact_time", {})
                for force in forces_g1:
                    imp_list = impulses_dict.get(force, [])
                    ivals = contact_time.get(force, [])
                    for (imp, (t0, t1)) in zip(imp_list, ivals):
                        try:
                            imp_val = float(imp)
                            if abs(imp_val) > 0 and t1 > t0:
                                # Place label at right end of interval, 3/4 up the y-axis
                                x_pos = t1+0.5
                                ylims = ax_right.get_ylim()
                                y_pos = ylims[0] + 0.75 * (ylims[1] - ylims[0])
                                ct = t1 - t0
                                txt = f"{imp_val:.1f}\n({ct:.2f}s)"
                                ax_right.text(x_pos, y_pos, txt, color="grey", ha="left", va="center", fontsize=6)
                        except Exception as e:
                            logger.warning(f"Fehler beim Anzeigen von Impuls {force} für G1R: {e}")
                combine_legends(ax_right, sec_ax_right, loc="upper left", ncol=PLOT_CONFIG["legend_ncol"])
        # Zeitbereich mit kleinem Puffer setzen, damit Linien nicht abgeschnitten werden
        time_min = min(time_left.min(), time_right.min()) if has_g1 else time_left.min()
        time_max = max(time_left.max(), time_right.max()) if has_g1 else time_left.max()
        time_range = time_max - time_min
        axes[0].set_xlim([time_min - 0.01 * time_range, time_max + 0.05 * time_range])
        if has_g1:
            axes[1].set_xlim([time_min - 0.01 * time_range, time_max + 0.05 * time_range])
        
        plt.tight_layout()

        # Aufräumen und Standard-Stil erneut anwenden
        apply_default_plot_style(fig, normalizebyweight=normalizebyweight)
        if save_plot:
            save_figure_with_title(fig, filename, grip_label, save_plot=save_plot, figstyle=figstyle, save_folder=save_folder)
    except Exception as e:
        # Ausnahmebehandlung: Bei Fehlern in der Plot-Erstellung ausgeben, aber Programm nicht abbrechen
        logger.exception(f"plot_data_per_hold failed for {filename}: {e}")

# ======================================================
# Funktion: plot_selected_forces_comparison
# Vergleicht jeweils zwei Kräfte (z.B. Fz und Fy) zwischen G1R und G2L
# ======================================================
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
    """
    print("executing: plo_selected_forces_comparison")
    try:
        # Haupt-Schleife: für jede gewählte Kraft eine Achse erstellen
        time = file_dict["G1R"]["data"]["Time [s]"]
        holdNameList = {"G1R", "G2L"}
        # Liste der darzustellenden Kräfte
        allForcesList = []
        for f in forces_g1 + forces_g2:
            if f not in allForcesList:
                allForcesList.append(f)
        fig, axes = plt.subplots(2, 1, figsize=PLOT_CONFIG["default_figsize"])
        file_identity = file_dict.get("file_identity", filename)
        fig.suptitle(file_identity, fontsize=12)
        axes_list = list(axes)
        logger.debug(f"allForcesList in plot_selected_forces_comparison: {allForcesList}")
        for idx, current_force in enumerate(allForcesList[:2]):
            ax = axes_list[idx]
            # In jeder Achse beide Griffe in unterschiedlichen Farbtönen plotten
            for current_hold in ["G1R", "G2L"]:
                suffix = {"G1R": "_R", "G2L": "_L"}[current_hold]
                alpha = 1 if current_hold == "G1R" else 0.5
                curr_color = get_color_for(current_force, current_hold)
                cols = [c for c in file_dict[current_hold]["data"].columns if current_force in c]
                if not cols:
                    continue
                df = file_dict[current_hold]["data"]
                _plot_force_lines(ax, df, [current_force], COLOR_MAPPING, alpha=alpha, suffix=suffix)
        # Einheitliche y-Grenzen basierend auf zusammengesetzten Daten beider Griffe
        for idx, current_force in enumerate(allForcesList[:2]):
            ax = axes_list[idx]
            cols_g1 = [c for c in file_dict["G1R"]["data"].columns if current_force in c]
            cols_g2 = [c for c in file_dict["G2L"]["data"].columns if current_force in c]
            df_cmp = pd.concat([
                file_dict["G1R"]["data"][cols_g1],
                file_dict["G2L"]["data"][cols_g2]
            ], axis=1)
            y_min_cmp, y_max_cmp = compute_ylimits(df_cmp, margin=margin)
            ax.set_ylim([y_min_cmp, y_max_cmp])
        # Optional: Kontaktzeit-Markierungen für jede Kraft in den Vergleichsplots
        if show_contact_time:
            contact_time_g1 = file_dict["G1R"].get("contact_time", {})
            contact_time_g2 = file_dict["G2L"].get("contact_time", {})
            force_top = allForcesList[0]
            if force_top in contact_time_g1:
                t0, t1 = contact_time_g1[force_top]
                axes[0].axvspan(t0, t1, color=get_color_for(force_top, "G1R"), alpha=0.15)
            if len(allForcesList) > 1:
                force_bot = allForcesList[1]
                if force_bot in contact_time_g2:
                    t0, t1 = contact_time_g2[force_bot]
                    axes[1].axvspan(t0, t1, color=get_color_for(force_bot, "G2L"), alpha=0.15)
        # Legenden, Achsentitel, Stil setzen und Plot speichern/anzeigen
        axes[0].set_title(f"Vergleich: {filename}")
        axes[0].set_xlabel("Time [s]")
        axes[0].legend(ncol=PLOT_CONFIG["legend_ncol"])
        axes[1].set_xlabel("Time [s]")
        axes[1].legend(ncol=PLOT_CONFIG["legend_ncol"])
        apply_default_plot_style(fig=fig, normalizebyweight=normalizebyweight)
        if save_plot:
            plt.savefig(f"{save_folder}/{filename}.png")
        plt.show()
    except Exception as e:
        # Ausnahmebehandlung: Bei Fehlern in der Plot-Erstellung ausgeben, aber Programm nicht abbrechen
        logger.exception(f"plot_selected_forces_comparison failed for {filename}: {e}")














# ======================================================
# Funktion: plot_impulses_bar
# Erstellt Balkendiagramm für Impulse verschiedener Dateien (Athleten) und Griffe
# Optionale Anzeige von Impulswert und Kontaktzeit über jedem Balken
# ======================================================
def plot_impulses_bar(
    all_lvm_data_dict,
    forces: List[str] = [],
    split_grips: bool = False,  # True: oben/unten, False: gruppiert nebeneinander
    show_values: bool = True,
    figsize: Tuple[int, int] = (8, 5),
    title: str = "Impulsvergleich",
    optional_suffix: str = "",  # Optionaler Zusatz für Dateinamen
    save_plot: bool = False,     # Speichern aktivieren
    save_folder: str = ".",      # Zielordner für gespeicherte Plots
    normalizebyweight: bool = False
) -> None:
    """
    Erstellt ein Balkendiagramm der Impulse für mehrere Dateien (Athleten).
    """
    # Initialisiere Listen für Athletennamen, Impulswerte und Kontaktzeiten
    print("\n +++++++++ in plot_impulse_bar ++++++++++")
    print("     forces: ", forces)
    print_nested_keys(all_lvm_data_dict,1)

    # variable init
    optional_suffix += "_BAR"
    y_titel = " [%s]"
    MARGIN = 1.1
    margin = MARGIN
    if not normalizebyweight:
        y_titel = " [Ns]"

    for force in forces:
        direction = map_force_to_axis(force=force)
        athlete_names = []
        g1r_impulses, g2l_impulses = [], []
        contact_times_g1 = []
        contact_times_g2 = []

        # loop over data
        for fname, dct in all_lvm_data_dict.items():
            curr_identity = dct.get("file_identity", "Unknown")

            # impulsdaten
            g1_imp_list = dct.get("G1R", {}).get("impulses", {}).get(force, [])
            g2_imp_list = dct.get("G2L", {}).get("impulses", {}).get(force, [])

            # contact time
            g1_intervals = dct.get("G1R", {}).get("contact_time", {}).get(force, [])
            g2_intervals = dct.get("G2L", {}).get("contact_time", {}).get(force, [])
            g1_imp = max(g1_imp_list, key=abs) if g1_imp_list else 0
            g2_imp = max(g2_imp_list, key=abs) if g2_imp_list else 0
            g1_ct = find_max_impulse_interval_length(g1_imp_list, g1_intervals)
            g2_ct = find_max_impulse_interval_length(g2_imp_list, g2_intervals)
            if g1_imp != 0 and g1_ct > 0:
                g1r_impulses.append(g1_imp)
                contact_times_g1.append(g1_ct)
            if g2_imp != 0 and g2_ct > 0:
                g2l_impulses.append(g2_imp)
                contact_times_g2.append(g2_ct)
            athlete_names.append(f"{curr_identity} – {force}")

        # plot options : splitview / all in one    
        if split_grips:
            optional_suffix += "_split"
            filtered_names_g1 = []
            filtered_g1 = []
            filtered_ct1 = []
            for name, g1, ct1 in zip(athlete_names, g1r_impulses, contact_times_g1):
                if g1 != 0 and ct1 and ct1 > 0:
                    filtered_names_g1.append(name)
                    filtered_g1.append(g1)
                    filtered_ct1.append(ct1)

            filtered_names_g2 = []
            filtered_g2 = []
            filtered_ct2 = []
            for name, g2, ct2 in zip(athlete_names, g2l_impulses, contact_times_g2):
                if g2 != 0 and ct2 and ct2 > 0:
                    filtered_names_g2.append(name)
                    filtered_g2.append(g2)
                    filtered_ct2.append(ct2)

            indices_g1 = np.arange(len(filtered_names_g1))
            indices_g2 = np.arange(len(filtered_names_g2))

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
            _plot_impulses_bar_split(ax1, ax2, indices_g1, filtered_names_g1, filtered_g1, filtered_ct1,
                                     indices_g2, filtered_names_g2, filtered_g2, filtered_ct2,
                                     force, title, direction, y_titel, margin, show_values)
        else:
            optional_suffix += "_combined"
            filtered_names = []
            filtered_g1 = []
            filtered_g2 = []
            filtered_ct1 = []
            filtered_ct2 = []
            for name, g1, g2, ct1, ct2 in zip(athlete_names, g1r_impulses, g2l_impulses, contact_times_g1, contact_times_g2):
                if (g1 != 0 and ct1 and ct1 > 0) or (g2 != 0 and ct2 and ct2 > 0):
                    filtered_names.append(name)
                    filtered_g1.append(g1 if g1 != 0 and ct1 and ct1 > 0 else 0)
                    filtered_g2.append(g2 if g2 != 0 and ct2 and ct2 > 0 else 0)
                    filtered_ct1.append(ct1 if g1 != 0 and ct1 and ct1 > 0 else None)
                    filtered_ct2.append(ct2 if g2 != 0 and ct2 and ct2 > 0 else None)
            indices = np.arange(len(filtered_names))
            fig, ax = plt.subplots(figsize=figsize)
            _plot_impulses_bar_combined(ax, indices, filtered_names, filtered_g1, filtered_g2, filtered_ct1,
                                        filtered_ct2, force, title, direction, y_titel, margin, show_values)

        plt.tight_layout()

        if split_grips:
            for axis in (ax1, ax2):
                apply_barplot_style(axis, margin=MARGIN)
        else:
            apply_barplot_style(ax, margin=MARGIN)
        if save_plot:
            filename = f"{title}{optional_suffix}.png"
            file_path = os.path.join(save_folder, filename)
            print(f"saving impulses bar plot to: {file_path}")
            fig.savefig(file_path)
        plt.show()


# === Helper functions for plot_impulses_bar ===
def _plot_impulses_bar_split(ax1, ax2, indices_g1, filtered_names_g1, filtered_g1, filtered_ct1,
                             indices_g2, filtered_names_g2, filtered_g2, filtered_ct2,
                             force, title, direction, y_title, margin, show_values):
    # Standardfarben
    bar_colors_g1 = ["deepskyblue"] * len(filtered_g1)
    bar_colors_g2 = ["coral"] * len(filtered_g2)


    bars1 = ax1.bar(indices_g2, filtered_g2, color=bar_colors_g2, label="GL")
    bars2 = ax2.bar(indices_g1, filtered_g1, color=bar_colors_g1, label="GR")
    
    # Achsenbeschriftungen
    ax1.set_title(f"{title} – {force} – GL")
    ax2.set_title(f"{title} – {force} – GR")
    ax2.set_ylabel(f"Impuls P{direction} {y_title}")
    ax1.set_xticks(indices_g2)
    ax1.set_xticklabels(filtered_names_g2, rotation=25, ha="right")
    ax2.set_xticks(indices_g1)
    ax2.set_xticklabels(filtered_names_g1, rotation=25, ha="right")

    # Gemeinsame y-Limits für beide Achsen setzen (auf Maximalbereich beider)
    min_g1 = min(filtered_g1) if filtered_g1 else 0
    max_g1 = max(filtered_g1) if filtered_g1 else 0
    min_g2 = min(filtered_g2) if filtered_g2 else 0
    max_g2 = max(filtered_g2) if filtered_g2 else 0
    global_min = min(min_g1, min_g2)
    global_max = max(max_g1, max_g2)
    y_range = global_max - global_min
    if y_range == 0:  # Alle Balken gleich oder leer
        global_max = global_max * margin if global_max != 0 else margin
        global_min = global_min * margin if global_min != 0 else 0
    else:
        buffer = y_range * (margin - 1) / 2
        global_min -= buffer
        global_max += buffer
    ax1.set_ylim(global_min, global_max)
    ax2.set_ylim(global_min, global_max)

    # Werte und Kontaktzeiten als kleine graue Texte direkt über den Balken anzeigen
    if show_values:
        fontsize = 6
        for bar, ct, g1 in zip(bars1, filtered_ct1, filtered_g1):
            txt = f"{g1:.1f}\n({ct:.1f}s)"
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), txt, ha="center", va="bottom", fontsize=fontsize, color="grey")
        for bar, ct, g2 in zip(bars2, filtered_ct2, filtered_g2):
            txt = f"{g2:.1f}\n({ct:.1f}s)"
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), txt, ha="center", va="bottom", fontsize=fontsize, color="grey")


def _plot_impulses_bar_combined(ax, indices, filtered_names, filtered_g1, filtered_g2, filtered_ct1,
                                filtered_ct2, force, title, direction, y_title, margin, show_values):
    width = 0.35
    # Erstelle Balken nebeneinander für G2L und G1R
    bars2 = []
    bars1 = []
    bar_colors_g2 = []
    bar_colors_g1 = []
    # Erzeuge Farbliste: grau falls 0, sonst Standardfarbe
    for g2, ct2 in zip(filtered_g2, filtered_ct2):
        if g2 == 0 or ct2 is None or ct2 == 0:
            bar_colors_g2.append("grey")
        else:
            bar_colors_g2.append("coral")
    for g1, ct1 in zip(filtered_g1, filtered_ct1):
        if g1 == 0 or ct1 is None or ct1 == 0:
            bar_colors_g1.append("grey")
        else:
            bar_colors_g1.append("deepskyblue")
            
    bars2 = ax.bar(indices - width/2, filtered_g2, width, color=bar_colors_g2, label="GL")
    bars1 = ax.bar(indices + width/2, filtered_g1, width, color=bar_colors_g1, label="GR")
    ax.set_xticks(indices)
    ax.set_xticklabels(filtered_names, rotation=25, ha="right")
    ax.set_ylabel(f"Impuls P{direction} {y_title}")
    ax.set_title(f"{title} – {force}")
    ax.legend()

    # Werte und Kontaktzeiten als kleine Texte direkt über den Balken anzeigen
    if show_values:
        fontsize = 6
        for i, (bar, ct, g2) in enumerate(zip(bars2, filtered_ct2, filtered_g2)):
            txt = f"{g2:.1f}"
            txt_ct = f"\n({ct:.1f}s)"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), txt + txt_ct, ha="center", va="bottom", fontsize=fontsize, color="grey")
        for i, (bar, ct, g1) in enumerate(zip(bars1, filtered_ct1, filtered_g1)):
            txt = f"{g1:.1f}"
            txt_ct = f"\n({ct:.1f}s)"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), txt + txt_ct, ha="center", va="bottom", fontsize=fontsize, color="grey")