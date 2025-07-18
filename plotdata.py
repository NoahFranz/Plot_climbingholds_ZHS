INTERVAL_SHADE_COLOR = "lightgrey"
import config
import matplotlib.pyplot as plt
import pandas as pd
from plotUITLS import (
    clean_label,
    compute_ylimits,
    apply_default_plot_style,
    save_figure_with_title,
    plot_mz_on_secondary_axis,
    combine_legends,
    plot_normal_forces,
)
from matplotlib.colors import to_rgb
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import os  # für Dateisystemoperationen
from utils import print_nested_keys
import json






def annotate_impulses_on_axis(ax: plt.Axes, side_data: Dict[str, Any], forces: List[str], labeloffset: float = 0.0):
    impulses_dict = side_data.get("impulses", {})
    contact_time = side_data.get("contact_time", {})
    print("forces in annotate_impulses_on_axis: ", forces)

    # Kontaktzeit nur einmal pro Intervall anzeigen – z. B. von 'Fz'
    primary_force = None
    for f in ["Fz", "Fy", "Fx"]:
        if f in contact_time:
            primary_force = f
            break

    if primary_force:
        ivals = contact_time.get(primary_force, [])
        for (t0, t1) in ivals:
            ct = t1 - t0
            x_pos = t1 + 0.1
            ylims = ax.get_ylim()
            y_pos_ct = ylims[0] + 0.85 * (ylims[1] - ylims[0])

            # position Contct time text
            ax.text(x_pos, y_pos_ct, f"{ct:.1f}s", color="grey", ha="left", va="center", fontsize=6)

    for force in forces:
        imp_list = impulses_dict.get(force, [])
        ivals = contact_time.get(force, [])

        for i, ((imp, (t0, t1))) in enumerate(zip(imp_list, ivals)):
            try:
                imp_val = float(imp)
                if abs(imp_val) > 0 and t1 > t0:
                    x_pos = t1 + 0.1
                    ylims = ax.get_ylim()
                    y_pos = ylims[0] + 0.82 * (ylims[1] - ylims[0]) - labeloffset
                    txt = f"{force.replace('F', 'P', 1).replace('res_', '')}: {imp_val:.1f}"
                    ax.text(x_pos, y_pos, txt, color="grey", ha="left", va="center", fontsize=6)
            except Exception as e:
                logger.warning(f"Fehler beim Anzeigen von Impuls {force}: {e}")

        labeloffset += 3  # Offset für die nächste Kraft erhöhen

    # Separate Anzeige für Fres_xyz, falls vorhanden
    intervals = side_data.get("intervals", {})
    for key, val in intervals.items():
        fres_data = val.get("Fres_xyz", {})
        timing = val.get("interval_timing", None)
        if fres_data and timing and "impuls" in fres_data:
            try:
                imp_val = float(fres_data["impuls"])
                t1 = timing[1]
                x_pos = t1 + 0.1
                ylims = ax.get_ylim()
                y_pos = ylims[0] + 0.82 * (ylims[1] - ylims[0]) - labeloffset
                ax.text(x_pos, y_pos, f"Pxyz: {imp_val:.1f}", color="grey", ha="left", va="center", fontsize=6)
            except Exception as e:
                logger.warning(f"Fehler beim Anzeigen von Fres_xyz Impuls: {e}")



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
    "legend_ncol": 5,
    "single_figsize": (8, 4.5)
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
    Wählt eine Farbe für 'force' und 'hold' basierend auf config.COLOR_MAPPING.
    Für G2L wird die Basisfarbe leicht angepasst.
    """
    # wählt je nach Griff (G2L vs. G1R) die passende Farbe aus
    base_key = next((f for f in config.COLOR_MAPPING if f in force), None)
    base_color = config.COLOR_MAPPING.get(base_key, "black")
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



def plot_single_hold_splitview(
    side_data: Dict[str, Any],
    forces: List[str],
    filename: str = "",
    grip_label: str = "",
    save_plot: bool = False,
    margin: float = 1.25,
    save_folder: str = ".",
    cutoff: Optional[Dict[str, float]] = None,
    normalizebyweight: bool = False,
    show_contact_time: bool = False,
    show_interval: bool = False,
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

        # Extract the DataFrame from the side_data dict
        hold_data = side_data["data"]

        # Zeitachse aus den Rohdaten extrahieren
        time = hold_data["Time [s]"]
        
        # Filtere Kraftspalten basierend auf GUI-Auswahl
        selected_forces = [f for f in forces if f != "Mz"]
        normal_cols = [col for col in hold_data.columns if any(f in col for f in selected_forces)]
        moment_cols = [col for col in hold_data.columns if "Mz" in col and "Mz" in forces]

        # Erstelle eine Figure mit 2 Zeilen und 1 Spalte; 
        # die Breite ist fix (6.3 Zoll, passend für LaTeX) und Höhe auf 8 Zoll gewählt.
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=PLOT_CONFIG["default_figsize"], sharex=True)
        file_identity = side_data.get("file_identity", filename)
        if config.show_title_in_plots:
            fig.suptitle(file_identity, fontsize=12)
        
        # Untere bzw. obere Achsen einrichten und Daten plotten
        for col in normal_cols:
            curret_forcen = next((f for f in config.COLOR_MAPPING if f in col), None)
            ax_top.plot(time, hold_data[col], label=clean_label(col), color=config.COLOR_MAPPING.get(curret_forcen))
        if config.show_title_in_plots:
            ax_top.set_title("Kräfte – " + grip_label)
        if normal_cols:
            data_subset = hold_data[normal_cols].dropna()
            ax_top.set_ylim(compute_ylimits(data_subset, margin=margin, fallback=(-100, 1000)))
        # Optional: Kontaktzeiten als halbtransparente Flächen markieren (nur für die Kontaktzeiten der ersten Kraft)
        if show_interval:
            contact_time = side_data.get("contact_time", {})
            if forces:
                first_force = forces[0]
                ivals = side_data.get("contact_time", {}).get(first_force, [])
                for (t0, t1) in ivals:
                    ax_top.axvspan(t0, t1, color=INTERVAL_SHADE_COLOR, alpha=0.15)
        # Füge eine Legende hinzu (innerhalb des Plots)
        ax_top.legend(loc="upper right", ncol=PLOT_CONFIG["legend_ncol"])
        # Nach Setzen der Achsenlimits: Optional Kontaktzeit annotieren
        if show_contact_time:
            annotate_impulses_on_axis(ax_top, side_data, forces)

        for col in moment_cols:
            ax_bottom.plot(time, hold_data[col], label=clean_label(col), color=config.COLOR_MAPPING["Mz"])
        if config.show_title_in_plots:
            ax_bottom.set_title("Moment – " + grip_label)
        ax_bottom.set_xlabel("Time [s]")
        ax_bottom.set_ylabel("Mz [Nm]")
        if moment_cols:
            data_subset = hold_data[moment_cols].dropna()
            y_min, y_max = compute_ylimits(data_subset, margin=margin, fallback=(-10, 10))
            ax_bottom.set_ylim([y_min, y_max])
            # Add interval shading for Mz if show_interval is True
            if show_interval:
                contact_time = side_data.get("contact_time", {})
                if "Mz" in contact_time:
                    ivals = contact_time["Mz"]
                    for (t0, t1) in ivals:
                        ax_bottom.axvspan(t0, t1, color=INTERVAL_SHADE_COLOR, alpha=0.15)
        ax_bottom.legend(loc="upper right", ncol=PLOT_CONFIG["legend_ncol"])
        
        # Beschriftungen und Achsenbegrenzungen setzen
        # Wende Plot-Stil ganz am Ende an, damit alle Änderungen übernommen werden
        apply_default_plot_style(fig, normalizebyweight=normalizebyweight)
        if save_plot:
            save_figure_with_title(fig, filename + config.optional_suffix, grip_label, save_plot=save_plot, figstyle=figstyle, save_folder=save_folder)
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
    show_interval: bool = False,
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
    #print("     current file_dict:")
    #print_nested_keys(plot_dict, 1)

#   # Remove "Fres_xyz" from forces_g1 and forces_g2 if it exists
#   if "Fres_xyz" in forces_g1:
#       forces_g1.remove("Fres_xyz")
#   if "Fres_xyz" in forces_g2:
#       forces_g2.remove("Fres_xyz")
#
    try:
        figstyle = "2G"
        grip_label = "OL_UR"

        # Farbzuordnungen für die verschiedenen Krafttypen (inkl. neu hinzugefügter)
        color_mapping = config.COLOR_MAPPING
        
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
        if config.show_title_in_plots:
            fig.suptitle(file_identity, fontsize=12)

        # --- Plot für linken Griff (G2L) ---
        if only_mz_g2:
            mz_cols = [col for col in plot_dict["G2L"]["data"].columns if "Mz" in col]
            ax_left = axes[0]
            time_left = plot_dict["G2L"]["data"]["Time [s]"]
            plot_mz_on_secondary_axis(ax_left, time_left, plot_dict["G2L"]["data"], mz_cols)
            if config.show_title_in_plots:
                ax_left.set_title("GL")
            ax_left.set_xlabel("Time [s]")
            ax_left.set_ylabel("Mz [Nm]")
            combine_legends(ax_left, None, loc="upper left", ncol=PLOT_CONFIG["legend_ncol"])
            apply_default_plot_style(fig, normalizebyweight=normalizebyweight)
        else:
            ax_left = axes[0]
            time_left = plot_dict["G2L"]["data"]["Time [s]"]
            # Falls nur Moment gewünscht: nur Mz darstellen
            plot_normal_forces(ax_left, plot_dict["G2L"]["data"], forces_g2, color_mapping)
            ax_left.set_title("GL")
            # Normalkräfte plotten und y-Grenzen berechnen
            data_left = plot_dict["G2L"]["data"][[col for col in plot_dict["G2L"]["data"].columns if any(f in col for f in forces_g2)]]
            if "global_y_limits" in plot_dict.get("G1R", {}) or "global_y_limits" in plot_dict.get("G2L", {}):
                global_limits = plot_dict.get("G2L", {}).get("global_y_limits") or plot_dict.get("G1R", {}).get("global_y_limits")
                y_min_left = global_limits["global_y_min"] * 1.2
                y_max_left = global_limits["global_y_max"] * 1.2
                try:
                    ax_left.set_ylim([y_min_left, y_max_left])
                except Exception as e:
                    logger.warning(f"Fehler beim Setzen der y-Limits für G2L: {e}")
                    ax_left.set_ylim([-5, 60])  # Fallback-Werte
            else:
                # Robust logic to guard against NaN or Inf in y-limits
                if not data_left.empty and data_left.notna().any().any():
                    y_min_left, y_max_left = compute_ylimits(data_left, margin=margin, fallback=(-100, 1000))
                    if not (np.isfinite(y_min_left) and np.isfinite(y_max_left)):
                        logger.warning("Computed y-limits contain NaN or Inf, skipping set_ylim.")
                    else:
                        ax_left.set_ylim([y_min_left, y_max_left])
                else:
                    logger.warning("No valid data in data_left for G2L; skipping ylim setting.")
            
            
            # Optional: Momente auf Sekundärachse anzeigen
            mz_cols = [col for col in plot_dict["G2L"]["data"].columns if "Mz" in col]
            sec_ax_left = None
            if mz_cols and "Mz" in forces_g2:
                sec_ax_left = ax_left.twinx()
                plot_mz_on_secondary_axis(sec_ax_left, time_left, plot_dict["G2L"]["data"], mz_cols)
                mz_df_left = plot_dict["G2L"]["data"][mz_cols]
                y_min_mz_left, y_max_mz_left = compute_ylimits(mz_df_left, margin=margin)
                sec_ax_left.set_ylim([y_min_mz_left, y_max_mz_left])
           
            # Optional: Kontaktzeiten als halbtransparente Flächen markieren (nur für die Kontaktzeiten der ersten Kraft)
            if show_interval:
                contact_time = plot_dict["G2L"].get("contact_time", {})
                if forces_g2:
                    first_force = forces_g2[0]
                    ivals = contact_time.get(first_force, [])
                    for (t0, t1) in ivals:
                        ax_left.axvspan(t0, t1,
                                        color=INTERVAL_SHADE_COLOR,
                                        alpha=0.15)

            # Alle Impulse pro Intervall anzeigen und Kontaktzeit nach letztem Impuls annotieren
            if show_contact_time:
                annotate_impulses_on_axis(ax_left, plot_dict["G2L"], forces_g2)

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

                if config.show_title_in_plots:
                    ax_right.set_title("GR")
                ax_right.set_xlabel("Time [s]")

                # Nach Plotten der Normalkräfte: lokale y-Limits berechnen
                data_right = plot_dict["G1R"]["data"][[col for col in plot_dict["G1R"]["data"].columns if any(f in col for f in forces_g1)]]
                y_min_right, y_max_right = compute_ylimits(data_right, margin=margin)
                y_min_right = plot_dict["G1R"]["global_y_limits"]["global_y_min"]*1.2
                y_max_right = plot_dict["G1R"]["global_y_limits"]["global_y_max"]*1.2

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
                if show_interval:
                    contact_time = plot_dict["G1R"].get("contact_time", {})
                    if forces_g1:
                        first_force = forces_g1[0]
                        ivals = contact_time.get(first_force, [])
                        for (t0, t1) in ivals:
                            ax_right.axvspan(t0, t1,
                                             color=INTERVAL_SHADE_COLOR,
                                             alpha=0.15)

                if show_contact_time:
                    annotate_impulses_on_axis(ax_right, plot_dict["G1R"], forces_g1, labeloffset=2)
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
    show_contact_time: bool = False,
    show_interval: bool = False,
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
        if config.show_title_in_plots:
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
                _plot_force_lines(ax, df, [current_force], config.COLOR_MAPPING, alpha=alpha, suffix=suffix)
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
        try:
            if show_contact_time:
                contact_time_g1 = file_dict["G1R"].get("contact_time", {})
                contact_time_g2 = file_dict["G2L"].get("contact_time", {})
                force_top = allForcesList[0]
                if show_interval and force_top in contact_time_g1:
                    t0, t1 = contact_time_g1[force_top]
                    axes[0].axvspan(t0, t1, color=get_color_for(force_top, "G1R"), alpha=0.15)
                if len(allForcesList) > 1:
                    force_bot = allForcesList[1]
                    if show_interval and force_bot in contact_time_g2:
                        t0, t1 = contact_time_g2[force_bot]
                        axes[1].axvspan(t0, t1, color=get_color_for(force_bot, "G2L"), alpha=0.15)
        except Exception as e:
            logger.warning(f"Fehler in show_contact_time Block: {e}")
        # Legenden, Achsentitel, Stil setzen und Plot speichern/anzeigen
        if config.show_title_in_plots:
            axes[0].set_title(f"Vergleich: {filename}")
        axes[0].set_xlabel("Time [s]")
        axes[0].legend(ncol=PLOT_CONFIG["legend_ncol"])
        axes[1].set_xlabel("Time [s]")
        axes[1].legend(ncol=PLOT_CONFIG["legend_ncol"])
        apply_default_plot_style(fig=fig, normalizebyweight=normalizebyweight)
        if save_plot:
            # Ensure target directory exists before saving
            full_path = f"{save_folder}/{filename}{config.optional_suffix}.png"
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            plt.savefig(full_path)
            print(f"Plot gespeichert unter: {full_path}")
        plt.show()
    except Exception as e:
        # Ausnahmebehandlung: Bei Fehlern in der Plot-Erstellung ausgeben, aber Programm nicht abbrechen
        logger.exception(f"plot_selected_forces_comparison failed for {filename}: {e}")



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
    if config.show_title_in_plots:
        ax1.set_title(f"{title} – {force} – GL")
        ax2.set_title(f"{title} – {force} – GR")
    ax2.set_ylabel(f"Impuls P{direction} {y_title}")
    ax1.set_xticks(indices_g2)
    ax1.set_xticklabels([config.file_acronyms_map.get(name, name) for name in filtered_names_g2], rotation=25, ha="right")
    ax2.set_xticks(indices_g1)
    ax2.set_xticklabels([config.file_acronyms_map.get(name, name) for name in filtered_names_g1], rotation=25, ha="right")

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
        for bar, ct, G1R in zip(bars1, filtered_ct1, filtered_g1):
            txt = f"{G1R:.1f}\n({ct:.1f}s)"
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), txt, ha="center", va="bottom", fontsize=fontsize, color="grey")
        for bar, ct, G2L in zip(bars2, filtered_ct2, filtered_g2):
            txt = f"{G2L:.1f}\n({ct:.1f}s)"
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
    for G2L, ct2 in zip(filtered_g2, filtered_ct2):
        if G2L == 0 or ct2 is None or ct2 == 0:
            bar_colors_g2.append("grey")
        else:
            bar_colors_g2.append("coral")
    for G1R, ct1 in zip(filtered_g1, filtered_ct1):
        if G1R == 0 or ct1 is None or ct1 == 0:
            bar_colors_g1.append("grey")
        else:
            bar_colors_g1.append("deepskyblue")
            
    bars2 = ax.bar(indices - width/2, filtered_g2, width, color=bar_colors_g2, label="GL")
    bars1 = ax.bar(indices + width/2, filtered_g1, width, color=bar_colors_g1, label="GR")
    ax.set_xticks(indices)
    ax.set_xticklabels([config.file_acronyms_map.get(name, name) for name in filtered_names], rotation=25, ha="right")
    ax.set_ylabel(f"Impuls P{direction} {y_title}")
    if config.show_title_in_plots:
        ax.set_title(f"{title} – {force}")
    ax.legend()

    # Werte und Kontaktzeiten als kleine Texte direkt über den Balken anzeigen
    if show_values:
        fontsize = 6
        for i, (bar, ct, G2L) in enumerate(zip(bars2, filtered_ct2, filtered_g2)):
            txt = f"{G2L:.1f}"
            txt_ct = f"\n({ct:.1f}s)"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), txt + txt_ct, ha="center", va="bottom", fontsize=fontsize, color="grey")
        for i, (bar, ct, G1R) in enumerate(zip(bars1, filtered_ct1, filtered_g1)):
            txt = f"{G1R:.1f}"
            txt_ct = f"\n({ct:.1f}s)"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), txt + txt_ct, ha="center", va="bottom", fontsize=fontsize, color="grey")


def plot_mean_metrics_bar(
    all_lvm_data_dict: Dict[str, Any],
    forces: List[str],
    metric: str = "impuls",  # "mean", "max", "min" etc.
    side: str = "G2L",
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Mean",
    save_plot: bool = False,
    save_folder: str = ".",
    normalizebyweight= bool(True),
    split_view= False,  # True: separate plots for G1R and G2L
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    if forces is None:
        forces = []

    # Map GUI-friendly names to data keys depending on side
    mapped_forces = []
    for f in forces:
        if f == "FgR":
            if side == "G2L":
                mapped_forces.append("FgR_2")
            elif side == "G1R":
                mapped_forces.append("FgR_1")
        else:
            mapped_forces.append(f)

    labels, plot_data = [], []

    is_contact_time = metric.lower() == "contact time" or "Contacttime" in metric
    # optinol for choosing order of files.
    if config.use_custom_bar_order:
        ordered_keys = []
        if config.plot_only_wooden_holds:
            holds_to_plot = config.order_for_wooden_holds
        else:
            holds_to_plot = config.file_order_bar_holds
        for prefix in holds_to_plot:
            match = next((k for k in all_lvm_data_dict if k.startswith(prefix)), None)
            if match:
                ordered_keys.append(match)
    else:
        ordered_keys = list(all_lvm_data_dict.keys())
    for fname in ordered_keys:
        content = all_lvm_data_dict[fname]
        mean_metrics = content.get(side, {}).get("intervals", {}).get("Mean-Metrics", {})
        if not mean_metrics:
            print(f"⚠️ Kein 'Mean-Metrics' für {side} in Datei: {fname}")
            print(json.dumps(content.get(side, {}).get("intervals", {}), indent=2))
            continue

        if is_contact_time:
            val_entry = mean_metrics.get("Contacttime", {})
            val = val_entry.get("mean", np.nan)
            plot_data.append([val])
            file_number = fname[:3]
            labels.append(config.file_acronyms_map.get(file_number, file_number))
        else:
            row = []
            for force in mapped_forces:
                val_entry = mean_metrics.get(force, None)
                if isinstance(val_entry, dict) and metric in val_entry:
                    val = val_entry[metric]
                elif isinstance(val_entry, (int, float)) and metric == "mean":
                    val = val_entry
                else:
                    val = None
                row.append(val if val is not None else np.nan)
            plot_data.append(row)
            file_number = fname[:3]
            labels.append(config.file_acronyms_map.get(file_number, file_number))

    # Check if plot_data is empty before proceeding
    if not plot_data:
        print(f"Keine gültigen Daten für {side}, Diagramm wird übersprungen.")
        return

    plot_data = np.array(plot_data)
    if plot_data.ndim == 1:
        plot_data = plot_data[:, np.newaxis]
    x = np.arange(len(labels))
    width = 0.8 / plot_data.shape[1]

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(plot_data.shape[1]):
        if metric.lower() == "impuls":
            force_label = forces[i]
            base_key = force_label.replace("F", "P", 1)
            color_key = force_label  # use F* as color source
        elif is_contact_time:
            base_key = r"$t{\mathrm{contact}}$"
            color_key = base_key
        else:
            base_key = forces[i]
            color_key = base_key
        color = config.COLOR_MAPPING.get(color_key, "black")
        ax.bar(x + i * width, plot_data[:, i], width, label=base_key, color=color)

    # Dynamischer Y-Achsentitel je nach Metrik
    metric_lower = metric.lower()
    if normalizebyweight:
        if metric_lower == "impuls":
            ylabel = r"P [%BWs]"
        elif metric_lower == "maxrofd":
            ylabel = r"∆F / ∆t [%BW/s]"
        elif metric_lower in ["contact time", "contacttime"]:
            ylabel = "Time [s]"
        elif metric_lower =="max":
            ylabel = r"F [%BW]"
        elif metric_lower =="hausdorff":
            ylabel = "HD"
        else:
            ylabel = r"F [%BW]"
            

    else:
        if metric_lower == "impuls":
            ylabel = "P [Ns]"
        elif metric_lower == "maxrofd":
            ylabel = "∆F / ∆t [N/s]"
        elif metric_lower in ["contact time", "contacttime"]:
            ylabel = "Time [s]"
        elif metric_lower =="max":
            ylabel = "F [N]"
        elif metric_lower =="hausdorff":
            ylabel = "HD"
        else:
            ylabel = "F [N]"

    ax.set_ylabel(ylabel)
    if config.show_title_in_plots:
        ax.set_title(f"{title} ({side})")
    if config.manual_y_limits_var:
        y_min = config.plot_settings["y_limits"][0]
        y_max = config.plot_settings["y_limits"][1]
     
    else:
        y_min = plot_data.min().min()/1.05
        y_max = plot_data.max().max()*1.1
    y_min = 0
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25) 
    ax.set_ylim([y_min,y_max])
    ax.legend(loc='upper right', ncol=PLOT_CONFIG["legend_ncol"])
    plt.tight_layout()
    if save_plot:
        safe_title = title.replace(":", "_").replace(" ", "_")
        path = os.path.join(save_folder, f"{safe_title}_{side}{config.optional_suffix}.png")
        # Ensure target directory exists before saving
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f"Plot gespeichert unter: {path}")
    
    plt.show()





def plot_force_vector_trace(
    df: pd.DataFrame,
    forces: Tuple[str, str] = ("Fy", "Fz"),
    step: int = 10,
    title: str = "Resultierende Kraftvektoren (2D)",
    save_plot: bool = False,
    save_folder: str = ".",
    filename: str = "force_vector_trace",
    intervals: Optional[List[Tuple[float, float]]] = None,
    plot_vector_interval_only: bool = False,
) -> None:
    """
    Visualisiert die Richtung der resultierenden Kraft in der YZ-Ebene über die Zeit als Pfeile.

    Parameters:
        df: DataFrame mit den Spalten "Fy", "Fz" und "Time [s]"
        forces: Tupel der zu plottenden Kraftkomponenten, standardmäßig ("Fy", "Fz")
        step: Abtastrate zur Darstellung (nicht jeden Zeitpunkt plotten)
        title: Titel des Plots
        save_plot: Falls True, speichert den Plot
        save_folder: Zielverzeichnis
        filename: Basisname für den Plot
    """
    fy_col, fz_col = forces
    time = df["Time [s]"]
    # Fuzzy matching for force columns (e.g., matches "Fy_1  [N]" etc.)
    fy_matches = [col for col in df.columns if fy_col in col]
    fz_matches = [col for col in df.columns if fz_col in col]

    if not fy_matches or not fz_matches:
        print(f"Spalten mit '{fy_col}' oder '{fz_col}' nicht gefunden.")
        return

    # Optionally restrict vector plotting to intervals only
    if plot_vector_interval_only and intervals:
        valid_mask = pd.Series(False, index=df.index)
        for (t0, t1) in intervals:
            valid_mask |= (df["Time [s]"] >= t0) & (df["Time [s]"] <= t1)
        df = df[valid_mask]
    elif intervals:
        valid_mask = pd.Series(False, index=df.index)
        for (t0, t1) in intervals:
            valid_mask |= (df["Time [s]"] >= t0) & (df["Time [s]"] <= t1)
        df = df[valid_mask]

    fy = df[fy_matches[0]]
    fz = df[fz_matches[0]]

    fig, ax = plt.subplots(figsize=(6, 6))
    # Rotate vectors by 40 degrees (wall tilt)
    theta = np.radians(40)
    fy_rot = np.cos(theta) * fy - np.sin(theta) * fz
    fz_rot = np.sin(theta) * fy + np.cos(theta) * fz

    from matplotlib.cm import get_cmap
    cmap = get_cmap("viridis")
    norm = np.linspace(0, 1, len(df[::step]))

    for i, j in enumerate(range(0, len(df), step)):
        color = cmap(norm[i])
        ax.arrow(0, 0, fz_rot.iloc[j], fy_rot.iloc[j],
                 head_width=0.05, head_length=0.1,
                 alpha=0.8, color="red", length_includes_head=True)

    ax.set_xlabel("Fz [%BW]", fontsize=12, color="darkred")
    ax.set_ylabel("Fy [%BW]", fontsize=12, color="darkred")
    if config.show_title_in_plots:
        ax.set_title("Vektore", fontsize=14)
    ax.grid(True)
    ax.set_aspect('equal')

    if save_plot:
        path = os.path.join(save_folder, f"{filename}{config.config.optional_suffix}.png")
        # Ensure target directory exists before saving
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f"Plot gespeichert unter: {path}")
    plt.show()
def plot_FgR_sum(
    file_dict: Dict[str, Any],
    forces_g1: Optional[List[str]] = None,
    forces_g2: Optional[List[str]] = None,
    filename: str = "",
    grip_label: str = "",
    save_plot: bool = False,
    save_folder: str = ".",
    normalizebyweight: bool = False
) -> None:
    """
    Plottet die Spalte 'FgR_sum [%]' aus 'total_df' sowie optional Kräfte aus G1R und G2L.
    """
    try:
        df = file_dict.get("total_df", None)
        if df is None or "FgR_sum [%]" not in df.columns:
            print(f"Keine gültige 'FgR_sum [%]'-Spalte in Datei {filename}")
            return

        fig, ax = plt.subplots(figsize=PLOT_CONFIG["single_figsize"])
        print(f"Figure size (inches): {fig.get_size_inches()}")
        time = df["Time [s]"]
        
        # FgR_sum immer plotten
        ax.plot(time, df["FgR_sum [%]"], label="FgR_sum", color=config.COLOR_MAPPING.get("FgR_sum", "black"), linewidth=1.2)

        # Optional: Weitere Kräfte aus G1R und G2L
        for side, forces in [("G1R", forces_g1), ("G2L", forces_g2)]:
            if not forces:
                continue
            forces = [f for f in forces if f != "FgR_sum"]

            side_df = file_dict.get(side, {}).get("data", None)
            if side_df is None:
                continue
            for force in forces:
                cols = [col for col in side_df.columns if force in col and "FgR_sum" not in col]
                for col in cols:
                    # Prüfe auf FgR-Krafttyp für Farblogik
                    if "FgR" in col:
                        color = config.COLOR_MAPPING.get("FgR_1" if side == "G1R" else "FgR_2", "grey")
                    else:
                        color = config.COLOR_MAPPING.get(force, "grey")
                    ax.plot(
                        side_df["Time [s]"],
                        side_df[col],
                        label=f"{clean_label(col)} ({'R' if side == 'G1R' else 'L'})",
                        color=color,
                        alpha=0.7
                    )
        # Nach allen ax.plot(...): Doppelte Legenden vermeiden
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), ncol=PLOT_CONFIG["legend_ncol"])

        file_identity = file_dict.get("file_identity", filename)
        if config.show_title_in_plots:
            ax.set_title(f"{file_identity}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Kraft [%BW]")
        apply_default_plot_style(fig, normalizebyweight=normalizebyweight)

        # Setze Y-Achse explizit auf 0–110 % (am Ende, direkt vor plt.show())
        ax.set_ylim([0, 110])

        if save_plot:
            save_figure_with_title(fig, filename + config.optional_suffix, grip_label, save_plot=save_plot, figstyle="FgR_sum", save_folder=save_folder)
        plt.show()

    except Exception as e:
        logger.exception(f"plot_FgR_sum failed for {filename} / {grip_label}: {e}")