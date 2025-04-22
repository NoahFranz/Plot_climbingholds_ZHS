import matplotlib.pyplot as plt
import pandas as pd
import math
from plotUITLS import*
from matplotlib.colors import to_rgb

def adjust_color(color, shift=0.2):
    r, g, b = to_rgb(color)
    r = max(0, min(1, r - shift))
    g = max(0, min(1, g + shift / 2))
    b = max(0, min(1, b + shift))
    return (r, g, b)
NCOL = 5
DEFAULT_FIGSIZE = (6.3, 8)
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

def plot_single_hold_splitview(hold_data, forces, filename="", grip_label="", save_plot=False, margin=1.25, save_folder=".", cutoff=None):
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
    figstyle = "1G_split"

    # Extrahiere die Zeitdaten aus der "Time [s]" Spalte
    time = hold_data["Time [s]"]
    
    # Filtere Kraftspalten basierend auf GUI-Auswahl
    selected_forces = [f for f in forces if f != "Mz"]
    normal_cols = [col for col in hold_data.columns if any(f in col for f in selected_forces)]
    moment_cols = [col for col in hold_data.columns if "Mz" in col and "Mz" in forces]

    # Erstelle eine Figure mit 2 Zeilen und 1 Spalte; 
    # die Breite ist fix (6.3 Zoll, passend für LaTeX) und Höhe auf 8 Zoll gewählt.
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=DEFAULT_FIGSIZE, sharex=True)
    
    # --- Oberer Subplot: Normalkräfte ---
    for col in normal_cols:
        curret_forcen = next((f for f in COLOR_MAPPING if f in col), None)
        ax_top.plot(time, hold_data[col], label=clean_label(col), color=COLOR_MAPPING.get(curret_forcen))
    ax_top.set_title(f"Kräfte – {grip_label}")
    ax_top.set_ylabel("F [%]" if only_fgr_in_plot(forces) else "F [N]")
    if normal_cols:
        data_subset = hold_data[normal_cols].dropna()
        ax_top.set_ylim(compute_ylimits(data_subset, margin=margin, fallback=(-100, 1000)))
    # Füge eine Legende hinzu (innerhalb des Plots)
    ax_top.legend(loc="upper right", ncol=NCOL)
    
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
    ax_bottom.legend(loc="upper right", ncol=NCOL)
    
    # Setze den eindeutigen Titel und speichere optional
    apply_default_plot_style(fig)
    if save_plot:
        save_figure_with_title(fig, filename, grip_label, save_plot=save_plot, figstyle=figstyle, save_folder=save_folder)

# =================================== plot_data_per_hold ====================================

def plot_data_per_hold(plot_dict, forces_g1, forces_g2, filename, save_plot=False, margin=1.2, save_folder=".", cutoff=None):
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
    figstyle = "2G"
    grip_label = "OL_UR"

    # Farbmapping für die Kräfte: Zuordnung von Kraftnamen zu Farben
    color_mapping = COLOR_MAPPING
    
    # Prüfe, ob ausschließlich der Momentenwert "Mz" ausgewählt wurde für jeden Griff
    only_mz_g1 = forces_g1 == ["Mz"]
    only_mz_g2 = forces_g2 == ["Mz"]

    # Lade globale Y-Grenzen
    y_min_global = plot_dict["G2L"]["global_limits"]["global_y_min"] * margin if plot_dict["G2L"]["global_limits"]["global_y_min"] < 0 else plot_dict["G2L"]["global_limits"]["global_y_min"] / margin
    y_max_global = plot_dict["G2L"]["global_limits"]["global_y_max"] * margin if plot_dict["G2L"]["global_limits"]["global_y_max"] > 0 else plot_dict["G2L"]["global_limits"]["global_y_max"] / margin

    # Überprüfe, ob für einen Griff überhaupt Kräfte vorhanden sind (um Achsen zu bestimmen)
    has_g1 = bool(forces_g1)
    has_g2 = bool(forces_g2)
    num_axes = has_g1 + has_g2

    # Erstelle die Figure mit passender Höhe basierend auf der Anzahl der Plots
    fig_height = 4 * num_axes
    fig, axes = plt.subplots(num_axes, 1, figsize=(6.3, fig_height), sharex=True)
    axes = [axes] if num_axes == 1 else axes

    # Setze die Achsenlimits für beide Subplots
    axes[0].set_ylim([y_min_global, y_max_global])
    if has_g1:
        axes[1].set_ylim([y_min_global, y_max_global])

    # ----- Linker Griff (G2L) -----
    if only_mz_g2:
        # Falls nur Mz ausgewählt ist, plotte nur Mz auf der primären Achse
        mz_cols = [col for col in plot_dict["G2L"]["data"].columns if "Mz" in col]
        ax_left = axes[0]
        time_left = plot_dict["G2L"]["data"]["Time [s]"]
        plot_mz_on_secondary_axis(ax_left, time_left, plot_dict["G2L"]["data"], mz_cols)
        ax_left.set_ylabel("Mz [Nm]")
        # Hole und setze die Legende
        combine_legends(ax_left, None, loc="upper left", ncol=NCOL)
    else:
        # Plotte die normalen Kräfte (außer Mz) auf dem oberen Plot
        ax_left = axes[0]
        time_left = plot_dict["G2L"]["data"]["Time [s]"]
        plot_normal_forces(ax_left, plot_dict["G2L"]["data"], forces_g2, color_mapping)
        ax_left.set_title("GL")
        ax_left.set_ylabel("F [%]" if only_fgr_in_plot(forces_g2) else "F [N]")
        ax_left.set_ylim([y_min_global, y_max_global])
        # Falls Mz ebenfalls aktiv ist, erstelle eine Sekundäxe und plotte Mz
        mz_cols = [col for col in plot_dict["G2L"]["data"].columns if "Mz" in col]
        sec_ax_left = None
        if mz_cols and "Mz" in forces_g2:
            sec_ax_left = ax_left.twinx()
            plot_mz_on_secondary_axis(sec_ax_left, time_left, plot_dict["G2L"]["data"], mz_cols)
        # Kombiniere Legenden von primärer und sekundärer Achse
        combine_legends(ax_left, sec_ax_left, loc="upper left", ncol=NCOL)
    
    # ----- Rechter Griff (G1R) -----
    if has_g1:
        if only_mz_g1:
            # Falls für den rechten Griff ausschließlich Mz ausgewählt wurde, plotte nur Mz
            mz_cols = [col for col in plot_dict["G1R"]["data"].columns if "Mz" in col]
            ax_right = axes[1]
            time_right = plot_dict["G1R"]["data"]["Time [s]"]
            plot_mz_on_secondary_axis(ax_right, time_right, plot_dict["G1R"]["data"], mz_cols)
            ax_right.set_ylabel("Mz [Nm]")
            combine_legends(ax_right, None, loc="upper left", ncol=NCOL)
        else:
            # Plotte normale Kräfte (außer Mz) für den rechten Griff
            ax_right = axes[1]
            time_right = plot_dict["G1R"]["data"]["Time [s]"]
            plot_normal_forces(ax_right, plot_dict["G1R"]["data"], forces_g1, color_mapping)
            ax_right.set_title("GR")
            ax_right.set_xlabel("Time [s]")
            ax_right.set_ylabel("F [%]" if only_fgr_in_plot(forces_g1) else "F [N]")
            ax_right.set_ylim([y_min_global, y_max_global])
            # Falls Mz aktiv ist, plotte zusätzlich Mz auf einer Sekundärachse
            mz_cols = [col for col in plot_dict["G1R"]["data"].columns if "Mz" in col]
            sec_ax_right = None
            if mz_cols and "Mz" in forces_g1:
                sec_ax_right = ax_right.twinx()
                plot_mz_on_secondary_axis(sec_ax_right, time_right, plot_dict["G1R"]["data"], mz_cols)
            combine_legends(ax_right, sec_ax_right, loc="upper left", ncol=NCOL)
    
    # Dynamische Berechnung der Zeitachsen-Grenzen mit 10% Puffer
    time_min = min(time_left.min(), time_right.min()) if has_g1 else time_left.min()
    time_max = max(time_left.max(), time_right.max()) if has_g1 else time_left.max()
    time_range = time_max - time_min
    axes[0].set_xlim([time_min - 0.01 * time_range, time_max + 0.05 * time_range])
    if has_g1:
        axes[1].set_xlim([time_min - 0.01 * time_range, time_max + 0.05 * time_range])
    
    plt.tight_layout()
    apply_default_plot_style(fig)
    if save_plot:
        save_figure_with_title(fig, filename, grip_label, save_plot=save_plot, figstyle=figstyle, save_folder=save_folder)
    # plt.show()

def plot_selected_forces_comparison(file_dict, forces_g1, forces_g2, filename, save_folder=".", save_plot=False, margin=1.2, cutoff=None):
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
    time = file_dict["G1R"]["data"]["Time [s]"]
    holdNameList = {"G1R", "G2L"}

    # Liste mit den darzustellenden Kräften
    allForcesList = []
    for f in forces_g1 + forces_g2:
        if f not in allForcesList:
            allForcesList.append(f)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6.3, 8))
    axFlag = 0
    print("allForcesList in plot_Selected_force:", allForcesList)
    for current_force in allForcesList:
        for current_hold in holdNameList:
            legend_suffix = {"G1R": "_R", "G2L": "_L"}.get(current_hold, "unknown")
            curr_alpha = {"G1R": 1, "G2L": 0.5}.get(current_hold, 0.8)
            if current_force == "Fy":
                curr_color = "#1f77b4" if current_hold == "G1R" else "#17becf"
            elif current_force == "Fz":
                curr_color = "#ff7f0e" if current_hold == "G1R" else "#F79F0B"
            else:
                base_force = next((f for f in COLOR_MAPPING if f in current_force), None)
                base_color = COLOR_MAPPING.get(base_force, "black")
                curr_color = adjust_color(base_color, shift=0.15) if current_hold == "G2L" else base_color
            matching_cols = [col for col in file_dict[current_hold]["data"].columns if current_force in col]
            plot_force = file_dict[current_hold]["data"][matching_cols[0]]
            if axFlag == 0:
                ax_top.plot(time, plot_force, 
                        label=clean_label(current_force)+legend_suffix, 
                        color=curr_color, alpha = curr_alpha)
            if axFlag == 1:
                ax_bottom.plot(time, plot_force, 
                    label=clean_label(current_force)+legend_suffix, 
                    color=curr_color, alpha = curr_alpha)
            else:
                    print("all plots populated already, only 2 Forces are possible per plot")
        axFlag = 1
    
    # equal limits for both forces
    y_min_global = file_dict["G1R"]["global_limits"]["global_y_min"]*margin
    y_max_global = file_dict["G1R"]["global_limits"]["global_y_max"]*margin

    # Top plot axis
    ax_top.set_title(f"Vergleich: {filename}")
    ax_top.set_xlabel("Time [s]")
    ax_top.set_ylabel("F [%]" if only_fgr_in_plot(allForcesList) else "Kräfte [N]")
    ax_top.set_ylim([y_min_global, y_max_global])
    ax_top.legend(ncol=NCOL)

    # Bottom plot Axis
    ax_bottom.set_xlabel("Time [s]")
    ax_bottom.set_ylabel("F [%]" if only_fgr_in_plot(allForcesList) else "Kräfte [N]")
    ax_bottom.legend(ncol=NCOL)
    ax_bottom.set_ylim([y_min_global, y_max_global])
    
    apply_default_plot_style(fig=fig)
    if save_plot:
        plt.savefig(f"{save_folder}/{filename}.png")
    plt.show()