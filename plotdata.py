import matplotlib.pyplot as plt
import pandas as pd
import math
from plotUITLS import*

DEFAULT_FIGSIZE = (6.3, 8)
COLOR_MAPPING = {"Fy": "blue","Fx": "green", "Fz": "orange", "Mz": "#8B1A1A"}  # Kaminrot
# COLOR_MAPPING = {
    # "Fx": "#3498db",  # Blue
    # "Fy": "#e74c3c",  # Red
    # "Fz": "#2ecc71",  # Green
    # "Mz": "#9b59b6"   # Purple
# }

def plot_single_hold_splitview(hold_data, forces, filename="", grip_label="", save_plot=False, margin=1.25, save_folder="."):
    """
    Erstellt eine Figure für einen einzelnen Griff, in der:
      - Im oberen Subplot die Normalkräfte (alle Spalten außer "Time [s]" und solchen, die "Mz" enthalten) geplottet werden.
      - Im unteren Subplot der Moment (alle Spalten, die "Mz" enthalten) geplottet wird.
    
    Parameter:
      hold_data : pandas.DataFrame
          DataFrame, das mindestens die Spalte "Time [s]" und 
          weitere Kraftdaten (z.B. Fx, Fy, Fz) sowie Momentdaten ("Mz") enthält.
      forces : list
          Liste der ausgewählten Kräfte.
      filename : str
          Der Name der Datei für die Speicherung des Plots.
      grip_label : str
          Bezeichnung des Griffs für den Titel.
      save_plot : bool
          Wenn True, wird die erstellte Figure als PNG abgespeichert.
    
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
        force = next((f for f in COLOR_MAPPING if f in col), None)
        ax_top.plot(time, hold_data[col], label=clean_label(col), color=COLOR_MAPPING.get(force))
    ax_top.set_title(f"Kräfte – {grip_label}")
    ax_top.set_ylabel("F [N]")
    if normal_cols:
        data_subset = hold_data[normal_cols].dropna()
        ax_top.set_ylim(compute_ylimits(data_subset, margin=margin, fallback=(-100, 1000)))
    # Füge eine Legende hinzu (innerhalb des Plots)
    ax_top.legend(loc="upper right", ncol=4)
    
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
    ax_bottom.legend(loc="upper right", ncol=4)
    
    # Setze den eindeutigen Titel und speichere optional
    apply_default_plot_style(fig)
    if save_plot:
        save_figure_with_title(fig, filename, grip_label, save_plot=save_plot, figstyle=figstyle, save_folder=save_folder)





# =================================== plot_data_per_hold ====================================






def plot_data_per_hold(plot_dict, forces_g1, forces_g2, filename, save_plot=False, margin=1.2, save_folder="."):
    """
    Erstellt eine Figure mit separaten Subplots für den linken (G2L) und rechten Griff (G1R).
    
    Für jeden Griff wird unterschieden:
      - Falls ausschließlich "Mz" ausgewählt ist, werden nur die Mz-Daten geplottet.
      - Andernfalls werden die normalen Kräfte (außer Mz) geplottet,
        und falls Mz zusätzlich aktiv ist, auf einer Sekundärachse ergänzt.
    
    Anschließend werden die Legenden der primären und sekundären Achsen kombiniert.
    """
    figstyle = "both_G_UL"
    grip_label = "Left_right"

    # Farbmapping für die Kräfte: Zuordnung von Kraftnamen zu Farben
    color_mapping = COLOR_MAPPING
    
    # Prüfe, ob ausschließlich der Momentenwert "Mz" ausgewählt wurde für jeden Griff
    only_mz_g1 = forces_g1 == ["Mz"]
    only_mz_g2 = forces_g2 == ["Mz"]
    
    # Verwende vorab berechnete Statistiken aus plot_dict["G2L"]["stats"] und ["G1R"]["stats"]
    stats_g2 = plot_dict["G2L"]["stats"]
    stats_g1 = plot_dict["G1R"]["stats"]

    y_min = float("inf")
    y_max = float("-inf")

    for key, val in stats_g2.items():
        if "Mz" not in key and "FgR" not in key:
            y_min = min(y_min, val["min"])
            y_max = max(y_max, val["max"])

    for key, val in stats_g1.items():
        if "Mz" not in key and "FgR" not in key:
            y_min = min(y_min, val["min"])
            y_max = max(y_max, val["max"])

    if y_min == float('inf') or y_max == float('-inf'):
        y_min, y_max = 0, 600

    y_min = y_min * margin if y_min < 0 else y_min / margin
    y_max = y_max * margin if y_max > 0 else y_max / margin

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
        combine_legends(ax_left, None, loc="upper left", ncol=4)
    else:
        # Plotte die normalen Kräfte (außer Mz) auf dem oberen Plot
        ax_left = axes[0]
        time_left = plot_dict["G2L"]["data"]["Time [s]"]
        normal_y_min, normal_y_max = plot_normal_forces(ax_left, plot_dict["G2L"]["data"], forces_g2, color_mapping)
        ax_left.set_title("GL")
        ax_left.set_ylabel("F [N]")
        ax_left.set_ylim([y_min, y_max + 0.25 * abs(y_max)])
        # Falls Mz ebenfalls aktiv ist, erstelle eine Sekundärachse und plotte Mz
        mz_cols = [col for col in plot_dict["G2L"]["data"].columns if "Mz" in col]
        sec_ax_left = None
        if mz_cols and "Mz" in forces_g2:
            sec_ax_left = ax_left.twinx()
            plot_mz_on_secondary_axis(sec_ax_left, time_left, plot_dict["G2L"]["data"], mz_cols)
        # Kombiniere Legenden von primärer und sekundärer Achse
        combine_legends(ax_left, sec_ax_left, loc="upper left", ncol=4)
    
    # ----- Rechter Griff (G1R) -----
    if has_g1:
        if only_mz_g1:
            # Falls für den rechten Griff ausschließlich Mz ausgewählt wurde, plotte nur Mz
            mz_cols = [col for col in plot_dict["G1R"]["data"].columns if "Mz" in col]
            ax_right = axes[1]
            time_right = plot_dict["G1R"]["data"]["Time [s]"]
            plot_mz_on_secondary_axis(ax_right, time_right, plot_dict["G1R"]["data"], mz_cols)
            ax_right.set_ylabel("Mz [Nm]")
            combine_legends(ax_right, None, loc="upper left", ncol=4)
        else:
            # Plotte normale Kräfte (außer Mz) für den rechten Griff
            ax_right = axes[1]
            time_right = plot_dict["G1R"]["data"]["Time [s]"]
            y_min_g1, y_max_g1 = plot_normal_forces(ax_right, plot_dict["G1R"]["data"], forces_g1, color_mapping)
            ax_right.set_title("GR")
            ax_right.set_xlabel("Time [s]")
            ax_right.set_ylabel("F [N]")
            ax_right.set_ylim([y_min_g1, y_max_g1 + 0.25 * abs(y_max_g1)])
            # Falls Mz aktiv ist, plotte zusätzlich Mz auf einer Sekundärachse
            mz_cols = [col for col in plot_dict["G1R"]["data"].columns if "Mz" in col]
            sec_ax_right = None
            if mz_cols and "Mz" in forces_g1:
                sec_ax_right = ax_right.twinx()
                plot_mz_on_secondary_axis(sec_ax_right, time_right, plot_dict["G1R"]["data"], mz_cols)
            combine_legends(ax_right, sec_ax_right, loc="upper left", ncol=4)
    
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