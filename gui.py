import tkinter as tk
from tkinter import filedialog
import config



def run_gui():
    """
    Erzeugt eine GUI zur Auswahl der Optionen:
      - Ob die Plots gespeichert werden sollen
      - Ob die Plots erstellt werden sollen
      - Griff-Options: All, G1R, G2L
      - Force-Options:
            Für G1R: all, Fy, Fx, Fz, Mz, FgR
            Für G2L: all, Fy, Fx, Fz, Mz, FgR
    Rückgabe:
      (create_plots, save_plots, griff_options, kraefte_options)
    """
    root = tk.Tk()
    root.title("Choose Options for Plots")
    root.geometry("700x500")

    # Diagrammtyp-Auswahlbereich (Radiobuttons)
    diagram_type_var = tk.StringVar(value="bar")

    # Funktion zum Umschalten der Sichtbarkeit von Plot-Options je nach Diagrammtyp
    def update_plot_option_visibility():
        if diagram_type_var.get() == "bar":
            # bar_container_frame sichtbar machen
            bar_container_frame.pack(fill="x", padx=10, pady=5)
            # Zeit-Frame ausblenden
            time_frame.pack_forget()
            time_options_container.pack_forget()
            # bar_frame nur anzeigen, wenn ausgeklappt
            if bar_frame.winfo_ismapped():
                bar_frame.pack_forget()
                bar_frame.pack(pady=5, padx=10, fill="x", anchor="center")
            # Vectoroptionen ausblenden
            vector_container_frame.pack_forget()
            vector_frame.pack_forget()
        elif diagram_type_var.get() == "time":
            # Zeitoptionen sichtbar machen
            time_options_container.pack(fill="x", padx=10, pady=5)
            time_frame.pack(pady=5, padx=10, fill="x", anchor="center")
            # bar_container_frame und bar_frame ausblenden
            bar_container_frame.pack_forget()
            bar_frame.pack_forget()
            # Vectoroptionen ausblenden
            vector_container_frame.pack_forget()
            vector_frame.pack_forget()
        elif diagram_type_var.get() == "vector":
            # Vectoroptionen sichtbar machen
            vector_container_frame.pack(fill="x", padx=10, pady=5)
            if vector_frame.winfo_ismapped():
                vector_frame.pack_forget()
                vector_frame.pack(pady=5, padx=10, fill="x", anchor="center")
            # bar und time ausblenden
            bar_container_frame.pack_forget()
            bar_frame.pack_forget()
            time_frame.pack_forget()
            time_options_container.pack_forget()
        else:
            # Default: alles ausblenden
            bar_container_frame.pack_forget()
            bar_frame.pack_forget()
            time_frame.pack_forget()
            time_options_container.pack_forget()
            vector_container_frame.pack_forget()
            vector_frame.pack_forget()
    # Diagrammtyp-Wechsel aktualisiert die Sichtbarkeit
    diagram_type_var.trace_add("write", lambda *args: update_plot_option_visibility())

    # Scrollbares Fenster mit Canvas
    main_canvas = tk.Canvas(root, borderwidth=0, highlightthickness=0)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
    scrollable_frame = tk.Frame(main_canvas)

    # Diagrammtyp-LabelFrame direkt nach Fenster-Setup
    diagram_type_frame = tk.LabelFrame(scrollable_frame, text="Choose Diagram Type")
    diagram_type_frame.pack(pady=8, padx=10, fill="x")
    bar_radio = tk.Radiobutton(diagram_type_frame, text="Barplot", variable=diagram_type_var, value="bar")
    time_radio = tk.Radiobutton(diagram_type_frame, text="Force-Time", variable=diagram_type_var, value="time")
    vector_radio = tk.Radiobutton(diagram_type_frame, text="Vectorplot", variable=diagram_type_var, value="vector")
    bar_radio.pack(side="left", padx=12, pady=3)
    time_radio.pack(side="left", padx=12, pady=3)
    vector_radio.pack(side="left", padx=12, pady=3)

    

    scrollable_frame.bind(
        "<Configure>",
        lambda e: main_canvas.configure(
            scrollregion=main_canvas.bbox("all")
        )
    )

    main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    main_canvas.configure(yscrollcommand=scrollbar.set)

    main_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Fenster zentrieren
    window_width = 700
    window_height = 1300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height/2 - window_height/2)
    position_right = int(screen_width/2 - window_width/2)
    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
    
    # Frame für Plot-Options
    plot_options_frame = tk.Frame(scrollable_frame)
    plot_options_frame.pack(side="top", fill="x")

    # Collapsible BAR-Options
    def toggle_bar_options():
        if bar_frame.winfo_ismapped():
            bar_frame.pack_forget()
            bar_toggle_btn.config(text="▶ Bar-Options")
        else:
            bar_frame.after(0, lambda: bar_frame.pack(pady=5, padx=10, fill="x", anchor="center"))
            bar_toggle_btn.config(text="▼ Bar-Options")

    bar_container_frame = tk.Frame(diagram_type_frame)
    # KEIN direktes pack() hier! Sichtbarkeit wird von update_plot_option_visibility() gesteuert.

    bar_toggle_btn = tk.Button(bar_container_frame, text="▶ Bar-Options", command=toggle_bar_options)
    bar_toggle_btn.pack(anchor="w", padx=10, pady=(10, 0))

    bar_frame = tk.LabelFrame(bar_container_frame, text="Bar-Options")


    # Option für FgR_sum (Checkbox, bleibt erhalten)
    plot_fgr_sum_var = tk.BooleanVar(value=False)

    # Collapsible Vector Options (container and frame)
    vector_container_frame = tk.Frame(diagram_type_frame)
    # KEIN direktes pack() hier! Sichtbarkeit wird von update_plot_option_visibility() gesteuert.
    def toggle_vector_options():
        if vector_frame.winfo_ismapped():
            vector_frame.pack_forget()
            vector_toggle_btn.config(text="▶ Vector-Options")
        else:
            vector_frame.after(0, lambda: vector_frame.pack(pady=5, padx=10, fill="x", anchor="center"))
            vector_toggle_btn.config(text="▼ Vector-Options")

    vector_toggle_btn = tk.Button(vector_container_frame, text="▶ Vector-Options", command=toggle_vector_options)
    vector_toggle_btn.pack(anchor="w", padx=10, pady=(10, 0))

    vector_frame = tk.LabelFrame(vector_container_frame, text="Vector-Options")
    # Checkboxes für Vector-Options
    plot_vector_var = tk.BooleanVar(value=False)
    plot_vector_checkbox = tk.Checkbutton(vector_frame, text="Vector plot", variable=plot_vector_var)
    plot_vector_checkbox.pack(anchor="w", padx=5)

    plot_vector_interval_only_var = tk.BooleanVar(value=False)
    plot_vector_interval_only_checkbox = tk.Checkbutton(vector_frame, text="Intervalls only (Vector)", variable=plot_vector_interval_only_var)
    plot_vector_interval_only_checkbox.pack(anchor="w", padx=20)

    # Auswahl der Metrik (z. B. "impuls", "mean", "max", "min")
    metric_label = tk.Label(bar_frame, text="Metric:")
    metric_label.pack(anchor="w")
    metric_var = tk.StringVar(value="impuls")
    metric_dropdown = tk.OptionMenu(bar_frame, metric_var, "impuls", "mean", "max", "maxROFD", "contact time", "hausdorff")
    metric_dropdown.pack(anchor="w", padx=5)

    # Collapsible Signal-/Daten-Options
    def toggle_signal_options():
        if signal_frame.winfo_ismapped():
            signal_frame.pack_forget()
            signal_toggle_btn.config(text="▶ Signal-Options")
        else:
            signal_frame.after(0, lambda: signal_frame.pack(pady=5, padx=10, fill="x", anchor="center"))
            signal_toggle_btn.config(text="▼ Signal-Options")

    signal_toggle_btn = tk.Button(scrollable_frame, text="▶ Signal-Options", command=toggle_signal_options)
    signal_toggle_btn.pack(anchor="w", padx=10, pady=(10, 0))

    signal_frame = tk.LabelFrame(scrollable_frame, text="Signal-Options")

    # Option zum manuellen Setzen der Y-Achsenlimits (nach Trimmen-Checkbox)
    set_y_limits_var = tk.BooleanVar(value=False)

    def toggle_y_limits_options():
        if set_y_limits_var.get():
            y_limits_frame.pack(pady=5, padx=10, fill="x")
        else:
            y_limits_frame.pack_forget()

    set_y_limits_checkbox = tk.Checkbutton(signal_frame, text="Y-Achsenlimits setzen", variable=set_y_limits_var, command=toggle_y_limits_options)
    set_y_limits_checkbox.pack(anchor="center", pady=2, padx=5)

    y_limits_frame = tk.LabelFrame(signal_frame, text="Y-Achsenlimits")

    y_min_var = tk.DoubleVar(value=0.0)
    y_max_var = tk.DoubleVar(value=0.0)

    tk.Label(y_limits_frame, text="Y min:").pack(anchor="w", padx=5)
    y_min_entry = tk.Entry(y_limits_frame, textvariable=y_min_var)
    y_min_entry.pack(fill="x", padx=5)

    tk.Label(y_limits_frame, text="Y max:").pack(anchor="w", padx=5)
    y_max_entry = tk.Entry(y_limits_frame, textvariable=y_max_var)
    y_max_entry.pack(fill="x", padx=5)

    # Zeit-Graphen-Options
    time_options_container = tk.Frame(diagram_type_frame)

    time_frame = tk.LabelFrame(time_options_container, text="Time-Plot-Options")

    submit_frame = tk.Frame(scrollable_frame)

    # Option zum Speichern der Plots
    save_plots_var = tk.BooleanVar(value=False)
    save_plots_checkbox = tk.Checkbutton(submit_frame, text="Save plots", variable=save_plots_var)
    save_plots_checkbox.pack(side="left", padx=10)
    # Option zum Exportieren der Berechnungen
    export_data_var = tk.BooleanVar(value=False)
    export_data_checkbox = tk.Checkbutton(submit_frame, text="Export Calculations", variable=export_data_var)
    export_data_checkbox.pack(side="left", padx=10)

    # --- Export Hausdof plots option ---
    config.create_hausdorff_plots = False  # Default

    hausdorff_plot_var = tk.BooleanVar(value=False)
    hausdorff_plot_checkbox = tk.Checkbutton(submit_frame, text="Export HD-plots", variable=hausdorff_plot_var)
    hausdorff_plot_checkbox.pack(side="left", padx=10)

    def update_hausdorff_setting(*args):
        config.create_hausdorff_plots = hausdorff_plot_var.get()

    hausdorff_plot_var.trace_add("write", update_hausdorff_setting)

    # Option zum Erstellen der Plots
    create_plots_var = tk.BooleanVar(value=True)
    create_plots_checkbox = tk.Checkbutton(plot_options_frame, text="Plots erstellen", variable=create_plots_var)
    create_plots_checkbox.pack(side="left", padx=10)

    # Option to show interval
    show_interval_var = tk.BooleanVar(value=False)
    show_interval_checkbox = tk.Checkbutton(time_frame, text="Show Intervals", variable=show_interval_var)
    show_interval_checkbox.pack(anchor="w", pady=2, padx=5)

    # Nested option to show impulse data (including contact time)
    show_impuls_data_var = tk.BooleanVar(value=False)
    show_impuls_data_checkbox = tk.Checkbutton(time_frame, text="    Show Impuls", variable=show_impuls_data_var)
    show_impuls_data_checkbox.pack(anchor="w", pady=2, padx=25)

    # Option für Kraft/Zeit pro Griff
    plot_data_per_hold_var = tk.BooleanVar(value=True)
    plot_data_per_hold_checkbox = tk.Checkbutton(time_frame, text="Force over time per Hold", variable=plot_data_per_hold_var)
    plot_data_per_hold_checkbox.pack(anchor="w", pady=2, padx=5)

    # Option für Bar-Splitview
   # bar_split_var = tk.BooleanVar(value=False)
    #bar_split_checkbox = tk.Checkbutton(bar_frame, text="Bar Splitview", variable=bar_split_var)
    #bar_split_checkbox.pack(anchor="w", padx=5)

    # Option zum Anzeigen der Werte über den Balken
   # show_values_var = tk.BooleanVar(value=True)
    #show_values_checkbox = tk.Checkbutton(bar_frame, text="Werte anzeigen", variable=show_values_var)
    #show_values_checkbox.pack(anchor="w", padx=5)
   
    # Option zur Trennung von NormalForcen und Moment in getrennten Plots
    split_fmz_var = tk.BooleanVar(value=False)
    split_fmz_checkbox = tk.Checkbutton(time_frame, text="  split F & Mz", variable=split_fmz_var)
    split_fmz_checkbox.pack(anchor="w", pady=2, padx=25)

    # Option zum Vergleichen von Forcen pro Griff
    compare_forces_var = tk.BooleanVar(value=False)
    compare_forces_checkbox = tk.Checkbutton(time_frame, text="Compare loads in single plot", variable=compare_forces_var)
    compare_forces_checkbox.pack(anchor="w", pady=2, padx=5)

    # Wenn "Force vergleichen pro Plot" aktiviert wird, Intervall und Impulsdaten deaktivieren
    def on_compare_forces_change(*args):
        if compare_forces_var.get():
            show_impuls_data_var.set(False)
            show_interval_var.set(False)
    compare_forces_var.trace_add("write", on_compare_forces_change)


    # Aufruf von update_plot_option_visibility jetzt am Ende des Zeit-Graphen-Bereichs

    # Option zum Trimmen der Plots
    trim_plot_var = tk.BooleanVar(value=False)

    def toggle_trim_options():
        if trim_plot_var.get():
            trim_frame.pack(pady=5, padx=10, fill="x")
        else:
            trim_frame.pack_forget()

    trim_plot_checkbox = tk.Checkbutton(signal_frame, text="Trim plot", variable=trim_plot_var, command=toggle_trim_options)
    trim_plot_checkbox.pack(anchor="center", pady=2, padx=5)

    # Collapsible Frame für Trim-Options
    trim_frame = tk.LabelFrame(signal_frame, text="Trim-Options")
    
    cutoff_start_var = tk.IntVar(value=0)
    cutoff_end_var = tk.IntVar(value=0)

    tk.Label(trim_frame, text="Cut from start (s):").pack(anchor="w", padx=5)
    cutoff_start_entry = tk.Entry(trim_frame, textvariable=cutoff_start_var)
    cutoff_start_entry.pack(fill="x", padx=5)

    tk.Label(trim_frame, text="Cut from ende (s):").pack(anchor="w", padx=5)
    cutoff_end_entry = tk.Entry(trim_frame, textvariable=cutoff_end_var)
    cutoff_end_entry.pack(fill="x", padx=5)

    # Option zur Verwendung des Savitzky-Golay Filters
    use_SVG_filter_var = tk.BooleanVar(value=True)
    use_SVG_filter_checkbox = tk.Checkbutton(signal_frame, text="Savitzky-Golay Filter", variable=use_SVG_filter_var)
    use_SVG_filter_checkbox.pack(anchor="center", pady=2, padx=5)

    # Collapsible Frame für Savitzky-Golay Optionen
    svg_options_frame = tk.LabelFrame(signal_frame, text="Savitzky-Golay Options")
    svg_options_frame.pack(pady=5, padx=10, fill="x")

    window_length_var = tk.IntVar(value=11)
    polyorder_var = tk.IntVar(value=5)

    # Checkbox: Force/Momente durch Körpergewicht normieren
    normalize_by_weight_var = tk.BooleanVar(value=True)
    normalize_by_weight_checkbox = tk.Checkbutton(
        signal_frame, text="Normalize by bodyweight", variable=normalize_by_weight_var
    )
    normalize_by_weight_checkbox.pack(anchor="center", pady=2, padx=5)

    # Neue Checkbox: Auto-Trimming aktivieren
    autotrim_var = tk.BooleanVar(value=True)
    autotrim_checkbox = tk.Checkbutton(signal_frame, text="Auto-trim", variable=autotrim_var)
    autotrim_checkbox.pack(anchor="center", pady=2, padx=5)

    tk.Label(svg_options_frame, text="Windowlength:").pack(anchor="w", padx=5)
    window_length_entry = tk.Entry(svg_options_frame, textvariable=window_length_var)
    window_length_entry.pack(fill="x", padx=5)

    tk.Label(svg_options_frame, text="Polynom Order:").pack(anchor="w", padx=5)
    polyorder_entry = tk.Entry(svg_options_frame, textvariable=polyorder_var)
    polyorder_entry.pack(fill="x", padx=5)

    svg_options_visible = True

    def toggle_svg_options():
        nonlocal svg_options_visible
        svg_options_visible = not svg_options_visible
        if not svg_options_visible:
            svg_options_frame.pack_forget()
        else:
            svg_options_frame.after(0, lambda: svg_options_frame.pack(pady=5, padx=10, fill="x"))

    svg_toggle_button = tk.Button(signal_frame, text="▶Filter Options", command=toggle_svg_options)
    svg_toggle_button.pack(anchor="w", pady=2, padx=5)
    toggle_svg_options()
    
    # Eingabe für Datenordner (für LVM-Dateien)
    data_folder_frame = tk.LabelFrame(scrollable_frame, text="Input-Folder  (.lvm)")
    data_folder_frame.pack(pady=5, padx=10, fill="x")
    data_folder_var = tk.StringVar()
    data_folder_entry = tk.Entry(data_folder_frame, textvariable=data_folder_var)
    data_folder_entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)
    def browse_data_folder():
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            data_folder_var.set(folder_selected)
    browse_data_button = tk.Button(data_folder_frame, text="Browse", command=browse_data_folder)
    browse_data_button.pack(side="right", padx=5, pady=5)


    # Eingabe für optionalen Speicherpfad
    save_folder_frame = tk.LabelFrame(scrollable_frame, text="Export-Folder (optional)")
    save_folder_frame.pack(pady=5, padx=10, fill="x")
    save_folder_var = tk.StringVar()
    save_folder_entry = tk.Entry(save_folder_frame, textvariable=save_folder_var)
    save_folder_entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)
    def browse_folder():
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            save_folder_var.set(folder_selected)
    browse_button = tk.Button(save_folder_frame, text="Browse", command=browse_folder)
    browse_button.pack(side="right", padx=5, pady=5)
    
    # Eingabe für Dateinamens-Suffix
    suffix_frame = tk.LabelFrame(scrollable_frame, text="File-Suffix (optional)")
    suffix_frame.pack(pady=5, padx=10, fill="x")
    suffix_var = tk.StringVar()
    suffix_entry = tk.Entry(suffix_frame, textvariable=suffix_var)
    suffix_entry.pack(fill="x", padx=5, pady=5)

    # Eingabe für zu überspringende Intervalle
    intervals_frame = tk.LabelFrame(scrollable_frame, text="Intervals to Skip (e.g. 2,3)")
    intervals_frame.pack(pady=5, padx=10, fill="x")
    intervals_var = tk.StringVar()
    intervals_entry = tk.Entry(intervals_frame, textvariable=intervals_var)
    intervals_entry.pack(fill="x", padx=5, pady=5)
    
    # Griff-Options
    griff_frame = tk.LabelFrame(scrollable_frame, text="Hold")
    griff_frame.pack(pady=10, padx=10, fill="both")
    
    griff_all_var = tk.BooleanVar(value=True)
    griff_g1_var = tk.BooleanVar(value=True)
    griff_g2_var = tk.BooleanVar(value=True)

    g1_all_var = tk.BooleanVar(value=False)
    
    def update_griff_all():
        griff_g1_var.set(griff_all_var.get())
        griff_g2_var.set(griff_all_var.get())
    
    def update_griff_g1():
        if not griff_g1_var.get():
            griff_all_var.set(False)
            g1_Fy_var.set(False)
            g1_Fx_var.set(False)
            g1_Fz_var.set(False)
            g1_Mz_var.set(False)
            g1_FgR_var.set(False)
            g1_Fres_var.set(False)
            g1_Fres_xyz_var.set(False)
            g1_phi_var.set(False)

    def update_griff_g2():
        if not griff_g2_var.get():
            griff_all_var.set(False)
            g2_Fy_var.set(False)
            g2_Fx_var.set(False)
            g2_Fz_var.set(False)
            g2_Mz_var.set(False)
            g2_FgR_var.set(False)
            g2_Fres_var.set(False)
            g2_Fres_xyz_var.set(False)
            g2_phi_var.set(False)
    
    griff_all_cb = tk.Checkbutton(griff_frame, text="All", variable=griff_all_var, command=update_griff_all)
    griff_g1_cb = tk.Checkbutton(griff_frame, text="G1R(right)", variable=griff_g1_var, command=update_griff_g1)
    griff_g2_cb = tk.Checkbutton(griff_frame, text="G2L(left)", variable=griff_g2_var, command=update_griff_g2)
    
    griff_all_cb.pack(side="left", padx=5)
    griff_g2_cb.pack(side="left", padx=5)
    griff_g1_cb.pack(side="left", padx=5)
    
    # Force-Options Ein-/Ausklappbar
    kraefte_visible = {"shown": True}

    def toggle_kraefte_options():
        if kraefte_visible["shown"]:
            kraefte_frame.pack_forget()
            kraefte_toggle_btn.config(text="▶ Force-Options")
            kraefte_visible["shown"] = False
        else:
            kraefte_frame.after(0, lambda: kraefte_frame.pack(pady=10, padx=10, fill="both"))
            kraefte_toggle_btn.config(text="▼ Force-Options")
            kraefte_visible["shown"] = True

    kraefte_toggle_btn = tk.Button(scrollable_frame, text="▼ Force-Options", command=toggle_kraefte_options)
    kraefte_toggle_btn.pack(anchor="w", padx=10, pady=(10, 0))

    kraefte_frame = tk.LabelFrame(scrollable_frame, text="Forces")
    kraefte_frame.pack(pady=10, padx=10, fill="both")

    # Für G2L (jetzt links)
    kraefte_g2_frame = tk.LabelFrame(kraefte_frame, text="G2L")
    kraefte_g2_frame.pack(side="left", padx=10, pady=10, fill="both")

    g2_all_var = tk.BooleanVar(value=False)
    g2_Fy_var = tk.BooleanVar(value=True)
    g2_Fz_var = tk.BooleanVar(value=True)
    g2_Fx_var = tk.BooleanVar(value=True)
    g2_Mz_var = tk.BooleanVar(value=False)
    g2_FgR_var = tk.BooleanVar(value=False)
    g2_FgR_sum_var = tk.BooleanVar(value=False)
    g2_Fres_var = tk.BooleanVar(value=False)
    g2_Fres_cb = tk.Checkbutton(kraefte_g2_frame, text="Fres_yz", variable=g2_Fres_var, command=lambda: update_kraft_g2_single())
    g2_Fres_xyz_var = tk.BooleanVar(value=False)
    g2_Fres_xyz_cb = tk.Checkbutton(kraefte_g2_frame, text="Fres_xyz", variable=g2_Fres_xyz_var, command=lambda: update_kraft_g2_single())
    g2_phi_var = tk.BooleanVar(value=False)
    g2_phi_cb = tk.Checkbutton(kraefte_g2_frame, text="φ_yz", variable=g2_phi_var, command=lambda: update_kraft_g2_single())

    def update_kraefte_g2():
        g2_Fy_var.set(g2_all_var.get())
        g2_Fz_var.set(g2_all_var.get())
        g2_Fx_var.set(g2_all_var.get())
        g2_Mz_var.set(g2_all_var.get())
        g2_FgR_var.set(g2_all_var.get())
        g2_FgR_sum_var.set(g2_all_var.get())
        g2_Fres_var.set(g2_all_var.get())
        g2_Fres_xyz_var.set(g2_all_var.get())
        g2_phi_var.set(g2_all_var.get())

    def update_kraft_g2_single():
        if not all([
            g2_Fy_var.get(), g2_Fx_var.get(), g2_Fz_var.get(),
            g2_Mz_var.get(), g2_FgR_var.get(), g2_FgR_sum_var.get(), g2_Fres_var.get(),
            g2_Fres_xyz_var.get(), g2_phi_var.get()
        ]):
            g2_all_var.set(False)
        if any([
            g2_Fy_var.get(), g2_Fx_var.get(), g2_Fz_var.get(),
            g2_Mz_var.get(), g2_FgR_var.get(), g2_FgR_sum_var.get(), g2_Fres_var.get(),
            g2_Fres_xyz_var.get(), g2_phi_var.get()
        ]):
            griff_g2_var.set(True)

    g2_all_cb = tk.Checkbutton(kraefte_g2_frame, text="all", variable=g2_all_var, command=update_kraefte_g2)
    g2_Fy_cb = tk.Checkbutton(kraefte_g2_frame, text="Fy", variable=g2_Fy_var, command=update_kraft_g2_single)
    g2_Fz_cb = tk.Checkbutton(kraefte_g2_frame, text="Fz", variable=g2_Fz_var, command=update_kraft_g2_single)
    g2_Fx_cb = tk.Checkbutton(kraefte_g2_frame, text="Fx", variable=g2_Fx_var, command=update_kraft_g2_single)
    g2_Mz_cb = tk.Checkbutton(kraefte_g2_frame, text="Mz", variable=g2_Mz_var, command=update_kraft_g2_single)
    g2_FgR_cb = tk.Checkbutton(kraefte_g2_frame, text="FgR", variable=g2_FgR_var, command=update_kraft_g2_single)
    g2_FgR_sum_cb = tk.Checkbutton(kraefte_g2_frame, text="FgR_sum", variable=g2_FgR_sum_var, command=update_kraft_g2_single)

    # Pack Reihenfolge: all zuerst, dann die anderen
    g2_all_cb.pack(anchor="w")
    g2_Fy_cb.pack(anchor="w")
    g2_Fz_cb.pack(anchor="w")
    g2_Fx_cb.pack(anchor="w")
    g2_Mz_cb.pack(anchor="w")
    g2_FgR_cb.pack(anchor="w")
    g2_FgR_sum_cb.pack(anchor="w")
    g2_Fres_cb.pack(anchor="w")
    g2_Fres_xyz_cb.pack(anchor="w")
    g2_phi_cb.pack(anchor="w")

    # Für G1R (jetzt rechts)
    kraefte_g1_frame = tk.LabelFrame(kraefte_frame, text="G1R")
    kraefte_g1_frame.pack(side="right", padx=10, pady=10, fill="both")

    g1_Fy_var = tk.BooleanVar(value=True)
    g1_Fz_var = tk.BooleanVar(value=True)
    g1_Fx_var = tk.BooleanVar(value=True)
    g1_Mz_var = tk.BooleanVar(value=False)
    g1_FgR_var = tk.BooleanVar(value=False)
    g1_FgR_sum_var = tk.BooleanVar(value=False)
    g1_Fres_var = tk.BooleanVar(value=False)
    g1_Fres_cb = tk.Checkbutton(kraefte_g1_frame, text="Fres_yz", variable=g1_Fres_var, command=lambda: update_kraft_g1_single())
    g1_Fres_xyz_var = tk.BooleanVar(value=False)
    g1_Fres_xyz_cb = tk.Checkbutton(kraefte_g1_frame, text="Fres_xyz", variable=g1_Fres_xyz_var, command=lambda: update_kraft_g1_single())
    g1_phi_var = tk.BooleanVar(value=False)
    g1_phi_cb = tk.Checkbutton(kraefte_g1_frame, text="φ_yz", variable=g1_phi_var, command=lambda: update_kraft_g1_single())

    def update_kraefte_g1():
        g1_Fy_var.set(g1_all_var.get())
        g1_Fx_var.set(g1_all_var.get())
        g1_Fz_var.set(g1_all_var.get())
        g1_Mz_var.set(g1_all_var.get())
        g1_FgR_var.set(g1_all_var.get())
        g1_FgR_sum_var.set(g1_all_var.get())
        g1_Fres_var.set(g1_all_var.get())
        g1_Fres_xyz_var.set(g1_all_var.get())
        g1_phi_var.set(g1_all_var.get())

    def update_kraft_g1_single():
        if not all([
            g1_Fy_var.get(), g1_Fx_var.get(), g1_Fz_var.get(),
            g1_Mz_var.get(), g1_FgR_var.get(), g1_FgR_sum_var.get(), g1_Fres_var.get(),
            g1_Fres_xyz_var.get(), g1_phi_var.get()
        ]):
            g1_all_var.set(False)
        if any([
            g1_Fy_var.get(), g1_Fx_var.get(), g1_Fz_var.get(),
            g1_Mz_var.get(), g1_FgR_var.get(), g1_FgR_sum_var.get(), g1_Fres_var.get(),
            g1_Fres_xyz_var.get(), g1_phi_var.get()
        ]):
            griff_g1_var.set(True)

    g1_all_cb = tk.Checkbutton(kraefte_g1_frame, text="all", variable=g1_all_var, command=update_kraefte_g1)
    g1_Fy_cb = tk.Checkbutton(kraefte_g1_frame, text="Fy", variable=g1_Fy_var, command=update_kraft_g1_single)
    g1_Fz_cb = tk.Checkbutton(kraefte_g1_frame, text="Fz", variable=g1_Fz_var, command=update_kraft_g1_single)
    g1_Fx_cb = tk.Checkbutton(kraefte_g1_frame, text="Fx", variable=g1_Fx_var, command=update_kraft_g1_single)
    g1_Mz_cb = tk.Checkbutton(kraefte_g1_frame, text="Mz", variable=g1_Mz_var, command=update_kraft_g1_single)
    g1_FgR_cb = tk.Checkbutton(kraefte_g1_frame, text="FgR", variable=g1_FgR_var, command=update_kraft_g1_single)
    g1_FgR_sum_cb = tk.Checkbutton(kraefte_g1_frame, text="FgR_sum", variable=g1_FgR_sum_var, command=update_kraft_g1_single)

    # Pack Reihenfolge: all zuerst, dann die anderen
    g1_all_cb.pack(anchor="w")
    g1_Fy_cb.pack(anchor="w")
    g1_Fz_cb.pack(anchor="w")
    g1_Fx_cb.pack(anchor="w")
    g1_Mz_cb.pack(anchor="w")
    g1_FgR_cb.pack(anchor="w")
    g1_FgR_sum_cb.pack(anchor="w")
    g1_Fres_cb.pack(anchor="w")
    g1_Fres_xyz_cb.pack(anchor="w")
    g1_phi_cb.pack(anchor="w")

    update_griff_all()

    # Neue Variable und Funktion für Dateipfad-Options
    paths_options_visible = True

    def toggle_paths_options():
        nonlocal paths_options_visible
        paths_options_visible = not paths_options_visible
        if paths_options_visible:
            data_folder_frame.pack(pady=5, padx=10, fill="x")
            save_folder_frame.pack(pady=5, padx=10, fill="x")
            suffix_frame.pack(pady=5, padx=10, fill="x")
        else:
            data_folder_frame.pack_forget()
            save_folder_frame.pack_forget()
            suffix_frame.pack_forget()

    paths_toggle_button = tk.Button(scrollable_frame, text="Show/Hide Path Settings ", command=toggle_paths_options)
    paths_toggle_button.pack(pady=10)
    toggle_paths_options()

    # Sichtbarkeit von Plot-Options initial setzen
    update_plot_option_visibility()

    # OK-Button
    cancelled = {"status": False}

    def submit():
        # Parse intervals to skip and save to config
        try:
            config.invalid_intervals_list = [int(x.strip()) for x in intervals_var.get().split(",") if x.strip().isdigit()]
        except Exception as e:
            print(f"Could not parse invalid intervals: {e}")
            config.invalid_intervals_list = []
        root.quit()
        root.destroy()

    def cancel():
        cancelled["status"] = True
        root.quit()
        root.destroy()

    submit_frame.pack(pady=20)

    submit_button = tk.Button(submit_frame, text="OK", command=submit)
    submit_button.pack(side="right", padx=10)

    cancel_button = tk.Button(submit_frame, text="Cancel", command=cancel)
    cancel_button.pack(side="right", padx=10)

    def _on_mousewheel(event):
        main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    # Support mouse wheel scrolling on Windows, Mac, and Linux
    main_canvas.bind_all("<MouseWheel>", _on_mousewheel)          # Windows, some Mac
    main_canvas.bind_all("<Button-4>", lambda e: main_canvas.yview_scroll(-1, "units"))  # Linux scroll up
    main_canvas.bind_all("<Button-5>", lambda e: main_canvas.yview_scroll(1, "units"))   # Linux scroll down

    root.mainloop()
    
    
    # Rückgabewerte vorbereiten
    if cancelled["status"]:
        return None

    plot_settings = {
        "create": create_plots_var.get(),
        "save": save_plots_var.get(),
        "export_data": export_data_var.get(),
        "split_fmz": split_fmz_var.get(),
        "compare_forces": compare_forces_var.get(),
        "show_interval": show_interval_var.get(),
        "show_impuls_data": show_impuls_data_var.get(),
        #"bar_split": bar_split_var.get(),
        "plot_bar": diagram_type_var.get() == "bar",
        "plot_mean_metrics_bar": diagram_type_var.get() == "bar",
        #"show_values": show_values_var.get(),
        "plot_vector": plot_vector_var.get(),
        "plot_vector_interval_only": plot_vector_interval_only_var.get(),
        "plot_fgr_sum": plot_fgr_sum_var.get(),
        "plot_data_per_hold": plot_data_per_hold_var.get(),
        "y_limits": (y_min_var.get(), y_max_var.get()) if set_y_limits_var.get() else None,
        "diagram_type": diagram_type_var.get()
    }

    filter_settings = {
        "use_filter": use_SVG_filter_var.get(),
        "window_length": window_length_var.get(),
        "polyorder": polyorder_var.get(),
        "normalize_by_weight": normalize_by_weight_var.get(),
        "autotrim": autotrim_var.get(),
    }

    file_paths = {
        "save_folder": save_folder_var.get(),
        "data_folder": data_folder_var.get(),
        "suffix": suffix_var.get(),
        
    }

    cutoff = {
        "active": trim_plot_var.get(),
        "start": cutoff_start_var.get(),
        "end": cutoff_end_var.get()
    }
    
    griff_options = {
        "all": griff_all_var.get(),
        "G1R": griff_g1_var.get(),
        "G2L": griff_g2_var.get()
    }
    kraefte_options = {
        "G1R": {
            "Fy": g1_Fy_var.get(),
            "Fx": g1_Fx_var.get(),
            "Fz": g1_Fz_var.get(),
            "Mz": g1_Mz_var.get(),
            "FgR": g1_FgR_var.get(),
            "FgR_sum": g1_FgR_sum_var.get(),
            "Fres_yz": g1_Fres_var.get(),
            "Fres_xyz": g1_Fres_xyz_var.get(),
            "φ_yz": g1_phi_var.get()
        },
        "G2L": {
            "Fy": g2_Fy_var.get(),
            "Fx": g2_Fx_var.get(),
            "Fz": g2_Fz_var.get(),
            "Mz": g2_Mz_var.get(),
            "FgR": g2_FgR_var.get(),
            "FgR_sum": g2_FgR_sum_var.get(),
            "Fres_yz": g2_Fres_var.get(),
            "Fres_xyz": g2_Fres_xyz_var.get(),
            "φ_yz": g2_phi_var.get()
        }
    }
    
    return plot_settings, griff_options, kraefte_options, filter_settings, file_paths, cutoff, metric_var.get()


