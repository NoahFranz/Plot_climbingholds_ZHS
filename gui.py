import tkinter as tk
from tkinter import filedialog

def run_gui():
    """
    Erzeugt eine GUI zur Auswahl der Optionen:
      - Ob die Plots gespeichert werden sollen
      - Ob die Plots erstellt werden sollen
      - Griff-Optionen: All, G1, G2
      - Kräfte-Optionen:
            Für G1: all, Fy, Fx, Fz, Mz, FgR
            Für G2: all, Fy, Fx, Fz, Mz, FgR
    Rückgabe:
      (create_plots, save_plots, griff_options, kraefte_options)
    """
    root = tk.Tk()
    root.title("Optionen wählen")
    root.geometry("700x400")

    # Scrollbares Fenster mit Canvas
    main_canvas = tk.Canvas(root, borderwidth=0, highlightthickness=0)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
    scrollable_frame = tk.Frame(main_canvas)

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
    window_width = 350
    window_height = 1300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height/2 - window_height/2)
    position_right = int(screen_width/2 - window_width/2)
    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
    
    # Frame für Plot-Optionen
    plot_options_frame = tk.Frame(scrollable_frame)
    plot_options_frame.pack(side="top", fill="x")

    # BAR-Optionen
    bar_frame = tk.LabelFrame(scrollable_frame, text="Bar-Optionen")
    bar_frame.pack(pady=5, padx=10, fill="x", anchor="center")

    # Neue Checkbox: Balkendiagramm plotten
    plot_bar_var = tk.BooleanVar(value=False)
    plot_bar_checkbox = tk.Checkbutton(bar_frame, text="Balkendiagramm plotten", variable=plot_bar_var)
    plot_bar_checkbox.pack(anchor="center", padx=5)

    # Signal-/Daten-Optionen
    signal_frame = tk.LabelFrame(scrollable_frame, text="Signal-Optionen")
    signal_frame.pack(pady=5, padx=10, fill="x", anchor="center")

    # Option zum manuellen Setzen der Y-Achsenlimits (nach Trimmen-Checkbox)
    set_y_limits_var = tk.BooleanVar(value=False)

    def toggle_y_limits_options():
        if set_y_limits_var.get():
            y_limits_frame.pack(pady=5, padx=10, fill="x")
        else:
            y_limits_frame.pack_forget()

    set_y_limits_checkbox = tk.Checkbutton(signal_frame, text="Y-Achsenlimits setzen?", variable=set_y_limits_var, command=toggle_y_limits_options)
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

    # Zeit-Graphen-Optionen
    time_frame = tk.LabelFrame(scrollable_frame, text="Zeit-Graphen")
    time_frame.pack(pady=5, padx=10, fill="x", anchor="center")

    submit_frame = tk.Frame(scrollable_frame)

    # Option zum Speichern der Plots
    save_plots_var = tk.BooleanVar(value=False)
    save_plots_checkbox = tk.Checkbutton(submit_frame, text="Plots speichern", variable=save_plots_var)
    save_plots_checkbox.pack(side="left", padx=10)

    # Option zum Erstellen der Plots
    create_plots_var = tk.BooleanVar(value=True)
    create_plots_checkbox = tk.Checkbutton(plot_options_frame, text="Plots erstellen", variable=create_plots_var)
    create_plots_checkbox.pack(side="left", padx=10)

    # Option für Impuls-Berechnung anzeigen
    show_impulses_var = tk.BooleanVar(value=True)
    show_impulses_checkbox = tk.Checkbutton(time_frame, text="Impuls anzeigen", variable=show_impulses_var)
    show_impulses_checkbox.pack(anchor="center", pady=2, padx=5)

    # Option für Bar-Splitview
    bar_split_var = tk.BooleanVar(value=False)
    bar_split_checkbox = tk.Checkbutton(bar_frame, text="BAR Splitview", variable=bar_split_var)
    bar_split_checkbox.pack(anchor="center", padx=5)

    # Option zum Anzeigen der Werte über den Balken
    show_values_var = tk.BooleanVar(value=True)
    show_values_checkbox = tk.Checkbutton(bar_frame, text="Werte anzeigen", variable=show_values_var)
    show_values_checkbox.pack(anchor="center", padx=5)
   
    # Option zur Trennung von Normalkräften und Moment in getrennten Plots
    split_fmz_var = tk.BooleanVar(value=False)
    split_fmz_checkbox = tk.Checkbutton(time_frame, text="F & Mz trennen", variable=split_fmz_var)
    split_fmz_checkbox.pack(anchor="center", pady=2, padx=5)

    # Option zum Vergleichen von Kräften pro Griff
    compare_forces_var = tk.BooleanVar(value=False)
    compare_forces_checkbox = tk.Checkbutton(time_frame, text="Kräfte vergleichen", variable=compare_forces_var)
    compare_forces_checkbox.pack(anchor="center", pady=2, padx=5)

    # Option zum Trimmen der Plots
    trim_plot_var = tk.BooleanVar(value=False)

    def toggle_trim_options():
        if trim_plot_var.get():
            trim_frame.pack(pady=5, padx=10, fill="x")
        else:
            trim_frame.pack_forget()

    trim_plot_checkbox = tk.Checkbutton(signal_frame, text="Plot trimmen?", variable=trim_plot_var, command=toggle_trim_options)
    trim_plot_checkbox.pack(anchor="center", pady=2, padx=5)

    # Collapsible Frame für Trim-Optionen
    trim_frame = tk.LabelFrame(signal_frame, text="Trim-Optionen")
    
    cutoff_start_var = tk.IntVar(value=0)
    cutoff_end_var = tk.IntVar(value=0)

    tk.Label(trim_frame, text="Von Start (s):").pack(anchor="w", padx=5)
    cutoff_start_entry = tk.Entry(trim_frame, textvariable=cutoff_start_var)
    cutoff_start_entry.pack(fill="x", padx=5)

    tk.Label(trim_frame, text="Von Ende (s):").pack(anchor="w", padx=5)
    cutoff_end_entry = tk.Entry(trim_frame, textvariable=cutoff_end_var)
    cutoff_end_entry.pack(fill="x", padx=5)

    # Option zur Verwendung des Savitzky-Golay Filters
    use_SVG_filter_var = tk.BooleanVar(value=True)
    use_SVG_filter_checkbox = tk.Checkbutton(signal_frame, text="Savitzky-Golay Filter", variable=use_SVG_filter_var)
    use_SVG_filter_checkbox.pack(anchor="center", pady=2, padx=5)

    # Collapsible Frame für Savitzky-Golay Optionen
    svg_options_frame = tk.LabelFrame(signal_frame, text="Savitzky-Golay Optionen")
    svg_options_frame.pack(pady=5, padx=10, fill="x")

    window_length_var = tk.IntVar(value=11)
    polyorder_var = tk.IntVar(value=5)

    # Checkbox: Kräfte/Momente durch Körpergewicht normieren
    normalize_by_weight_var = tk.BooleanVar(value=True)
    normalize_by_weight_checkbox = tk.Checkbutton(
        signal_frame, text="Normierung nach Körpergewicht", variable=normalize_by_weight_var
    )
    normalize_by_weight_checkbox.pack(anchor="center", pady=2, padx=5)

    # Neue Checkbox: Auto-Trimming aktivieren
    autotrim_var = tk.BooleanVar(value=True)
    autotrim_checkbox = tk.Checkbutton(signal_frame, text="Auto-Trimming aktivieren", variable=autotrim_var)
    autotrim_checkbox.pack(anchor="center", pady=2, padx=5)

    tk.Label(svg_options_frame, text="Fensterlänge:").pack(anchor="w", padx=5)
    window_length_entry = tk.Entry(svg_options_frame, textvariable=window_length_var)
    window_length_entry.pack(fill="x", padx=5)

    tk.Label(svg_options_frame, text="Polynomgrad:").pack(anchor="w", padx=5)
    polyorder_entry = tk.Entry(svg_options_frame, textvariable=polyorder_var)
    polyorder_entry.pack(fill="x", padx=5)

    svg_options_visible = True

    def toggle_svg_options():
        nonlocal svg_options_visible
        svg_options_visible = not svg_options_visible
        svg_options_frame.pack_forget() if not svg_options_visible else svg_options_frame.pack(pady=5, padx=10, fill="x")

    svg_toggle_button = tk.Button(signal_frame, text="Filter Optionen", command=toggle_svg_options)
    svg_toggle_button.pack(anchor="w", pady=2, padx=5)
    toggle_svg_options()
    
    # Eingabe für Datenordner (für LVM-Dateien)
    data_folder_frame = tk.LabelFrame(scrollable_frame, text="Datenordner (für LVM-Dateien)")
    data_folder_frame.pack(pady=5, padx=10, fill="x")
    data_folder_var = tk.StringVar()
    data_folder_entry = tk.Entry(data_folder_frame, textvariable=data_folder_var)
    data_folder_entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)
    def browse_data_folder():
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            data_folder_var.set(folder_selected)
    browse_data_button = tk.Button(data_folder_frame, text="Durchsuchen", command=browse_data_folder)
    browse_data_button.pack(side="right", padx=5, pady=5)


    # Eingabe für optionalen Speicherpfad
    save_folder_frame = tk.LabelFrame(scrollable_frame, text="Speicherordner (optional)")
    save_folder_frame.pack(pady=5, padx=10, fill="x")
    save_folder_var = tk.StringVar()
    save_folder_entry = tk.Entry(save_folder_frame, textvariable=save_folder_var)
    save_folder_entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)
    def browse_folder():
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            save_folder_var.set(folder_selected)
    browse_button = tk.Button(save_folder_frame, text="Durchsuchen", command=browse_folder)
    browse_button.pack(side="right", padx=5, pady=5)
    
    # Eingabe für Dateinamens-Suffix
    suffix_frame = tk.LabelFrame(scrollable_frame, text="Datei-Suffix (optional)")
    suffix_frame.pack(pady=5, padx=10, fill="x")
    suffix_var = tk.StringVar()
    suffix_entry = tk.Entry(suffix_frame, textvariable=suffix_var)
    suffix_entry.pack(fill="x", padx=5, pady=5)
    
    # Griff-Optionen
    griff_frame = tk.LabelFrame(scrollable_frame, text="Griff")
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

    def update_griff_g2():
        if not griff_g2_var.get():
            griff_all_var.set(False)
    
    griff_all_cb = tk.Checkbutton(griff_frame, text="All", variable=griff_all_var, command=update_griff_all)
    griff_g1_cb = tk.Checkbutton(griff_frame, text="G1(rechts)", variable=griff_g1_var, command=update_griff_g1)
    griff_g2_cb = tk.Checkbutton(griff_frame, text="G2(links)", variable=griff_g2_var, command=update_griff_g2)
    
    griff_all_cb.pack(side="left", padx=5)
    griff_g1_cb.pack(side="left", padx=5)
    griff_g2_cb.pack(side="left", padx=5)
    
    # Kräfte-Optionen
    kraefte_frame = tk.LabelFrame(scrollable_frame, text="Kräfte")
    kraefte_frame.pack(pady=10, padx=10, fill="both")
    
    # Für G1
    kraefte_g1_frame = tk.LabelFrame(kraefte_frame, text="G1")
    kraefte_g1_frame.pack(side="left", padx=10, pady=10, fill="both")

    g1_Fy_var = tk.BooleanVar(value=True)
    g1_Fz_var = tk.BooleanVar(value=True)
    g1_Fx_var = tk.BooleanVar(value=True)
    g1_Mz_var = tk.BooleanVar(value=False)
    g1_FgR_var = tk.BooleanVar(value=False)
    g1_Fres_var = tk.BooleanVar(value=False)
    g1_Fres_cb = tk.Checkbutton(kraefte_g1_frame, text="Fres_YZ", variable=g1_Fres_var, command=lambda: update_kraft_g1_single())
    g1_Fres_xyz_var = tk.BooleanVar(value=False)
    g1_Fres_xyz_cb = tk.Checkbutton(kraefte_g1_frame, text="Fres_XYZ", variable=g1_Fres_xyz_var, command=lambda: update_kraft_g1_single())
    g1_phi_var = tk.BooleanVar(value=False)
    g1_phi_cb = tk.Checkbutton(kraefte_g1_frame, text="φ_yz", variable=g1_phi_var, command=lambda: update_kraft_g1_single())

    def update_kraefte_g1():
        g1_Fy_var.set(g1_all_var.get())
        g1_Fx_var.set(g1_all_var.get())
        g1_Fz_var.set(g1_all_var.get())
        g1_Mz_var.set(g1_all_var.get())
        g1_FgR_var.set(g1_all_var.get())
        g1_Fres_var.set(g1_all_var.get())
        g1_Fres_xyz_var.set(g1_all_var.get())
        g1_phi_var.set(g1_all_var.get())

    def update_kraft_g1_single():
        if not all([
            g1_Fy_var.get(), g1_Fx_var.get(), g1_Fz_var.get(),
            g1_Mz_var.get(), g1_FgR_var.get(), g1_Fres_var.get(),
            g1_Fres_xyz_var.get(), g1_phi_var.get()
        ]):
            g1_all_var.set(False)
        if any([
            g1_Fy_var.get(), g1_Fx_var.get(), g1_Fz_var.get(),
            g1_Mz_var.get(), g1_FgR_var.get(), g1_Fres_var.get(),
            g1_Fres_xyz_var.get(), g1_phi_var.get()
        ]):
            griff_g1_var.set(True)

    g1_all_cb = tk.Checkbutton(kraefte_g1_frame, text="all", variable=g1_all_var, command=update_kraefte_g1)
    g1_Fy_cb = tk.Checkbutton(kraefte_g1_frame, text="Fy", variable=g1_Fy_var, command=update_kraft_g1_single)
    g1_Fz_cb = tk.Checkbutton(kraefte_g1_frame, text="Fz", variable=g1_Fz_var, command=update_kraft_g1_single)
    g1_Fx_cb = tk.Checkbutton(kraefte_g1_frame, text="Fx", variable=g1_Fx_var, command=update_kraft_g1_single)
    g1_Mz_cb = tk.Checkbutton(kraefte_g1_frame, text="Mz", variable=g1_Mz_var, command=update_kraft_g1_single)
    g1_FgR_cb = tk.Checkbutton(kraefte_g1_frame, text="FgR", variable=g1_FgR_var, command=update_kraft_g1_single)

    # Pack Reihenfolge: all zuerst, dann die anderen
    g1_all_cb.pack(anchor="w")
    g1_Fy_cb.pack(anchor="w")
    g1_Fz_cb.pack(anchor="w")
    g1_Fx_cb.pack(anchor="w")
    g1_Mz_cb.pack(anchor="w")
    g1_FgR_cb.pack(anchor="w")
    g1_Fres_cb.pack(anchor="w")
    g1_Fres_xyz_cb.pack(anchor="w")
    g1_phi_cb.pack(anchor="w")
    
    # Für G2
    kraefte_g2_frame = tk.LabelFrame(kraefte_frame, text="G2")
    kraefte_g2_frame.pack(side="right", padx=10, pady=10, fill="both")

    g2_all_var = tk.BooleanVar(value=False)

    g2_Fy_var = tk.BooleanVar(value=True)
    g2_Fz_var = tk.BooleanVar(value=True)
    g2_Fx_var = tk.BooleanVar(value=True)
    g2_Mz_var = tk.BooleanVar(value=False)
    g2_FgR_var = tk.BooleanVar(value=False)
    g2_Fres_var = tk.BooleanVar(value=False)
    g2_Fres_cb = tk.Checkbutton(kraefte_g2_frame, text="Fres_YZ", variable=g2_Fres_var, command=lambda: update_kraft_g2_single())
    g2_Fres_xyz_var = tk.BooleanVar(value=False)
    g2_Fres_xyz_cb = tk.Checkbutton(kraefte_g2_frame, text="Fres_XYZ", variable=g2_Fres_xyz_var, command=lambda: update_kraft_g2_single())
    g2_phi_var = tk.BooleanVar(value=False)
    g2_phi_cb = tk.Checkbutton(kraefte_g2_frame, text="φ_yz", variable=g2_phi_var, command=lambda: update_kraft_g2_single())

    def update_kraefte_g2():
        g2_Fy_var.set(g2_all_var.get())
        g2_Fz_var.set(g2_all_var.get())
        g2_Fx_var.set(g2_all_var.get())
        g2_Mz_var.set(g2_all_var.get())
        g2_FgR_var.set(g2_all_var.get())
        g2_Fres_var.set(g2_all_var.get())
        g2_Fres_xyz_var.set(g2_all_var.get())
        g2_phi_var.set(g2_all_var.get())

    def update_kraft_g2_single():
        if not all([
            g2_Fy_var.get(), g2_Fx_var.get(), g2_Fz_var.get(),
            g2_Mz_var.get(), g2_FgR_var.get(), g2_Fres_var.get(),
            g2_Fres_xyz_var.get(), g2_phi_var.get()
        ]):
            g2_all_var.set(False)
        if any([
            g2_Fy_var.get(), g2_Fx_var.get(), g2_Fz_var.get(),
            g2_Mz_var.get(), g2_FgR_var.get(), g2_Fres_var.get(),
            g2_Fres_xyz_var.get(), g2_phi_var.get()
        ]):
            griff_g2_var.set(True)

    g2_all_cb = tk.Checkbutton(kraefte_g2_frame, text="all", variable=g2_all_var, command=update_kraefte_g2)
    g2_Fy_cb = tk.Checkbutton(kraefte_g2_frame, text="Fy", variable=g2_Fy_var, command=update_kraft_g2_single)
    g2_Fz_cb = tk.Checkbutton(kraefte_g2_frame, text="Fz", variable=g2_Fz_var, command=update_kraft_g2_single)
    g2_Fx_cb = tk.Checkbutton(kraefte_g2_frame, text="Fx", variable=g2_Fx_var, command=update_kraft_g2_single)
    g2_Mz_cb = tk.Checkbutton(kraefte_g2_frame, text="Mz", variable=g2_Mz_var, command=update_kraft_g2_single)
    g2_FgR_cb = tk.Checkbutton(kraefte_g2_frame, text="FgR", variable=g2_FgR_var, command=update_kraft_g2_single)

    # Pack Reihenfolge: all zuerst, dann die anderen
    g2_all_cb.pack(anchor="w")
    g2_Fy_cb.pack(anchor="w")
    g2_Fz_cb.pack(anchor="w")
    g2_Fx_cb.pack(anchor="w")
    g2_Mz_cb.pack(anchor="w")
    g2_FgR_cb.pack(anchor="w")
    g2_Fres_cb.pack(anchor="w")
    g2_Fres_xyz_cb.pack(anchor="w")
    g2_phi_cb.pack(anchor="w")

    
    update_griff_all()

    # Neue Variable und Funktion für Dateipfad-Optionen
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

    paths_toggle_button = tk.Button(scrollable_frame, text="Dateipfad-Optionen anzeigen/ausblenden", command=toggle_paths_options)
    paths_toggle_button.pack(pady=10)
    toggle_paths_options()

    # OK-Button
    cancelled = {"status": False}

    def submit():
        root.quit()
        root.destroy()

    def cancel():
        cancelled["status"] = True
        root.quit()
        root.destroy()

    submit_frame.pack(pady=20)

    submit_button = tk.Button(submit_frame, text="OK", command=submit)
    submit_button.pack(side="right", padx=10)

    cancel_button = tk.Button(submit_frame, text="Abbrechen", command=cancel)
    cancel_button.pack(side="right", padx=10)

    def _on_mousewheel(event):
        main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    main_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    root.mainloop()
    
    # Rückgabewerte vorbereiten
    if cancelled["status"]:
        return None

    plot_settings = {
        "create": create_plots_var.get(),
        "save": save_plots_var.get(),
        "split_fmz": split_fmz_var.get(),
        "compare_forces": compare_forces_var.get(),
        "show_impulses": show_impulses_var.get(),
        "bar_split": bar_split_var.get(),
        "show_values": show_values_var.get(),
        "plot_bar": plot_bar_var.get(),
        "y_limits": (y_min_var.get(), y_max_var.get()) if set_y_limits_var.get() else None
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
        "G1": griff_g1_var.get(),
        "G2": griff_g2_var.get()
    }
    kraefte_options = {
        "G1": {
            "Fy": g1_Fy_var.get(),
            "Fx": g1_Fx_var.get(),
            "Fz": g1_Fz_var.get(),
            "Mz": g1_Mz_var.get(),
            "FgR": g1_FgR_var.get(),
            "Fres_YZ": g1_Fres_var.get(),
            "Fres_XYZ": g1_Fres_xyz_var.get(),
            "φ_yz": g1_phi_var.get()
        },
        "G2": {
            "Fy": g2_Fy_var.get(),
            "Fx": g2_Fx_var.get(),
            "Fz": g2_Fz_var.get(),
            "Mz": g2_Mz_var.get(),
            "FgR": g2_FgR_var.get(),
            "Fres_YZ": g2_Fres_var.get(),
            "Fres_XYZ": g2_Fres_xyz_var.get(),
            "φ_yz": g2_phi_var.get()
        }
    }
    
    return plot_settings, griff_options, kraefte_options, filter_settings, file_paths, cutoff