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
    root.geometry("700x1300")
    
    # Fenster zentrieren
    window_width = 700
    window_height = 1300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height/2 - window_height/2)
    position_right = int(screen_width/2 - window_width/2)
    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
    
    # Frame für Plot-Optionen
    plot_options_frame = tk.Frame(root)
    plot_options_frame.pack(side="top", fill="x")

    submit_frame = tk.Frame(root)

    # Option zum Speichern der Plots
    save_plots_var = tk.BooleanVar(value=False)
    save_plots_checkbox = tk.Checkbutton(submit_frame, text="Plots speichern", variable=save_plots_var)
    save_plots_checkbox.pack(side="left", padx=10)

    # Option zum Erstellen der Plots
    create_plots_var = tk.BooleanVar(value=True)
    create_plots_checkbox = tk.Checkbutton(plot_options_frame, text="Plots erstellen", variable=create_plots_var)
    create_plots_checkbox.pack(side="left", padx=10)
   
    # Option zur Trennung von Normalkräften und Moment in getrennten Plots
    trim_split_frame = tk.Frame(root)
    trim_split_frame.pack(pady=10)

    split_fmz_var = tk.BooleanVar(value=False)
    split_fmz_checkbox = tk.Checkbutton(trim_split_frame, text="F und Mz trennen?", variable=split_fmz_var)
    split_fmz_checkbox.pack(in_=trim_split_frame)

    # Option zum Vergleichen von Kräften pro Griff
    compare_forces_var = tk.BooleanVar(value=False)
    compare_forces_checkbox = tk.Checkbutton(root, text="Kräfte pro Griff vergleichen?", variable=compare_forces_var)
    compare_forces_checkbox.pack(pady=10)

    # Option zum Trimmen der Plots
    trim_plot_var = tk.BooleanVar(value=False)

    def toggle_trim_options():
        if trim_plot_var.get():
            trim_frame.pack(pady=5, padx=10, fill="x")
        else:
            trim_frame.pack_forget()

    trim_plot_checkbox = tk.Checkbutton(trim_split_frame, text="Plot trimmen?", variable=trim_plot_var, command=toggle_trim_options)
    trim_plot_checkbox.pack(in_=trim_split_frame)

    # Collapsible Frame für Trim-Optionen
    trim_frame = tk.LabelFrame(root, text="Trim-Optionen")
    
    cutoff_start_var = tk.IntVar(value=0)
    cutoff_end_var = tk.IntVar(value=0)

    tk.Label(trim_frame, text="Von Start (s):").pack(anchor="w", padx=5)
    cutoff_start_entry = tk.Entry(trim_frame, textvariable=cutoff_start_var)
    cutoff_start_entry.pack(fill="x", padx=5)

    tk.Label(trim_frame, text="Von Ende (s):").pack(anchor="w", padx=5)
    cutoff_end_entry = tk.Entry(trim_frame, textvariable=cutoff_end_var)
    cutoff_end_entry.pack(fill="x", padx=5)



    # Option zur verwendung des Savatzgi-Goolay filters
    use_SVG_filter_var = tk.BooleanVar(value=True)
    use_SVG_filter_checkbox = tk.Checkbutton(root, text="Savatzgi-golay filter verwenden?", variable=use_SVG_filter_var)
    use_SVG_filter_checkbox.pack(pady=10)

    # Collapsible Frame für Savitzky-Golay Optionen
    svg_options_frame = tk.LabelFrame(root, text="Savitzky-Golay Optionen")
    svg_options_frame.pack(pady=5, padx=10, fill="x")

    window_length_var = tk.IntVar(value=11)
    polyorder_var = tk.IntVar(value=5)

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

    svg_toggle_button = tk.Button(root, text="Savitzky-Golay Optionen anzeigen/ausblenden", command=toggle_svg_options)
    svg_toggle_button.pack(pady=10)
    toggle_svg_options()
    
    # Eingabe für Datenordner (für LVM-Dateien)
    data_folder_frame = tk.LabelFrame(root, text="Datenordner (für LVM-Dateien)")
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
    save_folder_frame = tk.LabelFrame(root, text="Speicherordner (optional)")
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
    suffix_frame = tk.LabelFrame(root, text="Datei-Suffix (optional)")
    suffix_frame.pack(pady=5, padx=10, fill="x")
    suffix_var = tk.StringVar()
    suffix_entry = tk.Entry(suffix_frame, textvariable=suffix_var)
    suffix_entry.pack(fill="x", padx=5, pady=5)
    
    # Griff-Optionen
    griff_frame = tk.LabelFrame(root, text="Griff")
    griff_frame.pack(pady=10, padx=10, fill="both")
    
    griff_all_var = tk.BooleanVar(value=True)
    griff_g1_var = tk.BooleanVar(value=True)
    griff_g2_var = tk.BooleanVar(value=True)

    g1_all_var = tk.BooleanVar(value=True)
    
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
    kraefte_frame = tk.LabelFrame(root, text="Kräfte")
    kraefte_frame.pack(pady=10, padx=10, fill="both")
    
    # Für G1
    kraefte_g1_frame = tk.LabelFrame(kraefte_frame, text="G1")
    kraefte_g1_frame.pack(side="left", padx=10, pady=10, fill="both")
    
    g1_Fy_var = tk.BooleanVar(value=True)
    g1_Fz_var = tk.BooleanVar(value=True)
    g1_Fx_var = tk.BooleanVar(value=True)
    g1_Mz_var = tk.BooleanVar(value=True)
    g1_FgR_var = tk.BooleanVar(value=True)
    g1_FgR_calc_var = tk.BooleanVar(value=False)
    g1_Fres_var = tk.BooleanVar(value=False)
    g1_Fres_cb = tk.Checkbutton(kraefte_g1_frame, text="Fres", variable=g1_Fres_var, command=lambda: update_kraft_g1_single())
    g1_Fres_cb.pack(anchor="w")

    g1_phi_var = tk.BooleanVar(value=False)
    g1_phi_cb = tk.Checkbutton(kraefte_g1_frame, text="φ_yz", variable=g1_phi_var, command=lambda: update_kraft_g1_single())
    g1_phi_cb.pack(anchor="w")
    
    def update_kraefte_g1():
        g1_Fy_var.set(g1_all_var.get())
        g1_Fx_var.set(g1_all_var.get())
        g1_Fz_var.set(g1_all_var.get())
        g1_Mz_var.set(g1_all_var.get())
        g1_FgR_var.set(g1_all_var.get())
        g1_FgR_calc_var.set(g1_all_var.get())
        g1_Fres_var.set(g1_all_var.get())
        g1_phi_var.set(g1_all_var.get())
    
    def update_kraft_g1_single():
        if not all([g1_Fy_var.get(), g1_Fx_var.get(), g1_Fz_var.get(), g1_Mz_var.get(), g1_FgR_var.get(), g1_FgR_calc_var.get(), g1_Fres_var.get(), g1_phi_var.get()]):
            g1_all_var.set(False)
        if any([g1_Fy_var.get(), g1_Fx_var.get(), g1_Fz_var.get(), g1_Mz_var.get(), g1_FgR_var.get(), g1_FgR_calc_var.get(), g1_Fres_var.get(), g1_phi_var.get()]):
            griff_g1_var.set(True)
    
    g1_all_cb = tk.Checkbutton(kraefte_g1_frame, text="all", variable=g1_all_var, command=update_kraefte_g1)
    g1_Fy_cb = tk.Checkbutton(kraefte_g1_frame, text="Fy", variable=g1_Fy_var, command=update_kraft_g1_single)
    g1_Fz_cb = tk.Checkbutton(kraefte_g1_frame, text="Fz", variable=g1_Fz_var, command=update_kraft_g1_single)
    g1_Fx_cb = tk.Checkbutton(kraefte_g1_frame, text="Fx", variable=g1_Fx_var, command=update_kraft_g1_single)
    g1_Mz_cb = tk.Checkbutton(kraefte_g1_frame, text="Mz", variable=g1_Mz_var, command=update_kraft_g1_single)
    g1_FgR_cb = tk.Checkbutton(kraefte_g1_frame, text="FgR", variable=g1_FgR_var, command=update_kraft_g1_single)
    g1_FgR_calc_cb = tk.Checkbutton(kraefte_g1_frame, text="FgR_calc", variable=g1_FgR_calc_var, command=update_kraft_g1_single)
    
    g1_all_cb.pack(anchor="w")
    g1_Fy_cb.pack(anchor="w")
    g1_Fz_cb.pack(anchor="w")
    g1_Fx_cb.pack(anchor="w")
    g1_Mz_cb.pack(anchor="w")
    g1_FgR_cb.pack(anchor="w")
    g1_FgR_calc_cb.pack(anchor="w")
    
    # Für G2
    kraefte_g2_frame = tk.LabelFrame(kraefte_frame, text="G2")
    kraefte_g2_frame.pack(side="right", padx=10, pady=10, fill="both")
    
    g2_all_var = tk.BooleanVar(value=True)
    
    g2_Fy_var = tk.BooleanVar(value=True)
    g2_Fz_var = tk.BooleanVar(value=True)
    g2_Fx_var = tk.BooleanVar(value=True)
    g2_Mz_var = tk.BooleanVar(value=True)
    g2_FgR_var = tk.BooleanVar(value=True)    
    g2_FgR_calc_var = tk.BooleanVar(value=False)
    g2_Fres_var = tk.BooleanVar(value=False)
    g2_Fres_cb = tk.Checkbutton(kraefte_g2_frame, text="Fres", variable=g2_Fres_var, command=lambda: update_kraft_g2_single())
    g2_Fres_cb.pack(anchor="w")

    g2_phi_var = tk.BooleanVar(value=False)
    g2_phi_cb = tk.Checkbutton(kraefte_g2_frame, text="φ_yz", variable=g2_phi_var, command=lambda: update_kraft_g2_single())
    g2_phi_cb.pack(anchor="w")

    def update_kraefte_g2():
        g2_Fy_var.set(g2_all_var.get())
        g2_Fz_var.set(g2_all_var.get())
        g2_Fx_var.set(g2_all_var.get()) 
        g2_Mz_var.set(g2_all_var.get())
        g2_FgR_var.set(g2_all_var.get())
        g2_FgR_calc_var.set(g2_all_var.get())
        g2_Fres_var.set(g2_all_var.get())
        g2_phi_var.set(g2_all_var.get())
    
    def update_kraft_g2_single():
        if not all([g2_Fy_var.get(), g2_Fx_var.get(), g2_Fz_var.get(), g2_Mz_var.get(), g2_FgR_var.get(), g2_FgR_calc_var.get(), g2_Fres_var.get(), g2_phi_var.get()]):
            g2_all_var.set(False)
        if any([g2_Fy_var.get(), g2_Fx_var.get(), g2_Fz_var.get(), g2_Mz_var.get(), g2_FgR_var.get(), g2_FgR_calc_var.get(), g2_Fres_var.get(), g2_phi_var.get()]):
            griff_g2_var.set(True)
    
    g2_all_cb = tk.Checkbutton(kraefte_g2_frame, text="all", variable=g2_all_var, command=update_kraefte_g2)
    g2_Fy_cb = tk.Checkbutton(kraefte_g2_frame, text="Fy", variable=g2_Fy_var, command=update_kraft_g2_single)
    g2_Fz_cb = tk.Checkbutton(kraefte_g2_frame, text="Fz", variable=g2_Fz_var, command=update_kraft_g2_single)
    g2_Fx_cb = tk.Checkbutton(kraefte_g2_frame, text="Fx", variable=g2_Fx_var, command=update_kraft_g2_single)  
    g2_Mz_cb = tk.Checkbutton(kraefte_g2_frame, text="Mz", variable=g2_Mz_var, command=update_kraft_g2_single)
    g2_FgR_cb = tk.Checkbutton(kraefte_g2_frame, text="FgR", variable=g2_FgR_var, command=update_kraft_g2_single)
    g2_FgR_calc_cb = tk.Checkbutton(kraefte_g2_frame, text="FgR_calc", variable=g2_FgR_calc_var, command=update_kraft_g2_single)
    
    g2_all_cb.pack(anchor="w")
    g2_Fy_cb.pack(anchor="w")
    g2_Fz_cb.pack(anchor="w")
    g2_Fx_cb.pack(anchor="w")
    g2_Mz_cb.pack(anchor="w")
    g2_FgR_cb.pack(anchor="w")
    g2_FgR_calc_cb.pack(anchor="w")    

    
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

    paths_toggle_button = tk.Button(root, text="Dateipfad-Optionen anzeigen/ausblenden", command=toggle_paths_options)
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

    root.mainloop()
    
    # Rückgabewerte vorbereiten
    if cancelled["status"]:
        return None

    plot_settings = {
        "create": create_plots_var.get(),
        "save": save_plots_var.get(),
        "split_fmz": split_fmz_var.get(),
        "compare_forces": compare_forces_var.get(),
    }

    filter_settings = {
        "use_filter": use_SVG_filter_var.get(),
        "window_length": window_length_var.get(),
        "polyorder": polyorder_var.get(),
    }

    file_paths = {
        "save_folder": save_folder_var.get(),
        "data_folder": data_folder_var.get(),
        "suffix": suffix_var.get()
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
            "FgR_calc": g1_FgR_calc_var.get(),
            "Fres": g1_Fres_var.get(),
            "φ_yz": g1_phi_var.get()
        },
        "G2": {
            "Fy": g2_Fy_var.get(),
            "Fx": g2_Fx_var.get(),
            "Fz": g2_Fz_var.get(),
            "Mz": g2_Mz_var.get(),
            "FgR": g2_FgR_var.get(),
            "FgR_calc": g2_FgR_calc_var.get(),
            "Fres": g2_Fres_var.get(),
            "φ_yz": g2_phi_var.get()
        }
    }
    
    return plot_settings, griff_options, kraefte_options, filter_settings, file_paths, cutoff