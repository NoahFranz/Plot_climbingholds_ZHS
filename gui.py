import tkinter as tk
from tkinter import filedialog

def run_gui():
    """
    Erzeugt eine GUI zur Auswahl der Optionen:
      - Ob die Plots gespeichert werden sollen
      - Ob die Plots erstellt werden sollen
      - Griff-Optionen: All, G1, G2
      - Kräfte-Optionen:
            Für G1: all, Fy, Fx, Fz, Mz
            Für G2: all, Fy, Fx, Fz, Mz
    Rückgabe:
      (create_plots, save_plots, griff_options, kraefte_options)
    """
    root = tk.Tk()
    root.title("Optionen wählen")
    root.geometry("500x1000")
    
    # Fenster zentrieren
    window_width = 500
    window_height = 1000
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height/2 - window_height/2)
    position_right = int(screen_width/2 - window_width/2)
    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
    
    # Option zum Speichern der Plots
    save_plots_var = tk.BooleanVar(value=False)
    save_plots_checkbox = tk.Checkbutton(root, text="Plots speichern", variable=save_plots_var)
    save_plots_checkbox.pack(pady=10)

    # Option zum Erstellen der Plots
    create_plots_var = tk.BooleanVar(value=True)
    create_plots_checkbox = tk.Checkbutton(root, text="Plots erstellen", variable=create_plots_var)
    create_plots_checkbox.pack(pady=10)
   
   
    # Option zur verwendung des Savatzgi-Goolay filters
    use_SVG_filter_var = tk.BooleanVar(value=True)
    use_SVG_filter_checkbox = tk.Checkbutton(root, text="Savatzgi-golay filter verwenden?", variable=use_SVG_filter_var)
    use_SVG_filter_checkbox.pack(pady=10)

    # Fensterlänge und Polynomgrad für Savitzky-Golay
    svg_frame = tk.LabelFrame(root, text="Savitzky-Golay Optionen")
    svg_frame.pack(pady=5, padx=10, fill="x")

    window_length_var = tk.IntVar(value=11)
    polyorder_var = tk.IntVar(value=5)

    tk.Label(svg_frame, text="Fensterlänge:").pack(anchor="w", padx=5)
    window_length_entry = tk.Entry(svg_frame, textvariable=window_length_var)
    window_length_entry.pack(fill="x", padx=5)

    tk.Label(svg_frame, text="Polynomgrad:").pack(anchor="w", padx=5)
    polyorder_entry = tk.Entry(svg_frame, textvariable=polyorder_var)
    polyorder_entry.pack(fill="x", padx=5)



    # Option zur Trennung von Normalkräften und Moment in getrennten Plots
    split_fmz_var = tk.BooleanVar(value=True)
    split_fmz_checkbox = tk.Checkbutton(root, text="Normalkräfte und Moment in separaten Plots? (oben: F, unten: Mz)", variable=split_fmz_var)
    split_fmz_checkbox.pack(pady=10)
    
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

    def update_griff_all():
        griff_g1_var.set(griff_all_var.get())
        griff_g2_var.set(griff_all_var.get())
    
    def update_griff_g1():
        if not griff_g1_var.get():
            griff_all_var.set(False)
            g1_all_var.set(False)
            g1_Fy_var.set(False)
            g1_Fx_var.set(False)
            g1_Fz_var.set(False)
            g1_Mz_var.set(False)

    def update_griff_g2():
        if not griff_g2_var.get():
            griff_all_var.set(False)
            g2_all_var.set(False)
            g2_Fy_var.set(False)
            g2_Fx_var.set(False)
            g2_Fz_var.set(False)
            g2_Mz_var.set(False)
    
    griff_all_cb = tk.Checkbutton(griff_frame, text="All", variable=griff_all_var, command=update_griff_all)
    griff_g1_cb = tk.Checkbutton(griff_frame, text="G1(rechts)", variable=griff_g1_var, command=update_griff_g1)
    griff_g2_cb = tk.Checkbutton(griff_frame, text="G2(links)", variable=griff_g2_var, command=update_griff_g2)
    
    griff_all_cb.pack(anchor="w")
    griff_g1_cb.pack(anchor="w")
    griff_g2_cb.pack(anchor="w")
    
    # Kräfte-Optionen
    kraefte_frame = tk.LabelFrame(root, text="Kräfte")
    kraefte_frame.pack(pady=10, padx=10, fill="both")
    
    # Für G1
    kraefte_g1_frame = tk.LabelFrame(kraefte_frame, text="G1")
    kraefte_g1_frame.pack(side="left", padx=10, pady=10, fill="both")
    
    g1_all_var = tk.BooleanVar(value=True)
    g1_Fy_var = tk.BooleanVar(value=True)
    g1_Fx_var = tk.BooleanVar(value=True)
    g1_Fz_var = tk.BooleanVar(value=True)
    g1_Mz_var = tk.BooleanVar(value=True)

    def update_kraefte_g1():
        g1_Fy_var.set(g1_all_var.get())
        g1_Fx_var.set(g1_all_var.get())
        g1_Fz_var.set(g1_all_var.get())
        g1_Mz_var.set(g1_all_var.get())
    
    def update_kraft_g1_single():
        if not all([g1_Fy_var.get(), g1_Fx_var.get(), g1_Fz_var.get(), g1_Mz_var.get()]):
            g1_all_var.set(False)
        if any([g1_Fy_var.get(), g1_Fx_var.get(), g1_Fz_var.get(), g1_Mz_var.get()]):
            griff_g1_var.set(True)
    
    g1_all_cb = tk.Checkbutton(kraefte_g1_frame, text="all", variable=g1_all_var, command=update_kraefte_g1)
    g1_Fy_cb = tk.Checkbutton(kraefte_g1_frame, text="Fy", variable=g1_Fy_var, command=update_kraft_g1_single)
    g1_Fx_cb = tk.Checkbutton(kraefte_g1_frame, text="Fx", variable=g1_Fx_var, command=update_kraft_g1_single)
    g1_Fz_cb = tk.Checkbutton(kraefte_g1_frame, text="Fz", variable=g1_Fz_var, command=update_kraft_g1_single)
    g1_Mz_cb = tk.Checkbutton(kraefte_g1_frame, text="Mz", variable=g1_Mz_var, command=update_kraft_g1_single)
    
    g1_all_cb.pack(anchor="w")
    g1_Fy_cb.pack(anchor="w")
    g1_Fx_cb.pack(anchor="w")
    g1_Fz_cb.pack(anchor="w")
    g1_Mz_cb.pack(anchor="w")
    
    # Für G2
    kraefte_g2_frame = tk.LabelFrame(kraefte_frame, text="G2")
    kraefte_g2_frame.pack(side="right", padx=10, pady=10, fill="both")
    
    g2_all_var = tk.BooleanVar(value=True)
    g2_Fy_var = tk.BooleanVar(value=True)
    g2_Fx_var = tk.BooleanVar(value=True)
    g2_Fz_var = tk.BooleanVar(value=True)
    g2_Mz_var = tk.BooleanVar(value=True)

    def update_kraefte_g2():
        g2_Fy_var.set(g2_all_var.get())
        g2_Fx_var.set(g2_all_var.get())
        g2_Fz_var.set(g2_all_var.get())
        g2_Mz_var.set(g2_all_var.get())
    
    def update_kraft_g2_single():
        if not all([g2_Fy_var.get(), g2_Fx_var.get(), g2_Fz_var.get(), g2_Mz_var.get()]):
            g2_all_var.set(False)
        if any([g2_Fy_var.get(), g2_Fx_var.get(), g2_Fz_var.get(), g2_Mz_var.get()]):
            griff_g2_var.set(True)
    
    g2_all_cb = tk.Checkbutton(kraefte_g2_frame, text="all", variable=g2_all_var, command=update_kraefte_g2)
    g2_Fy_cb = tk.Checkbutton(kraefte_g2_frame, text="Fy", variable=g2_Fy_var, command=update_kraft_g2_single)
    g2_Fx_cb = tk.Checkbutton(kraefte_g2_frame, text="Fx", variable=g2_Fx_var, command=update_kraft_g2_single)
    g2_Fz_cb = tk.Checkbutton(kraefte_g2_frame, text="Fz", variable=g2_Fz_var, command=update_kraft_g2_single)
    g2_Mz_cb = tk.Checkbutton(kraefte_g2_frame, text="Mz", variable=g2_Mz_var, command=update_kraft_g2_single)
    
    g2_all_cb.pack(anchor="w")
    g2_Fy_cb.pack(anchor="w")
    g2_Fx_cb.pack(anchor="w")
    g2_Fz_cb.pack(anchor="w")
    g2_Mz_cb.pack(anchor="w")
    
    if griff_all_var.get():
            griff_g1_var.set(True)
            griff_g2_var.set(True)

    if g1_all_var.get():
            g1_Fy_var.set(True)
            g1_Fx_var.set(True)
            g1_Fz_var.set(True)
            g1_Mz_var.set(True)

    if g2_all_var.get():
            g2_Fy_var.set(True)
            g2_Fx_var.set(True)
            g2_Fz_var.set(True)
            g2_Mz_var.set(True)





    # OK-Button
    def submit():
        root.quit()
    submit_button = tk.Button(root, text="OK", command=submit)
    submit_button.pack(pady=20)
    
    root.mainloop()
    
    # Rückgabewerte vorbereiten
    griff_options = {
        "all": griff_all_var.get(),
        "G1": griff_g1_var.get(),
        "G2": griff_g2_var.get()
    }
    kraefte_options = {
        "G1": {
            "all": g1_all_var.get(),
            "Fy": g1_Fy_var.get(),
            "Fx": g1_Fx_var.get(),
            "Fz": g1_Fz_var.get(),
            "Mz": g1_Mz_var.get()
        },
        "G2": {
            "all": g2_all_var.get(),
            "Fy": g2_Fy_var.get(),
            "Fx": g2_Fx_var.get(),
            "Fz": g2_Fz_var.get(),
            "Mz": g2_Mz_var.get()
        }
    }
    
    return create_plots_var.get(), save_plots_var.get(), griff_options, kraefte_options, split_fmz_var.get(), save_folder_var.get(), suffix_var.get(), data_folder_var.get(), use_SVG_filter_var.get(), window_length_var.get(), polyorder_var.get()