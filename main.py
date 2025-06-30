from gui import*
from loadData import load_lvm_data
from utils import *
from plotdata import *
import os
import datetime
import copy
import glob


# Export run settings to a .txt file

def export_plot_settings(settings, holds_to_plot, forces_to_plot, filters, file_paths, cutoff, selected_metric):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"RunSettings_{now}.txt"
    path = os.path.join(file_paths["save_folder"], filename)

    with open(path, "w") as f:
        f.write(f"Run Settings Export — {now}\n\n")
        f.write("Selected Metric:\n")
        f.write(f"  {selected_metric}\n\n")

        f.write("Plot Settings:\n")
        for k, v in settings.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nFilter Settings:\n")
        for k, v in filters.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nCutoff:\n")
        for k, v in cutoff.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nHolds to Plot:\n")
        for k, v in holds_to_plot.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nForces to Plot:\n")
        for side, forces in forces_to_plot.items():
            f.write(f"  {side}:\n")
            for k, v in forces.items():
                f.write(f"    {k}: {v}\n")

    print(f"Einstellungen exportiert nach: {path}")



def main():
    
    os.system('cls' if os.name == 'nt' else 'clear')
    result = run_gui()
    if result is None:
        print("GUI abgebrochen.")
        return
    plot_settings, holds_to_plot, forces_to_plot, filter_settings, file_paths, cutoff, selected_metric = result
    
    export_data = plot_settings.get("export_data", False)
    
    normalizebyweight = filter_settings["normalize_by_weight"]
    
    print("forces_to_plot:", forces_to_plot, "\n holds to plot", holds_to_plot)
    optional_suffix = ""

    if filter_settings["normalize_by_weight"] == True:
        print("\nForces Normalized by weight")
        optional_suffix = "_NBW"
    if plot_settings.get("show_impuls_data", False):
        optional_suffix += "_IMP"
    
 
    folder_path = file_paths["data_folder"] or "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/pipeline"
    save_folder = file_paths["save_folder"] or folder_path + "/Figures"
    optional_suffix = file_paths["suffix"] or ""
    optional_suffix = "_" + optional_suffix if optional_suffix else ""
    optional_suffix += get_force_suffix(forces_to_plot)

    unique_force_list = get_unique_selected_forces(forces_to_plot=forces_to_plot)
    print(" unqiue_force_lsit ",unique_force_list )

    settings = {
        "SVGwindowlength": filter_settings["window_length"],
        "SVGpolyorder": filter_settings["polyorder"],
        "use_filter": filter_settings["use_filter"],
        "normalizeByweight": filter_settings["normalize_by_weight"],
        "save_plot": plot_settings["save"],
        "autotrim": filter_settings["autotrim"]
    }

    folder_list = [
      #  "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Moment",
      "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Reliability"
       # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Test-Rest",
       #     "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2",
       # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/pipeline",
 #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Technik",
 #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Technik/Leftside_data",
 #  "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Technik/Rightside_data",
 # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/medium-yellow",
 #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/worst-black",
 #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/best-grey",
 # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Clipping",
 #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Griffe",
 # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/shakeout",
 # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/ForceDevRatio"
    
]
    
    for curr_folder in folder_list:
        folder_path = curr_folder
        save_folder = curr_folder + "/Figures"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        
        all_lvm_data_dict = load_lvm_data(folder_path, settings=settings, export=export_data)

        if filter_settings["use_filter"]:
            optional_suffix += "_filtered_with-" + str(filter_settings["window_length"]) + "-" + str(filter_settings["polyorder"])
        else:
            optional_suffix += "_raw"
        if cutoff.get("active", False):
            optional_suffix += f"_trimmed-S{cutoff['start']}-E{cutoff['end']}"
        
        all_lvm_data_dict, forces_g1, forces_g2 = prepare_data(all_lvm_data_dict, forces_to_plot, cutoff)
        if all_lvm_data_dict is None:
            return

        
    #  print("\nall_lvm_data_dict keys:", list(all_lvm_data_dict.keys()))

        # compute globale limits for y_limits and save it with every Hold such that every dataframe can access it
        ylims = plot_settings["y_limits"]
        for filename, file_data in all_lvm_data_dict.items():
            print("\n+++++++++++ new File ++++++++++++++")
            all_lvm_data_dict[filename] = compute_global_ylimits_for_plots(file_data, forces_g1, forces_g2, ylims=ylims)
            #print_current_dict_summary(current_dict=all_lvm_data_dict)

        export_plot_settings(settings, holds_to_plot, forces_to_plot, filter_settings, file_paths, cutoff, selected_metric)

        # Wenn "Plots erstellen" aktiviert ist:
        if plot_settings["create"]:
            print("\n ----------------------------------------------")
            print("\n ---------------- Plotting --------------------")  
            print("\n ----------------------------------------------")

            if plot_settings.get("diagram_type") == "time":   
                # Explizite Einzelprüfungen für die verschiedenen Plotoptionen
                if plot_settings.get("plot_data_per_hold", False) and not plot_settings.get("split_fmz", False) and not plot_settings.get("compare_forces", False):
                    print("Plot: plot_data_per_hold aktiviert")
                    for curr_filename, data_per_file in all_lvm_data_dict.items():
                        print(f"\n +++Plotting GL+GR: {curr_filename}")
                        #Zeitdiagramm Daten pro griff (Oben, Unten)
                        plot_data_per_hold(
                            data_per_file,
                            forces_g1,
                            forces_g2,
                            curr_filename + optional_suffix,
                            save_plot=plot_settings["save"],
                            save_folder=save_folder,
                            cutoff=cutoff,
                            normalizebyweight=normalizebyweight,
                            show_contact_time=plot_settings.get("show_impuls_data", False),
                            show_interval=plot_settings.get("show_interval", False)
                        )

                if plot_settings.get("compare_forces", False) and not plot_settings.get("split_fmz", False):
                    print("Plot: plot_selected_forces_comparison aktiviert")
                    for filename, file_data in all_lvm_data_dict.items():
                        #Zeitdiagramm gelcihe Kraft beider Griffe in einem Plot
                        plot_selected_forces_comparison(
                            file_data,
                            forces_g1,
                            forces_g2,
                            filename=filename + optional_suffix,
                            save_folder=save_folder,
                            save_plot=plot_settings["save"],
                            cutoff=cutoff,
                            normalizebyweight=normalizebyweight,
                            show_contact_time=plot_settings.get("show_impuls_data", False),
                            show_interval=plot_settings.get("show_interval", False)
                        )

                if plot_settings.get("split_fmz", False):
                    print("Plot: split_fmz aktiviert")
                    #Zeitdiagramm F oben Mz unten
                    for curr_filename, data_per_file in all_lvm_data_dict.items():
                        if holds_to_plot.get("G2L", False):
                            print(f"Plotting G2L splitview: {curr_filename}")
                            g1r_data = data_per_file.get("G1R", {}).get("data", None)
                            if g1r_data is None:
                                print(f"❌ Kein 'data' in G1R für Datei {curr_filename}")
                            elif g1r_data.empty:
                                print(f"⚠️ 'data' in G1R ist leer für Datei {curr_filename}")
                            else:
                                print(f"✅ G1R 'data' enthält {len(g1r_data)} Zeilen")
                            plot_single_hold_splitview(
                                data_per_file["G2L"],
                                forces_g2,
                                curr_filename + optional_suffix,
                                grip_label="links",
                                save_plot=plot_settings["save"],
                                save_folder=save_folder,
                                cutoff=cutoff,
                                normalizebyweight=normalizebyweight,
                                show_contact_time=plot_settings.get("show_impuls_data", False),
                                show_interval=plot_settings.get("show_interval", False)
                            )
                        print(f"Holds to plot G1R is {holds_to_plot.get('G1R', False)}")
                        if holds_to_plot.get("G1R", False):
                            print(f"Plotting G1R splitview: {curr_filename}")
                            plot_single_hold_splitview(
                                data_per_file["G1R"],
                                forces_g1,
                                curr_filename + optional_suffix,
                                grip_label="rechts",
                                save_plot=plot_settings["save"],
                                save_folder=save_folder,
                                cutoff=cutoff,
                                normalizebyweight=normalizebyweight,
                                show_contact_time=plot_settings.get("show_impuls_data", False),
                                show_interval=plot_settings.get("show_interval", False)
                            )
    #----------------------------- BAR PLOTS -----------------------------------
            if plot_settings.get("diagram_type") == "bar":

                if plot_settings.get("plot_mean_metrics_bar", False):
                    print("Plot: plot_mean_metrics_bar aktiviert")
                    for side in ["G1R", "G2L"]:
                        # Bar plot
                        plot_mean_metrics_bar(
                            all_lvm_data_dict=all_lvm_data_dict,
                            forces=unique_force_list,
                            metric=selected_metric,
                            side=side,
                            figsize=(8, 6),
                            title=f"Mean-{selected_metric}",
                            save_plot=plot_settings["save"],
                            save_folder=save_folder,
                            #split_view=plot_settings.get("bar_split", False)

                        )

                if plot_settings.get("plot_fgr_sum", False):
                    print("Plot: plot_FgR_sum aktiviert")
                    for filename, file_data in all_lvm_data_dict.items():
                        # Zeitdiagramm FgR Summe
                        plot_FgR_sum(
                            file_dict=file_data,
                            forces_g1=forces_g1,
                            forces_g2=forces_g2,
                            filename=filename + optional_suffix,
                            grip_label="G1R + G2L",
                            save_plot=plot_settings["save"],
                            save_folder=save_folder,
                            normalizebyweight=normalizebyweight
                        )

                # Zeige alle erzeugten Plots

                # Plot force vector trace if enabled
                if plot_settings.get("plot_vector", False):
                    for curr_filename, data_per_file in all_lvm_data_dict.items():
                        for side in ["G1R", "G2L"]:
                            df = data_per_file.get(side, {}).get("data", None)
                            if df is not None:
                                plot_force_vector_trace(
                                    df,
                                    forces=("Fy", "Fz"),
                                    title=f"{curr_filename} - {side} Kraftvektorverlauf",
                                    save_plot=plot_settings["save"],
                                    save_folder=save_folder,
                                    filename=f"{curr_filename}_{side}_force_vector",
                                    plot_vector_interval_only=plot_settings.get("plot_vector_interval_only", False),
                                    intervals=data_per_file[side].get("contact_time", {}).get("Fy", [])
                                )

    # Show all generated plots after all plotting is done
    import matplotlib.pyplot as plt
    plt.show()



if __name__ == "__main__":
    main()
   # folders = [
 #"/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Technik/Leftside_data",
  #  "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Technik/Rightside_data"
  # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/MEDIUM- YELLOW",
  # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/black",
  # "/Users/noah/LRZ Sync+Share/MA/ZHS_LabView_Messungen/Exploration_V2/Shoes_and_footholds/Best_grey",
    

 
