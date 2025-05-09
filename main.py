from gui import*
from loadData import load_lvm_data
from utils import *
from plotdata import *
import os



def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    result = run_gui()
    if result is None:
        print("GUI abgebrochen.")
        return
    plot_settings, holds_to_plot, forces_to_plot, filter_settings, file_paths, cutoff = result
    
    normalizebyweight = filter_settings["normalize_by_weight"]
    
    print("forces_to_plot:", forces_to_plot, "\n holds to plot", holds_to_plot)

    if filter_settings["normalize_by_weight"] == True:
        print("\nForces Normalized by weight")
        optional_suffix = "_NBW"
    if plot_settings["show_impulses"]:
        optional_suffix += "_IMP"
    
 
    folder_path = file_paths["data_folder"] or "/Users/noah/LRZ Sync+Share/MA/ZHS_ LabView_Messungen/Exploration_V1"
    save_folder = file_paths["save_folder"] or "/Users/noah/LRZ Sync+Share/MA/Plot_Figures"
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
    all_lvm_data_dict = load_lvm_data(folder_path, settings=settings)

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
    for filename, file_data in all_lvm_data_dict.items():
        print("\n+++++++++++ new File ++++++++++++++")
        all_lvm_data_dict[filename] = compute_global_ylimits_for_plots(file_data, forces_g1, forces_g2)
        print_current_dict_summary(current_dict=all_lvm_data_dict)

    # Wenn "Plots erstellen" aktiviert ist:
    if plot_settings["create"]:
        print("\n ----------------------------------------------")
        print("\n ---------------- Plotting --------------------")  
        print("\n ----------------------------------------------")      

        if plot_settings["plot_bar"] == True:
            for curr_force in unique_force_list:
                plot_impulses_bar(
                all_lvm_data_dict,
                forces="Fz",           # ODER "Fy" usw.
                split_grips=plot_settings["bar_split"],    # Nur ein Plot, beide Griffe nebeneinander
                show_values=plot_settings["show_values"],     # Werte über Balken anzeigen
                figsize=(6.3, 8),      # Plotgröße
                title="Impulsvergleich aller Athleten",
                normalizebyweight=normalizebyweight,
                save_folder=save_folder,
                save_plot=plot_settings["save"],
                optional_suffix=optional_suffix
                
            )
        else:    
            # Fall 1: Normale kombinierte Darstellung von G1R und G2L (kein Split, kein Vergleich)
            if not plot_settings["split_fmz"] and not plot_settings["compare_forces"]:
                for curr_filename, data_per_file in all_lvm_data_dict.items():
                    print(f"\n +++Plotting GL+GR: {curr_filename}")
                    plot_data_per_hold(
                        data_per_file,
                        forces_g1,
                        forces_g2,
                        curr_filename + optional_suffix,
                        save_plot=plot_settings["save"],
                        save_folder=save_folder,
                        cutoff=cutoff,
                        normalizebyweight=normalizebyweight,
                        show_contact_time=plot_settings["show_impulses"]
                    )
            
            # Fall 2: Vergleich der Kräfte pro Griff (kein Split, aber Vergleich aktiviert)
            elif not plot_settings["split_fmz"] and plot_settings["compare_forces"]:
                for filename, file_data in all_lvm_data_dict.items():
                    plot_selected_forces_comparison(
                        file_data,
                        forces_g1,
                        forces_g2,
                        filename=filename + optional_suffix,
                        save_folder=save_folder,
                        save_plot=plot_settings["save"],
                        cutoff=cutoff,
                        normalizebyweight=normalizebyweight,
                        show_contact_time=["show_impulses"]
                    )
            
            # Fall 3: Split-Modus aktiviert – Darstellung für G1 und G2 getrennt
            else:
                for curr_filename, data_per_file in all_lvm_data_dict.items():
                    if holds_to_plot["G2"]:
                        print(f"Plotting G2L splitview: {curr_filename}")
                        plot_single_hold_splitview(
                            data_per_file["G2L"]["data"],
                            forces_g2,
                            curr_filename + optional_suffix,
                            grip_label="links",
                            save_plot=plot_settings["save"],
                            save_folder=save_folder,
                            cutoff=cutoff,
                            normalizebyweight=normalizebyweight,
                            show_contact_time=["show_impulses"]
                        )
                    if holds_to_plot["G1"]:
                        print(f"Plotting G1R splitview: {curr_filename}")
                        plot_single_hold_splitview(
                            data_per_file["G1R"]["data"],
                            forces_g1,
                            curr_filename + optional_suffix,
                            grip_label="rechts",
                            save_plot=plot_settings["save"],
                            save_folder=save_folder,
                            cutoff=cutoff,
                            normalizebyweight=normalizebyweight,
                            show_contact_time=["show_impulses"]
                        )

            # Zeige alle erzeugten Plots
            plt.show()



if __name__ == "__main__":
    main()