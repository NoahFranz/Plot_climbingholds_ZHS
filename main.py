from gui import*
from loadData import load_lvm_data
from utils import *
from plotdata import *
import os


def main():
    plot_settings, holds_to_plot, forces_to_plot, filter_settings, file_paths, cutoff = run_gui()
    
    print("forces_to_plot:", forces_to_plot, "holds to plot", holds_to_plot)

    folder_path = file_paths["data_folder"] or "/Users/noah/LRZ Sync+Share/MA/ZHS_ LabView_Messungen/Tests/Test tag 2"
    save_folder = file_paths["save_folder"] or "/Users/noah/LRZ Sync+Share/MA/Plot_Figures"
    optional_suffix = file_paths["suffix"] or ""
    optional_suffix = optional_suffix + get_force_suffix(forces_to_plot)
    cutoff = {"start": cutoff["start"], "end": cutoff["end"]}

    sorted_data_dict, filtered_data_dict = load_lvm_data(folder_path, filter_settings["window_length"], filter_settings["polyorder"])

    if filter_settings["use_filter"]:
        current_dict = filtered_data_dict
        optional_suffix += "_filtered_" + str(filter_settings["window_length"]) + "_" + str(filter_settings["polyorder"])
    else:
        current_dict = sorted_data_dict
        optional_suffix += "_raw"

    print("------------------ current_dict in main end ------------------")
    if current_dict is None:
        print("Keine .lvm-Dateien gefunden.")
        return


    forces_g1 = [k for k, v in forces_to_plot["G1"].items() if k != "all" and v]
    forces_g2 = [k for k, v in forces_to_plot["G2"].items() if k != "all" and v]

    

    # compute globale limits for y_limits and save it with every Hold such that every dataframe can access it
    for filename, file_data in current_dict.items():
        current_dict[filename] = compute_global_ylimits_for_plots(file_data, forces_g1, forces_g2)
    for key in current_dict:
        print("Current_dict AFTER: global y_limits")
        print(f"{key} →")
        for side, content in current_dict[key].items():
            print(f"  {side}: {list(content.keys())}")
            print(f"    Spalten: {content['data'].columns.tolist()}")
                # global_limits anzeigen, falls vorhanden
            if "global_limits" in content:
                gl = content["global_limits"]
                print(f"    global_limits: min={gl['global_y_min']:.2f}, max={gl['global_y_max']:.2f}")
            else:
                print("    global_limits: Nicht gesetzt")

    if plot_settings["create"]:
        if not plot_settings["split_fmz"] and not plot_settings["compare_forces"]:
            for curr_filename, data_per_file in current_dict.items():
                print(f"Plotting GL+GR: {curr_filename}")
                plot_data_per_hold(data_per_file, forces_g1, forces_g2, curr_filename + optional_suffix, save_plot=plot_settings["save"], save_folder=save_folder, cutoff=cutoff)
        elif not plot_settings["split_fmz"] and plot_settings["compare_forces"]:
            for filename, file_data in current_dict.items():
                plot_selected_forces_comparison(file_data, forces_g1, forces_g2, filename=filename, save_folder=save_folder, save_plot=plot_settings["save"], cutoff=cutoff)
        else:
            for curr_filename, data_per_file in current_dict.items():
                if holds_to_plot["G2"]:
                    print(f"Plotting G2L splitview: {curr_filename}")
                    plot_single_hold_splitview(data_per_file["G2L"]["data"], forces_g2, curr_filename + optional_suffix, grip_label="GL", save_plot=plot_settings["save"], save_folder=save_folder, cutoff=cutoff)
                if holds_to_plot["G1"]:
                    print(f"Plotting G1R splitview: {curr_filename}")
                    plot_single_hold_splitview(data_per_file["G1R"]["data"], forces_g1, curr_filename + optional_suffix, grip_label="GR", save_plot=plot_settings["save"], save_folder=save_folder, cutoff=cutoff)

        plt.show()



if __name__ == "__main__":
    main()