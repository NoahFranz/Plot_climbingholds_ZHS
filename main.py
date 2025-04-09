from gui import*
from loadData import load_lvm_data
from utils import clean_data, split_data, get_min_max_values_per_column
from plotdata import *
import os


def main():
    create_plots, save_plots, holds_to_plot, forces_to_plot, split_fmz_var, gui_save_folder, optional_suffix, data_folder, usefilteredDict, SVGwindowlength, SVGpolyorder = run_gui()
    print("forces_to_plot:", forces_to_plot, "holds to plot", holds_to_plot)

    folder_path = data_folder or "/Users/noah/LRZ Sync+Share/MA/ZHS Messungen/Tests/erste Messungen"
    save_folder = gui_save_folder or "/Users/noah/LRZ Sync+Share/MA/Plot_Figures"
    optional_suffix = optional_suffix or ""

    sorted_data_dict, filtered_data_dict = load_lvm_data(folder_path, SVGwindowlength, SVGpolyorder)

    if usefilteredDict:
        current_dict = filtered_data_dict
        optional_suffix += "_filtered_" + str(SVGwindowlength) + "_" + str(SVGpolyorder)
    else:
        current_dict = sorted_data_dict
        optional_suffix += "_raw"

    for key in current_dict:
        print(f"{key} →")
        for side, content in current_dict[key].items():
            print(f"  {side}: {list(content.keys())}")
            print(f"    Spalten: {content['data'].columns.tolist()}")

    print("------------------ current_dict in main end ------------------")
    if current_dict is None:
        print("Keine .lvm-Dateien gefunden.")
        return

    forces_g1 = [k for k, v in forces_to_plot["G1"].items() if k != "all" and v]
    forces_g2 = [k for k, v in forces_to_plot["G2"].items() if k != "all" and v]

    if create_plots:
        if not split_fmz_var:
            for curr_filename, data_per_file in current_dict.items():
                print(f"Plotting GL+GR: {curr_filename}")
                plot_data_per_hold(data_per_file, forces_g1, forces_g2, curr_filename + optional_suffix, save_plot=save_plots, save_folder=save_folder)
        else:
            for curr_filename, data_per_file in current_dict.items():
                if holds_to_plot["G2"]:
                    print(f"Plotting G2L splitview: {curr_filename}")
                    plot_single_hold_splitview(data_per_file["G2L"]["data"], forces_g2, curr_filename + optional_suffix, grip_label="GL", save_plot=save_plots, save_folder=save_folder)
                if holds_to_plot["G1"]:
                    print(f"Plotting G1R splitview: {curr_filename}")
                    plot_single_hold_splitview(data_per_file["G1R"]["data"], forces_g1, curr_filename + optional_suffix, grip_label="GR", save_plot=save_plots, save_folder=save_folder)

        plt.show()



if __name__ == "__main__":
    main()