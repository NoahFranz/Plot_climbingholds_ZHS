# config.py
file_path = ""
save_folder = ""
file_number = "000"
optional_suffix = ""
current_folder = ""
processed_files_list = []
force_to_plot = ["Fx", "Fy", "Fz", "Mz", "Fres_yz", "Fres_xyz"]
create_hausdorff_plots = False  # Default setting for Hausdorff plots
filter_suffix = "noSVG"  # Default setting for SVG filtered plots
invalid_intervals_list = []

processing_settings = {}

deebug_mode = False  # Default setting for debug mode

COLOR_MAPPING = {
    "Fy": "orange",
    "Fx": "green",
    "Fz": "blue",
    "Mz": "#DB23EB",  # Kaminrot
    "FgR": "#02A15C",
    "FgR_1": "#C95C02",  # Lime
    "FgR_2": "#0032C7",  # 
    "FgR_sum": "#898111",
    "FgR_calc": "#32CD32",
    "Fres_yz": "#4B0082",     # Indigo
    "φ_yz": "#800080",      # Lila
    "Fres_xyz": "#9C1A1A"     # Orange-Rot
    
}

file_acronyms_map = {
    "039":"039-RedS-E",
    "040":"040-YelP-G",
    "041":"041-RedI-G",
    "042":"042-OrI-C", 
    "043":"043-RedF-C",
    "045":"045-W0-H",
    "046":"046-W15-C",
    "047":"047-W30-G",
    "048":"048-W45-G",
}


file_order_bar_holds = ["039", "040", "041", "048", "047", "043", "042", "046", "045"]
order_for_wooden_holds = ["048","047", "046", "045"]
plot_only_wooden_holds: bool = False  # Default setting for plotting only wooden holds

use_custom_bar_order: bool = True # dont change this

# setting y limits for plots
manual_y_limits_var = False
y_min = 0
y_max = 60

# show title in saved plots
show_title_in_plots = False

plot_settings = {}
filter_settings = {}


excluded_intervals_dict = {
    "031-Technik-3-rightside_crosstep_73kg_Noah": [2],
    "032-Technik-4-rightside_backflag_73kg_Noah": [2],
    "034-Technik-6-leftside_frontal_73kg_Noah_lvl-ADV_25-05-13_1821": [2, 3],
    "036-Technik-8-leftside_crosstep_73kg_Noah_lvl-ADV_25-05-13_1830": [1],
}