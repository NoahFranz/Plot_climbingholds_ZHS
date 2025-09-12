# config.py
file_path = ""
save_folder = ""
file_number = "000"
optional_suffix = ""
current_folder = ""
processed_files_list = []
force_to_plot = ["Fx", "Fy", "Fz", "Mz", "Fres_yz", "Fres_xyz"]
holds_to_plot = ["G1R", "G2L"]
create_hausdorff_plots = False  # Default setting for Hausdorff plots
filter_suffix = "noSVG"  # Default setting for SVG filtered plots
invalid_intervals_list = []
NBW = ""

gui_invalid_intervals_override = None  # Wird gesetzt, wenn die GUI benutzerdefinierte Intervalle vorgibt

processing_settings = {}

deebug_mode = False  # Default setting for debug mode

COLOR_MAPPING = {
    "Fy": "orange",
    "Fy_sum": "#CC8400",
    "Fx": "green",
    "Fx_sum": "#006400",
    "Fz": "blue",
    "Fz_sum": "#00008B",
    "Mz": "#B30930",  # Kaminrot
    "FgR": "#8E08A2",
    "FgR_1": "#8E08A2",  # Lime
    "FgR_2": "#8E08A2",  # 
    "FgR_sum": "#5D056A",
    "FgR_calc": "#32CD32",
    "Fres_yz": "#4B0082",     # Indigo
    "φ_yz": "#800080",      # Lila
    "Fres_xyz": "#9C1A1A",     # Orange-Rot
    "Fres_xyz_sum": "#690B0B"     # Orange-Rot
    
}

file_acronyms_map = {
    
    "001": "Trail-w-f",
    "002": "Trail-w-c",
    "003": "Basic-w-f",
    "004": "Basic-w-c",
    "005": "Perf-w-f",
    "006": "Perf-w-c",
    "007": "HE-w-f",
    "008": "HE-w-c",
    "009": "Trail-m-f",
    "010": "Trail-m-c",
    "011": "Basic-m-f",
    "012": "Basic-m-c",
    "013": "Perf-m-f",
    "014": "Perf-m-c",
    "015": "HE-m-f",
    "016": "HE-m-c",

    "017":"Trail-b-f",
    "019":"Basic-b-f",
    "021":"Perf-b-f",
    "023":"HE-b-f",
    "018":"Trail-b-c",
    "020":"Basic-b-c",
    "022":"Perf-b-c",
    "024":"HE-b-c",
    "025":"025-Low-gl",
    "026":"026-High-gl",
    "027":"027-Low-gr",
    "028":"028-High-gr",
    "029":"Frontal",
    "030":"Heelhook",
    "031":"Crosstep",
    "032":"Backflag",
    "033":"Rockover",
    "039":"039-RedS-E",
    "040":"040-YelP-G",
    "041":"041-RedI-G",
    "042":"042-OrI-C", 
    "043":"043-RedF-C",
    "045":"045-W0-H",
    "046":"046-W15-C",
    "047":"047-W30-G",
    "048":"048-W45-G",
    "049":"049-2FH",
    "050":"050-1FH",
    "051":"051-switch",
    "052":"052-2FH",
    "053":"053-1FH",
    "054":"054-switch",
    "055": "055-HtR",
    "056": "056-MtR",
    "057": "057-LtR",
    "058": "058-Ht15",
    "059": "059-Mt15",
    "060": "060-Lt15",
    "061": "061-Ht0",
    "062": "062-Mt0",
    "063": "063-Lt0",
    "065": "065-YP",
    "066": "066-W0",
    "067": "067-RMJ",
}


technique_plot_names = {
    "029":"029-Frontal",
    "030":"030-Heelhook",
    "031":"031-Crosstep",
    "032":"032-Backflag",
    "033":"033-Rockover",
 }

#all_shoes_plot_order = ["001", "002", "003", "004", "005", "006"]
COLOR_MAPPING_FH = {   
    "-w-": "#000000",
    "-b-": "#262525",
    "-w-": "#696767",
    
}

file_order_bar_holds = ["039", "040", "041", "048", "047", "043", "042", "046", "045"]
order_for_wooden_holds = ["048","047", "046", "045"]
plot_only_wooden_holds: bool = False  # Default setting for plotting only wooden holds

use_custom_bar_order: bool = True # needs to true when setting your own order of plots

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
    "034-Technik-6-leftside_frontal_73kg_Noah_lvl-ADV": [2, 3],
    "036-Technik-8-leftside_crosstep_73kg_Noah_lvl-ADV": [1],
    "017-Shoes-3_1_FH-best_Sh-trail_TK-front_73kg_Noah": [2,3],
    "001-Shoes-1_1_FH-Bad_Sh-trail_TK-front_73kg_Noah": [3],
    #"053-shakeout_2-2_left_hold_1_fh_73kg_Noah_lvl-ADV_25-05-14_1125": [2]
    #"049-shakeout_1-1_right_hold_2_fh_73kg_Noah": [2]

}