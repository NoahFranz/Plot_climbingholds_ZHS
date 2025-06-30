import pandas as pd
import glob
import os
import openpyxl
import re
from openpyxl.styles import numbers


def extract_fy1_std_values(xls):
    fy1_std_values = []
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        std_row = df[df.iloc[:, 0].astype(str).str.lower() == 'std']
        if not std_row.empty and 'Fy_1 [N]' in std_row.columns:
            std_value = pd.to_numeric(std_row['Fy_1 [N]'].values[0], errors='coerce')
            fy1_std_values.append(std_value)
    return fy1_std_values

def compute_mean_components(xls, components):
    component_means = {comp: [] for comp in components}
    sheet_names = []
    for sheet_name in xls.sheet_names:
        if not sheet_name.startswith("S"):
            continue
        df = xls.parse(sheet_name)
        sheet_names.append(sheet_name)
        for comp in components:
            if comp in df.columns:
                mean_val = pd.to_numeric(df[comp], errors='coerce').mean()
            else:
                mean_val = None
            component_means[comp].append(mean_val)
    mean_df = pd.DataFrame(component_means, index=sheet_names).T
    mean_df = mean_df.round(1)
    return mean_df, sheet_names

def compute_relative_components(mean_df, basename):
    import re
    main_force = None
    g_match = re.search(r'(G\d+)', basename)
    g_number = None
    if g_match:
        g_number = g_match.group(1)[1:]  # Extract just the number part after 'G'
    for force_tag in ['Fx', 'Fy', 'Fz']:
        if force_tag in basename and g_number is not None:
            main_force = f"{force_tag}_{g_number} [N]"
            break
    if main_force and main_force in mean_df.index:
        relative_df = mean_df.copy()
        main_values = mean_df.loc[main_force]
        for comp in mean_df.index:
            if comp != main_force:
                relative_df.loc[comp] = (mean_df.loc[comp] *100 / main_values)
        relative_df = relative_df.apply(pd.to_numeric, errors="coerce").round(0).astype("Int64")
        # Rename indices ending with '[N]' or '[Nm]' to end with '[%]', except main_force
        def rename_index(label):
            if label == main_force:
                return label
            if label.endswith('[N]'):
                return label[:-3] + '[%]'
            elif label.endswith('[Nm]'):
                return label[:-4] + '[%]'
            else:
                return label
        relative_df = relative_df.rename(index=rename_index)
        relative_df["mean"] = relative_df.mean(axis=1, skipna=True).round(0).astype("Int64")
    else:
        relative_df = None
    return relative_df

excel_folder = "/Users/noah/LRZ Sync+Share/MA/Plot_Figures/System_characteristics/Cross_Talk_plots/G1_static/excel stats"
excel_files = sorted(
    f for f in glob.glob(os.path.join(excel_folder, "*.xlsx"))
    if not os.path.basename(f).startswith("~$")
)

for file_path in excel_files:
    print(f"Processing {os.path.basename(file_path)}")
    xls = pd.ExcelFile(file_path)

    fy1_std_values = extract_fy1_std_values(xls)
   # print("STD values used for Fy_1 [N]:", [round(val, 3) for val in fy1_std_values])
    mean_std_fy1 = sum(fy1_std_values) / len(fy1_std_values)
    #print(f"Mean STD of Fy_1 [N]: {mean_std_fy1:.3f} N")

    basename = os.path.basename(file_path)
    g_match = re.search(r'G(\d+)', basename)
    if g_match:
        g_number = g_match.group(1)
    else:
        g_number = '1'  # fallback if no G number found
    components = [f'Fy_{g_number} [N]', f'Fz_{g_number} [N]', f'Fx_{g_number} [N]', f'Mz_{g_number} [Nm]']
    mean_df, sheet_names = compute_mean_components(xls, components)

    relative_df = compute_relative_components(mean_df, basename)

    output_path = file_path
    if relative_df is not None:
        with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            mean_df.to_excel(writer, sheet_name='Mean_Components')
            relative_df.to_excel(writer, sheet_name='Mean_Components_Relative')
    else:
        with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            mean_df.to_excel(writer, sheet_name='Mean_Components')