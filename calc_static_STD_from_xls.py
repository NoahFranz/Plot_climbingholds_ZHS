import pandas as pd
import os
import re
import numpy as np


# plotting data from raw data as excel files i.e. used for Crosstalk evaluation and plotting.

# === CONFIGURATION ===
input_folder = "/Users/noah/LRZ Sync+Share/MA/Plot_Figures/System_characteristics/Cross_Talk_plots/G2_static/excel stats"
output_file = "/Users/noah/LRZ Sync+Share/MA/Plot_Figures/System_characteristics/Cross_Talk_plots/G2_static/excel stats/G2_summary_statistics.xlsx"

# List of input Excel files
files = [
    "G2_Mz_stat_LR.xlsx",
    "G2_Fz_stat.xlsx",
    "G2_Fy_stat.xlsx",
    "G2_Fx_stat_LR.xlsx"
]

# === PROCESSING ===
# Create Excel writer
writer = pd.ExcelWriter(output_file, engine='openpyxl')

for file in files:
    filepath = os.path.join(input_folder, file)

    # Extract the main force component from filename, e.g., 'Fz' from 'G1_Fz_stat.xlsx'
    match = re.search(r"_([A-Z][a-zA-Z]+)", file)
    if not match:
        continue
    component = match.group(1)

    # Read all sheets from Excel file
    xls = pd.ExcelFile(filepath)
    summary_data = {}

    for sheet_name in xls.sheet_names:
        if not sheet_name.startswith("S"):
            continue  # Only process sheets starting with 'S'

        df = xls.parse(sheet_name)

        # Identify the relevant force column
        force_cols = [col for col in df.columns if isinstance(col, str) and component in col]
        if not force_cols:
            continue
        force_col = force_cols[0]

        # Extract relevant rows
        df_stats = df[df.iloc[:, 0].astype(str).str.lower().isin(['mean', 'max', 'min', 'std'])]
        df_stats = df_stats.set_index(df_stats.columns[0])
        series = df_stats.loc[['mean', 'max', 'min', 'std'], force_col]
        summary_data[sheet_name] = series

    if not summary_data:
        continue

    # Build a summary table
    summary = pd.DataFrame(summary_data)

    # Format rows
    summary.loc['mean'] = np.round(summary.loc['mean'].astype(float), 0)
    summary.loc['max'] = np.round(summary.loc['max'].astype(float), 0)
    summary.loc['min'] = np.round(summary.loc['min'].astype(float), 0)
    summary.loc['std'] = np.round(summary.loc['std'].astype(float), 2)

    # Compute and add relative std
    std_rel = summary.loc['std'] / summary.loc['mean'].replace(0, np.nan) * 100
    std_rel = std_rel.replace([np.inf, -np.inf], np.nan).round(1).astype(str) + '%'

    summary.loc['CoV [%]'] = std_rel

    # Ensure proper row order
    summary = summary.reindex(['mean', 'max', 'min', 'std', 'std rel'])

    # Write to Excel sheet
    sheet_title = os.path.splitext(file)[0]
    try:
        summary.to_excel(writer, sheet_name=sheet_title)
    except Exception as e:
        print(f"Failed to write {sheet_title}: {e}")

# Save the final output
writer.close()

print("Summary file created:", output_file)