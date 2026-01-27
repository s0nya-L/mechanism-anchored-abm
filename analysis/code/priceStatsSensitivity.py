import pandas as pd
import numpy as np

decay_rates = {
    0.90: "High Urgency (10% Decay)",
    0.95: "Base (5% Decay)",
    0.99: "Low Urgency (1% Decay)"
}

groups = [1, 2]
data_dir = "data/finalResults/"
all_data = []

for decay_multiplier, decay_label in decay_rates.items():
    decay_int = int(decay_multiplier * 100)

    for group_id in groups:
        file_name = f"{data_dir}{decay_int}simRESULTSgrp{group_id}_CLEAN.csv"

        try:
            df = pd.read_csv(file_name)
            if not df.empty:
                df.loc[:, 'Group'] = str(group_id)
                df.loc[:, 'Decay_Rate'] = decay_label
                all_data.append(df)
        except FileNotFoundError:
            continue

if not all_data:
    print("\nError: No sensitivity analysis files were successfully loaded. Check file paths and names.")
else:
    df_combined = pd.concat(all_data, ignore_index=True)

    summary_stats = df_combined.groupby(['Decay_Rate', 'Group'])['final_price'].agg(
        Mean=('mean'),
        Median=('median'),
        STDEV=('std')
    ).reset_index()

    summary_stats['CV (%)'] = (summary_stats['STDEV'] / summary_stats['Mean']) * 100

    for col in ['Mean', 'Median', 'STDEV', 'CV (%)']:
        if col in summary_stats.columns:
            summary_stats[col] = summary_stats[col].round(2)

    summary_stats = summary_stats[['Decay_Rate', 'Group', 'Mean', 'Median', 'STDEV', 'CV (%)']]

    output_filepath = "analysis/sensitivityPriceStats.csv"
    summary_stats.to_csv(output_filepath, index=False)

    print(f"Summary statistics saved to {output_filepath}")
