import pandas as pd
import numpy as np

file_names = {
    0: "data/finalResults/simRESULTSgrp0_CLEAN.csv",
    1: "data/finalResults/simRESULTSgrp1_CLEAN.csv",
    2: "data/finalResults/simRESULTSgrp2_CLEAN.csv"
}
all_data = []

for group_id, file_name in file_names.items():
    try:
        df = pd.read_csv(file_name)
        if not df.empty:
            df.loc[:, 'group'] = group_id
            all_data.append(df)
    except FileNotFoundError:
        continue

if not all_data:
    print("Error: No transaction results files were successfully loaded.")
else:
    df_combined = pd.concat(all_data, ignore_index=True)
    
    summary_stats = df_combined.groupby('group')['final_price'].agg(
        Mean=('mean'),
        Median=('median'),
        STDEV=('std')
    ).reset_index()

    summary_stats['CV (%)'] = (summary_stats['STDEV'] / summary_stats['Mean']) * 100
    
    summary_stats['Group'] = summary_stats['group'].astype(str)
    summary_stats.drop(columns=['group'], inplace=True)
    
    for col in ['Mean', 'Median', 'STDEV', 'CV (%)']:
        summary_stats[col] = summary_stats[col].round(2)

    summary_stats.to_csv("analysis/priceStats.csv", index=False)
    print(summary_stats)
