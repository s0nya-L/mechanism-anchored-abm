import pandas as pd
import numpy as np

file_name = "data/finalResults/99simRESULTSgrp1.csv"
output_file_name = "data/finalResults/99simRESULTSgrp1_CLEAN.csv"
df = pd.read_csv(file_name)


original_count = len(df)

df['final_price'] = pd.to_numeric(df['final_price'], errors='coerce')
df['final_volume'] = pd.to_numeric(df['final_volume'], errors='coerce')

df_cleaned_stock = df.dropna(subset=['final_price', 'final_volume'])

count_after_stock_clean = len(df_cleaned_stock)

zero_deal_mask = ~((df_cleaned_stock['final_price'] == 0.0) & (df_cleaned_stock['final_volume'] == 0.0))
df_final_clean = df_cleaned_stock[zero_deal_mask].copy()


final_count = len(df_final_clean)


df_final_clean.to_csv(output_file_name, index=False)








print(f"Original Deal Count: {original_count}")
print(f"Count after removing 'STOCK_EXCEEDED' errors: {count_after_stock_clean}")
print(f"Final Clean Deal Count (after removing 0 price/0 volume): {final_count}")
print(f"Total Corrupted/False Deals Removed: {original_count - final_count}")

print(f"\nCleaned data saved to: {output_file_name}")