import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools


file_names = {
    0: "data/finalResults/95simRESULTSgrp0_CLEAN.csv",
    1: "data/finalResults/95simRESULTSgrp1_CLEAN.csv",
    2: "data/finalResults/95simRESULTSgrp2_CLEAN.csv"
}

all_data = []


for group_id, file_name in file_names.items():
    try:
        df = pd.read_csv(file_name)
        df['round'] = df['round'] + 1  
        df['group'] = str(group_id)
        all_data.append(df)
    except FileNotFoundError:
        print(f"File not found: {file_name}. Skipping.")
        continue

if not all_data:
    print("Error: No data files were successfully loaded.")
else:
    df_combined = pd.concat(all_data, ignore_index=True)
    unique_groups = df_combined['group'].nunique()

    if unique_groups < 2:
        print(f"Error: Only {unique_groups} group(s) found. ANOVA requires at least two groups.")
    else:
        
        formula = 'round ~ C(group)'
        lm = ols(formula, data=df_combined).fit()
        anova_table = sm.stats.anova_lm(lm, typ=2)

        anova_table.to_csv("analysis/ANOVAefficiencyResults.csv")
        print("ANOVA results saved to ANOVAefficiencyResults.csv")

       
        p_value = anova_table.iloc[0, -1]

        if p_value < 0.05:
            m_comp = pairwise_tukeyhsd(endog=df_combined['round'], groups=df_combined['group'], alpha=0.05)

 
            all_pairs = list(itertools.combinations(sorted(df_combined['group'].unique()), 2))

            tukey_df = pd.DataFrame({
                'group1': [p[0] for p in all_pairs],
                'group2': [p[1] for p in all_pairs],
                'meandiff': m_comp.meandiffs,
                'lower_ci': m_comp.confint[:, 0],
                'upper_ci': m_comp.confint[:, 1],
                'reject': m_comp.reject
            })

            tukey_df.to_csv("analysis/TukeyHSDefficiencyPairs.csv", index=False)
            print("Tukey's HSD results saved to TukeyHSDefficiencyPairs.csv")
        else:
            print(f"ANOVA is NOT significant (p={p_value:.4f}). Skipping Tukey's HSD save.")


    plt.figure(figsize=(12, 8))
    group_order = sorted(df_combined['group'].unique())


    means = df_combined.groupby('group')['round'].mean()
    cis = df_combined.groupby('group')['round'].apply(lambda x: sm.stats.DescrStatsW(x).tconfint_mean()).apply(pd.Series)

    bars = plt.bar(group_order, means, yerr=(means - cis[0], cis[1] - means),
                   color=sns.color_palette("viridis", 3), capsize=5, alpha=0.8)

    plt.title('Negotiation Efficiency by Information Group (5% Decay Rate)', fontsize=16, fontweight='bold')
    plt.xlabel('Information Group', fontsize=14)
    plt.ylabel('Average Round Number', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

  
    p_value = anova_table.iloc[0, -1]  
    annotation_text = f'ANOVA Results:\nGroup effect: p = {p_value:.3f}'
    plt.annotate(annotation_text, xy=(0.75, 0.98), xycoords='axes fraction',
                 verticalalignment='top', horizontalalignment='right', fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

 
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.tight_layout()
    plt.savefig("analysis/EfficiencyAnalysis.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    print("Efficiency plot saved as EfficiencyAnalysis.pdf")
