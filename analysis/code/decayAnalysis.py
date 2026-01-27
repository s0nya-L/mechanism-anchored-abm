
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


file_names = {
    0: "data/simulationState/95simSTOCKgrp0.csv",
    1: "data/simulationState/95simSTOCKgrp1.csv",
    2: "data/simulationState/95simSTOCKgrp2.csv"
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
    df_combined.rename(columns={'decayed stock': 'decayed_stock'}, inplace=True)
    df_combined['group'] = df_combined['group'].astype(str)
    unique_groups = df_combined['group'].nunique()
    
    if unique_groups < 2:
        print(f"Error: Only {unique_groups} group(s) found. ANOVA/Tukey's requires at least two groups.")
    else:
 
        print("--- Running ANOVA and Tukey's HSD... ---")
        formula = 'decayed_stock ~ C(group)'
        lm = ols(formula, data=df_combined).fit()
        anova_table = sm.stats.anova_lm(lm, typ=2)
        

        anova_table.to_csv("analysis/ANOVAdecayResults.csv")
        print("ANOVA results saved to ANOVAdecayResults.csv")


        p_value = anova_table.iloc[0, -1] 
        
        if p_value < 0.05:
            m_comp = pairwise_tukeyhsd(endog=df_combined['decayed_stock'], groups=df_combined['group'], alpha=0.05)

         
            all_pairs = list(itertools.combinations(sorted(df_combined['group'].unique()), 2))


            tukey_df = pd.DataFrame({
                'group1': [p[0] for p in all_pairs],
                'group2': [p[1] for p in all_pairs],
                'meandiff': m_comp.meandiffs,
                'lower_ci': m_comp.confint[:, 0],
                'upper_ci': m_comp.confint[:, 1],
                'reject': m_comp.reject
            })
            

            tukey_df.to_csv("analysis/TukeyHSDdecayPairs.csv", index=False)
            print("Tukey's HSD results saved to TukeyHSDdecayPairs.csv")
        else:
            print(f"ANOVA is NOT significant (p={p_value:.4f}). Skipping Tukey's HSD save.")



    plt.figure(figsize=(12, 8))
    group_order = sorted(df_combined['group'].unique())

    sns.violinplot(x='group', y='decayed_stock', data=df_combined, order=group_order,
                   hue='group', palette="viridis", inner='box', linewidth=1.5, alpha=0.8, legend=False)

    plt.title('Distribution of Decayed Stock by Information Group', fontsize=14, fontweight='bold')
    plt.xlabel('Information Group', fontsize=12)
    plt.ylabel('Decayed Stock', fontsize=12)
    plt.ylim(0, None)  
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    p_value = anova_table.iloc[0, -1]  
    annotation_text = f'ANOVA Results:\nGroup effect: p = {p_value:.3f}'
    plt.annotate(annotation_text, xy=(0.75, 0.98), xycoords='axes fraction',
                 verticalalignment='top', horizontalalignment='right', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

 
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.tight_layout()
    plt.savefig("analysis/DecayedStockViolinPlot.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    print("Violin plot saved as DecayedStockViolinPlot.pdf")
