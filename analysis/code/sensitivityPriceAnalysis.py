import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

decay_rates = {
    0.90: "High Urgency (10% Decay)",
    0.95: "Base (5% Decay)",
    0.99: "Low Urgency (1% Decay)"
}

groups = [1, 2]
data_dir = "data/finalResults/" 
all_data = []

print("--- Loading Sensitivity Analysis Data ---")
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
                print(f"Loaded: {file_name}")
        except FileNotFoundError:
            print(f"File not found: {file_name}. Skipping.")
            continue

if not all_data:
    print("\nError: No sensitivity analysis files were successfully loaded. Check file paths and names.")
else:
    df_combined = pd.concat(all_data, ignore_index=True)
    
    df_combined['Group'] = df_combined['Group'].astype('category')
    df_combined['Decay_Rate'] = pd.Categorical(df_combined['Decay_Rate'], categories=["High Urgency (10% Decay)", "Base (5% Decay)", "Low Urgency (1% Decay)"], ordered=True)
    
    formula = 'final_price ~ C(Group) * C(Decay_Rate)'
    lm = ols(formula, data=df_combined).fit()
    
    print("\n--- Two-Way ANOVA Results for Final Price ---")
    anova_table = sm.stats.anova_lm(lm, typ=2) 
    print(anova_table)

    anova_table.to_csv("analysis/ANOVASensitivity2Way.csv")

    # Set font to serif for journal standards
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.figure(figsize=(12, 8))
    ax = sns.pointplot(data=df_combined, x='Decay_Rate', y='final_price', hue='Group',
                       errorbar=('ci', 95), dodge=0.1, capsize=0.05, palette="viridis",
                       markers=['o', 's'], linestyles=['-', '--'])

    plt.title('Impact of Information Group and Decay Rate on Final Transaction Prices', fontsize=16, fontweight='bold')
    plt.xlabel('Urgency Condition (Decay Rate)', fontsize=14)
    plt.ylabel('Mean Final Price ($)', fontsize=14)
    plt.legend(title='Information Group', fontsize=12, title_fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Annotate ANOVA results
    p_group = anova_table.loc['C(Group)', 'PR(>F)']
    p_decay = anova_table.loc['C(Decay_Rate)', 'PR(>F)']
    p_interaction = anova_table.loc['C(Group):C(Decay_Rate)', 'PR(>F)']

    annotation_text = f'ANOVA Results:\nGroup effect: p = {p_group:.3f}\nDecay effect: p = {p_decay:.3f}\nInteraction: p = {p_interaction:.3f}'
    plt.annotate(annotation_text, xy=(0.75, 0.98), xycoords='axes fraction',
                 verticalalignment='top', horizontalalignment='right', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig("analysis/SensitivityPrice.pdf", dpi=300, bbox_inches='tight')
    plt.show()
