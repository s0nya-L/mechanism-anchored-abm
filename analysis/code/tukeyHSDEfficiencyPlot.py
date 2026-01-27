import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


tukey_df = pd.read_csv("analysis/csvs/TukeyHSDefficiencyPairs.csv")


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

plt.figure(figsize=(14, 10))


comparisons = [f"Group {row['group1']} vs Group {row['group2']}" for _, row in tukey_df.iterrows()]
y_pos = np.arange(len(comparisons))


bars = plt.barh(y_pos, tukey_df['meandiff'],
                xerr=[tukey_df['meandiff'] - tukey_df['lower_ci'],
                      tukey_df['upper_ci'] - tukey_df['meandiff']],
                color=sns.color_palette("viridis", len(comparisons)),
                alpha=0.8, capsize=6, error_kw={'elinewidth': 2, 'capthick': 2})


plt.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)


plt.yticks(y_pos, comparisons, fontsize=12)
plt.xlabel('Mean Difference (Rounds)', fontsize=14, fontweight='bold')
plt.ylabel('Group Comparisons', fontsize=14, fontweight='bold')
plt.title('Tukey HSD Post-hoc Pairwise Comparisons\nNegotiation Efficiency by Information Group',
          fontsize=18, fontweight='bold', pad=20)

plt.grid(axis='x', linestyle='--', alpha=0.4)


annotation_text = ('Error bars show 95% confidence intervals\n' +
                  'All comparisons are statistically significant')
plt.annotate(annotation_text, xy=(0.02, 0.02), xycoords='axes fraction',
             verticalalignment='bottom', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.8))


for i, (_, row) in enumerate(tukey_df.iterrows()):

    label_x = row['upper_ci'] + max(abs(row['meandiff']) * 0.05, 0.1) 
    plt.text(label_x, i, f'{abs(row["meandiff"]):.2f}',  
             va='center', ha='left', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("analysis/TukeyHSDEfficiencyPlot.pdf", dpi=300, bbox_inches='tight')
plt.show()

print("Tukey HSD efficiency plot saved as TukeyHSDEfficiencyPlot.pdf")
