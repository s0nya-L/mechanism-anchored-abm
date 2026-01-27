from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

TOTAL_POTENTIAL_DEALS = 600
FILE_NAMES = {
    0: "data/finalResults/95simRESULTSgrp0_CLEAN.csv",
    1: "data/finalResults/95simRESULTSgrp1_CLEAN.csv",
    2: "data/finalResults/95simRESULTSgrp2_CLEAN.csv"
}

success_counts = {}

for group_id, file_name in FILE_NAMES.items():
    try:
        df = pd.read_csv(file_name)
        success_counts[group_id] = len(df)
    except FileNotFoundError:
        success_counts[group_id] = "File Not Found"

if any(isinstance(count, str) for count in success_counts.values()):
    print("Incomplete data: cannot perform analysis.")
else:
    successful_deals = [success_counts[0], success_counts[1], success_counts[2]]
    failed_deals = [TOTAL_POTENTIAL_DEALS - s for s in successful_deals]

    contingency_table = np.array([successful_deals, failed_deals])

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    df_contingency = pd.DataFrame(contingency_table,
                                  index=['Successful Deals', 'Failed Deals'],
                                  columns=[f'Group {g}' for g in range(3)])
    df_contingency.to_csv("analysis/ContingencyTableSuccessRate.csv")

    df_chi2_results = pd.DataFrame({
        'Statistic': ['Chi-squared', 'P-value', 'Degrees of Freedom'],
        'Value': [chi2, p, dof]
    })
    df_chi2_results.to_csv("analysis/ChiSquaredTestResults.csv", index=False)

    success_rates = [s / TOTAL_POTENTIAL_DEALS * 100 for s in successful_deals]
    groups = ['Group 0', 'Group 1', 'Group 2']

    # Set font to serif for journal standards
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.figure(figsize=(12, 8))
    bars = plt.bar(groups, success_rates, color=sns.color_palette("viridis", 3), alpha=0.8)

    plt.title('Negotiation Success Rate by Information Group', fontsize=16, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.xlabel('Information Group', fontsize=14)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=12)

    # Annotate chi-square results
    p_formatted = f"{p:.3f}" if p >= 0.001 else "<0.001"
    annotation_text = f'Chi-square Test:\nχ² = {chi2:.2f}, p = {p_formatted}'
    plt.annotate(annotation_text, xy=(0.75, 0.98), xycoords='axes fraction',
                 verticalalignment='top', horizontalalignment='right', fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plot_file_name = "analysis/successRateChart.pdf"
    plt.savefig(plot_file_name, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Results saved to analysis/ContingencyTableSuccessRate.csv, analysis/ChiSquaredTestResults.csv, and {plot_file_name}")
