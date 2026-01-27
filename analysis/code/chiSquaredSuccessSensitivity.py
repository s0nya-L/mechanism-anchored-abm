from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

TOTAL_POTENTIAL_DEALS = 600

decay_configs = {
    90: "High Urgency (10% Decay)",
    95: "Base (5% Decay)",
    99: "Low Urgency (1% Decay)"
}
data_dir = "data/finalResults/"
results_list = []

for decay_int, decay_label in decay_configs.items():
    file_g1 = f"{data_dir}{decay_int}simRESULTSgrp1_CLEAN.csv"
    file_g2 = f"{data_dir}{decay_int}simRESULTSgrp2_CLEAN.csv"

    success_counts = {}

    try:
        df_g1 = pd.read_csv(file_g1)
        success_counts[1] = len(df_g1)
    except FileNotFoundError:
        success_counts[1] = "File Not Found"

    try:
        df_g2 = pd.read_csv(file_g2)
        success_counts[2] = len(df_g2)
    except FileNotFoundError:
        success_counts[2] = "File Not Found"

    if any(isinstance(count, str) for count in success_counts.values()):
        results_list.append({
            'Decay_Rate': decay_label,
            'G1_Successes': success_counts.get(1, 'N/A'),
            'G2_Successes': success_counts.get(2, 'N/A'),
            'Chi2': 'N/A',
            'P_value': 'N/A',
            'Significant': 'N/A'
        })
    else:
        successful_deals = [success_counts[1], success_counts[2]]
        failed_deals = [TOTAL_POTENTIAL_DEALS - s for s in successful_deals]

        contingency_table = np.array([successful_deals, failed_deals])

        chi2, p, dof, expected = chi2_contingency(contingency_table)

        is_significant = 'Yes' if p < 0.05 else 'No'

        results_list.append({
            'Decay_Rate': decay_label,
            'G1_Successes': success_counts[1],
            'G2_Successes': success_counts[2],
            'Chi2': round(chi2, 3),
            'P_value': round(p, 5),
            'Significant': is_significant
        })

df_final_results = pd.DataFrame(results_list)

output_filepath = "analysis/sensitivityChiSquareResults.csv"
df_final_results.to_csv(output_filepath, index=False)

# Prepare data for plotting success rates
plot_data = []
for r in results_list:
    if r['G1_Successes'] != 'N/A':
        plot_data.append({
            'Decay_Rate': r['Decay_Rate'],
            'Group': 'Group 1',
            'Success_Rate': r['G1_Successes'] / TOTAL_POTENTIAL_DEALS * 100
        })
        plot_data.append({
            'Decay_Rate': r['Decay_Rate'],
            'Group': 'Group 2',
            'Success_Rate': r['G2_Successes'] / TOTAL_POTENTIAL_DEALS * 100
        })

if plot_data:
    df_plot = pd.DataFrame(plot_data)
    df_plot['Decay_Rate'] = pd.Categorical(df_plot['Decay_Rate'], categories=["High Urgency (10% Decay)", "Base (5% Decay)", "Low Urgency (1% Decay)"], ordered=True)

    # Set font to serif for journal standards
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.figure(figsize=(12, 8))
    ax = sns.pointplot(data=df_plot, x='Decay_Rate', y='Success_Rate', hue='Group',
                       palette=sns.color_palette("viridis", 2), markers=['o', 's'], linestyles=['-', '--'],
                       dodge=0.1, capsize=5, errorbar=('ci', 95), alpha=0.8)

    plt.title('Impact of Information Group and Decay Rate on Negotiation Success Rates', fontsize=16, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.xlabel('Urgency Condition (Decay Rate)', fontsize=14)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title='Information Group', fontsize=12, title_fontsize=13)

    # Annotate chi-square results
    annotation_lines = []
    for r in results_list:
        if r['P_value'] != 'N/A':
            decay_short = r['Decay_Rate'].split(' (')[0]  # e.g., "High Urgency"
            p_val = f"{r['P_value']:.3f}" if r['P_value'] >= 0.001 else "<0.001"
            annotation_lines.append(f"{decay_short}: p = {p_val}")

    annotation_text = 'Chi-square Results:\n' + '\n'.join(annotation_lines)
    plt.annotate(annotation_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plot_filepath = "analysis/sensitivitySuccessRateChart.pdf"
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Results saved to {output_filepath} and {plot_filepath}")
else:
    print(f"Results saved to {output_filepath}")
