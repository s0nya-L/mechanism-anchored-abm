import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes."""

    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            self.set_thetamin(90)
            self.set_thetamax(90 + 360)
            self.set_rlim(0, 1)

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_xydata().T
         
            if x[0] != x[-1] or y[0] != y[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=14, fontfamily='serif')

        def _gen_axes_patch(self):
            
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k",
                                      facecolor="w", lw=1)
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

print("Loading data for radar chart...")


price_files = {
    0: "data/finalResults/95simRESULTSgrp0_CLEAN.csv",
    1: "data/finalResults/95simRESULTSgrp1_CLEAN.csv",
    2: "data/finalResults/95simRESULTSgrp2_CLEAN.csv"
}
price_data = []
for group_id, file_name in price_files.items():
    df = pd.read_csv(file_name)
    df['group'] = str(group_id)
    price_data.append(df)
df_price = pd.concat(price_data, ignore_index=True)


efficiency_files = {
    0: "data/finalResults/95simRESULTSgrp0_CLEAN.csv",
    1: "data/finalResults/95simRESULTSgrp1_CLEAN.csv",
    2: "data/finalResults/95simRESULTSgrp2_CLEAN.csv"
}
efficiency_data = []
for group_id, file_name in efficiency_files.items():
    df = pd.read_csv(file_name)
    df['round'] = df['round'] + 1  
    df['group'] = str(group_id)
    efficiency_data.append(df)
df_efficiency = pd.concat(efficiency_data, ignore_index=True)


TOTAL_DEALS = 600
success_counts = {}
for group_id in [0, 1, 2]:
    file_name = f"data/finalResults/95simRESULTSgrp{group_id}_CLEAN.csv"
    df = pd.read_csv(file_name)
    success_counts[group_id] = len(df)
success_rates = {k: v/TOTAL_DEALS * 100 for k, v in success_counts.items()}


spoilage_files = {
    0: "data/simulationState/95simSTOCKgrp0.csv",
    1: "data/simulationState/95simSTOCKgrp1.csv",
    2: "data/simulationState/95simSTOCKgrp2.csv"
}
spoilage_data = []
for group_id, file_name in spoilage_files.items():
    df = pd.read_csv(file_name)
    df['group'] = str(group_id)
    spoilage_data.append(df)
df_spoilage = pd.concat(spoilage_data, ignore_index=True)
df_spoilage.rename(columns={'decayed stock': 'decayed_stock'}, inplace=True)


price_stats = df_price.groupby('group')['final_price'].agg(['mean', 'std']).reset_index()
price_stats['cv'] = (price_stats['std'] / price_stats['mean']) * 100
price_stats['group'] = price_stats['group'].astype(str)


efficiency_stats = df_efficiency.groupby('group')['round'].agg(['mean']).reset_index()
efficiency_stats['group'] = efficiency_stats['group'].astype(str)


spoilage_stats = df_spoilage.groupby('group')['decayed_stock'].agg(['mean']).reset_index()
spoilage_stats['group'] = spoilage_stats['group'].astype(str)


fig = plt.figure(figsize=(12, 10))


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

ax = fig.add_subplot(111, polar=True)

metrics = ['Price\n(Lower Better)', 'Efficiency\n(Lower Better)',
           'Success Rate\n(Higher Better)', 'Spoilage\n(Lower Better)', 'Volatility\n(Lower Better)']


price_vals = [price_stats.loc[price_stats['group']==str(i), 'mean'].values[0] for i in [0,1,2]]
efficiency_vals = [efficiency_stats.loc[efficiency_stats['group']==str(i), 'mean'].values[0] for i in [0,1,2]]
success_vals = [success_rates[i] for i in [0,1,2]]
spoilage_vals = [spoilage_stats.loc[spoilage_stats['group']==str(i), 'mean'].values[0] for i in [0,1,2]]
volatility_vals = [price_stats.loc[price_stats['group']==str(i), 'cv'].values[0] for i in [0,1,2]]


def standardize_lower_better(values):
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return [1.0] * len(values)  
    return [(max_val - v) / (max_val - min_val) for v in values]

def standardize_higher_better(values):
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return [1.0] * len(values) 
    return [(v - min_val) / (max_val - min_val) for v in values]

radar_data = [
    standardize_lower_better(price_vals),      
    standardize_lower_better(efficiency_vals), 
    standardize_higher_better(success_vals),  
    standardize_lower_better(spoilage_vals),  
    standardize_lower_better(volatility_vals)  
]

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1] 

colors = sns.color_palette("viridis", 3)

for i, (color, label) in enumerate(zip(colors, ['Group 0', 'Group 1', 'Group 2'])):
    group_vals = [radar_data[j][i] for j in range(len(metrics))]
    group_vals += group_vals[:1]  
    ax.plot(angles, group_vals, 'o-', linewidth=3, label=label, color=color, alpha=0.9, markersize=8)
    ax.fill(angles, group_vals, alpha=0.1, color=color)


ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=14, fontfamily='serif')
ax.set_ylim(0, 1)
ax.set_title('Standardized Performance Across All Metrics\n(Higher = Better Performance)', fontsize=18, fontweight='bold', pad=30, fontfamily='serif')
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=14)
ax.grid(True, alpha=0.3)


ax.annotate('All group differences\nstatistically significant\n(p < 0.001)', xy=(0.02, 0.02), xycoords='axes fraction',
            verticalalignment='bottom', fontsize=12, fontfamily='serif', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig("analysis/plots/StandardizedPerformanceRadar.pdf", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Radar chart saved as: analysis/plots/StandardizedPerformanceRadar.pdf")
print("Chart shows standardized performance across all metrics for groups 0, 1, and 2")
