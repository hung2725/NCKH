import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv('results/full_comparison_results.csv')

methods = [
    ('scp',      'Split CP (Baseline)'),
    ('crc',      'CRC (Theorem 2.1)'),
    ('ascp',     'Adaptive SCP'),
    ('acrc',     'Adaptive CRC'),
    ('compassl', 'COMPASS-L'),
    ('compassj', 'COMPASS-J'),
]

ALPHA = 0.1
print()
print('='*68)
print('SUMMARY RESULTS (ACDC - LV Volume, Target Coverage = 90%)')
print('='*68)
print(f'  {"Method":<38} {"Coverage":>10} {"Avg Width":>12} {"vs SCP":>8}')
print('-'*68)

scp_width = df['scp_width'].mean()
covs, wids = [], []
valid_methods = [(k,n) for k,n in methods if f'{k}_width' in df.columns]
min_wid = min(df[f'{k}_width'].mean() for k,_ in valid_methods)

for key, name in valid_methods:
    cov = df[f'{key}_covered'].mean() * 100
    wid = df[f'{key}_width'].mean()
    diff = (wid - scp_width) / scp_width * 100
    covs.append(cov)
    wids.append(wid)
    best = ' <-- BEST' if abs(wid - min_wid) < 1e-6 else ''
    print(f'  {name:<38} {cov:>8.1f}%  {wid:>10.2f} mL  {diff:>+7.1f}%{best}')

print('='*68)

# Plot
labels = [n for _,n in valid_methods]
colors = ['#E74C3C','#3498DB','#E67E22','#9B59B6','#1ABC9C','#2ECC71']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Conformal Prediction Methods - ACDC LV Volume (5-Fold CV)',
             fontsize=13, fontweight='bold')

ax1 = axes[0]
bars1 = ax1.bar(labels, covs, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
ax1.axhline(90.0, color='red', linestyle='--', linewidth=1.5, label='Target 90%')
ax1.set_ylabel('Empirical Coverage (%)', fontsize=12)
ax1.set_ylim(80, 108)
ax1.set_title('Coverage Comparison', fontsize=12)
ax1.legend(fontsize=10)
plt.setp(ax1.get_xticklabels(), rotation=20, ha='right', fontsize=9)
for bar, val in zip(bars1, covs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2 = axes[1]
bars2 = ax2.bar(labels, wids, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
ax2.set_ylabel('Average Interval Width (mL)', fontsize=12)
ax2.set_title('Interval Width (Smaller = Better)', fontsize=12)
ax2.set_ylim(0, max(wids) * 1.25)
plt.setp(ax2.get_xticklabels(), rotation=20, ha='right', fontsize=9)
for bar, val in zip(bars2, wids):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
out = 'results/figures/full_comparison_all_methods.png'
Path('results/figures').mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Figure saved to {out}')
