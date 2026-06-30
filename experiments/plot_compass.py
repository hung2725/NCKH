import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

WORKSPACE_DIR = Path("D:/Hoc_Tap/NCKH")
CSV_PATH = WORKSPACE_DIR / "results/compass_results_lv_volume.csv"
FIG_PATH = WORKSPACE_DIR / "results/figures/compass_vs_scp_comparison.png"

def plot_compass_results():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # Calculate metrics
    compass_cov = df['compass_covered'].mean() * 100
    compass_width = df['compass_width'].mean()
    
    scp_cov = df['scp_covered'].mean() * 100
    scp_width = df['scp_width'].mean()
    
    target_cov = 90.0 # From alpha=0.1
    
    print("=" * 50)
    print("SUMMARY METRICS FOR REPORT:")
    print("=" * 50)
    print(f"Target Coverage: {target_cov}%")
    print("-" * 50)
    print(f"Split CP (Baseline):")
    print(f"  Coverage: {scp_cov:.2f}%")
    print(f"  Average Width: {scp_width:.2f} mL")
    print("-" * 50)
    print(f"COMPASS-L (Feature Perturbation):")
    print(f"  Coverage: {compass_cov:.2f}%")
    print(f"  Average Width: {compass_width:.2f} mL")
    print("=" * 50)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Conformal Prediction: COMPASS vs Split CP (LV Volume)', fontsize=14, fontweight='bold')
    
    methods = ['Split CP (Baseline)', 'COMPASS-L (SOTA)']
    colors = ['#E74C3C', '#2ECC71']
    
    # Coverage Plot
    ax1 = axes[0]
    bars1 = ax1.bar(methods, [scp_cov, compass_cov], color=colors, alpha=0.85, edgecolor='black')
    ax1.axhline(target_cov, color='black', linestyle='--', label=f'Target ({target_cov}%)')
    ax1.set_ylabel('Empirical Coverage (%)')
    ax1.set_ylim(0, 110)
    ax1.set_title('Coverage Comparison')
    ax1.legend()
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')
                 
    # Width Plot
    ax2 = axes[1]
    bars2 = ax2.bar(methods, [scp_width, compass_width], color=colors, alpha=0.85, edgecolor='black')
    ax2.set_ylabel('Average Interval Width (mL)')
    ax2.set_title('Prediction Interval Width Comparison\n(Smaller is better)')
    # Set y-limit with some headroom
    max_width = max(scp_width, compass_width)
    ax2.set_ylim(0, max_width * 1.2)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max_width * 0.02),
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontweight='bold')
                 
    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {FIG_PATH}")

if __name__ == "__main__":
    plot_compass_results()
