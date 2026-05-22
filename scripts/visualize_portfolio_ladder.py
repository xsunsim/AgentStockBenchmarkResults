import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
OUTPUT_PATH = Path("AgentStockBenchmarkResults/leaderboard/portfolio_ladder.png")
MAX_DOLLARS = 250.0

def generate_illustrative_ladder():
    # We'll simulate 20 positions for the visual but label them as 1,2,3...501,502,503
    # Left side (Short): Ranks 503, 502, 501
    # Right side (Long): Ranks 3, 2, 1
    
    # Values for the 6 bars we will actually show
    positions = [-250.0, -240.0, -230.0, 230.0, 240.0, 250.0]
    labels = ["Stock Z\n(Rank 503)", "Stock Y\n(Rank 502)", "Stock X\n(Rank 501)", 
              "Stock C\n(Rank 3)", "Stock B\n(Rank 2)", "Stock A\n(Rank 1)"]
    
    x = [1, 2, 3, 7, 8, 9] # Indices with a gap in the middle
    colors = ['#d62728', '#d62728', '#d62728', '#2ca02c', '#2ca02c', '#2ca02c']

    # 2. Setup Figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    plt.style.use('bmh')
    
    # 3. Plot bars
    ax.bar(x, positions, color=colors, width=0.8, alpha=0.9)
    
    # 4. Middle Omission
    ax.text(5, 0, "...", fontsize=40, fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(5, -50, "(~500 stocks total)", fontsize=10, style='italic', horizontalalignment='center')

    # 5. Add Labels to bars
    for i, label in enumerate(labels):
        y_pos = positions[i]
        va = 'bottom' if y_pos > 0 else 'top'
        offset = 10 if y_pos > 0 else -10
        ax.text(x[i], y_pos + offset, label, ha='center', va=va, fontsize=9, fontweight='bold')

    # 6. Directional Arrows
    # Top of chart - Left (Low Rank)
    ax.annotate('Low Rank\n(Short side)', xy=(0.5, 300), xytext=(3.5, 300),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=11, ha='center', va='center')
    
    # Top of chart - Right (High Rank)
    ax.annotate('High Rank\n(Long side)', xy=(9.5, 300), xytext=(6.5, 300),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=11, ha='center', va='center')

    # 7. Formatting
    ax.set_title("How the Portfolio is Constructed", fontsize=18, fontweight='bold', pad=40)
    ax.set_ylabel("Fixed Dollar Allocation ($)", fontsize=12)
    
    # Grid and axis lines
    ax.axhline(0, color='black', linewidth=1.5, alpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Clean up axes
    ax.set_xticks([])
    ax.set_xlim(0, 10)
    ax.set_ylim(-350, 350)
    
    # Add dollar markers on Y axis
    ax.set_yticks([-250, -125, 0, 125, 250])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches='tight')
    print(f"Successfully generated illustrative portfolio ladder plot at {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_illustrative_ladder()
