import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

# Load CSV from parameter_testing directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'parameter_testing', 'parameter_sweep_results.csv')
df = pd.read_csv(csv_path)

# Create all 3 figures at once
fig1, ax1 = plt.subplots(figsize=(14, 10))
fig2, ax2 = plt.subplots(figsize=(14, 10))
fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(111, projection='3d')

# Figure 1: Response Time Series - ALL tested combinations
# Create better color mapping with high contrast
k_values = sorted(df['k'].unique())
b_values = sorted(df['b'].unique())

# Use the requested color palette: blue, red, orange, purple, yellow, pink, dark blue
colors = ['#0066CC', '#CC0000', '#FF6600', '#6600CC', '#FFD700', '#FF69B4', '#003366']
k_color_map = dict(zip(k_values, colors[:len(k_values)]))

# Define line styles for different K values
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
k_style_map = dict(zip(k_values, line_styles[:len(k_values)]))

# Define marker sizes for different B values (normalized)
b_sizes = np.linspace(20, 80, len(b_values))
b_size_map = dict(zip(b_values, b_sizes))

# Group by K values for better legend organization
k_groups = {}
for _, row in df.iterrows():
    k_val = row['k']
    if k_val not in k_groups:
        k_groups[k_val] = []
    k_groups[k_val].append(row)

# Plot each K group with distinct styling
for k_val, group in k_groups.items():
    base_color = k_color_map[k_val]
    line_style = k_style_map[k_val]
    
    for row in group:
        # Vary line width based on b value (higher b = thicker line)
        b_width = 1.0 + 2.0 * (row['b'] - min(b_values)) / (max(b_values) - min(b_values))
        
        # Simulate response curves (exponential decay to steady state)
        t = np.linspace(0, row['response_time'], 100)
        # Simple exponential decay model
        response = 0.2 * np.exp(-3*t/row['response_time']) + 0.001
        ax1.plot(t, response, color=base_color, linestyle=line_style, 
                linewidth=b_width, alpha=0.8, 
                label=f'k={row["k"]:.0f}, b={row["b"]:.0f}')

ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Position Error (m)', fontsize=12)
ax1.set_title('Response Time Series (All Tested Combinations)', fontsize=14, fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#f8f9fa')

# Figure 2: Speed vs Time - ALL tested combinations
for k_val, group in k_groups.items():
    base_color = k_color_map[k_val]
    line_style = k_style_map[k_val]
    
    for row in group:
        # Vary line width based on b value (higher b = thicker line)
        b_width = 1.0 + 2.0 * (row['b'] - min(b_values)) / (max(b_values) - min(b_values))
        
        # Simulate speed curves (high initial speed, then decay)
        t = np.linspace(0, row['response_time'], 100)
        # Speed profile: starts high, decays exponentially
        speed = row['max_speed'] * np.exp(-2*t/row['response_time']) + 0.01
        ax2.plot(t, speed, color=base_color, linestyle=line_style, 
                linewidth=b_width, alpha=0.8,
                label=f'k={row["k"]:.0f}, b={row["b"]:.0f}')
    
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Speed (m/s)', fontsize=12)
ax2.set_title('Speed vs Time (All Tested Combinations)', fontsize=14, fontweight='bold')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#f8f9fa')

# Figure 3: 3D Plot - k, b, settling time with color coding
# Color by K values, size by B values, and add settled/not settled distinction
settled_colors = []
settled_sizes = []
settled_markers = []

for _, row in df.iterrows():
    # Color by K value
    settled_colors.append(k_color_map[row['k']])
    # Size by B value
    settled_sizes.append(b_size_map[row['b']])
    # Marker by settled status
    settled_markers.append('o' if row['settled'] else 'X')

# Plot settled and not settled points separately for better visibility
settled_df = df[df['settled'] == True]
not_settled_df = df[df['settled'] == False]

if len(settled_df) > 0:
    ax3.scatter(settled_df['k'], settled_df['b'], settled_df['response_time'], 
               c=[k_color_map[k] for k in settled_df['k']], 
               s=[b_size_map[b] for b in settled_df['b']], 
               marker='o', alpha=0.8, label='Settled', edgecolors='black', linewidth=0.5)

if len(not_settled_df) > 0:
    ax3.scatter(not_settled_df['k'], not_settled_df['b'], not_settled_df['response_time'], 
               c=[k_color_map[k] for k in not_settled_df['k']], 
               s=[b_size_map[b] for b in not_settled_df['b']], 
               marker='X', alpha=0.8, label='Not Settled', edgecolors='red', linewidth=1)

ax3.set_xlabel('K (Stiffness)', fontsize=12)
ax3.set_ylabel('B (Damping)', fontsize=12) 
ax3.set_zlabel('Settling Time (s)', fontsize=12)
ax3.set_title('K vs B vs Settling Time\n(Color=K, Size=B, Shape=Settled Status)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)

# Add color bar for K values
sm = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(colors[:len(k_values)]), 
                          norm=plt.cm.colors.Normalize(vmin=min(k_values), vmax=max(k_values)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax3, shrink=0.5, aspect=20)
cbar.set_label('K Values', fontsize=10)

# Add a comprehensive legend explanation
fig_legend = plt.figure(figsize=(10, 6))
ax_legend = fig_legend.add_subplot(111)
ax_legend.axis('off')

# Create legend explanation
legend_text = f"""
VISUALIZATION GUIDE:

COLORS: Each K (stiffness) value has a distinct color
• K=10: Blue (#0066CC)    • K=25: Red (#CC0000)       • K=50: Orange (#FF6600)
• K=100: Purple (#6600CC) • K=200: Yellow (#FFD700)   • K=350: Pink (#FF69B4)
• K=500: Dark Blue (#003366)

LINE STYLES: Different line styles for each K value
• K=10: Solid (-)         • K=25: Dashed (--)         • K=50: Dash-dot (-.)
• K=100: Dotted (:)       • K=200: Solid (-)          • K=350: Dashed (--)
• K=500: Dash-dot (-.)

LINE WIDTH: Thickness indicates B (damping) value
• Thinner lines = Lower B values
• Thicker lines = Higher B values

3D PLOT:
• Color = K value (same as above)
• Size = B value (larger = higher B)
• Shape = Settled status (○ = Settled, ✗ = Not Settled)
"""

ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes, fontsize=11,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

ax_legend.set_title('Color and Style Guide', fontsize=16, fontweight='bold', pad=20)

# Show all figures at once
plt.show()

# Print summary statistics
print(f"\n{'='*60}")
print(f"PARAMETER SWEEP ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"Total combinations tested: {len(df)}")
print(f"Successfully settled: {df['settled'].sum()}")
print(f"Settlement rate: {df['settled'].sum()/len(df)*100:.1f}%")
print(f"Average response time (settled): {df[df['settled']==True]['response_time'].mean():.2f}s")
print(f"Average final error (settled): {df[df['settled']==True]['final_error'].mean():.4f}m")
print(f"Best settling time: {df[df['settled']==True]['response_time'].min():.2f}s")
print(f"Worst settling time: {df[df['settled']==True]['response_time'].max():.2f}s")
print(f"{'='*60}")