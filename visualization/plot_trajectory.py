import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fields.potential_field_class import PotentialField
from control.impedance_control_spikes import create_potential_field

def plot_trajectory(csv_filename):
    # Load the data
    df = pd.read_csv(csv_filename)
    # Sort by timestamp to ensure continuous plotting
    df = df.sort_values('timestamp').reset_index(drop=True)
    # Remove any NaN values that could cause breaks in the plot
    df = df.dropna()
    time_seconds = (df['timestamp'] - df['timestamp'].iloc[0])  # Convert to seconds
    
    # Use the same potential field configuration as impedance_control_spikes.py
    pf = create_potential_field()
    Z_pf = pf.calculate_potential()
    
    # Plot 1: 3D trajectory with potential field overlay
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Overlay potential field surface
    ax1.plot_surface(pf.X, pf.Y, Z_pf, cmap='viridis', alpha=0.5, linewidth=0, antialiased=True)
    
    # Compute trajectory relative to goal (q_0) if available; else use absolute
    if {'q_0_x','q_0_y'}.issubset(df.columns):
        traj_x = df['q_0_x'].to_numpy()
        traj_y = df['q_0_y'].to_numpy()
        # Sample PF Z along the trajectory by nearest grid index
        x_grid = pf.X[0, :]
        y_grid = pf.Y[:, 0]
        ix = np.clip(np.searchsorted(x_grid, traj_x), 0, len(x_grid)-1)
        iy = np.clip(np.searchsorted(y_grid, traj_y), 0, len(y_grid)-1)
        traj_z = Z_pf[iy, ix] + 0.05  # small lift so the line is visible above surface
        ax1.plot(traj_x, traj_y, traj_z, 'k-', linewidth=2, label='Robot Path (on PF)', linestyle='-', marker='None')
        # Markers
        ax1.scatter(traj_x[0], traj_y[0], traj_z[0], color='green', s=60, label='Start')
        ax1.scatter(traj_x[-1], traj_y[-1], traj_z[-1], color='red', s=60, label='End')
        ax1.set_xlabel('Relative X (m)')
        ax1.set_ylabel('Relative Y (m)')
        ax1.set_zlabel('Potential Z')
        ax1.set_title('3D Trajectory over Potential Field (relative)')
    else:
        # Fallback to absolute trajectory without projection
        ax1.plot(df['x'], df['y'], df['z'], 'k-', linewidth=2, label='Robot Path', linestyle='-', marker='None')
        ax1.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], color='green', s=60, label='Start')
        ax1.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], color='red', s=60, label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Robot Trajectory with Potential Field')
    ax1.legend()
    plt.tight_layout()
    
    # Plot 2: XY trajectory (top view) with potential field overlay
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    
    # Overlay potential field contours
    contour = ax2.contour(pf.X, pf.Y, Z_pf, levels=20, alpha=0.5, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    
    if {'q_0_x','q_0_y'}.issubset(df.columns):
        ax2.plot(df['q_0_x'], df['q_0_y'], 'b-', linewidth=2, label='Robot Path (relative)', zorder=3, linestyle='-', marker='None')
        ax2.scatter(df['q_0_x'].iloc[0], df['q_0_y'].iloc[0], color='green', s=60, label='Start', zorder=4)
        ax2.scatter(df['q_0_x'].iloc[-1], df['q_0_y'].iloc[-1], color='red', s=60, label='End', zorder=4)
        ax2.set_xlabel('Relative X (m)')
        ax2.set_ylabel('Relative Y (m)')
        # Set axis limits to match potential field bounds
        ax2.set_xlim(pf.x_bounds)
        ax2.set_ylim(pf.y_bounds)
    else:
        ax2.plot(df['x'], df['y'], 'b-', linewidth=2, label='Robot Path', zorder=3, linestyle='-', marker='None')
        ax2.scatter(df['x'].iloc[0], df['y'].iloc[0], color='green', s=60, label='Start', zorder=4)
        ax2.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='red', s=60, label='End', zorder=4)
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
    ax2.set_title('XY Trajectory with Potential Field Overlay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    plt.tight_layout()
    
    # Plot 3: Forces over time
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(time_seconds, df['Fx'], 'r-', label='Fx', linestyle='-', marker='None')
    ax3.plot(time_seconds, df['Fy'], 'g-', label='Fy', linestyle='-', marker='None')
    ax3.plot(time_seconds, df['Fz'], 'b-', label='Fz', linestyle='-', marker='None')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Force (N)')
    ax3.set_title('Forces Over Time')
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    
    # Plot 4: Velocity over time
    fig4 = plt.figure(figsize=(8, 6))
    ax4 = fig4.add_subplot(111)
    ax4.plot(time_seconds, df['velocity_x'], 'r-', label='Vx', linestyle='-', marker='None')
    ax4.plot(time_seconds, df['velocity_y'], 'g-', label='Vy', linestyle='-', marker='None')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Velocity Over Time')
    ax4.legend()
    ax4.grid(True)
    plt.tight_layout()
    
    # Show all plots at once
    plt.show()
    
    # Print summary statistics
    print(f"\nTrajectory Summary:")
    print(f"Total points: {len(df)}")
    print(f"Duration: {time_seconds.iloc[-1]:.2f} seconds")
    if {'q_0_x','q_0_y'}.issubset(df.columns):
        print(f"Start (rel): ({df['q_0_x'].iloc[0]:.3f}, {df['q_0_y'].iloc[0]:.3f})  End (rel): ({df['q_0_x'].iloc[-1]:.3f}, {df['q_0_y'].iloc[-1]:.3f})")
    else:
        print(f"Start position: ({df['x'].iloc[0]:.3f}, {df['y'].iloc[0]:.3f}, {df['z'].iloc[0]:.3f})")
        print(f"End position: ({df['x'].iloc[-1]:.3f}, {df['y'].iloc[-1]:.3f}, {df['z'].iloc[-1]:.3f})")
    print(f"Max force magnitude: {np.sqrt(df['Fx']**2 + df['Fy']**2 + df['Fz']**2).max():.2f} N")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot robot trajectory from CSV file')
    parser.add_argument('target_file', nargs='?', default=None, 
                        help='Path to specific trajectory CSV file (default: most recent)')
    args = parser.parse_args()
    
    # Default trajectory directory (relative to project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trajectory_dir = os.path.join(base_dir, "trajectories")
    
    # If target file is specified, use it; otherwise find the most recent
    if args.target_file is not None:
        csv_filename = args.target_file
        # If it's just a filename without path, check in trajectories folder
        if not os.path.dirname(csv_filename):
            csv_filename = os.path.join(trajectory_dir, csv_filename)
        
        if not os.path.exists(csv_filename):
            print(f"Error: File '{csv_filename}' not found.")
            sys.exit(1)
        
        print(f"Using specified file: {csv_filename}")
    else:
        # Find the most recent CSV file in trajectories folder
        csv_files = glob.glob(os.path.join(trajectory_dir, "robot_trajectory_*.csv"))
        
        if not csv_files:
            print(f"No robot trajectory CSV files found in '{trajectory_dir}' directory.")
            print("Make sure you have run the impedance control script first.")
            sys.exit(1)
        
        # Get the most recent file
        csv_filename = max(csv_files, key=os.path.getctime)
        print(f"Using most recent file: {csv_filename}")
    
    try:
        plot_trajectory(csv_filename)
    except Exception as e:
        print(f"Error plotting trajectory: {e}")
        import traceback
        traceback.print_exc()
