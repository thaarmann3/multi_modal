import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
import sys

def plot_combined_trajectory(csv_filename):
    """Plot trajectory data from combined impedance control CSV file."""
    # Load the data
    df = pd.read_csv(csv_filename)
    time_seconds = (df['timestamp'] - df['timestamp'].iloc[0])  # Convert to seconds
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: XY trajectory (top view)
    ax1 = plt.subplot(2, 3, 1)
    if 'q_0_x' in df.columns and 'q_0_y' in df.columns:
        ax1.plot(df['q_0_x'], df['q_0_y'], 'b-', linewidth=2, label='Robot Path')
        ax1.scatter(df['q_0_x'].iloc[0], df['q_0_y'].iloc[0], color='green', s=100, label='Start', zorder=5)
        ax1.scatter(df['q_0_x'].iloc[-1], df['q_0_y'].iloc[-1], color='red', s=100, label='End', zorder=5)
        ax1.scatter(0, 0, color='orange', s=150, marker='*', label='Goal', zorder=5)
        ax1.set_xlabel('Relative X (m)')
        ax1.set_ylabel('Relative Y (m)')
    else:
        ax1.plot(df['x'], df['y'], 'b-', linewidth=2, label='Robot Path')
        ax1.scatter(df['x'].iloc[0], df['y'].iloc[0], color='green', s=100, label='Start', zorder=5)
        ax1.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='red', s=100, label='End', zorder=5)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
    ax1.set_title('XY Trajectory (Top View)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Z trajectory over time
    ax2 = plt.subplot(2, 3, 2)
    if 'q_0_z' in df.columns:
        ax2.plot(time_seconds, df['q_0_z'], 'b-', linewidth=2, label='Z Position')
        ax2.axhline(y=0, color='orange', linestyle='--', linewidth=2, label='Goal (Z=0)')
        ax2.set_ylabel('Relative Z (m)')
    else:
        ax2.plot(time_seconds, df['z'], 'b-', linewidth=2, label='Z Position')
        ax2.set_ylabel('Z (m)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Z Trajectory Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 3D trajectory
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    if 'q_0_x' in df.columns and 'q_0_y' in df.columns and 'q_0_z' in df.columns:
        ax3.plot(df['q_0_x'], df['q_0_y'], df['q_0_z'], 'b-', linewidth=2, label='Robot Path')
        ax3.scatter(df['q_0_x'].iloc[0], df['q_0_y'].iloc[0], df['q_0_z'].iloc[0], 
                   color='green', s=100, label='Start')
        ax3.scatter(df['q_0_x'].iloc[-1], df['q_0_y'].iloc[-1], df['q_0_z'].iloc[-1], 
                   color='red', s=100, label='End')
        ax3.scatter(0, 0, 0, color='orange', s=150, marker='*', label='Goal')
        ax3.set_xlabel('Relative X (m)')
        ax3.set_ylabel('Relative Y (m)')
        ax3.set_zlabel('Relative Z (m)')
    else:
        ax3.plot(df['x'], df['y'], df['z'], 'b-', linewidth=2, label='Robot Path')
        ax3.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], 
                   color='green', s=100, label='Start')
        ax3.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], 
                   color='red', s=100, label='End')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
    ax3.set_title('3D Trajectory')
    ax3.legend()
    
    # Plot 4: Forces over time
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(time_seconds, df['Fx'], 'r-', label='Fx', linewidth=1.5)
    ax4.plot(time_seconds, df['Fy'], 'g-', label='Fy', linewidth=1.5)
    ax4.plot(time_seconds, df['Fz'], 'b-', label='Fz', linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Force (N)')
    ax4.set_title('Input Forces Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Potential field forces over time
    ax5 = plt.subplot(2, 3, 5)
    if 'F_pot_x' in df.columns and 'F_pot_y' in df.columns:
        ax5.plot(time_seconds, df['F_pot_x'], 'r--', label='F_pot_x', linewidth=1.5, alpha=0.7)
        ax5.plot(time_seconds, df['F_pot_y'], 'g--', label='F_pot_y', linewidth=1.5, alpha=0.7)
    if 'F_pot_z' in df.columns:
        ax5.plot(time_seconds, df['F_pot_z'], 'b--', label='F_pot_z', linewidth=1.5, alpha=0.7)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Potential Force (N)')
    ax5.set_title('Potential Field Forces Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Velocities over time
    ax6 = plt.subplot(2, 3, 6)
    if 'velocity_x' in df.columns:
        ax6.plot(time_seconds, df['velocity_x'], 'r-', label='Vx', linewidth=1.5)
    if 'velocity_y' in df.columns:
        ax6.plot(time_seconds, df['velocity_y'], 'g-', label='Vy', linewidth=1.5)
    if 'velocity_z' in df.columns:
        ax6.plot(time_seconds, df['velocity_z'], 'b-', label='Vz', linewidth=1.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Velocity (m/s)')
    ax6.set_title('Velocities Over Time')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"TRAJECTORY SUMMARY")
    print(f"{'='*60}")
    print(f"Total points: {len(df)}")
    print(f"Duration: {time_seconds.iloc[-1]:.2f} seconds")
    print(f"Average frequency: {len(df) / time_seconds.iloc[-1]:.1f} Hz")
    
    if 'q_0_x' in df.columns and 'q_0_y' in df.columns:
        print(f"\nXY Position (relative to goal):")
        print(f"  Start: ({df['q_0_x'].iloc[0]:.4f}, {df['q_0_y'].iloc[0]:.4f}) m")
        print(f"  End: ({df['q_0_x'].iloc[-1]:.4f}, {df['q_0_y'].iloc[-1]:.4f}) m")
        print(f"  Distance from goal: {np.sqrt(df['q_0_x'].iloc[-1]**2 + df['q_0_y'].iloc[-1]**2):.4f} m")
    
    if 'q_0_z' in df.columns:
        print(f"\nZ Position (relative to goal):")
        print(f"  Start: {df['q_0_z'].iloc[0]:.4f} m")
        print(f"  End: {df['q_0_z'].iloc[-1]:.4f} m")
        print(f"  Distance from goal: {abs(df['q_0_z'].iloc[-1]):.4f} m")
    
    print(f"\nForces:")
    print(f"  Max |F|: {np.sqrt(df['Fx']**2 + df['Fy']**2 + df['Fz']**2).max():.2f} N")
    print(f"  Mean |F|: {np.sqrt(df['Fx']**2 + df['Fy']**2 + df['Fz']**2).mean():.2f} N")
    
    if 'velocity_x' in df.columns:
        print(f"\nVelocities:")
        if 'velocity_x' in df.columns:
            print(f"  Max |V_xy|: {np.sqrt(df['velocity_x']**2 + df['velocity_y']**2).max():.4f} m/s")
        if 'velocity_z' in df.columns:
            print(f"  Max |V_z|: {abs(df['velocity_z']).max():.4f} m/s")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot combined robot trajectory from CSV file')
    parser.add_argument('target_file', nargs='?', default=None, 
                        help='Path to specific trajectory CSV file (default: most recent combined file)')
    args = parser.parse_args()
    
    # Default trajectory directory (relative to project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trajectory_dir = os.path.join(base_dir, "trajectories")
    
    # If target file is specified, use it; otherwise find the most recent combined file
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
        # Find the most recent combined CSV file
        csv_files = glob.glob(os.path.join(trajectory_dir, "robot_trajectory_combined_*.csv"))
        
        if not csv_files:
            print(f"No combined robot trajectory CSV files found in '{trajectory_dir}' directory.")
            print("Make sure you have run the combined impedance control script first.")
            sys.exit(1)
        
        # Get the most recent file
        csv_filename = max(csv_files, key=os.path.getctime)
        print(f"Using most recent combined file: {csv_filename}")
    
    try:
        plot_combined_trajectory(csv_filename)
    except Exception as e:
        print(f"Error plotting trajectory: {e}")
        import traceback
        traceback.print_exc()

