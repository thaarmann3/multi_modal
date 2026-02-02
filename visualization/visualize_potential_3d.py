"""
Visualization function for 3D potential field from discretized tunneling control results.

This function reads trajectory CSV files from impedance_control_discretized_tunneling.py
and displays a 3D graph of the potential function, including any bores created during
the trajectory.

Usage:
    # From command line:
    python visualization/visualize_potential_3d.py [csv_filename]
    
    # Or in Python:
    from visualization.visualize_potential_3d import visualize_potential_3d
    visualize_potential_3d('trajectories/robot_trajectory_discretized_*.csv')
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fields.potential_field_discrete import PotentialFieldDiscreteRemodelable


def reconstruct_potential_field_from_trajectory(csv_filename, config_path=None, 
                                                reconstruct_bores=True, 
                                                dt=0.0025):
    """
    Reconstruct the potential field state from a trajectory CSV file.
    
    Args:
        csv_filename: Path to the trajectory CSV file
        config_path: Path to the config JSON file (if None, tries to find it)
        reconstruct_bores: If True, replay the trajectory to reconstruct bores
        dt: Control loop time step (default 0.0025)
    
    Returns:
        pf: PotentialFieldDiscreteRemodelable object with reconstructed state
        df: DataFrame with trajectory data
    """
    # Load trajectory data
    df = pd.read_csv(csv_filename)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.dropna()
    
    # Find config file if not provided
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "configs", 
                                   "impedance_control_discretized_tunneling.json")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize potential field from config
    pf_config = config["potential_field"]
    pf = PotentialFieldDiscreteRemodelable(
        x_bounds=tuple(pf_config["x_bounds"]),
        y_bounds=tuple(pf_config["y_bounds"]),
        resolution=pf_config["resolution"],
        alpha=pf_config["alpha"],
    )
    
    # Add obstacles from config
    for obs in pf_config.get("obstacles", []):
        pf.add_obstacle(obs["x"], obs["y"], obs["height"], obs["width"])
    
    # Reconstruct bores by replaying the trajectory
    if reconstruct_bores and 'gamma' in df.columns and 'intent_I' in df.columns:
        print("Reconstructing bores from trajectory data...")
        
        # Get tunneling parameters from config
        tunneling_config = config.get("tunneling", {})
        min_grad_norm = tunneling_config.get("min_grad_norm", 10.0)
        
        # Replay trajectory to reconstruct bores
        for idx, row in df.iterrows():
            if pd.isna(row.get('gamma', 0)) or pd.isna(row.get('intent_I', 0)):
                continue
            
            gamma = row['gamma']
            intent_I = row['intent_I']
            
            # Only update bores when gamma is significant (tunneling is happening)
            if gamma > 0.01:
                # Get robot position (relative coordinates)
                if 'q_0_x' in df.columns and 'q_0_y' in df.columns:
                    robot_x = row['q_0_x']
                    robot_y = row['q_0_y']
                else:
                    # Fallback: would need initial position, but we don't have it
                    # Skip this case for now
                    continue
                
                # Get external force
                if 'Fx' in df.columns and 'Fy' in df.columns:
                    force_vector = np.array([row['Fx'], row['Fy']])
                else:
                    continue
                
                # Check if we're within bounds
                x_bounds, y_bounds = pf.x_bounds, pf.y_bounds
                if (x_bounds[0] <= robot_x <= x_bounds[1] and 
                    y_bounds[0] <= robot_y <= y_bounds[1]):
                    # Update bore using the tunneling intent
                    pf.update_bore_from_force(
                        robot_x=robot_x,
                        robot_y=robot_y,
                        force_vector=force_vector,
                        tunneling_intent=gamma
                    )
        
        print(f"Reconstruction complete. Created {pf.get_field_bore_count()} field bores "
              f"and {sum(len(b) for b in pf.obstacle_bores.values())} obstacle bores.")
    
    return pf, df


def visualize_potential_3d(csv_filename, config_path=None, 
                           reconstruct_bores=True,
                           show_trajectory=True,
                           show_obstacles=True,
                           show_bores=True,
                           elev=60, azim=45,
                           save_path=None):
    """
    Visualize the potential field as a 3D surface plot.
    
    Args:
        csv_filename: Path to the trajectory CSV file
        config_path: Path to the config JSON file (if None, tries to find it)
        reconstruct_bores: If True, replay the trajectory to reconstruct bores
        show_trajectory: If True, overlay the robot trajectory on the surface
        show_obstacles: If True, mark obstacle positions
        show_bores: If True, visualize bore directions
        elev: Elevation angle for 3D plot (default 60)
        azim: Azimuth angle for 3D plot (default 45)
        save_path: If provided, save the figure to this path
    """
    # Reconstruct potential field
    pf, df = reconstruct_potential_field_from_trajectory(
        csv_filename, config_path, reconstruct_bores
    )
    
    # Calculate potential over the grid
    Z = pf.calculate_potential()
    X, Y = pf.X, pf.Y
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the potential surface
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.85, 
                          linewidth=0, antialiased=True, shade=True)
    
    # Add wireframe for better depth perception
    ax.plot_wireframe(X, Y, Z, rstride=15, cstride=15, 
                     alpha=0.2, color='black', linewidth=0.3)
    
    # Overlay trajectory if requested
    if show_trajectory:
        if 'q_0_x' in df.columns and 'q_0_y' in df.columns:
            traj_x = df['q_0_x'].to_numpy()
            traj_y = df['q_0_y'].to_numpy()
            
            # Sample potential along trajectory
            x_grid = X[0, :]
            y_grid = Y[:, 0]
            ix = np.clip(np.searchsorted(x_grid, traj_x), 0, len(x_grid)-1)
            iy = np.clip(np.searchsorted(y_grid, traj_y), 0, len(y_grid)-1)
            traj_z = Z[iy, ix] + 0.1  # Small lift to make it visible above surface
            
            # Plot trajectory
            ax.plot(traj_x, traj_y, traj_z, 'k-', linewidth=2.5, 
                   label='Robot Trajectory', zorder=10)
            
            # Mark start and end points
            ax.scatter(traj_x[0], traj_y[0], traj_z[0], 
                      color='green', s=150, marker='o', 
                      label='Start', zorder=11, edgecolors='black', linewidths=1.5)
            ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], 
                      color='red', s=150, marker='s', 
                      label='End', zorder=11, edgecolors='black', linewidths=1.5)
    
    # Mark obstacles if requested
    if show_obstacles:
        for obs in pf.obstacles.values():
            obs_x, obs_y = obs['x'], obs['y']
            # Get potential at obstacle center
            obs_z = pf.get_potential(obs_x, obs_y)
            ax.scatter(obs_x, obs_y, obs_z, 
                      color='red', s=200, marker='^', 
                      label='Obstacle' if obs == list(pf.obstacles.values())[0] else '',
                      zorder=12, edgecolors='black', linewidths=2)
    
    # Visualize bores if requested
    if show_bores:
        # Draw obstacle bore directions
        for obs_key, bores in pf.obstacle_bores.items():
            obs = pf.obstacles[obs_key]
            obs_x, obs_y = obs['x'], obs['y']
            obs_z = pf.get_potential(obs_x, obs_y)
            
            for bore in bores:
                dir_vec = np.array(bore['direction'])
                arrow_length = bore['width'] * 0.4
                # Project arrow onto the potential surface
                arrow_end_x = obs_x + dir_vec[0] * arrow_length
                arrow_end_y = obs_y + dir_vec[1] * arrow_length
                arrow_end_z = pf.get_potential(arrow_end_x, arrow_end_y)
                
                ax.quiver(obs_x, obs_y, obs_z,
                         dir_vec[0] * arrow_length,
                         dir_vec[1] * arrow_length,
                         arrow_end_z - obs_z,
                         color='orange', arrow_length_ratio=0.3,
                         linewidth=2, alpha=0.8, zorder=13)
        
        # Draw field bore locations
        for bore in pf.field_bores:
            bore_x, bore_y = bore['x'], bore['y']
            bore_z = pf.get_potential(bore['x'], bore['y'])
            
            # Mark bore location
            ax.scatter(bore_x, bore_y, bore_z,
                      color='cyan', s=100, marker='o',
                      zorder=12, edgecolors='black', linewidths=1.5, alpha=0.9)
            
            # Draw bore direction
            dir_vec = np.array(bore['direction'])
            arrow_length = bore['width'] * 0.4
            arrow_end_x = bore_x + dir_vec[0] * arrow_length
            arrow_end_y = bore_y + dir_vec[1] * arrow_length
            arrow_end_z = pf.get_potential(arrow_end_x, arrow_end_y)
            
            ax.quiver(bore_x, bore_y, bore_z,
                     dir_vec[0] * arrow_length,
                     dir_vec[1] * arrow_length,
                     arrow_end_z - bore_z,
                     color='cyan', arrow_length_ratio=0.3,
                     linewidth=2, alpha=0.8, zorder=13)
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)
    
    # Add colorbar
    cbar = plt.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Potential Value', rotation=270, labelpad=20, fontsize=12)
    
    # Labels and title
    ax.set_xlabel('Relative X (m)', fontsize=12)
    ax.set_ylabel('Relative Y (m)', fontsize=12)
    ax.set_zlabel('Potential', fontsize=12)
    ax.set_title('3D Potential Field with Bore Tunneling', fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio for better visualization
    ax.set_box_aspect([1, 1, 0.6])  # Adjust Z scaling for better view
    
    # Add legend
    if show_trajectory or show_obstacles:
        ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Print summary
    print(f"\nPotential Field Summary:")
    print(f"  Bounds: X={pf.x_bounds}, Y={pf.y_bounds}")
    print(f"  Resolution: {pf.resolution}x{pf.resolution}")
    print(f"  Alpha: {pf.alpha}")
    print(f"  Obstacles: {len(pf.obstacles)}")
    print(f"  Obstacle bores: {sum(len(b) for b in pf.obstacle_bores.values())}")
    print(f"  Field bores: {pf.get_field_bore_count()}")
    print(f"  Potential range: [{np.min(Z):.2f}, {np.max(Z):.2f}]")
    print(f"  Trajectory points: {len(df)}")
    if 'q_0_x' in df.columns and 'q_0_y' in df.columns:
        print(f"  Trajectory start: ({df['q_0_x'].iloc[0]:.3f}, {df['q_0_y'].iloc[0]:.3f})")
        print(f"  Trajectory end: ({df['q_0_x'].iloc[-1]:.3f}, {df['q_0_y'].iloc[-1]:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize 3D potential field from discretized tunneling control results'
    )
    parser.add_argument('target_file', nargs='?', default=None,
                       help='Path to specific trajectory CSV file (default: most recent discretized)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON file (default: auto-detect)')
    parser.add_argument('--no-bores', action='store_true',
                       help='Do not reconstruct bores from trajectory')
    parser.add_argument('--no-trajectory', action='store_true',
                       help='Do not show robot trajectory')
    parser.add_argument('--no-obstacles', action='store_true',
                       help='Do not mark obstacle positions')
    parser.add_argument('--no-bore-vis', action='store_true',
                       help='Do not visualize bore directions')
    parser.add_argument('--elev', type=float, default=60,
                       help='Elevation angle for 3D plot (default: 60)')
    parser.add_argument('--azim', type=float, default=45,
                       help='Azimuth angle for 3D plot (default: 45)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save figure to this path')
    
    args = parser.parse_args()
    
    # Default trajectory directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trajectory_dir = os.path.join(base_dir, "trajectories")
    
    # Find CSV file
    if args.target_file is not None:
        csv_filename = args.target_file
        if not os.path.dirname(csv_filename):
            csv_filename = os.path.join(trajectory_dir, csv_filename)
        
        if not os.path.exists(csv_filename):
            print(f"Error: File '{csv_filename}' not found.")
            sys.exit(1)
        
        print(f"Using specified file: {csv_filename}")
    else:
        # Find the most recent discretized CSV file
        csv_files = glob.glob(os.path.join(trajectory_dir, 
                                          "robot_trajectory_discretized_*.csv"))
        
        if not csv_files:
            print(f"No discretized trajectory CSV files found in '{trajectory_dir}' directory.")
            print("Make sure you have run the impedance_control_discretized_tunneling.py script first.")
            sys.exit(1)
        
        csv_filename = max(csv_files, key=os.path.getctime)
        print(f"Using most recent file: {csv_filename}")
    
    try:
        visualize_potential_3d(
            csv_filename,
            config_path=args.config,
            reconstruct_bores=not args.no_bores,
            show_trajectory=not args.no_trajectory,
            show_obstacles=not args.no_obstacles,
            show_bores=not args.no_bore_vis,
            elev=args.elev,
            azim=args.azim,
            save_path=args.save
        )
    except Exception as e:
        print(f"Error visualizing potential field: {e}")
        import traceback
        traceback.print_exc()
