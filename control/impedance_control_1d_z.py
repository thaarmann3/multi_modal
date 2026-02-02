from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import numpy as np
import sympy as sp
import math
import time
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import signal
import sys
from scipy import signal as scipy_signal

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fields.potential_field_class import PotentialField1DFlat

robot_ip = "169.254.9.43"

rtde_c = RTDEControlInterface(robot_ip)
rtde_r = RTDEReceiveInterface(robot_ip)

# Control parameters
alpha = 0.15  # Low-pass filter coefficient for force
k = 30.0       # Scaling factor for gradient response (increased proportionally)
b = 90.0      # Damping constant (increased proportionally, maintaining k/b balance ~0.33)
              # Higher damping makes system stiffer (less responsive) while maintaining balance
dt = 1.0/400.0  # Control loop time step

# Butterworth filter parameters
sampling_freq = 1.0 / dt  # Hz - match control loop rate (400 Hz)
cutoff_freq = 25.0         # Hz
filter_order = 4
deadband_threshold = 1.0  # N - force components below this magnitude are set to 0

# Design 4th order Butterworth low-pass filter using SOS form for better numerical stability
nyquist = sampling_freq / 2.0
normal_cutoff = cutoff_freq / nyquist
sos_butter = scipy_signal.butter(filter_order, normal_cutoff, btype='low', analog=False, output='sos')

# Initialize filter state for each force component (x, y, z)
# sosfilt_zi returns initial conditions for steady-state step response
zi_x = scipy_signal.sosfilt_zi(sos_butter)
zi_y = scipy_signal.sosfilt_zi(sos_butter)
zi_z = scipy_signal.sosfilt_zi(sos_butter)

# Clamp mass to valid range [0, 10] kg (inclusive)
min_mass = 0.0  # kg
max_mass = 5.0  # kg

# Enable/disable live plotting
ENABLE_LIVE_PLOTTING = True

# Initialize 1D potential field for Z axis
# Potential field magnitude is much greater than spike magnitude
# With scaling_factor=10.0 and height=0.12, alpha = 10.0*0.12 = 1.2, so F_parab = -2.4*x
# The paraboloid (potential field) dominates, with the spike creating a small perturbation
pf = PotentialField1DFlat(x_bounds=(-0.5, 0.5), resolution=1000, scaling_factor=10.0, alpha=10.0)
pf.add_obstacle(0.0, 0.1, 0.05)  # Obstacle at origin (small spike relative to potential field)

# Setup CSV logging
# Create trajectories folder if it doesn't exist
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trajectory_dir = os.path.join(base_dir, "trajectories")
os.makedirs(trajectory_dir, exist_ok=True)

# Generate filename with date/time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(trajectory_dir, f"robot_trajectory_1d_z_{timestamp}.csv")
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'Fx', 'Fy', 'Fz', 'F_pot_z', 'q_0_z', 'velocity_z'])

# Reset Joints
j_goal = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, 0]
rtde_c.moveJ(j_goal, 0.5)
time.sleep(1.0)

# Goal position (origin in Z direction)
# Get current position and use Z as goal
qx_init, qy_init, qz_init, qrx_init, qry_init, qrz_init = rtde_r.getActualTCPPose()
q_goal_z = qz_init  # Goal is current Z position (origin in relative coordinates)

print(f"Goal Z position: {q_goal_z:.4f} m")

# Move to the goal position in Cartesian space (keep current Z)
goal_pose = [qx_init, qy_init, qz_init, 0, 0, 0]  # [x, y, z, rx, ry, rz]
rtde_c.moveL(goal_pose, 0.1, 0.1)
time.sleep(2.0)  # Wait for movement to complete
print("Moved to goal position")
rtde_c.zeroFtSensor()

# Set initial payload mass (default value before control loop)
initial_mass = 0.1  # kg
# rtde_c.setPayload(initial_mass, [0.0, 0.0, 0.0])

# Setup live plotting (if enabled)
if ENABLE_LIVE_PLOTTING:
    plt.ion()  # Turn on interactive mode
    fig, (ax_traj, ax_pot) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Trajectory (position vs time) with force overlay
    ax_traj.set_xlabel('Time (s)')
    ax_traj.set_ylabel('Relative Z position (m)', color='b')
    ax_traj.tick_params(axis='y', labelcolor='b')
    ax_traj.set_title('Robot Trajectory and Control Force')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axhline(y=0, color='g', linestyle=':', linewidth=2, label='Goal (Z=0)')
    
    # Add second y-axis for force on trajectory plot
    ax_traj_force = ax_traj.twinx()
    ax_traj_force.set_ylabel('Total Control Force (N)', color='m')
    ax_traj_force.tick_params(axis='y', labelcolor='m')
    ax_traj_force.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    
    # Right plot: Potential field (potential vs position) with force overlay
    z_range = pf.x_positions
    potential_values = pf.calculate_potential()
    ax_pot.plot(z_range, potential_values, 'b-', linewidth=2, label='Potential Field')
    ax_pot.set_xlabel('Relative Z position (m)')
    ax_pot.set_ylabel('Potential', color='b')
    ax_pot.tick_params(axis='y', labelcolor='b')
    ax_pot.set_title('Potential Field vs Position')
    ax_pot.grid(True, alpha=0.3)
    ax_pot.axvline(x=0, color='g', linestyle=':', linewidth=2, label='Goal (Z=0)')
    
    # Add second y-axis for force
    ax_pot_force = ax_pot.twinx()
    # Calculate potential force field (force from potential vs position)
    force_values = np.array([pf.get_gradient(z) for z in z_range])
    force_line, = ax_pot_force.plot(z_range, force_values, 'r--', linewidth=2, alpha=0.7, label='Potential Force Field')
    ax_pot_force.set_ylabel('Force (N)', color='r')
    ax_pot_force.tick_params(axis='y', labelcolor='r')
    ax_pot_force.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    
    # Initialize trajectory storage
    trajectory_z = []
    trajectory_time = []
    trajectory_total_force = []  # Store total control force (potential + input)
    start_time = time.time()
    
    # Trajectory plot elements
    trajectory_line, = ax_traj.plot([], [], 'b-', linewidth=2, label='Robot Path')
    current_point_traj, = ax_traj.plot([], [], 'ro', markersize=8, label='Current Position')
    start_point_traj, = ax_traj.plot([], [], 'go', markersize=10, label='Start')
    
    # Total control force line on trajectory plot (force vs time)
    total_force_line, = ax_traj_force.plot([], [], 'm-', linewidth=2, alpha=0.7, label='Total Control Force')
    
    # Combine legends from both axes on trajectory plot
    lines_traj1, labels_traj1 = ax_traj.get_legend_handles_labels()
    lines_traj2, labels_traj2 = ax_traj_force.get_legend_handles_labels()
    ax_traj.legend(lines_traj1 + lines_traj2, labels_traj1 + labels_traj2, loc='upper right')
    
    # Potential field plot elements
    current_point_pot, = ax_pot.plot([], [], 'ro', markersize=10, label='Current Position', zorder=5)
    start_point_pot, = ax_pot.plot([], [], 'go', markersize=10, label='Start', zorder=5)
    
    # Combine legends from both axes on potential field plot
    lines1, labels1 = ax_pot.get_legend_handles_labels()
    lines2, labels2 = ax_pot_force.get_legend_handles_labels()
    ax_pot.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    fig.canvas.draw()  # Initial draw
    plt.show(block=False)
    
    # Plot update counter (update every N iterations to avoid affecting control loop timing)
    plot_update_counter = 0
    plot_update_interval = 15  # Update plot every 15 iterations (~30 Hz at 500 Hz control loop)
else:
    # Initialize empty trajectory storage (for CSV logging only)
    trajectory_z = []
    trajectory_time = []
    trajectory_total_force = []
    start_time = time.time()
    fig = None
    ax_traj = None
    ax_traj_force = None
    ax_pot = None
    ax_pot_force = None

# Flag for clean shutdown
shutdown_flag = False

# Signal handler for KeyboardInterrupt (works even when plot has focus)
def signal_handler(sig, frame):
    global shutdown_flag
    shutdown_flag = True
    print("\nInterrupt received, shutting down gracefully...")

signal.signal(signal.SIGINT, signal_handler)

F_inp = np.array([0, 0, 0])

while True:
    try:
        # Check for shutdown flag (set by signal handler)
        if shutdown_flag:
            raise KeyboardInterrupt
       
        qx, qy, qz, qrx, qry, qrz = rtde_r.getActualTCPPose()
        q_0_z = qz - q_goal_z  # Relative Z position (goal is at origin)

        Fx, Fy, Fz, Tx, Ty, Tz = rtde_r.getActualTCPForce()
        F_inp_sensor = np.array([Fx, Fy, Fz])
        
        # Apply 4th order Butterworth filter to each force component using SOS form
        F_inp_x, zi_x = scipy_signal.sosfilt(sos_butter, [F_inp_sensor[0]], zi=zi_x)
        F_inp_y, zi_y = scipy_signal.sosfilt(sos_butter, [F_inp_sensor[1]], zi=zi_y)
        F_inp_z, zi_z = scipy_signal.sosfilt(sos_butter, [F_inp_sensor[2]], zi=zi_z)
        F_inp = np.array([F_inp_x[0], F_inp_y[0], F_inp_z[0]])
        
        # Apply deadband: set components below threshold to zero
        F_inp[np.abs(F_inp) < deadband_threshold] = 0.0

        # ========== Z AXIS CONTROL ==========
        # Check if q_0_z is outside potential field bounds
        z_bounds = pf.x_bounds
        F_bounds_z = 0.0
        
        # Add strong repulsive force if outside bounds
        if q_0_z < z_bounds[0]:
            F_bounds_z = 1000 * (z_bounds[0] - q_0_z)  # Push up (positive direction)
        elif q_0_z > z_bounds[1]:
            F_bounds_z = 1000 * (z_bounds[1] - q_0_z)  # Push down (negative direction)
        
        # Get potential field force for Z
        if z_bounds[0] <= q_0_z <= z_bounds[1]:
            # get_gradient returns -dV/dx (force), so we use it directly
            # Convert to float in case it's a numpy scalar
            F_pot_z = float(pf.get_gradient(q_0_z))
        else:
            F_pot_z = 0.0  # No potential field force outside bounds
        
        # Combine bounds force with potential field force
        F_pot_z = F_pot_z + F_bounds_z
        
        qdot = np.zeros(6)
        
        # Impedance control in Z direction only
        # Response magnitude scales with gradient magnitude: high gradient = strong response
        # Robot moves based on input force and scaled potential field gradient
        # Will naturally come to rest at local minima where net force (gradient) is zero
        # Goal position is only for visualization (q_0_z), not used in control
        # F_pot_z is the gradient (force from potential field), scaled by k
        qdot[2] = (F_inp[2] + k * F_pot_z)/b

        # Calculate total control force magnitude and update payload mass
        F_total_z = F_inp[2] + F_pot_z
        F_total_magnitude = abs(F_total_z)
        
        # Convert force to equivalent mass using gravity (F = mg -> m = F/g)
        mass = np.clip(F_total_magnitude / 9.81, min_mass, max_mass)
        
        # Update payload mass dynamically to prevent protective stops
        # rtde_c.setPayload(mass, [0.0, 0.0, 0.0])

        # Limit velocity to maximum of 0.75 for Z component
        qdot = np.clip(qdot, -0.75, 0.75)
        rtde_c.speedL(qdot, 0.3, dt)
        
        # Log data to CSV
        current_time = time.time()
        csv_writer.writerow([
            current_time,
            qx, qy, qz, qrx, qry, qrz,  # Pose
            Fx, Fy, Fz,  # Force
            F_pot_z,  # Potential force
            q_0_z,  # Position relative to goal
            qdot[2]  # Velocity in Z
        ])
        
        # Store trajectory points
        trajectory_z.append(q_0_z)
        trajectory_time.append(current_time - start_time)
        trajectory_total_force.append(F_total_z)
        
        # Update plot periodically (non-blocking, doesn't affect dt) - only if enabled
        if ENABLE_LIVE_PLOTTING:
            plot_update_counter += 1
            if plot_update_counter >= plot_update_interval:
                if len(trajectory_z) > 0:
                    # Update trajectory plot (left subplot: position vs time)
                    trajectory_line.set_data(trajectory_time, trajectory_z)
                    current_point_traj.set_data([trajectory_time[-1]], [trajectory_z[-1]])
                    
                    # Update start point (first position)
                    if len(trajectory_z) == 1:
                        start_point_traj.set_data([trajectory_time[0]], [trajectory_z[0]])
                    
                    # Auto-scale trajectory axes
                    if len(trajectory_z) > 1:
                        ax_traj.set_xlim([0, max(trajectory_time)])
                        z_min, z_max = min(trajectory_z), max(trajectory_z)
                        z_range = max(abs(z_min), abs(z_max)) * 1.2
                        ax_traj.set_ylim([-z_range, z_range])
                    
                    # Update total control force line on trajectory plot (force vs time)
                    if len(trajectory_total_force) > 0:
                        total_force_line.set_data(trajectory_time, trajectory_total_force)
                        # Auto-scale force axis
                        if len(trajectory_total_force) > 1:
                            force_min, force_max = min(trajectory_total_force), max(trajectory_total_force)
                            force_range = max(abs(force_min), abs(force_max)) * 1.2
                            if force_range > 0:
                                ax_traj_force.set_ylim([-force_range, force_range])
                    
                    # Update potential field plot (right subplot: potential vs position)
                    current_point_pot.set_data([trajectory_z[-1]], [pf.get_potential(trajectory_z[-1])])
                    
                    # Update start point on potential plot
                    if len(trajectory_z) == 1:
                        start_point_pot.set_data([trajectory_z[0]], [pf.get_potential(trajectory_z[0])])
                    
                    # Keep potential field plot bounds fixed to show full potential field
                    ax_pot.set_xlim(pf.x_bounds)
                    pot_min, pot_max = min(potential_values), max(potential_values)
                    pot_range = pot_max - pot_min
                    ax_pot.set_ylim([pot_min - 0.1 * pot_range, pot_max + 0.1 * pot_range])
                    
                    # Auto-scale force axis based on potential force field values
                    force_min, force_max = min(force_values), max(force_values)
                    force_range = force_max - force_min
                    if force_range > 0:
                        ax_pot_force.set_ylim([force_min - 0.1 * force_range, force_max + 0.1 * force_range])
                
                # Non-blocking plot update - use draw_idle only, avoid flush_events which can block
                fig.canvas.draw_idle()
                # Try to flush events, but don't let it block interrupts
                try:
                    # Only process events if shutdown hasn't been requested
                    if not shutdown_flag:
                        fig.canvas.flush_events()
                except:
                    pass  # Ignore errors during event flushing
                plot_update_counter = 0
        
        # Optional: print debug info
        print(f"Z relative: {q_0_z:.4f}, Force: {F_inp[2]:.2f}, Pot Force: {F_pot_z:.2f}, Velocity: {qdot[2]:.4f}")

    except KeyboardInterrupt:
        print("\nShutting down...")
        csv_file.close()
        
        # Stop robot motion immediately
        try:
            rtde_c.speedStop()
            rtde_c.disconnect()
        except:
            pass
        
        # Final plot update (if plotting was enabled)
        if ENABLE_LIVE_PLOTTING and len(trajectory_z) > 0:
            # Update trajectory plot
            trajectory_line.set_data(trajectory_time, trajectory_z)
            current_point_traj.set_data([trajectory_time[-1]], [trajectory_z[-1]])
            
            # Update total control force line on trajectory plot
            if len(trajectory_total_force) > 0:
                total_force_line.set_data(trajectory_time, trajectory_total_force)
            
            # Update potential field plot
            current_point_pot.set_data([trajectory_z[-1]], [pf.get_potential(trajectory_z[-1])])
            
            # Final axis scaling
            if len(trajectory_z) > 1:
                ax_traj.set_xlim([0, max(trajectory_time)])
                z_min, z_max = min(trajectory_z), max(trajectory_z)
                z_range = max(abs(z_min), abs(z_max)) * 1.2
                ax_traj.set_ylim([-z_range, z_range])
                
                # Final force axis scaling for trajectory plot
                if len(trajectory_total_force) > 1:
                    force_min, force_max = min(trajectory_total_force), max(trajectory_total_force)
                    force_range = max(abs(force_min), abs(force_max)) * 1.2
                    if force_range > 0:
                        ax_traj_force.set_ylim([-force_range, force_range])
            
            # Recalculate potential values for final scaling
            final_potential_values = pf.calculate_potential()
            final_force_values = np.array([pf.get_gradient(z) for z in pf.x_positions])
            ax_pot.set_xlim(pf.x_bounds)
            pot_min, pot_max = min(final_potential_values), max(final_potential_values)
            pot_range = pot_max - pot_min
            ax_pot.set_ylim([pot_min - 0.1 * pot_range, pot_max + 0.1 * pot_range])
            
            # Final force axis scaling for potential field plot
            force_min, force_max = min(final_force_values), max(final_force_values)
            force_range = force_max - force_min
            if force_range > 0:
                ax_pot_force.set_ylim([force_min - 0.1 * force_range, force_max + 0.1 * force_range])
            
            try:
                fig.canvas.draw()
            except:
                pass
            
            print(f"Stopping... Data saved to {csv_filename}")
            print("Close the plot window to exit.")
            
            # Keep plot open until user closes it
            plt.ioff()  # Turn off interactive mode
            try:
                plt.show(block=True)  # Block until window is closed
            except:
                pass
        else:
            print(f"Stopping... Data saved to {csv_filename}")
        break

