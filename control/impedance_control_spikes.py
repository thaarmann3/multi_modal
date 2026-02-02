from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import numpy as np
import sympy as sp
from typing import Tuple
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
from fields.potential_field_class import PotentialFieldFlat

def create_potential_field():
    """Create and configure the potential field. Update this function to change the field configuration."""
    pf = PotentialFieldFlat(x_bounds=(-.5, 1.0), y_bounds=(-1.0, 1.0), resolution=10000, scaling_factor=2.0)
    pf.add_obstacle(0.0, 0.0, 10, 0.05)    # Center spike
    pf.add_obstacle(0.0, 0.4, 10, 0.05)  
    return pf

if __name__ == "__main__":
    # robot_ip = "192.168.1.104"
    robot_ip = "169.254.9.43"

    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)

    # Control parameters
    alpha = 0.3  # Low-pass filter coefficient for force
    k = 2.5      # Scaling factor for gradient response
    b = 50.0      # Damping constant
    dt = 1.0/400.0  # Control loop time step
    
    # Butterworth filter parameters
    # Note: Control loop runs at 1000 Hz (dt = 1.0/1000.0), so filter must match this rate
    sampling_freq = 1.0 / dt  # Hz - match control loop rate (1000 Hz)
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

    # Initialize potential field
    pf = create_potential_field()

    # Setup CSV logging
    # Create trajectories folder if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trajectory_dir = os.path.join(base_dir, "trajectories")
    os.makedirs(trajectory_dir, exist_ok=True)

    # Generate filename with date/time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(trajectory_dir, f"robot_trajectory_{timestamp}.csv")
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'Fx', 'Fy', 'Fz', 'F_pot_x', 'F_pot_y', 'q_0_x', 'q_0_y', 'q_0_z', 'velocity_x', 'velocity_y'])

    # near axis
    # q_goal = sp.Point(-0.00600044, -0.1427485, 0.73381762)

    # outright
    # q_goal = sp.Point(-0.29139538, -0.13562899, 0.68669564)
    # # Convert SymPy Point to regular Python floats
    # goal_pos = np.array([float(q_goal.x), float(q_goal.y), float(q_goal.z)])

    # Reset Joints
    j_goal = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, 0]
    rtde_c.moveJ(j_goal, 0.5)

    # Goal position (origin in Z direction)
    # Get current position and use Z as goal
    qx_init, qy_init, qz_init, qrx_init, qry_init, qrz_init = rtde_r.getActualTCPPose()
    q_goal = np.array([qx_init, qy_init, qz_init])  # Goal is current Z position (origin in relative coordinates)

    print(f"Goal position: [{q_goal[0]:.4f}, {q_goal[1]:.4f}, {q_goal[2]:.4f}] m")

    # Move to the goal position in Cartesian space
    goal_pose = [q_goal[0], q_goal[1], q_goal[2], 0, 0, 0]  # [x, y, z, rx, ry, rz]
    rtde_c.moveL(goal_pose, 0.1, 0.1)
    time.sleep(2.0)  # Wait longer for movement to complete
    print("Moved to goal position")
    rtde_c.zeroFtSensor()

    # Set initial payload mass (default value before control loop)
    initial_mass = 0.5  # kg
    # rtde_c.setPayload(initial_mass, [0.0, 0.0, 0.0])

    # Setup live plotting (if enabled)
    if ENABLE_LIVE_PLOTTING:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate and display potential field
        Z_pf = pf.calculate_potential()
        contour = ax.contour(pf.X, pf.Y, Z_pf, levels=20, alpha=0.5, cmap='viridis')
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('Relative X (m)')
        ax.set_ylabel('Relative Y (m)')
        ax.set_title('Live Robot Trajectory with Potential Field')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(pf.x_bounds)
        ax.set_ylim(pf.y_bounds)
        
        # Initialize trajectory storage
        trajectory_x = []
        trajectory_y = []
        trajectory_line, = ax.plot([], [], 'b-', linewidth=2, label='Robot Path')
        current_point, = ax.plot([], [], 'ro', markersize=8, label='Current Position')
        start_point, = ax.plot([], [], 'go', markersize=10, label='Start')
        goal_point = ax.plot(0, 0, 'g*', markersize=15, label='Goal')[0]  # Goal is at origin in relative coordinates
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        fig.canvas.draw()  # Initial draw
        plt.show(block=False)
        
        # Plot update counter (update every N iterations to avoid affecting control loop timing)
        plot_update_counter = 0
        plot_update_interval = 15  # Update plot every 15 iterations (~30 Hz at 500 Hz control loop)
    else:
        # Initialize empty trajectory storage (for CSV logging only)
        trajectory_x = []
        trajectory_y = []
        fig = None
        ax = None

    # Flag for clean shutdown (use list to allow modification in nested function)
    shutdown_flag = [False]

    # Signal handler for KeyboardInterrupt (works even when plot has focus)
    def signal_handler(sig, frame):
        shutdown_flag[0] = True
        print("\nInterrupt received, shutting down gracefully...")

    signal.signal(signal.SIGINT, signal_handler)

    F_inp = np.array([0, 0, 0])

    while True:
        try:
            # Check for shutdown flag (set by signal handler)
            if shutdown_flag[0]:
                raise KeyboardInterrupt
           
            qx, qy, qz, qrx, qry, qrz = rtde_r.getActualTCPPose()
            q_0 = np.array([qx, qy, qz]) - q_goal

            Fx, Fy, Fz, T, Ty, Tz = rtde_r.getActualTCPForce()
            F_inp_sensor = np.array([Fx, Fy, Fz])
            
            # Apply 4th order Butterworth filter to each force component using SOS form
            F_inp_x, zi_x = scipy_signal.sosfilt(sos_butter, [F_inp_sensor[0]], zi=zi_x)
            F_inp_y, zi_y = scipy_signal.sosfilt(sos_butter, [F_inp_sensor[1]], zi=zi_y)
            F_inp_z, zi_z = scipy_signal.sosfilt(sos_butter, [F_inp_sensor[2]], zi=zi_z)
            F_inp = np.array([F_inp_x[0], F_inp_y[0], F_inp_z[0]])
            
            # Apply deadband: set components below threshold to zero
            F_inp[np.abs(F_inp) < deadband_threshold] = 0.0
            
            # Old low-pass filter (commented out - replaced with Butterworth filter)
            # F_inp = alpha*F_inp + (1-alpha)*F_inp_sensor

            # # Check if q_0 is outside potential field bounds
            x_bounds = pf.x_bounds
            y_bounds = pf.y_bounds
            # F_bounds = np.zeros(2)
            
            # # Add strong repulsive force if outside bounds
            # if q_0[0] < x_bounds[0]:
            #     F_bounds[0] = 1000 * (x_bounds[0] - q_0[0])  # Push right
            # elif q_0[0] > x_bounds[1]:
            #     F_bounds[0] = 1000 * (x_bounds[1] - q_0[0])  # Push left
                
            # if q_0[1] < y_bounds[0]:
            #     F_bounds[1] = 1000 * (y_bounds[0] - q_0[1])  # Push up
            # elif q_0[1] > y_bounds[1]:
            #     F_bounds[1] = 1000 * (y_bounds[1] - q_0[1])  # Push down
            
            # Get potential field force
            if x_bounds[0] <= q_0[0] <= x_bounds[1] and y_bounds[0] <= q_0[1] <= y_bounds[1]:
                F_pot = -np.array(pf.get_gradient(q_0[0], q_0[1]))
            else:
                F_pot = np.zeros(2)  # No potential field force outside bounds
            
            F_pot[np.abs(F_pot) < deadband_threshold] = 0.0

            # Combine bounds force with potential field force
            # F_pot = F_pot + F_bounds
            
            qdot = np.zeros(6)
           
            # Impedance control in X and Y directions
            # Robot moves based on input force and scaled potential field gradient
            # Will naturally come to rest at local minima where net force (gradient) is zero
            # Goal position is only for visualization (q_0), not used in control
            # F_pot is the gradient (force from potential field), scaled by k
            qdot[:2] = (F_inp[:2] + k * F_pot[:2])/b

            # Calculate total control force magnitude and update payload mass
            F_total_2d = F_inp[:2] + F_pot[:2]
            F_total_magnitude = np.linalg.norm(F_total_2d)
            
            # Convert force to equivalent mass using gravity (F = mg -> m = F/g)
            mass = np.clip(F_total_magnitude / 9.81, min_mass, max_mass)
            
            # Update payload mass dynamically to prevent protective stops
            # rtde_c.setPayload(mass, [0.0, 0.0, 0.0])

            # Limit velocity to maximum of 0.5 for any component
            # qdot = np.clip(qdot, -0.75, 0.75)
            rtde_c.speedL(qdot, 0.3, dt)
            
            # Log data to CSV
            current_time = time.time()
            csv_writer.writerow([
                current_time,
                qx, qy, qz, qrx, qry, qrz,  # Pose
                F_inp[0], F_inp[1], F_inp[2],  # Force
                F_pot[0], F_pot[1],  # Potential force
                q_0[0], q_0[1], q_0[2],  # Position relative to goal
                qdot[0], qdot[1]  # Velocity
            ])
            
            # Store trajectory points
            trajectory_x.append(q_0[0])
            trajectory_y.append(q_0[1])
            
            # Update plot periodically (non-blocking, doesn't affect dt) - only if enabled
            if ENABLE_LIVE_PLOTTING:
                plot_update_counter += 1
                if plot_update_counter >= plot_update_interval:
                    if len(trajectory_x) > 0:
                        # Update trajectory line
                        trajectory_line.set_data(trajectory_x, trajectory_y)
                        
                        # Update current position marker
                        current_point.set_data([trajectory_x[-1]], [trajectory_y[-1]])
                        
                        # Update start point (first position)
                        if len(trajectory_x) == 1:
                            start_point.set_data([trajectory_x[0]], [trajectory_y[0]])
                    
                    # Non-blocking plot update - use draw_idle only, avoid flush_events which can block
                    fig.canvas.draw_idle()
                    # Try to flush events, but don't let it block interrupts
                    try:
                        # Only process events if shutdown hasn't been requested
                        if not shutdown_flag[0]:
                            fig.canvas.flush_events()
                    except:
                        pass  # Ignore errors during event flushing
                    plot_update_counter = 0
        
            print(f"Force: {F_inp}")
            print(f"Potential Force: {F_pot}")
            print(f"Velocity: {qdot}")
            # time.sleep(0.1)



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
            if ENABLE_LIVE_PLOTTING and len(trajectory_x) > 0:
                trajectory_line.set_data(trajectory_x, trajectory_y)
                current_point.set_data([trajectory_x[-1]], [trajectory_y[-1]])
                
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
    

