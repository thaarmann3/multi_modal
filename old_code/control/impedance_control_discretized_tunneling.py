from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import numpy as np
import time
import csv
import os
import sys
import json
from datetime import datetime
import signal
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fields.potential_field_discrete import PotentialFieldDiscreteRemodelable

# Load configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, "configs", "impedance_control_discretized_tunneling.json")

with open(config_path, 'r') as f:
    config = json.load(f)

# Robot parameters
ROBOT_IP = config["robot"]["ip"]
DT = 1.0/config["robot"]["control_loop_dt"]
JOINT_GOAL = config["robot"]["joint_goal"]
MOVE_SPEED = config["robot"]["move_speed"]
MOVE_ACCELERATION = config["robot"]["move_acceleration"]
SPEED_LIMIT = config["robot"]["speed_limit"]
SPEEDL_ACCELERATION = config["robot"].get("speedl_acceleration", 0.2)  # Acceleration limit for speedL (m/s²)

# Visualization flag
ENABLE_LIVE_VISUALIZATION = config["visualization"]["enabled"]
PLOT_UPDATE_INTERVAL = config["visualization"]["plot_update_interval"]
PRINT_INTERVAL = config["visualization"]["print_interval"]

# Control parameters
DAMPING = config["control"]["damping"]
STIFFNESS = config["control"]["stiffness"]
FORCE_SCALE = config["control"]["force_scale"]

# Filter parameters
CUTOFF_FREQ = config["filter"]["cutoff_freq"]
FILTER_ORDER = config["filter"]["filter_order"]
DEADBAND_THRESHOLD = config["filter"]["deadband_threshold"]

# Tunneling helper function
def sigmoid(z: float) -> float:
    if z >= 0:
        ez = np.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = np.exp(z)
    return ez / (1.0 + ez)

# Tunneling intent integrator parameters
s0 = config["tunneling"]["s0"]
rho = config["tunneling"]["rho"]
I0 = config["tunneling"]["I0"]
beta_I = config["tunneling"]["beta_I"]
eps_grad = config["tunneling"]["eps_grad"]
use_angle_gate = config["tunneling"]["use_angle_gate"]
cos0 = config["tunneling"]["cos0"]
min_grad_norm = config["tunneling"]["min_grad_norm"]

# Initialize robot interfaces
rtde_c = RTDEControlInterface(ROBOT_IP)
rtde_r = RTDEReceiveInterface(ROBOT_IP)

# Setup Butterworth filter
nyquist = (1.0 / DT) / 2.0
normal_cutoff = CUTOFF_FREQ / nyquist
sos_butter = scipy_signal.butter(FILTER_ORDER, normal_cutoff, btype='low', output='sos')
zi_x = scipy_signal.sosfilt_zi(sos_butter)
zi_y = scipy_signal.sosfilt_zi(sos_butter)

# Initialize discretized potential field with bore tunneling from config
# Using relative coordinates (centered at initial position)
pf_config = config["potential_field"]
# Increase force alignment threshold to make bores less sensitive
# Higher threshold = bores only created when force is more aligned with gradient
force_alignment_threshold = pf_config.get("force_alignment_threshold", 0.85)  # Default 0.85 (was 0.75)
# Tunable parameters for bore reduction rate and sensitivity
field_bore_strength_multiplier = pf_config.get("field_bore_strength_multiplier", 0.3)
field_bore_reduction_curve = pf_config.get("field_bore_reduction_curve", 1.0)
bore_strength_accumulation_rate = pf_config.get("bore_strength_accumulation_rate", 0.05)
bore_width_default = pf_config.get("bore_width_default", 0.5)  # Bore radius/spatial extent
pf_xy = PotentialFieldDiscreteRemodelable(
    x_bounds=tuple(pf_config["x_bounds"]),
    y_bounds=tuple(pf_config["y_bounds"]),
    resolution=pf_config["resolution"],
    alpha=pf_config["alpha"],
    force_alignment_threshold=force_alignment_threshold,  # Stricter alignment requirement
    field_bore_strength_multiplier=field_bore_strength_multiplier,  # Base reduction strength
    field_bore_reduction_curve=field_bore_reduction_curve,  # Reduction curve (1.0=linear, >1.0=faster, <1.0=slower)
    bore_strength_accumulation_rate=bore_strength_accumulation_rate,  # How fast strength accumulates
    bore_width_default=bore_width_default,  # Bore radius/spatial extent (smaller = narrower bores)
)

# Setup CSV logging
log_config = config["logging"]
if log_config["enabled"]:
    trajectory_dir = os.path.join(base_dir, log_config["trajectory_dir"])
    os.makedirs(trajectory_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(trajectory_dir, f"{log_config['filename_prefix']}_{timestamp}.csv")
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'Fx', 'Fy', 'Fz', 
                         'F_pot_x', 'F_pot_y', 'q_0_x', 'q_0_y', 
                         'velocity_x', 'velocity_y', 'gamma', 'intent_I'])
else:
    csv_file = None
    csv_writer = None
    csv_filename = None

# Initialize robot
rtde_c.moveJ(JOINT_GOAL, MOVE_SPEED)
time.sleep(1.0)

qx_init, qy_init, qz_init, _, _, _ = rtde_r.getActualTCPPose()
o_nom = np.array([qx_init, qy_init])

# Tunneling intent integrator
intent_I = 0.0

# Field bore parameters from config
field_bore_config = config.get("field_bores", {})
FIELD_BORE_GAMMA_THRESHOLD = field_bore_config.get("gamma_threshold", 0.15)  # Higher threshold for field bore creation
FIELD_BORE_COOLDOWN = field_bore_config.get("cooldown_seconds", 0.5)  # Minimum time (seconds) between field bore updates
last_field_bore_time = 0.0

# Add obstacles from config (relative to initial position)

# {
#         "x": 0.0,
#         "y": 0.2,
#         "height": 200.0,
#         "width": 0.2
#       },
#       {
#         "x": -0.25,
#         "y": -0.25,
#         "height": 300.0,
#         "width": 0.2
#       },
#       {
#         "x": 0.0,
#         "y": 0.30,
#         "height": 150.0,
#         "width": 0.1
#       }

for obs in pf_config["obstacles"]:
    pf_xy.add_obstacle(obs["x"], obs["y"], obs["height"], obs["width"])

print(f"Nominal position: [{o_nom[0]:.4f}, {o_nom[1]:.4f}] m")
print(f"Potential field bounds: x={pf_xy.x_bounds}, y={pf_xy.y_bounds}")
print(f"Added {len(pf_xy.obstacles)} obstacles")

rtde_c.moveL([qx_init, qy_init, qz_init, 0, 0, 0], MOVE_ACCELERATION, MOVE_ACCELERATION)
time.sleep(0.5)
print("Moved to goal position")
rtde_c.zeroFtSensor()

# Setup live visualization (if enabled)
if ENABLE_LIVE_VISUALIZATION:
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel('Relative X (m)')
    ax.set_ylabel('Relative Y (m)')
    ax.set_title('Live Robot Control - Discretized Potential Field with Bore Tunneling')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    x_bounds, y_bounds = pf_xy.x_bounds, pf_xy.y_bounds
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    
    # Plot elements will be redrawn each update
    
    plt.tight_layout()
    fig.canvas.draw()
    plt.show(block=False)
    
    # Plot update counter (update every N iterations to avoid affecting control loop timing)
    plot_update_counter = 0
    last_plot_time = time.time()
else:
    fig = None
    ax = None

# Print counter for force output
print_counter = 0
last_print_time = time.time()

# Shutdown handling
shutdown_flag = [False]

def signal_handler(sig, frame):
    shutdown_flag[0] = True
    print("\nInterrupt received, shutting down gracefully...")

signal.signal(signal.SIGINT, signal_handler)


def get_potential_force_xy(q_0_xy):
    """Get potential field force for XY plane."""
    x_bounds, y_bounds = pf_xy.x_bounds, pf_xy.y_bounds
    
    if x_bounds[0] <= q_0_xy[0] <= x_bounds[1] and y_bounds[0] <= q_0_xy[1] <= y_bounds[1]:
        F_pot = np.array(pf_xy.get_gradient(q_0_xy[0], q_0_xy[1]))
    else:
        F_pot = np.zeros(2)

    return F_pot

# Main control loop
while True:
    try:
        if shutdown_flag[0]:
            raise KeyboardInterrupt
        
        # Get current pose and compute error
        qx, qy, qz, qrx, qry, qrz = rtde_r.getActualTCPPose()
        q_0_xy = np.array([qx, qy]) - o_nom
        
        # Get and filter force
        Fx, Fy, Fz, _, _, _ = rtde_r.getActualTCPForce()
        F_inp_sensor = np.array([Fx, Fy])
        F_inp_x, zi_x = scipy_signal.sosfilt(sos_butter, [F_inp_sensor[0]], zi=zi_x)
        F_inp_y, zi_y = scipy_signal.sosfilt(sos_butter, [F_inp_sensor[1]], zi=zi_y)
        F_inp = np.array([F_inp_x[0], F_inp_y[0]])
        F_inp[np.abs(F_inp) < DEADBAND_THRESHOLD] = 0.0
        
        # Get potential field forces (matching simulation)
        potential_force = -get_potential_force_xy(q_0_xy)
        
        # External force (matching simulation)
        f_ext = F_inp.copy()
        gradV = -potential_force
        grad_norm = float(np.linalg.norm(gradV))
        
        # Tunneling logic (matching simulation exactly)
        if grad_norm < min_grad_norm:
            gamma = 0.0
            intent_I = max(0.0, intent_I - DT * rho * intent_I)
        else:
            ghat = gradV / (grad_norm + eps_grad)
            f_up = float(np.dot(f_ext, ghat))
            
            if use_angle_gate:
                f_norm = float(np.linalg.norm(f_ext))
                if f_norm < 1e-12:
                    angle_ok = False
                else:
                    cosang = f_up / (f_norm + 1e-12)
                    angle_ok = (cosang >= cos0)
            else:
                angle_ok = True
            
            if (f_up > 0.0) and angle_ok:
                u = max(0.0, f_up - s0)
                intent_I = max(0.0, intent_I + DT * (u - rho * intent_I))
            else:
                intent_I = max(0.0, intent_I - DT * rho * intent_I)
            
            gamma = sigmoid(beta_I * (intent_I - I0))
            
            # Update bore based on tunneling intent with cooldown and higher threshold
            # Only create NEW bores when:
            # 1. Gamma is above threshold (strong tunneling intent)
            # 2. Enough time has passed since last bore creation (cooldown)
            # Updates to existing bores bypass cooldown to allow faster accumulation
            current_time = time.time()
            time_since_last_bore = current_time - last_field_bore_time
            
            if gamma > FIELD_BORE_GAMMA_THRESHOLD:
                # Check if we should update (bypass cooldown for existing bores)
                should_update = time_since_last_bore >= FIELD_BORE_COOLDOWN
                
                if should_update:
                    # Try to update/create bore - returns True if updated existing, False if created new
                    updated_existing = pf_xy.update_bore_from_force(
                        robot_x=q_0_xy[0],
                        robot_y=q_0_xy[1],
                        force_vector=f_ext,
                        tunneling_intent=gamma
                    )
                    
                    # Recalculate gradient after bore update to account for newly formed bores
                    # This ensures the control loop uses the updated potential field
                    potential_force = -get_potential_force_xy(q_0_xy)
                    gradV = -potential_force
                    grad_norm = float(np.linalg.norm(gradV))
                    
                    # Only apply cooldown for new bore creation, not updates
                    # This allows strength to accumulate faster when held at constant location
                    if not updated_existing:
                        last_field_bore_time = current_time
        
        # Impedance control (matching simulation dynamics: velocity = (potential_force + stiffness * external_force) / damping)
        # Use the most up-to-date potential_force (recalculated after bore updates if bores were updated)
        qdot = np.zeros(6)
        qdot[:2] = STIFFNESS * (potential_force + f_ext) / DAMPING
        qdot = np.clip(qdot, -SPEED_LIMIT, SPEED_LIMIT)
        
        # Use configurable acceleration limit for speedL
        rtde_c.speedL(qdot, SPEEDL_ACCELERATION, DT)
        
        # Print counter (currently disabled)
        print_counter += 1
        current_time = time.time()
        time_since_last_print = current_time - last_print_time
        
        if print_counter >= PRINT_INTERVAL and time_since_last_print >= 0.1:
            last_print_time = time.time()
            print_counter = 0
        
        # Update live visualization (if enabled)
        # Use time-based throttling to avoid affecting control loop timing
        if ENABLE_LIVE_VISUALIZATION:
            plot_update_counter += 1
            current_time = time.time()
            time_since_last_plot = current_time - last_plot_time
            
            # Update plot only if enough iterations AND enough time has passed (throttle to ~10 Hz max)
            if plot_update_counter >= PLOT_UPDATE_INTERVAL and time_since_last_plot >= 0.1:
                plot_start_time = time.time()
                
                # Clear previous dynamic elements
                ax.clear()
                ax.set_xlabel('Relative X (m)')
                ax.set_ylabel('Relative Y (m)')
                ax.set_title('Live Robot Control - Discretized Potential Field with Bore Tunneling')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                ax.set_xlim(x_bounds)
                ax.set_ylim(y_bounds)
                
                # Calculate and display potential field (this is the expensive operation)
                Z = pf_xy.calculate_potential()
                X, Y = pf_xy.X, pf_xy.Y
                Z_min = np.min(Z)
                Z_vis = Z - Z_min
                Z_max = np.max(Z_vis)
                
                # Create more contour levels, with more density at lower values
                # Use logarithmic spacing for lower values, linear for higher values
                num_low_levels = 15  # More levels in lower range
                num_high_levels = 10  # Fewer levels in higher range
                
                # Low value contours (more granular)
                low_max = Z_max * 0.3  # Focus on bottom 30% of range
                low_levels = np.linspace(0, low_max, num_low_levels)
                
                # High value contours (less granular)
                high_levels = np.linspace(low_max, Z_max, num_high_levels + 1)[1:]  # Skip first to avoid overlap
                
                # Combine levels
                contour_levels = np.concatenate([low_levels, high_levels])
                
                # Contour lines with colors and labeled values
                contour = ax.contour(X, Y, Z_vis, levels=contour_levels, cmap='viridis', alpha=0.8, linewidths=0.8)
                ax.clabel(contour, inline=True, fontsize=8, fmt='%.0f')
                
                # Draw obstacles
                for obs in pf_xy.obstacles.values():
                    circle = plt.Circle((obs['x'], obs['y']), obs['width'], 
                                      color='red', alpha=0.5, fill=False, linewidth=2)
                    ax.add_patch(circle)
                    ax.plot(obs['x'], obs['y'], 'ro', markersize=8, zorder=10)
                
                # Draw obstacle bore directions
                for obs_key, bores in pf_xy.obstacle_bores.items():
                    obs = pf_xy.obstacles[obs_key]
                    for bore in bores:
                        dir_vec = np.array(bore['direction'])
                        arrow_length = bore['width'] * 0.6
                        ax.arrow(obs['x'], obs['y'], 
                               dir_vec[0] * arrow_length, dir_vec[1] * arrow_length,
                               head_width=0.15, head_length=0.1, 
                               fc='orange', ec='orange', alpha=0.7, linewidth=2,
                               zorder=9)
                
                # Draw field bore locations
                for bore in pf_xy.field_bores:
                    dir_vec = np.array(bore['direction'])
                    arrow_length = bore['width'] * 0.6
                    ax.plot(bore['x'], bore['y'], 'o', color='cyan', markersize=8, zorder=10)
                    ax.arrow(bore['x'], bore['y'], 
                           dir_vec[0] * arrow_length, dir_vec[1] * arrow_length,
                           head_width=0.12, head_length=0.08, 
                           fc='cyan', ec='cyan', alpha=0.8, linewidth=2,
                           zorder=9)
                
                # Draw robot position
                ax.plot(q_0_xy[0], q_0_xy[1], 'bo', markersize=14, zorder=11)
                
                # Draw force vector
                if np.linalg.norm(f_ext) > 0.1:
                    force_vis = f_ext / FORCE_SCALE * 0.3
                    ax.arrow(q_0_xy[0], q_0_xy[1],
                           force_vis[0], force_vis[1],
                           head_width=0.08, head_length=0.06,
                           fc='blue', ec='blue', alpha=0.8, linewidth=2,
                           zorder=12)
                
                # Info text
                info_text_str = f"Force scale: {FORCE_SCALE:.1f}\n"
                info_text_str += f"I={intent_I:.2f}\n"
                info_text_str += f"γ={gamma:.2f}\n"
                info_text_str += f"Obstacle bores: {sum(len(b) for b in pf_xy.obstacle_bores.values())}\n"
                info_text_str += f"Field bores: {pf_xy.get_field_bore_count()}\n"
                info_text_str += f"Position: [{q_0_xy[0]:.3f}, {q_0_xy[1]:.3f}]"
                
                ax.text(
                    0.02, 0.98,
                    info_text_str,
                    transform=ax.transAxes,
                    va='top',
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=1)
                )
                
                # Non-blocking plot update (with timeout protection)
                try:
                    if not shutdown_flag[0]:
                        fig.canvas.draw_idle()
                        # Only flush events if we haven't spent too much time already
                        plot_elapsed = time.time() - plot_start_time
                        if plot_elapsed < 0.05:  # Only flush if plot took < 50ms
                            fig.canvas.flush_events()
                except:
                    pass  # Ignore errors during event flushing
                
                last_plot_time = time.time()
                plot_update_counter = 0
        
        # Log data
        if log_config["enabled"]:
            csv_writer.writerow([
                time.time(), qx, qy, qz, qrx, qry, qrz,
                Fx, Fy, Fz,
                potential_force[0], potential_force[1],
                q_0_xy[0], q_0_xy[1],
                qdot[0], qdot[1], gamma, intent_I
            ])
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        if log_config["enabled"]:
            csv_file.close()
            print(f"Data saved to {csv_filename}")
        try:
            rtde_c.speedStop()
            rtde_c.disconnect()
        except:
            pass
        if ENABLE_LIVE_VISUALIZATION and fig is not None:
            plt.close(fig)
        break
