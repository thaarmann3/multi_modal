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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fields.potential_field_class import PotentialField, PotentialField1D, PotentialFieldFlat, PotentialField1DFlat

# Load configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, "configs", "impedance_control_tunneling.json")

with open(config_path, 'r') as f:
    config = json.load(f)

# Robot parameters
ROBOT_IP = config["robot"]["ip"]
DT = config["robot"]["control_loop_dt"]
JOINT_GOAL = config["robot"]["joint_goal"]
MOVE_SPEED = config["robot"]["move_speed"]
MOVE_ACCELERATION = config["robot"]["move_acceleration"]
SPEED_LIMIT = config["robot"]["speed_limit"]

# Control parameters
K_XY = config["control"]["k_xy"]
B_XY = config["control"]["b_xy"]
K_Z = config["control"]["k_z"]
B_Z = config["control"]["b_z"]

# Filter parameters
CUTOFF_FREQ = config["filter"]["cutoff_freq"]
FILTER_ORDER = config["filter"]["filter_order"]
DEADBAND_THRESHOLD = config["filter"]["deadband_threshold"]
BOUNDARY_FORCE_GAIN = config["filter"]["boundary_force_gain"]

# Tunneling helper functions
def sigmoid(z: float) -> float:
    if z >= 0:
        ez = np.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = np.exp(z)
    return ez / (1.0 + ez)

def clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12 or n <= max_norm:
        return v
    return v * (max_norm / n)

def clamp_scalar(v: float, max_val: float) -> float:
    return max(-max_val, min(max_val, v))

# XY tunneling parameters
xy_s0 = config["tunneling_xy"]["s0"]
xy_rho = config["tunneling_xy"]["rho"]
xy_I0 = config["tunneling_xy"]["I0"]
xy_beta_I = config["tunneling_xy"]["beta_I"]
xy_alpha_o = config["tunneling_xy"]["alpha_o"]
xy_vmax_o = config["tunneling_xy"]["vmax_o"]
xy_eps_grad = config["tunneling_xy"]["eps_grad"]
xy_use_angle_gate = config["tunneling_xy"]["use_angle_gate"]
xy_cos0 = config["tunneling_xy"]["cos0"]
xy_min_grad_norm = config["tunneling_xy"]["min_grad_norm"]

# Z tunneling parameters
z_s0 = config["tunneling_z"]["s0"]
z_rho = config["tunneling_z"]["rho"]
z_I0 = config["tunneling_z"]["I0"]
z_beta_I = config["tunneling_z"]["beta_I"]
z_alpha_o = config["tunneling_z"]["alpha_o"]
z_vmax_o = config["tunneling_z"]["vmax_o"]
z_eps_grad = config["tunneling_z"]["eps_grad"]
z_min_grad_norm = config["tunneling_z"]["min_grad_norm"]

# Initialize robot interfaces
rtde_c = RTDEControlInterface(ROBOT_IP)
rtde_r = RTDEReceiveInterface(ROBOT_IP)

# Setup Butterworth filter
nyquist = (1.0 / DT) / 2.0
normal_cutoff = CUTOFF_FREQ / nyquist
sos_butter = scipy_signal.butter(FILTER_ORDER, normal_cutoff, btype='low', output='sos')
zi_x = scipy_signal.sosfilt_zi(sos_butter)
zi_y = scipy_signal.sosfilt_zi(sos_butter)
zi_z = scipy_signal.sosfilt_zi(sos_butter)

# Initialize potential fields from config
pf_xy_config = config["potential_field_xy"]
pf_xy = PotentialFieldFlat(
    x_bounds=tuple(pf_xy_config["x_bounds"]),
    y_bounds=tuple(pf_xy_config["y_bounds"]),
    resolution=pf_xy_config["resolution"],
    scaling_factor=pf_xy_config["scaling_factor"]
)
for obs in pf_xy_config["obstacles"]:
    pf_xy.add_obstacle(obs["x"], obs["y"], obs["height"], obs["width"])

pf_z_config = config["potential_field_z"]
pf_z = PotentialField1DFlat(
    x_bounds=tuple(pf_z_config["x_bounds"]),
    resolution=pf_z_config["resolution"],
    scaling_factor=pf_z_config["scaling_factor"],
    alpha=pf_z_config["alpha"]
)
for obs in pf_z_config["obstacles"]:
    pf_z.add_obstacle(obs["x"], obs["height"], obs["width"])

# Setup CSV logging
log_config = config["logging"]
if log_config["enabled"]:
    trajectory_dir = os.path.join(base_dir, log_config["trajectory_dir"])
    os.makedirs(trajectory_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(trajectory_dir, f"{log_config['filename_prefix']}_{timestamp}.csv")
else:
    csv_filename = None
    csv_file = None
    csv_writer = None
if log_config["enabled"]:
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'Fx', 'Fy', 'Fz', 
                         'F_pot_x', 'F_pot_y', 'F_pot_z', 'q_0_x', 'q_0_y', 'q_0_z', 
                         'velocity_x', 'velocity_y', 'velocity_z'])

# Initialize robot
rtde_c.moveJ(JOINT_GOAL, MOVE_SPEED)
time.sleep(1.0)

qx_init, qy_init, qz_init, _, _, _ = rtde_r.getActualTCPPose()
o_nom = np.array([qx_init, qy_init])
o_nom_z = qz_init

# Tunneling intent integrators
intent_I_xy = 0.0
intent_I_z = 0.0

print(f"Nominal position: [{o_nom[0]:.4f}, {o_nom[1]:.4f}] m")
print(f"Nominal Z: {o_nom_z:.4f} m")

rtde_c.moveL([qx_init, qy_init, qz_init, 0, 0, 0], MOVE_ACCELERATION, MOVE_ACCELERATION)
time.sleep(0.5)
print("Moved to goal position")
rtde_c.zeroFtSensor()

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

def get_potential_force_z(q_0_z):
    """Get potential field force for Z axis."""
    z_bounds = pf_z.x_bounds
    
    if z_bounds[0] <= q_0_z <= z_bounds[1]:
        F_pot = float(pf_z.get_gradient(q_0_z))
    else:
        F_pot = 0.0
    
    return F_pot

# Main control loop
while True:
    try:
        if shutdown_flag[0]:
            raise KeyboardInterrupt
        
        # Get current pose and compute error
        qx, qy, qz, qrx, qry, qrz = rtde_r.getActualTCPPose()
        q_0_xy = np.array([qx, qy]) - o_nom
        q_0_z = qz - o_nom_z
        
        # Get and filter force
        Fx, Fy, Fz, _, _, _ = rtde_r.getActualTCPForce()
        F_inp_sensor = np.array([Fx, Fy, Fz])
        F_inp_x, zi_x = scipy_signal.sosfilt(sos_butter, [F_inp_sensor[0]], zi=zi_x)
        F_inp_y, zi_y = scipy_signal.sosfilt(sos_butter, [F_inp_sensor[1]], zi=zi_y)
        F_inp_z, zi_z = scipy_signal.sosfilt(sos_butter, [F_inp_sensor[2]], zi=zi_z)
        F_inp = np.array([F_inp_x[0], F_inp_y[0], F_inp_z[0]])
        F_inp[np.abs(F_inp) < DEADBAND_THRESHOLD] = 0.0
        
        # Get potential field forces
        F_pot_xy = -get_potential_force_xy(q_0_xy)
        F_pot_z = -get_potential_force_z(q_0_z)
        
        # Tunneling logic for XY
        f_ext_xy = F_inp[:2].copy()
        gradV_xy = -F_pot_xy
        grad_norm_xy = float(np.linalg.norm(gradV_xy))
        
        if grad_norm_xy < xy_min_grad_norm:
            o_dot_xy = np.zeros(2)
            intent_I_xy = max(0.0, intent_I_xy - DT * xy_rho * intent_I_xy)
        else:
            ghat_xy = gradV_xy / (grad_norm_xy + xy_eps_grad)
            f_up_xy = float(np.dot(f_ext_xy, ghat_xy))
            
            if xy_use_angle_gate:
                f_norm_xy = float(np.linalg.norm(f_ext_xy))
                if f_norm_xy < 1e-12:
                    angle_ok = False
                else:
                    cosang = f_up_xy / (f_norm_xy + 1e-12)
                    angle_ok = (cosang >= xy_cos0)
            else:
                angle_ok = True
            
            if (f_up_xy > 0.0) and angle_ok:
                u = max(0.0, f_up_xy - xy_s0)
                intent_I_xy = max(0.0, intent_I_xy + DT * (u - xy_rho * intent_I_xy))
            else:
                intent_I_xy = max(0.0, intent_I_xy - DT * xy_rho * intent_I_xy)
            
            gamma_xy = sigmoid(xy_beta_I * (intent_I_xy - xy_I0))
            o_dot_xy = xy_alpha_o * gamma_xy * f_up_xy * ghat_xy
        
        o_dot_xy = clamp_norm(o_dot_xy, xy_vmax_o)
        o_nom = o_nom + o_dot_xy * DT
        
        # Tunneling logic for Z
        f_ext_z = F_inp[2]
        gradV_z = -F_pot_z
        grad_norm_z = abs(gradV_z)
        
        if grad_norm_z < z_min_grad_norm:
            o_dot_z = 0.0
            intent_I_z = max(0.0, intent_I_z - DT * z_rho * intent_I_z)
        else:
            ghat_z = 1.0 if gradV_z >= 0 else -1.0
            f_up_z = f_ext_z * ghat_z
            
            if f_up_z > 0.0:
                u = max(0.0, f_up_z - z_s0)
                intent_I_z = max(0.0, intent_I_z + DT * (u - z_rho * intent_I_z))
            else:
                intent_I_z = max(0.0, intent_I_z - DT * z_rho * intent_I_z)
            
            gamma_z = sigmoid(z_beta_I * (intent_I_z - z_I0))
            o_dot_z = z_alpha_o * gamma_z * f_up_z * ghat_z
        
        o_dot_z = clamp_scalar(o_dot_z, z_vmax_o)
        o_nom_z = o_nom_z + o_dot_z * DT
        
        # Impedance control
        qdot = np.zeros(6)
        qdot[:2] = (F_inp[:2] + K_XY * F_pot_xy) / B_XY
        qdot[2] = (F_inp[2] + K_Z * F_pot_z) / B_Z
        qdot = np.clip(qdot, -SPEED_LIMIT, SPEED_LIMIT)
        
        rtde_c.speedL(qdot, 0.3, DT)
        
        # Log data
        if log_config["enabled"]:
            csv_writer.writerow([
                time.time(), qx, qy, qz, qrx, qry, qrz,
                Fx, Fy, Fz,
                F_pot_xy[0], F_pot_xy[1], F_pot_z,
                q_0_xy[0], q_0_xy[1], q_0_z,
                qdot[0], qdot[1], qdot[2]
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
        break
