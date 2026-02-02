from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import numpy as np
import time
import csv
import os
from datetime import datetime
import signal
from scipy import signal as scipy_signal
from potential_field_class import PotentialField, PotentialField1D, PotentialFieldFlat, PotentialField1DFlat

# robot_ip = "192.168.1.104"
ROBOT_IP = "169.254.9.43"
DT = 1.0 / 400.0

# Control parameters
K_XY, B_XY = 2.5, 50.0
K_Z, B_Z = 30.0, 90.0

# Filter parameters
CUTOFF_FREQ = 25.0
FILTER_ORDER = 4
DEADBAND_THRESHOLD = 1.0
BOUNDARY_FORCE_GAIN = 1000.0

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

# Initialize potential fields
pf_xy = PotentialFieldFlat(x_bounds=(-0.5, 1.0), y_bounds=(-1.0, 1.0), resolution=1000, scaling_factor=1.5)
pf_xy.add_obstacle(0.2, 0.0, 10, 0.05)
pf_xy.add_obstacle(0.0, 0.4, 10, 0.05)

pf_z = PotentialField1DFlat(x_bounds=(-0.5, 0.5), resolution=1000, scaling_factor=10.0, alpha=10.0)
pf_z.add_obstacle(0.0, 0.1, 0.05)

# Setup CSV logging
os.makedirs("trajectories", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join("trajectories", f"robot_trajectory_combined_{timestamp}.csv")
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'Fx', 'Fy', 'Fz', 
                     'F_pot_x', 'F_pot_y', 'F_pot_z', 'q_0_x', 'q_0_y', 'q_0_z', 
                     'velocity_x', 'velocity_y', 'velocity_z'])

# Initialize robot
j_goal = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, 0]
rtde_c.moveJ(j_goal, 0.5)
time.sleep(1.0)

qx_init, qy_init, qz_init, _, _, _ = rtde_r.getActualTCPPose()
q_goal_xy = np.array([qx_init, qy_init])
q_goal_z = qz_init

print(f"Goal XY: [{q_goal_xy[0]:.4f}, {q_goal_xy[1]:.4f}] m")
print(f"Goal Z: {q_goal_z:.4f} m")

rtde_c.moveL([qx_init, qy_init, qz_init, 0, 0, 0], 0.1, 0.1)
time.sleep(2.0)
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
        q_0_xy = np.array([qx, qy]) - q_goal_xy
        q_0_z = qz - q_goal_z
        
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
        
        # Impedance control
        qdot = np.zeros(6)
        qdot[:2] = (F_inp[:2] + K_XY * F_pot_xy) / B_XY
        qdot[2] = (F_inp[2] + K_Z * F_pot_z) / B_Z
        qdot = np.clip(qdot, -0.75, 0.75)
        
        rtde_c.speedL(qdot, 0.3, DT)
        
        # Log data
        csv_writer.writerow([
            time.time(), qx, qy, qz, qrx, qry, qrz,
            Fx, Fy, Fz,
            F_pot_xy[0], F_pot_xy[1], F_pot_z,
            q_0_xy[0], q_0_xy[1], q_0_z,
            qdot[0], qdot[1], qdot[2]
        ])
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        csv_file.close()
        try:
            rtde_c.speedStop()
            rtde_c.disconnect()
        except:
            pass
        print(f"Data saved to {csv_filename}")
        break
