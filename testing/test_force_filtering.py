from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import numpy as np
import time
from scipy import signal as scipy_signal
import signal
import sys

# Configuration
ENABLE_PLOTTING = False
robot_ip = "169.254.9.43"

# Filter parameters
sampling_freq = 1000.0  # Hz
cutoff_freq = 20.0       # Hz
filter_order = 4
deadband_threshold = 0.15  # N

# Initialize robot
rtde_c = RTDEControlInterface(robot_ip)
rtde_r = RTDEReceiveInterface(robot_ip)

# Design filter
nyquist = sampling_freq / 2.0
normal_cutoff = cutoff_freq / nyquist
b, a = scipy_signal.butter(filter_order, normal_cutoff, btype='low')

# Initialize filter state
zi_x = scipy_signal.lfilter_zi(b, a) * 0.0
zi_y = scipy_signal.lfilter_zi(b, a) * 0.0
zi_z = scipy_signal.lfilter_zi(b, a) * 0.0

# Data storage (only if plotting enabled)
if ENABLE_PLOTTING:
    import matplotlib.pyplot as plt
    timestamps = []
    F_raw_x, F_raw_y, F_raw_z = [], [], []
    F_filtered_x, F_filtered_y, F_filtered_z = [], [], []

# Shutdown handler
shutdown_flag = [False]
def signal_handler(sig, frame):
    shutdown_flag[0] = True
    print("\nStopping...")
signal.signal(signal.SIGINT, signal_handler)

# Zero sensor and start
print("Zeroing force sensor...")
rtde_c.zeroFtSensor()
time.sleep(0.5)
print("Collecting data. Press Ctrl+C to stop...")

start_time = time.time()
dt = 1.0 / sampling_freq

try:
    while not shutdown_flag[0]:
        # Read force
        Fx, Fy, Fz, _, _, _ = rtde_r.getActualTCPForce()
        F_raw = np.array([Fx, Fy, Fz])
        
        # Filter
        F_filt_x, zi_x = scipy_signal.lfilter(b, a, [F_raw[0]], zi=zi_x)
        F_filt_y, zi_y = scipy_signal.lfilter(b, a, [F_raw[1]], zi=zi_y)
        F_filt_z, zi_z = scipy_signal.lfilter(b, a, [F_raw[2]], zi=zi_z)
        F_filtered = np.array([F_filt_x[0], F_filt_y[0], F_filt_z[0]])
        
        # Apply deadband
        F_filtered[np.abs(F_filtered) < deadband_threshold] = 0.0
        
        # Store data (if plotting enabled)
        if ENABLE_PLOTTING:
            elapsed = time.time() - start_time
            timestamps.append(elapsed)
            F_raw_x.append(F_raw[0])
            F_raw_y.append(F_raw[1])
            F_raw_z.append(F_raw[2])
            F_filtered_x.append(F_filtered[0])
            F_filtered_y.append(F_filtered[1])
            F_filtered_z.append(F_filtered[2])
        
        # Print
        print(f"Raw: [{F_raw[0]:6.2f}, {F_raw[1]:6.2f}, {F_raw[2]:6.2f}] N | "
              f"Filtered: [{F_filtered[0]:6.2f}, {F_filtered[1]:6.2f}, {F_filtered[2]:6.2f}] N")
        
        time.sleep(dt)

except KeyboardInterrupt:
    pass
finally:
    rtde_c.disconnect()
    rtde_r.disconnect()
    if ENABLE_PLOTTING and len(timestamps) > 0:
        print(f"\nCollected {len(timestamps)} samples")
        
        # Convert to arrays
        timestamps = np.array(timestamps)
        F_raw_x = np.array(F_raw_x)
        F_raw_y = np.array(F_raw_y)
        F_raw_z = np.array(F_raw_z)
        F_filtered_x = np.array(F_filtered_x)
        F_filtered_y = np.array(F_filtered_y)
        F_filtered_z = np.array(F_filtered_z)
        
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'Force Filtering Test - {filter_order}th Order Butterworth, {cutoff_freq} Hz cutoff', fontsize=14)
        
        axes[0].plot(timestamps, F_raw_x, 'r-', alpha=0.5, linewidth=1, label='Raw')
        axes[0].plot(timestamps, F_filtered_x, 'r-', linewidth=2, label='Filtered')
        axes[0].set_ylabel('Fx (N)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].plot(timestamps, F_raw_y, 'g-', alpha=0.5, linewidth=1, label='Raw')
        axes[1].plot(timestamps, F_filtered_y, 'g-', linewidth=2, label='Filtered')
        axes[1].set_ylabel('Fy (N)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        axes[2].plot(timestamps, F_raw_z, 'b-', alpha=0.5, linewidth=1, label='Raw')
        axes[2].plot(timestamps, F_filtered_z, 'b-', linewidth=2, label='Filtered')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Fz (N)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
