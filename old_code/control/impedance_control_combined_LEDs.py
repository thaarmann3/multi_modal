from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import numpy as np
import time
import csv
import os
from datetime import datetime
import signal
import math
import threading
import queue
from scipy import signal as scipy_signal
from potential_field_class import PotentialField, PotentialField1D, PotentialFieldFlat, PotentialField1DFlat

try:
    import serial  # pyserial
except Exception:
    serial = None

# robot_ip = "192.168.1.104"
ROBOT_IP = "169.254.9.43"
DT = 1.0 / 400.0

# --- NeoPixel / Arduino LED feedback (optional but enabled if serial port opens) ---
# The Arduino sketch expects 16 RGB triples in the format:
#   r,g,b;r,g,b;... (16 total) + '\n'
ARDUINO_PORT = "COM6"  # e.g. "/dev/ttyACM0" or "COM5"
ARDUINO_BAUD = 250000
LED_COUNT = 16
LED_RADIUS_M = 0.0508  # 2 inches in meters
LED_MAX_BRIGHTNESS = 30  # cap at 30/255
LED_UPDATE_HZ = 30.0     # decimate LED updates so the 400 Hz control loop isn't slowed
# Convert (scalar) gradient to brightness: brightness = min(LED_MAX_BRIGHTNESS, abs(g) * gain)
# Tune this if the LEDs are always off or always saturated.
GRAD_TO_BRIGHTNESS_GAIN = 10.0
GRAD_DEADBAND = 0.0


class ArduinoLedStreamer:
    """Non-blocking LED writer: enqueue frames in control loop, write in a background thread."""

    def __init__(self, port: str, baud: int):
        self._enabled = False
        self._ser = None
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=1)  # keep only latest frame
        self._stop = threading.Event()
        self._thread = None

        if not port:
            print("[LED] ARDUINO_PORT not set; LED streaming disabled.")
            return
        if serial is None:
            print("[LED] pyserial not available; LED streaming disabled.")
            return

        try:
            # write_timeout=0 keeps writes from blocking for long if OS buffer is full
            self._ser = serial.Serial(port, baud, timeout=0, write_timeout=0)
            # Many Arduinos reset when the port opens; wait a moment so we don't drop the first frames.
            time.sleep(2.0)
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            self._enabled = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            print(f"[LED] Streaming enabled on {port} @ {baud} baud")
        except Exception as e:
            print(f"[LED] Failed to open serial port '{port}': {e}. LED streaming disabled.")

    def _run(self):
        while not self._stop.is_set():
            try:
                line = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._enabled or self._ser is None:
                continue

            try:
                self._ser.write(line.encode("ascii"))
            except Exception:
                # If the cable disconnects or the port errors, disable quietly.
                self._enabled = False

    def try_send(self, rgb_list):
        """rgb_list: iterable of (r,g,b) length 16. Non-blocking; drops older frame if needed."""
        if not self._enabled:
            return

        parts = [f"{int(r)},{int(g)},{int(b)}" for (r, g, b) in rgb_list]
        line = ";".join(parts) + "\n"

        try:
            self._q.put_nowait(line)
        except queue.Full:
            try:
                _ = self._q.get_nowait()  # drop previous
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(line)
            except queue.Full:
                pass

    def close(self):
        if not self._enabled and self._ser is None:
            return

        self._stop.set()
        try:
            # Best-effort: turn LEDs off
            off_line = ";".join(["0,0,0"] * LED_COUNT) + "\n"
            if self._ser is not None:
                try:
                    self._ser.write(off_line.encode("ascii"))
                except Exception:
                    pass
        finally:
            try:
                if self._ser is not None:
                    self._ser.close()
            except Exception:
                pass


def _clamp_u8(v: float) -> int:
    v = int(v)
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


def gradient_to_rgb(g_scalar: float):
    """Map signed gradient to RGB with requested scheme.

    - g == 0 -> off
    - g > 0  -> red (brighter as g increases)
    - g < 0  -> green (brighter as |g| increases)
    Brightness is capped at LED_MAX_BRIGHTNESS (out of 255).
    """
    if abs(g_scalar) <= GRAD_DEADBAND:
        return (0, 0, 0)

    brightness = min(LED_MAX_BRIGHTNESS, int(abs(g_scalar) * GRAD_TO_BRIGHTNESS_GAIN))
    if brightness <= 0:
        return (0, 0, 0)

    if g_scalar > 0:
        return (_clamp_u8(brightness), 0, 0)
    else:
        return (0, _clamp_u8(brightness), 0)


def sample_ring_gradients_xy(q_center_xy: np.ndarray):
    """Sample 16 points on a ring (radius LED_RADIUS_M) around q_center_xy.

    Returns:
      rgb_list: length-16 list of (r,g,b) mapped from scalar gradient.

    Note on "gradient": pf_xy.get_gradient(x,y) returns [dV/dx, dV/dy].
    We convert this vector to a scalar per sample direction using a radial projection:
      g_scalar = grad · u_r
    where u_r is the unit vector from the center to that sample point.
    """
    rgb_list = []

    x_bounds, y_bounds = pf_xy.x_bounds, pf_xy.y_bounds

    for i in range(LED_COUNT):
        theta = 2.0 * math.pi * (i / LED_COUNT)
        ur = np.array([math.cos(theta), math.sin(theta)], dtype=float)
        q_s = q_center_xy + LED_RADIUS_M * ur

        # Bounds-check like get_potential_force_xy
        if x_bounds[0] <= q_s[0] <= x_bounds[1] and y_bounds[0] <= q_s[1] <= y_bounds[1]:
            grad_vec = np.array(pf_xy.get_gradient(q_s[0], q_s[1]), dtype=float)
            g_scalar = float(np.dot(grad_vec, ur))
        else:
            g_scalar = 0.0

        rgb_list.append(gradient_to_rgb(g_scalar))

    return rgb_list

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

# Start LED streaming (non-blocking) if the Arduino serial port is available.
led_streamer = ArduinoLedStreamer(ARDUINO_PORT, ARDUINO_BAUD)
_led_period_s = (1.0 / LED_UPDATE_HZ) if LED_UPDATE_HZ > 0 else 1e9
_next_led_t = time.monotonic()

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

        # LED feedback: sample 16 points on a ring around the *current* XY (in the same
        # coordinate system used by the potential field, i.e., relative to q_goal_xy),
        # map signed scalar gradient to (R,G,B), and stream to Arduino without
        # blocking the 400 Hz control loop.
        _t_now = time.monotonic()
        if _t_now >= _next_led_t:
            led_streamer.try_send(sample_ring_gradients_xy(q_0_xy))
            _next_led_t = _t_now + _led_period_s
        
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
        try:
            led_streamer.close()
        except Exception:
            pass
        csv_file.close()
        try:
            rtde_c.speedStop()
            rtde_c.disconnect()
        except:
            pass
        print(f"Data saved to {csv_filename}")
        break
