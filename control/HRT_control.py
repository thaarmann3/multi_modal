"""
HRT (Human Subject Trials) Control - Impedance control with modality switching for HST testing.

Supports 5 modalities:
  1. Fixed bar (non-adjustable) - robot holds position, no response to force
  2. Touch - impedance control only
  3. Touch + LED ring - impedance + LED gradient feedback
  4. Touch + voice - impedance + voice obstacle addition
  5. Touch + voice + LED ring - full feature set

Operator commands (type in terminal, press Enter):
  modality <1-5>       - Switch modality (1=fixed, 2=touch, 3=touch+LED, 4=touch+voice, 5=all)
  reset                - Move handlebar to global minimum, clear bores
  position_bar left    - Move bar left of user (for sit-to-stand trials)
  position_bar right   - Move bar right of user (for sit-to-stand trials)
  start_trial <dir>    - Start cardinal trial (dir: north, south, east, west)
  start_trial          - Start sit-to-stand trial (when in sit_to_stand mode, no target, manual end)
  end_trial            - Manually end trial and record metrics
  qualitative <1-7>    - Record qualitative score for current modality
  sit_to_stand         - Toggle sit-to-stand trial mode
  quit                 - Shutdown
"""

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
import math
import threading
import queue

try:
    import serial  # pyserial
except Exception:
    serial = None

# --- Modality constants ---
MODALITY_FIXED = 1
MODALITY_TOUCH = 2
MODALITY_TOUCH_LED = 3
MODALITY_TOUCH_VOICE = 4
MODALITY_TOUCH_VOICE_LED = 5

MODALITY_NAMES = {
    MODALITY_FIXED: "Fixed bar (non-adjustable)",
    MODALITY_TOUCH: "Touch",
    MODALITY_TOUCH_LED: "Touch + LED ring",
    MODALITY_TOUCH_VOICE: "Touch + voice",
    MODALITY_TOUCH_VOICE_LED: "Touch + voice + LED ring",
}


class ArduinoLedStreamer:
    """Non-blocking LED writer: enqueue frames in control loop, write in a background thread."""

    def __init__(self, port: str, baud: int, led_count: int, print_status: bool = False):
        self._enabled = False
        self._ser = None
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None
        self._led_count = led_count

        if not port:
            if print_status:
                print("[LED] ARDUINO_PORT not set; LED streaming disabled.")
            return
        if serial is None:
            if print_status:
                print("[LED] pyserial not available; LED streaming disabled.")
            return

        try:
            self._ser = serial.Serial(port, baud, timeout=0, write_timeout=0)
            time.sleep(2.0)
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            self._enabled = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            if print_status:
                print(f"[LED] Streaming enabled on {port} @ {baud} baud")
        except Exception as e:
            if print_status:
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
                self._enabled = False

    def try_send(self, rgb_list):
        if not self._enabled:
            return
        parts = [f"{int(r)},{int(g)},{int(b)}" for (r, g, b) in rgb_list]
        line = ";".join(parts) + "\n"
        try:
            self._q.put_nowait(line)
        except queue.Full:
            try:
                _ = self._q.get_nowait()
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
            off_line = ";".join(["0,0,0"] * self._led_count) + "\n"
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


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fields.potential_field_discrete import PotentialFieldDiscreteRemodelable
from sbert.embedding_to_pf_subscriber import EmbeddingToPFPipeline

# Load configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, "configs", "HRT_control.json")

with open(config_path, 'r') as f:
    config = json.load(f)

# Printing flags
print_cfg = config.get("printing", {})
PRINT_LED_STATUS = print_cfg.get("led_status", False)
PRINT_STARTUP_INFO = print_cfg.get("startup_info", True)
PRINT_SHUTDOWN_INTERRUPT = print_cfg.get("shutdown_interrupt", False)
PRINT_VOICE_TRANSCRIPTION = print_cfg.get("voice_transcription", True)
PRINT_VOICE_OBSTACLE = print_cfg.get("voice_obstacle", False)
PRINT_LOOP_RATE = print_cfg.get("loop_rate", False)
PRINT_SHUTDOWN_MESSAGE = print_cfg.get("shutdown_message", False)
PRINT_LOG_SAVED = print_cfg.get("log_saved", False)

# Robot parameters
ROBOT_IP = config["robot"]["ip"]
DT = 1.0 / config["robot"]["control_loop_dt"]
JOINT_GOAL = config["robot"]["joint_goal"]
MOVE_SPEED = config["robot"]["move_speed"]
MOVE_ACCELERATION = config["robot"]["move_acceleration"]
SPEED_LIMIT = config["robot"]["speed_limit"]
SPEEDL_ACCELERATION = config["robot"].get("speedl_acceleration", 0.2)
PAYLOAD_MASS = config["robot"].get("payload_mass_kg", 0.0)
PAYLOAD_COG = config["robot"].get("payload_cog", [0.0, 0.0, 0.0])

# Control parameters
DAMPING = config["control"]["damping"]
FORCE_SCALE = config["control"]["force_scale"]

# Filter parameters
CUTOFF_FREQ = config["filter"]["cutoff_freq"]
FILTER_ORDER = config["filter"]["filter_order"]
DEADBAND_THRESHOLD = config["filter"]["deadband_threshold"]

# HRT trial parameters
hrt_config = config.get("hrt", {})
CARDINAL_TARGET_M = hrt_config.get("cardinal_target_distance_m", 0.15)
TARGET_REACH_TOLERANCE_M = hrt_config.get("target_reach_tolerance_m", 0.03)
SIT_TO_STAND_OFFSET_M = hrt_config.get("sit_to_stand_bar_offset_m", 0.2)

# Tunneling parameters
s0 = config["tunneling"]["s0"]
rho = config["tunneling"]["rho"]
I0 = config["tunneling"]["I0"]
beta_I = config["tunneling"]["beta_I"]
eps_grad = config["tunneling"]["eps_grad"]
use_angle_gate = config["tunneling"]["use_angle_gate"]
cos0 = config["tunneling"]["cos0"]
min_grad_norm = config["tunneling"]["min_grad_norm"]

# LED parameters
ARDUINO_PORT = config["leds"]["port"]
ARDUINO_BAUD = config["leds"]["baud"]
LED_COUNT = config["leds"]["count"]
LED_RADIUS_M = config["leds"]["radius"]
LED_MAX_BRIGHTNESS = config["leds"]["max_brightness"]
LED_UPDATE_HZ = config["leds"]["update_hz"]
GRAD_TO_BRIGHTNESS_GAIN = config["leds"]["grad_to_brightness_gain"]
GRAD_DEADBAND = config["leds"]["grad_deadband"]

# Potential field
pf_config = config["potential_field"]
force_alignment_threshold = pf_config.get("force_alignment_threshold", 0.85)
field_bore_strength_multiplier = pf_config.get("field_bore_strength_multiplier", 0.3)
field_bore_reduction_curve = pf_config.get("field_bore_reduction_curve", 1.0)
bore_strength_accumulation_rate = pf_config.get("bore_strength_accumulation_rate", 0.05)
bore_width_default = pf_config.get("bore_width_default", 0.5)
obstacle_bore_width_multiplier = pf_config.get("obstacle_bore_width_multiplier", 2.0)
obstacle_bore_strength_multiplier = pf_config.get("obstacle_bore_strength_multiplier", 0.5)
obstacle_bore_distance_falloff_multiplier = pf_config.get("obstacle_bore_distance_falloff_multiplier")
OBS_DELTA = pf_config.get("obs_delta", 0.1)

voice_config = config.get("voice", {})
VOICE_KEYWORDS = voice_config.get("keywords", ["robot"])
VOICE_REQUIRE_SIMILARITY = voice_config.get("require_similarity", True)
VOICE_DEBUG_SKIP = voice_config.get("debug_skip", False)
VOICE_OBSTACLE_DISTANCE_SCALE = voice_config.get("obstacle_distance_scale", 2.0)
VOICE_OBSTACLE_AMPLITUDE_SCALE = voice_config.get("obstacle_amplitude_scale", 2.5)
VOICE_OBSTACLE_MIN_DISTANCE_M = voice_config.get("obstacle_min_distance_m", 0.08)
VOICE_OBSTACLE_MIN_RADIUS_M = voice_config.get("obstacle_min_radius_m", 0.03)

# Visualization
ENABLE_LIVE_VISUALIZATION = config["visualization"]["enabled"]
PLOT_UPDATE_INTERVAL = config["visualization"]["plot_update_interval"]
PRINT_INTERVAL = config["visualization"]["print_interval"]

# Logging
log_config = config["logging"]
HRT_METRICS_DIR = log_config.get("hrt_metrics_dir", "hrt_metrics")
HRT_METRICS_PREFIX = log_config.get("hrt_metrics_prefix", "hrt_trial")
field_bore_config = config.get("field_bores", {})
FIELD_BORE_GAMMA_THRESHOLD = field_bore_config.get("gamma_threshold", 0.15)
FIELD_BORE_COOLDOWN = field_bore_config.get("cooldown_seconds", 0.5)


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = np.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = np.exp(z)
    return ez / (1.0 + ez)


def check_keyword(sentence: str, keywords: list) -> bool:
    if sentence is None or not keywords:
        return False
    sentence_lower = sentence.lower()
    return any(keyword.lower() in sentence_lower for keyword in keywords)


def check_similarity(ridge: dict, nn: dict, threshold: float) -> bool:
    if ridge is None or nn is None:
        return False
    rv = np.array([ridge["x_m"], ridge["y_m"]], dtype=np.float64)
    nv = np.array([nn["x_m"], nn["y_m"]], dtype=np.float64)
    return np.linalg.norm(rv - nv) <= threshold


def gradient_to_rgb(g_scalar: float):
    if abs(g_scalar) <= GRAD_DEADBAND:
        return (0, 0, 0)
    brightness = min(LED_MAX_BRIGHTNESS, int(abs(g_scalar) * GRAD_TO_BRIGHTNESS_GAIN))
    if brightness <= 0:
        return (0, 0, 0)
    if g_scalar > 0:
        return (_clamp_u8(brightness), 0, 0)
    return (0, _clamp_u8(brightness), 0)


def sample_ring_gradients_xy(q_center_xy: np.ndarray, pf_xy_obj):
    rgb_list = []
    x_bounds, y_bounds = pf_xy_obj.x_bounds, pf_xy_obj.y_bounds
    for i in range(LED_COUNT):
        theta = 2.0 * math.pi * (i / LED_COUNT)
        ur = np.array([math.cos(theta), math.sin(theta)], dtype=float)
        q_s = q_center_xy + LED_RADIUS_M * ur
        if x_bounds[0] <= q_s[0] <= x_bounds[1] and y_bounds[0] <= q_s[1] <= y_bounds[1]:
            grad_vec = np.array(pf_xy_obj.get_gradient(q_s[0], q_s[1]), dtype=float)
            g_scalar = float(np.dot(grad_vec, ur))
        else:
            g_scalar = 0.0
        rgb_list.append(gradient_to_rgb(g_scalar))
    return rgb_list


# --- HRT shared state (thread-safe) ---
hrt_state = {
    "modality": MODALITY_TOUCH_LED,  # Default for testing
    "trial_active": False,
    "trial_target_xy": None,  # [x, y] in relative coords, or None
    "trial_start_time": None,
    "trial_max_force": 0.0,
    "trial_start_pos": None,
    "trial_type": None,  # "cardinal" or "sit_to_stand"
    "trial_direction": None,  # north, south, east, west
    "qualitative_scores": [],  # [(modality, score, trial_type), ...]
    "command_queue": queue.Queue(),
    "reset_requested": False,
    "shutdown_requested": False,
    "sit_to_stand_mode": False,  # Bar offset for sit-to-stand
    "position_bar_requested": None,  # ("left"|"right", offset_m) or None
    "lock": threading.Lock(),
}

# Cardinal direction -> target offset (relative to global min 0,0)
CARDINAL_OFFSETS = {
    "north": np.array([0.0, CARDINAL_TARGET_M]),
    "south": np.array([0.0, -CARDINAL_TARGET_M]),
    "east": np.array([CARDINAL_TARGET_M, 0.0]),
    "west": np.array([-CARDINAL_TARGET_M, 0.0]),
}


def input_thread_fn():
    """Background thread: read operator commands from stdin."""
    print("\n[HRT] Operator commands: modality <1-5>, reset, start_trial <north|south|east|west>, end_trial, qualitative <1-7>, sit_to_stand, quit\n")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip().lower()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0] if parts else ""
            with hrt_state["lock"]:
                if hrt_state["shutdown_requested"]:
                    break
            hrt_state["command_queue"].put((cmd, parts[1:]))
        except EOFError:
            break
        except Exception:
            pass


# Initialize robot interfaces
rtde_c = RTDEControlInterface(ROBOT_IP)
rtde_r = RTDEReceiveInterface(ROBOT_IP)

# Butterworth filter
nyquist = (1.0 / DT) / 2.0
normal_cutoff = CUTOFF_FREQ / nyquist
sos_butter = scipy_signal.butter(FILTER_ORDER, normal_cutoff, btype='low', output='sos')
zi_x = scipy_signal.sosfilt_zi(sos_butter)
zi_y = scipy_signal.sosfilt_zi(sos_butter)

# Potential field
pf_xy = PotentialFieldDiscreteRemodelable(
    x_bounds=tuple(pf_config["x_bounds"]),
    y_bounds=tuple(pf_config["y_bounds"]),
    resolution=pf_config["resolution"],
    alpha=pf_config["alpha"],
    force_alignment_threshold=force_alignment_threshold,
    field_bore_strength_multiplier=field_bore_strength_multiplier,
    field_bore_reduction_curve=field_bore_reduction_curve,
    bore_strength_accumulation_rate=bore_strength_accumulation_rate,
    bore_width_default=bore_width_default,
    obstacle_bore_width_multiplier=obstacle_bore_width_multiplier,
    obstacle_bore_strength_multiplier=obstacle_bore_strength_multiplier,
    obstacle_bore_distance_falloff_multiplier=obstacle_bore_distance_falloff_multiplier,
)
pf_xy.clear_obstacles()
pf_xy.clear_bores(clear_field_bores=True)

for obs in pf_config["obstacles"]:
    pf_xy.add_obstacle(obs["x"], obs["y"], obs["height"], obs["width"])

# Global minimum in relative coords (center of bowl)
GLOBAL_MIN_REL = np.array([0.0, 0.0])


def get_potential_force_xy(q_0_xy, pf):
    x_bounds, y_bounds = pf.x_bounds, pf.y_bounds
    if x_bounds[0] <= q_0_xy[0] <= x_bounds[1] and y_bounds[0] <= q_0_xy[1] <= y_bounds[1]:
        return np.array(pf.get_gradient(q_0_xy[0], q_0_xy[1]))
    return np.zeros(2)


# Initialize robot
rtde_c.moveJ(JOINT_GOAL, MOVE_SPEED)
time.sleep(1.0)

qx_init, qy_init, qz_init, _, _, _ = rtde_r.getActualTCPPose()
o_nom = np.array([qx_init, qy_init])

intent_I = 0.0
last_field_bore_time = 0.0

# CSV trajectory logging
if log_config["enabled"]:
    trajectory_dir = os.path.join(base_dir, log_config["trajectory_dir"])
    os.makedirs(trajectory_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(trajectory_dir, f"{log_config['filename_prefix']}_HRT_{timestamp}.csv")
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'Fx', 'Fy', 'Fz', 'F_pot_x', 'F_pot_y', 'q_0_x', 'q_0_y',
                         'velocity_x', 'velocity_y', 'gamma', 'intent_I', 'modality', 'trial_active'])
else:
    csv_file = None
    csv_writer = None
    csv_filename = None

# HRT metrics CSV
hrt_metrics_dir = os.path.join(base_dir, HRT_METRICS_DIR)
os.makedirs(hrt_metrics_dir, exist_ok=True)
hrt_metrics_path = os.path.join(hrt_metrics_dir, f"{HRT_METRICS_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
hrt_metrics_file = open(hrt_metrics_path, 'w', newline='')
hrt_metrics_writer = csv.writer(hrt_metrics_file)
hrt_metrics_writer.writerow([
    'timestamp', 'modality', 'modality_name', 'trial_type', 'trial_direction',
    'time_to_target_s', 'target_distance_m', 'max_force_N', 'deviation_m', 'target_x', 'target_y',
    'final_x', 'final_y', 'qualitative_score', 'sit_to_stand_mode'
])

# Voice pipeline
pipeline = EmbeddingToPFPipeline()
_ = pipeline.model
pipeline.start_background(use_ridge=True, use_nn=True, print_params=False)
time.sleep(0.5)

rtde_c.moveL([qx_init, qy_init, qz_init, 0, 0, 0], MOVE_ACCELERATION, MOVE_ACCELERATION)
time.sleep(0.5)
rtde_c.zeroFtSensor()

# Set payload for gravity compensation (reduces protective stops from force/torque limits)
if PAYLOAD_MASS > 0:
    try:
        rtde_c.setPayload(PAYLOAD_MASS, PAYLOAD_COG)
        if PRINT_STARTUP_INFO:
            print(f"[HRT] Payload set: {PAYLOAD_MASS} kg, CoG={PAYLOAD_COG}")
    except Exception as e:
        if PRINT_STARTUP_INFO:
            print(f"[HRT] setPayload failed (may be unsupported): {e}")

# LED streamer
led_streamer = ArduinoLedStreamer(ARDUINO_PORT, ARDUINO_BAUD, LED_COUNT, PRINT_LED_STATUS)
_led_period_s = (1.0 / LED_UPDATE_HZ) if LED_UPDATE_HZ > 0 else 1e9
_next_led_t = time.monotonic()

# Visualization
if ENABLE_LIVE_VISUALIZATION:
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel('Relative X (m)')
    ax.set_ylabel('Relative Y (m)')
    ax.set_title('HRT Control - Human Subject Trials')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    x_bounds, y_bounds = pf_xy.x_bounds, pf_xy.y_bounds
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    plt.tight_layout()
    fig.canvas.draw()
    plt.show(block=False)
    plot_update_counter = 0
    last_plot_time = time.time()
else:
    fig = None
    ax = None

print_counter = 0
last_print_time = time.time()

shutdown_flag = [False]


def signal_handler(sig, frame):
    shutdown_flag[0] = True
    with hrt_state["lock"]:
        hrt_state["shutdown_requested"] = True
    if PRINT_SHUTDOWN_INTERRUPT:
        print("\nInterrupt received, shutting down gracefully...")


signal.signal(signal.SIGINT, signal_handler)


def reset_to_global_minimum(rtde_ctrl, rtde_recv, o_nom_abs, qz):
    """Move robot to global minimum (center), clear bores."""
    rtde_ctrl.speedStop()
    time.sleep(0.1)
    target_abs = np.array([o_nom_abs[0], o_nom_abs[1], qz, 0, 0, 0])
    rtde_ctrl.moveL(target_abs.tolist(), MOVE_ACCELERATION, MOVE_ACCELERATION)
    time.sleep(0.5)
    pf_xy.clear_bores(clear_field_bores=True)
    rtde_ctrl.zeroFtSensor()
    if PRINT_STARTUP_INFO:
        print("[HRT] Reset to global minimum, bores cleared.")


def process_commands():
    """Process any pending operator commands."""
    global intent_I, last_field_bore_time
    while True:
        try:
            cmd, args = hrt_state["command_queue"].get_nowait()
        except queue.Empty:
            return

        try:
            with hrt_state["lock"]:
                if cmd in ("quit", "q", "exit"):
                    hrt_state["shutdown_requested"] = True
                    shutdown_flag[0] = True
                    return

                if cmd == "modality" or cmd == "m":
                    if args and args[0].isdigit():
                        mod = int(args[0])
                        if 1 <= mod <= 5:
                            old_mod = hrt_state["modality"]
                            hrt_state["modality"] = mod
                            if mod != old_mod:
                                # Reset field on modality switch: clear voice obstacles, all bores
                                pf_xy.clear_obstacles()
                                pf_xy.clear_bores(clear_field_bores=True)
                                for obs in pf_config["obstacles"]:
                                    pf_xy.add_obstacle(obs["x"], obs["y"], obs["height"], obs["width"])
                                intent_I = 0.0
                                last_field_bore_time = time.time()
                                print(f"[HRT] Modality: {MODALITY_NAMES[mod]} (field reset)")
                            else:
                                print(f"[HRT] Modality: {MODALITY_NAMES[mod]}")
                    continue

                if cmd in ("reset", "r"):
                    hrt_state["reset_requested"] = True
                    hrt_state["trial_active"] = False
                    hrt_state["trial_target_xy"] = None
                    continue

                if cmd == "position_bar" or cmd == "pb":
                    if not args:
                        print("[HRT] Usage: position_bar left|right")
                        continue
                    side = args[0].lower()
                    if side == "left":
                        hrt_state["position_bar_requested"] = ("left", SIT_TO_STAND_OFFSET_M)
                    elif side == "right":
                        hrt_state["position_bar_requested"] = ("right", SIT_TO_STAND_OFFSET_M)
                    else:
                        print("[HRT] Use: position_bar left|right")
                    continue

                if cmd in ("start_trial", "st"):
                    if args:
                        # Cardinal direction trial
                        direction = args[0]
                        if direction not in CARDINAL_OFFSETS:
                            print("[HRT] Invalid direction. Use: north, south, east, west")
                            continue
                        if hrt_state["modality"] == MODALITY_FIXED:
                            print("[HRT] Fixed bar: cardinal trials skipped per procedure.")
                            continue
                        target_rel = CARDINAL_OFFSETS[direction]
                        hrt_state["trial_target_xy"] = target_rel.copy()
                        hrt_state["trial_type"] = "sit_to_stand" if hrt_state["sit_to_stand_mode"] else "cardinal"
                        hrt_state["trial_direction"] = direction
                    else:
                        # Sit-to-stand trial (no target, manual end)
                        if not hrt_state.get("sit_to_stand_mode"):
                            print("[HRT] Enable sit_to_stand mode first, or use: start_trial north|south|east|west")
                            continue
                        hrt_state["trial_target_xy"] = None  # No target for sit-to-stand
                        hrt_state["trial_type"] = "sit_to_stand"
                        hrt_state["trial_direction"] = ""

                    hrt_state["trial_active"] = True
                    hrt_state["trial_start_time"] = time.time()
                    hrt_state["trial_max_force"] = 0.0
                    qx, qy, _, _, _, _ = rtde_r.getActualTCPPose()
                    hrt_state["trial_start_pos"] = np.array([qx, qy]) - o_nom
                    tgt = hrt_state["trial_target_xy"]
                    print(f"[HRT] Trial started: {hrt_state['trial_type']}" + (f", target={tgt}" if tgt is not None else " (manual end)"))

                if cmd in ("end_trial", "et"):
                    if hrt_state["trial_active"]:
                        _record_trial_metrics()
                        hrt_state["trial_active"] = False
                        hrt_state["trial_target_xy"] = None
                        print("[HRT] Trial ended, metrics recorded.")
                    else:
                        print("[HRT] No active trial.")

                if cmd in ("qualitative", "qual"):
                    if args and args[0].isdigit():
                        score = int(args[0])
                        if 1 <= score <= 7:
                            mod = hrt_state["modality"]
                            trial_type = hrt_state.get("trial_type") or "N/A"
                            hrt_metrics_writer.writerow([
                                datetime.now().isoformat(), mod, MODALITY_NAMES.get(mod, ""),
                                "qualitative", "", "", "", "", "", "", "", "", "", score,
                                hrt_state.get("sit_to_stand_mode", False)
                            ])
                            hrt_metrics_file.flush()
                            print(f"[HRT] Qualitative score {score} recorded for {MODALITY_NAMES[mod]}.")
                        else:
                            print("[HRT] Qualitative score must be 1-7.")
                    else:
                        print("[HRT] Usage: qualitative <1-7>")

                if cmd in ("sit_to_stand", "sts"):
                    hrt_state["sit_to_stand_mode"] = not hrt_state["sit_to_stand_mode"]
                    print(f"[HRT] Sit-to-stand mode: {'ON' if hrt_state['sit_to_stand_mode'] else 'OFF'}")
        except Exception as e:
            print(f"[HRT] Command error: {e}")


def _record_trial_metrics():
    """Record trial metrics to HRT CSV."""
    with hrt_state["lock"]:
        mod = hrt_state["modality"]
        trial_type = hrt_state.get("trial_type", "cardinal")
        direction = hrt_state.get("trial_direction", "")
        start_time = hrt_state.get("trial_start_time")
        max_force = hrt_state.get("trial_max_force", 0.0)
        target_xy = hrt_state.get("trial_target_xy")

    qx, qy, _, _, _, _ = rtde_r.getActualTCPPose()
    final_rel = np.array([qx, qy]) - o_nom

    time_to_target = (time.time() - start_time) if start_time else None
    target_distance = np.linalg.norm(target_xy) if target_xy is not None else None
    deviation = np.linalg.norm(final_rel - target_xy) if target_xy is not None else None

    hrt_metrics_writer.writerow([
        datetime.now().isoformat(), mod, MODALITY_NAMES.get(mod, ""),
        trial_type, direction,
        f"{time_to_target:.2f}" if time_to_target is not None else "",
        f"{target_distance:.4f}" if target_distance is not None else "",
        f"{max_force:.2f}" if max_force is not None else "",
        f"{deviation:.4f}" if deviation is not None else "",
        target_xy[0] if target_xy is not None else "",
        target_xy[1] if target_xy is not None else "",
        final_rel[0], final_rel[1],
        "",  # qualitative filled separately
        hrt_state.get("sit_to_stand_mode", False)
    ])
    hrt_metrics_file.flush()


# Start input thread
input_thread = threading.Thread(target=input_thread_fn, daemon=True)
input_thread.start()

if PRINT_STARTUP_INFO:
    print(f"[HRT] Nominal position: [{o_nom[0]:.4f}, {o_nom[1]:.4f}] m")
    print(f"[HRT] Potential field: {len(pf_xy.obstacles)} obstacles")
    print(f"[HRT] Modalities: 1=Fixed, 2=Touch, 3=Touch+LED, 4=Touch+Voice, 5=All")
    print(f"[HRT] Metrics saved to: {hrt_metrics_path}")

# Main control loop
while True:
    try:
        process_commands()

        if shutdown_flag[0]:
            raise KeyboardInterrupt

        # Handle reset
        if hrt_state.get("reset_requested"):
            hrt_state["reset_requested"] = False
            qx, qy, qz, _, _, _ = rtde_r.getActualTCPPose()
            reset_to_global_minimum(rtde_c, rtde_r, o_nom, qz)
            intent_I = 0.0
            continue

        # Handle position_bar (for sit-to-stand trials)
        pos_req = hrt_state.get("position_bar_requested")
        if pos_req is not None:
            hrt_state["position_bar_requested"] = None
            side, offset = pos_req
            rtde_c.speedStop()
            time.sleep(0.1)
            qx, qy, qz, _, _, _ = rtde_r.getActualTCPPose()
            dx = offset if side == "left" else -offset  # left = +X, right = -X
            target_abs = np.array([o_nom[0] + dx, o_nom[1], qz, 0, 0, 0])
            rtde_c.moveL(target_abs.tolist(), MOVE_ACCELERATION, MOVE_ACCELERATION)
            time.sleep(0.5)
            rtde_c.zeroFtSensor()
            print(f"[HRT] Bar positioned to {side} (offset {abs(dx)*100:.0f} cm)")
            continue

        qx, qy, qz, qrx, qry, qrz = rtde_r.getActualTCPPose()
        q_0_xy = np.array([qx, qy]) - o_nom

        mod = hrt_state["modality"]

        # LED feedback (only when modality includes LED)
        use_led = mod in (MODALITY_TOUCH_LED, MODALITY_TOUCH_VOICE_LED)
        if use_led:
            _t_now = time.monotonic()
            if _t_now >= _next_led_t:
                led_streamer.try_send(sample_ring_gradients_xy(q_0_xy, pf_xy))
                _next_led_t = _t_now + _led_period_s

        # Get and filter force
        Fx, Fy, Fz, _, _, _ = rtde_r.getActualTCPForce()
        F_inp_sensor = np.array([Fx, Fy])
        F_inp_x, zi_x = scipy_signal.sosfilt(sos_butter, [F_inp_sensor[0]], zi=zi_x)
        F_inp_y, zi_y = scipy_signal.sosfilt(sos_butter, [F_inp_sensor[1]], zi=zi_y)
        F_inp = np.array([F_inp_x[0], F_inp_y[0]])
        F_inp[np.abs(F_inp) < DEADBAND_THRESHOLD] = 0.0
        f_ext = F_inp.copy()

        # Update trial max force
        if hrt_state.get("trial_active"):
            force_mag = np.linalg.norm(f_ext)
            if force_mag > hrt_state.get("trial_max_force", 0):
                with hrt_state["lock"]:
                    hrt_state["trial_max_force"] = force_mag

        # Check if target reached (auto-end trial)
        if hrt_state.get("trial_active") and hrt_state.get("trial_target_xy") is not None:
            target = np.array(hrt_state["trial_target_xy"])
            dist_to_target = np.linalg.norm(q_0_xy - target)
            if dist_to_target <= TARGET_REACH_TOLERANCE_M:
                _record_trial_metrics()
                with hrt_state["lock"]:
                    hrt_state["trial_active"] = False
                    hrt_state["trial_target_xy"] = None
                if PRINT_STARTUP_INFO:
                    print(f"[HRT] Target reached! Deviation ~{dist_to_target*100:.1f} cm")

        # --- Fixed bar: hold position, no movement ---
        if mod == MODALITY_FIXED:
            qdot = np.zeros(6)
            rtde_c.speedL(qdot, SPEEDL_ACCELERATION, DT)
            potential_force = np.zeros(2)
            gamma = 0.0
        else:
            # --- Impedance control (touch-based modalities) ---
            potential_force = -get_potential_force_xy(q_0_xy, pf_xy)
            gradV = -potential_force
            grad_norm = float(np.linalg.norm(gradV))

            # Voice: add obstacle (only when modality includes voice)
            use_voice = mod in (MODALITY_TOUCH_VOICE, MODALITY_TOUCH_VOICE_LED)
            if use_voice:
                sentence, emb, ridge, nn = pipeline.get_latest()
                if sentence is not None:
                    if PRINT_VOICE_TRANSCRIPTION:
                        print(f"[transcription] {sentence}", flush=True)
                    keyword_ok = check_keyword(sentence, VOICE_KEYWORDS)
                    similarity_ok = check_similarity(ridge, nn, OBS_DELTA) if VOICE_REQUIRE_SIMILARITY else (nn is not None)
                    if keyword_ok and similarity_ok and nn is not None:
                        # Filter: push obstacle farther from handlebar, increase amplitude (stabler behavior)
                        offset = np.array([nn["x_m"], nn["y_m"]], dtype=float)
                        dist = float(np.linalg.norm(offset))
                        if dist < 1e-6:
                            offset = np.array([0.1, 0.0])  # default: right
                            dist = 0.1
                        direction = offset / dist
                        new_dist = max(dist * VOICE_OBSTACLE_DISTANCE_SCALE, VOICE_OBSTACLE_MIN_DISTANCE_M)
                        new_offset = direction * new_dist
                        obs_x = q_0_xy[0] + new_offset[0]
                        obs_y = q_0_xy[1] + new_offset[1]
                        amplitude = nn["amplitude"] * VOICE_OBSTACLE_AMPLITUDE_SCALE
                        radius = max(float(nn.get("radius", 0.05)), VOICE_OBSTACLE_MIN_RADIUS_M)
                        pf_xy.add_obstacle(obs_x, obs_y, amplitude, radius)
                        potential_force = -get_potential_force_xy(q_0_xy, pf_xy)
                        gradV = -potential_force
                        grad_norm = float(np.linalg.norm(gradV))
                        if PRINT_VOICE_OBSTACLE:
                            print(f"[voice] added obstacle at ({obs_x:.3f}, {obs_y:.3f}) amp={amplitude:.0f}", flush=True)
                    elif VOICE_DEBUG_SKIP and sentence:
                        why = []
                        if not keyword_ok:
                            why.append("no_keyword")
                        if nn is None:
                            why.append("nn_none")
                        elif VOICE_REQUIRE_SIMILARITY and not similarity_ok:
                            why.append("similarity_fail")
                        print(f"[voice] skip: {sentence[:40]!r} -> {why}", flush=True)
            else:
                sentence, emb, ridge, nn = pipeline.get_latest()  # Still consume to avoid backlog

            # Tunneling logic
            if grad_norm < min_grad_norm:
                gamma = 0.0
                intent_I = max(0.0, intent_I - DT * rho * intent_I)
            else:
                ghat = gradV / (grad_norm + eps_grad)
                f_up = float(np.dot(f_ext, ghat))
                if use_angle_gate:
                    f_norm = float(np.linalg.norm(f_ext))
                    angle_ok = (f_norm >= 1e-12) and (f_up / (f_norm + 1e-12) >= cos0)
                else:
                    angle_ok = True
                if (f_up > 0.0) and angle_ok:
                    u = max(0.0, f_up - s0)
                    intent_I = max(0.0, intent_I + DT * (u - rho * intent_I))
                else:
                    intent_I = max(0.0, intent_I - DT * rho * intent_I)
                gamma = sigmoid(beta_I * (intent_I - I0))

                current_time = time.time()
                time_since_last_bore = current_time - last_field_bore_time
                if gamma > FIELD_BORE_GAMMA_THRESHOLD and time_since_last_bore >= FIELD_BORE_COOLDOWN:
                    updated_existing = pf_xy.update_bore_from_force(
                        robot_x=q_0_xy[0], robot_y=q_0_xy[1],
                        force_vector=f_ext, tunneling_intent=gamma
                    )
                    potential_force = -get_potential_force_xy(q_0_xy, pf_xy)
                    gradV = -potential_force
                    grad_norm = float(np.linalg.norm(gradV))
                    if not updated_existing:
                        last_field_bore_time = current_time

            qdot = np.zeros(6)
            qdot[:2] = (potential_force + f_ext) / DAMPING
            qdot = np.clip(qdot, -SPEED_LIMIT, SPEED_LIMIT)
            rtde_c.speedL(qdot, SPEEDL_ACCELERATION, DT)

        # Busy-wait for loop timing
        for i in range(1, 12000):
            _ = i * 56.1613

        print_counter += 1
        current_time = time.time()
        if print_counter >= PRINT_INTERVAL and (current_time - last_print_time) >= 0.1:
            last_print_time = current_time
            print_counter = 0

        # Visualization
        if ENABLE_LIVE_VISUALIZATION and fig is not None:
            plot_update_counter += 1
            if plot_update_counter >= PLOT_UPDATE_INTERVAL and (current_time - last_plot_time) >= 0.1:
                ax.clear()
                ax.set_xlabel('Relative X (m)')
                ax.set_ylabel('Relative Y (m)')
                ax.set_title(f'HRT - {MODALITY_NAMES.get(mod, "?")}')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                ax.set_xlim(pf_xy.x_bounds)
                ax.set_ylim(pf_xy.y_bounds)
                Z = pf_xy.calculate_potential()
                X, Y = pf_xy.X, pf_xy.Y
                Z_min = np.min(Z)
                Z_vis = Z - Z_min
                contour = ax.contour(X, Y, Z_vis, levels=15, cmap='viridis', alpha=0.8)
                for obs in pf_xy.obstacles.values():
                    ax.add_patch(plt.Circle((obs['x'], obs['y']), obs['width'], color='red', alpha=0.5, fill=False))
                ax.plot(q_0_xy[0], q_0_xy[1], 'bo', markersize=14)
                if hrt_state.get("trial_target_xy") is not None:
                    t = hrt_state["trial_target_xy"]
                    ax.plot(t[0], t[1], 'g*', markersize=20, label='Target')
                try:
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                except Exception:
                    pass
                last_plot_time = time.time()
                plot_update_counter = 0

        # Log trajectory
        if log_config["enabled"] and csv_writer is not None:
            csv_writer.writerow([
                time.time(), qx, qy, qz, Fx, Fy, Fz,
                potential_force[0], potential_force[1],
                q_0_xy[0], q_0_xy[1],
                qdot[0], qdot[1],
                gamma if mod != MODALITY_FIXED else 0, intent_I, mod, hrt_state.get("trial_active", False)
            ])

    except KeyboardInterrupt:
        if PRINT_SHUTDOWN_MESSAGE:
            print("\nShutting down...")
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            if led_streamer is not None:
                led_streamer.close()
        except Exception:
            pass
        if log_config["enabled"] and csv_file is not None:
            csv_file.close()
            if PRINT_LOG_SAVED:
                print(f"Trajectory saved to {csv_filename}")
        hrt_metrics_file.close()
        if PRINT_LOG_SAVED:
            print(f"HRT metrics saved to {hrt_metrics_path}")
        try:
            rtde_c.speedStop()
            rtde_c.disconnect()
        except Exception:
            pass
        if ENABLE_LIVE_VISUALIZATION and fig is not None:
            plt.close(fig)
        break
