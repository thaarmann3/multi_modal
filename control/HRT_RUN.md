# Running HRT Control

Quick guide to run `HRT_control.py` with `HRT_control.json`.

## Prerequisites

- **UR robot** powered on and reachable at the IP in config
- **Python 3.8+** with dependencies installed
- *(Optional)* Arduino LED ring on serial port (modalities 3, 5)
- *(Optional)* Microphone for voice modalities (4, 5)

## Dependencies

```bash
pip install ur_rtde numpy scipy matplotlib pyserial sentence-transformers
pip install -r stt/requirements.txt   # vosk, sounddevice
```

## Run

From the **project root** (`darb/`):

```bash
python control/HRT_control.py
```

The script loads `configs/HRT_control.json` automatically (path is relative to project root).

## Config

Edit `configs/HRT_control.json` before running:


| Key                     | Purpose                                                    |
| ----------------------- | ---------------------------------------------------------- |
| `robot.ip`              | UR robot IP (default `169.254.9.43`)                       |
| `leds.port`             | Arduino serial port (e.g. `/dev/cu.usbmodem1101` on macOS) |
| `visualization.enabled` | Set `true` for live potential-field plot                   |
| `logging.enabled`       | Set `true` to log trajectories and metrics                 |


## Operator Commands

Type in the terminal and press Enter:


| Command               | Description                                                           |
| --------------------- | --------------------------------------------------------------------- |
| `modality <1-5>`      | Switch modality (1=fixed, 2=touch, 3=touch+LED, 4=touch+voice, 5=all) |
| `reset`               | Move handlebar to global minimum, clear bores                         |
| start_trial south     | Starts trial to move handlebar a set distance in that direction       |
| `end_trial` or `stop` | End trial and record metrics                                          |
| `lock` / `unlock`     | Freeze robot or resume                                                |
| `quit`                | Shutdown                                                              |


## Output

- **Metrics**: `hrt_metrics/hrt_trial_YYYYMMDD_HHMMSS.csv`
- **Trajectories**: `trajectories/` (if `logging.enabled` is true)

## Testing Procedure

### Metrics to Measure


| Metric                   | Description                                                                                                      | Plot                           |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| Time to target           | Time to move handlebar from initial position to goal, or time to perform sit-to-stand movement (phone stopwatch) | Bar plot (a)                   |
| Maximum force            | Maximum force applied to the handlebar                                                                           | Plot (b), colored bar series 1 |
| Total distance traveled  | Total distance traveled over trial                                                                               | —                              |
| Deviation from target    | Deviation between final handlebar position and target position (e.g. "15 cm to the right")                       | Plot (b), colored bar series 2 |
| Qualitative scores (1–7) | Ease of use, understandability, and overall preference per modality                                              | Plot (c), heatmap grid         |
| Sit-to-stand impulse     | Force integrated over time during sit-to-stand                                                                   | —                              |


**Modalities for qualitative comparison:**

1. Fixed bar (non-adjustable)
2. Touch
3. Touch + LED ring
4. Touch + voice
5. Touch + voice + LED ring

### Procedure

1. **Setup**
  - Ensure robot has a bowl-shaped potential field with appropriate control gains.
  - Field should have 2 obstacles and a clear global minimum. Same for each user.
  - Explain to the user how each modality works.
  - Place direction sticky note and 15 cm line on the table; ensure user is aware of them.
2. **Cardinal trials (12 trials total: 4 modalities × 3 trials each)**
  - For each modality (2–5): Touch, Touch+LED, Touch+voice, Touch+voice+LED:
    - Give user time to practice moving the handlebar around.
    - Return handlebar to global minimum (`reset`).
    - We are testing using the "south" direction. At the start of each trial, reset to global minimum.
    - Example: `reset` → `start_trial south` → `stop` → `reset` → `start_trial south` → `stop` etc.
    - Record quantitative metrics for each trial.
  - *Skip fixed bar for cardinal trials.*
3. **Sit-to-stand trials**
  - Have user sit down on the stool. (Make sure the stool is within the tape).
  - Return the handlebar to the global minimum.
  - For each modality (1–5):
    - Give user 10 seconds to position the handlebar (strict time limit) then tell them to remove their hands.
    - Lock the handlebar, enter `lock`.
    - Ask user to perform sit-to-stand transition.
    - Enter `unlock` to finish data recording.
    - Record time to complete movement on phone.
    - Reset handlebar to original position `reset`.
    - After sit to stand, record qualitative metrics for the modality and then move on.

