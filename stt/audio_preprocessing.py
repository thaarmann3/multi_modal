# Audio preprocessing classes: AGC, filtering, etc. Apply in sequence before Vosk.
import numpy as np


class AGC:
    """Automatic Gain Control: normalize audio level to target RMS."""
    def __init__(self, target_rms: float = 0.3, attack_rate: float = 0.1, release_rate: float = 0.01):
        self.target_rms = target_rms  # Target RMS level (0.0-1.0)
        self.attack_rate = attack_rate  # How fast to increase gain
        self.release_rate = release_rate  # How fast to decrease gain
        self.current_gain = 1.0  # Current gain multiplier

    def process(self, audio_bytes: bytes) -> bytes:
        # Convert bytes to int16 array, compute RMS, adjust gain, convert back.
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio ** 2)) / 32768.0  # Normalize to 0-1 range
        
        if rms > 0:
            desired_gain = self.target_rms / rms
            # Smooth gain changes: attack (increase) faster, release (decrease) slower.
            if desired_gain > self.current_gain:
                self.current_gain += (desired_gain - self.current_gain) * self.attack_rate
            else:
                self.current_gain += (desired_gain - self.current_gain) * self.release_rate
            # Clamp gain to reasonable range (0.1x to 10x).
            self.current_gain = np.clip(self.current_gain, 0.1, 10.0)
            audio = audio * self.current_gain
        
        # Convert back to int16 and bytes.
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        return audio.tobytes()
