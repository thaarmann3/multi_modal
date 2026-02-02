#!/usr/bin/env python3
"""
Voice Processing Pipeline - Continuous Speech-to-Text
=====================================================

Fast, reliable, continuous speech-to-text using Whisper.
Optimized for real-time transcription with minimal latency.
"""

import numpy as np
import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Callable, List
from collections import deque

import pyaudio

# Try faster-whisper first (much faster), fallback to OpenAI Whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    try:
        import whisper
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False
        whisper = None


@dataclass
class SpeechEvent:
    """Speech transcription event."""
    text: str
    timestamp: float
    duration: float


@dataclass
class SpeechUtteranceFinal:
    """Speech event for compatibility."""
    seq_id: int
    t_capture_start: float
    t_capture_end: float
    utterance_ms: int
    transcript_raw: str
    transcript_normalized: str
    vad_speech_ratio: float = 1.0
    session_id: Optional[str] = None
    is_final: bool = True
    language: Optional[str] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    snr_estimate: Optional[float] = None
    dropped_frames: int = 0


@dataclass
class StopLatch:
    """Safety event."""
    timestamp: float
    trigger_source: str


class ContinuousVoiceListener:
    """
    Continuous voice listener optimized for fast, reliable transcription.
    
    Uses Whisper (faster-whisper if available) for real-time speech-to-text.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_size_ms: int = 25,
                 pause_threshold: float = 0.5,
                 model_size: str = "base",
                 device: str = "cpu",
                 compute_type: str = "int8"):
        """
        Initialize voice listener.
        
        Args:
            sample_rate: Audio sample rate (Hz) - 16000 for Whisper
            frame_size_ms: Frame size in milliseconds
            pause_threshold: Seconds of silence before processing speech
            model_size: Model size ("tiny", "base", "small", "medium", "large")
            device: "cpu" or "cuda"
            compute_type: "int8" (fastest), "int8_float16", "float16", "float32"
        """
        self.sample_rate = sample_rate
        self.frame_size_ms = frame_size_ms
        self.frame_size_samples = int(sample_rate * frame_size_ms / 1000)
        self.pause_threshold = pause_threshold
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        
        # Audio buffers
        self.current_speech: List[np.ndarray] = []
        self.last_speech_time: Optional[float] = None
        
        # State
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        
        # Audio stream
        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self.audio_stream: Optional[pyaudio.Stream] = None
        
        # ASR model
        self.model = None
        self.use_faster_whisper = False
        
        # Event subscribers
        self.subscribers: List[Callable[[SpeechEvent], None]] = []
    
    def subscribe(self, callback: Callable[[SpeechEvent], None]):
        """Subscribe to speech events."""
        self.subscribers.append(callback)
    
    def _publish_event(self, event: SpeechEvent):
        """Publish event to all subscribers."""
        for callback in self.subscribers:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in subscriber: {e}")
    
    def start(self):
        """Start listening."""
        if self.running:
            return
        
        # Load model
        if FASTER_WHISPER_AVAILABLE:
            print(f"[Voice] Loading faster-whisper model '{self.model_size}' ({self.device}, {self.compute_type})...")
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
                self.use_faster_whisper = True
                print(f"[Voice] ✓ faster-whisper model loaded")
            except Exception as e:
                print(f"[Voice] ✗ Failed to load faster-whisper: {e}")
                print(f"[Voice] Falling back to OpenAI Whisper...")
                self._load_openai_whisper()
        elif WHISPER_AVAILABLE:
            self._load_openai_whisper()
        else:
            print(f"[Voice] ✗ No Whisper available!")
            print(f"[Voice] Install: pip install faster-whisper")
            print(f"[Voice] Or: pip install openai-whisper")
            return
        
        if self.model is None:
            return
        
        self.running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        print(f"[Voice] ✓ Started listening (pause threshold: {self.pause_threshold}s)")
    
    def _load_openai_whisper(self):
        """Load OpenAI Whisper model."""
        print(f"[Voice] Loading OpenAI Whisper model '{self.model_size}'...")
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.use_faster_whisper = False
            print(f"[Voice] ✓ OpenAI Whisper model loaded")
        except Exception as e:
            print(f"[Voice] ✗ Failed to load Whisper: {e}")
    
    def stop(self):
        """Stop listening."""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
        
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except:
                pass
    
    def _capture_loop(self):
        """Audio capture loop."""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                frames_per_buffer=self.frame_size_samples,
                input=True,
                start=False
            )
            self.audio_stream.start_stream()
            
            while self.running:
                audio_data = self.audio_stream.read(
                    self.frame_size_samples,
                    exception_on_overflow=False
                )
                
                # Convert to float32
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Simple speech detection: check energy
                energy = np.sqrt(np.mean(audio_array ** 2))
                is_speech = energy > 0.01
                
                if is_speech:
                    self.current_speech.append(audio_array)
                    self.last_speech_time = time.monotonic()
                else:
                    # Check if we should process accumulated speech
                    if self.current_speech and self.last_speech_time:
                        silence_duration = time.monotonic() - self.last_speech_time
                        if silence_duration >= self.pause_threshold:
                            # Process speech in background thread
                            speech_audio = np.concatenate(self.current_speech)
                            threading.Thread(
                                target=self._process_speech,
                                args=(speech_audio,),
                                daemon=True
                            ).start()
                            self.current_speech = []
                            self.last_speech_time = None
        
        except Exception as e:
            print(f"[Voice] Capture error: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
        finally:
            if self.audio_stream:
                try:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except:
                    pass
            if self.pyaudio_instance:
                try:
                    self.pyaudio_instance.terminate()
                except:
                    pass
    
    def _process_speech(self, audio: np.ndarray):
        """Process speech audio and transcribe."""
        if len(audio) == 0 or self.model is None:
            return
        
        duration = len(audio) / self.sample_rate
        
        # Skip if too short
        if duration < 0.1:
            return
        
        start_time = time.time()
        max_amp = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))
        
        print(f"[Voice] Processing speech ({duration:.2f}s, RMS={rms:.3f})...")
        
        try:
            if self.use_faster_whisper:
                # Use faster-whisper (much faster)
                segments, info = self.model.transcribe(
                    audio,
                    language="en",
                    beam_size=1,  # Faster decoding
                    vad_filter=True,  # Use built-in VAD
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Collect all segments
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text.strip())
                
                text = " ".join(text_parts).strip()
                
            else:
                # Use OpenAI Whisper
                result = self.model.transcribe(
                    audio,
                    language="en",
                    fp16=(self.device == "cuda"),
                    verbose=False,
                    condition_on_previous_text=False,
                    initial_prompt=None
                )
                text = result["text"].strip()
            
            processing_time = time.time() - start_time
            
            if text:
                print(f"[Voice] ✓ Transcribed ({processing_time:.2f}s): '{text}'")
                
                # Create and publish event
                event = SpeechEvent(
                    text=text,
                    timestamp=time.time(),
                    duration=duration
                )
                self._publish_event(event)
            else:
                print(f"[Voice] No text detected")
        
        except Exception as e:
            print(f"[Voice] Transcription error: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN PIPELINE (for compatibility)
# ============================================================================

class VoiceProcessingPipeline:
    """
    Main pipeline wrapper for compatibility.
    """
    
    def __init__(self, config: Optional[dict] = None, model_size: str = "base", device: str = "cpu"):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration dictionary (optional)
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: "cpu" or "cuda" (if GPU available)
        """
        self.listener = ContinuousVoiceListener(model_size=model_size, device=device)
        self.event_publisher = self  # Self-reference for compatibility
    
    def start(self):
        """Start pipeline."""
        self.listener.start()
    
    def stop(self):
        """Stop pipeline."""
        self.listener.stop()
    
    def process(self):
        """Process one iteration (non-blocking, processing is async)."""
        pass
    
    def subscribe_speech(self, callback: Callable):
        """Subscribe to speech events."""
        def wrapper(event: SpeechEvent):
            # Convert SpeechEvent to SpeechUtteranceFinal for compatibility
            utterance_event = SpeechUtteranceFinal(
                seq_id=0,
                t_capture_start=event.timestamp - event.duration,
                t_capture_end=event.timestamp,
                utterance_ms=int(event.duration * 1000),
                transcript_raw=event.text,
                transcript_normalized=event.text.lower(),
                vad_speech_ratio=1.0
            )
            callback(utterance_event)
        
        self.listener.subscribe(wrapper)
    
    def subscribe_safety(self, callback: Callable):
        """Subscribe to safety events (placeholder)."""
        pass
    
    # Compatibility attributes
    @property
    def audio_capture(self):
        return self
    
    @property
    def vad_segmenter(self):
        return self
    
    @property
    def asr_decoder(self):
        return self
    
    @property
    def quality_gate(self):
        return self
    
    @property
    def pending_utterances(self):
        return {}
    
    @property
    def frame_buffer(self):
        return deque()
