# Mic -> Vosk -> enqueue events. A separate thread calls subscribers so STT never blocks.
import json
import queue
import sys
import threading
import time
from typing import Callable, List

import sounddevice as sd

from .config import STTConfig
from .events import TranscriptEvent
from .audio_preprocessing import AGC


def _load_vosk():
    from vosk import KaldiRecognizer, Model
    return Model, KaldiRecognizer


class STTPublisher:
    def __init__(self, config: STTConfig | None = None):
        self.config = config or STTConfig()
        self._subscribers: List[Callable[[TranscriptEvent], None]] = []
        self._subscribers_lock = threading.Lock()
        self._audio_queue: queue.Queue[bytes] = queue.Queue()  # mic -> process loop
        self._event_queue: queue.Queue[TranscriptEvent] = queue.Queue()  # process loop -> notify thread
        self._running = False
        self._thread: threading.Thread | None = None  # capture + Vosk
        self._notify_thread: threading.Thread | None = None
        self._model = None
        self._recognizer = None
        self._agc = AGC() if self.config.enable_agc else None

    def subscribe(self, callback: Callable[[TranscriptEvent], None]) -> None:
        with self._subscribers_lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[TranscriptEvent], None]) -> None:
        with self._subscribers_lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _notify(self, event: TranscriptEvent) -> None:
        # Enqueue only; never blocks the STT loop.
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            pass

    def _notify_loop(self) -> None:
        # Runs in its own thread: pull events and call each subscriber.
        while self._running or not self._event_queue.empty():
            try:
                event = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._subscribers_lock:
                callbacks = list(self._subscribers)
            for cb in callbacks:
                try:
                    cb(event)
                except Exception:
                    pass

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        # Called by sounddevice from its thread; push raw bytes into queue.
        if status:
            sys.stderr.write(str(status) + "\n")
        try:
            self._audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def _process_loop(self) -> None:
        # Pull audio, feed Vosk, enqueue partial/final text. Runs in capture thread.
        while self._running:
            try:
                data = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # Apply AGC if enabled.
            if self._agc is not None:
                data = self._agc.process(data)
            if self._recognizer.AcceptWaveform(data):
                obj = json.loads(self._recognizer.Result())
                text = (obj.get("text") or "").strip()
                if text:
                    self._notify(TranscriptEvent(text=text, is_final=True, timestamp=time.time()))
            else:
                obj = json.loads(self._recognizer.PartialResult())
                text = (obj.get("partial") or "").strip()
                if text:
                    self._notify(TranscriptEvent(text=text, is_final=False, timestamp=time.time()))

    def _ensure_model(self) -> None:
        if self._recognizer is not None:
            return
        Model, KaldiRecognizer = _load_vosk()
        self._model = Model(str(self.config.model_path))
        self._recognizer = KaldiRecognizer(self._model, self.config.sample_rate)

    def _start_notify_thread(self) -> None:
        if self._notify_thread is not None:
            return
        self._notify_thread = threading.Thread(target=self._notify_loop, daemon=True)
        self._notify_thread.start()

    def _stop_notify_thread(self) -> None:
        self._running = False
        if self._notify_thread is not None:
            self._notify_thread.join(timeout=2.0)
            self._notify_thread = None

    def start(self) -> None:
        # Blocking: runs capture + recognition in this thread until stop().
        if self._running:
            return
        self._running = True
        self._start_notify_thread()
        self._ensure_model()
        block_frames = int(self.config.sample_rate * self.config.block_ms / 1000)
        stream = sd.RawInputStream(
            samplerate=self.config.sample_rate,
            blocksize=block_frames,
            device=self.config.device,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        )
        stream.start()
        try:
            self._process_loop()
        finally:
            stream.stop()
            stream.close()
            self._stop_notify_thread()

    def start_background(self) -> None:
        # Returns immediately; capture + Vosk run in a daemon thread.
        if self._running or self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_background, daemon=True)
        self._thread.start()

    def _run_background(self) -> None:
        # Same as start() but in a thread; used by start_background().
        self._start_notify_thread()
        self._ensure_model()
        block_frames = int(self.config.sample_rate * self.config.block_ms / 1000)
        stream = sd.RawInputStream(
            samplerate=self.config.sample_rate,
            blocksize=block_frames,
            device=self.config.device,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        )
        stream.start()
        try:
            self._process_loop()
        finally:
            stream.stop()
            stream.close()
            self._running = False
            self._thread = None
            self._stop_notify_thread()

    def stop(self) -> None:
        self._running = False
