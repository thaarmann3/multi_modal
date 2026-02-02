#!/usr/bin/env python3
"""
Test Voice Control
==================

Simple test script that listens for voice commands and prints them out.
"""

import time
import signal
import queue
from typing import Optional, Tuple, List
from voice_processing import (
    VoiceProcessingPipeline,
    SpeechUtteranceFinal,
    StopLatch
)


def split_into_commands(text: str) -> List[str]:
    """
    Split text into multiple command segments.
    
    Splits on conjunctions like "then", "and", "also", etc.
    
    Args:
        text: Full transcript text
    
    Returns:
        List of command segments
    """
    import re
    
    # Conjunctions that separate commands (case insensitive)
    separators = [r"\s+then\s+", r"\s+and\s+then\s+", r"\s+also\s+", 
                  r"\s+and\s+", r"\s+next\s+", r"\s+after\s+that\s+"]
    
    # Split on separators (case insensitive)
    segments = [text]
    
    for sep_pattern in separators:
        new_segments = []
        for segment in segments:
            # Split using regex (case insensitive)
            parts = re.split(sep_pattern, segment, flags=re.IGNORECASE)
            new_segments.extend(parts)
        segments = new_segments
    
    # Clean up segments (strip whitespace, remove empty)
    cleaned = [s.strip() for s in segments if s.strip()]
    
    return cleaned if cleaned else [text]


def parse_single_command(text: str) -> Tuple[Optional[str], bool]:
    """
    Parse a single voice command and return action message and shutdown flag.
    
    Args:
        text: Single command text
    
    Returns:
        Tuple of (action message string or None, shutdown flag)
    """
    text_lower = text.lower()
    
    # Shutdown commands
    if any(word in text_lower for word in ["stop", "freeze", "kill"]):
        return "Shutting down", True
    
    # Movement commands
    if any(word in text_lower for word in ["move", "go"]) and "left" in text_lower:
        return "Moving left", False
    elif any(word in text_lower for word in ["move", "go"]) and "right" in text_lower:
        return "Moving right", False
    elif any(word in text_lower for word in ["move", "go"]) and ("forward" in text_lower or "ahead" in text_lower):
        return "Moving forward", False
    elif any(word in text_lower for word in ["move", "go"]) and ("back" in text_lower or "backward" in text_lower):
        return "Moving backward", False
    elif any(word in text_lower for word in ["move", "go"]) and "up" in text_lower:
        return "Moving up", False
    elif any(word in text_lower for word in ["move", "go"]) and "down" in text_lower:
        return "Moving down", False
    
    # No recognized command
    return None, False


def parse_commands(text: str) -> List[Tuple[str, bool]]:
    """
    Parse multiple commands from a single phrase.
    
    Args:
        text: Full transcript text that may contain multiple commands
    
    Returns:
        List of tuples (command_message, shutdown_flag) for each recognized command
    """
    # Split into command segments
    segments = split_into_commands(text)
    
    # Parse each segment
    commands = []
    for segment in segments:
        command, should_shutdown = parse_single_command(segment)
        if command:
            commands.append((command, should_shutdown))
    
    return commands


def on_speech_event(event, command_queue: queue.Queue, shutdown_flag: list, verbose: bool = True):
    """Handle speech event - parse multiple commands and add to queue."""
    # Extract text (handle both SpeechEvent and SpeechUtteranceFinal)
    if hasattr(event, 'transcript_normalized'):
        text = event.transcript_normalized
    elif hasattr(event, 'text'):
        text = event.text.lower()
    else:
        text = str(event).lower()
    
    # Always show what was heard if verbose
    if verbose:
        print(f"🎤 Heard: '{text}'")
    
    # Parse multiple commands from the phrase
    commands = parse_commands(text)
    
    if commands:
        # Add all commands to the queue (FIFO)
        for command, should_shutdown in commands:
            command_queue.put((command, should_shutdown))
        
        if verbose:
            print(f"   → Queued {len(commands)} command(s)")
    elif verbose:
        print(f"   (No command recognized)")


def process_command(command: str, shutdown_flag: list) -> bool:
    """
    Process a single command.
    
    Args:
        command: Command message to process
        shutdown_flag: Shutdown flag to set if needed
    
    Returns:
        True if command was processed successfully, False otherwise
    """
    print(f"▶ Executing: {command}")
    
    # Simulate command execution (in real system, this would control the robot)
    # For now, just print and return immediately
    # In production, this would wait for the command to complete
    
    return True


def on_stop_latch(event: StopLatch):
    """Handle safety event."""
    print(f"⚠ STOP LATCHED: {event.trigger_source}")


def main():
    """Main test loop."""
    print("=" * 60)
    print("Voice Control Test")
    print("=" * 60)
    print("Listening for voice commands...")
    print("Commands are parsed after a 0.5 second pause")
    print("Commands are processed in FIFO order (first in, first out)")
    print("")
    print("TROUBLESHOOTING:")
    print("- Speak clearly and at normal volume")
    print("- Wait for '[VAD] Utterance detected' message")
    print("- Check '[ASR]' messages for decoding status")
    print("- If 'Could not understand audio', try speaking louder")
    print("")
    print("LOCAL PROCESSING:")
    print("- All ASR processing is done locally (no internet required)")
    print("- Using Whisper for fast, reliable transcription")
    print("- Install: pip install faster-whisper (recommended, fastest)")
    print("- Or: pip install openai-whisper")
    print("")
    print("Say commands like: 'move robot left', 'go right', 'stop'")
    print("Or multiple: 'move left then go right then move forward'")
    print("Press Ctrl+C to exit")
    print("=" * 60)
    
    # Initialize pipeline
    # Model sizes: "tiny" (fastest), "base" (default), "small", "medium", "large" (most accurate)
    # Device: "cpu" or "cuda" (if you have GPU and PyTorch CUDA installed)
    
    model_size = "base"  # Change to "small" or "medium" for better accuracy
    device = "cpu"  # Change to "cuda" if you have GPU
    
    pipeline = VoiceProcessingPipeline(model_size=model_size, device=device)
    
    # VAD is now improved with adaptive thresholding - defaults should work well
    # But we can still tune if needed:
    # pipeline.vad_segmenter.vad_threshold = 0.005  # Even more sensitive if needed
    
    # Quality gate is already relaxed with new defaults, but can adjust:
    # pipeline.quality_gate.min_vad_ratio = 0.03  # Even more lenient if needed
    # pipeline.quality_gate.min_duration_ms = 50   # Catch very short commands
    
    # Command queue (FIFO)
    command_queue: queue.Queue = queue.Queue()
    currently_processing = False
    
    # Debug counters
    frame_count = 0
    utterance_count = 0
    transcript_count = 0
    event_count = 0
    last_debug_time = time.time()
    
    # Handle shutdown gracefully
    shutdown = [False]  # Use list for mutable reference in closure
    
    def signal_handler(sig, frame):
        shutdown[0] = True
        print("\nShutting down...")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Subscribe to events (with shutdown flag and command queue)
    verbose = True  # Set to False to only show recognized commands
    debug = True  # Set to True to see pipeline diagnostics
    
    def speech_handler(event: SpeechUtteranceFinal):
        nonlocal event_count
        event_count += 1
        on_speech_event(event, command_queue, shutdown, verbose)
    
    pipeline.event_publisher.subscribe_speech(speech_handler)
    pipeline.event_publisher.subscribe_safety(on_stop_latch)
    
    # Start pipeline
    print("Starting pipeline...")
    pipeline.start()
    print("Pipeline started. Listening...")
    
    # Main loop - process pipeline at ~100 Hz
    try:
        while not shutdown[0]:
            # Process voice pipeline
            pipeline.process()
            
            # Process command queue (FIFO) - one command at a time
            if not currently_processing:
                try:
                    command, should_shutdown = command_queue.get_nowait()
                    
                    # Mark as processing
                    currently_processing = True
                    
                    # Process the command
                    process_command(command, shutdown)
                    
                    # Mark as done processing
                    currently_processing = False
                    
                    # Check shutdown flag
                    if should_shutdown:
                        shutdown[0] = True
                        break
                    
                except queue.Empty:
                    # No commands in queue
                    pass
            
            # Debug output every 5 seconds
            if debug:
                current_time = time.time()
                if current_time - last_debug_time >= 5.0:
                    frames_received = len(pipeline.audio_capture.frame_buffer)
                    pending_utterances = len(pipeline.pending_utterances)
                    queue_size = command_queue.qsize()
                    
                    print(f"\n[DEBUG] Frames in buffer: {frames_received}, "
                          f"Pending utterances: {pending_utterances}, "
                          f"Events published: {event_count}, "
                          f"Commands in queue: {queue_size}")
                    last_debug_time = current_time
            
            time.sleep(0.01)  # ~100 Hz
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nError in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
