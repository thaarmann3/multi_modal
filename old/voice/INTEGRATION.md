# Voice Processing Pipeline Integration Guide

This document describes how to integrate the voice processing pipeline with the control loop, neural network, and potential field systems.

## Overview

The voice processing pipeline publishes events asynchronously that can be consumed by downstream systems. The key principle is that **speech never directly modulates velocity** - it updates parameters/state that are then applied by the control loop.

## Integration Points

### 1. Control Loop Integration

The control loop runs at 100 Hz and must never be blocked by voice processing.

#### Pattern: Non-blocking Event Consumption

```python
from voice.voice_processing import VoiceProcessingPipeline, SpeechUtteranceFinal, StopLatch

# Initialize pipeline
pipeline = VoiceProcessingPipeline()
pipeline.start()

# State that voice events can modify
voice_state = {
    'stop_latched': False,
    'parameters': {}
}

# Subscribe to events
def on_speech_event(event: SpeechUtteranceFinal):
    """Handle speech event - update state only, never block."""
    # Parse event and update state
    # Example: extract command and update parameters
    voice_state['parameters'] = parse_command(event.transcript_normalized)

def on_stop_latch(event: StopLatch):
    """Handle safety event - immediately latch stop."""
    voice_state['stop_latched'] = True

pipeline.event_publisher.subscribe_speech(on_speech_event)
pipeline.event_publisher.subscribe_safety(on_stop_latch)

# In control loop (100 Hz)
while True:
    # Process voice pipeline (non-blocking)
    pipeline.process()
    
    # Check stop latch
    if voice_state['stop_latched']:
        rtde_c.speedStop()
        voice_state['stop_latched'] = False  # Reset after handling
    
    # Apply voice parameters to potential field
    if voice_state['parameters']:
        update_potential_field(voice_state['parameters'])
    
    # Main control loop logic
    # ... (existing control code)
    
    time.sleep(0.01)  # 100 Hz
```

#### Key Points:
- `pipeline.process()` is called every control loop iteration
- Event handlers update state only, never block
- StopLatch immediately suppresses motion
- Parameters are applied at control loop rate (100 Hz)

### 2. Neural Network Integration

The neural network subscribes to speech events and processes normalized transcripts.

#### Pattern: Async Event Subscription

```python
from voice.voice_processing import VoiceProcessingPipeline, SpeechUtteranceFinal
import queue

# Initialize pipeline
pipeline = VoiceProcessingPipeline()
pipeline.start()

# Queue for NN processing
nn_input_queue = queue.Queue()

def on_speech_event(event: SpeechUtteranceFinal):
    """Forward speech events to NN input queue."""
    # Only forward events that passed quality gates
    nn_input_queue.put(event)

pipeline.event_publisher.subscribe_speech(on_speech_event)

# NN processing thread (separate from control loop)
def nn_worker():
    while True:
        try:
            event = nn_input_queue.get(timeout=0.1)
            # Process normalized transcript
            nn_output = process_with_nn(event.transcript_normalized)
            # Update potential field parameters
            update_potential_field_from_nn(nn_output)
        except queue.Empty:
            continue

nn_thread = threading.Thread(target=nn_worker, daemon=True)
nn_thread.start()
```

#### Key Points:
- NN subscribes to Speech Event Stream
- Events consumed asynchronously in separate thread
- NN processes `transcript_normalized` field
- NN outputs influence potential field parameters

### 3. Potential Field Integration

Potential field parameters are updated from NN outputs or direct voice commands.

#### Pattern: Parameter Updates

```python
from voice.voice_processing import VoiceProcessingPipeline, SpeechUtteranceFinal
from fields.potential_field_discrete import PotentialField

# Initialize systems
pipeline = VoiceProcessingPipeline()
pipeline.start()
pf = PotentialField(...)

# Parameter update function
def update_potential_field_from_voice(event: SpeechUtteranceFinal):
    """Update potential field based on voice command."""
    text = event.transcript_normalized.lower()
    
    # Parse commands (example)
    if "add obstacle" in text:
        # Extract position from text
        x, y = parse_position(text)
        pf.add_obstacle(x, y, height=40.0, width=0.1)
    elif "remove obstacle" in text:
        x, y = parse_position(text)
        pf.remove_obstacle(x, y)
    elif "increase strength" in text:
        pf.scaling_factor *= 1.1
    # ... more commands

pipeline.event_publisher.subscribe_speech(update_potential_field_from_voice)

# In control loop
while True:
    pipeline.process()
    
    # Get potential field forces (uses updated parameters)
    potential_force = -pf.get_potential_force_xy(q_0_xy)
    
    # Apply to control
    qdot = compute_velocity(potential_force, ...)
    rtde_c.speedL(qdot, 0.3, DT)
    
    time.sleep(0.01)
```

#### Key Points:
- Potential field parameters updated from voice events
- Updates applied at control loop rate (100 Hz)
- No direct velocity modulation from speech
- Speech influences potential field, which influences control

### 4. Complete Integration Example

Here's a complete example integrating all systems:

```python
#!/usr/bin/env python3
"""
Complete integration example: Voice + Control Loop + Potential Field
"""

import time
import threading
import queue
from voice.voice_processing import (
    VoiceProcessingPipeline, 
    SpeechUtteranceFinal, 
    StopLatch
)
from fields.potential_field_discrete import PotentialField
# ... other imports

class IntegratedSystem:
    def __init__(self):
        # Initialize voice pipeline
        self.pipeline = VoiceProcessingPipeline()
        self.pipeline.start()
        
        # Initialize potential field
        self.pf = PotentialField(...)
        
        # State
        self.stop_latched = False
        self.voice_parameters = {}
        
        # Setup event handlers
        self.pipeline.event_publisher.subscribe_speech(self.on_speech)
        self.pipeline.event_publisher.subscribe_safety(self.on_stop_latch)
    
    def on_speech(self, event: SpeechUtteranceFinal):
        """Handle speech event - update parameters."""
        text = event.transcript_normalized.lower()
        
        # Parse and update parameters
        if "move obstacle" in text:
            # Extract direction and distance
            direction, distance = parse_movement(text)
            self.voice_parameters['obstacle_move'] = (direction, distance)
        elif "adjust field" in text:
            # Extract adjustment parameters
            params = parse_field_adjustment(text)
            self.voice_parameters['field_adjust'] = params
    
    def on_stop_latch(self, event: StopLatch):
        """Handle safety event."""
        self.stop_latched = True
    
    def update_potential_field(self):
        """Apply voice parameters to potential field."""
        if 'obstacle_move' in self.voice_parameters:
            direction, distance = self.voice_parameters.pop('obstacle_move')
            # Move obstacle
            # ...
        
        if 'field_adjust' in self.voice_parameters:
            params = self.voice_parameters.pop('field_adjust')
            # Adjust field
            # ...
    
    def control_loop(self):
        """Main control loop (100 Hz)."""
        while True:
            # Process voice pipeline (non-blocking)
            self.pipeline.process()
            
            # Check stop latch
            if self.stop_latched:
                rtde_c.speedStop()
                self.stop_latched = False
                continue
            
            # Update potential field from voice
            self.update_potential_field()
            
            # Get current pose
            qx, qy, qz, qrx, qry, qrz = rtde_r.getActualTCPPose()
            q_0_xy = np.array([qx, qy]) - o_nom
            
            # Get potential field forces
            potential_force = -self.pf.get_potential_force_xy(q_0_xy)
            
            # Compute velocity
            qdot = compute_velocity(potential_force, ...)
            
            # Apply control
            rtde_c.speedL(qdot, 0.3, DT)
            
            time.sleep(0.01)  # 100 Hz

# Run system
if __name__ == "__main__":
    system = IntegratedSystem()
    system.control_loop()
```

## Event Flow Diagram

```
┌─────────────────┐
│  Microphone     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Audio Capture   │ (Stage A)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ VAD Segmenter   │ (Stage B)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ASR Decoder     │ (Stage C)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Normalizer │ (Stage D)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Event Publisher │ (Stage E)
└────────┬────────┘
         │
         ├─────────────────┬─────────────────┐
         ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Control Loop │  │ Neural Net   │  │ Potential    │
│ (100 Hz)     │  │ (Async)      │  │ Field        │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Safety Considerations

### StopLatch Handling

StopLatch events must be handled immediately:

```python
def on_stop_latch(event: StopLatch):
    """Immediate motion suppression."""
    # Immediately stop robot
    rtde_c.speedStop()
    
    # Latch the stop state
    stop_state['latched'] = True
    stop_state['timestamp'] = event.timestamp
    
    # Log for debugging
    print(f"STOP LATCHED: {event.trigger_source} at {event.timestamp}")

# In control loop
if stop_state['latched']:
    # Keep stopped until explicitly released
    rtde_c.speedStop()
    return  # Skip control computation
```

### Quality Gating

Only high-quality events should influence control:

```python
# Quality gate is applied automatically by pipeline
# But you can add additional checks:

def on_speech_event(event: SpeechUtteranceFinal):
    # Additional quality checks
    if event.utterance_ms < 500:  # Too short
        return
    
    if event.vad_speech_ratio < 0.3:  # Too much noise
        return
    
    # Process event
    process_command(event.transcript_normalized)
```

## Performance Considerations

1. **Non-blocking**: All voice processing must be non-blocking
2. **Thread isolation**: Heavy compute (ASR) runs in separate thread/process
3. **Queue sizes**: Limit queue sizes to prevent memory growth
4. **Rate limiting**: Control how often events are processed
5. **Latency**: Acceptable latency is 300-800 ms for end-of-speech + ASR decode

## Testing Integration

Test integration points:

1. **Unit tests**: Test event handlers independently
2. **Integration tests**: Test full pipeline with mock control loop
3. **Latency tests**: Verify control loop timing is unaffected
4. **Safety tests**: Verify StopLatch behavior
5. **Quality tests**: Verify only good events influence control

## Troubleshooting

### Control loop is slow
- Check that `pipeline.process()` is non-blocking
- Verify ASR decoder runs in separate thread
- Check queue sizes aren't growing unbounded

### Events not received
- Verify subscribers are registered before pipeline starts
- Check event publisher is initialized correctly
- Verify quality gates aren't filtering all events

### StopLatch not working
- Verify safety subscriber is registered
- Check that control loop checks stop state
- Ensure `speedStop()` is called immediately
