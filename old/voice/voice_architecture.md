# Voice Processing Pipeline Architecture Overview

## System Purpose

The voice processing pipeline (Process 2) converts microphone audio into clean, low-rate natural-language events that can safely influence a neural network and downstream potential-field control, without disturbing the 100 Hz velocity control loop. Speech is treated as a semantic sensor, not a direct control input.

## High-Level Architecture

The system follows a multi-stage pipeline architecture with strict non-blocking constraints:

```
Microphone → Audio Capture → VAD/Segmentation → ASR Decode → Text Normalization → Event Publication
                                                                    ↓
                                                          Safety Event Stream (StopLatch)
```

### Key Design Principles

- **Non-blocking interfaces**: All stages must not block the control loop
- **Isolated compute**: Heavy ML operations run in separate processes
- **Event-driven**: Speech published as events, not commands
- **Safety-first**: StopLatch events preempt all other logic
- **Quality gating**: Events filtered by quality metrics before NN consumption

## Pipeline Stages

### Stage A: Audio Capture

**Location**: `voice/voice_processing.py` - `AudioCapture` class

**Responsibilities**:
- Continuous microphone capture
- Fixed-size audio frames (20-30 ms)
- Push frames into ring buffer
- Timestamp each frame

**Constraints**:
- Must not block
- Must not perform ML or allocation-heavy work
- Must return immediately

**Output**: `TimestampedAudioFrame` objects containing PCM audio and timestamp

**Implementation Notes**:
- Use threading or async I/O for non-blocking capture
- Ring buffer size: ~2-3 seconds of audio (pre-roll buffer)
- Frame size: 20-30 ms (configurable)

### Stage B: VAD + Utterance Segmentation

**Location**: `voice/voice_processing.py` - `VADSegmenter` class

**Responsibilities**:
- Classify frames as speech or non-speech
- Assemble frames into utterances
- Track VAD statistics

**Key Parameters**:
- Frame size: 20-30 ms
- Start debounce: 60-150 ms of speech
- End hangover: 300-800 ms of silence
- Pre-roll buffer: 200-500 ms
- Max utterance length: 8-12 s

**Output**: `UtteranceAudio` objects containing:
- PCM audio data
- `t_start`, `t_end` timestamps
- VAD statistics (speech ratio, frame counts)

**Purpose**:
- Gate ASR compute (only process actual speech)
- Prevent CPU spikes
- Improve transcription quality

### Stage C: ASR Decode (Offline)

**Location**: `voice/voice_processing.py` - `ASRDecoder` class (or separate process)

**Responsibilities**:
- Convert utterance audio into text
- Produce decoding quality metrics

**Design**:
- Offline ASR only (no streaming initially)
- One decode at a time
- Prefer running in its own OS process
- Use queue for utterance submission

**Output**: `TranscriptResult` objects containing:
- `text`: Raw transcript
- `timing`: Word-level timing (optional)
- `quality_metrics`: ASR confidence scores
  - `avg_logprob`: Average log probability
  - `no_speech_prob`: Probability of no speech (if available)

**Implementation Notes**:
- Consider using Whisper, Wav2Vec2, or similar offline ASR
- Process isolation prevents blocking main thread
- Queue-based communication between stages

### Stage D: Text Normalization

**Location**: `voice/voice_processing.py` - `TextNormalizer` class

**Responsibilities**:
- Normalize casing (lowercase)
- Normalize numbers ("point two" → "0.2")
- Remove transcription artifacts cautiously
- Standardize phrasing for NN input distribution

**Output**: 
- `transcript_normalized`: Cleaned text for NN
- `transcript_raw`: Preserved for debugging

**Rationale**: Neural nets benefit from consistent input distributions

### Stage E: Event Publication

**Location**: `voice/voice_processing.py` - `EventPublisher` class

**Two Independent Streams**:

1. **Speech Event Stream**
   - Finalized utterance events
   - Delivered asynchronously
   - Non-blocking publish

2. **Safety Event Stream**
   - StopLatch events
   - High priority
   - Immediate publication

**Transport**: ZeroMQ PUB/SUB (or similar non-blocking message queue)
- Messages are small and self-contained
- Publisher never blocks
- Subscribers consume asynchronously

## Data Structures

### Core Speech Event Payload

```python
@dataclass
class SpeechUtteranceFinal:
    seq_id: int  # Monotonically increasing
    session_id: Optional[str]  # For conversational context
    t_capture_start: float  # Timestamp of speech start
    t_capture_end: float  # Timestamp of speech end
    utterance_ms: int  # Duration
    transcript_raw: str  # Raw ASR output
    transcript_normalized: str  # Cleaned text for NN
    is_final: bool  # Always True for finalized events
    language: Optional[str]
    
    # Quality metrics
    avg_logprob: Optional[float]
    no_speech_prob: Optional[float]
    vad_speech_ratio: float  # Fraction of frames classified as speech
    snr_estimate: Optional[float]
    dropped_frames: int
```

### Safety Event

```python
@dataclass
class StopLatch:
    timestamp: float
    trigger_source: str  # "keyword_spotter" or "asr_partial"
```

### Internal Data Structures

```python
@dataclass
class TimestampedAudioFrame:
    audio: np.ndarray  # PCM audio data
    timestamp: float
    frame_size_ms: int

@dataclass
class UtteranceAudio:
    audio: np.ndarray  # Complete utterance PCM
    t_start: float
    t_end: float
    vad_speech_ratio: float
    frame_count: int

@dataclass
class TranscriptResult:
    text: str
    timing: Optional[Dict[str, float]]  # Word-level timing
    avg_logprob: Optional[float]
    no_speech_prob: Optional[float]
```

## Gating Rules

Before publishing speech events to NN, apply quality gates:

**Publish if**:
- Utterance duration within bounds (e.g., 0.5s - 12s)
- VAD speech ratio > threshold (e.g., > 0.3)
- `no_speech_prob` < threshold (e.g., < 0.5) if available
- ASR quality metric > threshold (e.g., `avg_logprob` > -0.5)

**Otherwise**:
- Suppress event, or
- Publish diagnostic-only event (for debugging)

## Latency Expectations

- End-of-speech hangover: 300-800 ms
- ASR decode time: Model dependent (typically 1-3x real-time)
- Speech events update parameters/state only
- **Critical**: The 100 Hz speedL control loop remains unaffected

## Integration Points

### With Control Loop

- Voice events update state/parameters asynchronously
- Control loop polls for state changes (non-blocking)
- StopLatch immediately suppresses motion (latched downstream)

### With Neural Network

- NN subscribes to Speech Event Stream
- Events consumed asynchronously
- NN processes normalized transcripts
- NN outputs influence potential field parameters

### With Potential Field Control

- Potential field parameters updated from NN outputs
- Updates applied at control loop rate (100 Hz)
- No direct velocity modulation from speech

## Event Types

### Required Events

1. **SpeechUtteranceFinal**: Finalized utterance with all metadata
2. **StopLatch**: Safety event for immediate motion suppression

### Optional Events (Future)

3. **SpeechUtterancePartial**: Partial transcripts (for streaming)
4. **SpeechRejected**: Low-quality utterances (for diagnostics)

## File Structure

```
voice/
├── voice_processing.py          # Main pipeline implementation
│   ├── AudioCapture             # Stage A
│   ├── VADSegmenter             # Stage B
│   ├── ASRDecoder               # Stage C
│   ├── TextNormalizer           # Stage D
│   └── EventPublisher           # Stage E
├── voice_plan.txt               # Detailed specification
├── voice_architecture.md        # This document
└── adjustable_potential.py      # Integration example
```

## Configuration

Key parameters should be configurable:

- Audio capture: sample rate, frame size, buffer size
- VAD: debounce times, hangover, thresholds
- ASR: model selection, quality thresholds
- Normalization: rules and patterns
- Event publishing: transport, topics, serialization

## Testing Strategy

1. **Unit tests**: Each stage independently
2. **Integration tests**: Full pipeline with mock audio
3. **Latency tests**: Verify non-blocking behavior
4. **Quality tests**: Verify gating rules
5. **Safety tests**: Verify StopLatch behavior

## Implementation Phases

1. **Phase 1**: Basic pipeline structure, audio capture, simple VAD
2. **Phase 2**: ASR integration, text normalization
3. **Phase 3**: Event publishing, quality gating
4. **Phase 4**: StopLatch, safety events
5. **Phase 5**: Optimization, process isolation, performance tuning
