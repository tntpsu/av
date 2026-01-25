# Autonomous Vehicle Stack for Unity 3D Simulation

A complete autonomous vehicle stack implementing perception, trajectory planning, and control for Unity 3D simulation. This project implements single-camera lane following with ground truth following capabilities and comprehensive analysis tools.

## Current System State

### ✅ Implemented Features

- **Perception**: Segmentation-based lane detection (default) with CV fallback (color masks, edge detection, Hough lines) and polynomial fitting
- **Trained Segmentation Model**: Supports running a trained checkpoint via `--segmentation-checkpoint`
- **Trajectory Planning**: Rule-based path planning with reference point smoothing and bias correction
- **Control**: PID controller with feedforward (path curvature) + feedback (error correction) architecture
- **Ground Truth Following**: Direct velocity control mode for precise ground truth path following
- **Data Recording**: Automatic HDF5 recording of all frames, vehicle state (including Unity time/frame count), control commands, and ground truth data
- **Analysis Tools**: Comprehensive analysis suite for evaluating drive performance
- **Debug Visualizer**: Web-based tool for visualizing recorded data with overlays
- **Testing**: Extensive test suite covering control, trajectory, perception, and integration scenarios
- **Standalone Unity Player Workflow**: Build and run the Unity player directly from scripts for automated testing

### 🔧 Current Architecture

```
Unity Simulator (C#)
    ↓ (camera feed, vehicle state, ground truth)
Python Bridge/API (FastAPI)
    ↓
Perception (Segmentation default, CV fallback) → Lane Detection
    ↓ (lane line coefficients, positions)
Trajectory Planner (Rule-based) → Path Planning
    ↓ (reference point: x, y, heading)
Control Stack (PID + Feedforward) → Steering/Throttle/Brake
    ↓ (control commands)
Unity Simulator (C#) → Vehicle Control
    ↑
Data Recorder (HDF5) ← All sensor data + commands + ground truth
```

### Key Components

- **Perception**: `perception/inference.py` - Segmentation default with CV fallback and temporal filtering
- **Trajectory**: `trajectory/inference.py` - Rule-based planner with reference point smoothing
- **Control**: `control/pid_controller.py` - Feedforward + feedback PID controller
- **Bridge**: `bridge/server.py` - FastAPI server for Unity-Python communication
- **Data**: `data/recorder.py` - HDF5 recording with ground truth support
- **Main Stack**: `av_stack.py` - Integration of all components

## Quick Start

### 1. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Unity

Follow the detailed instructions in [setup_unity.md](setup_unity.md)

### 3. Running the System

#### Primary Run Scripts

**Option A: Standard AV Stack (Perception → Trajectory → Control)**
```bash
# Basic startup (segmentation default)
./start_av_stack.sh

# With Unity auto-launch and auto-play
./start_av_stack.sh --launch-unity --unity-auto-play

# Run for specific duration (e.g., 60 seconds)
./start_av_stack.sh --duration 60 --launch-unity

# Force CV-only mode
./start_av_stack.sh --use-cv

# Use a trained segmentation checkpoint
./start_av_stack.sh --segmentation-checkpoint /path/to/checkpoint.pt

# Force kill existing processes on port 8000
./start_av_stack.sh --force
```

**Option B: Ground Truth Follower (Direct GT Path Following)**
```bash
# Follow ground truth path for 60 seconds
python tools/ground_truth_follower.py --duration 60

# Custom output filename
python tools/ground_truth_follower.py --duration 60 --output my_test_run

# With Unity auto-launch
python tools/ground_truth_follower.py --duration 60 --launch-unity
```

**Option C: Standalone Unity Player (Automated Workflow)**
```bash
# Build and run Unity player for a 60s test (no editor interaction)
./start_av_stack.sh --build-unity-player --skip-unity-build-if-clean --run-unity-player --duration 60
```

**What each does:**
- `start_av_stack.sh`: Runs full AV stack (perception → trajectory → control) - tests your perception and control
- `ground_truth_follower.py`: Follows ground truth path directly - collects data for perception testing

#### Manual Startup (Alternative)

```bash
# Terminal 1 - Bridge Server
python -m bridge.server

# Terminal 2 - AV Stack
python av_stack.py  # Data recording enabled by default

# Or Ground Truth Follower
python tools/ground_truth_follower.py --duration 60
```

### 4. Unity Setup

**Option A: Auto-launch (Recommended)**
```bash
./start_av_stack.sh --launch-unity --unity-auto-play
```

**Option B: Manual**
1. Open Unity project (`unity/AVSimulation`)
2. Load the `SampleScene` scene
3. Select your Car GameObject
4. In AV Bridge component → Check **"Enable AV Control"**
5. Press **▶ PLAY**

The system will automatically:
- Capture camera frames (30 FPS)
- Run perception model
- Plan trajectory
- Control the vehicle
- Record all data to HDF5 files in `data/recordings/`

## Analysis and Debugging

### Overall Drive Analysis

**Quick performance overview:**
```bash
# Analyze latest recording
python tools/analyze/analyze_drive_overall.py --latest

# Analyze specific recording
python tools/analyze/analyze_drive_overall.py data/recordings/recording_YYYYMMDD_HHMMSS.h5

# List available recordings
python tools/analyze/analyze_drive_overall.py --list
```

**Detailed diagnostics:**
```bash
# Comprehensive analysis with root cause identification
python tools/analyze/analyze_recording_comprehensive.py --latest
```

See [tools/analyze/README.md](tools/analyze/README.md) for all analysis tools.

### Debug Visualizer

**Start the visualizer server:**
```bash
cd tools/debug_visualizer
python server.py
```

The server runs on `http://localhost:5000`.

**Open the visualizer:**
```bash
# Option 1: Use Python's built-in server
cd tools/debug_visualizer
python -m http.server 8000
# Then open http://localhost:8000/index.html in your browser

# Option 2: Open directly (may have CORS issues)
open tools/debug_visualizer/index.html
```

**Features:**

**✅ Phase 1: Frame-Level Diagnostics (Complete)**
- **Polynomial Inspector**: Analyze polynomial fitting for any frame
  - Shows recorded vs. re-run detection
  - Full system validation (what av_stack.py would do)
  - Explains why detections would be rejected
  - Provides recommendations for fixes
- **On-Demand Debug Overlays**: Generate edges, yellow_mask, and combined for ANY frame
  - No longer limited to every 30th frame
  - Visualize detected points/edges that led to bad polynomial fits
- Frame-by-frame navigation with keyboard controls
- Visual overlays for lane lines, trajectory, and ground truth
- Data side panel showing all frame data
- Export frames as PNG

**🚧 Phase 2: Recording-Level Analysis (In Progress)**
- Recording Summary tab (overall metrics and health graphs)
- Issues Detection (auto-detect problematic frames and jump to them)
- Trajectory vs Steering Diagnostic (identify which component is failing)

See [tools/debug_visualizer/README.md](tools/debug_visualizer/README.md) for full details and [tools/debug_visualizer/CONSOLIDATION_PLAN.md](tools/debug_visualizer/CONSOLIDATION_PLAN.md) for the consolidation roadmap.

### Other Analysis Tools

```bash
# Trajectory accuracy analysis
python tools/analyze/analyze_trajectory.py --latest

# Oscillation root cause analysis
python tools/analyze/analyze_oscillation_root_cause.py --latest

# Jerkiness analysis
python tools/analyze/analyze_jerkiness.py --latest

# Perception quality analysis
python tools/analyze/analyze_perception_questions.py --latest
```

## Project Structure

```
av/
├── unity/                           # Unity project files
│   └── AVSimulation/
│       ├── Assets/
│       │   ├── Scripts/            # C# scripts (AVBridge, CarController, etc.)
│       │   ├── Scenes/             # Unity scenes
│       │   ├── Materials/         # Lane marking materials
│       │   └── Prefabs/           # Car prefab with camera
│       └── .unity_autoplay         # Auto-play flag file
├── perception/                     # Perception module
│   ├── inference.py                # Segmentation + CV fallback
│   └── models/                     # Model definitions (checkpoints are gitignored)
├── trajectory/                     # Trajectory planning
│   ├── inference.py                # Trajectory planning inference
│   └── models/                     # Trajectory planning models
├── control/                        # Control stack
│   ├── pid_controller.py           # PID + feedforward controller
│   └── vehicle_model.py           # Bicycle model
├── bridge/                         # Unity-Python communication
│   ├── server.py                   # FastAPI server
│   └── client.py                   # Unity bridge client
├── data/                           # Data recording and replay
│   ├── recorder.py                 # HDF5 data recorder
│   ├── formats/                    # Data format definitions
│   └── recordings/                 # HDF5 recording files
├── tools/                          # Analysis and utility tools (see tools/README.md)
│   ├── analyze/                    # Analysis scripts (see tools/analyze/README.md)
│   │   ├── analyze_drive_overall.py      # Primary overall analysis
│   │   ├── analyze_recording_comprehensive.py  # Detailed diagnostics
│   │   └── ...                    # Specialized analysis tools
│   ├── debug_visualizer/           # Web-based debug visualizer (see tools/debug_visualizer/README.md)
│   │   ├── server.py              # Visualizer backend server
│   │   ├── index.html             # Visualizer frontend
│   │   ├── visualizer.js          # Visualization logic
│   │   ├── backend/               # Analysis backend modules (Phase 2)
│   │   └── CONSOLIDATION_PLAN.md  # Tool consolidation roadmap
│   ├── ground_truth_follower.py   # Ground truth path follower
│   ├── replay_perception.py       # Perception replay tool
│   └── calibrate_perception.py    # Perception calibration
├── tests/                          # Test suite (see tests/README.md)
│   ├── test_control.py            # Control system tests
│   ├── test_trajectory.py         # Trajectory planning tests
│   ├── test_perception_*.py       # Perception tests
│   └── test_integration.py        # Integration tests
├── start_av_stack.sh              # Primary startup script
├── launch_unity.sh                 # Unity launcher script
├── av_stack.py                     # Main AV stack integration
└── config.yaml                     # Configuration file
```

## Data Collection

Data recording is **enabled by default** when running `av_stack.py` or `ground_truth_follower.py`. All frames, vehicle state, control commands, and ground truth data are automatically saved to HDF5 files in `data/recordings/`.

**Recording format:**
- `camera/`: Camera frames (images)
- `vehicle_state/`: Position, speed, heading, etc.
- `perception/`: Lane detection results
- `trajectory/`: Trajectory planning results
- `control/`: Control commands (steering, throttle, brake)
- `ground_truth/`: Ground truth lane positions (when available)

**View recordings:**
```bash
# List recordings
python tools/list_recordings.py

# Replay recording
python -m data.replay --file data/recordings/recording_YYYYMMDD_HHMMSS.h5
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_control.py -v

# Run tests by category (using markers)
pytest tests/ -m unit          # Fast unit tests
pytest tests/ -m integration   # Integration tests
pytest tests/ -m control       # Control system tests
pytest tests/ -m trajectory    # Trajectory planning tests
pytest tests/ -m perception    # Perception tests

# Run with coverage
pytest tests/ --cov=perception --cov=trajectory --cov=control --cov-report=term-missing

# Run tests and drop into debugger on failure
pytest tests/ --pdb
```

### Test Categories

- **Control Tests** (`-m control`): PID controller, steering logic, integral accumulation
- **Trajectory Tests** (`-m trajectory`): Reference point calculation, smoothing, bias correction
- **Perception Tests** (`-m perception`): Lane detection, coordinate conversion
- **Integration Tests** (`-m integration`): End-to-end scenarios, system stability
- **Unit Tests** (`-m unit`): Fast, isolated unit tests

See [tests/README.md](tests/README.md) for comprehensive test documentation.

### Debugging Workflow

1. **Reproduce the bug** - Create a minimal test case
2. **Write a failing test** - Test should fail before fix, pass after
3. **Fix the bug** - Address root cause, not symptoms
4. **Run full test suite** - Ensure no regressions: `pytest tests/`
5. **Verify in Unity** - Test in actual simulation if applicable

**Important**: Every bug fix must include a test that reproduces the original issue.

## Configuration

Configuration is managed through `config.yaml`. Key sections:

- `control/`: PID gains, steering limits, rate limiting
- `trajectory/`: Lookahead distance, smoothing parameters, bias correction
- `safety/`: Emergency stop thresholds, bounds checking

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for detailed configuration options.

## Development

### Code Structure

- `perception/`: Lane detection models and inference
- `trajectory/`: Path planning algorithms
- `control/`: Vehicle control (PID with feedforward)
- `bridge/`: Unity-Python communication layer
- `data/`: Data recording and replay utilities
- `tools/`: Analysis and utility tools

### Pre-commit Hooks (Optional)

Install pre-commit hooks to run tests and linting automatically:

```bash
pip install pre-commit
pre-commit install
```

This will run tests, format code, and check for issues before each commit.

## Requirements

- Python 3.8+
- PyTorch 1.12+ (required for segmentation model)
- Unity 2021.3 LTS or later
- FastAPI
- NumPy, OpenCV, h5py

See `requirements.txt` for complete list.

## Documentation

### Core Documentation
- **[README.md](README.md)** - This file (project overview and quick start)
- **[setup_unity.md](setup_unity.md)** - Unity setup instructions
- **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)** - Configuration system guide

### Additional Documentation
- **[docs/README.md](docs/README.md)** - Documentation index
- **[docs/TODO.md](docs/TODO.md)** - Active and backlog TODO tracker
- **[docs/README_STARTUP.md](docs/README_STARTUP.md)** - Detailed startup instructions and troubleshooting
- **[docs/DEVELOPMENT_GUIDELINES.md](docs/DEVELOPMENT_GUIDELINES.md)** - Development best practices and critical lessons learned
- **[docs/AI_MEMORY_GUIDE.md](docs/AI_MEMORY_GUIDE.md)** - AI assistant memory and context guide
- **[docs/archive/](docs/archive/)** - Historical analysis and investigation notes (archived)

### Component Documentation
- **[tools/README.md](tools/README.md)** - Tools directory documentation (data collection, tuning, diagnostics)
- **[tools/analyze/README.md](tools/analyze/README.md)** - Analysis tools documentation
- **[tools/debug_visualizer/README.md](tools/debug_visualizer/README.md)** - Debug visualizer documentation
- **[tests/README.md](tests/README.md)** - Test suite documentation

## Roadmap

- [x] Basic Unity setup
- [x] Unity-Python bridge
- [x] Data recorder
- [x] CV-based lane detection
- [x] Trajectory planner
- [x] PID controller with feedforward
- [x] Ground truth following
- [x] Analysis tools
- [x] Debug visualizer
- [x] Comprehensive test suite
- [ ] ML-based perception models
- [ ] Multi-camera support
- [ ] Lidar integration
- [ ] Radar integration
- [ ] Sensor fusion
- [ ] Advanced trajectory planning (MPC)
- [ ] Reinforcement learning

## License

MIT License

## Contributing

Contributions welcome! Please read `docs/DEVELOPMENT_GUIDELINES.md` before submitting PRs.
