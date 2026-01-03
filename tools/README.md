# Tools Directory

This directory contains utility scripts for data collection, analysis, tuning, and diagnostics. The `analyze/` and `debug_visualizer/` subdirectories have their own README files.

## Quick Reference

### Data Collection & Recording
- **[ground_truth_follower.py](#ground_truth_followerpy)** - Collect ground truth data by following the lane center
- **[list_recordings.py](#list_recordingspy)** - List available recordings with metadata
- **[check_recording_data.py](#check_recording_datapy)** - Check what data exists in a recording

### Data Replay & Testing
- **[replay_perception.py](#replay_perceptionpy)** - Replay perception stack on recorded data (offline testing)
- **[reprocess_recording.py](#reprocess_recordingpy)** - Reprocess recordings with current code

### Calibration & Tuning
- **[calibrate_perception.py](#calibrate_perceptionpy)** - Calibrate perception coordinate conversion parameters
- **[parameter_sweep_tuning.py](#parameter_sweep_tuningpy)** - Automated parameter sweep for controller tuning
- **[parameter_sweep_with_metrics.py](#parameter_sweep_with_metricspy)** - Parameter sweep with detailed metrics
- **[quick_controller_tune.py](#quick_controller_tunepy)** - Quick controller parameter tuning using tests
- **[tune_controller_with_tests.py](#tune_controller_with_testspy)** - Tune controller with comprehensive test validation

### Diagnostics
- **[diagnose_ground_truth.py](#diagnose_ground_truthpy)** - Diagnose ground truth following issues
- **[diagnose_trajectory_vs_steering.py](#diagnose_trajectory_vs_steeringpy)** - Determine if issue is trajectory or steering

### Visualization & Validation
- **[visualize_frame_comparison.py](#visualize_frame_comparisonpy)** - Visualize detected lanes vs ground truth on images
- **[validate_analysis_consistency.py](#validate_analysis_consistencypy)** - Validate analysis tool consistency
- **[validate_coordinate_assumptions.py](#validate_coordinate_assumptionspy)** - Validate coordinate conversion assumptions

### Diagnostic Test Utilities
**Note:** These are diagnostic scripts (not pytest tests). They help debug specific issues by testing components in isolation. For automated unit tests, see `tests/` directory.

- **[test_ground_truth_alignment.py](#test_ground_truth_alignmentpy)** - Diagnostic: Find correct parameters for ground truth coordinate alignment
- **[test_ground_truth_reference.py](#test_ground_truth_referencepy)** - Diagnostic: Verify ground truth reference point calculation
- **[test_trajectory_heading_isolation.py](#test_trajectory_heading_isolationpy)** - Diagnostic: Test trajectory heading calculation in isolation

### Utilities
- **[collect_metrics.py](#collect_metricspy)** - Collect performance metrics from logs
- **[correlate_debug_with_state.py](#correlate_debug_with_statepy)** - Correlate debug visualizer data with vehicle state
- **[align_tests_with_unity.py](#align_tests_with_unitypy)** - Align test coordinate assumptions with Unity

### Analysis Wrappers
- **[analyze.py](#analyzepy)** - Wrapper for comprehensive analysis (see `analyze/README.md`)

---

## Detailed Tool Descriptions

### Data Collection & Recording

#### `ground_truth_follower.py`
**Purpose:** Collect ground truth data by driving the car along the ground truth lane center path.

**When to use:**
- Before testing perception improvements
- To collect training/validation data
- To generate reference recordings for comparison

**Usage:**
```bash
# Follow ground truth for 60 seconds
python tools/ground_truth_follower.py --duration 60

# Custom output filename
python tools/ground_truth_follower.py --duration 60 --output my_test_run

# With Unity auto-launch
python tools/ground_truth_follower.py --duration 60 --launch-unity
```

**Workflow:**
1. Run this script to collect ground truth data
2. Use `replay_perception.py` to test perception on the recorded data
3. Compare perception output vs ground truth

---

#### `list_recordings.py`
**Purpose:** List available recordings with their types and metadata.

**When to use:**
- To find a specific recording
- To see what recordings are available
- To check recording metadata

**Usage:**
```bash
# List recent recordings
python tools/list_recordings.py

# List all recordings
python tools/list_recordings.py --all
```

---

#### `check_recording_data.py`
**Purpose:** Check what data exists in a recording file.

**When to use:**
- To verify a recording has the expected data
- To debug missing data issues
- To understand recording structure

**Usage:**
```bash
python tools/check_recording_data.py <recording_file>
```

---

### Data Replay & Testing

#### `replay_perception.py`
**Purpose:** Replay perception stack on recorded data (offline testing without Unity).

**When to use:**
- To test perception improvements without Unity
- To compare perception output vs ground truth
- To debug perception issues on specific frames

**Usage:**
```bash
# Replay latest recording
python tools/replay_perception.py

# Replay specific recording
python tools/replay_perception.py data/recordings/recording_20240103_120000.h5

# Custom output name
python tools/replay_perception.py --output perception_test
```

**What it does:**
- Loads frames from recording
- Runs current perception code on each frame
- Compares against ground truth
- Saves results to new recording

---

#### `reprocess_recording.py`
**Purpose:** Reprocess an existing recording with current (fixed) code.

**When to use:**
- After fixing perception bugs
- To regenerate data with corrected calculations
- To test fixes without re-running Unity

**Usage:**
```bash
# Reprocess latest recording
python tools/reprocess_recording.py

# Reprocess specific recording
python tools/reprocess_recording.py data/recordings/recording_20240103_120000.h5
```

---

### Calibration & Tuning

#### `calibrate_perception.py`
**Purpose:** Find optimal perception coordinate conversion parameters using ground truth data.

**When to use:**
- When perception lane positions don't match ground truth
- After changing camera parameters
- To optimize distance scaling or camera offset

**Usage:**
```bash
# Calibrate using latest recording
python tools/calibrate_perception.py

# Calibrate using specific recording
python tools/calibrate_perception.py data/recordings/recording_20240103_120000.h5

# List available recordings
python tools/calibrate_perception.py --list
```

**Note:** Requires a recording with both perception and ground truth data (run `replay_perception.py` first).

---

#### `parameter_sweep_tuning.py`
**Purpose:** Automated parameter sweep for controller tuning using fast synthetic tests.

**When to use:**
- To find optimal PID gains
- To tune controller parameters systematically
- When you need to test many parameter combinations

**Usage:**
```bash
python tools/parameter_sweep_tuning.py
```

**Advantages:**
- Much faster than Unity tests (~0.05s per test vs 30s+)
- Tests multiple parameter combinations automatically
- Uses realistic synthetic tests

---

#### `parameter_sweep_with_metrics.py`
**Purpose:** Parameter sweep with detailed performance metrics.

**When to use:**
- When you need detailed metrics for each parameter combination
- For comprehensive parameter optimization
- To understand parameter impact on multiple metrics

**Usage:**
```bash
python tools/parameter_sweep_with_metrics.py
```

---

#### `quick_controller_tune.py`
**Purpose:** Quick controller parameter tuning using fast tests.

**When to use:**
- For rapid iteration on controller parameters
- When you want to test specific parameter values
- For quick validation of parameter changes

**Usage:**
```bash
# Interactive tuning
python tools/quick_controller_tune.py

# Test specific parameters
python tools/quick_controller_tune.py --kp 0.4 --kd 0.5
```

**Advantages:**
- Very fast (~1 second vs 30+ seconds in Unity)
- Interactive parameter testing
- Immediate feedback

---

#### `tune_controller_with_tests.py`
**Purpose:** Tune controller with comprehensive test validation.

**When to use:**
- For thorough parameter validation
- When you need to ensure parameters work across all test scenarios
- Before deploying parameter changes

**Usage:**
```bash
python tools/tune_controller_with_tests.py
```

---

### Diagnostics

#### `diagnose_ground_truth.py`
**Purpose:** Diagnose why ground truth following isn't working.

**When to use:**
- When ground truth follower isn't working
- To check if ground truth data is being received
- To verify steering calculations
- To test coordinate system interpretation

**Usage:**
```bash
python tools/diagnose_ground_truth.py
```

**What it checks:**
- Bridge connection
- Ground truth data availability
- Steering calculation correctness
- Coordinate system interpretation

---

#### `diagnose_trajectory_vs_steering.py`
**Purpose:** Determine if the issue is in trajectory planning or steering control.

**When to use:**
- When car isn't following the path correctly
- To isolate whether problem is trajectory or control
- To debug path following issues

**Usage:**
```bash
# Diagnose latest recording
python tools/diagnose_trajectory_vs_steering.py

# Diagnose specific recording
python tools/diagnose_trajectory_vs_steering.py data/recordings/recording_20240103_120000.h5
```

**What it analyzes:**
- Reference point accuracy (trajectory)
- Steering following trajectory (control)
- Where the problem lies

---

### Visualization & Validation

#### `visualize_frame_comparison.py`
**Purpose:** Visualize detected lanes vs ground truth lanes on camera images.

**When to use:**
- To see where perception is failing visually
- To debug lane detection issues
- To compare detection vs ground truth side-by-side

**Usage:**
```bash
# Visualize latest recording
python tools/visualize_frame_comparison.py

# Visualize specific recording
python tools/visualize_frame_comparison.py data/recordings/recording_20240103_120000.h5

# Visualize specific frame
python tools/visualize_frame_comparison.py --frame 100
```

**Output:** Saves images with detected lanes (red) and ground truth lanes (green) overlaid.

---

#### `validate_analysis_consistency.py`
**Purpose:** Validate that analysis tools produce consistent results.

**When to use:**
- After modifying analysis tools
- To ensure analysis consistency
- To verify analysis tool correctness

**Usage:**
```bash
python tools/validate_analysis_consistency.py
```

---

#### `validate_coordinate_assumptions.py`
**Purpose:** Validate coordinate conversion assumptions.

**When to use:**
- After changing camera parameters
- To verify coordinate conversion correctness
- To debug coordinate system issues

**Usage:**
```bash
python tools/validate_coordinate_assumptions.py
```

---

### Test Utilities

#### `test_ground_truth_alignment.py`
**Purpose:** Diagnostic script to find correct parameters for ground truth coordinate alignment.

**When to use:**
- When ground truth lanes don't align with detected lanes in visualizations
- To find optimal camera parameters (height, FOV, lookahead) for alignment
- To debug coordinate conversion issues

**Usage:**
```bash
# Test parameter combinations to find best alignment
python tools/test_ground_truth_alignment.py --recording <recording_file> --frame 0

# Test more combinations
python tools/test_ground_truth_alignment.py --recording <recording_file> --max-combinations 200
```

**Note:** This is a diagnostic tool, not a pytest test. It varies parameters to find optimal alignment.

---

#### `test_ground_truth_reference.py`
**Purpose:** Diagnostic script to verify ground truth reference point calculation.

**When to use:**
- When reference points seem incorrect
- To check if Unity is using correct reference point
- To debug trajectory planning reference point issues

**Usage:**
```bash
# Analyze reference point for specific frame
python tools/test_ground_truth_reference.py --recording <recording_file> --frame 0
```

**Note:** This is a diagnostic tool, not a pytest test. It analyzes and reports reference point data.

---

#### `test_trajectory_heading_isolation.py`
**Purpose:** Diagnostic script to test trajectory heading calculation in isolation.

**When to use:**
- To test heading fixes without running full system
- To debug heading calculation issues
- To validate trajectory heading logic with real Unity data

**Usage:**
```bash
# Test heading with latest recording
python tools/test_trajectory_heading_isolation.py

# Test with specific recording and frame
python tools/test_trajectory_heading_isolation.py --recording <recording_file> --frame 0
```

**Note:** This is a diagnostic tool, not a pytest test. It tests heading calculation in isolation using recorded data.

---

### Utilities

#### `collect_metrics.py`
**Purpose:** Collect performance metrics from logs.

**When to use:**
- After running a test drive
- To analyze system performance
- To generate performance reports

**Usage:**
```bash
# Collect metrics from latest log
python tools/collect_metrics.py --log tmp/logs/av_stack.log --output metrics.json

# Or use the shell wrapper
./tools/collect_metrics --log tmp/logs/av_stack.log --output metrics.json
```

**Output:** JSON file with comprehensive metrics (lane detection rate, speed statistics, etc.)

---

#### `correlate_debug_with_state.py`
**Purpose:** Correlate debug visualizer data with vehicle state.

**When to use:**
- To understand debug visualizer data
- To correlate visualizations with actual state
- To debug visualization issues

**Usage:**
```bash
python tools/correlate_debug_with_state.py
```

---

#### `align_tests_with_unity.py`
**Purpose:** Align test coordinate assumptions with Unity.

**When to use:**
- After changing Unity camera setup
- To ensure tests match Unity behavior
- To validate test assumptions

**Usage:**
```bash
python tools/align_tests_with_unity.py <recording_file>
```

---

### Analysis Wrappers

#### `analyze.py`
**Purpose:** Wrapper for comprehensive analysis tool.

**When to use:**
- As a convenient entry point for analysis
- For quick analysis of recordings

**Usage:**
```bash
# Analyze latest recording
python tools/analyze.py --latest

# Analyze specific recording
python tools/analyze.py data/recordings/recording_20240103_120000.h5
```

**Note:** This is a wrapper for `tools/analyze/analyze_recording_comprehensive.py`. See `tools/analyze/README.md` for more analysis tools.

---

## Common Workflows

### Testing Perception Improvements
1. Collect ground truth data: `python tools/ground_truth_follower.py --duration 60`
2. Replay perception: `python tools/replay_perception.py`
3. Visualize results: `python tools/visualize_frame_comparison.py`
4. Analyze: `python tools/analyze.py --latest`

### Tuning Controller Parameters
1. Quick test: `python tools/quick_controller_tune.py --kp 0.4 --kd 0.5`
2. Comprehensive sweep: `python tools/parameter_sweep_tuning.py`
3. Validate with tests: `python tools/tune_controller_with_tests.py`
4. Test in Unity: `./start_av_stack.sh --duration 30`

### Debugging Issues
1. Check recording: `python tools/check_recording_data.py <recording>`
2. Diagnose: `python tools/diagnose_trajectory_vs_steering.py <recording>`
3. Visualize: `python tools/visualize_frame_comparison.py <recording>`
4. Analyze: `python tools/analyze.py <recording>`

### Calibrating Perception
1. Collect GT data: `python tools/ground_truth_follower.py --duration 60`
2. Replay perception: `python tools/replay_perception.py`
3. Calibrate: `python tools/calibrate_perception.py`
4. Validate: `python tools/validate_coordinate_assumptions.py`

---

## Related Documentation

- **[tools/analyze/README.md](analyze/README.md)** - Analysis tools documentation
- **[tools/debug_visualizer/README.md](debug_visualizer/README.md)** - Debug visualizer documentation
- **[README.md](../README.md)** - Main project documentation

