# Analysis Tools

This directory contains analysis scripts for evaluating AV stack performance from recorded data.

## Quick Start

**For overall drive performance evaluation:**
```bash
python tools/analyze/analyze_drive_overall.py --latest
```

**For detailed diagnostics:**
```bash
python tools/analyze/analyze_recording_comprehensive.py --latest
```

## Analysis Scripts

### Primary Analysis Tools

#### `analyze_drive_overall.py` ⭐ **RECOMMENDED FOR QUICK OVERVIEW**
**Purpose:** Comprehensive overall drive performance evaluation with industry-standard metrics.

**Metrics:**
- Executive summary (overall score, key issues, drive duration)
- Path tracking (lateral/heading error, time in lane)
- Control smoothness (jerk, steering rate, oscillation)
- Perception quality (detection rate, confidence, stale data)
- Trajectory quality (accuracy, smoothness, availability)
- System health (PID health, error conflicts, stale commands)
- Safety metrics (out-of-lane events)

**Usage:**
```bash
# Analyze latest recording
python tools/analyze/analyze_drive_overall.py --latest

# Analyze specific recording
python tools/analyze/analyze_drive_overall.py data/recordings/recording_YYYYMMDD_HHMMSS.h5

# List available recordings
python tools/analyze/analyze_drive_overall.py --list
```

**When to use:** Start here for a quick overview of drive performance. Provides a single score and prioritized recommendations.

---

#### `analyze_recording_comprehensive.py` ⭐ **RECOMMENDED FOR DETAILED DIAGNOSTICS**
**Purpose:** Deep dive into all system components with root cause analysis.

**Features:**
- Basic metrics and statistics
- Control system analysis (PID, steering, speed)
- Oscillation and convergence analysis
- Root cause diagnostics (perception, trajectory, controller)
- Prioritized recommendations with severity levels

**Usage:**
```bash
# Analyze latest recording
python tools/analyze/analyze_recording_comprehensive.py --latest

# Analyze specific recording
python tools/analyze/analyze_recording_comprehensive.py data/recordings/recording_YYYYMMDD_HHMMSS.h5
```

**When to use:** When you need detailed diagnostics to identify specific issues. Provides component-level analysis and root cause identification.

---

### Specialized Analysis Tools

#### `analyze_trajectory.py`
**Purpose:** Analyze trajectory planning accuracy against ground truth.

**What it analyzes:**
- Trajectory centering accuracy (does it follow lane center?)
- Reference point accuracy (at lookahead distance)
- Trajectory vs ground truth path comparison
- Trajectory quality metrics (smoothness, curvature)

**Usage:**
```bash
python tools/analyze/analyze_trajectory.py <recording_file>
python tools/analyze/analyze_trajectory.py --list
```

**When to use:** When investigating trajectory planning issues or validating trajectory accuracy.

---

#### `analyze_oscillation_root_cause.py`
**Purpose:** Deep analysis of oscillation to identify root cause using frequency analysis.

**What it analyzes:**
- Oscillation frequency (FFT analysis)
- Phase relationships (steering vs error)
- Component-level oscillation (perception, trajectory, controller)
- Root cause identification

**Usage:**
```bash
python tools/analyze/analyze_oscillation_root_cause.py <recording_file>
```

**When to use:** When experiencing oscillation issues and need to identify which component is causing it.

---

#### `analyze_jerkiness.py`
**Purpose:** Analyze jerkiness/oscillation in steering during curves.

**What it analyzes:**
- Steering oscillation patterns (turn -> straight -> correcting)
- Error oscillation patterns
- Feedforward vs feedback contributions
- Root cause of jerkiness

**Usage:**
```bash
python tools/analyze/analyze_jerkiness.py <recording_file>
```

**When to use:** When experiencing jerky steering, especially on curves.

---

#### `analyze_control_layers.py`
**Purpose:** Analyze control layer architecture and identify potential issues from stacked controllers/filters.

**What it analyzes:**
- Control layer delays (lane smoothing, reference smoothing, PID, rate limiting)
- Phase relationships between layers
- Total system delay
- Potential phase issues

**Usage:**
```bash
python tools/analyze/analyze_control_layers.py <recording_file>
```

**When to use:** When investigating control system delays or phase issues.

---

#### `analyze_lateral_error_convergence.py`
**Purpose:** Analyze lateral error convergence and identify why error isn't dropping below threshold.

**What it analyzes:**
- Error convergence over time
- Error distribution by time segments
- Factors preventing convergence (oscillation, bias, etc.)

**Usage:**
```bash
python tools/analyze/analyze_lateral_error_convergence.py <recording_file>
```

**When to use:** When lateral error is not converging or staying above acceptable threshold.

---

#### `analyze_heading_opposition.py`
**Purpose:** Analyze heading vs lateral error conflicts.

**What it analyzes:**
- Frequency of heading/lateral error conflicts
- Impact on steering direction
- Root cause of conflicts

**Usage:**
```bash
python tools/analyze/analyze_heading_opposition.py <recording_file>
```

**When to use:** When experiencing steering direction errors or conflicting error signals.

---

#### `analyze_perception_questions.py`
**Purpose:** Analyze perception data to systematically answer 7 key questions about perception accuracy.

**What it analyzes:**
1. How accurate is lane detection?
2. How consistent is lane detection?
3. Are lanes detected in the correct positions?
4. How does detection vary with road conditions?
5. Are there systematic biases?
6. How does detection compare to ground truth?
7. What are common failure modes?

**Usage:**
```bash
python tools/analyze/analyze_perception_questions.py <recording_file>
```

**When to use:** When evaluating perception model performance or debugging perception issues.

---

#### `analyze_perception_freeze.py`
**Purpose:** Detect and analyze perception freeze events (when perception output stops changing).

**What it analyzes:**
- Perception freeze detection
- Duration of freeze events
- Impact on control system
- Root cause of freezes

**Usage:**
```bash
python tools/analyze/analyze_perception_freeze.py <recording_file>
```

**When to use:** When suspecting perception is frozen or not updating.

---

#### `analyze_frame_by_frame.py`
**Purpose:** Detailed frame-by-frame analysis with error metrics.

**What it analyzes:**
- Per-frame error metrics
- Frame-by-frame comparison (detected vs ground truth)
- Error distribution across frames
- Problematic frame identification

**Usage:**
```bash
python tools/analyze/analyze_frame_by_frame.py <recording_file>
```

**When to use:** When you need to examine specific frames or understand error patterns at the frame level.

---

#### `analyze_detection_failures.py`
**Purpose:** Analyze lane detection failures and identify patterns.

**What it analyzes:**
- Detection failure frequency
- Failure patterns (when/where failures occur)
- Impact of failures on system
- Common failure modes

**Usage:**
```bash
python tools/analyze/analyze_detection_failures.py <recording_file>
```

**When to use:** When investigating why lane detection is failing.

---

#### `analyze_lane_detection_consistency.py`
**Purpose:** Analyze consistency of lane detection over time.

**What it analyzes:**
- Detection consistency metrics
- Temporal stability
- Variance in detection
- Consistency issues

**Usage:**
```bash
python tools/analyze/analyze_lane_detection_consistency.py <recording_file>
```

**When to use:** When evaluating detection stability or debugging inconsistent detections.

---

#### `analyze_circular_drive.py` / `analyze_circular_drive_simple.py`
**Purpose:** Analyze performance on circular/oval tracks.

**What it analyzes:**
- Performance on curves
- Steering behavior on circular paths
- Error patterns on curves
- Curve-specific metrics

**Usage:**
```bash
python tools/analyze/analyze_circular_drive.py <recording_file>
python tools/analyze/analyze_circular_drive_simple.py <recording_file>
```

**When to use:** When testing on circular/oval tracks or analyzing curve performance.

---

#### `analyze_debug_visualizations.py`
**Purpose:** Analyze debug visualization data.

**What it analyzes:**
- Debug visualization accuracy
- Visualization vs actual data comparison
- Debug data quality

**Usage:**
```bash
python tools/analyze/analyze_debug_visualizations.py <recording_file>
```

**When to use:** When validating debug visualizations or debugging visualization issues.

---

## Workflow Recommendations

### For Regular Performance Evaluation
1. Start with `analyze_drive_overall.py` for quick overview
2. If issues detected, use `analyze_recording_comprehensive.py` for detailed diagnostics

### For Specific Issues

**Oscillation/Jerkiness:**
1. `analyze_drive_overall.py` - check oscillation frequency
2. `analyze_jerkiness.py` - detailed jerkiness analysis
3. `analyze_oscillation_root_cause.py` - root cause identification

**Perception Issues:**
1. `analyze_perception_questions.py` - comprehensive perception analysis
2. `analyze_detection_failures.py` - failure pattern analysis
3. `analyze_perception_freeze.py` - freeze detection

**Trajectory Issues:**
1. `analyze_trajectory.py` - trajectory accuracy analysis
2. `analyze_control_layers.py` - control layer delays

**Error Convergence:**
1. `analyze_lateral_error_convergence.py` - convergence analysis
2. `analyze_heading_opposition.py` - error conflict analysis

---

## Finding Recordings

Most scripts support `--latest` flag to analyze the most recent recording:

```bash
python tools/analyze/analyze_drive_overall.py --latest
```

Or use the list command:
```bash
python tools/analyze/analyze_drive_overall.py --list
```

Recordings are stored in `data/recordings/` with format `recording_YYYYMMDD_HHMMSS.h5`.

---

## Notes

- All scripts require HDF5 recordings from the AV stack
- Most scripts work with both ground truth and non-ground truth recordings
- Some scripts (like `analyze_trajectory.py`) work best with ground truth data
- Scripts automatically handle missing data gracefully
- Use `--latest` flag when available to analyze most recent recording

