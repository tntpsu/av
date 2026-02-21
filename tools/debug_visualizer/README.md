# AV Stack Debug Visualizer

Interactive web-based debug visualizer for analyzing AV stack recordings. Displays synchronized camera frames with all associated data, debug visualizations, and ground truth comparisons.

## Features

### ✅ Phase 1: Frame-Level Diagnostics (Complete)
- **Polynomial Inspector**: Analyze polynomial fitting for any frame
  - Shows recorded vs. re-run detection
  - Full system validation (what av_stack.py would do)
  - Explains why detections would be rejected
  - Provides recommendations for fixes
- **On-Demand Debug Overlays**: Generate edges, yellow_mask, and combined for ANY frame
  - No longer limited to every 30th frame
  - Visualize detected points/edges that led to bad polynomial fits
- **Frame Navigation**: Slider, keyboard controls, and direct frame jump
- **Data Overlays**: Visual overlays for lane lines, trajectory, and reference point
- **Ground Truth Comparison**: Compare detected lanes with ground truth data
- **Data Side Panel**: View all frame data organized by category

### ✅ Phase 2: Recording-Level Analysis (Complete)
- **Recording Summary Tab**: Overall metrics and health graphs ✅
- **Issues Detection & Navigation**: Auto-detect problematic frames and jump to them ✅
- **Trajectory vs Steering Diagnostic**: Identify which component is failing ✅
- **Projection Tab Diagnostics**:
  - First-visible trajectory source distance, turn-sign checks
  - Planner-vs-oracle lateral error at 5m/10m/15m
  - Right-lane fiducial reprojection error (5m/10m/15m + mean/max)
- **Right-Lane Fiducials Overlay**:
  - Unity screen-truth fiducials (`WorldToScreenPoint`) vs current projection
  - Per-point connector lines for quick visual residual checks in front camera

## Setup

### Prerequisites

- Python 3.7+
- Flask and Flask-CORS
- h5py, numpy, Pillow, opencv-python

Install Python dependencies:

```bash
pip install flask flask-cors h5py numpy Pillow opencv-python
```

### Starting the Server

1. Start the Python server:

```bash
cd tools/debug_visualizer
python server.py
```

The server will start on `http://localhost:5001`.

### Opening the Visualizer

1. Open `tools/debug_visualizer/index.html` in a web browser
2. The visualizer will automatically connect to the server

**Note**: Due to browser security restrictions, you may need to serve the HTML file through a local web server. You can use Python's built-in server:

```bash
cd tools/debug_visualizer
python -m http.server 8000
```

Then open `http://localhost:8000/index.html` in your browser.

## Usage

### Loading and Navigating

1. **Load Recording**: 
   - Select a recording from the dropdown
   - Click "Load" to load the recording

2. **Navigate Frames**:
   - Use the frame slider at the bottom
   - Use arrow keys (← →) to move frame by frame
   - Use spacebar to play/pause
   - Type frame number in "Jump to frame" box and click "Go"
   - Adjust playback speed with the speed slider

3. **View Data**:
   - Switch between tabs (Perception, Trajectory, Control, Vehicle, Ground Truth) to view different data
   - All values update automatically as you navigate frames

### Frame-Level Diagnostics

#### Polynomial Inspector
1. Navigate to a frame of interest (e.g., frame 399)
2. Click "Analyze Current Frame" button in the Perception tab
3. View:
   - **Recorded Detection**: What was actually saved in the HDF5 file
   - **Re-Run Detection**: What the current code would detect
   - **Full System Validation**: Whether av_stack.py would ACCEPT or REJECT
   - **Rejection Reasons**: Specific reasons if rejected
   - **Recommendations**: How to fix the issues

#### Debug Overlays
1. Navigate to any frame
2. Click "Generate Debug Overlays (Current Frame)" button
3. Check the "Edges", "Yellow Mask", or "Combined" checkboxes
4. See what points/edges were detected that led to the polynomial fit

### Toggle Overlays

- Use checkboxes to toggle lane lines, trajectory, reference point, and ground truth
- Use debug overlay checkboxes to show edges, masks, histograms
- Adjust opacity of debug overlays with the opacity slider
- Adjust ground truth distance with the dropdown (5.0m to 10.0m)

### Export

- Click "Export Frame" to save the current frame with all overlays as PNG

## Keyboard Shortcuts

- `←` (Left Arrow): Previous frame
- `→` (Right Arrow): Next frame
- `Space`: Play/Pause

## API Endpoints

The server provides the following REST API endpoints:

### Frame Data
- `GET /api/recordings` - List all available recordings
- `GET /api/recording/<filename>/frames` - Get frame count
- `GET /api/recording/<filename>/frame/<index>` - Get frame data as JSON
- `GET /api/recording/<filename>/frame/<index>/image` - Get camera frame as base64

### Debug Tools
- `GET /api/recording/<filename>/frame/<index>/polynomial-analysis` - Analyze polynomial fitting
- `GET /api/recording/<filename>/frame/<index>/generate-debug` - Generate debug overlays on-demand

### Recording Analysis
- `GET /api/recording/<filename>/summary?analyze_to_failure=<true|false>` - Get recording summary metrics
- `GET /api/recording/<filename>/issues?analyze_to_failure=<true|false>` - Get detected issues
- `GET /api/recording/<filename>/diagnostics?analyze_to_failure=<true|false>` - Get trajectory vs steering diagnostics
- `GET /api/recording/<filename>/topdown-diagnostics` - Timing/projection trust diagnostics for top-down trajectory overlay
- `POST /api/recording/<filename>/run-perception-questions` - Run `tools/analyze/analyze_perception_questions.py` and return Q1-Q8 output (used by Summary tab "Run Q Script" button)

### Debug Images (Legacy)
- `GET /api/debug/<filename>/<frame_id>` - Get debug visualization image (if saved)

## File Structure

```
tools/debug_visualizer/
├── index.html              # Main HTML interface
├── visualizer.js           # Core visualization logic
├── data_loader.js          # Data loading utilities
├── overlay_renderer.js     # Overlay rendering functions
├── style.css               # Styling
├── server.py               # Python backend server
├── backend/                # Analysis backend modules (Phase 2)
│   ├── summary_analyzer.py    # Recording summary analysis
│   ├── issue_detector.py       # Issue detection logic
│   └── diagnostics.py         # Trajectory vs steering diagnostics
├── CONSOLIDATION_PLAN.md   # Tool consolidation roadmap
└── README.md               # This file
```

## Troubleshooting

### Server Connection Issues

- Ensure the server is running on port 5001
- Check browser console for CORS errors
- Verify the API_BASE URL in `data_loader.js` matches your server

### Debug Images Not Loading

- Debug images are only saved every 30 frames during recording
- Use "Generate Debug Overlays" button for on-demand generation
- Check browser console for error messages

### Performance Issues

- The visualizer caches frames (up to 50 frames)
- For large recordings, consider loading frames on demand only
- Debug images are loaded lazily when toggled

## Tool Consolidation Status

### ✅ Phase 1: Frame-Level Diagnostics (Complete)
- Polynomial Fitting Inspector
- On-Demand Debug Overlay Generation
- Frame-by-Frame Analysis (integrated into tabs)

### ✅ Phase 2: Recording-Level Analysis (Complete)
- Recording Summary Tab ✅
- Issues Detection & Navigation ✅
- Trajectory vs Steering Diagnostic ✅

### 📋 Phase 3: Advanced Tools (Planned)
- Perception Replay
- Calibration Assistant

See [CONSOLIDATION_PLAN.md](CONSOLIDATION_PLAN.md) for full details.

## Related Tools

For command-line analysis and batch processing, see:
- `tools/analyze/` - Analysis scripts for detailed diagnostics
- `tools/README.md` - Documentation for all tools
