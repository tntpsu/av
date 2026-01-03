# AV Stack Debug Visualizer

Interactive web-based debug visualizer for analyzing AV stack recordings. Displays synchronized camera frames with all associated data, debug visualizations, and ground truth comparisons.

## Features

- **Frame Navigation**: Slider and keyboard controls to navigate through frames
- **Data Overlays**: Visual overlays for lane lines, trajectory, and reference point
- **Debug Visualizations**: Toggle debug images (edges, masks, histograms) as overlays
- **Ground Truth Comparison**: Compare detected lanes with ground truth data
- **Data Side Panel**: View all frame data organized by category (Perception, Trajectory, Control, Vehicle, Ground Truth)
- **Export**: Export individual frames as PNG images

## Setup

### Prerequisites

- Python 3.7+
- Flask and Flask-CORS
- h5py, numpy, Pillow

Install Python dependencies:

```bash
pip install flask flask-cors h5py numpy Pillow
```

### Starting the Server

1. Start the Python server:

```bash
cd tools/debug_visualizer
python server.py
```

The server will start on `http://localhost:5000`.

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

1. **Load Recording**: 
   - Select a recording from the dropdown
   - Click "Load" to load the recording

2. **Navigate Frames**:
   - Use the frame slider at the bottom
   - Use arrow keys (← →) to move frame by frame
   - Use spacebar to play/pause
   - Adjust playback speed with the speed slider

3. **View Data**:
   - Switch between tabs (Perception, Trajectory, Control, Vehicle, Ground Truth) to view different data
   - All values update automatically as you navigate frames

4. **Toggle Overlays**:
   - Use checkboxes to toggle lane lines, trajectory, reference point, and ground truth
   - Use debug overlay checkboxes to show edges, masks, histograms
   - Adjust opacity of debug overlays with the opacity slider

5. **Export**:
   - Click "Export Frame" to save the current frame with all overlays as PNG

## Keyboard Shortcuts

- `←` (Left Arrow): Previous frame
- `→` (Right Arrow): Next frame
- `Space`: Play/Pause

## API Endpoints

The server provides the following REST API endpoints:

- `GET /api/recordings` - List all available recordings
- `GET /api/recording/<filename>/frames` - Get frame count
- `GET /api/recording/<filename>/frame/<index>` - Get frame data as JSON
- `GET /api/recording/<filename>/frame/<index>/image` - Get camera frame as base64
- `GET /api/debug/<filename>/<frame_id>` - Get debug visualization image

## File Structure

```
tools/debug_visualizer/
├── index.html              # Main HTML interface
├── visualizer.js           # Core visualization logic
├── data_loader.js          # Data loading utilities
├── overlay_renderer.js     # Overlay rendering functions
├── style.css               # Styling
├── server.py               # Python backend server
└── README.md               # This file
```

## Troubleshooting

### Server Connection Issues

- Ensure the server is running on port 5000
- Check browser console for CORS errors
- Verify the API_BASE URL in `data_loader.js` matches your server

### Debug Images Not Loading

- Ensure debug visualization images exist in `tmp/debug_visualizations/`
- Check that frame numbers match (e.g., `frame_000000.png` for frame 0)
- Some debug images may not exist for all frames

### Performance Issues

- The visualizer caches frames (up to 50 frames)
- For large recordings, consider loading frames on demand only
- Debug images are loaded lazily when toggled

## Future Enhancements

- Video export functionality
- Timeline view showing data trends
- Multi-frame comparison
- Filter/search frames by criteria
- Annotations/notes per frame

