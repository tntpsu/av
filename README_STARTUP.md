# Quick Start Guide

## Starting the AV Stack

### Option 1: Automated Startup (Recommended)

Run the startup script:

```bash
# Basic startup
./start_av_stack.sh

# With Unity auto-launch and auto-play
./start_av_stack.sh --launch-unity --unity-auto-play

# Force kill existing processes on port 8000
./start_av_stack.sh --force --launch-unity --unity-auto-play
```

This script will:
- ✅ Check/create virtual environment
- ✅ Install dependencies (first time only)
- ✅ Start the bridge server (kills existing if `--force` used)
- ✅ Wait for services to be ready
- ✅ Start the AV stack (with data recording enabled by default)
- ✅ Optionally launch Unity Editor (`--launch-unity`)
- ✅ Optionally auto-enter play mode (`--unity-auto-play`)
- ✅ Show you clear instructions for Unity

### Option 2: Manual Startup

If you prefer to start services manually:

**Terminal 1 - Bridge Server:**
```bash
source venv/bin/activate
python -m bridge.server
```

**Terminal 2 - AV Stack:**
```bash
source venv/bin/activate
python av_stack.py
```

## Stopping the AV Stack

```bash
./stop_av_stack.sh
```

Or press `Ctrl+C` in the terminal running the startup script.

## What You'll See

### In Terminal:
- Bridge server starting on `http://localhost:8000`
- AV stack processing frames
- Status messages every 30 frames showing:
  - Number of lanes detected
  - Confidence score
  - Current speed
  - Steering commands

### In Unity:
**If using auto-launch:**
- Unity will automatically open and enter play mode
- No manual steps needed!

**If starting manually:**
1. Open Unity Editor
2. Open `Assets/Scenes/SampleScene.unity`
3. Select your Car GameObject
4. In AV Bridge component → Check "Enable AV Control"
5. Press **▶ PLAY**

The car should start driving automatically!

**Scene Features:**
- Extended 200-unit straight road
- White lane markings (automatically assigned)
- Camera attached to car (AVCamera GameObject)
- RoadGenerator component available for oval track generation

## Troubleshooting

### Port 8000 already in use
The script will ask if you want to kill the existing process. Say 'y' to restart.

### Bridge server won't start
- Check if Python is installed: `python3 --version`
- Check if dependencies are installed: `pip list | grep fastapi`
- Check logs: `/tmp/av_bridge.log`

### Unity can't connect
- Make sure bridge server is running (check terminal)
- Verify API URL in AV Bridge component: `http://localhost:8000`
- Check Unity console for connection errors

### Car not moving
- Make sure "Enable AV Control" is checked in AV Bridge
- Check that Car Controller and Camera Capture are assigned
- Verify Python AV stack is running and processing frames
- Check Unity console for errors
- Verify lane lines are visible (should be white) - if gray, materials may not be assigned

### Lane detection not working
- Verify lane lines have white material assigned (should appear bright white in scene)
- Check camera is positioned correctly on car (AVCamera GameObject)
- Review CV detection logs: `grep "CV fallback" /tmp/av_stack.log`
- Test CV detection on recorded frames: `python perception/test_cv_detection.py`

### Unity not auto-playing
- Check `.unity_autoplay` flag file exists in Unity project root
- Verify `AutoPlayScene.cs` script is in `Assets/Scripts/Editor/`
- Check Unity console for auto-play messages

## Advanced Options

### Startup Script Flags

```bash
# Force kill existing processes on port 8000
./start_av_stack.sh --force

# Launch Unity automatically
./start_av_stack.sh --launch-unity

# Auto-enter play mode in Unity
./start_av_stack.sh --unity-auto-play

# Combine all options
./start_av_stack.sh --force --launch-unity --unity-auto-play
```

### AV Stack Options

```bash
# Use trained model
python av_stack.py --model_path checkpoints/best.pth

# Limit number of frames
python av_stack.py --max_frames 1000

# Disable data recording
python av_stack.py --no-record

# Custom bridge URL
python av_stack.py --bridge_url http://localhost:8000
```

### Generate Oval Road

1. Open Unity Editor
2. Select `RoadGenerator` GameObject in scene
3. In Inspector:
   - Assign `WhiteLaneMarking` to White Lane Material
   - Assign `YellowLaneMarking` to Yellow Lane Material
   - Set `generateOnStart = true`
   - Set `replaceExistingRoad = true` (optional)
4. Enter Play mode - oval road will be generated automatically

### Collect Performance Metrics

```bash
# After running a test, analyze the logs
python tools/collect_metrics.py --log tmp/logs/av_stack.log --output metrics.json
```

This generates comprehensive metrics including lane detection rate, speed statistics, and system performance.


