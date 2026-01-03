# Trajectory Visualization Setup Guide

## Overview
The trajectory visualization shows the planned path in front of the car as a colored line. The line color indicates the lateral error:
- **Green**: Low error (< 0.5m)
- **Yellow**: Moderate error (0.5m - 2.0m)
- **Red**: High error (> 2.0m)

## Setup Instructions

### Option 1: Add to Car GameObject (Recommended)
1. Select the Car GameObject in the Unity hierarchy
2. Click "Add Component" in the Inspector
3. Search for "TrajectoryVisualizer" and add it
4. The component will automatically:
   - Create a LineRenderer for the trajectory line
   - Create a blue sphere for the reference point
   - Create a white sphere for the vehicle position
   - Find the vehicle transform automatically

### Option 2: Create Empty GameObject
1. Create an empty GameObject (right-click in hierarchy → Create Empty)
2. Name it "TrajectoryVisualizer"
3. Add the TrajectoryVisualizer component
4. In the Inspector, assign:
   - **Vehicle Transform**: Drag the Car GameObject here (or leave empty to auto-find)

## Configuration

### Line Appearance
- **Line Width**: 0.2m (default) - Adjust for visibility
- **Good Color**: Green (low error)
- **Warning Color**: Yellow (moderate error)
- **Error Color**: Red (high error)

### Reference Point Marker
- **Size**: 0.3m sphere (blue)
- Shows the target point the controller is trying to reach

### Vehicle Position Marker
- **Size**: 0.3m sphere (white)
- Attached to vehicle, shows current position

## Troubleshooting

### Line Not Visible
1. Check that TrajectoryVisualizer component is enabled
2. Verify the line width is not too small (try 0.3m)
3. Check that trajectory data is being sent (check Python console logs)
4. Verify bridge server is running and accessible

### Line in Wrong Position
- The trajectory points are in vehicle coordinates (relative to car)
- Unity transforms them to world coordinates automatically
- If line appears offset, check vehicle transform assignment

### No Trajectory Data
- Verify Python AV stack is running
- Check bridge server is accessible (http://localhost:8000)
- Check Python console for trajectory data sending logs
- Verify `/api/trajectory` endpoint is working

## How It Works

1. **Python Side** (`av_stack.py`):
   - Computes trajectory points (in vehicle frame)
   - Sends trajectory data to bridge server via `/api/trajectory` POST

2. **Bridge Server** (`bridge/server.py`):
   - Stores trajectory data
   - Serves trajectory data via `/api/trajectory` GET

3. **Unity Side** (`TrajectoryVisualizer.cs`):
   - Fetches trajectory data from bridge (30 FPS)
   - Converts vehicle coordinates to world coordinates
   - Draws line using LineRenderer
   - Updates color based on lateral error

## Coordinate System

- **Python/Vehicle Frame**: 
  - X = lateral (right positive)
  - Y = forward
  - Z = up

- **Unity World Frame**:
  - X = right
  - Y = up
  - Z = forward

The TrajectoryVisualizer automatically transforms vehicle coordinates to world coordinates using the vehicle's transform.

