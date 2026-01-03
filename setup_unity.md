# Unity Setup Guide for AV Stack

This guide will walk you through setting up Unity 3D for the autonomous vehicle simulation.

## Prerequisites

1. **Unity Hub**: Download and install from [unity.com](https://unity.com/download)
2. **Unity Editor**: Install Unity 2021.3 LTS or later (recommended: 2022.3 LTS)
   - When installing, include these modules:
     - Windows/Mac/Linux Build Support
     - Visual Studio / Visual Studio Code (for C# scripting)

## Step 1: Create New Unity Project

1. Open Unity Hub
2. Click "New Project"
3. Select "3D (URP)" template (Universal Render Pipeline recommended for better performance)
4. Name your project: `AVSimulation`
5. Choose a location and click "Create"

## Step 2: Project Structure Setup

1. In the Unity Editor, create the following folder structure in the `Assets` folder:
   ```
   Assets/
   ├── Scripts/
   ├── Scenes/
   ├── Materials/
   ├── Prefabs/
   └── Textures/
   ```

## Step 3: Create Basic Road Scene

### 3.1 Create Ground/Road

1. Right-click in Hierarchy → 3D Object → Plane
2. Rename to "Road"
3. Scale: (10, 1, 50) to create a long road
4. Position: (0, 0, 0)
5. Create a Material for the road:
   - Right-click in Materials folder → Create → Material
   - Name it "RoadMaterial"
   - Set color to dark gray (RGB: 50, 50, 50)
   - Assign to Road object

### 3.2 Add Lane Markings

1. Create lane marking lines:
   - Right-click in Hierarchy → 3D Object → Cube
   - Rename to "LaneLine"
   - Scale: (0.1, 0.01, 50)
   - Position: (0, 0.01, 0) for center line
   - Create Material "LaneLineMaterial" (white/yellow)
   - Duplicate and position at lane boundaries

2. Alternative: Use ProBuilder (Window → Package Manager → ProBuilder) for more advanced road creation

### 3.3 Add Environment

1. Add skybox: Window → Rendering → Lighting → Environment → Skybox Material
2. Add lighting: Create → Light → Directional Light
3. Position camera for scene view: Main Camera at (0, 5, -10) looking forward

## Step 4: Create Vehicle

### 4.1 Create Car Body

1. Create empty GameObject: Right-click Hierarchy → Create Empty → Rename "Vehicle"
2. Add Rigidbody component:
   - Select Vehicle → Add Component → Rigidbody
   - Set Mass: 1500 (kg)
   - Set Drag: 0.3
   - Set Angular Drag: 3
   - Freeze Rotation on X and Z axes (constraints)

3. Create car body:
   - Right-click Vehicle → 3D Object → Cube
   - Rename to "CarBody"
   - Scale: (2, 1, 4)
   - Position: (0, 0.5, 0)
   - Add Material "CarMaterial" (any color)

### 4.2 Add Wheels (WheelCollider Method)

1. For each wheel (FrontLeft, FrontRight, BackLeft, BackRight):
   - Create Empty GameObject under Vehicle
   - Add Component → Wheel Collider
   - Position at wheel locations:
     - FrontLeft: (-0.8, 0.3, 1.2)
     - FrontRight: (0.8, 0.3, 1.2)
     - BackLeft: (-0.8, 0.3, -1.2)
     - BackRight: (0.8, 0.3, -1.2)
   - Configure Wheel Collider:
     - Radius: 0.3
     - Suspension Distance: 0.3
     - Spring: 35000
     - Damper: 4500
     - Target Position: 0.5

2. Create visual wheel meshes (optional):
   - Create Cylinder for each wheel
   - Position to match Wheel Collider positions
   - Scale: (0.6, 0.2, 0.6)

### 4.3 Add Camera

1. Create Empty GameObject under Vehicle → Rename "CameraMount"
2. Position: (0, 1.2, 0) - above car, looking forward
3. Create Camera:
   - Right-click CameraMount → Camera
   - Rename to "MainCamera"
   - Position: (0, 0, 0) relative to CameraMount
   - Rotation: (0, 0, 0) - looking forward
   - Field of View: 60
   - Near Plane: 0.1
   - Far Plane: 1000
   - Set as Main Camera

## Step 5: Configure Camera for AV Stack

1. Select MainCamera
2. Set Resolution: 640x480 or 1280x720 (in CameraCapture script)
3. Enable Render Texture (optional for better control):
   - Create → Render Texture
   - Set size: 640x480
   - Assign to Camera's Target Texture

## Step 6: Add Physics Settings

1. Edit → Project Settings → Physics
2. Set Gravity: (0, -9.81, 0)
3. Set Default Material with appropriate friction

## Step 7: Save Scene

1. File → Save Scene As
2. Name: "AVSimulation"
3. Save in Assets/Scenes/

## Step 8: Import C# Scripts

1. Copy the C# scripts from `unity/Assets/Scripts/` to your Unity project's `Assets/Scripts/` folder
2. Unity will automatically compile them

## Step 9: Attach Scripts to GameObjects

1. **CameraCapture.cs**:
   - Attach to MainCamera GameObject
   - Set API URL: `http://localhost:8000` (or your Python server address)
   - Set Capture Rate: 30 (frames per second)

2. **VehicleController.cs**:
   - Attach to Vehicle GameObject
   - Assign all 4 Wheel Colliders in the inspector
   - Set Max Motor Torque: 1500
   - Set Max Steering Angle: 30 degrees
   - Set Brake Torque: 3000

3. **AVBridge.cs**:
   - Attach to Vehicle GameObject
   - Set API URL: `http://localhost:8000`
   - Set Update Rate: 30 Hz

## Step 10: Test Basic Setup

1. Press Play in Unity
2. Verify:
   - Camera is capturing frames
   - Vehicle can be controlled manually (if you add temporary keyboard controls)
   - No console errors

## Step 11: Python Server Setup

Before running Unity, ensure the Python server is running:

```bash
cd /path/to/av
python -m bridge.server
```

The server should start on `http://localhost:8000`

## Troubleshooting

### Camera not capturing
- Check CameraCapture script is attached
- Verify API URL is correct
- Check Python server is running
- Check firewall settings

### Vehicle physics issues
- Ensure Rigidbody is attached
- Check Wheel Colliders are properly positioned
- Verify mass and drag settings are reasonable

### Communication errors
- Verify Python server is accessible
- Check CORS settings in FastAPI server
- Ensure Unity is sending data in correct format

## Next Steps

After setup is complete:
1. Run the Python bridge server
2. Start Unity simulation
3. Begin data collection
4. Train perception models

## Additional Resources

- Unity Manual: https://docs.unity3d.com/Manual/
- Unity Wheel Collider: https://docs.unity3d.com/Manual/class-WheelCollider.html
- Unity Camera: https://docs.unity3d.com/Manual/class-Camera.html

