# Unity Setup Guide for AV Stack

The Unity project is already included in this repo at `unity/AVSimulation/`. You do **not** need to create a new project — just install Unity and open the existing one.

## Quick Start (Existing Project)

### 1. Install Unity Hub & Editor

1. Download and install **Unity Hub** from [unity.com](https://unity.com/download)
2. Sign in or create a Unity account
3. In Unity Hub, go to **Installs → Install Editor**
4. Install **Unity 6000.4 LTS** (or the version shown in `unity/AVSimulation/ProjectSettings/ProjectVersion.txt`)
   - Include **Mac Build Support** (or Windows/Linux as appropriate)

### 2. Open the Project

1. In Unity Hub, click **Open → Add project from disk**
2. Navigate to `unity/AVSimulation/` in this repo and select it
3. Unity will import the project (first open may take a few minutes to regenerate the Library)

### 3. Run the Simulation

The easiest way is through the startup script, which builds the player and launches everything:

```bash
./start_av_stack.sh --launch-unity --unity-auto-play --duration 60
```

Or to open the project in the Unity Editor directly:
1. Open the project in Unity Hub
2. Open the scene in `Assets/Scenes/`
3. Press Play

Make sure the Python bridge server is running first:

```bash
python -m bridge.server
```

That's it — you're ready to go.

---

## Optional: Creating a New Unity Project from Scratch

If you want to build a fresh Unity project instead of using the one in the repo (e.g., for experimentation or a different simulation setup), follow the steps below. **This is not required for normal development.**

<details>
<summary>Click to expand full project creation guide</summary>

### Create New Project

1. Open Unity Hub → **New Project**
2. Select **3D (URP)** template (Universal Render Pipeline)
3. Name your project `AVSimulation`
4. Choose a location and click **Create**

### Project Structure

Create the following folder structure in `Assets/`:
```
Assets/
├── Scripts/
├── Scenes/
├── Materials/
├── Prefabs/
└── Textures/
```

### Create Basic Road Scene

**Ground/Road:**
1. Hierarchy → 3D Object → Plane, rename to "Road"
2. Scale: (10, 1, 50), Position: (0, 0, 0)
3. Create a Material "RoadMaterial" — dark gray (RGB: 50, 50, 50), assign to Road

**Lane Markings:**
1. Hierarchy → 3D Object → Cube, rename to "LaneLine"
2. Scale: (0.1, 0.01, 50), Position: (0, 0.01, 0)
3. Create Material "LaneLineMaterial" (white/yellow)
4. Duplicate and position at lane boundaries

**Environment:**
1. Skybox: Window → Rendering → Lighting → Environment → Skybox Material
2. Lighting: Create → Light → Directional Light

### Create Vehicle

**Car Body:**
1. Hierarchy → Create Empty → rename "Vehicle"
2. Add Rigidbody: Mass=1500, Drag=0.3, Angular Drag=3, freeze rotation X/Z
3. Add child Cube "CarBody": Scale (2, 1, 4), Position (0, 0.5, 0)

**Wheels (WheelCollider):**
For each wheel (FrontLeft, FrontRight, BackLeft, BackRight):
- Create Empty GameObject under Vehicle, add Wheel Collider
- Positions: FL (-0.8, 0.3, 1.2), FR (0.8, 0.3, 1.2), BL (-0.8, 0.3, -1.2), BR (0.8, 0.3, -1.2)
- Radius: 0.3, Suspension Distance: 0.3, Spring: 35000, Damper: 4500

**Camera:**
1. Create Empty "CameraMount" under Vehicle, Position (0, 1.2, 0)
2. Add Camera child: FOV 60, Near 0.1, Far 1000

### Configure & Attach Scripts

Copy C# scripts from `unity/Assets/Scripts/` into your new project, then attach:
- **CameraCapture.cs** → MainCamera (API URL: `http://localhost:8000`, rate: 30 FPS)
- **VehicleController.cs** → Vehicle (assign 4 Wheel Colliders, torque: 1500, steering: 30 deg, brake: 3000)
- **AVBridge.cs** → Vehicle (API URL: `http://localhost:8000`, rate: 30 Hz)

### Physics Settings

Edit → Project Settings → Physics: Gravity (0, -9.81, 0), set appropriate friction material.

</details>

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Camera not capturing | Verify CameraCapture script is attached and Python server is running |
| Vehicle physics issues | Check Rigidbody is attached and Wheel Colliders are positioned correctly |
| Communication errors | Verify Python server is accessible at `http://localhost:8000` |
| Unity version mismatch warning | Install the exact version from `ProjectSettings/ProjectVersion.txt` |

