# Art Assets (Free Car Model)

This project currently uses a simple blue box mesh for the car. To replace it with
an actual car model, use a free, Unity-friendly asset and swap the visuals in the
car prefab while keeping physics and scripts unchanged.

## Recommended Free Model
- **Kenney Car Kit (CC0)**  
  Download: https://kenney.nl/assets/car-kit

This pack is free and Unity-compatible. The CC0 license allows use without attribution.
Recommended model file: `Models/FBX format/sedan-sports.fbx`.

## Integration Steps
1. Download and unzip the asset.
2. Place the asset under Unity: `unity/AVSimulation/Assets/Art/KenneyCarKit/`.
3. Open `Assets/Prefabs/CarPrefab.prefab`.
4. Replace the current visual mesh (blue box) with the imported car mesh:
   - Keep **CarController**, **AVBridge**, and **Camera** objects unchanged.
   - Keep the **Rigidbody** and **Colliders** (adjust collider size if needed).
5. Ensure the model scale and pivot are correct:
   - Car sits on the ground.
   - Forward axis matches Unity +Z (if needed, rotate the mesh under a parent).
6. If the model has wheels:
   - Parent wheel meshes under existing wheel transforms (if present),
     or keep them as static visuals.

## Notes
- Prefer a low-poly model to keep render cost low.
- Do not change the physics root hierarchy; only swap the visual mesh.
- We keep only the selected car model in-repo to reduce repo size.
