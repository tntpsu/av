# Mathematical Proof: Why Ground Truth Following Will Work

## Purpose

We're following a **known path** (oval track in Unity) to record perfect drive data. This data will be used to test each part of the AV stack (perception, trajectory planning, control) against ground truth.

## Coordinate System Verification

### Unity Vehicle Coordinate System

From `GroundTruthReporter.cs`:
```csharp
// Vehicle frame: car's forward = +Y, car's right = +X
Vector3 carRight = Vector3.Cross(Vector3.up, carForward).normalized;
float leftXVehicle = Vector3.Dot(toLeftLane, carRight);
float rightXVehicle = Vector3.Dot(toRightLane, carRight);
float lane_center = (leftX + rightX) / 2.0;
```

**Interpretation:**
- `lane_center > 0` = Lane center is to the **RIGHT** of car → Car is **LEFT** of center
- `lane_center < 0` = Lane center is to the **LEFT** of car → Car is **RIGHT** of center
- `lane_center = 0` = Car is at center

### Unity Steering System

From `CarController.cs`:
```csharp
float steerAngle = steerInput * maxSteerAngle;  // steerInput: -1 to 1
Quaternion rotation = Quaternion.Euler(0, steerAngle * Time.fixedDeltaTime, 0);
```

**Interpretation:**
- `steering > 0` (positive) = Turn **RIGHT**
- `steering < 0` (negative) = Turn **LEFT**
- `steering = 0` = Go straight

## Mathematical Proof with Actual Data

### Example 1: Car LEFT of Center (from latest logs)

**Input from logs:**
```
groundTruthLaneCenterX = 7.28m (POSITIVE)
```

**Interpretation:**
- Lane center is 7.28m to the RIGHT of car
- Car is 7.28m to the LEFT of lane center
- **Should steer RIGHT (positive) to correct**

**Current Implementation:**
```python
steering = kp * lane_center
steering = 1.5 * 7.28 = 10.92 (clipped to 1.0)
```
- Result: `steering = 1.0` (POSITIVE) = Turn RIGHT ✓ **CORRECT**

**If we used opposite sign (WRONG):**
```python
steering = -kp * lane_center
steering = -1.5 * 7.28 = -10.92 (clipped to -1.0)
```
- Result: `steering = -1.0` (NEGATIVE) = Turn LEFT ✗ **WRONG** (makes it worse!)

### Example 2: Car RIGHT of Center (from earlier logs)

**Input from logs:**
```
groundTruthLaneCenterX = -2.58m (NEGATIVE)
```

**Interpretation:**
- Lane center is 2.58m to the LEFT of car
- Car is 2.58m to the RIGHT of lane center
- **Should steer LEFT (negative) to correct**

**Current Implementation:**
```python
steering = kp * lane_center
steering = 1.5 * (-2.58) = -3.87 (clipped to -1.0)
```
- Result: `steering = -1.0` (NEGATIVE) = Turn LEFT ✓ **CORRECT**

### Example 3: Car at Center

**Input:**
```
groundTruthLaneCenterX = 0.0m
```

**Current Implementation:**
```python
steering = kp * lane_center
steering = 1.5 * 0.0 = 0.0
```
- Result: `steering = 0.0` = No steering ✓ **CORRECT**

## Step-by-Step Process

### Frame-by-Frame Example

**Initial State:**
- Car position: LEFT of lane center
- `lane_center = 7.28m`

**Frame 1:**
```
lane_center = 7.28m
steering = 1.5 * 7.28 = 10.92 → clipped to 1.0
Car steers RIGHT (maximum)
```

**Frame 2 (after moving right):**
```
lane_center = 6.50m (decreased - car moved closer)
steering = 1.5 * 6.50 = 9.75 → clipped to 1.0
Car continues steering RIGHT
```

**Frame 3:**
```
lane_center = 5.20m (decreased further)
steering = 1.5 * 5.20 = 7.80 → clipped to 1.0
Car continues steering RIGHT
```

**Frame N (near center):**
```
lane_center = 0.10m (very close to center)
steering = 1.5 * 0.10 = 0.15 (small steering)
Car makes small correction
```

**Frame N+1 (at center):**
```
lane_center = 0.00m
steering = 1.5 * 0.00 = 0.00
Car goes straight ✓
```

## Why This Will Work

### 1. Mathematically Correct

The formula `steering = kp * lane_center` is **mathematically correct**:
- Same sign ensures steering direction matches error direction
- Proportional control is a proven control law
- Works for any `lane_center` value (positive, negative, or zero)

### 2. Simple and Predictable

For a **known path** (oval track), we don't need complex control:
- We know exactly where the lane center is
- We just need to steer toward it
- Simple proportional control is sufficient

### 3. Ground Truth Mode

Unity's ground truth mode uses **direct velocity control**:
- No physics forces to fight
- Precise, predictable movement
- Perfect for recording perfect drive data

### 4. Tested Logic

The synthetic test validates the logic:
```python
✓ PASS: Car RIGHT of center (lane_center < 0) → steer LEFT (negative)
✓ PASS: Car LEFT of center (lane_center > 0) → steer RIGHT (positive)
✓ PASS: Car at center (lane_center = 0) → no steering
```

## Expected Behavior

### For a Simple Oval Track

1. **Start:** Car at any position on track
2. **Frame 1:** Calculate `lane_center`, compute `steering = kp * lane_center`
3. **Frame 2:** Car moves toward center, `lane_center` decreases
4. **Frame 3:** Continue correcting, `lane_center` approaches 0
5. **Frame N:** Car at center, `lane_center ≈ 0`, `steering ≈ 0`
6. **Continue:** Car follows path, maintaining `lane_center ≈ 0`

### Success Criteria

- `lane_center` decreases from initial value toward 0
- Car moves toward lane center (not away)
- Car stabilizes near center (`lane_center < 0.5m`)
- Car follows path smoothly

## Why Previous Runs Failed

### Run 1: Wrong Sign (My Mistake)
- Used `steering = -kp * lane_center` (opposite sign)
- Car steered AWAY from center
- Made error worse, not better

### Run 2: Path-Based Steering Interference
- Added heading error term
- May have interfered with lateral correction
- Now disabled - using only lateral error

### Current: Correct Sign, Simple Control
- `steering = kp * lane_center` (same sign) ✓
- Only lateral error (no heading term) ✓
- Should work correctly

## Confidence Level

**HIGH** - The math is correct, the coordinate system is verified, and the logic is simple and proven.

The only remaining question is whether Unity is actually applying the steering correctly, which we can verify by checking:
1. Is ground truth mode active? (Unity console logs)
2. Is steering being received? (Bridge logs)
3. Is steering being applied? (Unity Inspector)

