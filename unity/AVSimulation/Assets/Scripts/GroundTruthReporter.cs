using UnityEngine;

/// <summary>
/// Reports ground truth lane positions for validation and testing.
/// Calculates actual lane center position in world and vehicle coordinates.
/// Supports both straight roads (fixed positions) and curved roads (dynamic from RoadGenerator).
/// </summary>
public class GroundTruthReporter : MonoBehaviour
{
    [Header("Lane Configuration (Straight Road Only)")]
    [Tooltip("Left lane line X position in world coordinates (straight road only)")]
    public float leftLaneXWorld = -3.5f;  // Left edge of road/lane
    
    [Tooltip("Right lane line X position in world coordinates (straight road only)")]
    public float rightLaneXWorld = 3.5f;  // Right edge of road/lane
    
    [Tooltip("Lane width in meters (total road width if single lane)")]
    public float laneWidth = 7.0f;  // Full road width (single lane, no center line yet)
    
    [Header("Current Lane")]
    [Tooltip("Which lane the car is in (0=left, 1=right)")]
    public int currentLane = 1; // Right lane by default
    
    [Header("Road Detection")]
    [Tooltip("Auto-detect RoadGenerator for dynamic ground truth (oval track)")]
    public bool autoDetectRoadGenerator = true;
    
    [Header("Time-Based Reference Path")]
    [Tooltip("Use time-based reference path (moves at constant speed) instead of car position")]
    public bool useTimeBasedReference = true;
    
    [Tooltip("Reference speed for time-based path (m/s)")]
    public float referenceSpeed = 5.0f;

    [Header("Ground Truth Lookahead")]
    [Tooltip("Lookahead distance (m) used for ground truth lane positions and road-center debug")]
    public float groundTruthLookaheadDistance = 8.0f;
    
    private Transform carTransform;
    private Camera avCamera;  // NEW: Camera reference for calculating ground truth from camera position
    private RoadGenerator roadGenerator;
    private bool useDynamicGroundTruth = false;
    private float pathStartTime = 0f;
    private float ovalPathLength = 0f;
    
    // PERFORMANCE: Cache Rigidbody reference to avoid GetComponent() calls every frame
    private Rigidbody cachedCarRigidbody = null;
    
    void Start()
    {
        // If attached to car, use this transform directly
        // Otherwise, find the car GameObject
        if (transform.name.Contains("Car") || GetComponent<CarController>() != null)
        {
            // This component is on the car itself
            carTransform = transform;
            Debug.Log($"GroundTruthReporter.Start: Using attached transform '{transform.name}'");
        }
        else
        {
            // Try multiple ways to find the car
            GameObject car = GameObject.Find("CarPrefab");
            if (car == null)
            {
                car = GameObject.Find("Car");
            }
            if (car == null)
            {
                // Try finding by component
                CarController carController = FindObjectOfType<CarController>();
                if (carController != null)
                {
                    car = carController.gameObject;
                }
            }
            
            if (car != null)
            {
                carTransform = car.transform;
                Debug.Log($"GroundTruthReporter.Start: Found car '{car.name}' at position {carTransform.position}");
            }
            else
            {
                Debug.LogError("GroundTruthReporter.Start: Could not find car! " +
                             $"Transform name: '{transform.name}', " +
                             $"Has CarController: {GetComponent<CarController>() != null}");
            }
        }
        
        // NEW: Find camera for calculating ground truth from camera position (matches perception)
        // Camera is typically a child of the car, named "AVCamera"
        if (carTransform != null)
        {
            Transform cameraTransform = carTransform.Find("AVCamera");
            if (cameraTransform != null)
            {
                avCamera = cameraTransform.GetComponent<Camera>();
                if (avCamera != null)
                {
                    Debug.Log($"GroundTruthReporter.Start: Found camera '{avCamera.name}' at local position {cameraTransform.localPosition}");
                }
            }
            // Fallback: search by name
            if (avCamera == null)
            {
                GameObject cameraObj = GameObject.Find("AVCamera");
                if (cameraObj != null)
                {
                    avCamera = cameraObj.GetComponent<Camera>();
                    if (avCamera != null)
                    {
                        Debug.Log($"GroundTruthReporter.Start: Found camera '{avCamera.name}' by name");
                    }
                }
            }
            if (avCamera == null)
            {
                Debug.LogWarning("GroundTruthReporter.Start: Camera not found! Ground truth will use car position instead of camera position.");
            }
        }
        
        // PERFORMANCE: Initialize cached Rigidbody in Start() to ensure it's always available
        // This prevents null reference issues on first frame
        if (carTransform != null && cachedCarRigidbody == null)
        {
            cachedCarRigidbody = carTransform.GetComponent<Rigidbody>();
            if (cachedCarRigidbody == null)
            {
                Debug.LogWarning("GroundTruthReporter.Start: Could not find Rigidbody on carTransform. Some features may not work correctly.");
            }
        }
        
        // Try to find RoadGenerator for dynamic ground truth (oval track)
        if (autoDetectRoadGenerator)
        {
            roadGenerator = FindObjectOfType<RoadGenerator>();
            if (roadGenerator != null)
            {
                useDynamicGroundTruth = true;
                laneWidth = roadGenerator.roadWidth; // Use road width from RoadGenerator
                
                // Calculate oval path length for time-based reference
                CalculateOvalPathLength();
                
                // Initialize time-based reference
                if (useTimeBasedReference)
                {
                    // Optional: Override start position from track config
                    if (roadGenerator.TryGetStartT(out float overrideStartT))
                    {
                        Vector3 startPos = roadGenerator.GetOvalCenterPoint(overrideStartT);
                        Vector3 startDir = roadGenerator.GetOvalDirection(overrideStartT);

                        if (carTransform != null)
                        {
                            if (cachedCarRigidbody != null)
                            {
                                cachedCarRigidbody.position = startPos;
                                cachedCarRigidbody.rotation = Quaternion.LookRotation(startDir, Vector3.up);
                                cachedCarRigidbody.velocity = Vector3.zero;
                                cachedCarRigidbody.angularVelocity = Vector3.zero;
                            }
                            else
                            {
                                carTransform.position = startPos;
                                carTransform.rotation = Quaternion.LookRotation(startDir, Vector3.up);
                            }
                        }

                        float startDistance = GetDistanceAlongPath(0f, overrideStartT);
                        pathStartTime = Time.time - (startDistance / referenceSpeed);
                        Debug.Log($"GroundTruthReporter: Using start override t={overrideStartT:F3}, distance={startDistance:F2}m");
                    }
                    else
                    {
                    // CRITICAL FIX: Initialize path reference based on car's starting position
                    // This prevents the car from jumping to path start (t=0) when ground truth mode activates
                    if (carTransform != null)
                    {
                        Vector3 carStartPos;
                        // PERFORMANCE: Use cached Rigidbody (already initialized above)
                        if (cachedCarRigidbody != null)
                        {
                            carStartPos = cachedCarRigidbody.position;
                        }
                        else
                        {
                            carStartPos = carTransform.position;
                        }
                        
                        // Find closest point on path to car's starting position
                        float startT = FindClosestPointOnOval(carStartPos);
                        
                        // Calculate distance from path start (t=0) to car's starting position
                        float startDistance = GetDistanceAlongPath(0f, startT);
                        
                        // Set pathStartTime so that at Time.time, we're at the car's starting position
                        // elapsedTime = 0 when ground truth mode starts
                        // distanceTraveled = referenceSpeed * 0 = 0
                        // We want: normalizedDistance = startDistance
                        // So we need: pathStartTime = Time.time - (startDistance / referenceSpeed)
                        pathStartTime = Time.time - (startDistance / referenceSpeed);
                        
                        // Verify distance calculation matches
                        float verifyDistance = GetDistanceAlongPath(0f, startT);
                        Debug.Log($"GroundTruthReporter: Using TIME-BASED reference path at {referenceSpeed}m/s. " +
                                 $"Path length: {ovalPathLength}m. " +
                                 $"Car starts at t={startT:F3}, distance={startDistance:F2}m from path start (verified: {verifyDistance:F2}m). " +
                                 $"Adjusted pathStartTime to prevent jump.");
                    }
                    else
                    {
                        // Fallback: start at path beginning
                        pathStartTime = Time.time;
                        Debug.Log($"GroundTruthReporter: Using TIME-BASED reference path at {referenceSpeed}m/s. Path length: {ovalPathLength}m. " +
                                 $"WARNING: carTransform not found, starting at path beginning (t=0)");
                    }
                    }
                }
                else
                {
                    Debug.Log($"GroundTruthReporter: Using CAR POSITION-based reference path");
                }
                
                Debug.Log($"GroundTruthReporter: Found RoadGenerator - using DYNAMIC ground truth for oval track. Road width: {laneWidth}m");
            }
            else
            {
                useDynamicGroundTruth = false;
                Debug.Log($"GroundTruthReporter: No RoadGenerator found - using STATIC ground truth (straight road). " +
                         $"leftLaneXWorld={leftLaneXWorld}m, rightLaneXWorld={rightLaneXWorld}m, laneWidth={laneWidth}m");
            }
        }
        else
        {
            useDynamicGroundTruth = false;
            Debug.Log($"GroundTruthReporter: Auto-detect disabled - using STATIC ground truth. " +
                     $"leftLaneXWorld={leftLaneXWorld}m, rightLaneXWorld={rightLaneXWorld}m, laneWidth={laneWidth}m");
        }
    }
    
    /// <summary>
    /// Calculate the total length of the oval path.
    /// </summary>
    void CalculateOvalPathLength()
    {
        if (roadGenerator == null) return;
        
        // Sample the path to calculate approximate length
        int samples = 1000;
        float totalLength = 0f;
        Vector3 prevPoint = roadGenerator.GetOvalCenterPoint(0f);
        
        for (int i = 1; i <= samples; i++)
        {
            float t = (float)i / samples;
            Vector3 currentPoint = roadGenerator.GetOvalCenterPoint(t);
            totalLength += Vector3.Distance(prevPoint, currentPoint);
            prevPoint = currentPoint;
        }
        
        ovalPathLength = totalLength;
    }
    
    /// <summary>
    /// Get time-based reference point on oval path.
    /// Returns parameter t (0-1) based on elapsed time and reference speed.
    /// </summary>
    public float GetTimeBasedReferenceT()
    {
        if (ovalPathLength <= 0f) return 0f;
        
        float elapsedTime = Time.time - pathStartTime;
        float distanceTraveled = referenceSpeed * elapsedTime;
        
        // Wrap around if we've completed a full lap
        // Use Mathf.Repeat to handle wrapping correctly (handles negative values too)
        float normalizedDistance = Mathf.Repeat(distanceTraveled, ovalPathLength);
        
        // Convert distance to parameter t (0-1)
        float t = FindTFromDistance(normalizedDistance);
        
        // DEBUG: Log if t is stuck or if we're near problematic areas
        if (Time.frameCount % 60 == 0)  // Every ~2 seconds
        {
            // Check if t is near segment boundaries where issues might occur
            bool nearBoundary = (Mathf.Abs(t - 0.5f) < 0.02f) || (Mathf.Abs(t - 0.25f) < 0.02f) || 
                               (Mathf.Abs(t - 0.75f) < 0.02f) || (t > 0.98f) || (t < 0.02f);
            
            if (nearBoundary)
            {
                Debug.LogWarning($"GroundTruthReporter: GetTimeBasedReferenceT - near boundary: " +
                               $"t={t:F4}, normalizedDistance={normalizedDistance:F2}m, " +
                               $"elapsedTime={elapsedTime:F2}s, distanceTraveled={distanceTraveled:F2}m");
            }
        }
        
        return t;
    }
    
    /// <summary>
    /// Get the current reference position and direction on the path for direct car positioning.
    /// Returns (position, direction) in world coordinates.
    /// </summary>
    public (Vector3 position, Vector3 direction) GetCurrentReferencePath()
    {
        if (roadGenerator == null)
        {
            return (Vector3.zero, Vector3.forward);
        }
        
        float t;
        if (useTimeBasedReference && useDynamicGroundTruth)
        {
            t = GetTimeBasedReferenceT();
            
            // Debug: Log if t is stuck or invalid (every ~2 seconds)
            if (Time.frameCount % 60 == 0)
            {
                float elapsedTime = Time.time - pathStartTime;
                float distanceTraveled = referenceSpeed * elapsedTime;
                float normalizedDistance = Mathf.Repeat(distanceTraveled, ovalPathLength);  // FIX: Use Mathf.Repeat, not %
                Debug.Log($"GroundTruthReporter: Time-based reference - elapsedTime={elapsedTime:F2}s, " +
                         $"distanceTraveled={distanceTraveled:F2}m, normalizedDistance={normalizedDistance:F2}m, " +
                         $"t={t:F4}, pathLength={ovalPathLength:F2}m");
            }
        }
        else
        {
            // Fallback to closest point
            Vector3 carPos;
            // PERFORMANCE: Use cached Rigidbody instead of GetComponent() every frame
            if (cachedCarRigidbody != null)
            {
                carPos = cachedCarRigidbody.position;
            }
            else if (carTransform != null)
            {
                // Cache Rigidbody if not already cached
                if (cachedCarRigidbody == null)
                {
                    cachedCarRigidbody = carTransform.GetComponent<Rigidbody>();
                }
                if (cachedCarRigidbody != null)
                {
                    carPos = cachedCarRigidbody.position;
                }
                else
                {
                    carPos = carTransform.position;
                }
            }
            else
            {
                Debug.LogWarning("GroundTruthReporter: GetCurrentReferencePath - carTransform is null, returning zero");
                return (Vector3.zero, Vector3.forward);
            }
            t = FindClosestPointOnOval(carPos);
        }
        
        Vector3 position = roadGenerator.GetOvalCenterPoint(t);
        Vector3 direction = roadGenerator.GetOvalDirection(t);
        
        // Validate direction vector
        if (direction.sqrMagnitude < 0.01f)
        {
            // Direction is zero or near-zero - this is a problem!
            if (Time.frameCount % 60 == 0) // Log every ~2 seconds
            {
                Debug.LogError($"GroundTruthReporter: GetOvalDirection returned zero vector! " +
                             $"t={t:F4}, position={position}, direction={direction}, " +
                             $"sqrMagnitude={direction.sqrMagnitude}");
            }
            // Return forward direction as fallback
            direction = Vector3.forward;
        }
        
        return (position, direction);
    }

    /// <summary>
    /// Reset time-based reference to the car's current position.
    /// This prevents large jumps when ground truth mode is enabled after a delay.
    /// </summary>
    public bool ResetTimeBasedReferenceToCar(string reason = null)
    {
        if (!useTimeBasedReference || !useDynamicGroundTruth || roadGenerator == null)
        {
            return false;
        }
        
        Vector3 carPos;
        if (cachedCarRigidbody != null)
        {
            carPos = cachedCarRigidbody.position;
        }
        else if (carTransform != null)
        {
            cachedCarRigidbody = carTransform.GetComponent<Rigidbody>();
            carPos = cachedCarRigidbody != null ? cachedCarRigidbody.position : carTransform.position;
        }
        else
        {
            Debug.LogWarning("GroundTruthReporter: ResetTimeBasedReferenceToCar - carTransform is null");
            return false;
        }
        
        float startT = FindClosestPointOnOval(carPos);
        float startDistance = GetDistanceAlongPath(0f, startT);
        if (referenceSpeed > 0.01f)
        {
            pathStartTime = Time.time - (startDistance / referenceSpeed);
        }
        else
        {
            pathStartTime = Time.time;
        }
        
        string reasonText = string.IsNullOrWhiteSpace(reason) ? "" : $" Reason: {reason}.";
        Debug.Log($"GroundTruthReporter: Reset time-based reference to car position at t={startT:F3}, " +
                  $"distance={startDistance:F2}m.{reasonText}");
        return true;
    }

    /// <summary>
    /// Move the car to a random point on the oval and align its heading.
    /// </summary>
    public bool SetCarToRandomStart(int seed = -1)
    {
        if (!useDynamicGroundTruth || roadGenerator == null || carTransform == null)
        {
            Debug.LogWarning("GroundTruthReporter: Random start requires dynamic ground truth and car transform.");
            return false;
        }

        if (seed >= 0)
        {
            UnityEngine.Random.InitState(seed);
        }

        float t = UnityEngine.Random.value;
        Vector3 referencePosition = roadGenerator.GetOvalCenterPoint(t);
        Vector3 referenceDirection = roadGenerator.GetOvalDirection(t);
        if (referenceDirection.sqrMagnitude < 0.01f)
        {
            referenceDirection = Vector3.forward;
        }

        Vector3 currentPos = carTransform.position;
        referencePosition.y = currentPos.y;  // Keep car height consistent
        Quaternion targetRotation = Quaternion.LookRotation(referenceDirection.normalized, Vector3.up);

        if (cachedCarRigidbody == null)
        {
            cachedCarRigidbody = carTransform.GetComponent<Rigidbody>();
        }

        if (cachedCarRigidbody != null)
        {
            cachedCarRigidbody.position = referencePosition;
            cachedCarRigidbody.rotation = targetRotation;
            cachedCarRigidbody.linearVelocity = Vector3.zero;
            cachedCarRigidbody.angularVelocity = Vector3.zero;
        }
        else
        {
            carTransform.position = referencePosition;
            carTransform.rotation = targetRotation;
        }

        ResetTimeBasedReferenceToCar("Random start");
        Debug.Log($"GroundTruthReporter: Random start set to t={t:F3}, position={referencePosition}, direction={referenceDirection}");
        return true;
    }
    
    /// <summary>
    /// Find parameter t (0-1) that corresponds to given distance along path.
    /// Uses binary search for accuracy.
    /// </summary>
    float FindTFromDistance(float targetDistance)
    {
        if (roadGenerator == null) return 0f;
        
        // Handle edge cases
        if (targetDistance < 0f)
        {
            // Negative distance is invalid - clamp to 0
            targetDistance = 0f;
        }
        
        // Special case: distance = 0 means t = 0 (start of path)
        if (targetDistance <= 0.001f) // Use small epsilon to handle floating point precision
        {
            return 0f;
        }
        
        // Use ovalPathLength directly (now that GetDistanceAlongPath uses matching sampling)
        // Both CalculateOvalPathLength() and GetDistanceAlongPath(0f, 1f) should now match
        float actualPathLength = ovalPathLength;
        
        // Handle case when targetDistance is very close to or exceeds actual path length
        float epsilon = 0.01f; // 1cm tolerance for path length comparison
        if (targetDistance >= (actualPathLength - epsilon))
        {
            // Very close to end - use direct calculation to avoid binary search issues
            // Try a value slightly before 1.0 to avoid wrap issues
            float testT = 0.999f;
            float distanceAtTestT = GetDistanceAlongPath(0f, testT);
            if (Mathf.Abs(distanceAtTestT - targetDistance) < epsilon)
            {
                return testT;
            }
            // If we're at or past the end, wrap to start (but Mathf.Repeat should prevent this)
            // For safety, clamp to just before 1.0
            return 0.999f;
        }
        
        // Binary search for t
        float low = 0f;
        float high = 1f;
        float tolerance = 0.01f; // 1cm accuracy (relaxed from 1mm for better convergence)
        
        for (int i = 0; i < 30; i++) // Increased to 30 iterations for better accuracy
        {
            float mid = (low + high) * 0.5f;
            float distanceAtMid = GetDistanceAlongPath(0f, mid);
            float error = distanceAtMid - targetDistance;
            
            if (Mathf.Abs(error) < tolerance)
            {
                return mid;
            }
            
            if (error < 0f)  // distanceAtMid < targetDistance
            {
                low = mid;
            }
            else  // distanceAtMid > targetDistance
            {
                high = mid;
            }
        }
        
        // Binary search didn't converge within tolerance - return midpoint and log warning
        float result = (low + high) * 0.5f;
        float finalDistance = GetDistanceAlongPath(0f, result);
        float finalError = Mathf.Abs(finalDistance - targetDistance);
        
        // Only log if error is significant (avoid spam)
        if (finalError > 0.1f && Time.frameCount % 300 == 0)  // Every ~10 seconds
        {
            Debug.LogWarning($"GroundTruthReporter: FindTFromDistance binary search didn't converge. " +
                           $"targetDistance={targetDistance:F2}m, result_t={result:F4}, " +
                           $"actualDistance={finalDistance:F2}m, error={finalError:F2}m");
        }
        
        return result;
    }
    
    /// <summary>
    /// Get distance along path from t0 to t1.
    /// </summary>
    float GetDistanceAlongPath(float t0, float t1)
    {
        if (roadGenerator == null) return 0f;
        
        // CRITICAL FIX: Always use FIXED 1000 samples for consistency
        // This ensures GetDistanceAlongPath(0f, t) always uses the same precision
        // regardless of t value, which is critical for binary search convergence.
        // Variable sampling causes inconsistent distance calculations that break
        // binary search (e.g., GetDistanceAlongPath(0f, 0.5) uses 500 samples but
        // GetDistanceAlongPath(0f, 0.25) uses 250 samples = inconsistent precision).
        const int SAMPLES = 1000;  // Always use 1000 samples, matching CalculateOvalPathLength()
        
        float distance = 0f;
        Vector3 prevPoint = roadGenerator.GetOvalCenterPoint(t0);
        
        for (int i = 1; i <= SAMPLES; i++)
        {
            float t = Mathf.Lerp(t0, t1, (float)i / SAMPLES);
            Vector3 currentPoint = roadGenerator.GetOvalCenterPoint(t);
            distance += Vector3.Distance(prevPoint, currentPoint);
            prevPoint = currentPoint;
        }
        
        return distance;
    }
    
    /// <summary>
    /// Find the closest point on the oval track to the car's position.
    /// Returns the parameter t (0-1) for that point.
    /// </summary>
    float FindClosestPointOnOval(Vector3 carPosition)
    {
        if (roadGenerator == null) return 0f;
        
        // Sample points around the oval to find closest
        float bestT = 0f;
        float bestDistance = float.MaxValue;
        
        // Sample at higher resolution for accuracy
        int samples = 200;
        for (int i = 0; i < samples; i++)
        {
            float t = (float)i / samples;
            Vector3 point = roadGenerator.GetOvalCenterPoint(t);
            float distance = Vector3.Distance(carPosition, point);
            
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestT = t;
            }
        }
        
        return bestT;
    }
    
    /// <summary>
    /// Get lane positions from RoadGenerator (dynamic, for oval track).
    /// </summary>
    (float leftX, float rightX) GetLanePositionsFromRoadGenerator()
    {
        if (roadGenerator == null || carTransform == null)
        {
            return (0f, 0f);
        }
        
        // CRITICAL FIX: Use time-based reference path instead of car position
        // This creates a "virtual leader" that moves at constant speed, independent of car position
        float t;
        
        // Get car position for coordinate conversion (always needed)
        Vector3 carPos;
        // PERFORMANCE: Use cached Rigidbody
        if (cachedCarRigidbody == null)
        {
            cachedCarRigidbody = carTransform.GetComponent<Rigidbody>();
        }
        if (cachedCarRigidbody != null)
        {
            carPos = cachedCarRigidbody.position;  // Use Rigidbody position (more accurate for physics)
        }
        else
        {
            carPos = carTransform.position;  // Fallback to transform position
        }
        
        if (useTimeBasedReference && useDynamicGroundTruth)
        {
            // Time-based reference: moves at constant speed around oval
            t = GetTimeBasedReferenceT();
        }
        else
        {
            // Car position-based: finds closest point to car (old behavior)
            // Find closest point on oval track
            t = FindClosestPointOnOval(carPos);
        }
        
        // Get road center and direction at that point
        Vector3 roadCenter = roadGenerator.GetOvalCenterPoint(t);
        Vector3 direction = roadGenerator.GetOvalDirection(t);
        Vector3 roadRight = Vector3.Cross(Vector3.up, direction).normalized;
        
        // Compute left and right lane positions in world space
        float halfWidth = roadGenerator.roadWidth * 0.5f;
        Vector3 leftLaneWorld = roadCenter - roadRight * halfWidth;
        Vector3 rightLaneWorld = roadCenter + roadRight * halfWidth;
        
        // Convert to vehicle coordinates (lateral offset)
        // Vehicle frame: car's forward = +Y, car's right = +X
        // Project world positions onto car's local coordinate system
        Vector3 carForward = carTransform.forward;
        Vector3 carRight = Vector3.Cross(Vector3.up, carForward).normalized;
        
        // Compute lateral offset from car to each lane
        Vector3 toLeftLane = leftLaneWorld - carPos;
        Vector3 toRightLane = rightLaneWorld - carPos;
        
        // Project onto car's right vector (lateral direction)
        float leftXVehicle = Vector3.Dot(toLeftLane, carRight);
        float rightXVehicle = Vector3.Dot(toRightLane, carRight);
        
        return (leftXVehicle, rightXVehicle);
    }
    
    /// <summary>
    /// Get ground truth lane center position in world coordinates.
    /// </summary>
    /// <param name="laneIndex">0 for left lane, 1 for right lane</param>
    /// <returns>Lane center X position in world coordinates</returns>
    public float GetLaneCenterWorld(int laneIndex)
    {
        if (laneIndex == 0)
        {
            // Left lane center = left line + half lane width
            return leftLaneXWorld + (laneWidth / 2.0f);
        }
        else
        {
            // Right lane center = right line - half lane width
            return rightLaneXWorld - (laneWidth / 2.0f);
        }
    }
    
    /// <summary>
    /// Get ground truth lane center position in vehicle coordinates.
    /// </summary>
    /// <param name="laneIndex">0 for left lane, 1 for right lane</param>
    /// <returns>Lane center X position in vehicle coordinates (relative to car)</returns>
    public float GetLaneCenterVehicle(int laneIndex)
    {
        if (carTransform == null)
        {
            return 0f;
        }
        
        float laneCenterWorld = GetLaneCenterWorld(laneIndex);
        float carXWorld = carTransform.position.x;
        
        // Vehicle coordinates: relative to car position
        // For straight road, just subtract car X
        // For curved roads, would need to account for heading
        return laneCenterWorld - carXWorld;
    }
    
    /// <summary>
    /// Get ground truth lane positions at a lookahead distance ahead of the car.
    /// This is used for comparison with perception, which evaluates lanes at a lookahead distance.
    /// 
    /// CRITICAL FIX: Uses STRAIGHT-AHEAD distance (not curved path distance) to match perception.
    /// Perception evaluates at 8m straight ahead in camera view, so ground truth should do the same.
    /// </summary>
    /// <param name="lookaheadDistance">Distance ahead in meters (default 8.0m to match perception)</param>
    /// <returns>Tuple of (left_lane_line_x, right_lane_line_x) in vehicle coordinates at lookahead distance</returns>
    public (float leftX, float rightX) GetLanePositionsAtLookahead(float lookaheadDistance = 8.0f)
    {
        if (roadGenerator == null || carTransform == null)
        {
            // Fallback to car position if road generator not available
            return GetLanePositionsVehicle();
        }
        
        // CRITICAL FIX: Calculate ground truth from CAMERA position (not car center)
        // This matches perception, which evaluates lanes from the camera's perspective
        // Camera LOCAL POSITION (mounting): Logged at Start() - check Unity console for actual values
        // Camera FORWARD DIRECTION: Typically (0, negative, 0) or (0, negative, small) - looking DOWN at road
        // NOTE: forward.z = 0 means camera is looking straight down or mostly down (correct for road camera!)
        Vector3 referencePos;
        Vector3 referenceForward;
        
        if (avCamera != null)
        {
            // Use camera position and forward direction (matches perception)
            referencePos = avCamera.transform.position;
            referenceForward = avCamera.transform.forward;
        }
        else if (useTimeBasedReference && useDynamicGroundTruth)
        {
            // Fallback: Use time-based reference path (virtual leader)
            var (refPos, refDir) = GetCurrentReferencePath();
            referencePos = refPos;
            referenceForward = refDir.normalized;
        }
        else
        {
            // Fallback: Use car's current position and heading (old behavior)
            // PERFORMANCE: Use cached Rigidbody
            if (cachedCarRigidbody == null)
            {
                cachedCarRigidbody = carTransform.GetComponent<Rigidbody>();
            }
            if (cachedCarRigidbody != null)
            {
                referencePos = cachedCarRigidbody.position;
            }
            else
            {
                referencePos = carTransform.position;
            }
            referenceForward = carTransform.forward;
        }
        
        // Get car position for coordinate conversion (always needed for vehicle coords)
        Vector3 carPos;
        // PERFORMANCE: Use cached Rigidbody
        if (cachedCarRigidbody == null)
        {
            cachedCarRigidbody = carTransform.GetComponent<Rigidbody>();
        }
        if (cachedCarRigidbody != null)
        {
            carPos = cachedCarRigidbody.position;
        }
        else
        {
            carPos = carTransform.position;
        }
        
        // CRITICAL FIX: Calculate point EXACTLY 8m away from camera (not projected point)
        // The issue: When we project 8m straight ahead onto a curved road, the road center
        // at that point might not be exactly 8.0m away (e.g., 7.885m). This causes the visualizer
        // to convert using 8.0m but ground truth was calculated at 7.885m, causing misalignment.
        //
        // Solution: Find the point on the road center line that IS exactly 8.0m away from camera.
        // This ensures ground truth and perception use the same distance (8.0m).
        Vector3 straightAheadPoint = referencePos + referenceForward * lookaheadDistance;
        
        // Find closest point on road to get initial estimate
        float tLookahead = FindClosestPointOnOval(straightAheadPoint);
        Vector3 roadCenterAtClosest = roadGenerator.GetOvalCenterPoint(tLookahead);
        Vector3 directionLookahead = roadGenerator.GetOvalDirection(tLookahead);
        
        // Iteratively find the point on road center that is exactly lookaheadDistance away
        // Use binary search or iterative refinement to find exact distance
        Vector3 roadCenterAtStraightAhead = roadCenterAtClosest;
        float currentDistance = Vector3.Distance(referencePos, roadCenterAtStraightAhead);
        const float distanceTolerance = 0.01f; // 1cm tolerance
        const int maxIterations = 10;
        
        for (int i = 0; i < maxIterations && Mathf.Abs(currentDistance - lookaheadDistance) > distanceTolerance; i++)
        {
            // Calculate how far along the road direction we need to move
            float distanceError = lookaheadDistance - currentDistance;
            
            // Move along road direction to correct distance
            // If too close, move forward; if too far, move backward
            Vector3 correction = directionLookahead * distanceError;
            roadCenterAtStraightAhead = roadCenterAtStraightAhead + correction;
            
            // Recalculate distance
            currentDistance = Vector3.Distance(referencePos, roadCenterAtStraightAhead);
            
            // Update road direction at new position (for next iteration)
            float tNew = FindClosestPointOnOval(roadCenterAtStraightAhead);
            directionLookahead = roadGenerator.GetOvalDirection(tNew);
        }
        
        // Final road direction at the exact 8m point
        float tFinal = FindClosestPointOnOval(roadCenterAtStraightAhead);
        directionLookahead = roadGenerator.GetOvalDirection(tFinal);
        
        // CRITICAL FIX: Use camera's coordinate system for BOTH calculation and conversion!
        // Perception uses camera's coordinate system, so ground truth must match.
        // On curves, roadRightLookahead ≠ camera right, causing width compression.
        // Solution: Calculate lane positions using camera's right vector, not roadRightLookahead.
        Vector3 coordReferencePos = referencePos; // Use camera position (or time-based reference)
        // Use camera's right vector (matches perception's coordinate system)
        // Camera right = Cross(up, camera forward) = Cross(up, referenceForward)
        Vector3 coordReferenceRight = Vector3.Cross(Vector3.up, referenceForward).normalized;
        
        // Calculate lane positions using camera's coordinate system
        // This ensures width is preserved when converting to vehicle coordinates
        float halfWidth = roadGenerator.roadWidth * 0.5f;
        Vector3 leftLaneWorld = roadCenterAtStraightAhead - coordReferenceRight * halfWidth;
        Vector3 rightLaneWorld = roadCenterAtStraightAhead + coordReferenceRight * halfWidth;
        
        // Convert to vehicle coordinates (lateral offset from camera, not car)
        // CRITICAL FIX: Use camera position for coordinate conversion to match perception
        // Perception evaluates lanes from camera's perspective, so ground truth should too
        // This ensures green lines align with red lines in the visualizer
        // 
        // Since we already calculated lane positions using camera's right vector,
        // the conversion is now just a simple projection (width is preserved)
        Vector3 toLeftLane = leftLaneWorld - coordReferencePos;
        Vector3 toRightLane = rightLaneWorld - coordReferencePos;
        
        // Project onto camera's right vector (lateral direction) - matches perception's coordinate system
        // This gives us vehicle coordinates relative to camera, matching perception
        // Width is preserved because we used camera's right vector for calculation
        float leftXVehicle = Vector3.Dot(toLeftLane, coordReferenceRight);
        float rightXVehicle = Vector3.Dot(toRightLane, coordReferenceRight);
        
        return (leftXVehicle, rightXVehicle);
    }
    
    /// <summary>
    /// Get debug information about road center positions for diagnosing offset issues.
    /// Returns road center at car's location and at lookahead distance.
    /// </summary>
    public (Vector3 roadCenterAtCar, Vector3 roadCenterAtLookahead, float referenceT) GetRoadCenterDebugInfo(float lookaheadDistance = 8.0f)
    {
        if (roadGenerator == null || carTransform == null)
        {
            return (Vector3.zero, Vector3.zero, 0.0f);
        }
        
        // Get car position
        Vector3 carPos;
        // PERFORMANCE: Use cached Rigidbody
        if (cachedCarRigidbody == null)
        {
            cachedCarRigidbody = carTransform.GetComponent<Rigidbody>();
        }
        if (cachedCarRigidbody != null)
        {
            carPos = cachedCarRigidbody.position;
        }
        else
        {
            carPos = carTransform.position;
        }
        
        // Get road center at car's location
        float tCar = FindClosestPointOnOval(carPos);
        Vector3 roadCenterAtCar = roadGenerator.GetOvalCenterPoint(tCar);
        
        // Get road center at lookahead (same calculation as GetLanePositionsAtLookahead)
        Vector3 referencePos;
        Vector3 referenceForward;
        
        if (avCamera != null)
        {
            referencePos = avCamera.transform.position;
            referenceForward = avCamera.transform.forward;
        }
        else if (useTimeBasedReference && useDynamicGroundTruth)
        {
            var (refPos, refDir) = GetCurrentReferencePath();
            referencePos = refPos;
            referenceForward = refDir.normalized;
        }
        else
        {
            referencePos = carPos;
            referenceForward = carTransform.forward;
        }
        
        Vector3 straightAheadPoint = referencePos + referenceForward * lookaheadDistance;
        float tLookahead = FindClosestPointOnOval(straightAheadPoint);
        Vector3 roadCenterAtClosest = roadGenerator.GetOvalCenterPoint(tLookahead);
        Vector3 directionLookahead = roadGenerator.GetOvalDirection(tLookahead);
        Vector3 toStraightAhead = straightAheadPoint - roadCenterAtClosest;
        float projectionDistance = Vector3.Dot(toStraightAhead, directionLookahead);
        Vector3 roadCenterAtLookahead = roadCenterAtClosest + directionLookahead * projectionDistance;
        
        return (roadCenterAtCar, roadCenterAtLookahead, tLookahead);
    }
    
    /// <summary>
    /// Get ground truth lane positions (left and right) in vehicle coordinates.
    /// Uses dynamic calculation from RoadGenerator if available, otherwise uses static positions.
    /// NOTE: This returns lane lines at the car's current position (0m ahead).
    /// For comparison with perception (which evaluates at 8m ahead), use GetLanePositionsAtLookahead().
    /// </summary>
    /// <returns>Tuple of (left_lane_line_x, right_lane_line_x) in vehicle coordinates</returns>
    public (float leftX, float rightX) GetLanePositionsVehicle()
    {
        // Safety check: component must be enabled
        if (!enabled)
        {
            Debug.LogWarning("GroundTruthReporter.GetLanePositionsVehicle: Component is disabled!");
            return (0f, 0f);
        }
        
        // Ensure carTransform is set (in case Start() hasn't run yet)
        if (carTransform == null)
        {
            if (transform.name.Contains("Car") || GetComponent<CarController>() != null)
            {
                carTransform = transform;
                Debug.Log($"GroundTruthReporter: Using attached transform '{transform.name}'");
            }
            else
            {
                // Try multiple ways to find the car
                GameObject car = GameObject.Find("CarPrefab");
                if (car == null)
                {
                    car = GameObject.Find("Car");
                }
                if (car == null)
                {
                    // Try finding by component
                    CarController carController = FindObjectOfType<CarController>();
                    if (carController != null)
                    {
                        car = carController.gameObject;
                    }
                }
                
                if (car != null)
                {
                    carTransform = car.transform;
                    Debug.Log($"GroundTruthReporter: Found car '{car.name}' at position {carTransform.position}");
                }
                else
                {
                    Debug.LogError("GroundTruthReporter: Could not find car! Returning (0,0). " +
                                 $"Transform name: '{transform.name}', " +
                                 $"Has CarController: {GetComponent<CarController>() != null}");
                    return (0f, 0f);
                }
            }
        }
        
        // Use dynamic ground truth from RoadGenerator if available
        if (useDynamicGroundTruth && roadGenerator != null)
        {
            // CRITICAL: Ensure carTransform is valid and position is updating
            if (carTransform == null)
            {
                Debug.LogError("GroundTruthReporter: carTransform is null in GetLanePositionsVehicle!");
                return (0f, 0f);
            }
            
            // Debug: Check if car position is updating (every 30 frames)
            if (Time.frameCount % 30 == 0)
            {
                Debug.Log($"GroundTruthReporter [DEBUG]: Car position={carTransform.position}, " +
                         $"Car forward={carTransform.forward}, " +
                         $"Car right={Vector3.Cross(Vector3.up, carTransform.forward).normalized}");
            }
            
            var (leftX, rightX) = GetLanePositionsFromRoadGenerator();
            
            // Debug log every 30 frames
            if (Time.frameCount % 30 == 0)
            {
                // Get car position for debug
                Vector3 carPosDebug;
                // PERFORMANCE: Use cached Rigidbody
                if (cachedCarRigidbody == null)
                {
                    cachedCarRigidbody = carTransform.GetComponent<Rigidbody>();
                }
                if (cachedCarRigidbody != null)
                {
                    carPosDebug = cachedCarRigidbody.position;
                }
                else
                {
                    carPosDebug = carTransform.position;
                }
                
                float tDebug;
                if (useTimeBasedReference && useDynamicGroundTruth)
                {
                    tDebug = GetTimeBasedReferenceT();
                }
                else
                {
                    tDebug = FindClosestPointOnOval(carPosDebug);
                }
                
                Vector3 roadCenter = roadGenerator.GetOvalCenterPoint(tDebug);
                Vector3 direction = roadGenerator.GetOvalDirection(tDebug);
                Vector3 roadRight = Vector3.Cross(Vector3.up, direction).normalized;
                float halfWidth = roadGenerator.roadWidth * 0.5f;
                Vector3 leftLaneWorld = roadCenter - roadRight * halfWidth;
                Vector3 rightLaneWorld = roadCenter + roadRight * halfWidth;
                
                // Show coordinate conversion
                Vector3 carForward = carTransform.forward;
                Vector3 carRight = Vector3.Cross(Vector3.up, carForward).normalized;
                Vector3 toLeftLane = leftLaneWorld - carPosDebug;
                Vector3 toRightLane = rightLaneWorld - carPosDebug;
                float leftXVehicleDebug = Vector3.Dot(toLeftLane, carRight);
                float rightXVehicleDebug = Vector3.Dot(toRightLane, carRight);
                
                Debug.Log($"GroundTruthReporter [DYNAMIC]: Car position={carPosDebug}, " +
                         $"Reference t={tDebug:F3}, Road center={roadCenter}, " +
                         $"Left lane world={leftLaneWorld}, Right lane world={rightLaneWorld}, " +
                         $"Left lane vehicle={leftXVehicleDebug:F3}m, Right lane vehicle={rightXVehicleDebug:F3}m, " +
                         $"Lane width={rightXVehicleDebug - leftXVehicleDebug:F3}m, " +
                         $"Center vehicle={(leftXVehicleDebug + rightXVehicleDebug)/2.0f:F3}m");
            }
            
            return (leftX, rightX);
        }
        else
        {
            // Use static ground truth (straight road)
            float carXWorld = carTransform.position.x;
            float leftXVehicle = leftLaneXWorld - carXWorld;
            float rightXVehicle = rightLaneXWorld - carXWorld;
            
            // Debug log every 30 frames
            if (Time.frameCount % 30 == 0)
            {
                Debug.Log($"GroundTruthReporter [STATIC]: Car X={carXWorld:F3}m, " +
                         $"Left lane={leftXVehicle:F3}m, Right lane={rightXVehicle:F3}m, " +
                         $"Lane width={rightXVehicle - leftXVehicle:F3}m");
            }
            
            return (leftXVehicle, rightXVehicle);
        }
    }
    
    /// <summary>
    /// Get ground truth lane center for current lane in vehicle coordinates.
    /// For single-lane roads (no center line), returns the road center.
    /// </summary>
    /// <returns>Lane center X position in vehicle coordinates</returns>
    public float GetCurrentLaneCenterVehicle()
    {
        if (carTransform == null)
        {
            return 0f;
        }
        
        // Get lane positions (works for both static and dynamic)
        var (leftX, rightX) = GetLanePositionsVehicle();
        
        // Road center = midpoint between left and right lane lines
        float roadCenterVehicle = (leftX + rightX) / 2.0f;
        
        return roadCenterVehicle;
    }
    
    /// <summary>
    /// Get desired heading from path at lookahead distance.
    /// Returns the heading the car should have to follow the path.
    /// </summary>
    /// <param name="lookaheadDistance">Distance ahead to look (meters)</param>
    /// <returns>Desired heading in degrees (0-360), or current car heading if path not available</returns>
    public float GetDesiredHeading(float lookaheadDistance = 5.0f)
    {
        if (carTransform == null)
        {
            return 0f;
        }
        
        // Use dynamic ground truth from RoadGenerator if available
        if (useDynamicGroundTruth && roadGenerator != null)
        {
            // Use time-based reference or car position
            float t;
            if (useTimeBasedReference)
            {
                t = GetTimeBasedReferenceT();
            }
            else
            {
                // Use Rigidbody position if available (more accurate for physics-based movement)
                Vector3 carPos;
                // PERFORMANCE: Use cached Rigidbody
                if (cachedCarRigidbody == null)
                {
                    cachedCarRigidbody = carTransform.GetComponent<Rigidbody>();
                }
                if (cachedCarRigidbody != null)
                {
                    carPos = cachedCarRigidbody.position;
                }
                else
                {
                    carPos = carTransform.position;
                }
                
                // Find closest point on path
                t = FindClosestPointOnOval(carPos);
            }
            
            // Get direction at that point (this is the path tangent)
            Vector3 direction = roadGenerator.GetOvalDirection(t);
            
            // Convert direction vector to heading angle
            // Unity: forward is +Z, right is +X
            // Heading: 0° = +Z (north), 90° = +X (east)
            float heading = Mathf.Atan2(direction.x, direction.z) * Mathf.Rad2Deg;
            
            // Normalize to 0-360
            if (heading < 0) heading += 360f;
            
            return heading;
        }
        else
        {
            // Static road: heading is constant (assume straight road, heading = 0° or car's current heading)
            // For straight road, desired heading is forward (0° in Unity)
            return 0f;
        }
    }
    
    /// <summary>
    /// Get path curvature at current position (for anticipatory steering).
    /// </summary>
    /// <returns>Curvature in 1/meters (positive = left turn, negative = right turn, 0 = straight)</returns>
    public float GetPathCurvature()
    {
        if (!useDynamicGroundTruth || roadGenerator == null || carTransform == null)
        {
            return 0f; // Straight road
        }
        
        // Find closest point on path
        Vector3 carPos = carTransform.position;
        float t = FindClosestPointOnOval(carPos);
        
        // Get direction at current point
        Vector3 direction = roadGenerator.GetOvalDirection(t);
        
        // Sample nearby points to estimate curvature
        float dt = 0.01f;
        float tPrev = Mathf.Clamp01(t - dt);
        float tNext = Mathf.Clamp01(t + dt);
        
        Vector3 dirPrev = roadGenerator.GetOvalDirection(tPrev);
        Vector3 dirNext = roadGenerator.GetOvalDirection(tNext);
        
        // Compute change in direction (curvature)
        Vector3 dirChange = dirNext - dirPrev;
        float curvature = dirChange.magnitude / (2.0f * dt * roadGenerator.turnRadius);
        
        // Determine sign (left vs right turn)
        Vector3 cross = Vector3.Cross(dirPrev, dirNext);
        if (cross.y < 0) curvature = -curvature;
        
        return curvature;
    }
    
    void OnDrawGizmos()
    {
        // Ensure carTransform is set for visualization
        if (carTransform == null)
        {
            if (transform.name.Contains("Car") || GetComponent<CarController>() != null)
            {
                carTransform = transform;
            }
            else
            {
                GameObject car = GameObject.Find("CarPrefab");
                if (car != null)
                {
                    carTransform = car.transform;
                }
                else
                {
                    return; // Can't visualize without car position
                }
            }
        }
        
        // FIXED: Draw ground truth based on mode (dynamic or static)
        if (useDynamicGroundTruth && roadGenerator != null)
        {
            // Dynamic mode: Draw lane centers from RoadGenerator
            Vector3 carPos = carTransform.position;
            float t = FindClosestPointOnOval(carPos);
            Vector3 roadCenter = roadGenerator.GetOvalCenterPoint(t);
            Vector3 direction = roadGenerator.GetOvalDirection(t);
            Vector3 roadRight = Vector3.Cross(Vector3.up, direction).normalized;
            
            float halfWidth = roadGenerator.roadWidth * 0.5f;
            Vector3 leftLaneWorld = roadCenter - roadRight * halfWidth;
            Vector3 rightLaneWorld = roadCenter + roadRight * halfWidth;
            
            // Draw left lane center (blue)
            Gizmos.color = Color.blue;
            Gizmos.DrawLine(leftLaneWorld + direction * 5f, leftLaneWorld - direction * 5f);
            
            // Draw right lane center (green) - this is the green line you see
            Gizmos.color = Color.green;
            Gizmos.DrawLine(rightLaneWorld + direction * 5f, rightLaneWorld - direction * 5f);
            
            // Draw road center (yellow) for reference
            Gizmos.color = Color.yellow;
            Gizmos.DrawLine(roadCenter + direction * 5f, roadCenter - direction * 5f);
        }
        else
        {
            // Static mode: Draw fixed positions (straight road)
            float leftCenterWorld = GetLaneCenterWorld(0);
            Vector3 leftCenterPos = new Vector3(leftCenterWorld, 0.1f, carTransform.position.z);
            Gizmos.color = Color.blue;
            Gizmos.DrawLine(leftCenterPos + Vector3.forward * 5f, leftCenterPos - Vector3.forward * 5f);
            
            float rightCenterWorld = GetLaneCenterWorld(1);
            Vector3 rightCenterPos = new Vector3(rightCenterWorld, 0.1f, carTransform.position.z);
            Gizmos.color = Color.green;
            Gizmos.DrawLine(rightCenterPos + Vector3.forward * 5f, rightCenterPos - Vector3.forward * 5f);
        }
    }
}

