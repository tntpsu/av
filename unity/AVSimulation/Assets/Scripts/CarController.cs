// 12/21/2025 AI-Tag
// This was created with the help of Assistant, a Unity Artificial Intelligence product.

using UnityEngine;
using UnityEngine.InputSystem;
#if ENABLE_PROFILER
using UnityEngine.Profiling;
#endif

public class CarController : MonoBehaviour
{
    public float maxSteerAngle = 35f;
    [Tooltip("Wheelbase in meters (distance between front and rear axles)")]
    public float wheelbaseMeters = 2.5f;
    [Tooltip("Acceleration in m/s² (ForceMode.Acceleration ignores mass, so this is direct acceleration) - Realistic car: 3-6 m/s²")]
    public float motorForce = 6f;  // Tuned down to reduce acceleration spikes and jerk
                                   // Increased from 5f to 8f to ensure it overcomes Unity's default friction
                                   // Previous values (400f, 200f) were 50-100x too fast!
                                   // ForceMode.Acceleration ignores mass, so motorForce = acceleration in m/s²
    [Tooltip("Brake force - ONLY used for reporting brakeTorque in vehicle state, NOT for actual physics braking")]
    [System.Obsolete("brakeForce is only used for reporting, not actual braking. Actual braking uses linearDamping and velocity reduction.")]
    public float brakeForce = 5000f;  // NOTE: This is ONLY for reporting brakeTorque, NOT for physics!
    [Tooltip("Maximum allowed speed in m/s - hard limit")]
    public float maxSpeed = 13.0f;  // FIXED: Hard speed limit
    [Tooltip("Start reducing throttle at this fraction of max speed")]
    public float speedPreventionThreshold = 0.90f;
    [Tooltip("Cut throttle completely at this fraction of max speed")]
    public float speedPreventionCutoff = 0.98f;
    [Tooltip("Throttle multiplier when near max speed")]
    public float speedPreventionThrottleFactor = 0.15f;

    
    [Header("Input")]
    public InputActionAsset inputActionAsset;

    [Header("AV Control")]
    [Tooltip("Enable AV control from Python stack")]
    public bool avControlEnabled = false;
    [Tooltip("Priority: AV control overrides manual input when enabled")]
    public bool avControlPriority = true;
    [Tooltip("Max age (seconds) of control commands before safe fallback")]
    public float maxCommandAgeSeconds = 0.2f;
    [Tooltip("Brake applied when command is stale")]
    public float staleCommandBrake = 0.3f;
    private float lastAvCommandTime = -999f;
    
    [Header("Ground Truth Mode")]
    [Tooltip("Use direct velocity/position control for precise ground truth following (bypasses physics forces)")]
    public bool groundTruthMode = false;
    [Tooltip("Speed for ground truth mode (m/s)")]
    public float groundTruthSpeed = 5.0f;
    [Tooltip("GT experiment: derive rotation (and velocity direction) from road-frame tangent at car position.")]
    public bool gtRotationFromRoadFrame = false;
    [Tooltip("GT experiment: skip MoveRotation() and leave heading unchanged.")]
    public bool gtDisableMoveRotation = false;

    [Header("Speed Limit")]
    [Tooltip("Speed limit at current reference point (m/s)")]
    public float speedLimit = 0.0f;

    private Rigidbody rb;
    private InputActionMap playerActionMap;
    private InputAction moveAction;
    private RigidbodyConstraints baseConstraints;
    private bool baseIsKinematic;
    
    // Teleportation prevention: Track last position to detect large jumps
    private Vector3 lastPosition;
    private float lastPositionTime;
    private const float MAX_TELEPORT_DISTANCE = 2.0f; // Maximum allowed teleport distance (meters)

    [Header("Diagnostics")]
    [SerializeField] private bool logProfilerOnHitch = true;
    [SerializeField] private float hitchThresholdSeconds = 0.2f;
    [SerializeField] private int periodicProfilerLogFrames = 300;
    private int lastProfilerLogFrame = -999;
    [SerializeField] private bool logFixedUpdateGapSummary = true;
    [SerializeField] private int fixedUpdateGapSummaryFrames = 300;
    [SerializeField] private bool logCollisionEvents = true;
    [SerializeField] private bool logSpeedDropEvents = true;
    [SerializeField] private float speedDropThreshold = 3.0f; // m/s drop in one FixedUpdate
    [SerializeField] private float speedDropAccelThreshold = 50.0f; // m/s^2 magnitude
    private float lastFixedUpdateRealtime = 0f;
    private float fixedUpdateGapSum = 0f;
    private float fixedUpdateGapMax = 0f;
    private int fixedUpdateGapCount = 0;
    private float lastPhysicsSpeed = 0f;
    private float lastPhysicsTime = -1f;
    private float lastCollisionTime = -1f;
    private string lastCollisionInfo = null;
    [SerializeField] private float throttleSlewRatePerSecond = 0.7f; // max throttle change per second
    private float lastThrottleApplied = 0f;

    // PERFORMANCE: Cache GroundTruthReporter to avoid FindObjectOfType() every frame
    private GroundTruthReporter cachedGroundTruthReporter = null;

    public float steerInput; // [-1, 1]
    public float throttleInput; // [0, 1]
    public float brakeInput; // [0, 1]
    public float steerSlewRatePerSecond = 6.0f;
    [Header("Friction (Option A)")]
    public float carStaticFriction = 0.5f;
    public float carDynamicFriction = 0.4f;
    public PhysicsMaterialCombine carFrictionCombine = PhysicsMaterialCombine.Average;
    [Header("Wheel Colliders (Option B)")]
    public bool useWheelColliders = true;
    public float wheelRadius = 0.35f;
    public float wheelSuspensionDistance = 0.2f;
    public float wheelSpring = 30000f;
    public float wheelDamper = 6000f;
    public float wheelMass = 20f;
    public float maxMotorTorque = 1500f;
    public float maxBrakeTorque = 3000f;
    public float wheelPlacementWidthScale = 0.9f;
    public float wheelPlacementLengthScale = 0.9f;
    public bool logWheelColliderDebug = true;
    [Header("Debug")]
    public bool forceFreezeYForDebug = true;
    [Header("Steering Actuator")]
    [Tooltip("Max steer angle at low speed (deg)")]
    public float maxSteerAngleLowSpeed = 30f;
    [Tooltip("Max steer angle at high speed (deg)")]
    public float maxSteerAngleHighSpeed = 16f;
    [Tooltip("Speed (m/s) where max steer reaches the high-speed limit")]
    public float maxSteerAngleFullSpeed = 12f;
    [Tooltip("Steering rate limit (deg/s)")]
    public float steeringRateLimitDegPerSec = 90f;
    [Tooltip("Steering actuator time constant (s)")]
    public float steeringTimeConstant = 0.25f;
    [Tooltip("Disable steering actuator model and apply direct steering input.")]
    public bool disableSteeringActuator = false;

    // AV control inputs (set by AVBridge)
    private float avSteering = 0f;
    private float avThrottle = 0f;
    private float avBrake = 0f;
    private float lastSteerInput = 0f;
    private float steeringAngleActual = 0f;
    private float desiredSteerAngle = 0f;

    private WheelCollider frontLeftWheel;
    private WheelCollider frontRightWheel;
    private WheelCollider rearLeftWheel;
    private WheelCollider rearRightWheel;
    private bool wheelCollidersReady = false;
    private bool wheelOrientationLogged = false;
    
    // Ground truth cache (updated in Update, applied in FixedUpdate)
    private Vector3 gtCachedReferencePosition = Vector3.zero;
    private Vector3 gtCachedReferenceDirection = Vector3.zero;
    private bool gtCachedReferenceValid = false;
    private float gtCachedReferenceTime = 0f;
    private Vector3 gtLastValidRoadFrameDirection = Vector3.forward;
    private bool gtRotationDebugValid = false;
    private bool gtRotationUsedRoadFrame = false;
    private bool gtRotationRejectedRoadFrameHop = false;
    private float gtRotationReferenceHeadingDeg = 0f;
    private float gtRotationRoadFrameHeadingDeg = 0f;
    private float gtRotationInputHeadingDeg = 0f;
    private float gtRotationRoadVsRefDeltaDeg = 0f;
    private float gtRotationAppliedDeltaDeg = 0f;

    private void Awake()
    {
        // Option B: force wheel-collider drivetrain on for tuning runs.
        useWheelColliders = true;
        logWheelColliderDebug = true;
        rb = GetComponent<Rigidbody>();
        
        // FIX: Prevent car from sinking into ground due to physics settling
        // Freeze Y position to keep car at correct height (prevents 0.5m → 0.2m drop)
        if (rb != null)
        {
            if (forceFreezeYForDebug)
            {
                rb.constraints = RigidbodyConstraints.FreezePositionY
                    | RigidbodyConstraints.FreezeRotationX
                    | RigidbodyConstraints.FreezeRotationZ;
            }
            else if (useWheelColliders)
            {
                // Freeze rotation to keep car upright. Allow Y movement for wheel colliders.
                rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
            }
            else
            {
                // Freeze Y position to prevent sinking in non-wheel-collider mode.
                rb.constraints = RigidbodyConstraints.FreezePositionY | RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
            }
            baseConstraints = rb.constraints;
            baseIsKinematic = rb.isKinematic;
            
            // CRITICAL: Disable Rigidbody sleep threshold to prevent car from going to sleep
            // Unity puts Rigidbody to sleep when velocity < sleep threshold (0.005 m/s)
            // This prevents the car from starting to move when throttle is low
            // Setting to -1.0 disables sleep completely
            rb.sleepThreshold = -1.0f;
            
            // CRITICAL: Ensure drag is zero to prevent forces from being counteracted
            // High drag would prevent the car from accelerating from rest
            // NOTE: Unity has both 'drag' (old) and 'linearDamping' (new) - set both to 0
            rb.linearDamping = 0f;
            rb.angularDamping = 0f;
            
            // In GT mode we directly move the rigidbody; interpolation can cause visual bounce.
            rb.interpolation = RigidbodyInterpolation.None;
            // Also set the old 'drag' property if it exists (for compatibility)
            #if UNITY_2021_1_OR_NEWER
            // In Unity 2021+, drag is deprecated but might still exist
            #else
            rb.drag = 0f;
            rb.angularDrag = 0f;
            #endif
            
            if (!useWheelColliders)
            {
                // Ensure car starts at correct position (Y = 0.5m if car is 1m tall)
                // This ensures bottom of car is at Y=0 (ground level)
                Vector3 pos = transform.position;
                if (pos.y < 0.4f || pos.y > 0.6f)
                {
                    // Car position seems wrong - set to 0.5m (assuming 1m tall car)
                    pos.y = 0.5f;
                    transform.position = pos;
                    Debug.Log($"CarController: Fixed car Y position to {pos.y}m (was {transform.position.y}m)");
                }
            }
            
            // Initialize teleportation tracking
            lastPosition = rb.position;
            lastPositionTime = Time.time;
        }

        ApplyCarFrictionMaterial();
        if (useWheelColliders)
        {
            SetupWheelColliders();
        }
    }

    private void SetupWheelColliders()
    {
        Collider bodyCollider = GetComponent<Collider>();
        Bounds bounds = bodyCollider != null
            ? bodyCollider.bounds
            : new Bounds(transform.position, new Vector3(1.6f, 1.0f, 2.5f));
        float halfWidth = bounds.extents.x * wheelPlacementWidthScale;
        float halfLength = bounds.extents.z * wheelPlacementLengthScale;
        float wheelY = bounds.min.y + wheelRadius;

        Vector3 center = bounds.center;
        Vector3[] wheelWorldPositions = new[]
        {
            center + new Vector3(-halfWidth, wheelY - center.y, halfLength),
            center + new Vector3(halfWidth, wheelY - center.y, halfLength),
            center + new Vector3(-halfWidth, wheelY - center.y, -halfLength),
            center + new Vector3(halfWidth, wheelY - center.y, -halfLength)
        };

        frontLeftWheel = CreateWheelCollider("WheelFL", wheelWorldPositions[0]);
        frontRightWheel = CreateWheelCollider("WheelFR", wheelWorldPositions[1]);
        rearLeftWheel = CreateWheelCollider("WheelRL", wheelWorldPositions[2]);
        rearRightWheel = CreateWheelCollider("WheelRR", wheelWorldPositions[3]);

        wheelCollidersReady = frontLeftWheel != null && frontRightWheel != null
            && rearLeftWheel != null && rearRightWheel != null;
        Debug.Log(
            $"WheelColliders: ready={wheelCollidersReady} boundsCenter={bounds.center} " +
            $"halfWidth={halfWidth:F2} halfLength={halfLength:F2} wheelRadius={wheelRadius:F2}"
        );
    }

    private void LogWheelOrientation(string label, WheelCollider wheel)
    {
        if (wheel == null)
        {
            Debug.LogWarning($"WheelOrientation: {label} is null");
            return;
        }
        Vector3 fwd = wheel.transform.forward.normalized;
        Vector3 right = wheel.transform.right.normalized;
        Vector3 up = wheel.transform.up.normalized;
        Debug.Log(
            $"WheelOrientation: {label} forward=({fwd.x:F3},{fwd.y:F3},{fwd.z:F3}) " +
            $"right=({right.x:F3},{right.y:F3},{right.z:F3}) " +
            $"up=({up.x:F3},{up.y:F3},{up.z:F3})"
        );
    }

    private WheelCollider CreateWheelCollider(string name, Vector3 worldPosition)
    {
        GameObject wheelObject = new GameObject(name);
        wheelObject.transform.SetParent(transform, true);
        wheelObject.transform.position = worldPosition;

        WheelCollider wheel = wheelObject.AddComponent<WheelCollider>();
        wheel.radius = wheelRadius;
        wheel.suspensionDistance = wheelSuspensionDistance;
        wheel.mass = wheelMass;

        JointSpring spring = wheel.suspensionSpring;
        spring.spring = wheelSpring;
        spring.damper = wheelDamper;
        spring.targetPosition = 0.5f;
        wheel.suspensionSpring = spring;

        WheelFrictionCurve forward = wheel.forwardFriction;
        forward.extremumSlip = 0.4f;
        forward.extremumValue = 1.0f;
        forward.asymptoteSlip = 0.8f;
        forward.asymptoteValue = 0.5f;
        forward.stiffness = 1.2f;
        wheel.forwardFriction = forward;

        WheelFrictionCurve sideways = wheel.sidewaysFriction;
        sideways.extremumSlip = 0.3f;
        sideways.extremumValue = 1.0f;
        sideways.asymptoteSlip = 0.7f;
        sideways.asymptoteValue = 0.5f;
        sideways.stiffness = 1.2f;
        wheel.sidewaysFriction = sideways;

        return wheel;
    }

    private void ApplyCarFrictionMaterial()
    {
        PhysicsMaterial carMaterial = new PhysicsMaterial("CarPhysicMaterial")
        {
            staticFriction = carStaticFriction,
            dynamicFriction = carDynamicFriction,
            frictionCombine = carFrictionCombine,
            bounciness = 0.0f,
            bounceCombine = PhysicsMaterialCombine.Minimum
        };

        Collider[] colliders = GetComponentsInChildren<Collider>();
        foreach (Collider collider in colliders)
        {
            collider.material = carMaterial;
        }
    }

    private void Start()
    {
        bool? cliGtRotationFromRoadFrame = ParseCommandLineBool(
            System.Environment.GetCommandLineArgs(),
            "--gt-rotation-from-road-frame"
        );
        if (cliGtRotationFromRoadFrame.HasValue)
        {
            gtRotationFromRoadFrame = cliGtRotationFromRoadFrame.Value;
        }
        bool? cliGtDisableMoveRotation = ParseCommandLineBool(
            System.Environment.GetCommandLineArgs(),
            "--gt-disable-move-rotation"
        );
        if (cliGtDisableMoveRotation.HasValue)
        {
            gtDisableMoveRotation = cliGtDisableMoveRotation.Value;
        }

        // CRITICAL: Initialize cached GroundTruthReporter BEFORE first frame
        // This ensures it's available from frame 0, preventing null reference issues
        if (cachedGroundTruthReporter == null)
        {
            cachedGroundTruthReporter = GetComponent<GroundTruthReporter>();
            if (cachedGroundTruthReporter == null)
            {
                // Fallback: search for it in the scene (only once, then cache)
                cachedGroundTruthReporter = FindObjectOfType<GroundTruthReporter>();
            }
        }
        gtLastValidRoadFrameDirection = transform.forward.sqrMagnitude > 0.01f
            ? transform.forward.normalized
            : Vector3.forward;
        
        // Try to load the InputActionAsset if not assigned
        if (inputActionAsset == null)
        {
            inputActionAsset = UnityEngine.Resources.Load<InputActionAsset>("InputSystem_Actions");
        }
        
        // If still null, try to find it in the project
        if (inputActionAsset == null)
        {
            InputActionAsset[] allAssets = Resources.FindObjectsOfTypeAll<InputActionAsset>();
            if (allAssets.Length > 0)
            {
                inputActionAsset = allAssets[0];
            }
        }

        // ProfilerRecorder not available in this target; using legacy Profiler API when enabled.
        
        // Set up input actions
        if (inputActionAsset != null)
        {
            playerActionMap = inputActionAsset.FindActionMap("Player");
            if (playerActionMap != null)
            {
                moveAction = playerActionMap.FindAction("Move");
            }
        }
    }

    private void OnEnable()
    {
        // Ensure InputActionAsset is initialized before enabling maps
        if (inputActionAsset == null)
        {
            inputActionAsset = UnityEngine.Resources.Load<InputActionAsset>("InputSystem_Actions");
        }
        
        if (inputActionAsset == null)
        {
            InputActionAsset[] allAssets = Resources.FindObjectsOfTypeAll<InputActionAsset>();
            if (allAssets.Length > 0)
            {
                inputActionAsset = allAssets[0];
            }
        }
        
        // Initialize action map if not already done
        if (inputActionAsset != null && playerActionMap == null)
        {
            playerActionMap = inputActionAsset.FindActionMap("Player");
            if (playerActionMap != null)
            {
                moveAction = playerActionMap.FindAction("Move");
            }
        }
        
        // Enable the asset first, then the maps
        if (inputActionAsset != null)
        {
            if (!inputActionAsset.enabled)
            {
                inputActionAsset.Enable();
            }
            
            // Now safely enable the maps
            try
            {
                playerActionMap?.Enable();
                moveAction?.Enable();
            }
            catch (System.Exception e)
            {
                // Input system might not be fully initialized - this is OK for AV control mode
                Debug.LogWarning($"CarController: Could not enable input actions (this is OK if using AV control): {e.Message}");
            }
        }
    }

    private void OnDisable()
    {
        // Safely disable actions
        if (moveAction != null && moveAction.enabled)
        {
            moveAction.Disable();
        }
        if (playerActionMap != null && playerActionMap.enabled)
        {
            playerActionMap.Disable();
        }

        // No profiler recorder disposal needed.
    }

    private void Update()
    {
        // Read manual input
        Vector2 moveInput = Vector2.zero;
        
        if (moveAction != null)
        {
            // Read input from Input System
            moveInput = moveAction.ReadValue<Vector2>();
        }
        else
        {
            // Fallback: Read directly from keyboard/gamepad
            Keyboard keyboard = Keyboard.current;
            Gamepad gamepad = Gamepad.current;
            
            if (gamepad != null)
            {
                moveInput = gamepad.leftStick.ReadValue();
            }
            else if (keyboard != null)
            {
                // Read WASD or arrow keys
                float horizontal = 0f;
                float vertical = 0f;
                
                if (keyboard.wKey.isPressed || keyboard.upArrowKey.isPressed) vertical = 1f;
                if (keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed) vertical = -1f;
                if (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed) horizontal = -1f;
                if (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed) horizontal = 1f;
                
                moveInput = new Vector2(horizontal, vertical);
            }
        }
        
        // Apply AV control or manual input based on priority
        // CRITICAL FIX: Check if AV commands are being received (non-zero values)
        // This handles the race condition where SetAVControls() is called before
        // AVBridge's coroutine sets avControlEnabled = true
        bool hasAVCommands = Mathf.Abs(avSteering) > 0.01f || Mathf.Abs(avThrottle) > 0.01f || Mathf.Abs(avBrake) > 0.01f;
        
        if ((avControlEnabled && avControlPriority) || hasAVCommands)
        {
            // Use AV control (either explicitly enabled, or commands detected)
            // If commands detected but not enabled, auto-enable now
            if (hasAVCommands && !avControlEnabled)
            {
                Debug.Log($"[AUTO-ENABLE] CarController: Auto-enabling AV control (received commands: steering={avSteering:F3}, throttle={avThrottle:F3}, brake={avBrake:F3})");
                avControlEnabled = true;
                avControlPriority = true;
            }
            
            steerInput = avSteering;
            throttleInput = avThrottle;
            brakeInput = avBrake;
        }
        else if (!avControlEnabled)
        {
            // Use manual input
            steerInput = moveInput.x;
            throttleInput = moveInput.y;
            brakeInput = 0f; // Manual brake would need separate input
        }
        else
        {
            // AV enabled but not priority - blend or use manual
            steerInput = moveInput.x;
            throttleInput = moveInput.y;
            brakeInput = 0f;
        }

        // Steering actuator slew to reduce oscillation.
        float desiredSteerInput = steerInput;
        float maxSteerDelta = steerSlewRatePerSecond * Time.deltaTime;
        steerInput = Mathf.MoveTowards(lastSteerInput, desiredSteerInput, maxSteerDelta);
        lastSteerInput = steerInput;

        UpdateGroundTruthCache();
    }

    private void FixedUpdate()
    {
        if (logFixedUpdateGapSummary && fixedUpdateGapSummaryFrames > 0)
        {
            float realtimeNow = Time.realtimeSinceStartup;
            if (lastFixedUpdateRealtime > 0f)
            {
                float gap = realtimeNow - lastFixedUpdateRealtime;
                fixedUpdateGapCount += 1;
                fixedUpdateGapSum += gap;
                if (gap > fixedUpdateGapMax)
                {
                    fixedUpdateGapMax = gap;
                }

                if (fixedUpdateGapCount >= fixedUpdateGapSummaryFrames)
                {
                    float avgGap = fixedUpdateGapSum / Mathf.Max(1, fixedUpdateGapCount);
                    Debug.Log(
                        $"[UNITY FIXED GAP] max={fixedUpdateGapMax:F4}s avg={avgGap:F4}s " +
                        $"frames={fixedUpdateGapCount} frame={Time.frameCount} time={Time.time:F3}"
                    );
                    fixedUpdateGapCount = 0;
                    fixedUpdateGapSum = 0f;
                    fixedUpdateGapMax = 0f;
                }
            }
            lastFixedUpdateRealtime = realtimeNow;
        }

        // Update position tracking for teleportation detection (even in physics mode)
        if (rb != null)
        {
            Vector3 currentPosition = rb.position;
            float timeSinceLastPosition = Time.time - lastPositionTime;
            
            // Check for large position jumps in physics mode too (could indicate Unity pause/resume)
            if (timeSinceLastPosition > 0.1f && lastPositionTime > 0) // More than 100ms gap and not first frame
            {
                float positionChange = Vector3.Distance(currentPosition, lastPosition);
                if (positionChange > MAX_TELEPORT_DISTANCE)
                {
                    Debug.LogWarning($"CarController: Large position change detected in physics mode: " +
                                   $"{positionChange:F2}m over {timeSinceLastPosition:F3}s. " +
                                   $"Unity may have paused/resumed.");
                }

                if (logProfilerOnHitch && timeSinceLastPosition > hitchThresholdSeconds)
                {
                    LogProfilerSnapshot($"HITCH (dt={timeSinceLastPosition:F3}s)", positionChange);
                }
            }

            // Also log based on Unity's reported deltaTime (captures stalls even if position doesn't move)
            if (logProfilerOnHitch && Time.deltaTime > hitchThresholdSeconds)
            {
                float positionChangeForDelta = Vector3.Distance(currentPosition, lastPosition);
                LogProfilerSnapshot($"HITCH_DELTA (dt={Time.deltaTime:F3}s)", positionChangeForDelta);
            }

            if (logProfilerOnHitch && Time.frameCount - lastProfilerLogFrame >= periodicProfilerLogFrames)
            {
                lastProfilerLogFrame = Time.frameCount;
                LogProfilerSnapshot("Periodic", 0.0f);
            }
            
            // Update tracking (but don't update if we're about to teleport in GT mode)
            // GT mode will update this after checking teleport distance
            if (!groundTruthMode)
            {
                lastPosition = currentPosition;
                lastPositionTime = Time.time;
            }
        }
        
        ApplyControls();
    }

    private void UpdateGroundTruthCache()
    {
        if (!groundTruthMode)
        {
            gtCachedReferenceValid = false;
            return;
        }

        // Cache GroundTruthReporter reference (Update timing) for FixedUpdate application.
        if (cachedGroundTruthReporter == null)
        {
            cachedGroundTruthReporter = GetComponent<GroundTruthReporter>();
            if (cachedGroundTruthReporter == null)
            {
                cachedGroundTruthReporter = FindObjectOfType<GroundTruthReporter>();
            }
        }

        if (cachedGroundTruthReporter != null)
        {
            var (referencePosition, referenceDirection) = cachedGroundTruthReporter.GetCurrentReferencePath();
            if (referenceDirection.sqrMagnitude > 0.01f)
            {
                gtCachedReferencePosition = referencePosition;
                gtCachedReferenceDirection = referenceDirection;
                gtCachedReferenceValid = true;
                gtCachedReferenceTime = Time.time;
            }
        }
    }

    private void ApplyControls()
    {
        // CRITICAL DEBUG: Log which mode we're using (every 60 frames = ~2 seconds)
        if (Time.frameCount % 60 == 0)
        {
            Debug.Log($"CarController: Mode={((groundTruthMode) ? "GROUND TRUTH (direct velocity)" : "PHYSICS (force-based)")}, " +
                     $"steerInput={steerInput:F3}, throttleInput={throttleInput:F3}, brakeInput={brakeInput:F3}");
        }
        
            if (groundTruthMode)
            {
                // Ground Truth Mode: Direct velocity/position control for precise path following
                // This bypasses physics forces for exact control, perfect for ground truth data collection
                
                // CRITICAL: Check for Unity pause/resume BEFORE applying ground truth controls
                // Unity pause causes Time.time to pause, but when it resumes, there's a large time gap
                // We need to detect this and prevent teleportation
                float currentTime = Time.time;
                float timeSinceLastUpdate = currentTime - lastPositionTime;
                
                // If time gap is large (>0.1s), Unity likely paused/resumed
                // In this case, we should NOT apply ground truth controls immediately
                // Instead, keep the car at its current position to prevent teleportation
                if (timeSinceLastUpdate > 0.1f && lastPositionTime > 0)
                {
                    Debug.LogWarning($"CarController: Large time gap detected ({timeSinceLastUpdate:F3}s) in ground truth mode. " +
                                   $"Unity may have paused. Skipping ground truth update to prevent teleportation.");
                    // Don't update position - keep car where it is
                    // Update lastPositionTime to current time to reset the check
                    lastPositionTime = currentTime;
                    return; // Skip ground truth controls this frame
                }
                
                ApplyGroundTruthControls();
            }
        else
        {
            // Normal Mode: Physics-based control for realistic car behavior
            // This emulates real-world car physics with forces, suitable for AV stack testing
            // If AV control is enabled but commands are stale, apply safe fallback
            if (avControlEnabled && !groundTruthMode)
            {
                float commandAge = Time.time - lastAvCommandTime;
                if (commandAge > maxCommandAgeSeconds)
                {
                    avSteering = 0f;
                    avThrottle = 0f;
                    avBrake = Mathf.Clamp01(staleCommandBrake);
                    steerInput = avSteering;
                    throttleInput = avThrottle;
                    brakeInput = avBrake;
                }
            }
            ApplyPhysicsControls();
        }
    }

    private void LogProfilerSnapshot(string reason, float positionChange)
    {
        long gcReserved = 0;
        long gcUsed = 0;
#if ENABLE_PROFILER
        gcReserved = Profiler.GetTotalReservedMemoryLong();
        gcUsed = Profiler.GetTotalAllocatedMemoryLong();
#endif
        long managedBytes = System.GC.GetTotalMemory(false);
        Debug.Log(
            $"[PROFILER] {reason} | frame={Time.frameCount} time={Time.time:F3}s " +
            $"posJump={positionChange:F2}m | " +
            $"gcReserved={gcReserved / (1024 * 1024)}MB gcUsed={gcUsed / (1024 * 1024)}MB " +
            $"managed={managedBytes / (1024 * 1024)}MB " +
            $"dt={Time.deltaTime:F4} sdt={Time.smoothDeltaTime:F4} udt={Time.unscaledDeltaTime:F4} " +
            $"timescale={Time.timeScale:F2}"
        );
    }
    
    private void ApplyGroundTruthControls()
    {
        // NEW APPROACH: Direct path positioning instead of control commands
        // This eliminates all control errors - car is always exactly on the path!
        // The car moves at constant speed along the path, just like the ground truth reference!
        
        // PERFORMANCE: Cache GroundTruthReporter reference to avoid FindObjectOfType() every frame
        // FindObjectOfType() is VERY expensive (searches entire scene)
        if (cachedGroundTruthReporter == null)
        {
            cachedGroundTruthReporter = GetComponent<GroundTruthReporter>();
            if (cachedGroundTruthReporter == null)
            {
                // Fallback: search for it in the scene (only once, then cache)
                cachedGroundTruthReporter = FindObjectOfType<GroundTruthReporter>();
            }
        }
        GroundTruthReporter gtReporter = cachedGroundTruthReporter;
        
        if (gtReporter != null)
        {
            // Get the current reference position and direction on the path
            Vector3 referencePosition;
            Vector3 referenceDirection;
            if (gtCachedReferenceValid && (Time.time - gtCachedReferenceTime) < 0.2f)
            {
                referencePosition = gtCachedReferencePosition;
                referenceDirection = gtCachedReferenceDirection;
            }
            else
            {
                // If cache is stale, skip this physics tick to avoid jitter from Update/FixedUpdate mismatch.
                return;
            }
            
            if (referenceDirection.sqrMagnitude > 0.01f)
            {
                // CRITICAL FIX: Prevent large teleportations (e.g., after Unity pause/resume)
                // If the car would teleport more than MAX_TELEPORT_DISTANCE, reject it and keep current position
                // This prevents the car from jumping when Unity pauses and the reference path advances
                Vector3 currentPosition = rb.position;
                float teleportDistance = Vector3.Distance(currentPosition, referencePosition);
                float timeSinceLastPosition = Time.time - lastPositionTime;
                
                // Check for large time gap FIRST (indicates Unity pause/resume)
                // This is more reliable than just checking distance, as Unity pause causes time gaps
                bool isLargeTimeGap = timeSinceLastPosition > 0.1f && lastPositionTime > 0;
                bool isLargeTeleport = teleportDistance > MAX_TELEPORT_DISTANCE;
                
                // Reject teleportation if either condition is met
                if (isLargeTimeGap || isLargeTeleport)
                {
                    // Large teleportation or time gap detected - likely due to Unity pause/resume
                    // Instead of teleporting, keep the car where it is and log a warning
                    
                    Debug.LogError($"CarController: ⚠️ LARGE TELEPORTATION REJECTED!");
                    Debug.LogError($"  Current position: {currentPosition}");
                    Debug.LogError($"  Reference position: {referencePosition}");
                    Debug.LogError($"  Teleport distance: {teleportDistance:F2}m (max allowed: {MAX_TELEPORT_DISTANCE}m)");
                    Debug.LogError($"  Time since last position: {timeSinceLastPosition:F3}s");
                    if (isLargeTimeGap)
                    {
                        Debug.LogError($"  ⚠️ LARGE TIME GAP DETECTED - Unity likely paused/resumed!");
                    }
                    Debug.LogError($"  Car will stay at current position to prevent teleportation.");
                    
                    // Keep car at current position instead of teleporting
                    // The car will continue from where it was, not jump ahead
                    // This prevents perception failures from sudden position changes
                    referencePosition = currentPosition; // Use current position instead
                    
                    // Reset lastPositionTime to current time to prevent repeated warnings
                    // But only if this is a time gap (not just a distance check)
                    if (isLargeTimeGap)
                    {
                        lastPositionTime = Time.time;
                    }
                }
                
                // In GT mode use direct position assignment for deterministic path progression.
                rb.position = referencePosition;
                
                // Update last position tracking
                lastPosition = rb.position;
                lastPositionTime = Time.time;
                
                Vector3 orientationDirection = referenceDirection.normalized;
                bool usedRoadFrame = false;
                bool rejectedRoadFrameHop = false;
                float roadFrameHeadingDeg = HeadingDegFromDirection(orientationDirection);
                float referenceHeadingDeg = HeadingDegFromDirection(referenceDirection);
                float roadVsRefDeltaDeg = 0f;
                if (gtRotationFromRoadFrame && gtReporter != null)
                {
                    // Use reference-progress road tangent for stable GT heading.
                    if (gtReporter.TryGetReferencePathDirection(out Vector3 roadFrameDirection))
                    {
                        Vector3 referenceDir = referenceDirection.normalized;
                        Vector3 lastDir = gtLastValidRoadFrameDirection.sqrMagnitude > 0.01f
                            ? gtLastValidRoadFrameDirection.normalized
                            : referenceDir;

                        float roadDeltaDeg = Vector3.Angle(lastDir, roadFrameDirection);
                        float refDeltaDeg = Vector3.Angle(lastDir, referenceDir);
                        roadVsRefDeltaDeg = Vector3.Angle(referenceDir, roadFrameDirection);

                        // If closest-point tangent jumps sharply while reference tangent remains smooth,
                        // treat it as a nearest-point alias and keep continuity with reference tangent.
                        bool rejectRoadFrameHop = roadDeltaDeg > 1.5f && refDeltaDeg < roadDeltaDeg;
                        rejectedRoadFrameHop = rejectRoadFrameHop;
                        usedRoadFrame = !rejectRoadFrameHop;
                        orientationDirection = rejectRoadFrameHop ? referenceDir : roadFrameDirection;
                        gtLastValidRoadFrameDirection = orientationDirection;
                        roadFrameHeadingDeg = HeadingDegFromDirection(roadFrameDirection);
                    }
                    else if (gtLastValidRoadFrameDirection.sqrMagnitude > 0.01f)
                    {
                        // Keep heading source consistent when road-frame lookup blips.
                        orientationDirection = gtLastValidRoadFrameDirection;
                        usedRoadFrame = true;
                        roadFrameHeadingDeg = HeadingDegFromDirection(gtLastValidRoadFrameDirection);
                        roadVsRefDeltaDeg = Vector3.Angle(referenceDirection.normalized, gtLastValidRoadFrameDirection.normalized);
                    }
                }

                // Set rotation to match selected orientation direction unless explicitly disabled for experiment.
                if (!gtDisableMoveRotation)
                {
                    float currentHeadingDeg = HeadingDegFromDirection(transform.forward);
                    float inputHeadingDeg = HeadingDegFromDirection(orientationDirection);
                    gtRotationAppliedDeltaDeg = Mathf.Abs(Mathf.DeltaAngle(currentHeadingDeg, inputHeadingDeg));
                    gtRotationInputHeadingDeg = inputHeadingDeg;
                    Quaternion targetRotation = Quaternion.LookRotation(orientationDirection, Vector3.up);
                    rb.rotation = targetRotation;
                }
                else
                {
                    gtRotationAppliedDeltaDeg = 0f;
                    gtRotationInputHeadingDeg = HeadingDegFromDirection(orientationDirection);
                }
                gtRotationDebugValid = true;
                gtRotationUsedRoadFrame = usedRoadFrame;
                gtRotationRejectedRoadFrameHop = rejectedRoadFrameHop;
                gtRotationReferenceHeadingDeg = referenceHeadingDeg;
                gtRotationRoadFrameHeadingDeg = roadFrameHeadingDeg;
                gtRotationRoadVsRefDeltaDeg = roadVsRefDeltaDeg;
                
                // Keep translational motion from reference path direction; rotation experiment only affects heading.
                if (!rb.isKinematic)
                {
                    rb.linearVelocity = referenceDirection.normalized * groundTruthSpeed;
                }
                else
                {
                    rb.linearVelocity = Vector3.zero;
                }
                rb.angularVelocity = Vector3.zero;
                
                return; // Success - car is now directly on the path!
            }
            else
            {
                // CRITICAL: Reference direction is zero or too small - this should not happen!
                // Log error to help diagnose issue
                if (Time.frameCount % 60 == 0) // Log every ~2 seconds to avoid spam
                {
                    Debug.LogError($"CarController: Ground truth reference direction is zero! " +
                                 $"position={referencePosition}, direction={referenceDirection}, " +
                                 $"sqrMagnitude={referenceDirection.sqrMagnitude}. " +
                                 $"Falling back to steering-based control (car will veer off!).");
                }
            }
        }
        else
        {
            // GroundTruthReporter not found - log error
            if (Time.frameCount % 60 == 0) // Log every ~2 seconds to avoid spam
            {
                Debug.LogError("CarController: GroundTruthReporter not found! Cannot apply ground truth controls.");
            }
        }
        
        // FALLBACK: Original direct velocity control if we can't access path directly
        // This maintains backward compatibility
        float steerAngle = steerInput * maxSteerAngle;
        Quaternion rotation = Quaternion.Euler(0, steerAngle * Time.fixedDeltaTime, 0);
        rb.MoveRotation(rb.rotation * rotation);
        
        float targetSpeed = throttleInput * groundTruthSpeed;
        if (brakeInput > 0.01f)
        {
            targetSpeed = 0f;
        }
        
        Vector3 forwardDirection = transform.forward;
        rb.linearVelocity = forwardDirection * targetSpeed;
        rb.angularVelocity = Vector3.zero;
    }
    
    private void ApplyPhysicsControls()
    {
        // CRITICAL: Reset drag at the START of each frame to prevent lingering drag from previous frames
        // This ensures drag is always 0 unless we explicitly set it for braking
        // Unity might have default drag values or drag might persist from previous frames
        rb.linearDamping = 0f;
        
        // CRITICAL FIX: Reduce steering when speed is very low to prevent spinning in place
        // When speed < 0.1 m/s, high steering causes car to spin instead of move forward
        // Reduce steering proportionally to allow forward movement
        Vector3 currentVelocity = rb.linearVelocity;
        float currentSpeed = currentVelocity.magnitude;
        
        float adjustedSteerInput = steerInput;
        if (currentSpeed < 0.1f && Mathf.Abs(steerInput) > 0.3f)
        {
            // At very low speed, reduce steering to allow forward movement
            // Scale steering from 0.3 to 0.0 as speed approaches 0
            float steeringReduction = Mathf.Clamp01(currentSpeed / 0.1f); // 0 at speed=0, 1 at speed=0.1
            float maxSteeringAtLowSpeed = 0.3f; // Maximum steering when speed is very low
            float steeringScale = maxSteeringAtLowSpeed + (1.0f - maxSteeringAtLowSpeed) * steeringReduction;
            adjustedSteerInput = steerInput * steeringScale;
            
            if (Time.frameCount % 60 == 0)
            {
                Debug.Log($"CarController: Reducing steering from {steerInput:F3} to {adjustedSteerInput:F3} at low speed ({currentSpeed:F3} m/s) to allow forward movement");
            }
        }
        
        float steerAngle;
        if (disableSteeringActuator)
        {
            desiredSteerAngle = adjustedSteerInput * maxSteerAngle;
            steeringAngleActual = desiredSteerAngle;
            steerAngle = steeringAngleActual;
        }
        else
        {
            float maxSteerAtSpeed = Mathf.Lerp(
                maxSteerAngleLowSpeed,
                maxSteerAngleHighSpeed,
                Mathf.InverseLerp(0f, maxSteerAngleFullSpeed, currentSpeed)
            );
            desiredSteerAngle = adjustedSteerInput * maxSteerAtSpeed;
            float actuatorAlpha = steeringTimeConstant > 0.0f
                ? Time.fixedDeltaTime / (steeringTimeConstant + Time.fixedDeltaTime)
                : 1.0f;
            float filteredSteerAngle = Mathf.Lerp(steeringAngleActual, desiredSteerAngle, actuatorAlpha);
            float maxSteerDelta = steeringRateLimitDegPerSec * Time.fixedDeltaTime;
            steeringAngleActual = Mathf.MoveTowards(steeringAngleActual, filteredSteerAngle, maxSteerDelta);
            steerAngle = steeringAngleActual;
        }
        if (!useWheelColliders || !wheelCollidersReady)
        {
            // Steering - rotate around Y axis only (horizontal steering)
            Quaternion rotation = Quaternion.Euler(0, steerAngle * Time.fixedDeltaTime, 0);
            rb.MoveRotation(rb.rotation * rotation);
        }
        else if (!wheelOrientationLogged)
        {
            wheelOrientationLogged = true;
            LogWheelOrientation("FrontLeft", frontLeftWheel);
            LogWheelOrientation("FrontRight", frontRightWheel);
            LogWheelOrientation("RearLeft", rearLeftWheel);
            LogWheelOrientation("RearRight", rearRightWheel);
            Vector3 carFwd = transform.forward.normalized;
            Vector3 carRight = transform.right.normalized;
            Debug.Log(
                $"CarOrientation: forward=({carFwd.x:F3},{carFwd.y:F3},{carFwd.z:F3}) " +
                $"right=({carRight.x:F3},{carRight.y:F3},{carRight.z:F3})"
            );
        }

        // Braking and friction
        // FIXED: More aggressive braking with direct velocity reduction
        // Note: currentSpeed and currentVelocity already calculated above for steering adjustment
        // No need to recalculate - they're already set
        
        // CRITICAL: Unity-side speed prevention - check BEFORE applying forces
        // This prevents overshoot when Python control loop is slower than physics timestep
        float effectiveThrottle = throttleInput;
        if (brakeInput > 0.02f)
        {
            // Avoid throttle+brake overlap to prevent jerk spikes.
            effectiveThrottle = 0f;
        }
        
        float effectiveMaxSpeed = (speedLimit > 0.0f) ? speedLimit : maxSpeed;
        if (currentSpeed > effectiveMaxSpeed * speedPreventionThreshold)
        {
            // Reduce throttle input to prevent further acceleration
            effectiveThrottle = throttleInput * speedPreventionThrottleFactor;
        }
        if (currentSpeed > effectiveMaxSpeed * speedPreventionCutoff)
        {
            // Almost at limit - cut throttle completely
            effectiveThrottle = 0.0f;
        }

        // Apply throttle slew limiting to avoid step inputs causing jerk spikes
        float desiredThrottle = effectiveThrottle;
        float maxDelta = throttleSlewRatePerSecond * Time.fixedDeltaTime;
        float smoothedThrottle = Mathf.MoveTowards(lastThrottleApplied, desiredThrottle, maxDelta);
        lastThrottleApplied = smoothedThrottle;

        // DEBUG: Log throttle values to diagnose why car isn't moving
        if (Time.frameCount % 60 == 0) // Log every 60 frames (~2 seconds at 30fps)
        {
            Debug.Log($"CarController: throttleInput={throttleInput:F3}, brakeInput={brakeInput:F3}, currentSpeed={currentSpeed:F3} m/s, effectiveThrottle={effectiveThrottle:F3}, smoothedThrottle={smoothedThrottle:F3}, motorForce={motorForce:F1}");
        }
        
        if (useWheelColliders && wheelCollidersReady)
        {
            float motorTorque = smoothedThrottle * maxMotorTorque;
            float brakeTorque = brakeInput * maxBrakeTorque;
            frontLeftWheel.steerAngle = steerAngle;
            frontRightWheel.steerAngle = steerAngle;
            frontLeftWheel.motorTorque = motorTorque;
            frontRightWheel.motorTorque = motorTorque;
            rearLeftWheel.motorTorque = motorTorque;
            rearRightWheel.motorTorque = motorTorque;
            frontLeftWheel.brakeTorque = brakeTorque;
            frontRightWheel.brakeTorque = brakeTorque;
            rearLeftWheel.brakeTorque = brakeTorque;
            rearRightWheel.brakeTorque = brakeTorque;
            if (logWheelColliderDebug && Time.frameCount % 60 == 0)
            {
                Debug.Log(
                    $"WheelColliders: grounded FL={frontLeftWheel.isGrounded} FR={frontRightWheel.isGrounded} " +
                    $"RL={rearLeftWheel.isGrounded} RR={rearRightWheel.isGrounded}, " +
                    $"motorTorque={motorTorque:F1}, brakeTorque={brakeTorque:F1}"
                );
            }
        }
        else if (Mathf.Abs(smoothedThrottle) > 0.01f)
        {
            // CRITICAL: Wake up Rigidbody if it's sleeping (sleeping Rigidbody doesn't respond to forces!)
            if (rb.IsSleeping())
            {
                rb.WakeUp();
                if (Time.frameCount % 60 == 0)
                {
                    Debug.Log($"CarController: Woke up sleeping Rigidbody (throttle={smoothedThrottle:F3})");
                }
            }

            float minEffectiveThrottle = 0.6f; // Minimum throttle to overcome static friction
            float actualThrottle = smoothedThrottle;
            if (currentSpeed < 0.1f && smoothedThrottle > 0.01f && smoothedThrottle < minEffectiveThrottle)
            {
                actualThrottle = minEffectiveThrottle;
                if (Time.frameCount % 60 == 0)
                {
                    Debug.Log($"CarController: Boosting throttle from {smoothedThrottle:F3} to {actualThrottle:F3} to overcome static friction (speed={currentSpeed:F3} m/s, steering={steerInput:F3})");
                }
            }

            rb.WakeUp();

            Vector3 forwardForce = transform.forward * actualThrottle * motorForce;
            rb.AddForce(forwardForce, ForceMode.Acceleration);

            if (currentSpeed < 0.1f || Time.frameCount % 60 == 0)
            {
                Vector3 currentVel = rb.linearVelocity;
                Debug.Log($"CarController: Applied force = {forwardForce.magnitude:F2} m/s² (throttle={actualThrottle:F3} * motorForce={motorForce:F1}), " +
                         $"velocity=({currentVel.x:F3}, {currentVel.y:F3}, {currentVel.z:F3}), speed={currentVel.magnitude:F3} m/s, " +
                         $"isSleeping={rb.IsSleeping()}, drag={rb.linearDamping:F2}, constraints={rb.constraints}");
            }
        }
        
        // CRITICAL: Wake up Rigidbody if braking (might be sleeping)
        if (brakeInput > 0 && rb.IsSleeping())
        {
            rb.WakeUp();
        }
        
        if (!useWheelColliders || !wheelCollidersReady)
        {
            if (brakeInput > 0)
            {
                // Cap brake to avoid aggressive physics spikes
                float brakeApplied = Mathf.Min(brakeInput, 0.2f);

                // Increase linear damping for braking, but keep it gentle to avoid spikes
                rb.linearDamping = brakeApplied * 5f;
            }
            else
            {
                // CRITICAL: Reset damping to 0 when not braking (allows car to coast)
                // This MUST be done every frame to prevent drag from persisting
                rb.linearDamping = 0f;
                
                // Also ensure old 'drag' property is reset (for compatibility)
            #if !UNITY_2021_1_OR_NEWER
            rb.drag = 0f;
            #endif
            }
        }
        
        // FIXED: Hard speed limit - cap velocity at the END after all forces are applied
        // This ensures the limit is enforced regardless of any forces applied earlier
        // CRITICAL: This is the LAST line - it overrides any physics forces that exceeded the limit
        // Recalculate speed after applying forces (in case forces increased it)
        currentVelocity = rb.linearVelocity;
        currentSpeed = currentVelocity.magnitude;
        if (currentSpeed > effectiveMaxSpeed)
        {
            // Cap velocity to max speed immediately
            rb.linearVelocity = currentVelocity.normalized * effectiveMaxSpeed;
        }

        // Diagnostics: detect sudden speed drops (physics discontinuities/collisions)
        if (logSpeedDropEvents)
        {
            float now = Time.time;
            float dt = (lastPhysicsTime > 0f) ? (now - lastPhysicsTime) : Time.fixedDeltaTime;
            if (dt > 0f)
            {
                float speedDrop = lastPhysicsSpeed - currentSpeed;
                float accel = speedDrop / dt; // positive when slowing
                if (speedDrop >= speedDropThreshold || accel >= speedDropAccelThreshold)
                {
                    string collisionNote = (lastCollisionTime > 0f && (now - lastCollisionTime) < 0.5f)
                        ? $" recentCollision=({lastCollisionInfo})"
                        : "";
                    Debug.LogWarning(
                        $"[SPEED_DROP] frame={Time.frameCount} dt={dt:F4}s " +
                        $"speed={currentSpeed:F2}m/s drop={speedDrop:F2}m/s accel={-accel:F2}m/s^2 " +
                        $"throttle={throttleInput:F3} brake={brakeInput:F3} " +
                        $"damping={rb.linearDamping:F2}{collisionNote}"
                    );
                }
            }
            lastPhysicsSpeed = currentSpeed;
            lastPhysicsTime = now;
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (!logCollisionEvents)
        {
            return;
        }
        float impulseMag = collision.impulse.magnitude;
        float relVel = collision.relativeVelocity.magnitude;
        lastCollisionTime = Time.time;
        lastCollisionInfo = $"{collision.collider.name} impulse={impulseMag:F2} relVel={relVel:F2} contacts={collision.contactCount}";
        Debug.LogWarning($"[COLLISION_ENTER] frame={Time.frameCount} {lastCollisionInfo}");
    }

    private void OnCollisionStay(Collision collision)
    {
        if (!logCollisionEvents)
        {
            return;
        }
        if (Time.frameCount % 30 != 0)
        {
            return;
        }
        float impulseMag = collision.impulse.magnitude;
        float relVel = collision.relativeVelocity.magnitude;
        lastCollisionTime = Time.time;
        lastCollisionInfo = $"{collision.collider.name} impulse={impulseMag:F2} relVel={relVel:F2} contacts={collision.contactCount}";
        Debug.LogWarning($"[COLLISION_STAY] frame={Time.frameCount} {lastCollisionInfo}");
    }

    private void OnTriggerEnter(Collider other)
    {
        if (!logCollisionEvents)
        {
            return;
        }
        lastCollisionTime = Time.time;
        lastCollisionInfo = $"{other.name} trigger";
        Debug.LogWarning($"[TRIGGER_ENTER] frame={Time.frameCount} collider={other.name}");
    }

    /// <summary>
    /// Set AV control inputs (called by AVBridge)
    /// </summary>
    public void SetAVControls(float steering, float throttle, float brake)
    {
        avSteering = Mathf.Clamp(steering, -1f, 1f);
        avThrottle = Mathf.Clamp(throttle, -1f, 1f);
        avBrake = Mathf.Clamp01(brake);
        lastAvCommandTime = Time.time;
    }
    
    /// <summary>
    /// Enable or disable ground truth mode (direct velocity control for precise path following)
    /// </summary>
    public void SetGroundTruthMode(bool enabled, float speed = 5.0f)
    {
        bool wasEnabled = groundTruthMode;
        groundTruthMode = enabled;
        groundTruthSpeed = speed;
        if (rb != null)
        {
            rb.constraints = enabled
                ? (baseConstraints | RigidbodyConstraints.FreezeRotationY)
                : baseConstraints;
            rb.isKinematic = enabled ? true : baseIsKinematic;
            rb.angularVelocity = Vector3.zero;
        }
        
        // Always log mode changes (not just when enabling)
        if (enabled && !wasEnabled)
        {
            if (cachedGroundTruthReporter == null)
            {
                cachedGroundTruthReporter = GetComponent<GroundTruthReporter>();
                if (cachedGroundTruthReporter == null)
                {
                    cachedGroundTruthReporter = FindObjectOfType<GroundTruthReporter>();
                }
            }
            
            if (cachedGroundTruthReporter != null)
            {
                cachedGroundTruthReporter.ResetTimeBasedReferenceToCar("Ground truth mode enabled");
            }
            else
            {
                Debug.LogWarning("CarController: GroundTruthReporter not found when enabling ground truth mode");
            }
            Debug.Log($"CarController: ✅ Ground Truth Mode ENABLED - Direct velocity control at {speed} m/s");
        }
        else if (!enabled && wasEnabled)
        {
            Debug.Log("CarController: ⚠️ Ground Truth Mode DISABLED - Using physics-based control");
        }
        else if (enabled && wasEnabled && Mathf.Abs(groundTruthSpeed - speed) > 0.1f)
        {
            Debug.Log($"CarController: Ground Truth Mode speed updated: {groundTruthSpeed} → {speed} m/s");
        }
    }

    /// <summary>
    /// Get current vehicle state (for AV stack)
    /// </summary>
    public VehicleState GetVehicleState()
    {
        float actualSteeringAngle = steeringAngleActual;
        if (useWheelColliders && wheelCollidersReady)
        {
            actualSteeringAngle = -(frontLeftWheel.steerAngle + frontRightWheel.steerAngle) * 0.5f;
        }

        return new VehicleState
        {
            position = transform.position,
            rotation = transform.rotation,
            velocity = rb.linearVelocity,
            angularVelocity = rb.angularVelocity,
            speed = groundTruthMode ? groundTruthSpeed : rb.linearVelocity.magnitude,
            steeringAngle = desiredSteerAngle,
            steeringAngleActual = actualSteeringAngle,
            steeringInput = steerInput,
            desiredSteerAngle = desiredSteerAngle,
            motorTorque = throttleInput * motorForce,
            brakeTorque = brakeInput * brakeForce,
            maxSteerAngle = maxSteerAngle,
            wheelbaseMeters = wheelbaseMeters,
            fixedDeltaTime = Time.fixedDeltaTime,
            unityTime = Time.time,
            unityFrameCount = Time.frameCount,
            unityDeltaTime = Time.deltaTime,
            unitySmoothDeltaTime = Time.smoothDeltaTime,
            unityUnscaledDeltaTime = Time.unscaledDeltaTime,
            unityTimeScale = Time.timeScale,
            gtRotationDebugValid = gtRotationDebugValid,
            gtRotationUsedRoadFrame = gtRotationUsedRoadFrame,
            gtRotationRejectedRoadFrameHop = gtRotationRejectedRoadFrameHop,
            gtRotationReferenceHeadingDeg = gtRotationReferenceHeadingDeg,
            gtRotationRoadFrameHeadingDeg = gtRotationRoadFrameHeadingDeg,
            gtRotationInputHeadingDeg = gtRotationInputHeadingDeg,
            gtRotationRoadVsRefDeltaDeg = gtRotationRoadVsRefDeltaDeg,
            gtRotationAppliedDeltaDeg = gtRotationAppliedDeltaDeg
        };
    }

    private static float HeadingDegFromDirection(Vector3 dir)
    {
        if (dir.sqrMagnitude < 1e-6f)
        {
            return 0f;
        }
        Vector3 n = dir.normalized;
        float heading = Mathf.Atan2(n.x, n.z) * Mathf.Rad2Deg;
        if (heading < 0f)
        {
            heading += 360f;
        }
        return heading;
    }

    private static bool? ParseCommandLineBool(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == name)
            {
                string value = args[i + 1].Trim().ToLowerInvariant();
                if (value == "true" || value == "1")
                {
                    return true;
                }
                if (value == "false" || value == "0")
                {
                    return false;
                }
                return null;
            }
        }
        return null;
    }

}

/// <summary>
/// Vehicle state data structure for AV stack
/// </summary>
[System.Serializable]
public class VehicleState
{
    public Vector3 position;
    public Quaternion rotation;
    public Vector3 velocity;
    public Vector3 angularVelocity;
    public float speed;
    public float steeringAngle;
    public float steeringAngleActual;
    public float steeringInput;
    public float desiredSteerAngle;
    public float motorTorque;
    public float brakeTorque;
    public float maxSteerAngle = 35f;
    public float wheelbaseMeters = 2.5f;
    public float fixedDeltaTime = 0.02f;
    public float unityTime = 0.0f;
    public int unityFrameCount = 0;
    public float unityDeltaTime = 0.0f;
    public float unitySmoothDeltaTime = 0.0f;
    public float unityUnscaledDeltaTime = 0.0f;
    public float unityTimeScale = 1.0f;
    // GT rotation input telemetry (exact values used to construct target rotation).
    public bool gtRotationDebugValid = false;
    public bool gtRotationUsedRoadFrame = false;
    public bool gtRotationRejectedRoadFrameHop = false;
    public float gtRotationReferenceHeadingDeg = 0.0f;
    public float gtRotationRoadFrameHeadingDeg = 0.0f;
    public float gtRotationInputHeadingDeg = 0.0f;
    public float gtRotationRoadVsRefDeltaDeg = 0.0f;
    public float gtRotationAppliedDeltaDeg = 0.0f;
    // Bridge correlation fields (set by AVBridge before send)
    public int requestId = 0;
    public float unitySendRealtime = 0.0f; // Time.realtimeSinceStartup at send time
    public long unitySendUtcMs = 0; // UTC epoch ms at send time
    
    // Ground truth lane line positions (optional, set by GroundTruthReporter)
    // These represent the painted lane line markings, not the drivable lanes
    public float groundTruthLeftLaneLineX = 0f;  // Left lane line (painted marking) position
    public float groundTruthRightLaneLineX = 0f;  // Right lane line (painted marking) position
    public float groundTruthLaneCenterX = 0f;  // Lane center (midpoint between lane lines)
    // NEW: Path-based steering data
    public float groundTruthDesiredHeading = 0.0f;  // Desired heading from path (degrees)
    public float groundTruthPathCurvature = 0.0f;  // Path curvature (1/meters)
    
    // NEW: Camera calibration - actual screen y pixel where 8m appears (from Unity's WorldToScreenPoint)
    public float camera8mScreenY = -1.0f;  // -1.0 means not calculated yet
    // NEW: Camera calibration - screen y pixel where ground truth lookahead appears
    public float cameraLookaheadScreenY = -1.0f;  // -1.0 means not calculated yet
    // NEW: Ground truth lookahead distance used for calibration
    public float groundTruthLookaheadDistance = 8.0f;
    // Oracle centerline samples in vehicle frame (flattened [x0,y0,x1,y1,...]).
    public float[] oracleTrajectoryXY = new float[0];
    public float[] oracleTrajectoryWorldXYZ = new float[0];
    public float[] oracleTrajectoryScreenXY = new float[0];
    public int oraclePointCount = 0;
    public float oracleHorizonMeters = 0.0f;
    public float oraclePointSpacingMeters = 0.0f;
    public bool oracleSamplesEnabled = false;
    // Right lane-line fiducials for projection diagnostics.
    public float[] rightLaneFiducialsVehicleXY = new float[0];
    public float[] rightLaneFiducialsVehicleTrueXY = new float[0];
    public float[] rightLaneFiducialsVehicleMonotonicXY = new float[0];
    public float[] rightLaneFiducialsWorldXYZ = new float[0];
    public float[] rightLaneFiducialsScreenXY = new float[0];
    public int rightLaneFiducialsPointCount = 0;
    public float rightLaneFiducialsHorizonMeters = 0.0f;
    public float rightLaneFiducialsSpacingMeters = 0.0f;
    public bool rightLaneFiducialsEnabled = false;
    // NEW: Camera FOV information - what Unity actually uses
    public float cameraFieldOfView = 0.0f;  // Unity's Camera.fieldOfView value (always vertical FOV)
    public float cameraHorizontalFOV = 0.0f;  // Calculated horizontal FOV
    // NEW: Camera position and forward for debugging alignment
    public float cameraPosX = 0.0f;  // Camera position X (world coords)
    public float cameraPosY = 0.0f;  // Camera position Y (world coords)
    public float cameraPosZ = 0.0f;  // Camera position Z (world coords)
    public float cameraForwardX = 0.0f;  // Camera forward X (normalized)
    public float cameraForwardY = 0.0f;  // Camera forward Y (normalized)
    public float cameraForwardZ = 0.0f;  // Camera forward Z (normalized)
    // Top-down camera calibration/projection (for top-down overlay diagnostics)
    public float topDownCameraPosX = 0.0f;
    public float topDownCameraPosY = 0.0f;
    public float topDownCameraPosZ = 0.0f;
    public float topDownCameraForwardX = 0.0f;
    public float topDownCameraForwardY = 0.0f;
    public float topDownCameraForwardZ = 0.0f;
    public float topDownCameraOrthographicSize = 0.0f;
    public float topDownCameraFieldOfView = 0.0f;
    
    // NEW: Debug fields for diagnosing ground truth offset issues
    // Road center world positions (for debugging alignment)
    public float roadCenterAtCarX = 0.0f;  // Road center X at car's location (world coords)
    public float roadCenterAtCarY = 0.0f;  // Road center Y at car's location (world coords)
    public float roadCenterAtCarZ = 0.0f;  // Road center Z at car's location (world coords)
    public float roadCenterAtLookaheadX = 0.0f;  // Road center X at 8m lookahead (world coords)
    public float roadCenterAtLookaheadY = 0.0f;  // Road center Y at 8m lookahead (world coords)
    public float roadCenterAtLookaheadZ = 0.0f;  // Road center Z at 8m lookahead (world coords)
    public float roadCenterReferenceT = 0.0f;  // Parameter t on road path for reference point
    // NEW: Road-frame alignment metrics (for diagnosing outside-of-lane bias)
    public float roadFrameLateralOffset = 0.0f;  // Lateral offset from road center (m, +right)
    public float roadHeadingDeg = 0.0f;  // Road tangent heading (deg)
    public float carHeadingDeg = 0.0f;  // Car heading (deg)
    public float headingDeltaDeg = 0.0f;  // Car - road heading delta (deg, -180..180)
    public float roadFrameLaneCenterOffset = 0.0f;  // Road center lookahead offset in road frame (m, +right)
    public float roadFrameLaneCenterError = 0.0f;  // Car offset vs lookahead center in road frame (m, +right)
    public float vehicleFrameLookaheadOffset = 0.0f;  // Lookahead road center offset in vehicle frame (m, +right)
    public float speedLimit = 0.0f;  // Speed limit at current reference point (m/s)
    public float speedLimitPreview = 0.0f;  // Speed limit at preview distance ahead (m/s)
    public float speedLimitPreviewDistance = 0.0f;  // Preview distance used for speed limit (m)
    public float speedLimitPreviewMinDistance = 0.0f;  // Distance to min limit in preview window (m)
    public float speedLimitPreviewMid = 0.0f;  // Speed limit at mid preview distance (m/s)
    public float speedLimitPreviewMidDistance = 0.0f;  // Mid preview distance (m)
    public float speedLimitPreviewMidMinDistance = 0.0f;  // Distance to min limit in mid window (m)
    public float speedLimitPreviewLong = 0.0f;  // Speed limit at long preview distance (m/s)
    public float speedLimitPreviewLongDistance = 0.0f;  // Long preview distance (m)
    public float speedLimitPreviewLongMinDistance = 0.0f;  // Distance to min limit in long window (m)
}
