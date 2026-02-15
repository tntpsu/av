using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using System;
using System.Text;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Bridge between Unity simulation and Python AV stack
/// Handles bidirectional communication for vehicle state and control commands
/// </summary>
public class AVBridge : MonoBehaviour
{
    private static int lastOverlayFrame = -1;
    [Header("Components")]
    public CarController carController;
    public CameraCapture cameraCapture;
    public TrajectoryVisualizer trajectoryVisualizer; // Optional trajectory visualizer
    public GroundTruthReporter groundTruthReporter; // Optional ground truth reporter

    [Header("Top-Down Debug Camera")]
    public bool enableTopDownCamera = true;
    public string topDownCameraName = "TopDownCamera";
    [Tooltip("Height above the car in meters.")]
    public float topDownHeight = 18f;
    [Tooltip("Orthographic size for the top-down camera.")]
    public float topDownOrthographicSize = 12f;
    [Tooltip("Viewport X position for the top-down camera (0..1).")]
    public float topDownViewportX = 0.7f;
    [Tooltip("Viewport width for the top-down camera (0..1).")]
    public float topDownViewportWidth = 0.3f;
    public int topDownCaptureWidth = 640;
    public int topDownCaptureHeight = 480;
    public int topDownTargetFps = 15;

    [Header("Oracle Trajectory Telemetry")]
    [Tooltip("Send Unity centerline samples (oracle) in vehicle frame.")]
    public bool enableOracleSamples = true;
    [Tooltip("Oracle sampling horizon in meters.")]
    public float oracleHorizonMeters = 30.0f;
    [Tooltip("Oracle sample spacing in meters.")]
    public float oraclePointSpacingMeters = 1.0f;

    [Header("Right Lane Fiducials (Projection Diagnostics)")]
    [Tooltip("Send right-lane fiducials and Unity screen projections for projection validation.")]
    public bool enableRightLaneFiducials = true;
    [Tooltip("Fiducial horizon in meters.")]
    public float rightLaneFiducialsHorizonMeters = 30.0f;
    [Tooltip("Fiducial spacing in meters.")]
    public float rightLaneFiducialsSpacingMeters = 5.0f;
    
    [Header("API Settings")]
    public string apiUrl = "http://localhost:8000";
    public string stateEndpoint = "/api/vehicle/state";
    public string controlEndpoint = "/api/vehicle/control";
    public string shutdownEndpoint = "/api/shutdown";
    public string playEndpoint = "/api/unity/play";  // NEW: Unity play request endpoint
    public string trajectoryEndpoint = "/api/trajectory";
    public string feedbackEndpoint = "/api/unity/feedback";  // NEW: Unity feedback endpoint
    public float updateRate = 30f; // Hz
    public float shutdownCheckInterval = 1.0f; // Check for shutdown every 1 second
    public float playCheckInterval = 0.5f; // Check for play request every 0.5 seconds (more frequent for responsiveness)
    public float feedbackSendInterval = 1.0f; // Send feedback every 1 second (not every frame to reduce overhead)
    
    [Header("Control")]
    [Tooltip("Enable AV control (overrides manual input)")]
    public bool enableAVControl = false;
    
    [Header("Debug")]
    public bool showDebugInfo = true;
    public bool logBridgeTimings = true;
    public float bridgeTimingWarnThreshold = 0.2f;
    public float bridgeStallLogCooldownSeconds = 1.0f;
    public bool logUpdateHitches = true;
    public float updateHitchThresholdSeconds = 0.2f;
    public float updateHitchCooldownSeconds = 1.0f;
    public bool logUpdateGapSummary = true;
    public int updateGapSummaryFrames = 120;
    public bool logFrameCountJumps = true;
    public int frameCountJumpThreshold = 1;
    public float unityTimeJumpThresholdSeconds = 0.2f;

    [Header("Speed Limit Preview")]
    [Tooltip("Preview distance ahead (m) for upcoming speed limit checks")]
    public float speedLimitPreviewDistance = 12.0f;
public float speedLimitPreviewDistanceMid = 0.0f;
public float speedLimitPreviewDistanceLong = 0.0f;
public bool speedLimitPreviewAllowWrap = false;
[Tooltip("Max normalized t jump allowed before preview is suppressed")]
public float speedLimitPreviewMaxTDelta = 0.15f;
    [Tooltip("Number of samples when scanning preview window for min speed limit")]
    public int speedLimitPreviewSamples = 8;
    
    private float updateInterval;
    private float lastUpdateTime;
    private bool updateInFlight = false;
    private float lastShutdownCheckTime;
    private float lastPlayCheckTime;
    private float lastFeedbackSendTime;
    private VehicleState lastVehicleState;
    private ControlCommand lastControlCommand;
    private int lastRandomizeRequestId = -1;
    private bool randomStartHandled = false;
    private int updateSequence = 0;
    private int currentUpdateSequence = 0;
    private float updateStartRealtime = 0f;
    private float lastUpdateStallLogRealtime = 0f;
    private float lastUpdateRealtime = 0f;
    private float lastUpdateHitchLogRealtime = 0f;
    private int updateGapSampleCount = 0;
    private float updateGapSum = 0f;
    private float updateGapMax = 0f;
    private int lastObservedFrameCount = -1;
    private float lastSentUnityTime = -1f;
    private int lastSentUnityFrameCount = -1;
    private bool stateSendInFlight = false;
    private float stateSendStartRealtime = 0f;
    private bool controlRequestInFlight = false;
    private float controlRequestStartRealtime = 0f;
    private float lastNonZeroSpeedLimit = 0f;
private float? lastCarT = null;
    private bool useFixedUpdateBridgeLoop = false;

    private Camera topDownCamera;
    private CameraCapture topDownCapture;
    
    // PERFORMANCE: Cache camera8mScreenY calculation - only recalculate when camera moves significantly
    private float cachedCamera8mScreenY = -1.0f;
    private Vector3 lastCameraPosition = Vector3.zero;
    private Quaternion lastCameraRotation = Quaternion.identity;
    private float camera8mScreenYRecalcInterval = 0.5f; // Recalculate every 0.5 seconds instead of every frame
    private float lastCamera8mScreenYRecalcTime = 0f;
    
    #if UNITY_EDITOR
    private double lastEditorPlayCheckTime = 0.0; // Use double for EditorApplication.timeSinceStartup
    private Queue<UnityWebRequest> pendingPlayChecks = new Queue<UnityWebRequest>(); // Queue for async requests in edit mode
    #endif

    private float ComputePreviewSpeedLimit(
        float carT,
        float previewDistance,
        float pathLength,
        float fallbackLimit,
        bool baseValid,
        out float previewMinDistance
    )
    {
        previewMinDistance = Mathf.Max(0f, previewDistance);
        if (!baseValid || previewDistance <= 0.01f || pathLength <= 0.01f)
        {
            return fallbackLimit;
        }

        float previewT = carT + (previewDistance / pathLength);
        if (!speedLimitPreviewAllowWrap && previewT > 1.0f)
        {
            return fallbackLimit;
        }

        int samples = Mathf.Max(2, speedLimitPreviewSamples);
        float startDistance = carT * pathLength;
        float endDistance = startDistance + previewDistance;
        float previewSpeedLimit = fallbackLimit;

        if (speedLimitPreviewAllowWrap)
        {
            float total = pathLength;
            float window = Mathf.Min(previewDistance, total);
            for (int i = 0; i < samples; i++)
            {
                float alpha = samples > 1 ? (float)i / (samples - 1) : 0f;
                float distance = startDistance + (alpha * window);
                float tSample = Mathf.Repeat(distance / total, 1f);
                float limit = groundTruthReporter.GetSpeedLimitAtT(tSample);
                if (limit > 0f && limit < previewSpeedLimit)
                {
                    previewSpeedLimit = limit;
                    previewMinDistance = alpha * window;
                }
            }
        }
        else
        {
            float clampedEnd = Mathf.Min(endDistance, pathLength);
            float span = Mathf.Max(0.01f, clampedEnd - startDistance);
            for (int i = 0; i < samples; i++)
            {
                float alpha = samples > 1 ? (float)i / (samples - 1) : 0f;
                float distance = startDistance + (alpha * span);
                float tSample = Mathf.Clamp01(distance / pathLength);
                float limit = groundTruthReporter.GetSpeedLimitAtT(tSample);
                if (limit > 0f && limit < previewSpeedLimit)
                {
                    previewSpeedLimit = limit;
                    previewMinDistance = distance - startDistance;
                }
            }
        }

        return previewSpeedLimit;
    }
    
    void Start()
    {
        bool? cliDisableTopDown = ParseCommandLineBool(System.Environment.GetCommandLineArgs(), "--gt-disable-topdown");
        if (cliDisableTopDown.HasValue && cliDisableTopDown.Value)
        {
            enableTopDownCamera = false;
            DisableExistingTopDownCaptures();
        }

        // Force debug overlay on for player runs (prefab may have it disabled).
        showDebugInfo = true;
        // Keep sending data when the player window loses focus.
        Application.runInBackground = true;
        if (showDebugInfo)
        {
            Debug.Log($"AVBridge: runInBackground={Application.runInBackground}");
        }

        // Get components if not assigned
        if (carController == null)
        {
            carController = GetComponent<CarController>();
        }
        
        if (cameraCapture == null)
        {
            cameraCapture = FindObjectOfType<CameraCapture>();
        }

        SetupTopDownCamera();
        
        if (groundTruthReporter == null)
        {
            groundTruthReporter = FindObjectOfType<GroundTruthReporter>();
            if (groundTruthReporter != null)
            {
                if (showDebugInfo) Debug.Log($"AVBridge: Found GroundTruthReporter on '{groundTruthReporter.gameObject.name}' (enabled: {groundTruthReporter.enabled})");
            }
            else
            {
                if (showDebugInfo) Debug.LogWarning("AVBridge: GroundTruthReporter not found in scene! Ground truth data will be 0.0");
            }
        }
        if (groundTruthReporter != null && !groundTruthReporter.enabled)
        {
            groundTruthReporter.enabled = true;
            if (showDebugInfo) Debug.Log("AVBridge: GroundTruthReporter was disabled; forcing enabled for track data.");
        }
        else
        {
            if (showDebugInfo) Debug.Log($"AVBridge: GroundTruthReporter assigned in Inspector on '{groundTruthReporter.gameObject.name}' (enabled: {groundTruthReporter.enabled})");
        }
        
        if (carController == null)
        {
            Debug.LogError("AVBridge: CarController not found!");
            enabled = false;
            return;
        }
        
        if (cameraCapture != null && cameraCapture.gtSyncCapture)
        {
            // Align bridge state/control loop with GT sync cadence to reduce periodic pose jumps.
            updateRate = 1.0f / Mathf.Max(0.001f, Time.fixedDeltaTime);
            useFixedUpdateBridgeLoop = true;
        }
        updateInterval = 1.0f / Mathf.Max(1f, updateRate);
        lastVehicleState = new VehicleState();
        lastShutdownCheckTime = Time.time;
        lastPlayCheckTime = Time.time;
        // CRITICAL: Initialize lastUpdateTime to negative value to ensure UpdateAVStack() runs on first frame
        // This ensures camera calibration data is calculated and sent from frame 0
        lastUpdateTime = -updateInterval;  // Force first UpdateAVStack() call on frame 0
        
        // CRITICAL: Initialize all cached values BEFORE first frame
        // This ensures Python receives valid data from frame 0, not -1.0 or null
        InitializeCachedValues();

        lastUpdateRealtime = Time.realtimeSinceStartup;
        
        // Enable AV control on car controller
        if (enableAVControl)
        {
            carController.avControlEnabled = true;
            carController.avControlPriority = true;
        }
        
        // Start trajectory visualization update (if visualizer exists)
        if (trajectoryVisualizer != null)
        {
            // TrajectoryVisualizer will fetch data itself, no need to do anything here
            if (showDebugInfo)
            {
                Debug.Log("AVBridge: TrajectoryVisualizer found - trajectory visualization enabled");
            }
        }
        
        if (showDebugInfo) Debug.Log($"AVBridge: Initialized - Update rate: {updateRate} Hz, AV Control: {enableAVControl}");
        
        // Start sending Unity feedback periodically
        StartCoroutine(SendUnityFeedback());
        
        // CRITICAL FIX: Register EditorApplication.update to check for play requests even when not in play mode
        // This allows auto-play to work when Unity Editor is open but not playing
        #if UNITY_EDITOR
        EditorApplication.update += OnEditorUpdate;
        #endif
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

    private void DisableExistingTopDownCaptures()
    {
        CameraCapture[] captures = FindObjectsOfType<CameraCapture>();
        foreach (CameraCapture capture in captures)
        {
            if (capture == null)
            {
                continue;
            }
            if (!string.Equals(capture.cameraId, "top_down", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            capture.enabled = false;
            if (capture.targetCamera != null)
            {
                capture.targetCamera.enabled = false;
            }
            if (showDebugInfo)
            {
                Debug.Log($"AVBridge: Disabled existing top-down capture '{capture.name}'.");
            }
        }
    }

    private void SetupTopDownCamera()
    {
        if (!enableTopDownCamera)
        {
            return;
        }

        Camera mainCamera = null;
        if (cameraCapture != null && cameraCapture.targetCamera != null)
        {
            mainCamera = cameraCapture.targetCamera;
        }
        else
        {
            GameObject avCam = GameObject.Find("AVCamera");
            if (avCam != null)
            {
                mainCamera = avCam.GetComponent<Camera>();
            }
            if (mainCamera == null)
            {
                mainCamera = Camera.main;
            }
        }

        float clampedX = Mathf.Clamp01(topDownViewportX);
        float clampedWidth = Mathf.Clamp01(topDownViewportWidth);
        if (mainCamera != null)
        {
            float mainWidth = Mathf.Clamp01(clampedX);
            if (mainWidth <= 0.0f)
            {
                mainWidth = Mathf.Clamp01(1.0f - clampedWidth);
                clampedX = Mathf.Clamp01(1.0f - clampedWidth);
            }
            mainCamera.rect = new Rect(0f, 0f, mainWidth, 1f);
        }

        GameObject existing = GameObject.Find(topDownCameraName);
        if (existing != null)
        {
            topDownCamera = existing.GetComponent<Camera>();
        }

        if (topDownCamera == null)
        {
            GameObject topDownObj = new GameObject(topDownCameraName);
            topDownObj.transform.SetParent(transform);
            topDownObj.transform.localPosition = new Vector3(0f, topDownHeight, 0f);
            topDownObj.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
            topDownCamera = topDownObj.AddComponent<Camera>();
        }

        topDownCamera.orthographic = true;
        topDownCamera.orthographicSize = topDownOrthographicSize;
        topDownCamera.rect = new Rect(clampedX, 0f, clampedWidth, 1f);
        if (mainCamera != null)
        {
            topDownCamera.depth = mainCamera.depth + 1f;
            topDownCamera.cullingMask = mainCamera.cullingMask;
        }

        topDownCapture = topDownCamera.GetComponent<CameraCapture>();
        if (topDownCapture == null)
        {
            topDownCapture = topDownCamera.gameObject.AddComponent<CameraCapture>();
        }

        if (cameraCapture != null)
        {
            topDownCapture.apiUrl = cameraCapture.apiUrl;
            topDownCapture.cameraEndpoint = cameraCapture.cameraEndpoint;
            topDownCapture.useAsyncGPUReadback = cameraCapture.useAsyncGPUReadback;
        }
        topDownCapture.targetCamera = topDownCamera;
        topDownCapture.cameraId = "top_down";
        topDownCapture.captureWidth = topDownCaptureWidth;
        topDownCapture.captureHeight = topDownCaptureHeight;
        topDownCapture.targetFPS = topDownTargetFps;
        topDownCapture.showDebugInfo = false;
        topDownCapture.logFrameMarkers = false;
        topDownCapture.enforceCameraFov = false;
        topDownCapture.enforceCameraAspect = false;
        topDownCapture.enforceScreenResolution = false;
        topDownCapture.enforceQualityLevel = false;
    }
    
    #if UNITY_EDITOR
    private void OnEditorUpdate()
    {
        // Check for play request periodically (works even when not in play mode)
        // Use EditorApplication.timeSinceStartup instead of Time.time (which only works in play mode)
        double currentTime = EditorApplication.timeSinceStartup;
        if (currentTime - lastEditorPlayCheckTime >= playCheckInterval)
        {
            // Only check if not already playing
            if (!EditorApplication.isPlaying)
            {
                // Start async request (will be checked in next update)
                string url = $"{apiUrl}{playEndpoint}";
                UnityWebRequest request = UnityWebRequest.Get(url);
                request.timeout = 1;
                request.SendWebRequest(); // Start async request
                pendingPlayChecks.Enqueue(request);
            }
            lastEditorPlayCheckTime = currentTime;
        }
        
        // Check pending requests
        while (pendingPlayChecks.Count > 0)
        {
            UnityWebRequest request = pendingPlayChecks.Peek();
            if (request.isDone)
            {
                pendingPlayChecks.Dequeue();
                if (request.result == UnityWebRequest.Result.Success)
                {
                    try
                    {
                        string jsonResponse = request.downloadHandler.text;
                        if (jsonResponse.Contains("\"status\":\"play\"") || jsonResponse.Contains("play_requested"))
                        {
                            Debug.Log($"[COMMAND RECEIVED] AVBridge: ▶️ PLAY command received (OnEditorUpdate) - Python script requested play mode, entering play mode...");
                            Debug.Log($"[COMMAND RECEIVED] AVBridge: Play request response: {jsonResponse}");
                            EditorApplication.isPlaying = true;
                        }
                        else
                        {
                            Debug.Log($"[COMMAND RECEIVED] AVBridge: Play check response (no play request): {jsonResponse}");
                        }
                    }
                    catch (Exception e)
                    {
                        // Silently ignore errors in edit mode
                    }
                }
                request.Dispose();
            }
            else
            {
                // Request still pending, check next frame
                break;
            }
        }
    }
    
    private void OnDestroy()
    {
        // Unregister editor update callback and clean up pending requests
        #if UNITY_EDITOR
        EditorApplication.update -= OnEditorUpdate;
        while (pendingPlayChecks.Count > 0)
        {
            UnityWebRequest request = pendingPlayChecks.Dequeue();
            if (!request.isDone)
            {
                request.Abort();
            }
            request.Dispose();
        }
        #endif
    }
    #endif
    
    void Update()
    {
        // CRITICAL: Don't send data when Unity is exiting play mode
        // During play mode exit, Update() may still be called but Time.time is frozen
        // This prevents sending duplicate frames with frozen timestamps
        #if UNITY_EDITOR
        if (!EditorApplication.isPlaying)
        {
            return; // Don't send data when exiting play mode
        }
        #endif

        // Detect main-thread hitches (Update gaps) even if FixedUpdate doesn't log them.
        float realtimeNow = Time.realtimeSinceStartup;
        float realtimeGap = realtimeNow - lastUpdateRealtime;
        if (logFrameCountJumps && lastObservedFrameCount >= 0)
        {
            int frameDelta = Time.frameCount - lastObservedFrameCount;
            if (frameDelta > frameCountJumpThreshold)
            {
                Debug.LogWarning(
                    $"[UNITY FRAME JUMP] delta={frameDelta} frame={Time.frameCount} " +
                    $"time={Time.time:F3} unscaled={Time.unscaledTime:F3} realtimeGap={realtimeGap:F3}s"
                );
            }
        }
        lastObservedFrameCount = Time.frameCount;
        if (logUpdateHitches && realtimeGap > updateHitchThresholdSeconds)
        {
            if (realtimeNow - lastUpdateHitchLogRealtime > updateHitchCooldownSeconds)
            {
                float timeSinceStartup = Time.time;
                float unscaledTime = Time.unscaledTime;
                float fixedDelta = Time.fixedDeltaTime;
                int vSync = QualitySettings.vSyncCount;
                int targetFps = Application.targetFrameRate;
                string targetFpsText = targetFps <= 0 ? "platform default" : targetFps.ToString();
                Debug.LogWarning(
                    $"[UNITY HITCH] Update gap {realtimeGap:F3}s frame={Time.frameCount} " +
                    $"dt={Time.deltaTime:F4} sdt={Time.smoothDeltaTime:F4} " +
                    $"udt={Time.unscaledDeltaTime:F4} timescale={Time.timeScale:F2} " +
                    $"time={timeSinceStartup:F3} unscaled={unscaledTime:F3} fixedDt={fixedDelta:F3} " +
                    $"vSync={vSync} targetFps={targetFpsText}"
                );
                lastUpdateHitchLogRealtime = realtimeNow;
            }
        }
        if (logUpdateGapSummary && updateGapSummaryFrames > 0 && lastUpdateRealtime > 0f)
        {
            updateGapSampleCount += 1;
            updateGapSum += realtimeGap;
            if (realtimeGap > updateGapMax)
            {
                updateGapMax = realtimeGap;
            }

            if (updateGapSampleCount >= updateGapSummaryFrames)
            {
                float avgGap = updateGapSum / Mathf.Max(1, updateGapSampleCount);
                Debug.Log(
                    $"[UNITY UPDATE GAP] max={updateGapMax:F4}s avg={avgGap:F4}s " +
                    $"frames={updateGapSampleCount} frame={Time.frameCount} time={Time.time:F3}"
                );
                updateGapSampleCount = 0;
                updateGapSum = 0f;
                updateGapMax = 0f;
            }
        }
        lastUpdateRealtime = realtimeNow;
        
        // Update AV control state
        if (enableAVControl != carController.avControlEnabled)
        {
            carController.avControlEnabled = enableAVControl;
        }
        
        // Check for shutdown signal periodically
        if (Time.time - lastShutdownCheckTime >= shutdownCheckInterval)
        {
            StartCoroutine(CheckShutdownSignal());
            lastShutdownCheckTime = Time.time;
        }
        
        // Send vehicle state and receive control commands (Update-driven when not GT sync).
        if (!useFixedUpdateBridgeLoop)
        {
            if (!updateInFlight && Time.time - lastUpdateTime >= updateInterval)
            {
                StartCoroutine(UpdateAVStack());
                lastUpdateTime = Time.time;
            }
            else if (updateInFlight && logBridgeTimings && showDebugInfo)
            {
                float inFlightDuration = Time.realtimeSinceStartup - updateStartRealtime;
                if (inFlightDuration > bridgeTimingWarnThreshold &&
                    Time.realtimeSinceStartup - lastUpdateStallLogRealtime > bridgeStallLogCooldownSeconds)
                {
                    Debug.LogWarning(
                        $"AVBridge: UpdateAVStack still in flight (id={currentUpdateSequence}, " +
                        $"elapsed={inFlightDuration:F3}s, frame={Time.frameCount}, time={Time.time:F3}s)"
                    );
                    lastUpdateStallLogRealtime = Time.realtimeSinceStartup;
                }
            }
        }
    }

    void FixedUpdate()
    {
        if (!useFixedUpdateBridgeLoop)
        {
            return;
        }
        if (!updateInFlight)
        {
            StartCoroutine(UpdateAVStack());
            lastUpdateTime = Time.time;
        }
    }
    
    // CRITICAL: Initialize all cached values before first frame
    // This ensures Python receives valid data from frame 0, not -1.0 or null
    void InitializeCachedValues()
    {
        // Initialize camera8mScreenY cache - calculate it once before first frame
        if (cachedCamera8mScreenY < 0 && cameraCapture != null && cameraCapture.targetCamera != null && carController != null)
        {
            CalculateCamera8mScreenY();
        }
        
        // Initialize lastCameraPosition and lastCameraRotation for change detection
        if (cameraCapture != null && cameraCapture.targetCamera != null)
        {
            lastCameraPosition = cameraCapture.targetCamera.transform.position;
            lastCameraRotation = cameraCapture.targetCamera.transform.rotation;
        }
        
        // Initialize lastCamera8mScreenYRecalcTime to force first calculation
        lastCamera8mScreenYRecalcTime = -camera8mScreenYRecalcInterval; // Force calculation on first frame
        
        if (showDebugInfo)
        {
            Debug.Log($"AVBridge.InitializeCachedValues: cachedCamera8mScreenY = {cachedCamera8mScreenY:F1}px");
        }
    }
    
    // Extract camera8mScreenY calculation into separate method for reuse
    void CalculateCamera8mScreenY()
    {
        if (cameraCapture == null || cameraCapture.targetCamera == null || carController == null)
        {
            return;
        }
        
        float imageY = CalculateCameraScreenYAtDistance(8.0f);
        if (imageY > 0)
        {
            cachedCamera8mScreenY = imageY;
            lastCameraPosition = cameraCapture.targetCamera.transform.position;
            lastCameraRotation = cameraCapture.targetCamera.transform.rotation;
            lastCamera8mScreenYRecalcTime = Time.time;
            if (showDebugInfo)
            {
                Debug.Log($"AVBridge.CalculateCamera8mScreenY: Calculated camera8mScreenY = {cachedCamera8mScreenY:F1}px");
            }
        }
    }

    float CalculateCameraScreenYAtDistance(float distanceMeters)
    {
        if (cameraCapture == null || cameraCapture.targetCamera == null || carController == null)
        {
            return -1.0f;
        }
        if (distanceMeters <= 0.01f)
        {
            return -1.0f;
        }
        
        Camera avCamera = cameraCapture.targetCamera;
        Vector3 cameraPos = avCamera.transform.position;
        Vector3 cameraForward = avCamera.transform.forward;
        Vector3 pointDirection = cameraPos + cameraForward * distanceMeters;
        
        // Project point onto ground plane
        float carY = carController.transform.position.y;
        float groundHeight = (carY < 0.25f) ? 0.0f : (carY - 0.5f);
        float t = (groundHeight - cameraPos.y) / cameraForward.y;
        
        Vector3 pointOnGround;
        if (cameraForward.y >= 0)
        {
            pointOnGround = pointDirection;
            pointOnGround.y = groundHeight;
        }
        else if (t <= 0)
        {
            return -1.0f;
        }
        else
        {
            pointOnGround = cameraPos + cameraForward * t;
            float distanceToGround = Vector3.Distance(cameraPos, pointOnGround);
            if (distanceToGround > distanceMeters)
            {
                pointOnGround = pointDirection;
                pointOnGround.y = groundHeight;
            }
        }
        
        Vector3 screenPoint = avCamera.WorldToScreenPoint(pointOnGround);
        if (screenPoint.z < 0 || screenPoint.y <= 0.1f || screenPoint.y >= Screen.height - 0.1f)
        {
            return -1.0f;
        }
        
        float screenY = screenPoint.y;
        if (screenY < 1.0f || screenY > Screen.height - 1.0f)
        {
            return -1.0f;
        }
        
        float imageHeight = cameraCapture.captureHeight;
        return imageHeight - screenY;
    }
    
    IEnumerator UpdateAVStack()
    {
        updateInFlight = true;
        updateSequence += 1;
        int updateId = updateSequence;
        currentUpdateSequence = updateId;
        updateStartRealtime = Time.realtimeSinceStartup;
        try
        {
        // CRITICAL: Double-check play mode state (may have changed during coroutine wait)
        // This prevents sending data during play mode exit transition
        #if UNITY_EDITOR
        if (!EditorApplication.isPlaying)
        {
            yield break; // Don't send data when exiting play mode
        }
        #endif
        
        // Get current vehicle state
        VehicleState currentState = carController.GetVehicleState();
        currentState.requestId = updateId;
        currentState.unitySendRealtime = Time.realtimeSinceStartup;
        currentState.unitySendUtcMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

        if (logFrameCountJumps && lastSentUnityTime > 0f)
        {
            float unityTimeGap = currentState.unityTime - lastSentUnityTime;
            int frameGap = currentState.unityFrameCount - lastSentUnityFrameCount;
            if (unityTimeGap > unityTimeJumpThresholdSeconds)
            {
                Debug.LogWarning(
                    $"[UNITY TIME JUMP] gap={unityTimeGap:F3}s frames={frameGap} " +
                    $"frame={currentState.unityFrameCount} time={currentState.unityTime:F3}"
                );
            }
        }
        lastSentUnityTime = currentState.unityTime;
        lastSentUnityFrameCount = currentState.unityFrameCount;
        
        // CRITICAL: Initialize camera8mScreenY to cached value (or -1.0 if never calculated)
        // This ensures we use cached value when not recalculating, preventing -1.0 from being sent
        // when we have a valid cached value
        currentState.camera8mScreenY = cachedCamera8mScreenY;
        
        // Add ground truth data if available
        if (groundTruthReporter != null)
        {
            float lookaheadDistance = groundTruthReporter.groundTruthLookaheadDistance;
            if (lookaheadDistance <= 0.01f)
            {
                lookaheadDistance = 8.0f;
            }
            currentState.groundTruthLookaheadDistance = lookaheadDistance;
            float lookaheadScreenY = CalculateCameraScreenYAtDistance(lookaheadDistance);
            currentState.cameraLookaheadScreenY = lookaheadScreenY > 0 ? lookaheadScreenY : -1.0f;
            // FIXED: Calculate ground truth at 8m lookahead to match perception evaluation distance
            // This ensures green lines in visualizer align with orange/red lines at the black line (y=350px)
            var (leftLaneLineX, rightLaneLineX) = groundTruthReporter.GetLanePositionsAtLookahead(lookaheadDistance);
            currentState.groundTruthLeftLaneLineX = leftLaneLineX;
            currentState.groundTruthRightLaneLineX = rightLaneLineX;
            // Calculate lane center at lookahead (use lane index from GroundTruthReporter)
            currentState.groundTruthLaneCenterX = groundTruthReporter.GetLaneCenterAtLookahead(
                lookaheadDistance,
                groundTruthReporter.currentLane
            );
            
            // NEW: Add path heading for path-based steering
            float desiredHeading = groundTruthReporter.GetDesiredHeading(5.0f);
            currentState.groundTruthDesiredHeading = desiredHeading;
            currentState.groundTruthPathCurvature = groundTruthReporter.GetPathCurvature();

            // Oracle centerline samples (instrumentation-only).
            currentState.oracleSamplesEnabled = enableOracleSamples;
            currentState.oracleHorizonMeters = Mathf.Clamp(oracleHorizonMeters, 0.5f, 30.0f);
            currentState.oraclePointSpacingMeters = Mathf.Max(0.1f, oraclePointSpacingMeters);
            if (enableOracleSamples)
            {
                float[] oracleSamples = groundTruthReporter.GetOracleTrajectorySamplesVehicle(
                    currentState.oracleHorizonMeters,
                    currentState.oraclePointSpacingMeters
                );
                currentState.oracleTrajectoryXY = oracleSamples ?? new float[0];
                currentState.oraclePointCount = currentState.oracleTrajectoryXY.Length / 2;
            }
            else
            {
                currentState.oracleTrajectoryXY = new float[0];
                currentState.oraclePointCount = 0;
            }

            // Right-lane fiducials for projection diagnostics.
            currentState.rightLaneFiducialsEnabled = enableRightLaneFiducials;
            currentState.rightLaneFiducialsHorizonMeters = Mathf.Clamp(rightLaneFiducialsHorizonMeters, 1.0f, 40.0f);
            currentState.rightLaneFiducialsSpacingMeters = Mathf.Clamp(rightLaneFiducialsSpacingMeters, 1.0f, 10.0f);
            if (enableRightLaneFiducials)
            {
                Vector3[] fiducialWorldPoints;
                float[] fiducialsVehicle = groundTruthReporter.GetRightLaneLineFiducialsVehicle(
                    currentState.rightLaneFiducialsHorizonMeters,
                    currentState.rightLaneFiducialsSpacingMeters,
                    out fiducialWorldPoints
                );
                currentState.rightLaneFiducialsVehicleXY = fiducialsVehicle ?? new float[0];
                int pointCount = currentState.rightLaneFiducialsVehicleXY.Length / 2;
                currentState.rightLaneFiducialsPointCount = pointCount;

                float[] screenSamples = new float[pointCount * 2];
                for (int i = 0; i < screenSamples.Length; i++)
                {
                    screenSamples[i] = -1.0f;
                }
                Camera avCamera = cameraCapture != null ? cameraCapture.targetCamera : null;
                if (avCamera != null && cameraCapture != null && fiducialWorldPoints != null)
                {
                    float imageWidth = Mathf.Max(1.0f, cameraCapture.captureWidth);
                    float imageHeight = Mathf.Max(1.0f, cameraCapture.captureHeight);
                    float screenWidth = Mathf.Max(1.0f, Screen.width);
                    float screenHeight = Mathf.Max(1.0f, Screen.height);
                    int usableCount = Mathf.Min(pointCount, fiducialWorldPoints.Length);
                    for (int i = 0; i < usableCount; i++)
                    {
                        Vector3 screenPoint = avCamera.WorldToScreenPoint(fiducialWorldPoints[i]);
                        if (screenPoint.z <= 0.0f) continue;
                        if (screenPoint.x < 0.0f || screenPoint.x > screenWidth || screenPoint.y < 0.0f || screenPoint.y > screenHeight)
                        {
                            continue;
                        }
                        float imageX = (screenPoint.x / screenWidth) * imageWidth;
                        float imageY = imageHeight - ((screenPoint.y / screenHeight) * imageHeight);
                        screenSamples[2 * i] = imageX;
                        screenSamples[2 * i + 1] = imageY;
                    }
                }
                currentState.rightLaneFiducialsScreenXY = screenSamples;
            }
            else
            {
                currentState.rightLaneFiducialsVehicleXY = new float[0];
                currentState.rightLaneFiducialsScreenXY = new float[0];
                currentState.rightLaneFiducialsPointCount = 0;
            }
            
            // NEW: Add debug information about road center positions (for diagnosing offset issues)
            var (roadCenterAtCar, roadCenterAtLookahead, referenceT, carT) = groundTruthReporter.GetRoadCenterDebugInfo(lookaheadDistance);
            currentState.roadCenterAtCarX = roadCenterAtCar.x;
            currentState.roadCenterAtCarY = roadCenterAtCar.y;
            currentState.roadCenterAtCarZ = roadCenterAtCar.z;
            currentState.roadCenterAtLookaheadX = roadCenterAtLookahead.x;
            currentState.roadCenterAtLookaheadY = roadCenterAtLookahead.y;
            currentState.roadCenterAtLookaheadZ = roadCenterAtLookahead.z;
            currentState.roadCenterReferenceT = referenceT;
            var (roadHeadingDeg, carHeadingDeg, headingDeltaDeg, roadLateralOffset) =
                groundTruthReporter.GetRoadFrameMetrics();
            currentState.roadHeadingDeg = roadHeadingDeg;
            currentState.carHeadingDeg = carHeadingDeg;
            currentState.headingDeltaDeg = headingDeltaDeg;
            currentState.roadFrameLateralOffset = roadLateralOffset;
            float headingRad = roadHeadingDeg * Mathf.Deg2Rad;
            Vector3 roadRight = new Vector3(Mathf.Cos(headingRad), 0f, -Mathf.Sin(headingRad));
            float laneCenterOffset = Vector3.Dot(roadCenterAtLookahead - roadCenterAtCar, roadRight);
            currentState.roadFrameLaneCenterOffset = laneCenterOffset;
            currentState.roadFrameLaneCenterError = roadLateralOffset - laneCenterOffset;
            if (carController != null)
            {
                Vector3 carPos = carController.transform.position;
                Vector3 carForward = carController.transform.forward.normalized;
                Vector3 carRight = Vector3.Cross(Vector3.up, carForward).normalized;
                float vehicleLookaheadOffset = Vector3.Dot(roadCenterAtLookahead - carPos, carRight);
                currentState.vehicleFrameLookaheadOffset = vehicleLookaheadOffset;
            }
            float computedSpeedLimit = groundTruthReporter.GetSpeedLimitAtT(carT);
            float defaultSpeedLimit = groundTruthReporter.GetDefaultSpeedLimit();
            if (computedSpeedLimit > 0f)
            {
                lastNonZeroSpeedLimit = computedSpeedLimit;
            }
            else if (lastNonZeroSpeedLimit > 0f)
            {
                computedSpeedLimit = lastNonZeroSpeedLimit;
            }
            else if (defaultSpeedLimit > 0f)
            {
                // Use top-level track default until per-segment limits are available.
                computedSpeedLimit = defaultSpeedLimit;
                lastNonZeroSpeedLimit = defaultSpeedLimit;
            }
            currentState.speedLimit = computedSpeedLimit;
            float previewDistance = Mathf.Max(0f, speedLimitPreviewDistance);
            float previewDistanceMid = Mathf.Max(0f, speedLimitPreviewDistanceMid);
            float previewDistanceLong = Mathf.Max(0f, speedLimitPreviewDistanceLong);
            float previewSpeedLimit = computedSpeedLimit;
            float previewMinDistance = previewDistance;
            float previewSpeedLimitMid = computedSpeedLimit;
            float previewMinDistanceMid = previewDistanceMid;
            float previewSpeedLimitLong = computedSpeedLimit;
            float previewMinDistanceLong = previewDistanceLong;
            bool previewValid = true;
            if (lastCarT.HasValue)
            {
                float deltaT = Mathf.Abs(carT - lastCarT.Value);
                float wrappedDeltaT = Mathf.Min(deltaT, 1f - deltaT);
                if (wrappedDeltaT > Mathf.Max(0.0f, speedLimitPreviewMaxTDelta))
                {
                    previewValid = false;
                }
            }
            if (previewDistance > 0.01f)
            {
                float pathLength = groundTruthReporter.GetPathLength();
                if (pathLength > 0.01f)
                {
                    previewSpeedLimit = ComputePreviewSpeedLimit(
                        carT,
                        previewDistance,
                        pathLength,
                        computedSpeedLimit,
                        previewValid,
                        out previewMinDistance
                    );
                    previewSpeedLimitMid = ComputePreviewSpeedLimit(
                        carT,
                        previewDistanceMid,
                        pathLength,
                        computedSpeedLimit,
                        previewValid,
                        out previewMinDistanceMid
                    );
                    previewSpeedLimitLong = ComputePreviewSpeedLimit(
                        carT,
                        previewDistanceLong,
                        pathLength,
                        computedSpeedLimit,
                        previewValid,
                        out previewMinDistanceLong
                    );
                    if (previewSpeedLimit <= 0f && lastNonZeroSpeedLimit > 0f)
                    {
                        previewSpeedLimit = lastNonZeroSpeedLimit;
                    }
                    if (previewSpeedLimitMid <= 0f && lastNonZeroSpeedLimit > 0f)
                    {
                        previewSpeedLimitMid = lastNonZeroSpeedLimit;
                    }
                    if (previewSpeedLimitLong <= 0f && lastNonZeroSpeedLimit > 0f)
                    {
                        previewSpeedLimitLong = lastNonZeroSpeedLimit;
                    }
                }
            }
            currentState.speedLimitPreview = previewSpeedLimit;
            currentState.speedLimitPreviewDistance = previewDistance;
            currentState.speedLimitPreviewMinDistance = previewMinDistance;
            currentState.speedLimitPreviewMid = previewSpeedLimitMid;
            currentState.speedLimitPreviewMidDistance = previewDistanceMid;
            currentState.speedLimitPreviewMidMinDistance = previewMinDistanceMid;
            currentState.speedLimitPreviewLong = previewSpeedLimitLong;
            currentState.speedLimitPreviewLongDistance = previewDistanceLong;
            currentState.speedLimitPreviewLongMinDistance = previewMinDistanceLong;
            lastCarT = carT;
            if (showDebugInfo && currentState.speedLimit <= 0f && Time.frameCount % 120 == 0)
            {
                Debug.LogWarning("AVBridge: Speed limit is 0; using fallback (if available). Check track speed limits.");
            }
            if (showDebugInfo && Time.frameCount % 120 == 0)
            {
                Debug.Log(
                    $"AVBridge: SpeedLimit debug t={referenceT:F3} " +
                    $"computed={computedSpeedLimit:F2}m/s default={defaultSpeedLimit:F2}m/s " +
                    $"lastNonZero={lastNonZeroSpeedLimit:F2}m/s"
                );
            }
            if (carController != null)
            {
                carController.speedLimit = currentState.speedLimit;
            }
            
            // Debug: Log ground truth values
            if (showDebugInfo && Time.frameCount % 30 == 0)
            {
                Debug.Log($"GroundTruthReporter: left={leftLaneLineX:F3}m, right={rightLaneLineX:F3}m, center={currentState.groundTruthLaneCenterX:F3}m, desiredHeading={desiredHeading:F1}°");
                Debug.Log($"GroundTruthReporter [DEBUG]: Road center at car=({roadCenterAtCar.x:F3}, {roadCenterAtCar.y:F3}, {roadCenterAtCar.z:F3}), " +
                         $"at lookahead=({roadCenterAtLookahead.x:F3}, {roadCenterAtLookahead.y:F3}, {roadCenterAtLookahead.z:F3}), t={referenceT:F3}");
            }
        }
        
        // NEW: Get camera FOV information (what Unity actually uses)
        // This helps us verify if Unity stores FOV as vertical or horizontal
        if (cameraCapture != null && cameraCapture.targetCamera != null)
        {
            Camera avCamera = cameraCapture.targetCamera;
            // Unity's Camera.fieldOfView ALWAYS returns vertical FOV, regardless of Inspector "Field of View Axis" setting
            float verticalFOV = avCamera.fieldOfView;
            float aspect = (float)avCamera.pixelWidth / (float)avCamera.pixelHeight;
            if (cameraCapture != null && cameraCapture.captureWidth > 0 && cameraCapture.captureHeight > 0)
            {
                // Use capture resolution aspect to match recorded frames (not Game view)
                aspect = (float)cameraCapture.captureWidth / (float)cameraCapture.captureHeight;
            }
            // Calculate horizontal FOV from vertical FOV
            float horizontalFOV = 2.0f * Mathf.Atan(Mathf.Tan(verticalFOV * Mathf.Deg2Rad / 2.0f) * aspect) * Mathf.Rad2Deg;
            
            currentState.cameraFieldOfView = verticalFOV;
            currentState.cameraHorizontalFOV = horizontalFOV;
            
            // NEW: Add camera position and forward direction for debugging alignment
            // CRITICAL: Report camera position as height above GROUND (Y=0), not world Y
            // Car center is at Y = 0.5 (car height = 1m, so bottom is at Y=0, top at Y=1)
            // Camera local Y = 1.2 (relative to car center)
            // Camera world Y = 0.5 + 1.2 = 1.7
            // User wants: camera height above ground = 1.2m (matches Inspector local Y)
            // 
            // SOLUTION: If camera is child of car, use localPosition.y directly (1.2m)
            // This represents the camera's height relative to car center, which the user
            // wants to see as "height above ground" (since car bottom is at ground level)
            Vector3 cameraPos = avCamera.transform.position;
            Vector3 cameraForward = avCamera.transform.forward;
            
            // Calculate camera height above ground
            // If camera is child of car, use localPosition.y (relative to car center)
            // Since car bottom is at ground (Y=0) and car center is at Y=0.5,
            // camera local Y (1.2m) represents height above car center
            // But user wants height above GROUND, which should be: localY + 0.5 = 1.2 + 0.5 = 1.7
            // However, user explicitly said they want 1.2m, so they want local Y, not world Y
            // 
            // INTERPRETATION: User considers "height above ground" to be the camera's local Y
            // (relative to car center), since the car is the reference frame for the camera
            float cameraHeightAboveGround;
            if (avCamera.transform.parent != null)
            {
                // Camera is child of car - use local position Y (relative to car center)
                // User wants this as "height above ground" (1.2m from Inspector)
                cameraHeightAboveGround = avCamera.transform.localPosition.y;
            }
            else
            {
                // Camera is not child of car - calculate from world position
                // FIXED: Handle case where car Y might be locked at 0.0m or 0.5m
                float carY = (carController != null) ? carController.transform.position.y : 0.5f;
                float groundHeight = (carY < 0.25f) ? 0.0f : (carY - 0.5f); // If car Y < 0.25m, assume ground at 0.0m, else use car Y - 0.5m
                cameraHeightAboveGround = cameraPos.y - groundHeight;
            }
            
            // Report camera position (X and Z are world, Y is height above ground)
            currentState.cameraPosX = cameraPos.x;
            currentState.cameraPosY = cameraHeightAboveGround; // Height above ground (local Y if child of car)
            currentState.cameraPosZ = cameraPos.z;
            currentState.cameraForwardX = cameraForward.x;
            currentState.cameraForwardY = cameraForward.y;
            currentState.cameraForwardZ = cameraForward.z;

            // Top-down camera calibration/projection (instrumentation-only).
            if (topDownCamera != null)
            {
                Vector3 tdPos = topDownCamera.transform.position;
                Vector3 tdForward = topDownCamera.transform.forward;
                currentState.topDownCameraPosX = tdPos.x;
                currentState.topDownCameraPosY = tdPos.y; // world Y for projection diagnostics
                currentState.topDownCameraPosZ = tdPos.z;
                currentState.topDownCameraForwardX = tdForward.x;
                currentState.topDownCameraForwardY = tdForward.y;
                currentState.topDownCameraForwardZ = tdForward.z;
                currentState.topDownCameraOrthographicSize = topDownCamera.orthographic
                    ? topDownCamera.orthographicSize
                    : 0.0f;
                currentState.topDownCameraFieldOfView = topDownCamera.orthographic
                    ? 0.0f
                    : topDownCamera.fieldOfView;
            }
            
            // Log once per second to avoid spam
            if (showDebugInfo && Time.frameCount % 30 == 0)
            {
                Debug.Log($"CameraFOV: Unity fieldOfView={verticalFOV:F2}° (vertical), calculated horizontal={horizontalFOV:F2}°");
                Debug.Log($"CameraPos: world=({cameraPos.x:F3}, {cameraPos.y:F3}, {cameraPos.z:F3}), " +
                         $"height_above_ground={currentState.cameraPosY:F3}m, forward=({cameraForward.x:F3}, {cameraForward.y:F3}, {cameraForward.z:F3})");
            }
        }
        
            // NEW: Calculate where 8m appears in camera screen using Unity's actual camera projection
            // This gives us the TRUE y pixel where 8m appears, independent of our simplified camera model
            // CRITICAL: Calculate point 8m ahead on the GROUND (not floating in air)
            // PERFORMANCE: Only recalculate when camera moves significantly or time interval elapsed
            // WorldToScreenPoint() is expensive - don't call it every frame
            bool shouldRecalcCamera8m = false;
            if (cameraCapture != null && cameraCapture.targetCamera != null && carController != null)
            {
                Camera avCamera = cameraCapture.targetCamera;
                Vector3 currentCameraPos = avCamera.transform.position;
                Quaternion currentCameraRot = avCamera.transform.rotation;
                
                // Recalculate if:
                // 1. Time interval elapsed (0.5s)
                // 2. Camera moved significantly (>0.1m)
                // 3. Camera rotated significantly (>5 degrees)
                // 4. Never calculated before (cachedCamera8mScreenY == -1.0f)
                float timeSinceLastRecalc = Time.time - lastCamera8mScreenYRecalcTime;
                float cameraPosChange = Vector3.Distance(currentCameraPos, lastCameraPosition);
                float cameraRotChange = Quaternion.Angle(currentCameraRot, lastCameraRotation);
                
                if (cachedCamera8mScreenY < 0 || 
                    timeSinceLastRecalc >= camera8mScreenYRecalcInterval ||
                    cameraPosChange > 0.1f ||
                    cameraRotChange > 5.0f)
                {
                    shouldRecalcCamera8m = true;
                    lastCameraPosition = currentCameraPos;
                    lastCameraRotation = currentCameraRot;
                    lastCamera8mScreenYRecalcTime = Time.time;
                }
            }
            
            if (shouldRecalcCamera8m && cameraCapture != null && cameraCapture.targetCamera != null && carController != null)
            {
                Camera avCamera = cameraCapture.targetCamera;
                
                // Calculate point 8m ahead along camera's forward direction
                Vector3 cameraPos = avCamera.transform.position;
                Vector3 cameraForward = avCamera.transform.forward;
                Vector3 point8mDirection = cameraPos + cameraForward * 8.0f;
            
            // CRITICAL FIX: Project point DOWN to ground plane (y=0 or road height)
            // The reference point is calculated on the road, so we need the ground point, not a floating point
            // Use car's position as reference for ground height (car is on the road)
            // FIXED: Handle case where car Y might be locked at 0.0m or 0.5m
            // If car Y = 0.0m, then ground is at 0.0m (car bottom is at ground)
            // If car Y = 0.5m, then ground is at 0.0m (car center is 0.5m above ground, bottom at 0.0m)
            float carY = carController.transform.position.y;
            float groundHeight = (carY < 0.25f) ? 0.0f : (carY - 0.5f); // If car Y < 0.25m, assume ground at 0.0m, else use car Y - 0.5m
            
            // Calculate intersection of camera forward ray with ground plane
            // Ray: cameraPos + t * cameraForward
            // Ground plane: y = groundHeight
            // Solve: cameraPos.y + t * cameraForward.y = groundHeight
            float t = (groundHeight - cameraPos.y) / cameraForward.y;
            
            // CRITICAL: Check if camera is looking down (cameraForward.y < 0)
            // If camera is looking down, t will be positive and point will be in front
            // If camera is looking up, t will be negative and point will be behind
            Vector3 point8mOnGround = Vector3.zero; // Initialize to avoid compiler error
            bool shouldCalculate = false;
            
            if (cameraForward.y >= 0)
            {
                // Camera is looking up or horizontal - ground intersection is behind or invalid
                // Fall back to projecting 8m point straight down
                if (showDebugInfo && Time.frameCount % 60 == 0)
                {
                    Debug.LogWarning($"CameraCalibration: Camera is not looking down! cameraForward.y={cameraForward.y:F3}, using fallback projection.");
                }
                point8mOnGround = point8mDirection;
                point8mOnGround.y = groundHeight;
                shouldCalculate = true; // Still try to calculate screen position
            }
            else if (t <= 0)
            {
                // Ground intersection is behind camera - invalid
                if (showDebugInfo && Time.frameCount % 60 == 0)
                {
                    Debug.LogWarning($"CameraCalibration: Ground intersection is BEHIND camera! t={t:F3}, cameraForward.y={cameraForward.y:F3}");
                }
                // Keep -1.0 (already set at start of UpdateAVStack)
                // Skip rest of camera calculation, but continue with rest of UpdateAVStack
                shouldCalculate = false; // Don't calculate - point is invalid
            }
            else
            {
                // Valid ground intersection
                point8mOnGround = cameraPos + cameraForward * t;
                
                // Clamp to 8m distance (if ground intersection is beyond 8m, use 8m point projected down)
                float distanceToGround = Vector3.Distance(cameraPos, point8mOnGround);
                if (distanceToGround > 8.0f)
                {
                    // Ground is further than 8m - project 8m point straight down to ground
                    point8mOnGround = point8mDirection;
                    point8mOnGround.y = groundHeight;
                }
                // else: use ground intersection point (distanceToGround < 8.0f)
                shouldCalculate = true; // Calculate screen position
            }
            
            // Use Unity's actual camera projection to get screen coordinates of GROUND point
            // Only calculate if we have a valid point
            if (shouldCalculate)
            {
                Vector3 screenPoint = avCamera.WorldToScreenPoint(point8mOnGround);
                
                // CRITICAL: Check if point is behind camera or outside view
                // WorldToScreenPoint returns screenPoint.z < 0 if point is behind camera
                // screenPoint.y = 0 or negative if point is outside view or behind
                if (screenPoint.z < 0)
                {
                    // Point is behind camera - invalid!
                    // Keep -1.0 (already set at start of UpdateAVStack)
                    if (showDebugInfo && Time.frameCount % 60 == 0)
                    {
                        Debug.LogWarning($"CameraCalibration: 8m point is BEHIND camera! point8mOnGround={point8mOnGround}, screenPoint.z={screenPoint.z:F2}");
                    }
                }
                else if (screenPoint.y <= 0.1f || screenPoint.y >= Screen.height - 0.1f)
                {
                    // Point is outside camera view (below or above screen)
                    // Use 0.1f tolerance to catch near-zero values that would give 480px
                    // Keep -1.0 (already set at start of UpdateAVStack)
                    if (showDebugInfo && Time.frameCount % 60 == 0)
                    {
                        Debug.LogWarning($"CameraCalibration: 8m point is OUTSIDE camera view! screenY={screenPoint.y:F3}px (Screen.height={Screen.height})");
                    }
                }
                else
                {
                    // Valid point - convert to image coordinates
                    // screenPoint.y is in screen coordinates (0 = bottom, Screen.height = top)
                    // Convert to image coordinates (0 = top, imageHeight = bottom) to match our image format
                    float imageHeight = cameraCapture.captureHeight;
                    float screenY = screenPoint.y;
                    
                    // CRITICAL: Additional safety check - screenY should be well within bounds
                    // If screenY is too close to 0 or Screen.height, the result will be wrong
                    if (screenY < 1.0f || screenY > Screen.height - 1.0f)
                    {
                        // Too close to edge - invalid
                        // Keep -1.0 (already set at start of UpdateAVStack)
                        if (showDebugInfo && Time.frameCount % 60 == 0)
                        {
                            Debug.LogWarning($"CameraCalibration: screenY too close to edge! screenY={screenY:F3}px (Screen.height={Screen.height})");
                        }
                    }
                    else
                    {
                        // Unity's screen coordinates: 0 = bottom, Screen.height = top
                        // Our image coordinates: 0 = top, imageHeight = bottom
                        // Conversion: imageY = imageHeight - screenY
                        float imageY = imageHeight - screenY;
                        cachedCamera8mScreenY = imageY; // Cache the result
                        currentState.camera8mScreenY = imageY;
                        
                        // Debug: Log calibration data periodically
                        if (showDebugInfo && Time.frameCount % 60 == 0)  // Every ~2 seconds
                        {
                            Debug.Log($"CameraCalibration: 8m point on ground at world={point8mOnGround}, screenY={screenY:F1}px, imageY={imageY:F1}px (imageHeight={imageHeight})");
                            Debug.Log($"CameraCalibration: currentState.camera8mScreenY = {currentState.camera8mScreenY:F1}px (will be sent to Python)");
                        }
                    }
                }
            }
            else
            {
                // Use cached value instead of recalculating
                currentState.camera8mScreenY = cachedCamera8mScreenY;
            }
        }
        else
        {
            // Camera not available - use cached value or -1.0
            currentState.camera8mScreenY = cachedCamera8mScreenY;
            if (showDebugInfo && Time.frameCount % 60 == 0)  // Every ~2 seconds
            {
                Debug.LogWarning($"CameraCalibration: Camera not available! cameraCapture={cameraCapture}, targetCamera={(cameraCapture != null ? cameraCapture.targetCamera : null)}");
            }
        }
        
        // Debug: Warn if GroundTruthReporter not found (moved outside camera check)
        if (groundTruthReporter == null && showDebugInfo && Time.frameCount % 90 == 0)  // Log every 3 seconds
        {
            Debug.LogWarning("AVBridge: GroundTruthReporter not found! Ground truth data will be 0.0");
        }
        
        // Send state to Python server (fire-and-forget; don't stall control loop).
        if (!stateSendInFlight)
        {
            StartCoroutine(SendVehicleState(currentState, updateId));
        }
        else if (logBridgeTimings && showDebugInfo)
        {
            float stateInFlightDuration = Time.realtimeSinceStartup - stateSendStartRealtime;
            if (stateInFlightDuration > bridgeTimingWarnThreshold)
            {
                Debug.LogWarning(
                    $"AVBridge: SendVehicleState still in flight (id={updateId}, " +
                    $"elapsed={stateInFlightDuration:F3}s, frame={Time.frameCount}, time={Time.time:F3}s)"
                );
            }
        }
        
        // Request control commands (only if AV control is enabled)
        if (enableAVControl)
        {
            if (!controlRequestInFlight)
            {
                StartCoroutine(RequestControlCommands(updateId));
            }
            else if (logBridgeTimings && showDebugInfo)
            {
                float controlInFlightDuration = Time.realtimeSinceStartup - controlRequestStartRealtime;
                if (controlInFlightDuration > bridgeTimingWarnThreshold)
                {
                    Debug.LogWarning(
                        $"AVBridge: RequestControlCommands still in flight (id={updateId}, " +
                        $"elapsed={controlInFlightDuration:F3}s, frame={Time.frameCount}, time={Time.time:F3}s)"
                    );
                }
            }
        }
        }
        finally
        {
            float totalDuration = Time.realtimeSinceStartup - updateStartRealtime;
            if (logBridgeTimings && showDebugInfo && totalDuration > bridgeTimingWarnThreshold)
            {
                Debug.LogWarning(
                    $"AVBridge: UpdateAVStack slow (id={currentUpdateSequence}, " +
                    $"duration={totalDuration:F3}s, frame={Time.frameCount}, time={Time.time:F3}s)"
                );
            }
            updateInFlight = false;
        }
        yield return null;
    }
    
    IEnumerator SendVehicleState(VehicleState state, int updateId)
    {
        if (stateSendInFlight)
        {
            yield break;
        }

        string url = $"{apiUrl}{stateEndpoint}";
        
        // Create JSON payload
        string json = JsonUtility.ToJson(state);
        byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
        
        stateSendInFlight = true;
        stateSendStartRealtime = Time.realtimeSinceStartup;

        using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.timeout = 1;

            float startRealtime = Time.realtimeSinceStartup;
            try
            {
                yield return request.SendWebRequest();
            }
            finally
            {
                stateSendInFlight = false;
            }

            float duration = Time.realtimeSinceStartup - startRealtime;

            if (request.result != UnityWebRequest.Result.Success)
            {
                if (showDebugInfo)
                {
                    Debug.LogWarning(
                        $"AVBridge: Failed to send state (id={updateId}) - {request.error}"
                    );
                }
            }
            else if (logBridgeTimings && showDebugInfo && duration > bridgeTimingWarnThreshold)
            {
                Debug.LogWarning(
                    $"AVBridge: SendVehicleState slow (id={updateId}, " +
                    $"duration={duration:F3}s, frame={Time.frameCount}, time={Time.time:F3}s)"
                );
            }

            lastVehicleState = state;
        }
    }
    
    IEnumerator CheckPlayRequest()
    {
        // Check if Python script is requesting Unity to start playing (use GET, not POST)
        // Only works in Unity Editor (not in builds)
        #if UNITY_EDITOR
        if (EditorApplication.isPlaying)
        {
            // Already playing, no need to check
            yield break;
        }
        
        string url = $"{apiUrl}{playEndpoint}";
        
        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            request.timeout = 1; // Short timeout for quick check
            yield return request.SendWebRequest();
            
            if (request.result == UnityWebRequest.Result.Success)
            {
                // Parse response to check if play was requested
                try
                {
                    string jsonResponse = request.downloadHandler.text;
                    // Check for play status in response (more flexible matching)
                    if (jsonResponse.Contains("\"status\":\"play\"") || jsonResponse.Contains("play_requested"))
                    {
                        // Python script is requesting play mode - enter play mode
                        Debug.Log($"[COMMAND RECEIVED] AVBridge: ▶️ PLAY command received (CheckPlayRequest) - Python script requested play mode, entering play mode...");
                        Debug.Log($"[COMMAND RECEIVED] AVBridge: Play request response: {jsonResponse}");
                        EditorApplication.isPlaying = true;
                    }
                    else
                    {
                        Debug.Log($"[COMMAND RECEIVED] AVBridge: Play check response (no play request): {jsonResponse}");
                    }
                    // If status is "no_request", do nothing (no play request)
                }
                catch (Exception e)
                {
                    if (showDebugInfo)
                    {
                        Debug.LogWarning($"AVBridge: Error parsing play request response: {e}");
                    }
                }
            }
            else if (request.result == UnityWebRequest.Result.ConnectionError || 
                     request.result == UnityWebRequest.Result.ProtocolError)
            {
                // Connection error - bridge server might not be running yet
                // This is expected if Python script hasn't started yet, so don't log
            }
        }
        #endif
        yield break;
    }
    
    IEnumerator CheckShutdownSignal()
    {
        // Check if AV stack is shutting down (use GET, not POST)
        string url = $"{apiUrl}{shutdownEndpoint}";
        
        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            request.timeout = 1; // Short timeout for quick check
            yield return request.SendWebRequest();
            
            if (request.result == UnityWebRequest.Result.Success)
            {
                // Parse response to check if shutdown was requested
                try
                {
                    string jsonResponse = request.downloadHandler.text;
                    // Simple check for shutdown status in response
                    if (jsonResponse.Contains("\"status\":\"shutdown\""))
                    {
                        // AV stack is shutting down - exit play mode gracefully
                        Debug.Log($"[COMMAND RECEIVED] AVBridge: ⚠️ SHUTDOWN command received - AV stack is shutting down, exiting play mode...");
                        Debug.Log($"[COMMAND RECEIVED] AVBridge: Shutdown response: {jsonResponse}");
                        #if UNITY_EDITOR
                        EditorApplication.isPlaying = false;
                        #endif
                    }
                    else
                    {
                        Debug.Log($"[COMMAND RECEIVED] AVBridge: Shutdown check response (not shutdown): {jsonResponse}");
                    }
                    // If status is "running", do nothing (AV stack is active)
                }
                catch (Exception e)
                {
                    if (showDebugInfo)
                    {
                        Debug.LogWarning($"AVBridge: Failed to parse shutdown response - {e.Message}");
                    }
                }
            }
            else if (request.result == UnityWebRequest.Result.ConnectionError)
            {
                // Bridge server is down (AV stack stopped) - exit play mode
                // This handles the case where the script stops the bridge server
                Debug.Log($"[COMMAND RECEIVED] AVBridge: ⚠️ BRIDGE DISCONNECTED - Connection error, exiting play mode...");
                #if UNITY_EDITOR
                EditorApplication.isPlaying = false;
                #endif
            }
            // Other errors (timeout, etc.) - don't exit play mode
            // (might be temporary network issues)
        }
    }
    
    IEnumerator RequestControlCommands(int updateId)
    {
        if (controlRequestInFlight)
        {
            yield break;
        }

        long sendUtcMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        string url = $"{apiUrl}{controlEndpoint}?request_id={updateId}&unity_send_utc_ms={sendUtcMs}";
        
        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            request.timeout = 1;
            controlRequestInFlight = true;
            controlRequestStartRealtime = Time.realtimeSinceStartup;
            float startRealtime = Time.realtimeSinceStartup;
            try
            {
                yield return request.SendWebRequest();
            }
            finally
            {
                controlRequestInFlight = false;
            }
            float duration = Time.realtimeSinceStartup - startRealtime;
            
            if (request.result == UnityWebRequest.Result.Success)
            {
                try
                {
                    string jsonResponse = request.downloadHandler.text;
                    
                    if (showDebugInfo && Time.frameCount % 30 == 0)
                    {
                        Debug.Log($"[COMMAND RECEIVED] AVBridge: Control command received - {jsonResponse}");
                    }
                    
                    ControlCommand command = JsonUtility.FromJson<ControlCommand>(jsonResponse);
                    
                    if (showDebugInfo && Time.frameCount % 30 == 0)
                    {
                        Debug.Log($"[COMMAND RECEIVED] AVBridge: Parsed command - steering={command.steering:F3}, " +
                                 $"throttle={command.throttle:F3}, brake={command.brake:F3}, " +
                                 $"ground_truth_mode={command.ground_truth_mode}, " +
                                 $"ground_truth_speed={command.ground_truth_speed}, " +
                                 $"randomize_start={command.randomize_start}, " +
                                 $"randomize_request_id={command.randomize_request_id}");
                    }
                    
                    // Check ground truth mode - Unity JsonUtility might not deserialize optional fields
                    // So we need to manually parse if needed
                    bool gtMode = command.ground_truth_mode;
                    float gtSpeed = command.ground_truth_speed;
                    bool randomizeStart = command.randomize_start;
                    int randomizeRequestId = command.randomize_request_id;
                    int randomizeSeed = command.randomize_seed;
                    
                    // Fallback: Try to parse from JSON string if JsonUtility didn't work
                    if (!gtMode && jsonResponse.Contains("\"ground_truth_mode\":true"))
                    {
                        Debug.LogWarning("AVBridge: JsonUtility didn't parse ground_truth_mode, parsing manually from JSON");
                        // Simple manual parsing
                        if (jsonResponse.Contains("\"ground_truth_mode\":true"))
                        {
                            gtMode = true;
                        }
                        // Try to extract speed
                        int speedIdx = jsonResponse.IndexOf("\"ground_truth_speed\":");
                        if (speedIdx >= 0)
                        {
                            int startIdx = speedIdx + 21; // Length of "ground_truth_speed":
                            int endIdx = jsonResponse.IndexOf(",", startIdx);
                            if (endIdx < 0) endIdx = jsonResponse.IndexOf("}", startIdx);
                            if (endIdx > startIdx)
                            {
                                string speedStr = jsonResponse.Substring(startIdx, endIdx - startIdx).Trim();
                                if (float.TryParse(speedStr, out float parsedSpeed))
                                {
                                    gtSpeed = parsedSpeed;
                                }
                            }
                        }
                    }

                    // Fallback: Parse random start fields if JsonUtility didn't populate them
                    if (!randomizeStart && jsonResponse.Contains("\"randomize_start\":true"))
                    {
                        randomizeStart = true;
                    }
                    int randomizeIdIdx = jsonResponse.IndexOf("\"randomize_request_id\":", StringComparison.Ordinal);
                    if (randomizeIdIdx >= 0)
                    {
                        int startIdx = randomizeIdIdx + 23; // Length of "randomize_request_id":
                        int endIdx = jsonResponse.IndexOf(",", startIdx);
                        if (endIdx < 0) endIdx = jsonResponse.IndexOf("}", startIdx);
                        if (endIdx > startIdx)
                        {
                            string idStr = jsonResponse.Substring(startIdx, endIdx - startIdx).Trim();
                            if (int.TryParse(idStr, out int parsedId))
                            {
                                randomizeRequestId = parsedId;
                            }
                        }
                    }
                    int randomizeSeedIdx = jsonResponse.IndexOf("\"randomize_seed\":", StringComparison.Ordinal);
                    if (randomizeSeedIdx >= 0)
                    {
                        int startIdx = randomizeSeedIdx + 17; // Length of "randomize_seed":
                        int endIdx = jsonResponse.IndexOf(",", startIdx);
                        if (endIdx < 0) endIdx = jsonResponse.IndexOf("}", startIdx);
                        if (endIdx > startIdx)
                        {
                            string seedStr = jsonResponse.Substring(startIdx, endIdx - startIdx).Trim();
                            if (int.TryParse(seedStr, out int parsedSeed))
                            {
                                randomizeSeed = parsedSeed;
                            }
                        }
                    }
                    
                    if (gtMode)
                    {
                        Debug.Log($"[COMMAND RECEIVED] AVBridge: ✅ ENABLING ground truth mode, speed={gtSpeed} m/s");
                        carController.SetGroundTruthMode(true, gtSpeed);
                    }
                    else if (carController.groundTruthMode)
                    {
                        // Disable ground truth mode if it was enabled but command doesn't request it
                        Debug.Log($"[COMMAND RECEIVED] AVBridge: ⚠️ DISABLING ground truth mode (command.ground_truth_mode=false)");
                        carController.SetGroundTruthMode(false, 0f);
                    }

                    if (randomizeStart && !randomStartHandled)
                    {
                        if (groundTruthReporter == null)
                        {
                            groundTruthReporter = FindObjectOfType<GroundTruthReporter>();
                        }
                        if (groundTruthReporter != null)
                        {
                            Debug.Log("[COMMAND RECEIVED] AVBridge: 🎲 Randomizing start position on oval");
                            groundTruthReporter.SetCarToRandomStart(randomizeSeed);
                            lastRandomizeRequestId = randomizeRequestId;
                            randomStartHandled = true;
                        }
                        else
                        {
                            Debug.LogWarning("AVBridge: GroundTruthReporter not found; cannot randomize start");
                        }
                    }
                    else if (randomizeStart && randomStartHandled)
                    {
                        Debug.Log("[COMMAND RECEIVED] AVBridge: Random start already handled; skipping.");
                    }
                    
                    // CRITICAL: Auto-enable AV control when first command is received
                    // This ensures the car responds to Python commands without manual setup
                    if (!enableAVControl || !carController.avControlEnabled)
                    {
                        Debug.Log($"[COMMAND RECEIVED] AVBridge: ⚠️ Auto-enabling AV control (was disabled)");
                        enableAVControl = true;
                        carController.avControlEnabled = true;
                        carController.avControlPriority = true;
                    }
                    
                    // Apply control command to vehicle
                    if (showDebugInfo && Time.frameCount % 30 == 0)
                    {
                        Debug.Log($"[COMMAND RECEIVED] AVBridge: Applying controls - steering={command.steering:F3}, throttle={command.throttle:F3}, brake={command.brake:F3}");
                    }
                    carController.SetAVControls(
                        command.steering,
                        command.throttle,
                        command.brake
                    );
                    
                    lastControlCommand = command;
                }
                catch (Exception e)
                {
                    if (showDebugInfo)
                    {
                        Debug.LogWarning($"AVBridge: Failed to parse control command - {e.Message}");
                    }
                }
            }
            else
            {
                if (showDebugInfo)
                {
                    Debug.LogWarning(
                        $"AVBridge: Failed to get control commands (id={updateId}) - {request.error}"
                    );
                }
            }
            
            if (logBridgeTimings && showDebugInfo && duration > bridgeTimingWarnThreshold)
            {
                Debug.LogWarning(
                    $"AVBridge: RequestControlCommands slow (id={updateId}, " +
                    $"duration={duration:F3}s, frame={Time.frameCount}, time={Time.time:F3}s)"
                );
            }
        }
    }
    
    IEnumerator SendUnityFeedback()
    {
        // Send Unity feedback periodically to Python for recording
        // This eliminates the need to check Unity console manually
        while (true)
        {
            yield return new WaitForSeconds(feedbackSendInterval);
            
            if (carController == null)
            {
                continue;
            }
            
            // Collect feedback data
            UnityFeedback feedback = new UnityFeedback();
            feedback.timestamp = Time.time;
            feedback.unity_frame_count = Time.frameCount;
            feedback.unity_time = Time.time;
            
            // Control status
            feedback.ground_truth_mode_active = carController.groundTruthMode;
            feedback.control_command_received = (lastControlCommand != null);
            feedback.actual_steering_applied = carController.steerInput * carController.maxSteerAngle;
            feedback.actual_throttle_applied = carController.throttleInput;
            feedback.actual_brake_applied = carController.brakeInput;
            
            // Ground truth data status
            if (groundTruthReporter != null)
            {
                feedback.ground_truth_reporter_enabled = groundTruthReporter.enabled;
                feedback.ground_truth_data_available = groundTruthReporter.enabled;
                
                // Check if path curvature is being calculated
                float curvature = groundTruthReporter.GetPathCurvature();
                feedback.path_curvature_calculated = (Mathf.Abs(curvature) > 1e-6f);
            }
            else
            {
                feedback.ground_truth_reporter_enabled = false;
                feedback.ground_truth_data_available = false;
                feedback.path_curvature_calculated = false;
            }
            
            // Car controller mode
            feedback.car_controller_mode = carController.groundTruthMode ? "ground_truth" : "physics";
            feedback.av_control_enabled = carController.avControlEnabled;
            
            // Send feedback to Python
            string url = $"{apiUrl}{feedbackEndpoint}";
            string json = JsonUtility.ToJson(feedback);
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
            
            using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
            {
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                
                yield return request.SendWebRequest();
                
                if (request.result != UnityWebRequest.Result.Success)
                {
                    if (showDebugInfo && Time.frameCount % 60 == 0)  // Log every 2 seconds
                    {
                        Debug.LogWarning($"AVBridge: Failed to send Unity feedback - {request.error}");
                    }
                }
            }
        }
    }
    
    void OnGUI()
    {
        showDebugInfo = true;
        //if (showDebugInfo)
        {
            if (Time.frameCount % 120 == 0 && Event.current.type == EventType.Repaint)
            {
                Debug.Log("[AVBridge] OnGUI tick");
            }
            if (Event.current.type != EventType.Repaint)
            {
                return;
            }
            if (Time.frameCount == lastOverlayFrame)
            {
                return;
            }
            lastOverlayFrame = Time.frameCount;
            GUI.depth = -1000;
            GUI.color = Color.white;
            GUIStyle labelStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 14,
                normal = { textColor = Color.white }
            };
            int yOffset = 10;
            int xOffset = 10;
            int lineHeight = 20;
            int boxWidth = 320;
            int boxHeight = 6 * lineHeight + 10;
            GUI.Box(new Rect(xOffset - 6, yOffset - 6, boxWidth, boxHeight), GUIContent.none);
            GUI.Label(new Rect(xOffset, yOffset, 300, lineHeight),
                $"AV Control: {(enableAVControl ? "ON" : "OFF")}", labelStyle);
            yOffset += lineHeight;
            
            if (lastControlCommand != null)
            {
                GUI.Label(new Rect(xOffset, yOffset, 300, lineHeight),
                    $"Steering: {lastControlCommand.steering:F3}", labelStyle);
                yOffset += lineHeight;
                GUI.Label(new Rect(xOffset, yOffset, 300, lineHeight),
                    $"Throttle: {lastControlCommand.throttle:F3}", labelStyle);
                yOffset += lineHeight;
                GUI.Label(new Rect(xOffset, yOffset, 300, lineHeight),
                    $"Brake Cmd: {lastControlCommand.brake:F3}", labelStyle);
                yOffset += lineHeight;
                if (carController != null)
                {
                    GUI.Label(new Rect(xOffset, yOffset, 300, lineHeight),
                        $"Brake Applied: {carController.brakeInput:F3}", labelStyle);
                    yOffset += lineHeight;
                }
            }
            if (lastVehicleState != null)
            {
                float speedMps = lastVehicleState.speed;
                float speedMph = speedMps * 2.236936f;
                float limitMps = lastVehicleState.speedLimit;
                string limitText = limitMps > 0.0f ? $"{(limitMps * 2.236936f):F1} mph" : "-";
                GUI.Label(new Rect(xOffset, yOffset, 300, lineHeight),
                    $"Speed: {speedMph:F1} mph ({speedMps:F1} m/s)", labelStyle);
                yOffset += lineHeight;
                GUI.Label(new Rect(xOffset, yOffset, 300, lineHeight),
                    $"Speed Limit: {limitText}", labelStyle);
            }
        }
    }
}

/// <summary>
/// Control command data structure
/// </summary>
[System.Serializable]
public class ControlCommand
{
    public float steering;  // -1.0 to 1.0
    public float throttle;  // -1.0 to 1.0 (negative = reverse)
    public float brake;     // 0.0 to 1.0
    public bool emergency_stop = false;
    // Optional: Ground truth mode settings
    public bool ground_truth_mode = false;  // Enable direct velocity control
    public float ground_truth_speed = 5.0f;  // Speed for ground truth mode (m/s)
    // Optional: Randomize start on oval
    public bool randomize_start = false;
    public int randomize_request_id = 0;
    public int randomize_seed = -1;
}

[System.Serializable]
public class UnityFeedback
{
    public float timestamp;
    // Control status
    public bool ground_truth_mode_active;
    public bool control_command_received;
    public float actual_steering_applied;
    public float actual_throttle_applied;
    public float actual_brake_applied;
    // Ground truth data status
    public bool ground_truth_data_available;
    public bool ground_truth_reporter_enabled;
    public bool path_curvature_calculated;
    // Errors/warnings (not implemented yet - could collect from Debug.Log)
    public string unity_errors;
    public string unity_warnings;
    // Internal state
    public string car_controller_mode;
    public bool av_control_enabled;
    // Frame info
    public int unity_frame_count;
    public float unity_time;
}

