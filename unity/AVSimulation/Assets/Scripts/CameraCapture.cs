using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using System;
using System.Reflection;
using System.Collections.Generic;
#if UNITY_2018_2_OR_NEWER
using UnityEngine.Rendering;
#endif
#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Captures camera frames and sends them to the Python AV stack server
/// </summary>
public class SyncPacketSourceBundleContext
{
    public string packetKey = "";
    public int ownerUpdateId = 0;
    public int ownerUnityFrameCount = 0;
    public float ownerUnityTime = 0.0f;
    public float createdRealtime = 0.0f;
    public float deadlineRealtime = 0.0f;
    public int inflightCount = 0;
    public bool cameraRequested = false;
    public bool cameraRequestAttempted = false;
    public bool cameraRequestAccepted = false;
    public string cameraRequestRejectedReason = "";
    public string cameraRequestSkippedReason = "";
    public string cameraRequestDispositionCode = "";
    public float cameraRequestAttemptRealtime = 0.0f;
    public float cameraRequestAcceptRealtime = 0.0f;
    public int cameraRequestQueueDepth = 0;
    public bool cameraCaptured = false;
    public bool cameraEnqueued = false;
    public bool cameraSent = false;
    public bool bundleAbortedBeforeVehicleSend = false;
    public string bundleAbortReason = "";
    public bool vehicleSendBlockedByCameraRequest = false;
    public bool vehicleStateBuilt = false;
    public bool vehicleStateEnqueued = false;
    public bool vehicleStateSent = false;
    public bool supersededBeforeSend = false;
    public string closeReason = "open";
    public bool closed = false;

    public float AgeMs(float nowRealtime)
    {
        return Mathf.Max(0.0f, (nowRealtime - createdRealtime) * 1000.0f);
    }

    public float DeadlineMs()
    {
        return Mathf.Max(0.0f, (deadlineRealtime - createdRealtime) * 1000.0f);
    }

    public float CameraRequestAttemptAgeMs(float nowRealtime)
    {
        if (cameraRequestAttemptRealtime <= 0.0f)
        {
            return 0.0f;
        }
        return Mathf.Max(0.0f, (nowRealtime - cameraRequestAttemptRealtime) * 1000.0f);
    }

    public float CameraRequestAcceptAgeMs(float nowRealtime)
    {
        if (cameraRequestAcceptRealtime <= 0.0f)
        {
            return 0.0f;
        }
        return Mathf.Max(0.0f, (nowRealtime - cameraRequestAcceptRealtime) * 1000.0f);
    }

    public void Close(string reason)
    {
        string normalized = string.IsNullOrEmpty(reason) ? "unknown" : reason;
        if (!closed)
        {
            closed = true;
            closeReason = normalized;
            return;
        }
        if (string.IsNullOrEmpty(closeReason) || closeReason == "open")
        {
            closeReason = normalized;
        }
    }

    public void TryMarkComplete()
    {
        if (cameraSent && vehicleStateSent)
        {
            Close("complete");
        }
    }
}

public struct SyncPacketContextRegistrationResult
{
    public bool attempted;
    public bool accepted;
    public string rejectedReason;
    public string dispositionCode;
    public int queueDepthBefore;
    public int queueDepthAfter;
}

public class CameraCapture : MonoBehaviour
{
    private static bool gtSyncTimeConfigured = false;

    [Header("Camera Settings")]
    public Camera targetCamera;
    public int captureWidth = 640;
    public int captureHeight = 480;
    public int targetFPS = 15;
    
    [Header("API Settings")]
    public string apiUrl = "http://localhost:8000";
    public string cameraEndpoint = "/api/camera";
    public string cameraId = "front_center";
    
    [Header("Debug")]
    public bool showDebugInfo = true;
    public bool saveLocalImages = false;
    public bool useAsyncGPUReadback = true;
    [Range(10, 100)]
    public int jpegQuality = 85;
    public bool logFrameMarkers = true;
    public float captureGapWarnSeconds = 0.2f;
    public float readbackWarnSeconds = 0.2f;

    [Header("Startup")]
    [Tooltip("Skip first N capture cycles after Start() to let GPU finish shader compilation.")]
    public int startupGraceFrames = 10;

    [Header("Runtime Consistency")]
    public bool enforceScreenResolution = true;
    public int screenWidth = 849;
    public int screenHeight = 439;
    public FullScreenMode screenMode = FullScreenMode.Windowed;
    public bool enforceQualityLevel = true;
    public int qualityLevelIndex = 0;
    public bool enforceCameraFov = true;
    public float cameraFieldOfView = 50f;
    public bool enforceCameraAspect = true;

    [Header("GT Sync Capture")]
    [Tooltip("When enabled, capture is driven from FixedUpdate and timestamps are deterministic.")]
    public bool gtSyncCapture = false;
    [Tooltip("Disable AsyncGPUReadback in GT sync mode. Default: false (async enabled " +
             "in GT sync for better performance with double buffering).")]
    public bool disableAsyncReadbackInGtSync = false;
    [Tooltip("Force Unity fixed timestep while in GT sync capture mode.")]
    public bool gtSyncForceFixedDelta = true;
    [Tooltip("Fixed timestep seconds for GT sync mode (default 1/30s).")]
    public float gtSyncFixedDeltaSeconds = 0.033333333f;
    [Tooltip("Queue camera uploads and send from a single worker coroutine (GT experiments).")]
    public bool gtCameraSendAsync = false;
    [Tooltip("Reduce top-down capture rate in GT sync mode to preserve front-camera cadence.")]
    public bool gtReduceTopDownRate = true;
    [Tooltip("Max queued camera frames before dropping oldest.")]
    public int maxUploadQueueSize = 4;
    [Tooltip("Maximum Unity frame delta allowed when pairing a camera capture to an AVBridge packet context.")]
    public int syncPacketContextMaxFrameDelta = 2;
    [Tooltip("Maximum Unity time delta in seconds allowed when pairing a camera capture to an AVBridge packet context.")]
    public float syncPacketContextMaxTimeDeltaSeconds = 0.05f;
    [Tooltip("Allow accepted source bundles to trigger a capture before the periodic interval.")]
    public bool requestDrivenSyncCapture = true;
    [Tooltip("Minimum seconds between request-driven captures while accepted source bundles are waiting.")]
    public float requestDrivenCaptureMinIntervalSeconds = 0.01f;
    
    // Double-buffered render textures: render to buffer[activeBufferIndex] while
    // async readback completes on the other buffer. This overlaps GPU render and
    // CPU readback, reducing steady-state jitter by ~50%.
    private RenderTexture[] renderTextures = new RenderTexture[2];
    private int activeBufferIndex = 0;
    private bool[] bufferReadbackInFlight = new bool[2];
    private int[] bufferPendingFrameId = new int[2];
    private float[] bufferPendingTimestamp = new float[2];
    private float[] bufferPendingReadbackStart = new float[2];
    private int[] bufferPendingUnityFrameCount = new int[2];
    private float[] bufferPendingUnityTime = new float[2];
    private float[] bufferPendingRealtime = new float[2];
    private float[] bufferPendingUnscaledTime = new float[2];
    private float[] bufferPendingTimeScale = new float[2];
    private SyncPacketSourceBundleContext[] bufferPendingSourceBundleContext = new SyncPacketSourceBundleContext[2];
    private string[] bufferPendingPacketKey = new string[2];
    private int[] bufferPendingOwnerUpdateId = new int[2];
    private int[] bufferPendingOwnerUnityFrameCount = new int[2];
    private float[] bufferPendingOwnerUnityTime = new float[2];
    private int[] bufferPendingSourceContextQueueDepth = new int[2];
    private int[] bufferPendingSourceContextDroppedStaleCount = new int[2];
    private int[] bufferPendingSourceContextMissingCount = new int[2];
    private int[] bufferPendingSourceContextFrameDelta = new int[2];
    private float[] bufferPendingSourceContextTimeDeltaMs = new float[2];
    private bool[] bufferPendingActiveTransportEligible = new bool[2];
    private bool[] bufferPendingDebugUnbundledCapture = new bool[2];
    private string[] bufferPendingCameraCaptureContractReason = new string[2];

    private Texture2D texture2D;
    private float captureInterval;
    private float lastCaptureTime;
    private int frameCount = 0;
    private bool warnedAsyncUnsupported = false;
    private int droppedAsyncFrames = 0;
    private int lastDropLogFrame = -999;
    private bool resolutionFixScheduled = false;
    private int lastFrameMarkerId = -1000;
    private float lastCaptureRealtime = -1f;
    private float lastCaptureUnityTime = -1f;
    private float lastSentTimestamp = -1f;
    private double captureClock = 0.0;
    private bool captureClockInitialized = false;
    private int gtSyncFixedTick = 0;
    private int gtSyncCaptureEveryTicks = 1;
    private readonly Queue<PendingUpload> uploadQueue = new Queue<PendingUpload>();
    private bool uploadWorkerRunning = false;
    private bool shuttingDown = false;
    private int startupGraceRemaining = 0;
    private string latestSyncPacketKey = "";
    private int latestSyncPacketFrameId = -1;
    private int latestSyncPacketUnityFrameCount = -1;
    private float latestSyncPacketTimestamp = -1f;
    private readonly Queue<SyncPacketSourceBundleContext> pendingSyncPacketContexts = new Queue<SyncPacketSourceBundleContext>();
    private int sourcePacketContextDroppedStaleCount = 0;
    private int sourcePacketContextMissingCount = 0;

    private struct PendingUpload
    {
        public byte[] imageData;
        public float timestamp;
        public int frameId;
        public int captureUnityFrameCount;
        public float captureUnityTime;
        public float captureRealtimeSinceStartup;
        public float captureUnscaledTime;
        public float captureTimeScale;
        public string packetKey;
        public SyncPacketSourceBundleContext sourceBundleContext;
        public int sourceContextQueueDepth;
        public int sourceContextDroppedStaleCount;
        public int sourceContextMissingCount;
        public int sourceContextFrameDelta;
        public float sourceContextTimeDeltaMs;
        public bool activeTransportEligible;
        public bool debugUnbundledCapture;
        public string cameraCaptureContractReason;
    }

    private struct SyncPacketContextSelection
    {
        public SyncPacketSourceBundleContext context;
        public bool matched;
        public bool activeTransportEligible;
        public bool debugUnbundledCapture;
        public string cameraCaptureContractReason;
        public int queueDepthBefore;
        public int staleDropDelta;
        public int missingDelta;
        public int contextFrameDelta;
        public float contextTimeDeltaMs;
    }

    public string LatestSyncPacketKey => latestSyncPacketKey;
    public int LatestSyncPacketFrameId => latestSyncPacketFrameId;
    public int LatestSyncPacketUnityFrameCount => latestSyncPacketUnityFrameCount;
    public float LatestSyncPacketTimestamp => latestSyncPacketTimestamp;

    public SyncPacketContextRegistrationResult RegisterSyncPacketContext(
        SyncPacketSourceBundleContext context
    )
    {
        SyncPacketContextRegistrationResult result = new SyncPacketContextRegistrationResult
        {
            attempted = false,
            accepted = false,
            rejectedReason = "",
            dispositionCode = "",
            queueDepthBefore = pendingSyncPacketContexts.Count,
            queueDepthAfter = pendingSyncPacketContexts.Count,
        };
        if (context == null)
        {
            result.rejectedReason = "invalid_context";
            result.dispositionCode = "rejected_invalid_context";
            return result;
        }
        string normalized = context.packetKey != null ? context.packetKey.Trim() : "";
        context.cameraRequestAttempted = true;
        context.cameraRequestAttemptRealtime = Time.realtimeSinceStartup;
        context.cameraRequestQueueDepth = pendingSyncPacketContexts.Count;
        context.cameraRequestDispositionCode = "attempted";
        result.attempted = true;
        if (string.IsNullOrEmpty(normalized))
        {
            context.cameraRequestRejectedReason = "invalid_packet_key";
            context.cameraRequestDispositionCode = "rejected_invalid_packet_key";
            result.rejectedReason = context.cameraRequestRejectedReason;
            result.dispositionCode = context.cameraRequestDispositionCode;
            return result;
        }
        if (targetCamera == null)
        {
            context.cameraRequestRejectedReason = "camera_unavailable";
            context.cameraRequestDispositionCode = "rejected_camera_unavailable";
            result.rejectedReason = context.cameraRequestRejectedReason;
            result.dispositionCode = context.cameraRequestDispositionCode;
            return result;
        }
        if (context.closed)
        {
            context.cameraRequestRejectedReason = "bundle_closed";
            context.cameraRequestDispositionCode = "rejected_bundle_closed";
            result.rejectedReason = context.cameraRequestRejectedReason;
            result.dispositionCode = context.cameraRequestDispositionCode;
            return result;
        }
        if (context.deadlineRealtime > 0.0f && Time.realtimeSinceStartup > context.deadlineRealtime)
        {
            context.cameraRequestRejectedReason = "out_of_budget";
            context.cameraRequestDispositionCode = "rejected_out_of_budget";
            result.rejectedReason = context.cameraRequestRejectedReason;
            result.dispositionCode = context.cameraRequestDispositionCode;
            return result;
        }
        context.packetKey = normalized;
        context.cameraRequested = true;
        int maxPendingContexts = Mathf.Max(4, maxUploadQueueSize * 4);
        if (pendingSyncPacketContexts.Count >= maxPendingContexts)
        {
            context.cameraRequestRejectedReason = "queue_full";
            context.cameraRequestDispositionCode = "rejected_queue_full";
            result.rejectedReason = context.cameraRequestRejectedReason;
            result.dispositionCode = context.cameraRequestDispositionCode;
            return result;
        }
        pendingSyncPacketContexts.Enqueue(context);
        context.cameraRequestAccepted = true;
        context.cameraRequestAcceptRealtime = Time.realtimeSinceStartup;
        context.cameraRequestQueueDepth = pendingSyncPacketContexts.Count;
        context.cameraRequestRejectedReason = "";
        context.cameraRequestSkippedReason = "";
        context.cameraRequestDispositionCode = "accepted";
        result.accepted = true;
        result.dispositionCode = context.cameraRequestDispositionCode;
        result.queueDepthAfter = pendingSyncPacketContexts.Count;
        return result;
    }
    
    void Start()
    {
        bool? cliGtSync = GroundTruthReporter.ParseCommandLineBool(
            System.Environment.GetCommandLineArgs(),
            "--gt-sync-capture"
        );
        if (cliGtSync.HasValue)
        {
            gtSyncCapture = cliGtSync.Value;
        }
        bool? cliGtCameraSendAsync = GroundTruthReporter.ParseCommandLineBool(
            System.Environment.GetCommandLineArgs(),
            "--gt-camera-send-async"
        );
        if (cliGtCameraSendAsync.HasValue)
        {
            gtCameraSendAsync = cliGtCameraSendAsync.Value;
        }
        bool? cliGtReduceTopDownRate = GroundTruthReporter.ParseCommandLineBool(
            System.Environment.GetCommandLineArgs(),
            "--gt-reduce-topdown-rate"
        );
        if (cliGtReduceTopDownRate.HasValue)
        {
            gtReduceTopDownRate = cliGtReduceTopDownRate.Value;
        }
        float? cliGtSyncFixedDelta = ParseCommandLineFloat(
            System.Environment.GetCommandLineArgs(),
            "--gt-sync-fixed-delta"
        );
        if (cliGtSyncFixedDelta.HasValue && cliGtSyncFixedDelta.Value > 0f)
        {
            gtSyncFixedDeltaSeconds = cliGtSyncFixedDelta.Value;
        }
        int? cliJpegQuality = ParseCommandLineInt(
            System.Environment.GetCommandLineArgs(),
            "--gt-jpeg-quality"
        );
        if (cliJpegQuality.HasValue)
        {
            jpegQuality = Mathf.Clamp(cliJpegQuality.Value, 10, 100);
        }
        if (gtSyncCapture && disableAsyncReadbackInGtSync)
        {
            useAsyncGPUReadback = false;
        }
        if (gtSyncCapture && gtSyncForceFixedDelta && !gtSyncTimeConfigured)
        {
            Time.fixedDeltaTime = Mathf.Max(0.001f, gtSyncFixedDeltaSeconds);
            gtSyncTimeConfigured = true;
            Debug.Log($"CameraCapture: GT sync fixedDeltaTime set to {Time.fixedDeltaTime:F6}s");
        }

        // Keep capturing when the player window loses focus.
        Application.runInBackground = true;
        if (showDebugInfo)
        {
            Debug.Log($"CameraCapture: runInBackground={Application.runInBackground}");
        }

        // Try to find AVCamera by name first (more reliable than tag)
        if (targetCamera == null)
        {
            Debug.Log("CameraCapture: targetCamera is null, searching for AVCamera by name...");
            GameObject avCameraObj = GameObject.Find("AVCamera");
            if (avCameraObj != null)
            {
                Debug.Log($"CameraCapture: Found GameObject 'AVCamera'");
                targetCamera = avCameraObj.GetComponent<Camera>();
                if (targetCamera != null)
                {
                    Debug.Log($"CameraCapture: ✅ Found AVCamera by name - Using '{targetCamera.name}' (moves with car)");
                }
                else
                {
                    Debug.LogWarning("CameraCapture: GameObject 'AVCamera' found but has no Camera component!");
                }
            }
            else
            {
                Debug.LogWarning("CameraCapture: GameObject 'AVCamera' not found by name!");
            }
        }
        else
        {
            Debug.Log($"CameraCapture: targetCamera already assigned: '{targetCamera.name}'");
        }
        
        // Fallback to main camera if AVCamera not found
        if (targetCamera == null)
        {
            Debug.LogWarning("CameraCapture: AVCamera not found, falling back to Camera.main...");
            targetCamera = Camera.main;
            if (targetCamera != null)
            {
                Debug.LogWarning($"CameraCapture: ⚠️ FALLBACK - Using Camera.main '{targetCamera.name}' (static scene camera, does NOT move with car!)");
                Debug.LogWarning($"CameraCapture: This means AVCamera was not found - check Unity hierarchy!");
            }
            else
            {
                Debug.LogError("CameraCapture: Camera.main is also null!");
            }
        }
        
        if (targetCamera == null)
        {
            Debug.LogError("CameraCapture: No camera assigned! Disabling component.");
            enabled = false;
            return;
        }
        
        // Final confirmation with details
        Debug.Log($"CameraCapture: ✅ FINAL - Using camera '{targetCamera.name}'");
        Debug.Log($"CameraCapture: Camera position: {targetCamera.transform.position}");
        Debug.Log($"CameraCapture: Camera parent: {(targetCamera.transform.parent != null ? targetCamera.transform.parent.name : "None (static)")}");
        Debug.Log($"CameraCapture: Camera moves with car: {(targetCamera.transform.parent != null ? "YES ✅" : "NO ⚠️ (static camera)")}");

        // Apply consistency settings before capture setup
        ApplyRuntimeConsistency();

        // Calculate capture interval
        captureInterval = 1.0f / targetFPS;
        if (gtSyncCapture)
        {
            float fixedHz = 1.0f / Mathf.Max(0.001f, Time.fixedDeltaTime);
            float targetHz = Mathf.Max(1f, targetFPS);
            gtSyncCaptureEveryTicks = Mathf.Max(1, Mathf.RoundToInt(fixedHz / targetHz));
            if (gtReduceTopDownRate && string.Equals(cameraId, "top_down", StringComparison.OrdinalIgnoreCase))
            {
                // Top-down is diagnostic-only; reduce capture load to preserve front-camera cadence.
                gtSyncCaptureEveryTicks = Mathf.Max(1, gtSyncCaptureEveryTicks * 2);
            }
            gtSyncFixedTick = 0;
            if (showDebugInfo)
            {
                Debug.Log(
                    $"CameraCapture: GT sync cadence fixedHz={fixedHz:F2} targetHz={targetHz:F2} " +
                    $"captureEveryTicks={gtSyncCaptureEveryTicks}"
                );
            }
        }
        
        // Ensure camera is enabled and rendering to screen (not RenderTexture)
        targetCamera.targetTexture = null;

        // Create double-buffered render textures for overlapping render + readback
        for (int i = 0; i < 2; i++)
        {
            renderTextures[i] = new RenderTexture(captureWidth, captureHeight, 24, RenderTextureFormat.ARGB32);
            bufferReadbackInFlight[i] = false;
            bufferPendingFrameId[i] = 0;
            bufferPendingTimestamp[i] = 0f;
            bufferPendingReadbackStart[i] = -1f;
        }
        activeBufferIndex = 0;

        // Create texture2D for encoding
        texture2D = new Texture2D(captureWidth, captureHeight, TextureFormat.RGBA32, false);
        
        Debug.Log($"CameraCapture: Initialized - {captureWidth}x{captureHeight} @ {targetFPS} FPS");
        Debug.Log($"CameraCapture: Camera '{targetCamera.name}' is enabled: {targetCamera.enabled}, targetTexture: {(targetCamera.targetTexture == null ? "null (rendering to screen)" : targetCamera.targetTexture.name)}");
        LogRuntimeSettings();
        if (useAsyncGPUReadback && !SystemInfo.supportsAsyncGPUReadback)
        {
            Debug.LogWarning("CameraCapture: AsyncGPUReadback not supported on this platform; using synchronous ReadPixels fallback.");
            warnedAsyncUnsupported = true;
        }
        else if (useAsyncGPUReadback)
        {
            Debug.Log("CameraCapture: AsyncGPUReadback enabled.");
        }
        if (gtCameraSendAsync)
        {
            StartCoroutine(UploadWorkerLoop());
        }
        startupGraceRemaining = Mathf.Max(0, startupGraceFrames);
        if (startupGraceRemaining > 0)
        {
            Debug.Log($"CameraCapture: Startup grace period: skipping first {startupGraceRemaining} capture cycles (shader warmup).");
        }
    }
    
    void Update()
    {
        // CRITICAL: Don't capture/send frames when Unity is exiting play mode
        // During play mode exit, Update() may still be called but Time.time is frozen
        // This prevents sending duplicate frames with frozen timestamps
        #if UNITY_EDITOR
        if (!EditorApplication.isPlaying)
        {
            return; // Don't send data when exiting play mode
        }
        #endif
        
        if (gtSyncCapture)
        {
            return;
        }

        float nowTime = Time.time;
        bool scheduledCaptureDue = nowTime - lastCaptureTime >= captureInterval;
        bool requestDrivenCaptureDue =
            requestDrivenSyncCapture &&
            pendingSyncPacketContexts.Count > 0 &&
            nowTime - lastCaptureTime >= Mathf.Max(0.0f, requestDrivenCaptureMinIntervalSeconds);
        if (scheduledCaptureDue || requestDrivenCaptureDue)
        {
            CaptureAndSend();
            lastCaptureTime = nowTime;
        }
    }

    void FixedUpdate()
    {
        if (!gtSyncCapture)
        {
            return;
        }
        #if UNITY_EDITOR
        if (!EditorApplication.isPlaying)
        {
            return;
        }
        #endif

        gtSyncFixedTick += 1;
        if (((gtSyncFixedTick - 1) % gtSyncCaptureEveryTicks) == 0)
        {
            CaptureAndSend();
        }
    }
    
    void CaptureAndSend()
    {
        // Startup grace period: skip first N capture cycles to let GPU finish
        // shader compilation from ShaderPrewarmer. Without this, the first few
        // frames see 200-300ms stalls as shaders compile on-demand.
        if (startupGraceRemaining > 0)
        {
            startupGraceRemaining--;
            if (startupGraceRemaining == 0)
            {
                Debug.Log("CameraCapture: Startup grace period complete. Capturing begins.");
            }
            return;
        }

        // Double buffering: check if the CURRENT buffer is still in-flight.
        // With 2 buffers, we can have 1 readback in flight while rendering to the other.
        // Only drop if BOTH buffers are in-flight (shouldn't happen at 30fps with <33ms readback).
        if (useAsyncGPUReadback && SystemInfo.supportsAsyncGPUReadback
            && bufferReadbackInFlight[activeBufferIndex])
        {
            droppedAsyncFrames++;
            if (showDebugInfo && (Time.frameCount - lastDropLogFrame) >= 30)
            {
                Debug.LogWarning($"CameraCapture: Dropping frame (buffer[{activeBufferIndex}] readback pending). Dropped={droppedAsyncFrames}");
                lastDropLogFrame = Time.frameCount;
            }
            return;
        }

        // Render to the active buffer
        RenderTexture currentRT = renderTextures[activeBufferIndex];
        RenderTexture previousActive = RenderTexture.active;
        RenderTexture previousTarget = targetCamera.targetTexture;
        Rect previousRect = targetCamera.rect;

        targetCamera.targetTexture = currentRT;
        targetCamera.rect = new Rect(0f, 0f, 1f, 1f);
        targetCamera.Render();
        targetCamera.targetTexture = previousTarget;
        targetCamera.rect = previousRect;

        // In GT sync mode, use fixed-step simulation time so timestamps reflect real tick cadence.
        // frameCount*captureInterval drifts when capture is intentionally subsampled (e.g., top-down).
        float captureTime = gtSyncCapture ? Time.fixedTime : Time.unscaledTime;
        int frameId = frameCount;
        float nowRealtime = Time.realtimeSinceStartup;
        float fallbackDelta = captureInterval;
        if (lastCaptureRealtime > 0f)
        {
            fallbackDelta = Mathf.Max(0.001f, nowRealtime - lastCaptureRealtime);
        }
        if (lastCaptureUnityTime > 0f && captureTime <= lastCaptureUnityTime)
        {
            float adjusted = lastCaptureUnityTime + fallbackDelta;
            if (showDebugInfo)
            {
                Debug.LogWarning(
                    $"CameraCapture: Time.time did not advance (t={captureTime:F6}). " +
                    $"Adjusting to {adjusted:F6} (frameId={frameId}, unityFrame={Time.frameCount})."
                );
            }
            captureTime = adjusted;
        }

        if (!captureClockInitialized)
        {
            captureClock = captureTime;
            captureClockInitialized = true;
        }
        else
        {
            captureClock += captureInterval;
            double drift = captureTime - captureClock;
            if (Mathf.Abs((float)drift) > captureGapWarnSeconds)
            {
                if (showDebugInfo)
                {
                    Debug.LogWarning(
                        $"CameraCapture: Capture clock drift {drift:F3}s. " +
                        $"Resyncing to {captureTime:F6} (frameId={frameId})."
                    );
                }
                captureClock = captureTime;
            }
        }
        captureTime = (float)captureClock;
        if (lastCaptureUnityTime > 0f)
        {
            float timeGap = captureTime - lastCaptureUnityTime;
            if (timeGap > captureGapWarnSeconds)
            {
                Debug.LogWarning(
                    $"CameraCapture: Time.time gap {timeGap:F3}s frameId={frameId} unityFrame={Time.frameCount}"
                );
            }
        }
        lastCaptureUnityTime = captureTime;
        if (lastCaptureRealtime > 0f)
        {
            float gap = nowRealtime - lastCaptureRealtime;
            if (gap > captureGapWarnSeconds)
            {
                Debug.LogWarning(
                    $"CameraCapture: Capture gap {gap:F3}s frameId={frameId} unityFrame={Time.frameCount}"
                );
            }
        }
        lastCaptureRealtime = nowRealtime;
        
        if (useAsyncGPUReadback && SystemInfo.supportsAsyncGPUReadback)
        {
#if UNITY_2018_2_OR_NEWER
            int bufIdx = activeBufferIndex;
            int captureUnityFrameCount = Time.frameCount;
            float captureUnityTime = Time.time;
            float captureRealtime = Time.realtimeSinceStartup;
            float captureUnscaled = Time.unscaledTime;
            float captureTimeScale = Time.timeScale;
            SyncPacketContextSelection syncSelection = SelectSyncPacketContextForCapture(
                captureUnityFrameCount,
                captureUnityTime
            );
            SyncPacketSourceBundleContext syncContext = syncSelection.context;
            string packetKey = syncSelection.activeTransportEligible &&
                syncContext != null &&
                !string.IsNullOrEmpty(syncContext.packetKey)
                ? syncContext.packetKey
                : "";
            bufferReadbackInFlight[bufIdx] = true;
            bufferPendingFrameId[bufIdx] = frameId;
            bufferPendingTimestamp[bufIdx] = captureTime;
            bufferPendingReadbackStart[bufIdx] = nowRealtime;
            bufferPendingUnityFrameCount[bufIdx] = captureUnityFrameCount;
            bufferPendingUnityTime[bufIdx] = captureUnityTime;
            bufferPendingRealtime[bufIdx] = captureRealtime;
            bufferPendingUnscaledTime[bufIdx] = captureUnscaled;
            bufferPendingTimeScale[bufIdx] = captureTimeScale;
            bufferPendingPacketKey[bufIdx] = packetKey;
            bufferPendingSourceBundleContext[bufIdx] = syncContext;
            bufferPendingOwnerUpdateId[bufIdx] = syncContext != null ? syncContext.ownerUpdateId : 0;
            bufferPendingOwnerUnityFrameCount[bufIdx] = syncContext != null ? syncContext.ownerUnityFrameCount : 0;
            bufferPendingOwnerUnityTime[bufIdx] = syncContext != null ? syncContext.ownerUnityTime : 0.0f;
            bufferPendingSourceContextQueueDepth[bufIdx] = syncSelection.queueDepthBefore;
            bufferPendingSourceContextDroppedStaleCount[bufIdx] = syncSelection.staleDropDelta;
            bufferPendingSourceContextMissingCount[bufIdx] = syncSelection.missingDelta;
            bufferPendingSourceContextFrameDelta[bufIdx] = syncSelection.contextFrameDelta;
            bufferPendingSourceContextTimeDeltaMs[bufIdx] = syncSelection.contextTimeDeltaMs;
            bufferPendingActiveTransportEligible[bufIdx] = syncSelection.activeTransportEligible;
            bufferPendingDebugUnbundledCapture[bufIdx] = syncSelection.debugUnbundledCapture;
            bufferPendingCameraCaptureContractReason[bufIdx] = syncSelection.cameraCaptureContractReason ?? "";
            AsyncGPUReadback.Request(currentRT, 0, TextureFormat.RGBA32,
                (AsyncGPUReadbackRequest req) => OnCompleteReadback(req, bufIdx));
            // Swap to the other buffer for the next frame
            activeBufferIndex = 1 - activeBufferIndex;
#endif
        }
        else
        {
            // Synchronous fallback — read from current buffer
            RenderTexture.active = currentRT;
            texture2D.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
            texture2D.Apply();
            RenderTexture.active = previousActive;
            
            // Encode to JPEG
            byte[] imageData = texture2D.EncodeToJPG(jpegQuality);
            
            // Save locally if debug enabled
            if (saveLocalImages && frameId % 30 == 0) // Save every 30 frames
            {
                System.IO.Directory.CreateDirectory($"{Application.dataPath}/../captures");
                System.IO.File.WriteAllBytes(
                    $"{Application.dataPath}/../captures/frame_{frameId:D6}.jpg",
                    imageData
                );
            }
            
            // Send to Python server
            LogFrameMarker(frameId, captureTime);
            int captureUnityFrameCount = Time.frameCount;
            float captureUnityTime = Time.time;
            float captureRealtime = Time.realtimeSinceStartup;
            float captureUnscaled = Time.unscaledTime;
            float captureTimeScale = Time.timeScale;
            SyncPacketContextSelection syncSelection = SelectSyncPacketContextForCapture(
                captureUnityFrameCount,
                captureUnityTime
            );
            SyncPacketSourceBundleContext syncContext = syncSelection.context;
            string packetKey = syncSelection.activeTransportEligible &&
                syncContext != null &&
                !string.IsNullOrEmpty(syncContext.packetKey)
                ? syncContext.packetKey
                : "";
            QueueOrSendImage(
                imageData,
                captureTime,
                frameId,
                captureUnityFrameCount,
                captureUnityTime,
                captureRealtime,
                captureUnscaled,
                captureTimeScale,
                packetKey,
                syncContext,
                syncSelection.queueDepthBefore,
                syncSelection.staleDropDelta,
                syncSelection.missingDelta,
                syncSelection.contextFrameDelta,
                syncSelection.contextTimeDeltaMs,
                syncSelection.activeTransportEligible,
                syncSelection.debugUnbundledCapture,
                syncSelection.cameraCaptureContractReason
            );
            
            frameCount++;
        }
    }

    private void ApplyRuntimeConsistency()
    {
        if (enforceQualityLevel && QualitySettings.names.Length > 0)
        {
            int clamped = Mathf.Clamp(qualityLevelIndex, 0, QualitySettings.names.Length - 1);
            QualitySettings.SetQualityLevel(clamped, true);
        }

        if (enforceScreenResolution)
        {
            // Ensure fullscreen mode is applied before setting resolution (builds can default to FullScreenWindow).
            Screen.fullScreenMode = screenMode;
            Screen.SetResolution(screenWidth, screenHeight, screenMode);
            if (!resolutionFixScheduled)
            {
                resolutionFixScheduled = true;
                StartCoroutine(ApplyResolutionAfterFrame());
            }
        }

        if (enforceCameraFov && targetCamera != null)
        {
            targetCamera.fieldOfView = cameraFieldOfView;
        }

        if (enforceCameraAspect && targetCamera != null)
        {
            targetCamera.aspect = (float)captureWidth / captureHeight;
        }
    }

    private void LogRuntimeSettings()
    {
        string rpName = "Built-in";
        float? renderScale = null;
        if (GraphicsSettings.currentRenderPipeline != null)
        {
            rpName = GraphicsSettings.currentRenderPipeline.name;
            PropertyInfo renderScaleProp = GraphicsSettings.currentRenderPipeline.GetType()
                .GetProperty("renderScale", BindingFlags.Public | BindingFlags.Instance);
            if (renderScaleProp != null && renderScaleProp.PropertyType == typeof(float))
            {
                renderScale = (float)renderScaleProp.GetValue(GraphicsSettings.currentRenderPipeline);
            }
        }

        string qualityName = QualitySettings.names.Length > 0
            ? QualitySettings.names[QualitySettings.GetQualityLevel()]
            : "unknown";
        string renderScaleText = renderScale.HasValue ? $" renderScale={renderScale.Value:F2}" : "";
        Debug.Log(
            $"CameraCapture: Runtime settings | Screen={Screen.width}x{Screen.height} {Screen.fullScreenMode} " +
            $"Quality={qualityName}({QualitySettings.GetQualityLevel()}) RP={rpName}{renderScaleText}"
        );
        Debug.Log(
            $"CameraCapture: Capture settings | capture={captureWidth}x{captureHeight} " +
            $"cameraPixels={targetCamera.pixelWidth}x{targetCamera.pixelHeight} " +
            $"cameraAspect={targetCamera.aspect:F3} fov={targetCamera.fieldOfView:F2}"
        );
    }

    private IEnumerator ApplyResolutionAfterFrame()
    {
        // Some platforms ignore SetResolution during the first frame if fullscreen is active.
        yield return null;
        Screen.fullScreenMode = screenMode;
        Screen.SetResolution(screenWidth, screenHeight, screenMode);
        LogRuntimeSettings();
    }

#if UNITY_2018_2_OR_NEWER
    private void OnCompleteReadback(AsyncGPUReadbackRequest request, int bufferIndex)
    {
        bufferReadbackInFlight[bufferIndex] = false;
        if (request.hasError)
        {
            if (showDebugInfo)
            {
                Debug.LogWarning($"CameraCapture: AsyncGPUReadback error on buffer[{bufferIndex}]");
            }
            return;
        }

        int fid = bufferPendingFrameId[bufferIndex];
        float ts = bufferPendingTimestamp[bufferIndex];
        float readbackStart = bufferPendingReadbackStart[bufferIndex];
        int captureUnityFrameCount = bufferPendingUnityFrameCount[bufferIndex];
        float captureUnityTime = bufferPendingUnityTime[bufferIndex];
        float captureRealtime = bufferPendingRealtime[bufferIndex];
        float captureUnscaled = bufferPendingUnscaledTime[bufferIndex];
        float captureTimeScale = bufferPendingTimeScale[bufferIndex];
        string packetKey = bufferPendingPacketKey[bufferIndex];
        SyncPacketSourceBundleContext sourceBundleContext = bufferPendingSourceBundleContext[bufferIndex];
        int sourceContextQueueDepth = bufferPendingSourceContextQueueDepth[bufferIndex];
        int sourceContextDroppedStaleCountDelta = bufferPendingSourceContextDroppedStaleCount[bufferIndex];
        int sourceContextMissingCountDelta = bufferPendingSourceContextMissingCount[bufferIndex];
        int sourceContextFrameDelta = bufferPendingSourceContextFrameDelta[bufferIndex];
        float sourceContextTimeDeltaMs = bufferPendingSourceContextTimeDeltaMs[bufferIndex];
        bool activeTransportEligible = bufferPendingActiveTransportEligible[bufferIndex];
        bool debugUnbundledCapture = bufferPendingDebugUnbundledCapture[bufferIndex];
        string cameraCaptureContractReason = bufferPendingCameraCaptureContractReason[bufferIndex];

        var data = request.GetData<byte>();
        texture2D.LoadRawTextureData(data);
        texture2D.Apply();

        if (readbackStart > 0f)
        {
            float readbackDuration = Time.realtimeSinceStartup - readbackStart;
            if (readbackDuration > readbackWarnSeconds)
            {
                Debug.LogWarning(
                    $"CameraCapture: Async readback duration {readbackDuration:F3}s " +
                    $"frameId={fid} buffer[{bufferIndex}] unityFrame={Time.frameCount}"
                );
            }
        }

        byte[] imageData = texture2D.EncodeToJPG(jpegQuality);

        if (saveLocalImages && fid % 30 == 0)
        {
            System.IO.Directory.CreateDirectory($"{Application.dataPath}/../captures");
            System.IO.File.WriteAllBytes(
                $"{Application.dataPath}/../captures/frame_{fid:D6}.jpg",
                imageData
            );
        }

        QueueOrSendImage(
            imageData,
            ts,
            fid,
            captureUnityFrameCount,
            captureUnityTime,
            captureRealtime,
            captureUnscaled,
            captureTimeScale,
            packetKey,
            sourceBundleContext,
            sourceContextQueueDepth,
            sourceContextDroppedStaleCountDelta,
            sourceContextMissingCountDelta,
            sourceContextFrameDelta,
            sourceContextTimeDeltaMs,
            activeTransportEligible,
            debugUnbundledCapture,
            cameraCaptureContractReason
        );
        LogFrameMarker(fid, ts);
        frameCount++;
    }
#endif

    private SyncPacketContextSelection SelectSyncPacketContextForCapture(
        int captureUnityFrameCount,
        float captureUnityTime
    )
    {
        SyncPacketContextSelection selection = new SyncPacketContextSelection
        {
            queueDepthBefore = pendingSyncPacketContexts.Count,
            contextFrameDelta = 0,
            contextTimeDeltaMs = float.NaN,
            activeTransportEligible = false,
            debugUnbundledCapture = true,
            cameraCaptureContractReason = "no_source_bundle_context",
        };
        if (pendingSyncPacketContexts.Count <= 0)
        {
            sourcePacketContextMissingCount += 1;
            selection.missingDelta = 1;
            return selection;
        }

        float timeBudgetMs =
            Mathf.Max(0.0f, syncPacketContextMaxTimeDeltaSeconds) * 1000f;
        int frameBudget = Mathf.Max(0, syncPacketContextMaxFrameDelta);
        int originalDepth = pendingSyncPacketContexts.Count;
        int staleDropDelta = 0;
        bool retainedPendingContext = false;
        Queue<SyncPacketSourceBundleContext> retainedContexts =
            new Queue<SyncPacketSourceBundleContext>(originalDepth);

        while (pendingSyncPacketContexts.Count > 0)
        {
            SyncPacketSourceBundleContext context = pendingSyncPacketContexts.Dequeue();
            if (context == null)
            {
                continue;
            }
            if (context.closed)
            {
                continue;
            }

            int frameDelta = captureUnityFrameCount - context.ownerUnityFrameCount;
            float timeDeltaMs = (captureUnityTime - context.ownerUnityTime) * 1000f;
            bool deadlineExpired =
                context.deadlineRealtime > 0.0f &&
                Time.realtimeSinceStartup > context.deadlineRealtime;
            bool withinFrameBudget = Mathf.Abs(frameDelta) <= frameBudget;
            bool withinTimeBudget = Mathf.Abs(timeDeltaMs) <= timeBudgetMs;
            bool matched = !deadlineExpired && withinFrameBudget && withinTimeBudget;

            if (selection.context == null && matched)
            {
                selection.context = context;
                selection.matched = true;
                selection.contextFrameDelta = frameDelta;
                selection.contextTimeDeltaMs = timeDeltaMs;
                context.cameraCaptured = true;
                selection.activeTransportEligible = true;
                selection.debugUnbundledCapture = false;
                selection.cameraCaptureContractReason = "";
                continue;
            }

            bool stale =
                deadlineExpired ||
                frameDelta > frameBudget ||
                timeDeltaMs > timeBudgetMs;
            if (stale)
            {
                staleDropDelta += 1;
                if (!context.closed)
                {
                    context.Close(
                        deadlineExpired
                            ? "camera_capture_deadline_expired"
                            : "camera_capture_superseded_before_send"
                    );
                }
                continue;
            }

            retainedPendingContext = true;
            retainedContexts.Enqueue(context);
        }

        pendingSyncPacketContexts.Clear();
        while (retainedContexts.Count > 0)
        {
            pendingSyncPacketContexts.Enqueue(retainedContexts.Dequeue());
        }

        if (selection.context == null)
        {
            sourcePacketContextMissingCount += 1;
            selection.missingDelta = 1;
            selection.cameraCaptureContractReason = retainedPendingContext
                ? "source_bundle_context_pending"
                : "source_bundle_context_mismatched";
        }
        if (staleDropDelta > 0)
        {
            sourcePacketContextDroppedStaleCount += staleDropDelta;
        }
        selection.queueDepthBefore = originalDepth;
        selection.staleDropDelta = staleDropDelta;
        return selection;
    }

    private void QueueOrSendImage(
        byte[] imageData,
        float timestamp,
        int frameId,
        int captureUnityFrameCount,
        float captureUnityTime,
        float captureRealtimeSinceStartup,
        float captureUnscaledTime,
        float captureTimeScale,
        string packetKeyOverride,
        SyncPacketSourceBundleContext sourceBundleContext,
        int sourceContextQueueDepth,
        int sourceContextDroppedStaleCountDelta,
        int sourceContextMissingCountDelta,
        int sourceContextFrameDelta,
        float sourceContextTimeDeltaMs,
        bool activeTransportEligible,
        bool debugUnbundledCapture,
        string cameraCaptureContractReason
    )
    {
        string packetKey = !string.IsNullOrEmpty(packetKeyOverride)
            ? packetKeyOverride
            : (sourceBundleContext != null ? sourceBundleContext.packetKey : "");
        if (activeTransportEligible && sourceBundleContext != null)
        {
            latestSyncPacketKey = packetKey ?? "";
            latestSyncPacketFrameId = frameId;
            latestSyncPacketUnityFrameCount = captureUnityFrameCount;
            latestSyncPacketTimestamp = timestamp;
        }

        if (!gtCameraSendAsync)
        {
            StartCoroutine(
                SendImageToServer(
                    imageData,
                    timestamp,
                    frameId,
                    captureUnityFrameCount,
                    captureUnityTime,
                    captureRealtimeSinceStartup,
                    captureUnscaledTime,
                    captureTimeScale,
                    packetKey,
                    sourceBundleContext,
                    sourceContextQueueDepth,
                    sourceContextDroppedStaleCountDelta,
                    sourceContextMissingCountDelta,
                    sourceContextFrameDelta,
                    sourceContextTimeDeltaMs,
                    activeTransportEligible,
                    debugUnbundledCapture,
                    cameraCaptureContractReason
                )
            );
            return;
        }

        if (uploadQueue.Count >= Mathf.Max(1, maxUploadQueueSize))
        {
            PendingUpload dropped = uploadQueue.Dequeue();
            if (dropped.sourceBundleContext != null && !dropped.sourceBundleContext.closed)
            {
                dropped.sourceBundleContext.Close("camera_requested_not_sent");
            }
        }
        uploadQueue.Enqueue(new PendingUpload
        {
            imageData = imageData,
            timestamp = timestamp,
            frameId = frameId,
            captureUnityFrameCount = captureUnityFrameCount,
            captureUnityTime = captureUnityTime,
            captureRealtimeSinceStartup = captureRealtimeSinceStartup,
            captureUnscaledTime = captureUnscaledTime,
            captureTimeScale = captureTimeScale,
            packetKey = packetKey,
            sourceBundleContext = sourceBundleContext,
            sourceContextQueueDepth = sourceContextQueueDepth,
            sourceContextDroppedStaleCount = sourceContextDroppedStaleCountDelta,
            sourceContextMissingCount = sourceContextMissingCountDelta,
            sourceContextFrameDelta = sourceContextFrameDelta,
            sourceContextTimeDeltaMs = sourceContextTimeDeltaMs,
            activeTransportEligible = activeTransportEligible,
            debugUnbundledCapture = debugUnbundledCapture,
            cameraCaptureContractReason = cameraCaptureContractReason ?? "",
        });
        if (sourceBundleContext != null)
        {
            sourceBundleContext.cameraEnqueued = true;
        }
    }

    private IEnumerator UploadWorkerLoop()
    {
        if (uploadWorkerRunning)
        {
            yield break;
        }
        uploadWorkerRunning = true;
        while (!shuttingDown)
        {
            if (uploadQueue.Count == 0)
            {
                yield return null;
                continue;
            }
            PendingUpload item = uploadQueue.Dequeue();
            yield return StartCoroutine(
                SendImageToServer(
                    item.imageData,
                    item.timestamp,
                    item.frameId,
                    item.captureUnityFrameCount,
                    item.captureUnityTime,
                    item.captureRealtimeSinceStartup,
                    item.captureUnscaledTime,
                    item.captureTimeScale,
                    item.packetKey,
                    item.sourceBundleContext,
                    item.sourceContextQueueDepth,
                    item.sourceContextDroppedStaleCount,
                    item.sourceContextMissingCount,
                    item.sourceContextFrameDelta,
                    item.sourceContextTimeDeltaMs,
                    item.activeTransportEligible,
                    item.debugUnbundledCapture,
                    item.cameraCaptureContractReason
                )
            );
        }
        uploadWorkerRunning = false;
    }

    private void LogFrameMarker(int frameId, float captureTime)
    {
        if (!logFrameMarkers)
        {
            return;
        }

        if (frameId % 100 == 0 && frameId != lastFrameMarkerId)
        {
            lastFrameMarkerId = frameId;
            Debug.Log(
                $"[CAMERA FRAME MARKER] frameId={frameId} unityFrame={Time.frameCount} " +
                $"time={Time.time:F3} unscaled={Time.unscaledTime:F3} " +
                $"realtime={Time.realtimeSinceStartup:F3} timescale={Time.timeScale:F2} " +
                $"captureTime={captureTime:F3}"
            );
        }
    }
    
    IEnumerator SendImageToServer(
        byte[] imageData,
        float timestamp,
        int frameId,
        int captureUnityFrameCount,
        float captureUnityTime,
        float captureRealtimeSinceStartup,
        float captureUnscaledTime,
        float captureTimeScale,
        string packetKeyOverride,
        SyncPacketSourceBundleContext sourceBundleContext,
        int sourceContextQueueDepth,
        int sourceContextDroppedStaleCountDelta,
        int sourceContextMissingCountDelta,
        int sourceContextFrameDelta,
        float sourceContextTimeDeltaMs,
        bool activeTransportEligible,
        bool debugUnbundledCapture,
        string cameraCaptureContractReason
    )
    {
        string url = $"{apiUrl}{cameraEndpoint}";
        float sendTimestamp = timestamp;
        if (lastSentTimestamp > 0f && sendTimestamp <= lastSentTimestamp)
        {
            float minStep = Mathf.Max(0.001f, Time.unscaledDeltaTime);
            float adjusted = lastSentTimestamp + minStep;
            if (showDebugInfo)
            {
                Debug.LogWarning(
                    $"CameraCapture: Non-monotonic timestamp {sendTimestamp:F6} " +
                    $"(last={lastSentTimestamp:F6}). Adjusting to {adjusted:F6}."
                );
            }
            sendTimestamp = adjusted;
        }
        lastSentTimestamp = sendTimestamp;
        
        // Create form data
        WWWForm form = new WWWForm();
        string packetKey = !string.IsNullOrEmpty(packetKeyOverride)
            ? packetKeyOverride
            : (sourceBundleContext != null ? sourceBundleContext.packetKey : "");
        int ownerUpdateId = sourceBundleContext != null ? sourceBundleContext.ownerUpdateId : 0;
        int ownerUnityFrameCount = sourceBundleContext != null ? sourceBundleContext.ownerUnityFrameCount : 0;
        float ownerUnityTime = sourceBundleContext != null ? sourceBundleContext.ownerUnityTime : 0.0f;
        float nowRealtime = Time.realtimeSinceStartup;
        form.AddBinaryData("image", imageData, "frame.jpg", "image/jpeg");
        form.AddField("timestamp", sendTimestamp.ToString("R"));
        form.AddField("frame_id", frameId.ToString());
        form.AddField("camera_id", cameraId);
        form.AddField("sync_packet_key", packetKey ?? "");
        form.AddField(
            "source_bundle_active_transport_eligible",
            activeTransportEligible ? "1" : "0"
        );
        form.AddField(
            "source_bundle_debug_unbundled_capture",
            debugUnbundledCapture ? "1" : "0"
        );
        form.AddField(
            "camera_capture_contract_reason",
            cameraCaptureContractReason ?? ""
        );
        form.AddField("source_packet_owner", "avbridge_update");
        form.AddField("source_packet_owner_update_id", ownerUpdateId.ToString());
        form.AddField("source_packet_owner_unity_frame_count", ownerUnityFrameCount.ToString());
        form.AddField("source_packet_owner_unity_time", ownerUnityTime.ToString("R"));
        form.AddField(
            "source_bundle_close_reason",
            sourceBundleContext != null ? (sourceBundleContext.closeReason ?? "") : ""
        );
        form.AddField(
            "source_bundle_deadline_ms",
            sourceBundleContext != null ? sourceBundleContext.DeadlineMs().ToString("R") : "0"
        );
        form.AddField(
            "source_bundle_age_ms",
            sourceBundleContext != null ? sourceBundleContext.AgeMs(nowRealtime).ToString("R") : "0"
        );
        form.AddField(
            "source_bundle_inflight_count",
            sourceBundleContext != null ? sourceBundleContext.inflightCount.ToString() : "0"
        );
        form.AddField(
            "source_bundle_camera_requested",
            sourceBundleContext != null && sourceBundleContext.cameraRequested ? "1" : "0"
        );
        form.AddField(
            "source_bundle_camera_request_attempted",
            sourceBundleContext != null && sourceBundleContext.cameraRequestAttempted ? "1" : "0"
        );
        form.AddField(
            "source_bundle_camera_request_accepted",
            sourceBundleContext != null && sourceBundleContext.cameraRequestAccepted ? "1" : "0"
        );
        form.AddField(
            "source_bundle_camera_request_rejected_reason",
            sourceBundleContext != null ? (sourceBundleContext.cameraRequestRejectedReason ?? "") : ""
        );
        form.AddField(
            "source_bundle_camera_request_skipped_reason",
            sourceBundleContext != null ? (sourceBundleContext.cameraRequestSkippedReason ?? "") : ""
        );
        form.AddField(
            "source_bundle_camera_request_disposition_code",
            sourceBundleContext != null ? (sourceBundleContext.cameraRequestDispositionCode ?? "") : ""
        );
        form.AddField(
            "source_bundle_camera_request_attempt_age_ms",
            sourceBundleContext != null
                ? sourceBundleContext.CameraRequestAttemptAgeMs(nowRealtime).ToString("R")
                : "0"
        );
        form.AddField(
            "source_bundle_camera_request_accept_age_ms",
            sourceBundleContext != null
                ? sourceBundleContext.CameraRequestAcceptAgeMs(nowRealtime).ToString("R")
                : "0"
        );
        form.AddField(
            "source_bundle_camera_request_queue_depth",
            sourceBundleContext != null ? sourceBundleContext.cameraRequestQueueDepth.ToString() : "0"
        );
        form.AddField(
            "source_bundle_camera_sent",
            sourceBundleContext != null && sourceBundleContext.cameraSent ? "1" : "0"
        );
        form.AddField(
            "source_bundle_aborted_before_vehicle_send",
            sourceBundleContext != null && sourceBundleContext.bundleAbortedBeforeVehicleSend ? "1" : "0"
        );
        form.AddField(
            "source_bundle_abort_reason",
            sourceBundleContext != null ? (sourceBundleContext.bundleAbortReason ?? "") : ""
        );
        form.AddField(
            "source_bundle_vehicle_send_blocked_by_camera_request",
            sourceBundleContext != null && sourceBundleContext.vehicleSendBlockedByCameraRequest ? "1" : "0"
        );
        form.AddField(
            "source_bundle_vehicle_state_built",
            sourceBundleContext != null && sourceBundleContext.vehicleStateBuilt ? "1" : "0"
        );
        form.AddField(
            "source_bundle_vehicle_state_enqueued",
            sourceBundleContext != null && sourceBundleContext.vehicleStateEnqueued ? "1" : "0"
        );
        form.AddField(
            "source_bundle_vehicle_state_sent",
            sourceBundleContext != null && sourceBundleContext.vehicleStateSent ? "1" : "0"
        );
        form.AddField(
            "source_bundle_superseded_before_send",
            sourceBundleContext != null && sourceBundleContext.supersededBeforeSend ? "1" : "0"
        );
        form.AddField("source_packet_context_queue_depth", sourceContextQueueDepth.ToString());
        form.AddField(
            "source_packet_context_dropped_stale_count",
            sourceContextDroppedStaleCountDelta.ToString()
        );
        form.AddField(
            "source_packet_context_missing_count",
            sourceContextMissingCountDelta.ToString()
        );
        form.AddField("source_packet_context_frame_delta", sourceContextFrameDelta.ToString());
        form.AddField(
            "source_packet_context_time_delta_ms",
            sourceContextTimeDeltaMs.ToString("R")
        );
        form.AddField("unity_frame_count", captureUnityFrameCount.ToString());
        form.AddField("unity_time", captureUnityTime.ToString("R"));
        form.AddField("realtime_since_startup", captureRealtimeSinceStartup.ToString("R"));
        form.AddField("unscaled_time", captureUnscaledTime.ToString("R"));
        form.AddField("time_scale", captureTimeScale.ToString("F3"));
        
        using (UnityWebRequest request = UnityWebRequest.Post(url, form))
        {
            request.timeout = 1;
            yield return request.SendWebRequest();
            bool requestSucceeded = request.result == UnityWebRequest.Result.Success;
            if (request.result != UnityWebRequest.Result.Success)
            {
                if (showDebugInfo)
                {
                    Debug.LogWarning($"CameraCapture: Failed to send frame - {request.error}");
                }
            }
            else if (showDebugInfo && frameId % 30 == 0)
            {
                Debug.Log($"CameraCapture: Sent frame {frameId}");
            }
            if (sourceBundleContext != null && requestSucceeded)
            {
                sourceBundleContext.cameraSent = true;
                sourceBundleContext.TryMarkComplete();
                if (activeTransportEligible)
                {
                    latestSyncPacketKey = sourceBundleContext.packetKey ?? "";
                    latestSyncPacketFrameId = frameId;
                    latestSyncPacketUnityFrameCount = captureUnityFrameCount;
                    latestSyncPacketTimestamp = sendTimestamp;
                }
            }
        }
    }

    private static float? ParseCommandLineFloat(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == name)
            {
                if (float.TryParse(args[i + 1], out float value))
                {
                    return value;
                }
                return null;
            }
        }
        return null;
    }

    private static int? ParseCommandLineInt(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == name)
            {
                if (int.TryParse(args[i + 1], out int value))
                {
                    return value;
                }
                return null;
            }
        }
        return null;
    }
    
    void OnDestroy()
    {
        shuttingDown = true;
        uploadQueue.Clear();
        // Reset camera target texture to ensure normal rendering
        if (targetCamera != null)
        {
            targetCamera.targetTexture = null;
        }
        
        for (int i = 0; i < renderTextures.Length; i++)
        {
            if (renderTextures[i] != null)
            {
                renderTextures[i].Release();
                Destroy(renderTextures[i]);
                renderTextures[i] = null;
            }
        }
        
        if (texture2D != null)
        {
            Destroy(texture2D);
        }
    }
}
