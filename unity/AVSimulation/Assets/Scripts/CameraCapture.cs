using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using System;
using System.Reflection;
#if UNITY_2018_2_OR_NEWER
using UnityEngine.Rendering;
#endif
#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Captures camera frames and sends them to the Python AV stack server
/// </summary>
public class CameraCapture : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera targetCamera;
    public int captureWidth = 640;
    public int captureHeight = 480;
    public int targetFPS = 30;
    
    [Header("API Settings")]
    public string apiUrl = "http://localhost:8000";
    public string cameraEndpoint = "/api/camera";
    
    [Header("Debug")]
    public bool showDebugInfo = true;
    public bool saveLocalImages = false;
    public bool useAsyncGPUReadback = true;
    public bool logFrameMarkers = true;
    public float captureGapWarnSeconds = 0.2f;
    public float readbackWarnSeconds = 0.2f;

    [Header("Runtime Consistency")]
    public bool enforceScreenResolution = true;
    public int screenWidth = 849;
    public int screenHeight = 439;
    public FullScreenMode screenMode = FullScreenMode.Windowed;
    public bool enforceQualityLevel = true;
    public int qualityLevelIndex = 0;
    public bool enforceCameraFov = true;
    public float cameraFieldOfView = 84.93f;
    public bool enforceCameraAspect = true;
    
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private float captureInterval;
    private float lastCaptureTime;
    private int frameCount = 0;
    private bool readbackInFlight = false;
    private int pendingFrameId = 0;
    private float pendingTimestamp = 0f;
    private bool warnedAsyncUnsupported = false;
    private int droppedAsyncFrames = 0;
    private int lastDropLogFrame = -999;
    private bool resolutionFixScheduled = false;
    private int lastFrameMarkerId = -1000;
    private float lastCaptureRealtime = -1f;
    private float lastCaptureUnityTime = -1f;
    private float pendingReadbackStartRealtime = -1f;
    
    void Start()
    {
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
        
        // Ensure camera is enabled and rendering to screen (not RenderTexture)
        targetCamera.targetTexture = null;
        
        // Create render texture (for manual rendering, not as targetTexture)
        renderTexture = new RenderTexture(captureWidth, captureHeight, 24, RenderTextureFormat.ARGB32);
        
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
        
        // Check if it's time to capture (use realtime to avoid focus/timescale stalls)
        if (Time.realtimeSinceStartup - lastCaptureTime >= captureInterval)
        {
            CaptureAndSend();
            lastCaptureTime = Time.realtimeSinceStartup;
        }
    }
    
    void CaptureAndSend()
    {
        if (useAsyncGPUReadback && SystemInfo.supportsAsyncGPUReadback && readbackInFlight)
        {
            droppedAsyncFrames++;
            if (showDebugInfo && (Time.frameCount - lastDropLogFrame) >= 30)
            {
                Debug.LogWarning($"CameraCapture: Dropping frame (async readback pending). Dropped={droppedAsyncFrames}");
                lastDropLogFrame = Time.frameCount;
            }
            return; // Drop frame if previous readback is still pending
        }
        // Manually render camera to render texture (allows camera to still render to Game view)
        // Use RenderTexture.GetTemporary to avoid RenderPass issues
        RenderTexture previousActive = RenderTexture.active;
        RenderTexture previousTarget = targetCamera.targetTexture;
        
        // Temporarily set target texture for rendering
        targetCamera.targetTexture = renderTexture;
        targetCamera.Render();
        targetCamera.targetTexture = previousTarget; // Restore immediately

        float captureTime = Time.time;
        int frameId = frameCount;
        float nowRealtime = Time.realtimeSinceStartup;
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
            readbackInFlight = true;
            pendingFrameId = frameId;
            pendingTimestamp = captureTime;
            pendingReadbackStartRealtime = nowRealtime;
            AsyncGPUReadback.Request(renderTexture, 0, TextureFormat.RGBA32, OnCompleteReadback);
#endif
        }
        else
        {
            // Read pixels from render texture (synchronous fallback)
            RenderTexture.active = renderTexture;
            texture2D.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
            texture2D.Apply();
            RenderTexture.active = previousActive;
            
            // Encode to JPEG
            byte[] imageData = texture2D.EncodeToJPG(85);
            
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
            StartCoroutine(SendImageToServer(imageData, captureTime, frameId));
            
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
    private void OnCompleteReadback(AsyncGPUReadbackRequest request)
    {
        readbackInFlight = false;
        if (request.hasError)
        {
            if (showDebugInfo)
            {
                Debug.LogWarning("CameraCapture: AsyncGPUReadback error");
            }
            return;
        }
        
        var data = request.GetData<byte>();
        texture2D.LoadRawTextureData(data);
        texture2D.Apply();

        if (pendingReadbackStartRealtime > 0f)
        {
            float readbackDuration = Time.realtimeSinceStartup - pendingReadbackStartRealtime;
            if (readbackDuration > readbackWarnSeconds)
            {
                Debug.LogWarning(
                    $"CameraCapture: Async readback duration {readbackDuration:F3}s frameId={pendingFrameId} unityFrame={Time.frameCount}"
                );
            }
        }
        
        byte[] imageData = texture2D.EncodeToJPG(85);
        
        if (saveLocalImages && pendingFrameId % 30 == 0) // Save every 30 frames
        {
            System.IO.Directory.CreateDirectory($"{Application.dataPath}/../captures");
            System.IO.File.WriteAllBytes(
                $"{Application.dataPath}/../captures/frame_{pendingFrameId:D6}.jpg",
                imageData
            );
        }
        
        StartCoroutine(SendImageToServer(imageData, pendingTimestamp, pendingFrameId));
        LogFrameMarker(pendingFrameId, pendingTimestamp);
        frameCount++;
    }
#endif

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
    
    IEnumerator SendImageToServer(byte[] imageData, float timestamp, int frameId)
    {
        string url = $"{apiUrl}{cameraEndpoint}";
        
        // Create form data
        WWWForm form = new WWWForm();
        form.AddBinaryData("image", imageData, "frame.jpg", "image/jpeg");
        form.AddField("timestamp", timestamp.ToString("F6"));
        form.AddField("frame_id", frameId.ToString());
        form.AddField("realtime_since_startup", Time.realtimeSinceStartup.ToString("F6"));
        form.AddField("unscaled_time", Time.unscaledTime.ToString("F6"));
        form.AddField("time_scale", Time.timeScale.ToString("F3"));
        
        using (UnityWebRequest request = UnityWebRequest.Post(url, form))
        {
            request.timeout = 1;
            yield return request.SendWebRequest();
            
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
        }
    }
    
    void OnDestroy()
    {
        // Reset camera target texture to ensure normal rendering
        if (targetCamera != null)
        {
            targetCamera.targetTexture = null;
        }
        
        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
        }
        
        if (texture2D != null)
        {
            Destroy(texture2D);
        }
    }
}

