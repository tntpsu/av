using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using System;
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
    
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private float captureInterval;
    private float lastCaptureTime;
    private int frameCount = 0;
    
    void Start()
    {
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
        
        // Calculate capture interval
        captureInterval = 1.0f / targetFPS;
        
        // Ensure camera is enabled and rendering to screen (not RenderTexture)
        targetCamera.targetTexture = null;
        
        // Create render texture (for manual rendering, not as targetTexture)
        renderTexture = new RenderTexture(captureWidth, captureHeight, 24);
        
        // Create texture2D for encoding
        texture2D = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
        
        Debug.Log($"CameraCapture: Initialized - {captureWidth}x{captureHeight} @ {targetFPS} FPS");
        Debug.Log($"CameraCapture: Camera '{targetCamera.name}' is enabled: {targetCamera.enabled}, targetTexture: {(targetCamera.targetTexture == null ? "null (rendering to screen)" : targetCamera.targetTexture.name)}");
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
        
        // Check if it's time to capture
        if (Time.time - lastCaptureTime >= captureInterval)
        {
            CaptureAndSend();
            lastCaptureTime = Time.time;
        }
    }
    
    void CaptureAndSend()
    {
        // Manually render camera to render texture (allows camera to still render to Game view)
        // Use RenderTexture.GetTemporary to avoid RenderPass issues
        RenderTexture previousActive = RenderTexture.active;
        RenderTexture previousTarget = targetCamera.targetTexture;
        
        // Temporarily set target texture for rendering
        targetCamera.targetTexture = renderTexture;
        targetCamera.Render();
        targetCamera.targetTexture = previousTarget; // Restore immediately
        
        // Read pixels from render texture
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        texture2D.Apply();
        RenderTexture.active = previousActive;
        
        // Encode to JPEG
        byte[] imageData = texture2D.EncodeToJPG(85);
        
        // Save locally if debug enabled
        if (saveLocalImages && frameCount % 30 == 0) // Save every 30 frames
        {
            System.IO.Directory.CreateDirectory($"{Application.dataPath}/../captures");
            System.IO.File.WriteAllBytes(
                $"{Application.dataPath}/../captures/frame_{frameCount:D6}.jpg",
                imageData
            );
        }
        
        // Send to Python server
        StartCoroutine(SendImageToServer(imageData));
        
        frameCount++;
    }
    
    IEnumerator SendImageToServer(byte[] imageData)
    {
        string url = $"{apiUrl}{cameraEndpoint}";
        
        // Create form data
        WWWForm form = new WWWForm();
        form.AddBinaryData("image", imageData, "frame.jpg", "image/jpeg");
        form.AddField("timestamp", Time.time.ToString("F6"));
        form.AddField("frame_id", frameCount.ToString());
        
        using (UnityWebRequest request = UnityWebRequest.Post(url, form))
        {
            yield return request.SendWebRequest();
            
            if (request.result != UnityWebRequest.Result.Success)
            {
                if (showDebugInfo)
                {
                    Debug.LogWarning($"CameraCapture: Failed to send frame - {request.error}");
                }
            }
            else if (showDebugInfo && frameCount % 30 == 0)
            {
                Debug.Log($"CameraCapture: Sent frame {frameCount}");
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

