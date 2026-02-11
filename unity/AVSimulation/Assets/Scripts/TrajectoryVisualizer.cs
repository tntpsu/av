using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System;
using System.Text;

/// <summary>
/// Visualizes trajectory path in front of the car using LineRenderer.
/// Receives trajectory data from AVBridge and displays it in real-time.
/// </summary>
public class TrajectoryVisualizer : MonoBehaviour
{
    [Header("Components")]
    public LineRenderer trajectoryLine;
    public GameObject referencePointMarker;
    public GameObject vehiclePositionMarker;
    
    [Header("API Settings")]
    public string apiUrl = "http://localhost:8000";
    public string trajectoryEndpoint = "/api/trajectory";
    public float updateRate = 30f; // Hz
    
    [Header("Visualization Settings")]
    public Color goodColor = Color.green;      // Good trajectory (low error)
    public Color warningColor = Color.yellow;  // Warning (moderate error)
    public Color errorColor = Color.red;       // Error (high error)
    public float errorThreshold = 0.5f;        // Lateral error threshold for color coding
    public float maxError = 2.0f;              // Maximum error for color scaling
    public float lineWidth = 0.2f;             // Line width in meters (increased for visibility)
    public float referencePointSize = 0.3f;    // Reference point marker size (increased)
    public float vehicleMarkerSize = 0.3f;     // Vehicle position marker size
    public bool alwaysVisible = true;          // Keep line visible even when no trajectory
    public int maxTrajectoryPoints = 30;       // Maximum points to display (increased from 20)
    public bool showReferenceMarker = false;
    public bool showVehicleMarker = false;

    [Header("Perception Overlay")]
    public bool showPerceptionLines = true;
    public Color perceptionLaneColor = Color.red;
    public Color perceptionCenterColor = Color.cyan;
    public float perceptionLineWidth = 0.15f;
    public float perceptionLineLength = 1.5f;
    
    [Header("Debug")]
    public bool showDebugInfo = true;
    
    private float updateInterval;
    private float lastUpdateTime;
    private Vector3[] trajectoryPoints = new Vector3[0];
    private Vector3 referencePoint = Vector3.zero;
    private float currentLateralError = 0f;
    private Transform vehicleTransform;
    private MaterialPropertyBlock materialPropertyBlock; // For per-instance material properties
    private LineRenderer perceptionLeftLine;
    private LineRenderer perceptionRightLine;
    private LineRenderer perceptionCenterLine;
    private float perceptionLeftX = float.NaN;
    private float perceptionRightX = float.NaN;
    private float perceptionCenterX = float.NaN;
    private float perceptionLookahead = float.NaN;
    private bool perceptionValid = false;
    
    void Start()
    {
        // CRITICAL FIX: Ensure goodColor is actually green (not yellow from Inspector)
        // If goodColor is yellow, the line will always be yellow!
        if (goodColor.r > 0.5f || goodColor.g < 0.5f || goodColor.b > 0.5f)
        {
            if (showDebugInfo)
            {
                Debug.LogWarning($"TrajectoryVisualizer: goodColor is not green! R={goodColor.r}, G={goodColor.g}, B={goodColor.b}. Setting to green.");
            }
            goodColor = Color.green;
        }
        
        // Auto-find vehicle if not assigned
        if (vehicleTransform == null)
        {
            GameObject vehicle = GameObject.FindGameObjectWithTag("Player");
            if (vehicle == null)
            {
                // Try to find CarController
                CarController car = FindObjectOfType<CarController>();
                if (car != null)
                {
                    vehicleTransform = car.transform;
                }
            }
            else
            {
                vehicleTransform = vehicle.transform;
            }
        }
        
        // Create LineRenderer if not assigned
        if (trajectoryLine == null)
        {
            GameObject lineObj = new GameObject("TrajectoryLine");
            lineObj.transform.SetParent(transform);
            trajectoryLine = lineObj.AddComponent<LineRenderer>();
            
            // CRITICAL FIX: Use shader specifically designed for LineRenderer
            // Research shows "Legacy Shaders/Particles/Alpha Blended Premultiply" works best
            // This shader supports both material.color and vertex colors (startColor/endColor)
            Shader lineShader = Shader.Find("Legacy Shaders/Particles/Alpha Blended Premultiply");
            if (lineShader == null)
            {
                // Fallback to Sprites/Default
                lineShader = Shader.Find("Sprites/Default");
            }
            if (lineShader == null)
            {
                Debug.LogError("TrajectoryVisualizer: Could not find compatible shader for LineRenderer!");
            }
            else
            {
                trajectoryLine.material = new Material(lineShader);
                // Set initial color to green (will be updated dynamically)
                trajectoryLine.material.color = new Color(0f, 1f, 0f, 1f);
                if (showDebugInfo)
                {
                    Debug.Log($"TrajectoryVisualizer: Created material with shader: {lineShader.name}, initial color: green");
                }
            }
            
            trajectoryLine.useWorldSpace = true;
            trajectoryLine.startWidth = lineWidth;
            trajectoryLine.endWidth = lineWidth;
            trajectoryLine.positionCount = 0;
            // Enable shadows and improve visibility
            trajectoryLine.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            trajectoryLine.receiveShadows = false;
            
            // CRITICAL: Use solid colors instead of gradient
            // Don't set colorGradient to null (Unity throws exception)
            // Instead, create a simple gradient with solid color
            Gradient gradient = new Gradient();
            gradient.SetKeys(
                new GradientColorKey[] { new GradientColorKey(Color.green, 0.0f), new GradientColorKey(Color.green, 1.0f) },
                new GradientAlphaKey[] { new GradientAlphaKey(1.0f, 0.0f), new GradientAlphaKey(1.0f, 1.0f) }
            );
            trajectoryLine.colorGradient = gradient;
            
            // CRITICAL: Ensure line is visible in Game view
            // Set layer to Default (layer 0) to ensure it's visible
            lineObj.layer = 0; // Default layer
            trajectoryLine.sortingOrder = 100; // Render on top
            trajectoryLine.sortingLayerName = "Default";
            
            // Set colors explicitly
            trajectoryLine.startColor = new Color(0f, 1f, 0f, 1f); // Pure green
            trajectoryLine.endColor = new Color(0f, 1f, 0f, 1f);   // Pure green
            
            // CRITICAL: Make sure line is enabled and visible
            trajectoryLine.enabled = true;
            
            if (showDebugInfo)
            {
                Debug.Log($"TrajectoryVisualizer: Created LineRenderer using Unity's default material (supports vertex colors), goodColor: R={goodColor.r}, G={goodColor.g}, B={goodColor.b}");
            }
        }

        if (showPerceptionLines)
        {
            perceptionLeftLine = CreatePerceptionLine("PerceptionLeftLine", perceptionLaneColor);
            perceptionRightLine = CreatePerceptionLine("PerceptionRightLine", perceptionLaneColor);
            perceptionCenterLine = CreatePerceptionLine("PerceptionCenterLine", perceptionCenterColor);
        }
        else
        {
            // If LineRenderer already exists, ensure it has a material with color property
            if (trajectoryLine.material == null)
            {
                // Create material if it doesn't exist
                Shader lineShader = Shader.Find("Sprites/Default");
                if (lineShader == null)
                {
                    lineShader = Shader.Find("Unlit/Color");
                }
                if (lineShader != null)
                {
                    trajectoryLine.material = new Material(lineShader);
                    trajectoryLine.material.color = new Color(0f, 1f, 0f, 1f);
                    if (showDebugInfo)
                    {
                        Debug.Log($"TrajectoryVisualizer: Created material for existing LineRenderer with shader: {lineShader.name}");
                    }
                }
            }
            
            if (showDebugInfo)
            {
                Debug.Log($"TrajectoryVisualizer: Using existing LineRenderer, goodColor: R={goodColor.r}, G={goodColor.g}, B={goodColor.b}");
            }
        }
        
        // Create reference point marker if enabled
        if (showReferenceMarker && referencePointMarker == null)
        {
            referencePointMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            referencePointMarker.name = "ReferencePointMarker";
            referencePointMarker.transform.SetParent(transform);
            referencePointMarker.transform.localScale = Vector3.one * referencePointSize;
            referencePointMarker.layer = 0; // Default layer - ensure visible in Game view
            referencePointMarker.SetActive(true); // Ensure it's active
            
            // CRITICAL: Create a new material instance (not shared) to avoid issues
            Renderer markerRenderer = referencePointMarker.GetComponent<Renderer>();
            if (markerRenderer != null)
            {
                // Use Unlit/Color shader - simpler and more reliable than Standard
                // This shader directly uses material.color without lighting calculations
                Shader markerShader = Shader.Find("Unlit/Color");
                if (markerShader == null)
                {
                    // Fallback to Standard if Unlit/Color not available
                    markerShader = Shader.Find("Standard");
                }
                
                if (markerShader != null)
                {
                    Material markerMaterial = new Material(markerShader);
                    markerMaterial.color = Color.yellow;
                    
                    // For Standard shader, set additional properties
                    if (markerShader.name == "Standard")
                    {
                        markerMaterial.SetFloat("_Metallic", 0f);
                        markerMaterial.SetFloat("_Glossiness", 0.5f);
                    }
                    
                    markerRenderer.material = markerMaterial;
                    markerRenderer.enabled = true;
                    markerRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    markerRenderer.receiveShadows = false;
                    
                    if (showDebugInfo)
                    {
                        Debug.Log($"TrajectoryVisualizer: Created reference marker material with shader: {markerShader.name}, color: yellow");
                    }
                }
                else
                {
                    Debug.LogError("TrajectoryVisualizer: Could not find shader for reference marker!");
                }
            }
            
            // Remove collider (not needed for visualization)
            Collider markerCollider = referencePointMarker.GetComponent<Collider>();
            if (markerCollider != null)
            {
                Destroy(markerCollider);
            }
            
            if (showDebugInfo)
            {
                bool rendererEnabled = markerRenderer != null ? markerRenderer.enabled : false;
                Debug.Log($"TrajectoryVisualizer: Created reference point marker, layer={referencePointMarker.layer}, active={referencePointMarker.activeSelf}, scale={referencePointMarker.transform.localScale}, renderer.enabled={rendererEnabled}");
            }
        }
        
        // Create vehicle position marker if enabled
        if (showVehicleMarker && vehiclePositionMarker == null && vehicleTransform != null)
        {
            vehiclePositionMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            vehiclePositionMarker.name = "VehiclePositionMarker";
            vehiclePositionMarker.transform.SetParent(vehicleTransform);
            vehiclePositionMarker.transform.localPosition = Vector3.zero;
            vehiclePositionMarker.transform.localScale = Vector3.one * vehicleMarkerSize;
            vehiclePositionMarker.GetComponent<Renderer>().material.color = Color.white;
            // Remove collider
            Destroy(vehiclePositionMarker.GetComponent<Collider>());
        }
        
        if (!showReferenceMarker && referencePointMarker != null)
        {
            referencePointMarker.SetActive(false);
        }
        if (!showVehicleMarker && vehiclePositionMarker != null)
        {
            vehiclePositionMarker.SetActive(false);
        }

        updateInterval = 1.0f / updateRate;
        lastUpdateTime = Time.time;
        
        // Initialize MaterialPropertyBlock for per-instance material properties
        materialPropertyBlock = new MaterialPropertyBlock();
        
        if (showDebugInfo)
        {
            Debug.Log("TrajectoryVisualizer: Initialized");
        }
    }
    
    void Update()
    {
        // Update trajectory periodically
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            StartCoroutine(FetchTrajectory());
            lastUpdateTime = Time.time;
        }
        
        // Update visualization
        UpdateVisualization();
    }
    
    IEnumerator FetchTrajectory()
    {
        string url = $"{apiUrl}{trajectoryEndpoint}";
        
        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            request.timeout = 1; // Short timeout
            yield return request.SendWebRequest();
            
            if (request.result == UnityWebRequest.Result.Success)
            {
                try
                {
                    string jsonResponse = request.downloadHandler.text;
                    TrajectoryData data = JsonUtility.FromJson<TrajectoryData>(jsonResponse);
                    
                    // Convert trajectory points to Unity world coordinates
                    // Python uses: +x right, +y forward, +z up
                    // Unity uses: +x right, +y up, +z forward
                    // Trajectory points are in vehicle frame: x=lateral (right positive), y=forward
                    List<Vector3> points = new List<Vector3>();
                    if (data.trajectory_points != null && data.trajectory_points.Length > 0)
                    {
                        int pointCount = Mathf.Min(data.trajectory_points.Length, maxTrajectoryPoints);
                        for (int i = 0; i < pointCount; i++)
                        {
                            float[] point = data.trajectory_points[i];
                            if (point.Length >= 2)
                            {
                                // Convert from vehicle frame to world frame
                                // Python: (x, y, heading) where x=lateral, y=forward
                                // Unity: (x, y, z) where x=right, y=up, z=forward
                                // Vehicle frame: x is lateral (right positive), y is forward
                                // Unity world: x is right, z is forward, y is up
                                Vector3 vehicleFramePoint = new Vector3(point[0], 0.1f, point[1]); // 0.1m above ground
                                Vector3 worldPoint = vehicleTransform != null 
                                    ? vehicleTransform.TransformPoint(vehicleFramePoint)
                                    : vehicleFramePoint;
                                points.Add(worldPoint);
                            }
                        }
                    }
                    
                    trajectoryPoints = points.ToArray();
                    
                    // Update reference point
                    if (data.reference_point != null)
                    {
                        float[] refPoint = data.reference_point;
                        if (refPoint.Length >= 2)
                        {
                            // Reference point is in vehicle frame: x=lateral, y=forward
                            Vector3 vehicleFrameRef = new Vector3(refPoint[0], 0.2f, refPoint[1]); // 0.2m above ground
                            referencePoint = vehicleTransform != null
                                ? vehicleTransform.TransformPoint(vehicleFrameRef)
                                : vehicleFrameRef;
                        }
                    }
                    
                    // Update lateral error for color coding
                    currentLateralError = data.lateral_error;
                    
                    // Update perception overlay data
                    perceptionValid = data.perception_valid;
                    perceptionLeftX = data.perception_left_lane_x;
                    perceptionRightX = data.perception_right_lane_x;
                    perceptionCenterX = data.perception_center_x;
                    perceptionLookahead = data.perception_lookahead_m;
                    
                    // CRITICAL: Force color update immediately when data is received
                    if (trajectoryLine != null && trajectoryPoints.Length > 0)
                    {
                        Color lineColor = GetColorForError(currentLateralError);
                        // Set material color directly (like lane lines do)
                        if (trajectoryLine.material != null)
                        {
                            trajectoryLine.material.color = lineColor;
                            // Also set _Color property if shader supports it
                            if (trajectoryLine.material.HasProperty("_Color"))
                            {
                                trajectoryLine.material.SetColor("_Color", lineColor);
                            }
                        }
                        
                        // Also set vertex colors as backup
                        trajectoryLine.startColor = lineColor;
                        trajectoryLine.endColor = lineColor;
                        
                        // REMOVED: Toggling enabled was causing Game view visibility issues
                        // Just ensure it's enabled
                        trajectoryLine.enabled = true;
                    }
                    
                    // Debug: Log lateral error periodically to diagnose color issues
                    if (showDebugInfo && Time.frameCount % 30 == 0) // Every ~1 second at 30 FPS
                    {
                        float absError = Mathf.Abs(currentLateralError);
                        string colorStatus = absError < errorThreshold ? "GREEN" : 
                                            (absError < maxError ? "YELLOW" : "RED");
                        Color calculatedColor = GetColorForError(currentLateralError);
                        Debug.Log($"TrajectoryVisualizer: lateral_error={currentLateralError:F4}m (abs={absError:F4}m), threshold={errorThreshold:F2}m, status={colorStatus}");
                        Debug.Log($"TrajectoryVisualizer: Calculated color R={calculatedColor.r:F2}, G={calculatedColor.g:F2}, B={calculatedColor.b:F2}");
                        if (trajectoryLine != null)
                        {
                            Debug.Log($"TrajectoryVisualizer: LineRenderer.startColor R={trajectoryLine.startColor.r:F2}, G={trajectoryLine.startColor.g:F2}, B={trajectoryLine.startColor.b:F2}");
                            Debug.Log($"TrajectoryVisualizer: LineRenderer.endColor R={trajectoryLine.endColor.r:F2}, G={trajectoryLine.endColor.g:F2}, B={trajectoryLine.endColor.b:F2}");
                        }
                    }
                }
                catch (Exception e)
                {
                    if (showDebugInfo)
                    {
                        Debug.LogWarning($"TrajectoryVisualizer: Failed to parse trajectory data - {e.Message}");
                    }
                }
            }
            // Don't log errors for connection issues (bridge might not be ready)
        }
    }
    
    void UpdateVisualization()
    {
        // Update trajectory line
        if (trajectoryLine != null)
        {
            if (trajectoryPoints.Length > 0)
            {
                trajectoryLine.positionCount = trajectoryPoints.Length;
                trajectoryLine.SetPositions(trajectoryPoints);
                trajectoryLine.enabled = true;
                
                // Color code based on lateral error
                Color lineColor = GetColorForError(currentLateralError);
                
                // CRITICAL: Set material color directly (like lane lines do)
                // This is more reliable than vertex colors for LineRenderer
                if (trajectoryLine.material != null)
                {
                    trajectoryLine.material.color = lineColor;
                    // Also set _Color property if shader supports it
                    if (trajectoryLine.material.HasProperty("_Color"))
                    {
                        trajectoryLine.material.SetColor("_Color", lineColor);
                    }
                }
                
                // Also set vertex colors as backup (though material.color takes precedence)
                // Use solid color gradient instead of null
                Gradient gradient = new Gradient();
                gradient.SetKeys(
                    new GradientColorKey[] { new GradientColorKey(lineColor, 0.0f), new GradientColorKey(lineColor, 1.0f) },
                    new GradientAlphaKey[] { new GradientAlphaKey(lineColor.a, 0.0f), new GradientAlphaKey(lineColor.a, 1.0f) }
                );
                trajectoryLine.colorGradient = gradient;
                trajectoryLine.startColor = lineColor;
                trajectoryLine.endColor = lineColor;
                
                // CRITICAL: Ensure line is enabled and visible in Game view
                trajectoryLine.enabled = true;
                
                // Debug: Log if line has points (every ~1 second)
                if (showDebugInfo && Time.frameCount % 30 == 0)
                {
                    Debug.Log($"TrajectoryVisualizer: LineRenderer enabled={trajectoryLine.enabled}, positionCount={trajectoryLine.positionCount}, visible={trajectoryLine.isVisible}");
                }
                
                // Debug: Log actual color being set (every ~1 second)
                if (showDebugInfo && Time.frameCount % 30 == 0)
                {
                    Color actualStartColor = trajectoryLine.startColor;
                    Color actualEndColor = trajectoryLine.endColor;
                    Color materialColor = trajectoryLine.material != null ? trajectoryLine.material.color : Color.clear;
                    
                    Debug.Log($"TrajectoryVisualizer: Calculated color R={lineColor.r:F2}, G={lineColor.g:F2}, B={lineColor.b:F2}");
                    Debug.Log($"TrajectoryVisualizer: LineRenderer.startColor R={actualStartColor.r:F2}, G={actualStartColor.g:F2}, B={actualStartColor.b:F2}");
                    Debug.Log($"TrajectoryVisualizer: LineRenderer.endColor R={actualEndColor.r:F2}, G={actualEndColor.g:F2}, B={actualEndColor.b:F2}");
                    Debug.Log($"TrajectoryVisualizer: Material.color R={materialColor.r:F2}, G={materialColor.g:F2}, B={materialColor.b:F2}");
                    Debug.Log($"TrajectoryVisualizer: goodColor R={goodColor.r:F2}, G={goodColor.g:F2}, B={goodColor.b:F2}");
                }
            }
            else if (alwaysVisible)
            {
                // Keep line visible but empty (will show nothing)
                trajectoryLine.positionCount = 0;
                trajectoryLine.enabled = true;
            }
            else
            {
                trajectoryLine.enabled = false;
            }
        }

        UpdatePerceptionOverlay();
        
        // Update reference point marker
        if (!showReferenceMarker)
        {
            if (referencePointMarker != null)
            {
                referencePointMarker.SetActive(false);
            }
        }
        else if (referencePointMarker == null)
        {
            if (showDebugInfo && Time.frameCount % 30 == 0)
            {
                Debug.LogWarning("TrajectoryVisualizer: referencePointMarker is NULL! Attempting to create it...");
            }
            // Try to create the marker now
            referencePointMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            referencePointMarker.name = "ReferencePointMarker";
            referencePointMarker.transform.SetParent(transform);
            referencePointMarker.transform.localScale = Vector3.one * referencePointSize;
            referencePointMarker.layer = 0; // Default layer - ensure visible in Game view
            referencePointMarker.SetActive(true); // Ensure it's active
            
            // CRITICAL: Create a new material instance (not shared) to avoid issues
            Renderer markerRenderer = referencePointMarker.GetComponent<Renderer>();
            if (markerRenderer != null)
            {
                // Use Unlit/Color shader - simpler and more reliable than Standard
                Shader markerShader = Shader.Find("Unlit/Color");
                if (markerShader == null)
                {
                    markerShader = Shader.Find("Standard");
                }
                
                if (markerShader != null)
                {
                    Material markerMaterial = new Material(markerShader);
                    markerMaterial.color = Color.yellow;
                    
                    // For Standard shader, set additional properties
                    if (markerShader.name == "Standard")
                    {
                        markerMaterial.SetFloat("_Metallic", 0f);
                        markerMaterial.SetFloat("_Glossiness", 0.5f);
                    }
                    
                    markerRenderer.material = markerMaterial;
                    markerRenderer.enabled = true;
                    markerRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    markerRenderer.receiveShadows = false;
                    
                    if (showDebugInfo)
                    {
                        Debug.Log($"TrajectoryVisualizer: Created reference marker material in UpdateVisualization with shader: {markerShader.name}, color: yellow");
                    }
                }
                else
                {
                    Debug.LogError("TrajectoryVisualizer: Could not find shader for reference marker in UpdateVisualization!");
                }
            }
            
            // Remove collider (not needed for visualization)
            Collider markerCollider = referencePointMarker.GetComponent<Collider>();
            if (markerCollider != null)
            {
                Destroy(markerCollider);
            }
            
            if (showDebugInfo)
            {
                Debug.Log($"TrajectoryVisualizer: Created reference point marker in UpdateVisualization, layer={referencePointMarker.layer}, active={referencePointMarker.activeSelf}");
            }
        }
        
        if (showReferenceMarker && referencePointMarker != null)
        {
            if (referencePoint != Vector3.zero)
            {
                referencePointMarker.transform.position = referencePoint;
                // CRITICAL: Ensure marker is active and visible in Game view
                referencePointMarker.SetActive(true);
                referencePointMarker.layer = 0; // Default layer - always set
                
                // CRITICAL: Ensure renderer is enabled and material color is set
                Renderer markerRenderer = referencePointMarker.GetComponent<Renderer>();
                if (markerRenderer != null)
                {
                    markerRenderer.enabled = true;
                    
                    // Ensure material exists and color is set correctly
                    if (markerRenderer.material == null)
                    {
                        // Recreate material if it was lost
                        Shader markerShader = Shader.Find("Unlit/Color");
                        if (markerShader == null)
                        {
                            markerShader = Shader.Find("Standard");
                        }
                        if (markerShader != null)
                        {
                            markerRenderer.material = new Material(markerShader);
                            if (showDebugInfo)
                            {
                                Debug.LogWarning("TrajectoryVisualizer: Reference marker material was null, recreated it!");
                            }
                        }
                    }
                    
                    if (markerRenderer.material != null)
                    {
                        markerRenderer.material.color = Color.yellow;
                        // Also set _Color property if shader supports it
                        if (markerRenderer.material.HasProperty("_Color"))
                        {
                            markerRenderer.material.SetColor("_Color", Color.yellow);
                        }
                    }
                }
                
                // Debug: Log marker status periodically (throttled to reduce spam)
                if (showDebugInfo && Time.frameCount % 300 == 0)
                {
                    bool rendererEnabled = markerRenderer != null ? markerRenderer.enabled : false;
                    Debug.Log($"TrajectoryVisualizer: Reference marker at {referencePoint}, active={referencePointMarker.activeSelf}, layer={referencePointMarker.layer}, renderer.enabled={rendererEnabled}");
                }
            }
            else
            {
                referencePointMarker.SetActive(alwaysVisible);
            }
        }
        
        // Vehicle position marker is attached to vehicle, so it updates automatically
    }
    
    Color GetColorForError(float lateralError)
    {
        float absError = Mathf.Abs(lateralError);
        
        if (absError < errorThreshold)
        {
            // CRITICAL: Explicitly return green to ensure it's not yellow
            // This overrides any Inspector setting that might be yellow
            // Use new Color(0, 1, 0, 1) to be absolutely sure it's green
            return new Color(0f, 1f, 0f, 1f); // Pure green: R=0, G=1, B=0, A=1
        }
        else if (absError < maxError)
        {
            // Interpolate between green and yellow
            float t = (absError - errorThreshold) / (maxError - errorThreshold);
            return Color.Lerp(new Color(0f, 1f, 0f, 1f), warningColor, t);
        }
        else
        {
            // Interpolate between warning and error
            float t = Mathf.Clamp01((absError - maxError) / maxError);
            return Color.Lerp(warningColor, errorColor, t);
        }
    }

    void UpdatePerceptionOverlay()
    {
        if (!showPerceptionLines)
        {
            return;
        }

        if (!perceptionValid || float.IsNaN(perceptionLookahead) || vehicleTransform == null)
        {
            if (perceptionLeftLine != null) perceptionLeftLine.enabled = false;
            if (perceptionRightLine != null) perceptionRightLine.enabled = false;
            if (perceptionCenterLine != null) perceptionCenterLine.enabled = false;
            return;
        }

        UpdatePerceptionLine(perceptionLeftLine, perceptionLeftX, perceptionLookahead);
        UpdatePerceptionLine(perceptionRightLine, perceptionRightX, perceptionLookahead);
        UpdatePerceptionLine(perceptionCenterLine, perceptionCenterX, perceptionLookahead);
    }

    LineRenderer CreatePerceptionLine(string name, Color color)
    {
        GameObject lineObj = new GameObject(name);
        lineObj.transform.SetParent(transform);
        LineRenderer line = lineObj.AddComponent<LineRenderer>();

        Shader lineShader = Shader.Find("Legacy Shaders/Particles/Alpha Blended Premultiply");
        if (lineShader == null)
        {
            lineShader = Shader.Find("Sprites/Default");
        }
        line.material = new Material(lineShader);
        line.material.color = color;

        line.useWorldSpace = true;
        line.startWidth = perceptionLineWidth;
        line.endWidth = perceptionLineWidth;
        line.positionCount = 0;
        line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        line.receiveShadows = false;

        return line;
    }

    void UpdatePerceptionLine(LineRenderer line, float x, float lookahead)
    {
        if (line == null || float.IsNaN(x) || float.IsNaN(lookahead) || vehicleTransform == null)
        {
            if (line != null) line.enabled = false;
            return;
        }

        float halfLen = perceptionLineLength * 0.5f;
        Vector3 startLocal = new Vector3(x, 0.05f, lookahead - halfLen);
        Vector3 endLocal = new Vector3(x, 0.05f, lookahead + halfLen);
        Vector3 startWorld = vehicleTransform.TransformPoint(startLocal);
        Vector3 endWorld = vehicleTransform.TransformPoint(endLocal);

        line.positionCount = 2;
        line.SetPosition(0, startWorld);
        line.SetPosition(1, endWorld);
        line.enabled = true;
    }
    
    void OnDrawGizmos()
    {
        // Draw trajectory in Scene view for debugging
        if (trajectoryPoints.Length > 1)
        {
            Gizmos.color = GetColorForError(currentLateralError);
            for (int i = 0; i < trajectoryPoints.Length - 1; i++)
            {
                Gizmos.DrawLine(trajectoryPoints[i], trajectoryPoints[i + 1]);
            }
        }
        
        // Draw reference point
        if (referencePoint != Vector3.zero)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawSphere(referencePoint, referencePointSize);
        }
    }
}

[Serializable]
public class TrajectoryData
{
    public float[][] trajectory_points;  // Array of [x, y, heading] points
    public float[] reference_point;      // [x, y, heading, velocity]
    public float lateral_error;
    public float timestamp;
    public float perception_left_lane_x;
    public float perception_right_lane_x;
    public float perception_center_x;
    public float perception_lookahead_m;
    public bool perception_valid;
}

