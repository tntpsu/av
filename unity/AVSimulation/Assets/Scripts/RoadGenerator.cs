using UnityEngine;
using System.Collections.Generic;
using System.Linq;
#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Procedural oval road generator for AV simulation.
/// Creates a racetrack-style oval with straight sections and curved turns.
/// </summary>
[ExecuteAlways]
public class RoadGenerator : MonoBehaviour
{
    [Header("Oval Dimensions")]
    [Tooltip("Length of straight sections (top and bottom)")]
    public float straightLength = 200f;  // Increased from 100f for proper oval shape
    
    [Tooltip("Radius of curved sections (left and right turns)")]
    public float turnRadius = 50f;
    
    [Tooltip("Width of the road")]
    public float roadWidth = 7f;  // Changed from 10f to 7f (3.5m per lane - standard)
    
    [Tooltip("Width of lane markings")]
    public float laneLineWidth = 0.3f;
    
    [Header("Lane Markings")]
    [Tooltip("Material for white lane lines")]
    public Material whiteLaneMaterial;
    
    [Tooltip("Material for yellow lane lines")]
    public Material yellowLaneMaterial;
    
    [Header("Dashed Line Settings")]
    [Tooltip("Length of each dash segment (meters)")]
    public float dashLength = 3f;
    
    [Tooltip("Length of gap between dashes (meters)")]
    public float gapLength = 0.75f;  // Reduced from 1.5f to 0.75f (half again) to make detection even easier
    
    [Header("Generation Settings")]
    [Tooltip("Number of segments per curve (higher = smoother)")]
    public int curveSegments = 32;
    
    [Tooltip("Generate road on Start()")]
    public bool generateOnStart = true;
    
    [Tooltip("Replace existing road GameObject")]
    public bool replaceExistingRoad = false;
    
    [Tooltip("Road surface material")]
    public Material roadMaterial;

    [Header("Placement")]
    [Tooltip("Offset the track from the origin")]
    public Vector3 trackOffset = Vector3.zero;
    
    private GameObject roadMeshObject;
    private GameObject leftLaneLineObject;
    private GameObject rightLaneLineObject;
    private List<GameObject> generatedObjects = new List<GameObject>();
    private bool hasGenerated = false;
    
    void OnEnable()
    {
        // Generate in edit mode if enabled
        // Only generate once to prevent duplicates
        if (generateOnStart && !hasGenerated)
        {
            #if UNITY_EDITOR
            if (!Application.isPlaying)
            {
                // Small delay to ensure scene is ready
                UnityEditor.EditorApplication.delayCall += () => {
                    if (this != null && generateOnStart && !hasGenerated)
                    {
                        GenerateOvalRoad();
                        hasGenerated = true;
                    }
                };
            }
            else
            #endif
            {
                GenerateOvalRoad();
                hasGenerated = true;
            }
        }
    }
    
    /// <summary>
    /// Public method to force regeneration (can be called from Inspector button)
    /// </summary>
    [ContextMenu("Regenerate Road")]
    public void RegenerateRoad()
    {
        hasGenerated = false;
        GenerateOvalRoad();
    }
    
    void Start()
    {
        if (generateOnStart && !hasGenerated)
        {
            GenerateOvalRoad();
            hasGenerated = true;
        }
    }
    
    /// <summary>
    /// Generate the complete oval road with lane markings.
    /// </summary>
    public void GenerateOvalRoad()
    {
        // Clean up existing generated objects
        Cleanup();
        
        // Remove existing road if requested
        if (replaceExistingRoad)
        {
            RemoveExistingRoad();
        }
        
        // Create road mesh
        CreateRoadMesh();
        
        // Create lane lines
        CreateLaneLines();
        
        hasGenerated = true;
        
        #if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            UnityEditor.EditorUtility.SetDirty(this);
        }
        #endif
        
        Debug.Log($"Oval road generated: {straightLength * 2 + Mathf.PI * turnRadius * 2:F1} units total length");
    }
    
    /// <summary>
    /// Remove existing road and lane line objects from scene.
    /// </summary>
    void RemoveExistingRoad()
    {
        GameObject existingRoad = GameObject.Find("Road");
        if (existingRoad != null)
        {
            DestroyImmediate(existingRoad);
        }
        
        GameObject leftLine = GameObject.Find("lanelinelft");
        if (leftLine != null)
        {
            DestroyImmediate(leftLine);
        }
        
        GameObject rightLine = GameObject.Find("lanelinert");
        if (rightLine != null)
        {
            DestroyImmediate(rightLine);
        }
    }
    
    /// <summary>
    /// Create the road mesh for the oval track.
    /// </summary>
    void CreateRoadMesh()
    {
        roadMeshObject = new GameObject("OvalRoad");
        roadMeshObject.transform.SetParent(transform);
        
        MeshFilter meshFilter = roadMeshObject.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = roadMeshObject.AddComponent<MeshRenderer>();
        
        Mesh roadMesh = new Mesh();
        roadMesh.name = "OvalRoadMesh";
        
        // Generate vertices and triangles for oval road
        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();
        List<Vector2> uvs = new List<Vector2>();
        
        // Calculate total path length for UV mapping
        // Curves are semicircles with radius = turnRadius
        float curveArcLength = Mathf.PI * turnRadius;
        float straightArcLength = straightLength;
        float totalPathLength = (straightArcLength * 2) + (curveArcLength * 2);
        float currentU = 0f;
        
        // Generate vertices along the oval path
        // Sample based on arc length: allocate segments proportionally to each section's length
        int straightSegments = Mathf.Max(10, curveSegments / 2);
        int totalSegments = (curveSegments * 2) + (straightSegments * 2);
        
        // Calculate how many segments each section should get based on arc length
        float totalArcLength = (straightArcLength * 2) + (curveArcLength * 2);
        int segmentsPerStraight = Mathf.RoundToInt((straightArcLength / totalArcLength) * totalSegments);
        int segmentsPerCurve = Mathf.RoundToInt((curveArcLength / totalArcLength) * totalSegments);
        
        // Ensure we have at least some segments for each section
        segmentsPerStraight = Mathf.Max(5, segmentsPerStraight);
        segmentsPerCurve = Mathf.Max(curveSegments, segmentsPerCurve);
        
        List<float> tValues = new List<float>();
        
        // Top straight: t = 0 to 0.25
        for (int i = 0; i <= segmentsPerStraight; i++)
        {
            float localT = (float)i / segmentsPerStraight;
            tValues.Add(localT * 0.25f);
        }
        
        // Right curve: t = 0.25 to 0.5
        for (int i = 1; i <= segmentsPerCurve; i++)
        {
            float localT = (float)i / segmentsPerCurve;
            tValues.Add(0.25f + localT * 0.25f);
        }
        
        // Bottom straight: t = 0.5 to 0.75
        for (int i = 1; i <= segmentsPerStraight; i++)
        {
            float localT = (float)i / segmentsPerStraight;
            tValues.Add(0.5f + localT * 0.25f);
        }
        
        // Left curve: t = 0.75 to 1.0
        // Don't include t=1.0 since it's the same as t=0.0 (already in the list)
        // The loop will be closed by connecting the last segment to the first
        for (int i = 1; i < segmentsPerCurve; i++)
        {
            float localT = (float)i / segmentsPerCurve;
            tValues.Add(0.75f + localT * 0.25f);
        }
        // Add the last point of left curve (just before t=1.0)
        tValues.Add(0.9999f); // Very close to 1.0 but not exactly, to avoid duplicate with t=0.0
        
        // Generate vertices for each t value
        for (int i = 0; i < tValues.Count; i++)
        {
            float t = Mathf.Clamp01(tValues[i]);
            Vector3 centerPoint = GetOvalCenterPoint(t);
            Vector3 direction = GetOvalDirection(t);
            Vector3 right = Vector3.Cross(Vector3.up, direction).normalized;
            
            // Left and right edges of road
            Vector3 leftEdge = centerPoint - right * (roadWidth * 0.5f);
            Vector3 rightEdge = centerPoint + right * (roadWidth * 0.5f);
            
            leftEdge.y = 0f;
            rightEdge.y = 0f;
            
            vertices.Add(leftEdge);
            vertices.Add(rightEdge);
            
            // UV mapping
            uvs.Add(new Vector2(0f, currentU));
            uvs.Add(new Vector2(1f, currentU));
            
            // Calculate U increment based on segment length
            if (i < tValues.Count - 1)
            {
                Vector3 nextCenter = GetOvalCenterPoint(tValues[i + 1]);
                float segmentDist = Vector3.Distance(centerPoint, nextCenter);
                currentU += segmentDist / totalPathLength;
            }
        }
        
        totalSegments = tValues.Count;
        
        // Create triangles (quads)
        for (int i = 0; i < totalSegments; i++)
        {
            int baseIdx = i * 2;
            int nextIdx = ((i + 1) % totalSegments) * 2; // Wrap around for closing the loop
            
            // First triangle
            triangles.Add(baseIdx);
            triangles.Add(nextIdx);
            triangles.Add(baseIdx + 1);
            
            // Second triangle
            triangles.Add(baseIdx + 1);
            triangles.Add(nextIdx);
            triangles.Add(nextIdx + 1);
        }
        
        roadMesh.vertices = vertices.ToArray();
        roadMesh.triangles = triangles.ToArray();
        roadMesh.uv = uvs.ToArray();
        roadMesh.RecalculateNormals();
        roadMesh.RecalculateBounds();
        
        meshFilter.mesh = roadMesh;
        
        // Add mesh collider
        MeshCollider meshCollider = roadMeshObject.AddComponent<MeshCollider>();
        meshCollider.sharedMesh = roadMesh;
        
        // Apply material - create default gray material if none provided
        if (roadMaterial != null)
        {
            meshRenderer.material = roadMaterial;
        }
        else
        {
            // Create default gray road material
            Material defaultRoadMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            defaultRoadMaterial.color = new Color(0.3f, 0.3f, 0.3f, 1f); // Dark gray
            meshRenderer.material = defaultRoadMaterial;
        }
        
        generatedObjects.Add(roadMeshObject);
    }
    
    /// <summary>
    /// Create lane lines that follow the oval path.
    /// </summary>
    void CreateLaneLines()
    {
        // Left lane line (yellow - left edge)
        leftLaneLineObject = CreateDashedLaneLine("LeftLaneLine", -roadWidth * 0.5f, yellowLaneMaterial);
        generatedObjects.Add(leftLaneLineObject);
        
        // Right lane line (white - right edge)
        rightLaneLineObject = CreateLaneLine("RightLaneLine", roadWidth * 0.5f, whiteLaneMaterial);
        generatedObjects.Add(rightLaneLineObject);
    }
    
    /// <summary>
    /// Create a single lane line along the oval path.
    /// </summary>
    GameObject CreateLaneLine(string name, float offsetFromCenter, Material material)
    {
        GameObject laneLine = new GameObject(name);
        laneLine.transform.SetParent(transform);
        
        // Create line segments along the path
        int straightSegments = Mathf.Max(10, curveSegments / 2);
        int totalSegments = (curveSegments * 2) + (straightSegments * 2);
        float segmentLength = 1f / totalSegments;
        
        List<Vector3> linePoints = new List<Vector3>();
        
        for (int i = 0; i <= totalSegments; i++)
        {
            float t = Mathf.Clamp01(i * segmentLength);
            Vector3 centerPoint = GetOvalCenterPoint(t);
            Vector3 direction = GetOvalDirection(t);
            Vector3 right = Vector3.Cross(Vector3.up, direction).normalized;
            
            Vector3 lanePoint = centerPoint + right * offsetFromCenter;
            lanePoint.y = 0.05f; // Slightly above road surface
            linePoints.Add(lanePoint);
        }
        
        // Create mesh for lane line (thin box along path)
        MeshFilter meshFilter = laneLine.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = laneLine.AddComponent<MeshRenderer>();
        
        Mesh lineMesh = CreateLineMesh(linePoints, laneLineWidth);
        meshFilter.mesh = lineMesh;
        
        if (material != null)
        {
            meshRenderer.material = material;
        }
        else
        {
            // Create default white material if none provided
            Material defaultMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            defaultMaterial.color = Color.white;
            meshRenderer.material = defaultMaterial;
        }
        
        return laneLine;
    }
    
    /// <summary>
    /// Create a dashed lane line along the oval path.
    /// </summary>
    GameObject CreateDashedLaneLine(string name, float offsetFromCenter, Material material)
    {
        GameObject dashedLineParent = new GameObject(name);
        dashedLineParent.transform.SetParent(transform);
        
        // Calculate total path length
        float totalPathLength = (straightLength * 2) + (Mathf.PI * turnRadius * 2);
        
        // Create line segments along the path with gaps
        int straightSegments = Mathf.Max(10, curveSegments / 2);
        int totalSegments = (curveSegments * 2) + (straightSegments * 2);
        float segmentLength = 1f / totalSegments;
        
        // Sample points along the path at high resolution
        List<Vector3> pathPoints = new List<Vector3>();
        for (int i = 0; i <= totalSegments * 10; i++)  // 10x resolution for smooth curves
        {
            float t = Mathf.Clamp01((float)i / (totalSegments * 10));
            Vector3 centerPoint = GetOvalCenterPoint(t);
            Vector3 direction = GetOvalDirection(t);
            Vector3 right = Vector3.Cross(Vector3.up, direction).normalized;
            
            Vector3 lanePoint = centerPoint + right * offsetFromCenter;
            lanePoint.y = 0.05f;
            pathPoints.Add(lanePoint);
        }
        
        // Calculate cumulative distances along path
        List<float> cumulativeDistances = new List<float>();
        cumulativeDistances.Add(0f);
        for (int i = 1; i < pathPoints.Count; i++)
        {
            float dist = Vector3.Distance(pathPoints[i - 1], pathPoints[i]);
            cumulativeDistances.Add(cumulativeDistances[i - 1] + dist);
        }
        
        // Create dashed segments
        float currentDistance = 0f;
        bool isDash = true;  // Start with a dash
        int dashIndex = 0;
        
        while (currentDistance < cumulativeDistances[cumulativeDistances.Count - 1])
        {
            float segmentDistance = isDash ? dashLength : gapLength;
            float endDistance = currentDistance + segmentDistance;
            
            if (isDash)
            {
                // Create a dash segment
                List<Vector3> dashPoints = new List<Vector3>();
                
                // Find points within this dash segment
                for (int i = 0; i < pathPoints.Count; i++)
                {
                    if (cumulativeDistances[i] >= currentDistance && cumulativeDistances[i] <= endDistance)
                    {
                        dashPoints.Add(pathPoints[i]);
                    }
                }
                
                // Interpolate start and end points if needed
                if (dashPoints.Count > 0)
                {
                    // Add start point if needed
                    if (cumulativeDistances[0] < currentDistance)
                    {
                        for (int i = 0; i < pathPoints.Count - 1; i++)
                        {
                            if (cumulativeDistances[i] <= currentDistance && cumulativeDistances[i + 1] > currentDistance)
                            {
                                float t = (currentDistance - cumulativeDistances[i]) / (cumulativeDistances[i + 1] - cumulativeDistances[i]);
                                Vector3 interpolated = Vector3.Lerp(pathPoints[i], pathPoints[i + 1], t);
                                dashPoints.Insert(0, interpolated);
                                break;
                            }
                        }
                    }
                    
                    // Add end point if needed
                    if (cumulativeDistances[cumulativeDistances.Count - 1] > endDistance)
                    {
                        for (int i = 0; i < pathPoints.Count - 1; i++)
                        {
                            if (cumulativeDistances[i] <= endDistance && cumulativeDistances[i + 1] > endDistance)
                            {
                                float t = (endDistance - cumulativeDistances[i]) / (cumulativeDistances[i + 1] - cumulativeDistances[i]);
                                Vector3 interpolated = Vector3.Lerp(pathPoints[i], pathPoints[i + 1], t);
                                dashPoints.Add(interpolated);
                                break;
                            }
                        }
                    }
                    
                    if (dashPoints.Count >= 2)
                    {
                        // Create mesh for this dash
                        GameObject dashSegment = new GameObject($"Dash_{dashIndex}");
                        dashSegment.transform.SetParent(dashedLineParent.transform);
                        
                        MeshFilter meshFilter = dashSegment.AddComponent<MeshFilter>();
                        MeshRenderer meshRenderer = dashSegment.AddComponent<MeshRenderer>();
                        
                        Mesh dashMesh = CreateLineMesh(dashPoints, laneLineWidth);
                        meshFilter.mesh = dashMesh;
                        
                        if (material != null)
                        {
                            meshRenderer.material = material;
                        }
                        else
                        {
                            // Create default white material if none provided
                            Material defaultMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
                            defaultMaterial.color = Color.white;
                            meshRenderer.material = defaultMaterial;
                        }
                        
                        dashIndex++;
                    }
                }
            }
            
            currentDistance = endDistance;
            isDash = !isDash;  // Alternate between dash and gap
        }
        
        return dashedLineParent;
    }
    
    /// <summary>
    /// Create a mesh for a line following a path of points.
    /// </summary>
    Mesh CreateLineMesh(List<Vector3> points, float width)
    {
        Mesh mesh = new Mesh();
        mesh.name = "LaneLineMesh";
        
        if (points.Count < 2)
        {
            return mesh;
        }
        
        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();
        List<Vector2> uvs = new List<Vector2>();
        
        for (int i = 0; i < points.Count; i++)
        {
            Vector3 point = points[i];
            Vector3 direction;
            
            if (i == 0)
            {
                direction = (points[i + 1] - point).normalized;
            }
            else if (i == points.Count - 1)
            {
                direction = (point - points[i - 1]).normalized;
            }
            else
            {
                direction = ((points[i + 1] - point) + (point - points[i - 1])).normalized;
            }
            
            Vector3 right = Vector3.Cross(Vector3.up, direction).normalized;
            Vector3 offset = right * (width * 0.5f);
            
            vertices.Add(point - offset);
            vertices.Add(point + offset);
            
            float u = (float)i / (points.Count - 1);
            uvs.Add(new Vector2(0f, u));
            uvs.Add(new Vector2(1f, u));
        }
        
        // Create triangles
        for (int i = 0; i < points.Count - 1; i++)
        {
            int baseIdx = i * 2;
            
            triangles.Add(baseIdx);
            triangles.Add(baseIdx + 2);
            triangles.Add(baseIdx + 1);
            
            triangles.Add(baseIdx + 1);
            triangles.Add(baseIdx + 2);
            triangles.Add(baseIdx + 3);
        }
        
        mesh.vertices = vertices.ToArray();
        mesh.triangles = triangles.ToArray();
        mesh.uv = uvs.ToArray();
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        
        return mesh;
    }
    
    /// <summary>
    /// Get center point on oval path at parameter t (0 to 1).
    /// Public so GroundTruthReporter can access it for dynamic ground truth.
    /// </summary>
    public Vector3 GetOvalCenterPoint(float t)
    {
        // Normalize t to 0-1 range
        t = Mathf.Repeat(t, 1f);
        
        // Calculate which segment we're on
        // Segment 0: Top straight (t: 0 to 0.25)
        // Segment 1: Right curve (t: 0.25 to 0.5)
        // Segment 2: Bottom straight (t: 0.5 to 0.75)
        // Segment 3: Left curve (t: 0.75 to 1.0)
        
        float segmentLength = 0.25f;
        float localT;
        Vector3 position;
        
        if (t < 0.25f)
        {
            // Top straight section
            // Connects (-straightLength/2, turnRadius) to (straightLength/2, turnRadius)
            localT = t / segmentLength;
            float z = turnRadius;
            float x = Mathf.Lerp(-straightLength * 0.5f, straightLength * 0.5f, localT);
            position = new Vector3(x, 0f, z);
        }
        else if (t < 0.5f)
        {
            // Right curve
            // Semicircle centered at (straightLength/2, 0) with radius = turnRadius
            // Connects top straight end (straightLength/2, turnRadius) to bottom straight start (straightLength/2, -turnRadius)
            localT = (t - 0.25f) / segmentLength;
            float angle = Mathf.Lerp(90f, -90f, localT) * Mathf.Deg2Rad;

            float cx = straightLength * 0.5f;
            float x = cx + turnRadius * Mathf.Cos(angle);
            float z = 0f + turnRadius * Mathf.Sin(angle);

            position = new Vector3(x, 0f, z);
        }
        else if (t < 0.75f)
        {
            // Bottom straight section
            // Connects (straightLength/2, -turnRadius) to (-straightLength/2, -turnRadius)
            localT = (t - 0.5f) / segmentLength;
            float z = -turnRadius;
            float x = Mathf.Lerp(straightLength * 0.5f, -straightLength * 0.5f, localT);
            position = new Vector3(x, 0f, z);
        }
        else
        {
            // Left curve
            // Semicircle centered at (-straightLength/2, 0) with radius = turnRadius
            // Connects bottom straight end (-straightLength/2, -turnRadius) to top straight start (-straightLength/2, turnRadius)
            localT = (t - 0.75f) / segmentLength;
            float angle = Mathf.Lerp(-90f, 90f, localT) * Mathf.Deg2Rad;

            float cx = -straightLength * 0.5f;
            float x = cx - turnRadius * Mathf.Cos(angle);
            float z = 0f + turnRadius * Mathf.Sin(angle);

            position = new Vector3(x, 0f, z);
        }
        
        return position + trackOffset;
    }
    
    /// <summary>
    /// Get direction vector along oval path at parameter t.
    /// Public so GroundTruthReporter can access it for dynamic ground truth.
    /// </summary>
    public Vector3 GetOvalDirection(float t)
    {
        t = Mathf.Repeat(t, 1f);

        float delta = 0.0025f;

        // CRITICAL FIX: Handle wrap-around at boundaries
        // When t is near 0 or 1, we need to avoid wrapping around to the other end
        // Instead, clamp tPrev and tNext to stay within valid range
        float tPrev, tNext;
        
        if (t < delta)
        {
            // Near start (t < 0.0025): don't wrap backward
            tPrev = 0f;
            tNext = t + delta;
        }
        else if (t > (1f - delta))
        {
            // Near end (t > 0.9975): don't wrap forward
            // Use 0.999 instead of 1.0 to avoid wrap-around in GetOvalCenterPoint
            tPrev = t - delta;
            tNext = 0.999f;  // Just before wrap point (1.0 wraps to 0.0 in GetOvalCenterPoint)
        }
        else
        {
            // Middle range: safe to use normal calculation
            tPrev = t - delta;
            tNext = t + delta;
        }

        Vector3 prev = GetOvalCenterPoint(tPrev);
        Vector3 next = GetOvalCenterPoint(tNext);

        Vector3 dir = (next - prev);
        if (dir.sqrMagnitude < 1e-8f) return Vector3.right;

        return dir.normalized;
    }
    
    /// <summary>
    /// Clean up generated objects.
    /// </summary>
    void Cleanup()
    {
        // Clean up tracked objects
        foreach (GameObject obj in generatedObjects)
        {
            if (obj != null)
            {
                DestroyImmediate(obj);
            }
        }
        generatedObjects.Clear();
        
        // Also find and destroy by name to catch any that weren't tracked
        // This handles cases where objects weren't properly tracked
        Transform[] allChildren = GetComponentsInChildren<Transform>(true);
        foreach (Transform child in allChildren)
        {
            // Check if child is null or destroyed before accessing properties
            if (child == null || child.gameObject == null)
                continue;
                
            if (child != transform && (child.name == "OvalRoad" || child.name == "LeftLaneLine" || child.name == "RightLaneLine"))
            {
                DestroyImmediate(child.gameObject);
            }
        }
    }
    
    void OnDestroy()
    {
        Cleanup();
    }
}

