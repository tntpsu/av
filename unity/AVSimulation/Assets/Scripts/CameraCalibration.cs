using UnityEngine;

/// <summary>
/// Reports actual camera parameters for coordinate conversion validation.
/// Add this script to the AVCamera GameObject to log camera parameters.
/// </summary>
public class CameraCalibration : MonoBehaviour
{
    [Header("Debug")]
    public bool logOnStart = true;
    public bool logContinuously = false;
    public float logInterval = 1.0f; // seconds
    
    private Camera targetCamera;
    private float lastLogTime = 0f;
    
    void Start()
    {
        targetCamera = GetComponent<Camera>();
        if (targetCamera == null)
        {
            Debug.LogError("CameraCalibration: No Camera component found!");
            enabled = false;
            return;
        }
        
        if (logOnStart)
        {
            LogCameraParameters();
        }
    }
    
    void Update()
    {
        if (logContinuously && Time.time - lastLogTime >= logInterval)
        {
            LogCameraParameters();
            lastLogTime = Time.time;
        }
    }
    
    void LogCameraParameters()
    {
        if (targetCamera == null) return;
        
        // Get camera transform
        Transform camTransform = targetCamera.transform;
        Vector3 worldPosition = camTransform.position;
        Vector3 localPosition = camTransform.localPosition;
        
        // Get camera parameters
        float fov = targetCamera.fieldOfView;
        float aspect = targetCamera.aspect;
        float nearClip = targetCamera.nearClipPlane;
        float farClip = targetCamera.farClipPlane;
        
        // Calculate horizontal FOV from vertical FOV
        // Unity uses vertical FOV by default
        float verticalFOV = fov;
        float horizontalFOV = 2.0f * Mathf.Atan(Mathf.Tan(verticalFOV * Mathf.Deg2Rad / 2.0f) * aspect) * Mathf.Rad2Deg;
        
        // Try to estimate camera height above ground
        // Cast ray downward to find ground
        float estimatedHeightAboveGround = worldPosition.y;
        RaycastHit hit;
        if (Physics.Raycast(worldPosition, Vector3.down, out hit, 100f))
        {
            estimatedHeightAboveGround = hit.distance;
        }
        
        Debug.Log("=" + new string('=', 79));
        Debug.Log("CAMERA CALIBRATION PARAMETERS");
        Debug.Log("=" + new string('=', 79));
        Debug.Log($"Camera Name: {targetCamera.name}");
        Debug.Log($"Camera Enabled: {targetCamera.enabled}");
        Debug.Log("");
        Debug.Log("Camera Transform:");
        Debug.Log($"  World Position: {worldPosition}");
        Debug.Log($"  Local Position: {localPosition}");
        Debug.Log($"  Rotation: {camTransform.eulerAngles}");
        Debug.Log("");
        Debug.Log("Camera Parameters:");
        Debug.Log($"  Field of View (Vertical): {fov}°");
        Debug.Log($"  Field of View (Horizontal): {horizontalFOV:F2}°");
        Debug.Log($"  Aspect Ratio: {aspect:F3} ({targetCamera.pixelWidth}x{targetCamera.pixelHeight})");
        Debug.Log($"  Near Clip: {nearClip}m");
        Debug.Log($"  Far Clip: {farClip}m");
        Debug.Log("");
        Debug.Log("Camera Height:");
        Debug.Log($"  World Y Position: {worldPosition.y:F3}m");
        Debug.Log($"  Estimated Height Above Ground: {estimatedHeightAboveGround:F3}m");
        Debug.Log($"  Local Y Position (relative to parent): {localPosition.y:F3}m");
        Debug.Log("");
        Debug.Log("Parent Transform:");
        if (camTransform.parent != null)
        {
            Debug.Log($"  Parent Name: {camTransform.parent.name}");
            Debug.Log($"  Parent Position: {camTransform.parent.position}");
        }
        else
        {
            Debug.Log("  No parent (static camera)");
        }
        Debug.Log("=" + new string('=', 79));
    }
    
    void OnDrawGizmos()
    {
        // Draw camera frustum for visualization
        if (targetCamera == null) return;
        
        Camera cam = targetCamera;
        float fov = cam.fieldOfView;
        float aspect = cam.aspect;
        float near = cam.nearClipPlane;
        float far = cam.farClipPlane;
        
        // Draw near and far planes
        Vector3[] nearCorners = GetFrustumCorners(cam, near);
        Vector3[] farCorners = GetFrustumCorners(cam, far);
        
        Gizmos.color = Color.yellow;
        DrawFrustum(nearCorners, farCorners);
    }
    
    Vector3[] GetFrustumCorners(Camera cam, float distance)
    {
        Vector3[] corners = new Vector3[4];
        float fov = cam.fieldOfView * Mathf.Deg2Rad;
        float aspect = cam.aspect;
        
        float halfHeight = distance * Mathf.Tan(fov / 2f);
        float halfWidth = halfHeight * aspect;
        
        corners[0] = cam.transform.position + cam.transform.forward * distance
            + cam.transform.right * -halfWidth + cam.transform.up * halfHeight;
        corners[1] = cam.transform.position + cam.transform.forward * distance
            + cam.transform.right * halfWidth + cam.transform.up * halfHeight;
        corners[2] = cam.transform.position + cam.transform.forward * distance
            + cam.transform.right * halfWidth + cam.transform.up * -halfHeight;
        corners[3] = cam.transform.position + cam.transform.forward * distance
            + cam.transform.right * -halfWidth + cam.transform.up * -halfHeight;
        
        return corners;
    }
    
    void DrawFrustum(Vector3[] near, Vector3[] far)
    {
        // Near plane
        Gizmos.DrawLine(near[0], near[1]);
        Gizmos.DrawLine(near[1], near[2]);
        Gizmos.DrawLine(near[2], near[3]);
        Gizmos.DrawLine(near[3], near[0]);
        
        // Far plane
        Gizmos.DrawLine(far[0], far[1]);
        Gizmos.DrawLine(far[1], far[2]);
        Gizmos.DrawLine(far[2], far[3]);
        Gizmos.DrawLine(far[3], far[0]);
        
        // Connecting lines
        for (int i = 0; i < 4; i++)
        {
            Gizmos.DrawLine(near[i], far[i]);
        }
    }
}

