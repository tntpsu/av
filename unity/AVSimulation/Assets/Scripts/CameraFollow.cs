using UnityEngine;

/// <summary>
/// Third-person camera that follows the car smoothly and looks at it
/// </summary>
public class CameraFollow : MonoBehaviour
{
    [Header("Target")]
    [Tooltip("The car to follow (auto-detected if not set)")]
    public Transform target;
    
    [Header("Position Settings")]
    [Tooltip("Offset from car position (relative to car's forward direction)")]
    public Vector3 positionOffset = new Vector3(0, 5, -10);
    
    [Header("Look Settings")]
    [Tooltip("Offset from car position where camera looks at (relative to car)")]
    public Vector3 lookAtOffset = new Vector3(0, 0, 5);
    
    [Header("Follow Mode")]
    [Tooltip("If true, camera follows car's rotation. If false, camera stays at fixed angle.")]
    public bool followRotation = true;
    
    [Header("Smoothing")]
    [Tooltip("Position smoothing speed (higher = smoother, but may lag)")]
    public float positionSmoothSpeed = 5f;
    
    [Tooltip("Rotation smoothing speed (higher = smoother)")]
    public float rotationSmoothSpeed = 5f;
    
    private void Start()
    {
        // Auto-detect car if not assigned
        if (target == null)
        {
            CarController car = FindObjectOfType<CarController>();
            if (car != null)
            {
                target = car.transform;
                Debug.Log($"CameraFollow: Auto-detected car target: {target.name}");
            }
            else
            {
                Debug.LogWarning("CameraFollow: No car found! Please assign target manually.");
            }
        }
    }
    
    private void LateUpdate()
    {
        if (target == null) return;
        
        // Calculate desired position (relative to car's rotation)
        Vector3 desiredPosition;
        if (followRotation)
        {
            // Position offset rotates with car (follows behind car)
            desiredPosition = target.position + target.TransformDirection(positionOffset);
        }
        else
        {
            // Fixed world-space offset (camera stays at fixed angle)
            desiredPosition = target.position + positionOffset;
        }
        
        // Smoothly move camera to desired position
        if (positionSmoothSpeed > 0)
        {
            transform.position = Vector3.Lerp(transform.position, desiredPosition, positionSmoothSpeed * Time.deltaTime);
        }
        else
        {
            transform.position = desiredPosition;
        }
        
        // Calculate look-at point (where camera should look)
        Vector3 lookAtPoint = target.position + target.TransformDirection(lookAtOffset);
        
        // Smoothly rotate camera to look at car
        Vector3 directionToTarget = lookAtPoint - transform.position;
        if (directionToTarget != Vector3.zero)
        {
            Quaternion desiredRotation = Quaternion.LookRotation(directionToTarget);
            
            if (rotationSmoothSpeed > 0)
            {
                transform.rotation = Quaternion.Slerp(transform.rotation, desiredRotation, rotationSmoothSpeed * Time.deltaTime);
            }
            else
            {
                transform.rotation = desiredRotation;
            }
        }
    }
}

