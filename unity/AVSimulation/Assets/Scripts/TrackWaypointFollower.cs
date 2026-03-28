using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Kinematic waypoint follower used by the lead vehicle.
///
/// Movement model: pure kinematic (no WheelCollider / physics).
/// Each FixedUpdate the object is translated along the track centerline
/// at the speed returned by SpeedProfiler, wrapping at track end if loop=true.
///
/// The transform is placed ON the track surface (Y = waypoint.y) so grade
/// changes are handled correctly — no separate grade FF needed.
///
/// Lane offset: laneOffsetM shifts the vehicle perpendicular to the road
/// tangent at each frame. Positive = right-hand side of travel direction
/// (Vector3.Cross(Vector3.up, forward)). Set to roadWidth/4 for right-lane.
///
/// Movement gate: isMovementEnabled must be true before the follower advances.
/// AVBridge sets this to true when the first Python control command arrives,
/// so the lead vehicle starts at the same moment as the ego car.
/// </summary>
[RequireComponent(typeof(SpeedProfiler))]
public class TrackWaypointFollower : MonoBehaviour
{
    // ── Configuration ─────────────────────────────────────────────────────────
    [Tooltip("Waypoints sampled from the track centerline (world coords).")]
    public List<Vector3> waypoints = new List<Vector3>();
    public bool loop = true;

    [Tooltip("Lateral offset from centerline (m). Positive = right lane. " +
             "Use roadWidth/4 for the right lane of a 2-lane road.")]
    public float laneOffsetM = 0.0f;

    [Tooltip("Travel direction along the sampled path: +1 = same direction, -1 = opposite direction.")]
    public int travelDirectionSign = 1;

    [Tooltip("When false the follower does not advance. Set to true by AVBridge " +
             "when the first Python control command is received so the lead vehicle " +
             "starts at the same time as the ego car.")]
    public bool isMovementEnabled = false;

    // ── Runtime ───────────────────────────────────────────────────────────────
    private SpeedProfiler _profiler;
    private int    _waypointIndex  = 0;
    private float  _segmentProgress = 0.0f;  // fraction [0,1] between current and next wp
    private List<float> _cumulativeDistances = new List<float>();
    private float _totalPathLength = 0.0f;

    /// <summary>Current arc-distance from the start waypoint (m).</summary>
    public float ArcDistance { get; private set; } = 0.0f;

    /// <summary>Current speed (m/s) as returned by SpeedProfiler.</summary>
    public float CurrentSpeed { get; private set; } = 0.0f;

    /// <summary>Current velocity vector (world space, m/s).</summary>
    public Vector3 Velocity { get; private set; } = Vector3.zero;

    void Awake()
    {
        _profiler = GetComponent<SpeedProfiler>();
    }

    public void SetTravelDirection(string travelDirection)
    {
        travelDirectionSign = string.Equals(travelDirection, "opposite", System.StringComparison.OrdinalIgnoreCase) ? -1 : 1;
    }

    /// <summary>Teleport to the waypoint nearest to the given arc distance.</summary>
    public void SetArcDistance(float targetDistanceM, List<float> cumulativeDistances)
    {
        if (waypoints == null || waypoints.Count < 2 || cumulativeDistances == null)
            return;

        _cumulativeDistances = cumulativeDistances;
        _totalPathLength = cumulativeDistances.Count > 0
            ? cumulativeDistances[cumulativeDistances.Count - 1]
            : 0.0f;

        SetArcDistanceInternal(targetDistanceM);
    }

    private void SetArcDistanceInternal(float targetDistanceM)
    {
        float normalizedDistance = NormalizeArcDistance(targetDistanceM);

        if (waypoints == null || waypoints.Count < 2 || _cumulativeDistances == null || _cumulativeDistances.Count < 2)
        {
            ArcDistance = normalizedDistance;
            return;
        }

        // Find segment containing targetDistanceM
        int idx = 0;
        for (int i = 0; i < _cumulativeDistances.Count - 1; i++)
        {
            if (_cumulativeDistances[i + 1] >= normalizedDistance)
            {
                idx = i;
                break;
            }
            idx = i;
        }
        idx = Mathf.Clamp(idx, 0, waypoints.Count - 2);
        _waypointIndex = idx;

        float segLen = Vector3.Distance(waypoints[idx], waypoints[idx + 1]);
        float remain  = normalizedDistance - _cumulativeDistances[idx];
        _segmentProgress = segLen > 0.001f
            ? Mathf.Clamp01(remain / segLen)
            : 0f;

        UpdateTransform();
        ArcDistance = normalizedDistance;
    }

    void FixedUpdate()
    {
        // Hold until AVBridge signals that the drive has started.
        if (!isMovementEnabled)
            return;

        if (waypoints == null || waypoints.Count < 2)
            return;

        float dt = Time.fixedDeltaTime;
        float speed = _profiler != null ? _profiler.GetSpeed(dt) : 0.0f;
        CurrentSpeed = speed;
        float signedDistance = speed * dt * travelDirectionSign;

        Advance(signedDistance, dt);
    }

    private void Advance(float signedDistance, float dt)
    {
        if (waypoints.Count < 2)
            return;

        Vector3 posBefore = transform.position;
        float targetArcDistance = ArcDistance + signedDistance;
        if (!loop)
        {
            if (targetArcDistance <= 0.0f)
            {
                targetArcDistance = 0.0f;
                CurrentSpeed = 0.0f;
            }
            else if (targetArcDistance >= _totalPathLength)
            {
                targetArcDistance = _totalPathLength;
                CurrentSpeed = 0.0f;
            }
        }

        SetArcDistanceInternal(targetArcDistance);
        Vector3 posAfter = transform.position;
        if (dt > 0.0001f)
        {
            Velocity = (posAfter - posBefore) / dt;
        }
    }

    private void UpdateTransform()
    {
        int nextIdx = _waypointIndex + 1;
        if (nextIdx >= waypoints.Count)
        {
            if (!loop) return;
            nextIdx = 0;
        }
        Vector3 from = waypoints[_waypointIndex];
        Vector3 to   = waypoints[nextIdx];

        Vector3 centerPos = Vector3.Lerp(from, to, _segmentProgress);
        Vector3 pathFwd = (to != from) ? (to - from).normalized : transform.forward;
        Vector3 fwd = travelDirectionSign >= 0 ? pathFwd : -pathFwd;

        // Apply lateral lane offset: Vector3.Cross(up, fwd) = right-hand side of travel.
        Vector3 right = Vector3.Cross(Vector3.up, fwd);
        transform.position = centerPos + right * laneOffsetM;

        transform.rotation = Quaternion.LookRotation(fwd, Vector3.up);
    }

    private float NormalizeArcDistance(float targetDistanceM)
    {
        if (_totalPathLength <= 0.0f)
        {
            return Mathf.Max(0.0f, targetDistanceM);
        }

        if (!loop)
        {
            return Mathf.Clamp(targetDistanceM, 0.0f, _totalPathLength);
        }

        float normalized = targetDistanceM % _totalPathLength;
        if (normalized < 0.0f)
        {
            normalized += _totalPathLength;
        }
        return normalized;
    }
}
