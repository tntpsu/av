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

    [Tooltip("When false the follower does not advance. Set to true by AVBridge " +
             "when the first Python control command is received so the lead vehicle " +
             "starts at the same time as the ego car.")]
    public bool isMovementEnabled = false;

    // ── Runtime ───────────────────────────────────────────────────────────────
    private SpeedProfiler _profiler;
    private int    _waypointIndex  = 0;
    private float  _segmentProgress = 0.0f;  // fraction [0,1] between current and next wp

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

    /// <summary>Teleport to the waypoint nearest to the given arc distance.</summary>
    public void SetArcDistance(float targetDistanceM, List<float> cumulativeDistances)
    {
        if (waypoints == null || waypoints.Count < 2 || cumulativeDistances == null)
            return;

        // Find segment containing targetDistanceM
        int idx = 0;
        for (int i = 0; i < cumulativeDistances.Count - 1; i++)
        {
            if (cumulativeDistances[i + 1] >= targetDistanceM)
            {
                idx = i;
                break;
            }
            idx = i;
        }
        idx = Mathf.Clamp(idx, 0, waypoints.Count - 2);
        _waypointIndex = idx;

        float segLen = Vector3.Distance(waypoints[idx], waypoints[idx + 1]);
        float remain  = targetDistanceM - cumulativeDistances[idx];
        _segmentProgress = segLen > 0.001f
            ? Mathf.Clamp01(remain / segLen)
            : 0f;

        UpdateTransform();
        ArcDistance = targetDistanceM;
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
        float distanceToTravel = speed * dt;

        Advance(distanceToTravel, dt);
    }

    private void Advance(float dist, float dt)
    {
        if (waypoints.Count < 2)
            return;

        Vector3 posBefore = transform.position;

        while (dist > 0.001f)
        {
            int nextIdx = _waypointIndex + 1;
            if (nextIdx >= waypoints.Count)
            {
                if (loop)
                {
                    nextIdx = 0;
                }
                else
                {
                    CurrentSpeed = 0.0f;
                    return;
                }
            }

            Vector3 current = waypoints[_waypointIndex];
            Vector3 next    = waypoints[nextIdx];
            float segLen    = Vector3.Distance(current, next);
            float remaining = segLen * (1.0f - _segmentProgress);

            if (dist >= remaining)
            {
                dist -= remaining;
                _segmentProgress = 0.0f;
                _waypointIndex   = nextIdx;
                if (_waypointIndex == 0)
                    ArcDistance = 0.0f;  // wrapped
            }
            else
            {
                _segmentProgress += dist / Mathf.Max(segLen, 0.001f);
                dist = 0.0f;
            }
        }

        UpdateTransform();
        Vector3 posAfter = transform.position;
        if (dt > 0.0001f)
        {
            Velocity = (posAfter - posBefore) / dt;
        }
        ArcDistance += CurrentSpeed * dt;
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
        Vector3 fwd = (to != from) ? (to - from).normalized : transform.forward;

        // Apply lateral lane offset: Vector3.Cross(up, fwd) = right-hand side of travel.
        Vector3 right = Vector3.Cross(Vector3.up, fwd);
        transform.position = centerPos + right * laneOffsetM;

        transform.rotation = Quaternion.LookRotation(fwd, Vector3.up);
    }
}
