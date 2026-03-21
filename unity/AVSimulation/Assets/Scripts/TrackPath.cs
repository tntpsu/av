using System.Collections.Generic;
using UnityEngine;

public class TrackPath
{
    public List<Vector3> Points { get; private set; }
    public List<float> Distances { get; private set; }
    public List<float> SpeedLimits { get; private set; }
    public List<float> Grades { get; private set; }
    public float TotalLength { get; private set; }

    public TrackPath(List<Vector3> points, List<float> distances, List<float> speedLimits, List<float> grades = null)
    {
        Points = points;
        Distances = distances;
        SpeedLimits = speedLimits ?? new List<float>();
        Grades = grades ?? new List<float>();
        TotalLength = distances.Count > 0 ? distances[distances.Count - 1] : 0f;
    }

    public Vector3 GetPointAtT(float t)
    {
        return GetPointAtDistance(t * TotalLength);
    }

    public Vector3 GetDirectionAtT(float t)
    {
        return GetDirectionAtDistance(t * TotalLength);
    }

    public Vector3 GetPointAtDistance(float distance)
    {
        if (Points == null || Points.Count == 0)
        {
            return Vector3.zero;
        }

        if (TotalLength <= 0f)
        {
            return Points[0];
        }

        float d = Mathf.Repeat(distance, TotalLength);
        int idx = FindSegmentIndex(d);
        if (idx <= 0)
        {
            return Points[0];
        }
        if (idx >= Points.Count)
        {
            return Points[Points.Count - 1];
        }

        float d0 = Distances[idx - 1];
        float d1 = Distances[idx];
        float t = d1 > d0 ? (d - d0) / (d1 - d0) : 0f;
        return Vector3.Lerp(Points[idx - 1], Points[idx], t);
    }

    public Vector3 GetDirectionAtDistance(float distance)
    {
        if (Points == null || Points.Count < 2)
        {
            return Vector3.forward;
        }

        float d = Mathf.Repeat(distance, TotalLength);
        int idx = FindSegmentIndex(d);
        int a = Mathf.Clamp(idx - 1, 0, Points.Count - 2);
        int b = Mathf.Clamp(idx, 1, Points.Count - 1);
        Vector3 dir = (Points[b] - Points[a]).normalized;
        return dir.sqrMagnitude > 0f ? dir : Vector3.forward;
    }

    public float GetSpeedLimitAtT(float t)
    {
        return GetSpeedLimitAtDistance(t * TotalLength);
    }

    public float GetSpeedLimitAtDistance(float distance)
    {
        if (SpeedLimits == null || SpeedLimits.Count == 0 || TotalLength <= 0f)
        {
            return 0f;
        }
        float d = Mathf.Repeat(distance, TotalLength);
        int idx = FindSegmentIndex(d);
        int a = Mathf.Clamp(idx - 1, 0, SpeedLimits.Count - 1);
        int b = Mathf.Clamp(idx, 0, SpeedLimits.Count - 1);
        float d0 = Distances[a];
        float d1 = Distances[b];
        float t = d1 > d0 ? (d - d0) / (d1 - d0) : 0f;
        return Mathf.Lerp(SpeedLimits[a], SpeedLimits[b], t);
    }

    public float GetGradeAtT(float t)
    {
        return GetGradeAtDistance(t * TotalLength);
    }

    public float GetGradeAtDistance(float distance)
    {
        if (Grades == null || Grades.Count == 0 || TotalLength <= 0f)
        {
            return 0f;
        }
        float d = Mathf.Repeat(distance, TotalLength);
        int idx = FindSegmentIndex(d);
        // Use piecewise-constant lookup (no interpolation — grade is constant per segment)
        return Grades[Mathf.Clamp(idx, 0, Grades.Count - 1)];
    }

    private int FindSegmentIndex(float distance)
    {
        // Binary search on sorted Distances — O(log N) vs the previous O(N) linear scan.
        // Critical for large tracks (e.g. highway_65 with 7012 points).
        int lo = 1, hi = Distances.Count - 1;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (Distances[mid] < distance)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

}
