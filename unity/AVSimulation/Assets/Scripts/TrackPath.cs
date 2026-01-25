using System.Collections.Generic;
using UnityEngine;

public class TrackPath
{
    public List<Vector3> Points { get; private set; }
    public List<float> Distances { get; private set; }
    public float TotalLength { get; private set; }

    public TrackPath(List<Vector3> points, List<float> distances)
    {
        Points = points;
        Distances = distances;
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

    private int FindSegmentIndex(float distance)
    {
        for (int i = 1; i < Distances.Count; i++)
        {
            if (Distances[i] >= distance)
            {
                return i;
            }
        }
        return Distances.Count - 1;
    }
}
