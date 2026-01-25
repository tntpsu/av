using System.Collections.Generic;
using UnityEngine;

public static class TrackBuilder
{
    public static TrackPath BuildPath(TrackConfig config, Vector3 offset)
    {
        float spacing = config.sampleSpacing > 0.01f ? config.sampleSpacing : 1f;
        List<Vector3> points = new List<Vector3>();
        List<float> distances = new List<float>();

        Vector3 position = Vector3.zero;
        float heading = 0f; // radians, 0 = +Z
        float elevation = 0f;

        points.Add(position + offset);
        distances.Add(0f);

        foreach (TrackSegment segment in config.segments)
        {
            if (segment.type == "straight")
            {
                AppendStraight(points, distances, ref position, ref elevation, ref heading, segment, spacing, offset);
            }
            else if (segment.type == "arc")
            {
                AppendArc(points, distances, ref position, ref elevation, ref heading, segment, spacing, offset);
            }
        }

        if (config.loop && points.Count > 1)
        {
            float gap = Vector3.Distance(points[0], points[points.Count - 1]);
            if (gap > spacing * 0.5f)
            {
                float lastDistance = distances[distances.Count - 1];
                distances.Add(lastDistance + gap);
                points.Add(points[0]);
            }
        }

        return new TrackPath(points, distances);
    }

    private static void AppendStraight(List<Vector3> points, List<float> distances, ref Vector3 position,
        ref float elevation, ref float heading, TrackSegment segment, float spacing, Vector3 offset)
    {
        float length = Mathf.Max(0f, segment.length);
        int steps = Mathf.Max(1, Mathf.CeilToInt(length / spacing));
        Vector3 forward = HeadingToForward(heading);

        float startY = elevation;
        for (int i = 1; i <= steps; i++)
        {
            float t = (float)i / steps;
            float d = length * t;
            float y = startY + (segment.grade * d);
            Vector3 p = position + forward * d;
            p.y = y;
            AddPoint(points, distances, p + offset);
        }

        elevation = startY + (segment.grade * length);
        position += forward * length;
    }

    private static void AppendArc(List<Vector3> points, List<float> distances, ref Vector3 position,
        ref float elevation, ref float heading, TrackSegment segment, float spacing, Vector3 offset)
    {
        float radius = Mathf.Max(0.1f, segment.radius);
        float angleRad = Mathf.Deg2Rad * Mathf.Abs(segment.angleDeg);
        float sign = segment.direction == "right" ? 1f : -1f;
        float arcLength = radius * angleRad;
        int steps = Mathf.Max(4, Mathf.CeilToInt(arcLength / spacing));

        Vector3 forward = HeadingToForward(heading);
        Vector3 right = HeadingToRight(heading);
        Vector3 center = position + right * sign * radius;
        Vector3 fromCenter = position - center;
        float startAngle = Mathf.Atan2(fromCenter.z, fromCenter.x);

        float startY = elevation;
        for (int i = 1; i <= steps; i++)
        {
            float t = (float)i / steps;
            float angle = startAngle + sign * angleRad * t;
            float y = startY + (segment.grade * arcLength * t);
            Vector3 p = new Vector3(
                center.x + Mathf.Cos(angle) * radius,
                y,
                center.z + Mathf.Sin(angle) * radius
            );
            AddPoint(points, distances, p + offset);
        }

        elevation = startY + (segment.grade * arcLength);
        heading += sign * angleRad;
        position = points[points.Count - 1] - offset;
    }

    private static void AddPoint(List<Vector3> points, List<float> distances, Vector3 point)
    {
        if (points.Count == 0)
        {
            points.Add(point);
            distances.Add(0f);
            return;
        }

        float d = Vector3.Distance(points[points.Count - 1], point);
        if (d < 0.001f)
        {
            return;
        }

        float lastDistance = distances[distances.Count - 1];
        distances.Add(lastDistance + d);
        points.Add(point);
    }

    private static Vector3 HeadingToForward(float heading)
    {
        return new Vector3(Mathf.Sin(heading), 0f, Mathf.Cos(heading));
    }

    private static Vector3 HeadingToRight(float heading)
    {
        return new Vector3(Mathf.Cos(heading), 0f, -Mathf.Sin(heading));
    }
}
