using System.Collections.Generic;
using UnityEngine;

public static class TrackBuilder
{
    public static TrackPath BuildPath(TrackConfig config, Vector3 offset)
    {
        float spacing = config.sampleSpacing > 0.01f ? config.sampleSpacing : 1f;
        List<Vector3> points = new List<Vector3>();
        List<float> distances = new List<float>();
        List<float> speedLimits = new List<float>();
        float defaultSpeedLimit = Mathf.Max(0f, config.speedLimit);

        if (config.template == "oval" && config.straightLength > 0f && config.turnRadius > 0f)
        {
            return BuildOval(config, offset, spacing);
        }

        Vector3 position = new Vector3(config.startX, config.startY, config.startZ);
        float heading = Mathf.Deg2Rad * config.startHeadingDeg; // radians, 0 = +Z
        float elevation = config.startY;

        points.Add(position + offset);
        distances.Add(0f);
        float firstSpeed = config.segments.Count > 0
            ? GetSegmentSpeedLimit(config.segments[0], defaultSpeedLimit)
            : defaultSpeedLimit;
        speedLimits.Add(firstSpeed);

        foreach (TrackSegment segment in config.segments)
        {
            if (segment.type == "straight")
            {
                AppendStraight(points, distances, speedLimits, ref position, ref elevation, ref heading, segment, spacing, offset, defaultSpeedLimit);
            }
            else if (segment.type == "arc")
            {
                AppendArc(points, distances, speedLimits, ref position, ref elevation, ref heading, segment, spacing, offset, defaultSpeedLimit);
            }
        }

        if (config.loop && points.Count > 1)
        {
            float gap = Vector3.Distance(points[0], points[points.Count - 1]);
            if (gap > spacing * 0.5f)
            {
                if (config.loopConnector)
                {
                    AddLoopConnector(points, distances, speedLimits, spacing);
                }
                else
                {
                    Debug.LogWarning($"TrackBuilder: Loop gap is {gap:F2}m (spacing={spacing:F2}). " +
                                     "Segments do not close; no straight connector will be added.");
                }
            }
        }

        if (points.Count > 1)
        {
            Vector3 min = points[0];
            Vector3 max = points[0];
            foreach (Vector3 p in points)
            {
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
            }
            Debug.Log($"TrackBuilder: Bounds min={min} max={max} length={distances[distances.Count - 1]:F1}m");
        }
        if (speedLimits.Count > 0)
        {
            float minSpeed = speedLimits[0];
            float maxSpeed = speedLimits[0];
            foreach (float limit in speedLimits)
            {
                minSpeed = Mathf.Min(minSpeed, limit);
                maxSpeed = Mathf.Max(maxSpeed, limit);
            }
            Debug.Log($"TrackBuilder: Speed limits min={minSpeed:F2} max={maxSpeed:F2} (m/s)");
        }

        return new TrackPath(points, distances, speedLimits);
    }

    private static TrackPath BuildOval(TrackConfig config, Vector3 offset, float spacing)
    {
        List<Vector3> points = new List<Vector3>();
        List<float> distances = new List<float>();
        List<float> speedLimits = new List<float>();
        float defaultSpeedLimit = Mathf.Max(0f, config.speedLimit);
        float totalLength = (config.straightLength * 2f) + (Mathf.PI * config.turnRadius * 2f);
        int steps = Mathf.Max(8, Mathf.CeilToInt(totalLength / spacing));

        points.Add(GetOvalPoint(config, 0f) + offset);
        distances.Add(0f);
        speedLimits.Add(defaultSpeedLimit);

        for (int i = 1; i <= steps; i++)
        {
            float t = (float)i / steps;
            Vector3 p = GetOvalPoint(config, t) + offset;
            AddPoint(points, distances, speedLimits, p, defaultSpeedLimit);
        }

        if (points.Count > 1)
        {
            Vector3 min = points[0];
            Vector3 max = points[0];
            foreach (Vector3 p in points)
            {
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
            }
            Debug.Log($"TrackBuilder: Bounds min={min} max={max} length={distances[distances.Count - 1]:F1}m");
        }
        if (speedLimits.Count > 0)
        {
            float minSpeed = speedLimits[0];
            float maxSpeed = speedLimits[0];
            foreach (float limit in speedLimits)
            {
                minSpeed = Mathf.Min(minSpeed, limit);
                maxSpeed = Mathf.Max(maxSpeed, limit);
            }
            Debug.Log($"TrackBuilder: Speed limits min={minSpeed:F2} max={maxSpeed:F2} (m/s)");
        }

        return new TrackPath(points, distances, speedLimits);
    }

    private static Vector3 GetOvalPoint(TrackConfig config, float t)
    {
        t = Mathf.Repeat(t, 1f);
        float segmentLength = 0.25f;
        float localT;

        if (t < 0.25f)
        {
            localT = t / segmentLength;
            float z = config.turnRadius;
            float x = Mathf.Lerp(-config.straightLength * 0.5f, config.straightLength * 0.5f, localT);
            return new Vector3(x, 0f, z);
        }
        if (t < 0.5f)
        {
            localT = (t - 0.25f) / segmentLength;
            float angle = Mathf.Lerp(90f, -90f, localT) * Mathf.Deg2Rad;
            float cx = config.straightLength * 0.5f;
            float x = cx + config.turnRadius * Mathf.Cos(angle);
            float z = 0f + config.turnRadius * Mathf.Sin(angle);
            return new Vector3(x, 0f, z);
        }
        if (t < 0.75f)
        {
            localT = (t - 0.5f) / segmentLength;
            float z = -config.turnRadius;
            float x = Mathf.Lerp(config.straightLength * 0.5f, -config.straightLength * 0.5f, localT);
            return new Vector3(x, 0f, z);
        }

        localT = (t - 0.75f) / segmentLength;
        float angle2 = Mathf.Lerp(-90f, 90f, localT) * Mathf.Deg2Rad;
        float cx2 = -config.straightLength * 0.5f;
        float x2 = cx2 - config.turnRadius * Mathf.Cos(angle2);
        float z2 = 0f + config.turnRadius * Mathf.Sin(angle2);
        return new Vector3(x2, 0f, z2);
    }

    private static void AppendStraight(List<Vector3> points, List<float> distances, List<float> speedLimits, ref Vector3 position,
        ref float elevation, ref float heading, TrackSegment segment, float spacing, Vector3 offset, float defaultSpeedLimit)
    {
        float length = Mathf.Max(0f, segment.length);
        int steps = Mathf.Max(1, Mathf.CeilToInt(length / spacing));
        Vector3 forward = HeadingToForward(heading);
        float segmentSpeedLimit = GetSegmentSpeedLimit(segment, defaultSpeedLimit);

        float startY = elevation;
        for (int i = 1; i <= steps; i++)
        {
            float t = (float)i / steps;
            float d = length * t;
            float y = startY + (segment.grade * d);
            Vector3 p = position + forward * d;
            p.y = y;
            AddPoint(points, distances, speedLimits, p + offset, segmentSpeedLimit);
        }

        elevation = startY + (segment.grade * length);
        position += forward * length;
    }

    private static void AppendArc(List<Vector3> points, List<float> distances, List<float> speedLimits, ref Vector3 position,
        ref float elevation, ref float heading, TrackSegment segment, float spacing, Vector3 offset, float defaultSpeedLimit)
    {
        float radius = Mathf.Max(0.1f, segment.radius);
        float angleRad = Mathf.Deg2Rad * Mathf.Abs(segment.angleDeg);
        float turnSign = segment.direction == "right" ? 1f : -1f;
        float segmentSpeedLimit = GetSegmentSpeedLimit(segment, defaultSpeedLimit);
        float angleSign = -turnSign;
        float arcLength = radius * angleRad;
        int steps = Mathf.Max(4, Mathf.CeilToInt(arcLength / spacing));

        Vector3 forward = HeadingToForward(heading);
        Vector3 right = HeadingToRight(heading);
        Vector3 center = position + right * turnSign * radius;
        Vector3 fromCenter = position - center;
        float startAngle = Mathf.Atan2(fromCenter.z, fromCenter.x);

        float startY = elevation;
        for (int i = 1; i <= steps; i++)
        {
            float t = (float)i / steps;
            float angle = startAngle + angleSign * angleRad * t;
            float y = startY + (segment.grade * arcLength * t);
            Vector3 p = new Vector3(
                center.x + Mathf.Cos(angle) * radius,
                y,
                center.z + Mathf.Sin(angle) * radius
            );
            AddPoint(points, distances, speedLimits, p + offset, segmentSpeedLimit);
        }

        elevation = startY + (segment.grade * arcLength);
        heading += turnSign * angleRad;
        position = points[points.Count - 1] - offset;
    }

    private static void AddPoint(List<Vector3> points, List<float> distances, List<float> speedLimits,
        Vector3 point, float speedLimit)
    {
        if (points.Count == 0)
        {
            points.Add(point);
            distances.Add(0f);
            speedLimits.Add(speedLimit);
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
        speedLimits.Add(speedLimit);
    }

    private static float GetSegmentSpeedLimit(TrackSegment segment, float defaultSpeedLimit)
    {
        if (segment == null)
        {
            return defaultSpeedLimit;
        }
        return segment.speedLimit > 0f ? segment.speedLimit : defaultSpeedLimit;
    }

    private static void AddLoopConnector(List<Vector3> points, List<float> distances, List<float> speedLimits, float spacing)
    {
        Vector3 start = points[0];
        Vector3 end = points[points.Count - 1];
        float gap = Vector3.Distance(start, end);
        if (gap < spacing * 0.5f)
        {
            return;
        }
        int steps = Mathf.Max(1, Mathf.CeilToInt(gap / spacing));
        float connectorSpeed = speedLimits.Count > 0 ? speedLimits[speedLimits.Count - 1] : 0f;
        for (int i = 1; i <= steps; i++)
        {
            float t = (float)i / steps;
            Vector3 p = Vector3.Lerp(end, start, t);
            AddPoint(points, distances, speedLimits, p, connectorSpeed);
        }
        Debug.Log($"TrackBuilder: Added loop connector (gap={gap:F2}m, steps={steps})");
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
