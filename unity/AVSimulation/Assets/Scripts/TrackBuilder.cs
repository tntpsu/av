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
        List<float> grades = new List<float>();
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
        float firstGrade = config.segments.Count > 0 ? config.segments[0].grade : 0f;
        grades.Add(firstGrade);

        foreach (TrackSegment segment in config.segments)
        {
            if (segment.type == "straight")
            {
                AppendStraight(points, distances, speedLimits, grades, ref position, ref elevation, ref heading, segment, spacing, offset, defaultSpeedLimit);
            }
            else if (segment.type == "arc")
            {
                AppendArc(points, distances, speedLimits, grades, ref position, ref elevation, ref heading, segment, spacing, offset, defaultSpeedLimit);
            }
        }

        // Post-process: smooth grade transition boundaries to avoid mesh creases.
        SmoothGradeTransitions(points, grades, spacing);

        if (config.loop && points.Count > 1)
        {
            float gap = Vector3.Distance(points[0], points[points.Count - 1]);
            if (gap > spacing * 0.5f)
            {
                if (config.loopConnector)
                {
                    AddLoopConnector(points, distances, speedLimits, grades, spacing);
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

        return new TrackPath(points, distances, speedLimits, grades);
    }

    private static TrackPath BuildOval(TrackConfig config, Vector3 offset, float spacing)
    {
        List<Vector3> points = new List<Vector3>();
        List<float> distances = new List<float>();
        List<float> speedLimits = new List<float>();
        List<float> grades = new List<float>();
        float defaultSpeedLimit = Mathf.Max(0f, config.speedLimit);
        float totalLength = (config.straightLength * 2f) + (Mathf.PI * config.turnRadius * 2f);
        int steps = Mathf.Max(8, Mathf.CeilToInt(totalLength / spacing));

        points.Add(GetOvalPoint(config, 0f) + offset);
        distances.Add(0f);
        speedLimits.Add(defaultSpeedLimit);
        grades.Add(0f);  // Ovals are flat

        for (int i = 1; i <= steps; i++)
        {
            float t = (float)i / steps;
            Vector3 p = GetOvalPoint(config, t) + offset;
            AddPoint(points, distances, speedLimits, grades, p, defaultSpeedLimit, 0f);
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

        return new TrackPath(points, distances, speedLimits, grades);
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

    private static void AppendStraight(List<Vector3> points, List<float> distances, List<float> speedLimits, List<float> grades, ref Vector3 position,
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
            AddPoint(points, distances, speedLimits, grades, p + offset, segmentSpeedLimit, segment.grade);
        }

        elevation = startY + (segment.grade * length);
        position += forward * length;
    }

    private static void AppendArc(List<Vector3> points, List<float> distances, List<float> speedLimits, List<float> grades, ref Vector3 position,
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
            AddPoint(points, distances, speedLimits, grades, p + offset, segmentSpeedLimit, segment.grade);
        }

        elevation = startY + (segment.grade * arcLength);
        heading += turnSign * angleRad;
        position = points[points.Count - 1] - offset;
    }

    /// <summary>
    /// Post-process: cosine-blend elevation at grade-change boundaries to avoid
    /// abrupt mesh creases that cause WheelCollider lateral impulses.
    /// Grade-zero proof: flat tracks have all grades == 0 → abs(0-0) < 0.005 → no-op.
    /// </summary>
    private static void SmoothGradeTransitions(
        List<Vector3> points, List<float> grades, float spacing, float blendRadius = 3.0f)
    {
        if (points.Count < 3 || grades.Count < 3) return;
        int blendPoints = Mathf.Max(1, Mathf.CeilToInt(blendRadius / Mathf.Max(spacing, 0.01f)));

        // Collect boundary indices where grade changes
        List<int> boundaries = new List<int>();
        for (int i = 1; i < grades.Count; i++)
        {
            if (Mathf.Abs(grades[i] - grades[i - 1]) > 0.005f)
                boundaries.Add(i);
        }

        foreach (int bnd in boundaries)
        {
            int start = Mathf.Max(1, bnd - blendPoints);
            int end = Mathf.Min(grades.Count - 1, bnd + blendPoints);
            if (end <= start) continue;

            float gradeA = grades[Mathf.Max(0, start - 1)];
            float gradeB = grades[Mathf.Min(grades.Count - 1, end)];

            for (int j = start; j <= end; j++)
            {
                float t = (float)(j - start) / (end - start);
                float blend = 0.5f * (1.0f - Mathf.Cos(t * Mathf.PI)); // cosine ease
                float blendedGrade = Mathf.Lerp(gradeA, gradeB, blend);

                // Recompute elevation from previous point
                float dx = Vector3.Distance(
                    new Vector3(points[j].x, 0f, points[j].z),
                    new Vector3(points[j - 1].x, 0f, points[j - 1].z));
                Vector3 p = points[j];
                p.y = points[j - 1].y + blendedGrade * dx;
                points[j] = p;
                grades[j] = blendedGrade;
            }
        }
    }

    private static void AddPoint(List<Vector3> points, List<float> distances, List<float> speedLimits, List<float> grades,
        Vector3 point, float speedLimit, float grade = 0f)
    {
        if (points.Count == 0)
        {
            points.Add(point);
            distances.Add(0f);
            speedLimits.Add(speedLimit);
            grades.Add(grade);
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
        grades.Add(grade);
    }

    private static float GetSegmentSpeedLimit(TrackSegment segment, float defaultSpeedLimit)
    {
        if (segment == null)
        {
            return defaultSpeedLimit;
        }
        return segment.speedLimit > 0f ? segment.speedLimit : defaultSpeedLimit;
    }

    private static void AddLoopConnector(List<Vector3> points, List<float> distances, List<float> speedLimits, List<float> grades, float spacing)
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
        // Connector grade: compute from elevation delta to close the loop
        float connectorGrade = gap > 0.01f ? (start.y - end.y) / gap : 0f;
        for (int i = 1; i <= steps; i++)
        {
            float t = (float)i / steps;
            Vector3 p = Vector3.Lerp(end, start, t);
            AddPoint(points, distances, speedLimits, grades, p, connectorSpeed, connectorGrade);
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
