using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

public static class TrackLoader
{
    public static bool TryLoadFromCommandLine(out TrackConfig config)
    {
        config = null;
        string[] args = Environment.GetCommandLineArgs();
        string yamlPath = GetArgValue(args, "-trackYaml", "--track-yaml");
        if (string.IsNullOrEmpty(yamlPath))
        {
            return false;
        }

        if (!File.Exists(yamlPath))
        {
            Debug.LogError($"Track YAML not found: {yamlPath}");
            return false;
        }

        config = LoadFromText(File.ReadAllText(yamlPath));
        ApplyStartOverrides(args, config);
        Debug.Log($"TrackLoader: Loaded YAML '{yamlPath}' name='{config.name}' template='{config.template}' segments={config.segments.Count}");
        return true;
    }

    public static TrackConfig LoadFromText(string yamlText)
    {
        TrackConfig config = new TrackConfig();
        TrackSegment currentSegment = null;
        bool inSegments = false;

        string[] lines = yamlText.Split('\n');
        foreach (string rawLine in lines)
        {
            string line = StripComment(rawLine).Trim();
            if (string.IsNullOrEmpty(line))
            {
                continue;
            }

            if (line.StartsWith("segments:"))
            {
                inSegments = true;
                continue;
            }

            if (inSegments)
            {
                if (line.StartsWith("-"))
                {
                    currentSegment = new TrackSegment();
                    config.segments.Add(currentSegment);
                    string remainder = line.Substring(1).Trim();
                    if (!string.IsNullOrEmpty(remainder))
                    {
                        ParseSegmentKeyValue(currentSegment, remainder);
                    }
                }
                else if (currentSegment != null)
                {
                    ParseSegmentKeyValue(currentSegment, line);
                }
            }
            else
            {
                ParseConfigKeyValue(config, line);
            }
        }

        if (config.speedLimit <= 0f)
        {
            float fallbackLimit = 0f;
            bool inSegmentsFallback = false;
            foreach (string rawLine in lines)
            {
                string fallbackLine = StripComment(rawLine).Trim();
                if (string.IsNullOrEmpty(fallbackLine))
                {
                    continue;
                }
                if (fallbackLine.StartsWith("segments:"))
                {
                    inSegmentsFallback = true;
                    continue;
                }
                if (inSegmentsFallback)
                {
                    if (fallbackLine.StartsWith("-"))
                    {
                        // new segment; stay in segments block
                    }
                    continue;
                }
                if (TrySplitKeyValue(fallbackLine, out string key, out string value))
                {
                    if (key == "speed_limit_mph" && TryParseFloat(value, out float mphValue))
                    {
                        fallbackLimit = MphToMps(mphValue);
                        break;
                    }
                    if (key == "speed_limit" && TryParseFloat(value, out float mpsValue))
                    {
                        fallbackLimit = mpsValue;
                        break;
                    }
                }
            }
            if (fallbackLimit > 0f)
            {
                config.speedLimit = fallbackLimit;
                Debug.Log($"TrackLoader: Parsed fallback speed limit {fallbackLimit:F2} m/s");
            }
        }
        if (config.speedLimit <= 0f)
        {
            float maxSegmentLimit = 0f;
            foreach (var segment in config.segments)
            {
                if (segment != null && segment.speedLimit > maxSegmentLimit)
                {
                    maxSegmentLimit = segment.speedLimit;
                }
            }
            if (maxSegmentLimit > 0f)
            {
                config.speedLimit = maxSegmentLimit;
                Debug.LogWarning($"TrackLoader: Default speed limit missing; using max segment limit {maxSegmentLimit:F2} m/s");
            }
        }
        if (config.speedLimit > 0f)
        {
            foreach (var segment in config.segments)
            {
                if (segment != null && segment.speedLimit <= 0f)
                {
                    segment.speedLimit = config.speedLimit;
                }
            }
        }

        return config;
    }

    private static void ApplyStartOverrides(string[] args, TrackConfig config)
    {
        string startT = GetArgValue(args, "--start-t");
        if (!string.IsNullOrEmpty(startT) && TryParseFloat(startT, out float startTVal))
        {
            config.startT = startTVal;
        }

        string startDistance = GetArgValue(args, "--start-distance");
        if (!string.IsNullOrEmpty(startDistance) && TryParseFloat(startDistance, out float startDistVal))
        {
            config.startDistance = startDistVal;
        }

        string startRandom = GetArgValue(args, "--start-random");
        if (!string.IsNullOrEmpty(startRandom))
        {
            config.startRandom = startRandom == "true" || startRandom == "1";
        }
    }

    private static void ParseConfigKeyValue(TrackConfig config, string line)
    {
        if (!TrySplitKeyValue(line, out string key, out string value))
        {
            return;
        }

        switch (key)
        {
            case "name":
                config.name = Unquote(value);
                break;
            case "template":
                config.template = Unquote(value).ToLowerInvariant();
                break;
            case "road_width":
                if (TryParseFloat(value, out float rw)) config.roadWidth = rw;
                break;
            case "lane_line_width":
                if (TryParseFloat(value, out float lw)) config.laneLineWidth = lw;
                break;
            case "sample_spacing":
                if (TryParseFloat(value, out float ss)) config.sampleSpacing = ss;
                break;
            case "loop":
                config.loop = value == "true" || value == "1";
                break;
            case "start_t":
                if (TryParseFloat(value, out float st)) config.startT = st;
                break;
            case "start_distance":
                if (TryParseFloat(value, out float sd)) config.startDistance = sd;
                break;
            case "start_random":
                config.startRandom = value == "true" || value == "1";
                break;
            case "start_x":
                if (TryParseFloat(value, out float sx)) config.startX = sx;
                break;
            case "start_y":
                if (TryParseFloat(value, out float sy)) config.startY = sy;
                break;
            case "start_z":
                if (TryParseFloat(value, out float sz)) config.startZ = sz;
                break;
            case "start_heading_deg":
                if (TryParseFloat(value, out float sh)) config.startHeadingDeg = sh;
                break;
            case "straight_length":
                if (TryParseFloat(value, out float sl)) config.straightLength = sl;
                break;
            case "turn_radius":
                if (TryParseFloat(value, out float tr)) config.turnRadius = tr;
                break;
            case "offset_x":
            case "track_offset_x":
                if (TryParseFloat(value, out float ox)) config.offsetX = ox;
                break;
            case "offset_y":
            case "track_offset_y":
                if (TryParseFloat(value, out float oy)) config.offsetY = oy;
                break;
            case "offset_z":
            case "track_offset_z":
                if (TryParseFloat(value, out float oz)) config.offsetZ = oz;
                break;
            case "loop_connector":
                if (TryParseBool(value, out bool loopConnector)) config.loopConnector = loopConnector;
                break;
            case "speed_limit":
                if (TryParseFloat(value, out float speedLimit)) config.speedLimit = speedLimit;
                break;
            case "speed_limit_mph":
                if (TryParseFloat(value, out float speedLimitMph)) config.speedLimit = MphToMps(speedLimitMph);
                break;
        }
    }

    private static void ParseSegmentKeyValue(TrackSegment segment, string line)
    {
        if (!TrySplitKeyValue(line, out string key, out string value))
        {
            return;
        }

        switch (key)
        {
            case "type":
                segment.type = Unquote(value).ToLowerInvariant();
                break;
            case "length":
                if (TryParseFloat(value, out float len)) segment.length = len;
                break;
            case "radius":
                if (TryParseFloat(value, out float rad)) segment.radius = rad;
                break;
            case "angle_deg":
            case "angle":
                if (TryParseFloat(value, out float ang)) segment.angleDeg = ang;
                break;
            case "direction":
                segment.direction = Unquote(value).ToLowerInvariant();
                break;
            case "grade":
                if (TryParseFloat(value, out float grade)) segment.grade = grade;
                break;
            case "speed_limit":
                if (TryParseFloat(value, out float limit)) segment.speedLimit = limit;
                break;
            case "speed_limit_mph":
                if (TryParseFloat(value, out float limitMph)) segment.speedLimit = MphToMps(limitMph);
                break;
        }
    }

    private static float MphToMps(float mph)
    {
        return mph * 0.44704f;
    }

    private static string StripComment(string line)
    {
        int idx = line.IndexOf('#');
        if (idx >= 0)
        {
            return line.Substring(0, idx);
        }
        return line;
    }

    private static bool TrySplitKeyValue(string line, out string key, out string value)
    {
        key = null;
        value = null;
        int idx = line.IndexOf(':');
        if (idx < 0)
        {
            return false;
        }

        key = line.Substring(0, idx).Trim();
        value = line.Substring(idx + 1).Trim();
        return true;
    }

    private static string Unquote(string value)
    {
        if (value.StartsWith("\"") && value.EndsWith("\""))
        {
            return value.Substring(1, value.Length - 2);
        }
        if (value.StartsWith("'") && value.EndsWith("'"))
        {
            return value.Substring(1, value.Length - 2);
        }
        return value;
    }

    private static bool TryParseFloat(string value, out float result)
    {
        return float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out result);
    }

    private static bool TryParseBool(string value, out bool result)
    {
        result = false;
        if (string.IsNullOrWhiteSpace(value))
        {
            return false;
        }
        string normalized = value.Trim().ToLowerInvariant();
        if (normalized == "true" || normalized == "1" || normalized == "yes" || normalized == "y")
        {
            result = true;
            return true;
        }
        if (normalized == "false" || normalized == "0" || normalized == "no" || normalized == "n")
        {
            result = false;
            return true;
        }
        return false;
    }

    private static string GetArgValue(string[] args, params string[] keys)
    {
        for (int i = 0; i < args.Length; i++)
        {
            foreach (string key in keys)
            {
                if (args[i] == key && i + 1 < args.Length)
                {
                    return args[i + 1];
                }
            }
        }
        return null;
    }
}
