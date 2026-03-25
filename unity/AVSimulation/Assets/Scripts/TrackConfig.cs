using System;
using System.Collections.Generic;

/// <summary>
/// Lead vehicle configuration parsed from the lead_vehicle: YAML block.
/// Used by TrackLoader and consumed by LeadVehicle.cs at runtime.
/// </summary>
[Serializable]
public class LeadVehicleConfig
{
    public bool enabled = false;
    public float startDistanceM = 40.0f;     // arc-distance offset ahead of ego spawn
    public string speedProfileType = "constant"; // constant | hard_brake | accel_away | stop_go | slower
    public float speedMps = 20.0f;           // base speed for profile
    public float brakeAtTimeS = 5.0f;        // hard_brake: time after start when lead brakes
    public float brakeToSpeedMps = 5.0f;     // hard_brake: target speed after braking
    public float stopGoPeriodS = 10.0f;      // stop_go: period of full stop→go cycle (s)
    public float stopGoTopSpeedMps = 20.0f;  // stop_go: top speed in cycle
    public float laneOffsetM = 0.0f;         // lateral offset from centerline (m); positive = right lane
                                             // Use roadWidth/4 for right lane of a 2-lane road.
}

[Serializable]
public class TrackConfig
{
    public string name = "track";
    public string template = "";
    public float roadWidth = 7.2f;
    public float laneLineWidth = 0.2f;
    public float sampleSpacing = 1f;
    public bool loop = true;
    public bool loopConnector = false;
    public float speedLimit = 0f; // Default speed limit (m/s) when segment limit not set
    public float startT = -1f;
    public float startDistance = -1f;
    public bool startRandom = false;
    public float startX = 0f;
    public float startY = 0f;
    public float startZ = 0f;
    public float startHeadingDeg = 0f;
    public float offsetX = 0f;
    public float offsetY = 0f;
    public float offsetZ = 0f;
    public float straightLength = 0f;
    public float turnRadius = 0f;
    public List<TrackSegment> segments = new List<TrackSegment>();
    public LeadVehicleConfig leadVehicle = new LeadVehicleConfig();
}

[Serializable]
public class TrackSegment
{
    public string type = "straight"; // straight | arc
    public float length = 0f;        // straight only
    public float radius = 0f;        // arc only
    public float angleDeg = 0f;      // arc only
    public string direction = "left";
    public float grade = 0f;         // slope (rise/run)
    public float speedLimit = 0f;    // optional
}
