using System;
using System.Collections.Generic;

[Serializable]
public class TrackConfig
{
    public string name = "track";
    public float roadWidth = 7f;
    public float laneLineWidth = 0.3f;
    public float sampleSpacing = 1f;
    public bool loop = true;
    public float startT = -1f;
    public float startDistance = -1f;
    public bool startRandom = false;
    public float offsetX = 0f;
    public float offsetY = 0f;
    public float offsetZ = 0f;
    public List<TrackSegment> segments = new List<TrackSegment>();
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
