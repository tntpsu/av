using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Lead vehicle actor for Step 5 ACC scenarios.
///
/// Instantiated at runtime by the scene initialisation code (see GroundTruthReporter
/// or the scene manager) when track YAML includes a lead_vehicle block with enabled=true.
///
/// Architecture:
///   LeadVehicle → TrackWaypointFollower (movement) + SpeedProfiler (speed policy)
///
/// The lead vehicle is a kinematic box — no WheelCollider, no rigidbody physics.
/// A BoxCollider (trigger) is present so the forward radar SphereCast can detect it.
/// A simple white MeshRenderer gives visual feedback in the Game view.
/// </summary>
[RequireComponent(typeof(TrackWaypointFollower))]
[RequireComponent(typeof(SpeedProfiler))]
public class LeadVehicle : MonoBehaviour
{
    // ── Visual representation ─────────────────────────────────────────────────
    [Header("Appearance")]
    [Tooltip("Physical dimensions matching a mid-size sedan (length × height × width).")]
    public Vector3 vehicleBoxSize = new Vector3(2.0f, 1.5f, 4.5f);  // W × H × L

    // ── Runtime references ────────────────────────────────────────────────────
    private TrackWaypointFollower _follower;
    private SpeedProfiler         _profiler;

    /// <summary>Returns the current velocity vector (world space, m/s).</summary>
    public Vector3 Velocity => _follower != null ? _follower.Velocity : Vector3.zero;

    /// <summary>Returns the current speed (m/s).</summary>
    public float Speed => _follower != null ? _follower.CurrentSpeed : 0.0f;

    /// <summary>True after a collision with the ego car was detected this run.</summary>
    public bool CollisionDetected { get; private set; } = false;

    // ──────────────────────────────────────────────────────────────────────────

    void Awake()
    {
        _follower = GetComponent<TrackWaypointFollower>();
        _profiler = GetComponent<SpeedProfiler>();
    }

    /// <summary>
    /// Initialise from YAML config. Called by the scene initialisation after
    /// TrackBuilder has built the path waypoints.
    /// </summary>
    public void Initialise(
        LeadVehicleConfig cfg,
        List<Vector3>     waypoints,
        List<float>       cumulativeDistances,
        bool              trackLoop)
    {
        // Wire speed profiler
        _profiler.Initialise(cfg);

        // Wire follower — movement is DISABLED until AVBridge calls StartMovement()
        _follower.waypoints        = waypoints;
        _follower.loop             = trackLoop;
        _follower.laneOffsetM      = cfg.laneOffsetM;
        _follower.SetTravelDirection(cfg.travelDirection);
        _follower.isMovementEnabled = false;

        // Place at start_distance_m ahead of arc origin
        _follower.SetArcDistance(cfg.startDistanceM, cumulativeDistances);

        // Build visual mesh (white box)
        var mf = gameObject.AddComponent<MeshFilter>();
        mf.mesh = BuildBoxMesh(vehicleBoxSize);
        var mr = gameObject.AddComponent<MeshRenderer>();
        var mat = BuildWhiteMaterial();
        if (mat != null) mr.material = mat;

        // Trigger collider for SphereCast detection
        var bc = gameObject.AddComponent<BoxCollider>();
        bc.size      = vehicleBoxSize;
        bc.center    = new Vector3(0f, vehicleBoxSize.y * 0.5f, 0f);  // ground-plane centred
        bc.isTrigger = true;

    }

    // ── Movement control ──────────────────────────────────────────────────────

    /// <summary>
    /// Called by AVBridge when the first Python control command arrives.
    /// Until this is called the lead vehicle stays at its spawn position.
    /// </summary>
    public void StartMovement()
    {
        if (_follower != null)
            _follower.isMovementEnabled = true;
    }

    /// <summary>
    /// Immediately halts the lead vehicle (e.g. on collision).
    /// </summary>
    public void StopMovement()
    {
        if (_follower != null)
            _follower.isMovementEnabled = false;
        // CurrentSpeed is read-only; it will drop to 0 naturally on the next
        // FixedUpdate when isMovementEnabled=false gates the advance step.
    }

    // ── Collision detection ───────────────────────────────────────────────────

    /// <summary>
    /// Detects ego–lead contact via the trigger BoxCollider.
    /// Stops the lead vehicle immediately and sets CollisionDetected so
    /// AVBridge can send an emergency_stop to Python on the next frame.
    /// </summary>
    void OnTriggerEnter(Collider other)
    {
        // Only react to the ego car (tagged "Player" by CarController, or has Rigidbody).
        if (other.GetComponent<Rigidbody>() == null && !other.CompareTag("Player"))
            return;

        if (!CollisionDetected)
        {
            CollisionDetected = true;
            StopMovement();
            Debug.LogWarning("LeadVehicle: collision with ego car detected — stopping lead vehicle.");
        }
    }

    // ── Mesh helpers ──────────────────────────────────────────────────────────

    private static Mesh BuildBoxMesh(Vector3 size)
    {
        // Unity's built-in Cube primitive scaled to the desired size.
        // We clone it here to avoid shared-asset mutation.
        GameObject tmp = GameObject.CreatePrimitive(PrimitiveType.Cube);
        Mesh mesh = tmp.GetComponent<MeshFilter>().sharedMesh;
        Destroy(tmp);

        // Scale UVs/verts to desired box dimensions
        Mesh scaled = new Mesh();
        scaled.name = "LeadVehicleBox";
        Vector3[] verts = mesh.vertices;
        for (int i = 0; i < verts.Length; i++)
        {
            verts[i] = Vector3.Scale(verts[i], size);
        }
        scaled.vertices  = verts;
        scaled.triangles = mesh.triangles;
        scaled.normals   = mesh.normals;
        scaled.uv        = mesh.uv;
        scaled.RecalculateBounds();
        return scaled;
    }

    private static Material BuildWhiteMaterial()
    {
        // Try render-pipeline shaders in priority order; the visual is purely decorative
        // so returning null is safe — the MeshRenderer will use its default error material.
        Shader s = Shader.Find("Universal Render Pipeline/Lit")
                ?? Shader.Find("Standard")
                ?? Shader.Find("Unlit/Color")
                ?? Shader.Find("Unlit/Texture");
        if (s == null)
        {
            Debug.LogWarning("LeadVehicle: no suitable shader found — using error material.");
            return null;
        }
        Material m = new Material(s);
        m.color = new Color(0.9f, 0.9f, 0.9f);
        return m;
    }
}
