using UnityEngine;

/// <summary>
/// Computes the desired speed for a lead vehicle at each simulation frame.
///
/// Profiles (matching step5_acc_plan.md scenario matrix):
///   constant    — always speedMps
///   slower      — constant at speedMps (< ego target); used for steady catch-up
///   hard_brake  — holds speedMps until brakeAtTimeS, then decelerates at 3 m/s²
///                 until brakeToSpeedMps, then holds
///   accel_away  — starts at speedMps, ramps up at 1.5 m/s² until stopGoTopSpeedMps
///   stop_go     — sinusoidal oscillation between 0 and stopGoTopSpeedMps over
///                 stopGoPeriodS seconds (smooth stop-and-go)
/// </summary>
public class SpeedProfiler : MonoBehaviour
{
    // ── Configuration (set by LeadVehicle.Initialise) ─────────────────────────
    public string profileType  = "constant";
    public float  speedMps     = 20.0f;
    public float  brakeAtTimeS = 5.0f;
    public float  brakeToSpeedMps  = 5.0f;
    public float  stopGoPeriodS    = 10.0f;
    public float  stopGoTopSpeedMps = 20.0f;

    // ── Runtime ──────────────────────────────────────────────────────────────
    private float _elapsed     = 0.0f;
    private float _currentSpeed = 0.0f;

    private const float HardBrakeDecelMps2  = 3.0f;
    private const float AccelAwayAccelMps2  = 1.5f;

    public void Initialise(LeadVehicleConfig cfg)
    {
        profileType        = cfg.speedProfileType;
        speedMps           = cfg.speedMps;
        brakeAtTimeS       = cfg.brakeAtTimeS;
        brakeToSpeedMps    = cfg.brakeToSpeedMps;
        stopGoPeriodS      = Mathf.Max(0.5f, cfg.stopGoPeriodS);
        stopGoTopSpeedMps  = cfg.stopGoTopSpeedMps;
        _elapsed           = 0.0f;
        _currentSpeed      = speedMps;
    }

    public float GetSpeed(float dt)
    {
        _elapsed += dt;

        switch (profileType)
        {
            case "constant":
            case "slower":
                _currentSpeed = speedMps;
                break;

            case "hard_brake":
                if (_elapsed < brakeAtTimeS)
                {
                    _currentSpeed = speedMps;
                }
                else
                {
                    float target = Mathf.Max(brakeToSpeedMps, 0.0f);
                    _currentSpeed = Mathf.MoveTowards(_currentSpeed, target,
                                                      HardBrakeDecelMps2 * dt);
                }
                break;

            case "accel_away":
                _currentSpeed = Mathf.MoveTowards(
                    _currentSpeed,
                    stopGoTopSpeedMps,
                    AccelAwayAccelMps2 * dt
                );
                break;

            case "stop_go":
            {
                // Smooth sinusoidal: speed = topSpeed * 0.5 * (1 − cos(2π t / T))
                // At t=0: speed=0; at t=T/2: speed=topSpeed; at t=T: speed=0
                float phase = (_elapsed % stopGoPeriodS) / stopGoPeriodS;
                _currentSpeed = stopGoTopSpeedMps * 0.5f *
                                (1.0f - Mathf.Cos(2.0f * Mathf.PI * phase));
                break;
            }

            default:
                _currentSpeed = speedMps;
                break;
        }

        return Mathf.Max(0.0f, _currentSpeed);
    }
}
