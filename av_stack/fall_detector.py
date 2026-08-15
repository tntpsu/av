"""
Vehicle-fell-off-the-world detection.

WHY THIS EXISTS
---------------
Tracks are a bare `MeshCollider` exactly `roadWidth` (7.2 m) wide, generated from
edges at +/-3.6 m. There is no terrain, ground plane, or shoulder beside it — past
the edge is void. A vehicle that leaves the road does not skid onto grass; it
falls.

This has happened at least once: `recording_20260506_042710.h5` (H9, 2026-05-06)
records pos.y going 0.80 -> -45.93 m and never recovering, with 12.5 m lateral
offset, 172 deg roll, and 19% of frames with all four wheels off the ground.

Before this module nothing in the stack read `position.y` or
`wheel_contact_normal_y`. That run was scored as an out-of-lane event plus an
e-stop — the same category of deduction a car drifting 0.6 m wide would receive.
A 47 m fall and a lane wobble were indistinguishable.

DESIGN
------
Two independent signatures, either sufficient, both together conclusive:

  1. DESCENT RATE — position.y falls faster than any grade could produce. This
     is physics, not tuning: a legitimate descent is bounded by
     `max_speed x max_grade` = 25 m/s x 0.10 = 2.5 m/s, while free fall passes
     4 m/s in 0.41 s. Measured: the H9 fall reaches 47.65 m/s; the steepest
     validated grade (hill_highway, +/-5%) peaks at 0.72 m/s. An absolute-drop
     test does NOT work — hill tracks legitimately descend ~5 m and trip it.
  2. NO GROUND CONTACT — all four wheel contact normals read 0 for
     `airborne_frames` consecutive frames. Individual wheels lift routinely in
     cornering (~6% of wheel-samples), so this deliberately requires ALL four.

Used by:
  - `av_stack/orchestrator.py` at runtime, to end a run shortly after a fall
    rather than record a vehicle falling through the void for the full duration.
  - `tools/drive_summary_core.py` offline, as a Safety hard-zero: a run where the
    vehicle left the world is not a low score, it is an invalid run. This also
    protects the scoring pipeline, since a fallen car keeps producing lateral
    error numbers that would otherwise be averaged in as if they meant something.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

# Descent rate above any achievable grade. max_speed x max_grade = 25 x 0.10 =
# 2.5 m/s is the physical ceiling for a legitimate descent; free fall passes 4 m/s
# in 0.41 s. Measured: H9 fall 47.65 m/s vs steepest grade 0.72 m/s.
DEFAULT_DESCENT_RATE_MPS: float = 4.0
# Consecutive frames above the descent-rate threshold before declaring a fall,
# so a single noisy sample cannot trigger it.
DEFAULT_DESCENT_FRAMES: int = 3
# Retained for reporting only — cumulative drop is informative but cannot be a
# trigger, because graded tracks legitimately descend several metres.
DEFAULT_DROP_THRESHOLD_M: float = 2.0
# All four wheels off the ground for this many consecutive frames. At ~13 FPS
# that is ~0.4 s — longer than any legitimate crest or suspension unload.
DEFAULT_AIRBORNE_FRAMES: int = 5
# How long to keep recording after a fall is confirmed, so the event itself is
# captured in the recording rather than truncated at the instant of detection.
DEFAULT_POST_FALL_GRACE_S: float = 2.0


@dataclass
class FallState:
    fallen: bool = False
    reason: Optional[str] = None
    frame: Optional[int] = None
    time_s: Optional[float] = None
    drop_m: float = 0.0
    airborne_run: int = 0
    descent_run: int = 0
    reference_y: Optional[float] = None
    max_drop_m: float = 0.0


@dataclass
class FallDetector:
    """Streaming detector for the runtime loop.

    Triggers on descent RATE, not absolute drop. Graded tracks legitimately lose
    several metres of altitude, so an absolute threshold false-positives on
    hill_highway and hill_g1; a rate threshold does not, because no grade at any
    drivable speed can produce free-fall velocity.
    """

    descent_rate_mps: float = DEFAULT_DESCENT_RATE_MPS
    descent_frames: int = DEFAULT_DESCENT_FRAMES
    airborne_frames: int = DEFAULT_AIRBORNE_FRAMES
    state: FallState = field(default_factory=FallState)
    _prev_y: Optional[float] = None
    _prev_t: Optional[float] = None

    def update(
        self,
        position_y: Optional[float],
        wheel_contact_normals: Optional[Sequence[float]] = None,
        frame: Optional[int] = None,
        time_s: Optional[float] = None,
    ) -> FallState:
        s = self.state
        if s.fallen:
            return s


        # ── 1. descent rate beyond any achievable grade ─────────────────────
        if position_y is not None:
            try:
                y = float(position_y)
            except (TypeError, ValueError):
                y = None
            if y is not None and y == y:  # not NaN
                if s.reference_y is None:
                    s.reference_y = y
                s.max_drop_m = max(s.max_drop_m, s.reference_y - y)
                s.drop_m = s.reference_y - y
                if self._prev_y is not None and time_s is not None and self._prev_t is not None:
                    dt = float(time_s) - float(self._prev_t)
                    if dt > 0:
                        descent = (self._prev_y - y) / dt      # positive = falling
                        if descent > self.descent_rate_mps:
                            s.descent_run += 1
                        else:
                            s.descent_run = 0
                        if s.descent_run >= self.descent_frames:
                            s.fallen = True
                            s.reason = (
                                f"free_fall — descending {descent:.1f} m/s for "
                                f"{s.descent_run} frames (grade ceiling "
                                f"{self.descent_rate_mps:.1f} m/s), dropped "
                                f"{s.max_drop_m:.1f} m"
                            )
                            s.frame, s.time_s = frame, time_s
                            return s
                self._prev_y, self._prev_t = y, time_s

        # ── 2. all four wheels off the ground ───────────────────────────────
        if wheel_contact_normals is not None:
            try:
                normals = [float(v) for v in wheel_contact_normals]
            except (TypeError, ValueError):
                normals = []
            if normals:
                if max(normals) <= 0.0:
                    s.airborne_run += 1
                else:
                    s.airborne_run = 0
                if s.airborne_run >= self.airborne_frames:
                    s.fallen = True
                    s.reason = (
                        f"no_ground_contact — all wheels off for "
                        f"{s.airborne_run} consecutive frames"
                    )
                    s.frame, s.time_s = frame, time_s
                    return s
        return s


def detect_fall_offline(
    position_y: Iterable[float],
    wheel_contact_normal_y=None,
    timestamps: Optional[Iterable[float]] = None,
    descent_rate_mps: float = DEFAULT_DESCENT_RATE_MPS,
    airborne_frames: int = DEFAULT_AIRBORNE_FRAMES,
) -> FallState:
    """Batch equivalent for scoring an existing recording.

    `wheel_contact_normal_y` may be (n, 4) or None.
    """
    import numpy as np

    y = np.asarray(list(position_y), dtype=float)
    if y.size == 0:
        return FallState()
    contacts = None
    if wheel_contact_normal_y is not None:
        c = np.asarray(wheel_contact_normal_y, dtype=float)
        if c.ndim == 1 and y.size and c.size % y.size == 0:
            c = c.reshape(y.size, -1)
        if c.ndim == 2 and c.shape[0] == y.size:
            contacts = c

    ts = np.asarray(list(timestamps), dtype=float) if timestamps is not None else None
    det = FallDetector(descent_rate_mps=descent_rate_mps, airborne_frames=airborne_frames)
    for i in range(y.size):
        det.update(
            y[i],
            contacts[i] if contacts is not None else None,
            frame=i,
            time_s=float(ts[i] - ts[0]) if ts is not None and ts.size == y.size else None,
        )
        if det.state.fallen:
            break
    return det.state
