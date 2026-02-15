#!/usr/bin/env python3
"""
Stage-5 cross-layer counterfactual evaluator.

This script runs (or ingests) trajectory-lock and control-lock matrices,
folds both comparators into one report, and produces an upstream/downstream
attribution scorecard.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module_func(module_relpath: str, fn_name: str):
    module_path = REPO_ROOT / module_relpath
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise RuntimeError(f"Function {fn_name} not found in {module_path}")
    return fn


replay_trajectory_locked = _load_module_func(
    "tools/analyze/replay_trajectory_locked.py",
    "replay_trajectory_locked",
)
replay_control_locked = _load_module_func(
    "tools/analyze/replay_control_locked.py",
    "replay_control_locked",
)

DEFAULT_SEGMENTATION_CHECKPOINT = (
    REPO_ROOT / "data" / "segmentation_dataset" / "checkpoints" / "segnet_best.pt"
)


def _f(v: Any) -> float:
    return float(v) if v is not None else 0.0


def _safe_ratio(num: float, den: float, eps: float = 1e-6) -> float:
    return float(num) / max(float(den), eps)


def _trajectory_sensitivity(
    self_a: Dict[str, Any],
    self_b: Dict[str, Any],
    cross_ab: Dict[str, Any],
    cross_ba: Dict[str, Any],
    strong_threshold: float,
    moderate_threshold: float,
) -> Dict[str, Any]:
    self_a_mean = _f(self_a.get("steering_abs_diff_mean_vs_source"))
    self_b_mean = _f(self_b.get("steering_abs_diff_mean_vs_source"))
    self_a_p95 = _f(self_a.get("steering_abs_diff_p95_vs_source"))
    self_b_p95 = _f(self_b.get("steering_abs_diff_p95_vs_source"))
    cross_ab_mean = _f(cross_ab.get("steering_abs_diff_mean_vs_source"))
    cross_ba_mean = _f(cross_ba.get("steering_abs_diff_mean_vs_source"))
    cross_ab_p95 = _f(cross_ab.get("steering_abs_diff_p95_vs_source"))
    cross_ba_p95 = _f(cross_ba.get("steering_abs_diff_p95_vs_source"))

    amp_a_mean = _safe_ratio(cross_ab_mean, self_a_mean)
    amp_b_mean = _safe_ratio(cross_ba_mean, self_b_mean)
    amp_a_p95 = _safe_ratio(cross_ab_p95, self_a_p95)
    amp_b_p95 = _safe_ratio(cross_ba_p95, self_b_p95)
    amp_mean_avg = (amp_a_mean + amp_b_mean) / 2.0
    amp_p95_avg = (amp_a_p95 + amp_b_p95) / 2.0

    if (
        amp_a_mean >= strong_threshold
        and amp_b_mean >= strong_threshold
        and amp_a_p95 >= strong_threshold
        and amp_b_p95 >= strong_threshold
    ):
        cls = "strong-trajectory-sensitivity"
    elif (
        amp_a_mean >= moderate_threshold
        and amp_b_mean >= moderate_threshold
    ) or (amp_a_p95 >= strong_threshold or amp_b_p95 >= strong_threshold):
        cls = "moderate-trajectory-sensitivity"
    else:
        cls = "low-trajectory-sensitivity"

    return {
        "classification": cls,
        "amplification": {
            "a_mean_ratio": amp_a_mean,
            "b_mean_ratio": amp_b_mean,
            "mean_ratio_avg": amp_mean_avg,
            "a_p95_ratio": amp_a_p95,
            "b_p95_ratio": amp_b_p95,
            "p95_ratio_avg": amp_p95_avg,
        },
        "thresholds": {
            "strong": float(strong_threshold),
            "moderate": float(moderate_threshold),
        },
    }


def _control_sensitivity(
    self_a: Dict[str, Any],
    self_b: Dict[str, Any],
    cross_ab: Dict[str, Any],
    cross_ba: Dict[str, Any],
    fidelity_threshold: float,
    steer_mean_strong: float,
    steer_p95_strong: float,
) -> Dict[str, Any]:
    fidelity_ok = all(
        _f(v) <= fidelity_threshold
        for v in [
            self_a.get("steering_abs_diff_mean_vs_lock"),
            self_a.get("throttle_abs_diff_mean_vs_lock"),
            self_a.get("brake_abs_diff_mean_vs_lock"),
            self_b.get("steering_abs_diff_mean_vs_lock"),
            self_b.get("throttle_abs_diff_mean_vs_lock"),
            self_b.get("brake_abs_diff_mean_vs_lock"),
            cross_ab.get("steering_abs_diff_mean_vs_lock"),
            cross_ab.get("throttle_abs_diff_mean_vs_lock"),
            cross_ab.get("brake_abs_diff_mean_vs_lock"),
            cross_ba.get("steering_abs_diff_mean_vs_lock"),
            cross_ba.get("throttle_abs_diff_mean_vs_lock"),
            cross_ba.get("brake_abs_diff_mean_vs_lock"),
        ]
    )

    cross_steer_mean = (
        _f(cross_ab.get("steering_abs_diff_mean_vs_input_source"))
        + _f(cross_ba.get("steering_abs_diff_mean_vs_input_source"))
    ) / 2.0
    cross_steer_p95 = (
        _f(cross_ab.get("steering_abs_diff_p95_vs_input_source"))
        + _f(cross_ba.get("steering_abs_diff_p95_vs_input_source"))
    ) / 2.0
    cross_throttle_mean = (
        _f(cross_ab.get("throttle_abs_diff_mean_vs_input_source"))
        + _f(cross_ba.get("throttle_abs_diff_mean_vs_input_source"))
    ) / 2.0
    cross_brake_mean = (
        _f(cross_ab.get("brake_abs_diff_mean_vs_input_source"))
        + _f(cross_ba.get("brake_abs_diff_mean_vs_input_source"))
    ) / 2.0

    if cross_steer_mean >= steer_mean_strong and cross_steer_p95 >= steer_p95_strong:
        cls = "strong-control-sensitivity"
    elif cross_steer_mean >= (0.5 * steer_mean_strong):
        cls = "moderate-control-sensitivity"
    else:
        cls = "low-control-sensitivity"

    return {
        "classification": cls,
        "fidelity_ok": bool(fidelity_ok),
        "fidelity_threshold": float(fidelity_threshold),
        "cross_shift": {
            "steering_mean_vs_input_source_avg": cross_steer_mean,
            "steering_p95_vs_input_source_avg": cross_steer_p95,
            "throttle_mean_vs_input_source_avg": cross_throttle_mean,
            "brake_mean_vs_input_source_avg": cross_brake_mean,
        },
        "thresholds": {
            "steer_mean_strong": float(steer_mean_strong),
            "steer_p95_strong": float(steer_p95_strong),
        },
    }


def _scorecard(
    trajectory_cmp: Dict[str, Any],
    control_cmp: Dict[str, Any],
    counterfactuals: Dict[str, Any],
) -> Dict[str, Any]:
    traj_cls = trajectory_cmp["classification"]
    ctrl_cls = control_cmp["classification"]
    traj_strength = float(trajectory_cmp["amplification"]["mean_ratio_avg"])
    ctrl_strength = float(control_cmp["cross_shift"]["steering_mean_vs_input_source_avg"])
    ctrl_fidelity_ok = bool(control_cmp["fidelity_ok"])

    if (traj_cls.startswith("strong") or traj_cls.startswith("moderate")) and ctrl_cls.startswith("low"):
        primary = "upstream-trajectory-dominant"
    elif (ctrl_cls.startswith("strong") or ctrl_cls.startswith("moderate")) and traj_cls.startswith("low"):
        primary = "downstream-control-dominant"
    elif ("strong" in traj_cls or "moderate" in traj_cls) and (
        "strong" in ctrl_cls or "moderate" in ctrl_cls
    ):
        primary = "mixed-cross-layer-coupling"
    else:
        primary = "low-isolation-signal"

    confidence = "low"
    if ctrl_fidelity_ok and primary != "low-isolation-signal":
        if abs(traj_strength - ctrl_strength) > 0.35:
            confidence = "high"
        else:
            confidence = "medium"

    recommendations = []
    if not ctrl_fidelity_ok:
        recommendations.append(
            "Fix control-lock fidelity first; safety overrides are masking attribution."
        )
    if primary == "upstream-trajectory-dominant":
        recommendations.append(
            "Prioritize trajectory generation and reference stability before control retuning."
        )
    elif primary == "downstream-control-dominant":
        recommendations.append(
            "Prioritize control policy/limiter tuning before upstream model changes."
        )
    elif primary == "mixed-cross-layer-coupling":
        recommendations.append(
            "Use constrained two-factor sweeps and promote only if both layers improve."
        )
    else:
        recommendations.append(
            "Increase scenario separation (stronger A/B pair) and rerun counterfactuals."
        )

    recommendations.append(
        "Gate promotion with repeated A/B significance and rollback criteria."
    )

    return {
        "primary_call": primary,
        "confidence": confidence,
        "trajectory_sensitivity": traj_cls,
        "control_sensitivity": ctrl_cls,
        "counterfactual_markers": counterfactuals,
        "recommendations": recommendations,
    }


def _run_counterfactuals(
    baseline: Path,
    treatment: Path,
    output_prefix: str,
    disable_vehicle_frame_lookahead_ref: bool,
    use_segmentation: bool,
    segmentation_checkpoint: str | None,
) -> Dict[str, Any]:
    # Stage-2 matrix
    tr_self_a = replay_trajectory_locked(
        input_recording=baseline,
        lock_recording=baseline,
        output_name=f"{output_prefix}_traj_self_a",
        use_segmentation=bool(use_segmentation),
        segmentation_checkpoint=segmentation_checkpoint,
        disable_vehicle_frame_lookahead_ref=disable_vehicle_frame_lookahead_ref,
    )
    tr_self_b = replay_trajectory_locked(
        input_recording=treatment,
        lock_recording=treatment,
        output_name=f"{output_prefix}_traj_self_b",
        use_segmentation=bool(use_segmentation),
        segmentation_checkpoint=segmentation_checkpoint,
        disable_vehicle_frame_lookahead_ref=disable_vehicle_frame_lookahead_ref,
    )
    tr_cross_ab = replay_trajectory_locked(
        input_recording=baseline,
        lock_recording=treatment,
        output_name=f"{output_prefix}_traj_cross_a_lock_b",
        use_segmentation=bool(use_segmentation),
        segmentation_checkpoint=segmentation_checkpoint,
        disable_vehicle_frame_lookahead_ref=disable_vehicle_frame_lookahead_ref,
    )
    tr_cross_ba = replay_trajectory_locked(
        input_recording=treatment,
        lock_recording=baseline,
        output_name=f"{output_prefix}_traj_cross_b_lock_a",
        use_segmentation=bool(use_segmentation),
        segmentation_checkpoint=segmentation_checkpoint,
        disable_vehicle_frame_lookahead_ref=disable_vehicle_frame_lookahead_ref,
    )

    # Stage-3 matrix
    ctrl_self_a = replay_control_locked(
        input_recording=baseline,
        lock_recording=baseline,
        output_name=f"{output_prefix}_ctrl_self_a",
        use_segmentation=bool(use_segmentation),
        segmentation_checkpoint=segmentation_checkpoint,
    )
    ctrl_self_b = replay_control_locked(
        input_recording=treatment,
        lock_recording=treatment,
        output_name=f"{output_prefix}_ctrl_self_b",
        use_segmentation=bool(use_segmentation),
        segmentation_checkpoint=segmentation_checkpoint,
    )
    ctrl_cross_ab = replay_control_locked(
        input_recording=baseline,
        lock_recording=treatment,
        output_name=f"{output_prefix}_ctrl_cross_a_lock_b",
        use_segmentation=bool(use_segmentation),
        segmentation_checkpoint=segmentation_checkpoint,
    )
    ctrl_cross_ba = replay_control_locked(
        input_recording=treatment,
        lock_recording=baseline,
        output_name=f"{output_prefix}_ctrl_cross_b_lock_a",
        use_segmentation=bool(use_segmentation),
        segmentation_checkpoint=segmentation_checkpoint,
    )

    # Stage-5 explicit swaps
    swap_base_perc_treat_ctrl = ctrl_cross_ab
    swap_treat_perc_base_ctrl = ctrl_cross_ba
    swap_traj_then_ctrl = replay_control_locked(
        input_recording=Path(tr_cross_ba["output_recording"]),
        lock_recording=treatment,
        output_name=f"{output_prefix}_swap_treat_perc_base_traj_treat_ctrl",
        use_segmentation=bool(use_segmentation),
        segmentation_checkpoint=segmentation_checkpoint,
    )

    return {
        "trajectory": {
            "self_a": tr_self_a,
            "self_b": tr_self_b,
            "cross_a_lock_b": tr_cross_ab,
            "cross_b_lock_a": tr_cross_ba,
        },
        "control": {
            "self_a": ctrl_self_a,
            "self_b": ctrl_self_b,
            "cross_a_lock_b": ctrl_cross_ab,
            "cross_b_lock_a": ctrl_cross_ba,
        },
        "counterfactual_swaps": {
            "baseline_perception_treatment_control": swap_base_perc_treat_ctrl,
            "treatment_perception_baseline_control": swap_treat_perc_base_ctrl,
            "baseline_trajectory_treatment_control": swap_traj_then_ctrl,
            "supporting_treatment_perception_baseline_trajectory": tr_cross_ba,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run Stage-5 cross-layer counterfactual layer-swap evaluator."
    )
    p.add_argument("baseline_recording", help="Baseline recording path (.h5)")
    p.add_argument("treatment_recording", help="Treatment recording path (.h5)")
    p.add_argument("--output-prefix", default="stage5_counterfactual")
    p.add_argument("--output-json", default=None)
    p.add_argument("--disable-vehicle-frame-lookahead-ref", action="store_true")
    p.add_argument(
        "--use-cv",
        action="store_true",
        help="Force CV-based perception (override default segmentation).",
    )
    p.add_argument(
        "--segmentation-checkpoint",
        default=str(DEFAULT_SEGMENTATION_CHECKPOINT),
        help="Segmentation checkpoint path (default: segnet_best.pt).",
    )
    p.add_argument("--trajectory-strong-threshold", type=float, default=2.0)
    p.add_argument("--trajectory-moderate-threshold", type=float, default=1.5)
    p.add_argument("--control-fidelity-threshold", type=float, default=1e-4)
    p.add_argument("--control-steer-mean-strong", type=float, default=0.25)
    p.add_argument("--control-steer-p95-strong", type=float, default=0.8)
    args = p.parse_args()

    baseline = Path(args.baseline_recording)
    treatment = Path(args.treatment_recording)
    if not baseline.exists():
        raise FileNotFoundError(f"Baseline recording not found: {baseline}")
    if not treatment.exists():
        raise FileNotFoundError(f"Treatment recording not found: {treatment}")
    use_segmentation = not bool(args.use_cv)
    if use_segmentation and not Path(args.segmentation_checkpoint).exists():
        raise FileNotFoundError(
            f"Segmentation checkpoint not found: {args.segmentation_checkpoint}. "
            "Provide --segmentation-checkpoint or run with --use-cv."
        )

    runs = _run_counterfactuals(
        baseline=baseline,
        treatment=treatment,
        output_prefix=args.output_prefix,
        disable_vehicle_frame_lookahead_ref=bool(args.disable_vehicle_frame_lookahead_ref),
        use_segmentation=use_segmentation,
        segmentation_checkpoint=args.segmentation_checkpoint,
    )

    trajectory_cmp = _trajectory_sensitivity(
        self_a=runs["trajectory"]["self_a"],
        self_b=runs["trajectory"]["self_b"],
        cross_ab=runs["trajectory"]["cross_a_lock_b"],
        cross_ba=runs["trajectory"]["cross_b_lock_a"],
        strong_threshold=float(args.trajectory_strong_threshold),
        moderate_threshold=float(args.trajectory_moderate_threshold),
    )
    control_cmp = _control_sensitivity(
        self_a=runs["control"]["self_a"],
        self_b=runs["control"]["self_b"],
        cross_ab=runs["control"]["cross_a_lock_b"],
        cross_ba=runs["control"]["cross_b_lock_a"],
        fidelity_threshold=float(args.control_fidelity_threshold),
        steer_mean_strong=float(args.control_steer_mean_strong),
        steer_p95_strong=float(args.control_steer_p95_strong),
    )

    counterfactual_markers = {
        "baseline_perception_treatment_control_steer_mean_vs_input": _f(
            runs["counterfactual_swaps"]["baseline_perception_treatment_control"].get(
                "steering_abs_diff_mean_vs_input_source"
            )
        ),
        "treatment_perception_baseline_control_steer_mean_vs_input": _f(
            runs["counterfactual_swaps"]["treatment_perception_baseline_control"].get(
                "steering_abs_diff_mean_vs_input_source"
            )
        ),
        "treatment_perception_baseline_trajectory_steer_mean_vs_source": _f(
            runs["counterfactual_swaps"]["supporting_treatment_perception_baseline_trajectory"].get(
                "steering_abs_diff_mean_vs_source"
            )
        ),
    }
    scorecard = _scorecard(
        trajectory_cmp=trajectory_cmp,
        control_cmp=control_cmp,
        counterfactuals=counterfactual_markers,
    )

    out = {
        "baseline_recording": str(baseline),
        "treatment_recording": str(treatment),
        "trajectory_sensitivity": trajectory_cmp,
        "control_sensitivity": control_cmp,
        "upstream_downstream_attribution_scorecard": scorecard,
        "counterfactual_swaps": runs["counterfactual_swaps"],
        "matrix_runs": {
            "trajectory": runs["trajectory"],
            "control": runs["control"],
        },
    }
    print(json.dumps(out, indent=2))

    output_json = (
        Path(args.output_json)
        if args.output_json
        else REPO_ROOT / "tmp" / "analysis" / f"{args.output_prefix}_report.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

