# Reporting Contract Cleanup Plan

## Goal

Remove or demote stale reporting paths that still describe the stack as if legacy `curve_intent`
metrics are the primary lateral owner, and replace them with reporting that matches the current
authoritative path:

1. local curve scheduler / phase state
2. turn-in owner
3. local curve reference ownership
4. GT cross-track contract
5. transport contract

The target is simple: if a run is clean under the current architecture, the summary, triage, and
PhilViz output should say so directly without requiring manual reinterpretation.

## Why This Exists

The validated run `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_230150.h5`
scored `97.4`, yet still surfaced:

- `Curve intent late arm (0.0% early)`

That warning is now stale. The active lateral path is not primarily owned by legacy
`curve_intent` arm timing anymore. Mild-curve activation, sustain, and MPC bias were already
moved into newer scheduler/owner contracts, and the GT cross-track contract is now explicit.

## Current Stale or Misweighted Surfaces

### 1. Legacy curve-intent diagnostics still drive recommendations

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`

Symptoms:
- `Curve intent late arm`
- `Curve intent undercall`
- `Curve Intent Arm Early Rate`
- `Curve Intent Curvature Ratio`

Problem:
- these metrics are still framed as primary causal signals
- they are now, at best, compatibility/proxy metrics
- they should not dominate recommendations when the newer local/phase owner path is healthy

### 2. Curvature contract health still centers a legacy owner field

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`

Symptoms:
- `curve_intent_commit_streak_max_frames`

Problem:
- this field still implies legacy `curve_intent` COMMIT is the main contract-health owner
- current runtime behavior is better described through local scheduler state and turn-in owner

### 3. Analyzer/PhilViz naming still overstates `curve_intent`

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Problem:
- several labels still present `curve_intent` as the main steering activation story
- that makes clean runs look suspicious even when the active local contract is healthy

### 4. Compatibility metrics are not explicitly labeled as compatibility metrics

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Problem:
- some legacy fields are still useful as proxies
- but they are not labeled `proxy`, `legacy`, or `compatibility`
- that creates false confidence and noisy recommendations

### 5. Secondary metric-contract issue: ACC comfort jerk still looks misleading

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- ACC scoring path files after identification

Problem:
- latest clean run is behaviorally calm, but ACC comfort still fails on raw jerk
- filtered and commanded jerk are both reasonable
- this is likely a separate metric-contract cleanup, not the same legacy curve-intent issue

This should be treated as a separate follow-up, not mixed into the curve-owner cleanup.

## Cleanup Principles

1. Do not delete useful proxy metrics immediately.
2. First, relabel them as `legacy` or `proxy`.
3. Promote authoritative owner metrics in recommendations and key-issue generation.
4. Only remove legacy surfaces after parity and migration gates pass.

## Phase R1: Inventory and Ownership Tagging

Goal:
- classify every lateral-reporting surface as one of:
  - `authoritative`
  - `proxy`
  - `legacy`

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Implementation:
1. Add explicit ownership metadata in summary builders:
   - `curve_intent_diagnostics.owner_class = "legacy_proxy"`
   - `turn_in_owner.owner_class = "authoritative"`
   - `curve_local_contract.owner_class = "authoritative"`
   - `local_curve_reference.owner_class = "authoritative"`
2. Ensure summary text generation can branch on that ownership.

Acceptance:
- legacy/proxy metrics are machine-readable as such

## Phase R2: Recommendation and Key-Issue Reweighting

Goal:
- stop emitting stale high-priority recommendations from legacy metrics when authoritative contracts are healthy

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`

Implementation:
1. Gate `Curve intent late arm` recommendation behind:
   - local scheduler issue present, or
   - local owner unavailable
2. Gate curvature undercall recommendation behind:
   - authoritative curvature-owner issue present
3. If local scheduler/turn-in owner/GT contract are healthy, legacy `curve_intent` should not
   appear in top key issues.
4. Replace with:
   - `Local curve scheduler healthy`
   - `Turn-in owner healthy`
   - `GT cross-track contract healthy`
   when appropriate

Acceptance:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_230150.h5`
  no longer reports `Curve intent late arm` as a top issue

## Phase R3: Analyzer and PhilViz Relabeling

Goal:
- make the UI/report language reflect architecture truth

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Implementation:
1. Rename curve-intent section headings to indicate compatibility status where needed.
   - example: `Legacy Curve-Intent Proxy`
2. Add an explicit `Lateral Owner Contract` summary block that prioritizes:
   - local scheduler state
   - turn-in owner
   - local curve reference
   - GT cross-track contract
3. Demote legacy curve-intent rows below authoritative owner rows.

Acceptance:
- a clean run reads as clean without manual interpretation

## Phase R4: Contract Health Cleanup

Goal:
- remove misleading contract-health dependence on legacy `curve_intent` COMMIT streak

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`

Implementation:
1. Demote `curve_intent_commit_streak_max_frames` to legacy compatibility telemetry.
2. Replace top-level health emphasis with:
   - local state progression stability
   - turn-in owner continuity
   - local reference availability / fallback
3. Keep the old field only for back-compat and debugging.

Acceptance:
- contract health no longer implies the old owner is primary

## Phase R5: Post-Stabilization Removal

Goal:
- surgically remove stale legacy issue text once compatibility value is low

Implementation:
1. remove stale key-issue generation
2. remove stale recommendation text
3. keep raw telemetry only if still useful for debugging

Gate:
- multiple clean runs where authoritative owner metrics explain behavior without legacy help

## Tests

1. Summary contract tests
- `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`

2. Analyzer wording tests
- add/update tests for clean-run issue ordering and recommendation output

3. PhilViz contract tests
- `/Users/philiptullai/Documents/Coding/av/tests/test_philviz_mpc_endpoints.py`

4. Regression checks
- clean run:
  - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_230150.h5`
- old GT mismatch run:
  - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_224921.h5`

## Priority

1. `curve_intent` recommendation/key-issue demotion
2. owner-contract relabeling in analyzer/PhilViz
3. contract-health cleanup
4. legacy removal after multiple clean runs

## Answer To “Is That The Only One?”

No.

The primary stale reporting issue is legacy `curve_intent` being treated as authoritative.
But there is also a second, separate metric-contract problem:

- ACC comfort jerk currently looks overstated relative to the actual calm behavior of the latest run

That should be handled in a separate longitudinal metric review after the lateral-reporting cleanup.
