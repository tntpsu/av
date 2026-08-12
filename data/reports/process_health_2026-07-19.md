# Process Health Report — 2026-07-19

**Log entries:** 14 | **Date range:** 2026-03-22 → 2026-04-17 | **Avg detection:** 2.3 iterations

---

## WHERE ISSUES ORIGINATE

```
WHERE ISSUES ORIGINATE
═══════════════════════════════════════════════════
Design          ████████████████████████  64%  (9)
Implementation  ████████                  21%  (3)
Testing         ███                        7%  (1)
Tooling         ███                        7%  (1)
Requirements                               0%  (0)
```

---

## DESIGN SUB-CATEGORIES

```
DESIGN SUB-CATEGORIES (9 entries)
═══════════════════════════════════════════════════
Proxy approximation     ████████████████  44%  (4)
Signal routing          ████               11%  (1)
Missing state machine   ████               11%  (1)
Temporal coupling       ████               11%  (1)
Missing physics model   ████               11%  (1)
State discretization    ████               11%  (1)
```

Proxy approximation entries: lateral-accel budget (6 proxy params → κ×v²), PP floor (speed table → physics formula), entry arm (distance → time), velocity profile (scalar collapse → spatial profile).

---

## DETECTION EFFICIENCY

```
DETECTION EFFICIENCY
═══════════════════════════════════════════════════
Avg iterations to root cause:   2.3
Issues caught by tooling:       93%  (13/14)
Issues requiring manual trace:   7%  ( 1/14)
Trend: IMPROVING
  First-half avg:  3.1 iterations  (entries 1–7,  2026-03-22→2026-04-05)
  Second-half avg: 1.4 iterations  (entries 8–14, 2026-04-06→2026-04-17)
```

The one manual-trace entry was entry 12 (pre-computed velocity profile), caught by architecture analysis rather than a diagnostic tool.

---

## PREVENTION THEMES

```
PREVENTION THEMES
═══════════════════════════════════════════════════
Theme                        Count  Example prevention
Physics-first design rule        4  "Derive from physics; use tables only as overrides"
Add regression/contract test     3  "Assert corr(mpc_e_lat, -lateral_error) > 0.95 on curves"
Design-review gate               3  "Any binary threshold on continuous signal requires justification"
Input validation / plausibility  1  "GT lane boundaries need range filter before scoring"
Architecture rate-domain sep.    1  "Assert control update rate >= minimum Hz in acceptance tests"
```

---

## TOP 3 PROCESS IMPROVEMENTS

```
TOP 3 PROCESS IMPROVEMENTS
════════════════════════════════════════════════════

1. Physics-First Design Gate
   Pattern: 4/9 design issues (44%) were proxy approximations — independent
            thresholds or tables substituting for a single physics formula.
            Each required 1–5 iterations to diagnose.
   Action:  Add a design-review checklist item: "What physical quantity does
            this threshold approximate? Can we compute it directly?" Block
            any config param with >2 proxy variables without dimensional
            analysis in the commit body.
   Impact:  Eliminates the most common issue class at origin. Historical
            examples each recovered 3–20 pts and removed per-track overlays.

2. E2E Validation Gate Before Merging Numeric Constants
   Pattern: Entry 13 (mpc_q_lat bundle) recovered 82.5 pts — the single
            highest-impact fix in the log — from a 2-line tuning change
            bundled without E2E validation. Unit tests passed; only E2E
            would have caught the oversteer runaway.
   Action:  Pre-merge rule: any diff touching _MPC_WEIGHT_AUTO_DERIVE_PARAMS
            or av_stack_config.yaml tuning keys requires an E2E run record
            (highway_65 + one curve-heavy track) in the commit body. Gate
            can be a grep-based check in SCRIPT_RUNBOOK.md.
   Impact:  Prevents the highest-cost class of regression (80+ pt drops
            from undocumented tuning). Detection delay was 1 iteration but
            only because /iterate bisect was used; without that tool it
            would have been much longer.

3. Phase-Isolation Contract for All Steering Terms
   Pattern: 3 entries (phase-gated FF, far-preview leak, regime blend
            asymmetry) involved a steering term active outside its intended
            driving phase. Each required 1–2 iterations to diagnose with
            steering decomposition trace.
   Action:  Steering term code review rule: every new term must declare
            its active phase (ENTRY / SUSTAIN / EXIT / STRAIGHT) in a
            comment adjacent to the add. Contract test: assert term == 0
            on a straight-segment slice for any term not declared GLOBAL.
   Impact:  Closes a recurring implementation bug class (3 hits in 2 weeks)
            via review + regression. Steering decomposition trace already
            exists — only the contract test is missing.
```

---

## NOTES

- Log coverage ends 2026-04-17. `/log-fix` has not been run for 93 days
  (2026-04-17 → 2026-07-19). Recent work (velocity profile activation,
  NMPC, ACC sweep) is unlogged — Pareto may shift when those fixes are
  recorded.
- `sweeping_highway` golden recording is missing from disk; any sweep FAIL
  on that track cannot be logged until recording is regenerated.
