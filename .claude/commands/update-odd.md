Update the Operational Design Domain (ODD) document to reflect the current state of what the system is designed to do, validated against, and known to NOT handle.

The user's request is: $ARGUMENTS

## When to run this skill

- After a milestone validation (new track passed all-layers-≥95)
- After a capability expansion (new sensor, new scenario class, new control mode)
- After a capability contraction (controller disabled, scenario regressed, sensor removed)
- After a platform change (Unity version, OS, hardware)
- Periodically (monthly) to catch drift between code reality and the ODD claim

If you're unsure whether to run it: check the revision history at the bottom of `docs/ODD.md`. If the last entry is >30 days old or pre-dates recent MEMORY.md changes, it's time.

## Step 1 — Read the current ODD

```bash
cat docs/ODD.md
```

Note the date of the last revision history entry. Everything that changed after that date is a candidate for an update.

## Step 2 — Identify the delta

Pull the sources of truth:

1. **Recent memories** — anything in MEMORY.md dated after the last ODD entry
2. **Current state** — `docs/agent/current_state.md` (session log)
3. **Task log** — `docs/agent/tasks.md` (what was worked on / deferred)
4. **Config deltas** — `config/av_stack_config.yaml` (check for new keys, removed keys, default changes in the target_speed, regime selector, controller enablement flags)
5. **Known limitations** — search project memories for words like "DISABLED", "SUPERSEDED", "regressed", "failed activation"

Build a list of deltas categorized into:
- **Capability additions** (new track, new sensor, new mode)
- **Capability removals / regressions** (disabled controllers, regressed scenarios)
- **Platform changes** (Unity/OS upgrade)
- **Threshold / gate changes** (scoring thresholds, validated speeds)
- **Known-limitation additions** (new gotchas surfaced)

## Step 3 — Cross-check for internal inconsistencies

Before editing, read the full ODD once more looking for:
- Contradictions between sections (e.g., "Unity 6000.3" in one section, "Unity 2021.3" in another)
- References to components/flags that have since been renamed or removed
- Validated scores that are stale vs current baselines
- Controllers listed as active that are actually disabled

Every inconsistency found is a bug in the ODD and gets fixed in this pass.

## Step 4 — Update the document

Target sections to keep current:
- **System Capabilities** (what is actively supported)
- **Track Constraints** (geometry limits, speed range)
- **Sensor Assumptions** (what the system consumes)
- **Control Modes table** (active controllers, speed/curvature regimes, activation logic)
- **Environmental Constraints** (platform, traffic, weather)
- **Known Limitations** (numbered list of gotchas — grows over time)
- **Validated Tracks table** (with current scores if relevant)
- **Revision History** (append a new row — never rewrite old rows)

Rules:
- Prefer surgical edits over full rewrites; the revision history is the change log.
- Every deferred / disabled / regressed capability gets a memory file pointer (`See \`project_xxx.md\``).
- If a controller is disabled, annotate it in the Control Modes table — don't just silently remove the row.
- When adding to Known Limitations, number continues from existing list.

## Step 5 — Append revision history entry

Format: `| YYYY-MM-DD | <1-2 sentence summary of what changed in the system, and hence in this ODD update>. |`

Focus on the *system* change, not the ODD edits themselves ("NMPC disabled" not "Added NMPC-disabled note to ODD").

## Step 6 — Report

Summarize:
- N lines changed
- M known limitations added
- K capability changes (add/remove/regress)
- Any inconsistencies found and fixed (highlight these — they're the strongest signal that ODD drift was real)
- Next recommended run date (default: 30 days, or after next milestone, whichever comes first)

## Anti-patterns

- **Don't blanket-rewrite.** Surgical edits preserve the document's audit trail.
- **Don't infer capability state from code.** Capability state is a *claim* — check the memory / task log to confirm the team considers it in-ODD.
- **Don't remove limitations without replacement evidence.** A limitation leaves the list only when there's a memory / test / E2E proving it was resolved.
- **Don't forget the revision history.** Silent edits break the document's usefulness as a change log.

## References

- ODD: `docs/ODD.md`
- Current state: `docs/agent/current_state.md`
- Tasks: `docs/agent/tasks.md`
- Memory index: `~/.claude/projects/-Users-philtullai-av/memory/MEMORY.md`
- Architecture: `docs/agent/architecture.md` (sister skill: `/update-arch`)
