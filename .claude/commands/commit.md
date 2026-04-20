Chain the post-commit rituals after a git commit has just landed. This skill does NOT do the commit itself — it runs the follow-up steps that are easy to forget, driven by what the commit actually touched.

The user's request is: $ARGUMENTS

## When to run this skill

Immediately after a commit lands, BEFORE moving to the next task. The whole point is to close the loop on the commit you just made while the context is fresh.

If you find yourself several commits later realizing you never ran this, run it against the specific SHA: `/commit <sha>` or `/commit HEAD~2`.

## Step 1 — Identify the commit

Default: `HEAD` (the commit that just landed).

`$ARGUMENTS` parsing:
- Empty → operate on `HEAD`
- A SHA or `HEAD~N` → operate on that commit
- `--retrospective` flag (alone or combined with a SHA) → skip journal-writing rituals (`/log-fix`, `/log-experiment`). Doc + validation rituals (`/update-arch`, `/update-odd`, comfort-gate) still run.

Rationale for `--retrospective`: if a commit landed >24h ago, or the relevant journal entry was already logged by a different flow, writing a fresh `/log-fix` or `/log-experiment` entry pollutes the Pareto (double-counts the outcome, misattributes the date). The doc/validation rituals remain valuable regardless of age because they check current state, not historical intent.

Capture:
- Commit SHA
- Commit message (subject + body)
- Commit age: `git show -s --format=%ct <sha>` vs `date +%s` (seconds since commit)
- List of files changed: `git show --stat --name-only <sha>`
- Full diff summary: `git show --stat <sha>`

**Auto-detect retrospective mode:** if commit age > 86400 (24h) and no explicit `--retrospective` flag was passed, ANNOUNCE the auto-upgrade in the report — do not silently skip. The user may still want journal entries for recent-but-not-fresh work.

## Step 2 — Classify the commit

Based on the subject line prefix and body keywords, classify:

| Prefix / keywords | Category |
|---|---|
| `fix:` / `fix(...)` / "fixes" / "resolves" / "bug" | **FIX** — run `/log-fix` |
| `feat:` / `feat(...)` / "add" / "introduce" / "land" | **FEATURE** — consider `/log-experiment` if this activated a new approach |
| `refactor:` / `chore:` / `docs:` / `test:` | **STRUCTURAL** — usually no rituals, but check files touched |
| `revert` / "disable" / "remove" / "back out" | **REGRESSION-PROTECTION** — `/log-experiment` (failure category), possibly `/update-odd` if a capability went away |
| Multiple categories (bundled commit) | Run multiple rituals; flag the bundle — prefer single-concern commits |

## Step 3 — File-touched triggers

For each file touched in the diff, fire the matching ritual:

| Files touched | Ritual to trigger |
|---|---|
| `control/pid_controller.py`, `av_stack/orchestrator.py`, `control/regime_selector.py`, `control/mpc_controller.py`, `trajectory/inference.py`, `trajectory/utils.py` | Recommend `/update-arch` — the pipeline diagram in `docs/agent/architecture.md` may be stale |
| `config/av_stack_config.yaml` (target_speed, regime selector keys, controller enablement flags, new capability gates) | Recommend `/update-odd` — capability envelope may have shifted. Also run `python tools/ci/check_config_regression.py --base HEAD~1 --critical-only` to surface scoring-critical param changes |
| `tools/drive_summary_core.py`, `tools/analyze/run_gate_and_triage.py`, `data/recorder.py`, `tools/scoring_registry.py` | REQUIRE running Tier 1 comfort gate tests per CLAUDE.md: `pytest tests/test_comfort_gate_replay.py -v -k "Synthetic or Boundary"` |
| Any file above + config change | Run scoring regression suite: `pytest tests/test_scoring_regression.py -v` |
| Unity `.cs` files under `unity/AVSimulation/Assets/Scripts/` | Flag: Unity player rebuild needed before next `/e2e`; note in output |
| `tests/fixtures/*.json`, `tests/conftest.py` | Baselines changed — verify re-freeze was intentional (cross-check commit message) |
| `docs/agent/*.md`, `docs/ODD.md`, `docs/CONFIG_GUIDE.md`, `CLAUDE.md` | Documentation-only — no further rituals |
| `.claude/commands/*.md` | New skill / skill edit — note for MEMORY.md cross-reference if this creates a new capability pattern |

## Step 4 — Run the rituals

For each ritual identified in Steps 2–3:

1. Announce what's being run and why (link back to the trigger that fired it)
2. Invoke the skill (e.g., `/log-fix`, `/update-arch`, `/update-odd`, `/log-experiment`)
3. Capture each skill's completion status
4. If a skill surfaces new findings (e.g., `/update-odd` reveals an inconsistency), note them for the report

Order matters:
1. `/log-fix` or `/log-experiment` first — these feed `/process-health` and should happen while context is fresh
2. `/update-arch` next if applicable — internal architecture truth
3. `/update-odd` last if applicable — external contract / capability claim (depends on arch being current)

### Retrospective-mode ritual filter

If `--retrospective` is active (either explicit or auto-detected per Step 1):

| Ritual | Runs in retrospective? | Reason |
|---|---|---|
| `/log-fix` | **SKIP** | Journal entry already exists or too stale to attribute; double-counting pollutes the Pareto. |
| `/log-experiment` | **SKIP** | Same. Use `/log-experiment` directly with `outcome_status` updates instead. |
| `/update-arch` | RUN | Checks current state, not historical intent. |
| `/update-odd` | RUN | Same. |
| comfort-gate / scoring regression | RUN | Validation is a snapshot, always applicable. |
| Unity rebuild flag | RUN | Current reality check. |

Each skipped ritual MUST be reported with the reason "skipped: retrospective mode (commit age: Xh)" — silent skips mask bugs in the trigger rules (per anti-pattern below).

## Step 5 — Validation gates

If Tier 1 comfort gate or scoring regression was required (Step 3), run it and fail loudly if red:

```bash
pytest tests/test_comfort_gate_replay.py -v -k "Synthetic or Boundary"
pytest tests/test_scoring_regression.py -v
```

If red: do NOT continue silently. Report the failure and recommend:
- Option A: revert the commit and re-do correctly
- Option B: fix the break and create a NEW commit (per CLAUDE.md commit protocol, NEVER amend)

## Step 6 — Report

Summarize what ran:

```
POST-COMMIT CHAIN — <sha_short>
═══════════════════════════════════════════════════════════
Commit: <subject line>
Category: <FIX / FEATURE / STRUCTURAL / REGRESSION-PROTECTION>
Files touched: <N files> (<key files>)

Rituals run:
  ✓ /log-fix       — <what was logged, if anything>
  ✓ /update-arch   — <sections updated, or "no changes needed">
  ○ /update-odd    — SKIPPED (reason: no capability envelope change)
  ✓ comfort-gate   — PASS / FAIL (<details>)

Rituals deferred (manual follow-up):
  - <ritual>: <why it was deferred, e.g., "Unity rebuild needed before /e2e">

Next recommended action: <skill to run next, or "work continues">
═══════════════════════════════════════════════════════════
```

## Anti-patterns

- **Don't re-commit inside this skill.** If a ritual (e.g., `/update-arch`) creates doc changes, those are separate follow-up commits. This skill is a chain, not a bundler.
- **Don't silently skip rituals.** If a trigger fires and the ritual isn't applicable, say so in the report with reasoning. Silent skips mask bugs in the trigger rules.
- **Don't run `/e2e` or `/sweep` here.** Validation burn cycles belong to `/iterate` and friends. This skill is lightweight bookkeeping.
- **Don't amend the commit you just made.** Per CLAUDE.md, never amend — any discovered issue becomes a NEW commit.

## What this skill replaces

Before this skill existed, the workflow was:
1. Agent commits
2. Agent may or may not remember CLAUDE.md line 76 about `/log-fix`
3. Agent may or may not remember to update `docs/agent/architecture.md`
4. Agent may or may not remember to update `docs/ODD.md`
5. Tier 1 comfort gate tests often skipped for "non-pipeline" changes that turned out to touch the pipeline

With this skill: all five of those "may or may not" items become deterministic based on file-touched triggers.

## References

- Log skills: `/log-fix`, `/log-experiment`
- Doc skills: `/update-arch`, `/update-odd`
- Testing protocol: `CLAUDE.md` § Testing Protocol — Comfort Gate Regression, § Testing Protocol — Scoring Regression
- Commit protocol: `CLAUDE.md` § Git Safety Protocol (built-in Claude Code behavior handles the commit itself)
- Process health: `/process-health` consumes `/log-fix` and `/log-experiment` entries
