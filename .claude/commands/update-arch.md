Update the architecture documentation after code changes to the control pipeline.

The user's request is: $ARGUMENTS

## Step 1 — Identify what changed

Parse $ARGUMENTS or check `git diff` for changes to:
- `control/pid_controller.py` — PP/FF/floor pipeline
- `trajectory/utils.py` — curve phase scheduler, lookahead computation
- `av_stack/orchestrator.py` — reference point construction
- `control/regime_selector.py` — PP/MPC regime logic
- `control/mpc_controller.py` — MPC pipeline

## Step 2 — Read current architecture docs

```bash
cat docs/agent/architecture.md
```

Check which sections are affected by the changes.

## Step 3 — Update the architecture doc

Update the relevant sections in `docs/agent/architecture.md` to reflect the new behavior. Key sections to keep current:
- PP Lookahead Signal Chain (parameters, gates, data flow)
- Control pipeline (which signals gate which outputs)
- Component descriptions (if interfaces changed)

## Step 4 — Update current_state.md

Update `docs/agent/current_state.md` with:
- What changed and why
- Current track sweep results
- Remaining bottlenecks

## Step 5 — Verify consistency

Check that the architecture doc matches the actual code:
- Grep for key parameter names mentioned in the doc
- Verify signal flow matches code paths
- Flag any stale references

References:
- Architecture: `docs/agent/architecture.md`
- Current state: `docs/agent/current_state.md`
- Config: `config/av_stack_config.yaml`
- Scoring: `tools/scoring_registry.py`
