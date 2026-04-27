# Nightly Test-Fix Rubric

This file is read by the **Nightly Test Fix** routine
(`https://claude.ai/code/routines`). The routine prompt points here so
classification rules live in version control, not embedded in the trigger.

## Failure classification

For each failing test, **read the test code first**, then classify:

| Category        | Signal                                                         | Action                                |
|-----------------|----------------------------------------------------------------|---------------------------------------|
| STALE_BASELINE  | Hardcoded value intentionally changed in config/code           | Update the expected value ONLY        |
| STALE_IMPORT    | Imports a symbol that was renamed or removed                   | Update the import                     |
| OBSOLETE_TEST   | Exercises a removed feature (grep-confirm feature is gone)     | Delete the test function              |
| REAL_BREAK      | Assertion is correct, code is wrong                            | DO NOT MODIFY — report only           |
| FLAKY           | Intermittent / timing / float edge                             | Report only                           |

## Hard rules

- **Default to REAL_BREAK** when unsure. Never widen the safe-fix categories.
- **Never modify a test's assertion** to make a REAL_BREAK pass.
- **Never change test logic** — STALE_BASELINE only updates hardcoded
  expected values, not control flow or computation.
- **Never delete a test** without `grep`-confirming the feature is gone
  from the codebase. The grep evidence goes in the report.
- If ≥10 tests fail from the same root cause (e.g. a renamed import in a
  shared helper), fix the cause once rather than touching ten tests.
- **Do not modify production code** — only `tests/`, `tests/fixtures/*`,
  and `tests/conftest.py BASELINE_SCORES` are in scope. `src/`, `control/`,
  `trajectory/`, `perception/`, `av_stack.py`, and `config/` are off-limits.
- **Always re-run pytest after fixes** to verify they actually work
  before committing. A "fix" that doesn't make pytest green is not a fix.

## Files that may be edited

| Path                                  | Why                                       |
|---------------------------------------|-------------------------------------------|
| `tests/**/*.py`                       | Test code (STALE_IMPORT, OBSOLETE_TEST)   |
| `tests/fixtures/scoring_baselines.json` | STALE_BASELINE for scoring regression    |
| `tests/fixtures/golden_recordings.json` | STALE_BASELINE for comfort-gate replay   |
| `tests/conftest.py` (`BASELINE_SCORES` only) | STALE_BASELINE for overall scores   |

## Report format

Write `data/reports/nightly_test_report.txt`:

```
NIGHTLY TEST REPORT — <YYYY-MM-DD>
==================================
Total: <N>
Passed: <N>
Fixed: <N>
Real breaks (unfixed): <N>
Flaky: <N>

## FIXES APPLIED
  [CATEGORY] tests/path.py::ClassName::test_name
    — <one-line description of fix>

## REAL BREAKS (need human review)
  [CATEGORY] tests/path.py::ClassName::test_name
    — <one-line description of what broke>
    — first-failure message: <first line of pytest error>

## FLAKY
  [path::name] — <description>
```

If the suite is fully green, the report is one line:
`<YYYY-MM-DD> — PASS (<N> tests)`.
