Fix broken tests by classifying each failure's root cause and applying the correct fix. Never silently delete a test that catches a real bug.

The user's request is: $ARGUMENTS

## Step 1 — Check: Run the test suite and capture failures

If $ARGUMENTS contains a specific test path or pattern, run only that:
```bash
python3 -m pytest $ARGUMENTS -v --tb=short 2>&1
```

Otherwise run the full suite:
```bash
python3 -m pytest tests/ -v --tb=short 2>&1
```

Collect all FAILED test names and their error messages. If all tests pass, report success and stop.

## Step 2 — Root cause: Classify each failure

For EACH failing test, run this classification sequence:

### 2a. Pre-existing check

Stash your changes and re-run just the failing test:
```bash
git stash
python3 -m pytest <test_path>::<test_name> -v --tb=short 2>&1
git stash pop
```

If it ALSO fails on the stashed (clean) code → **PRE-EXISTING**. Note it and move on — don't fix what was already broken.

### 2b. Read the failing test and your diff

For non-pre-existing failures, read:
1. The failing test code (understand what it asserts)
2. Your unstaged/staged diff (`git diff` + `git diff --cached`)

### 2c. Classify into one of these categories

| Category | Signal | Action |
|----------|--------|--------|
| **STALE_BASELINE** | Test hardcodes a value you intentionally changed (e.g., old config value, old score, old function signature) | Update the test's expected value to match the new reality |
| **STALE_IMPORT** | Test imports a function/class/constant you renamed or removed | Update import, or delete test if the tested behavior no longer exists |
| **OBSOLETE_TEST** | Test exercises behavior that was intentionally removed or replaced (e.g., old regime, deleted feature) | Delete the test — but ONLY if you can confirm the behavior is gone and not just refactored |
| **REAL_BREAK** | Test catches a genuine bug in your changes — the assertion is correct and your code is wrong | Fix your code, NOT the test |
| **FLAKY** | Test passes on clean code, fails intermittently (timing, ordering, floating-point edge) | Mark as the lowest-priority fix; do not block on it |

**Classification rules:**
- If the test asserts a hardcoded numeric value that matches a config/code value you changed → STALE_BASELINE (most common)
- If the test imports something you deleted/renamed → STALE_IMPORT
- If the test name references a feature you intentionally removed → OBSOLETE_TEST (verify by searching for the feature in current code)
- If the test assertion makes physical/logical sense and your code violates it → REAL_BREAK
- When in doubt, assume REAL_BREAK — it's safer to investigate than to delete

## Step 3 — Fix: Apply the correct remedy

### For STALE_BASELINE:
- Read the test file
- Update ONLY the hardcoded expected value(s) to match your new code/config
- Do NOT change the test's logic or structure
- If the test references a baselines file (e.g., `scoring_baselines.json`, `conftest.py` BASELINE_SCORES), update those too

### For STALE_IMPORT:
- Update the import to the new name/location
- If the import was deleted entirely, check if the test should be updated to test the replacement, or deleted

### For OBSOLETE_TEST:
- Confirm the feature is truly gone: `grep -r "feature_name"` across the codebase
- If gone: delete the test function (or test file if all tests in it are obsolete)
- If refactored (not gone): update the test to match the new API → reclassify as STALE_IMPORT

### For REAL_BREAK:
- Do NOT modify the test
- Report the issue clearly: what the test expects, what your code does, and why they disagree
- Suggest a fix to your code (not the test)
- Ask the user before applying the fix

### For FLAKY:
- Note it for the user
- Do not attempt to fix unless asked

### For PRE-EXISTING:
- Report it but do not fix it — it's not caused by current changes
- If user explicitly asks to fix pre-existing failures, treat them as a separate task

## Step 4 — Verify

After applying fixes, re-run the full suite (or the originally-scoped tests):
```bash
python3 -m pytest tests/ -v --tb=short 2>&1
```

If new failures appear, repeat from Step 2 for those.

## Step 5 — Report

Output a structured summary:

```
TEST FIX REPORT
===============
Tests run: <count>
Passed: <count>
Fixed: <count>
Pre-existing: <count>
Real breaks: <count>

FIXES APPLIED:
  [STALE_BASELINE] test_foo::test_bar — updated expected value 0.005→0.0
  [OBSOLETE_TEST]  test_baz::test_old_feature — deleted (feature removed in <commit>)

PRE-EXISTING (not touched):
  [PRE-EXISTING] test_qux::test_steering_direction — fails before and after changes

REAL BREAKS (need code fix):
  [REAL_BREAK] test_abc::test_safety — your code allows OOL but test requires 0
    → Suggested fix: <description>

REMAINING FAILURES: <count>
```

## Important rules

- NEVER delete a test just because it fails — always classify first
- NEVER change test assertions to make a REAL_BREAK pass — fix the code instead
- NEVER modify tests you haven't read
- When updating baselines, update ALL related baseline files (conftest.py, scoring_baselines.json, golden_recordings.json) — partial updates cause cascading failures
- If >10 tests fail from the same root cause (e.g., a renamed constant), fix the root cause once rather than patching each test individually
- Prefer `replace_all` edits when updating a renamed symbol across a test file
