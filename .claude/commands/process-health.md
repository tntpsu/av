Generate a continuous improvement Pareto from the historical improvement log, showing where issues originate and how to prevent them.

The user's request is: $ARGUMENTS

## Step 1 — Read the improvement log

```bash
cat data/reports/improvement_log.json
```

If the file doesn't exist or is empty, tell the user:
```
No improvement log found. Run /log-fix after commits to build history.
```

## Step 2 — Process Stage Pareto

Count entries by process_stage (requirements, design, implementation, testing, tooling).
Present as:

```
WHERE ISSUES ORIGINATE
═══════════════════════════════════════
Requirements    <bar>  <pct>%  (<count>)
Design          <bar>  <pct>%  (<count>)
Implementation  <bar>  <pct>%  (<count>)
Testing         <bar>  <pct>%  (<count>)
Tooling         <bar>  <pct>%  (<count>)
```

Use bar chart characters proportional to percentage.

## Step 3 — Sub-Category Pareto

For the top process_stage, break down by process_sub:

```
<TOP STAGE> SUB-CATEGORIES
═══════════════════════════════════════
<sub1>          <bar>  <pct>%
<sub2>          <bar>  <pct>%
...
```

## Step 4 — Detection Efficiency

Compute from the log:
- Average `detection_delay_iterations` across all entries
- Count entries where `detection_method` mentions a tool name vs "manual"
- Compare early entries vs recent entries for trend

```
DETECTION EFFICIENCY
═══════════════════════════════════════
Avg iterations to root cause: <val>
Issues caught by tooling:     <pct>%
Issues requiring manual trace: <pct>%
Trend: <improving/stable/degrading> (compare first half vs second half of log)
```

## Step 5 — Prevention Pattern Analysis

Group entries by `prevention` field. Identify recurring prevention themes:

```
PREVENTION THEMES
═══════════════════════════════════════
Theme                        Count  Example prevention
<theme>                      <N>    <example from log>
```

Common themes to look for:
- "Use physics-based formulas" → design rule needed
- "Add trace/diagnostic tool" → tooling gap
- "Add regression test" → testing gap
- "Review signal routing" → architecture review needed

## Step 6 — Top 3 Process Improvements

Based on all the Pareto data, recommend the top 3 improvements.

For each recommendation:
1. What pattern it addresses (from the Pareto)
2. Specific action (new tool, new test, new design rule, new gate)
3. Expected impact (fewer issues, faster detection, or prevented entirely)

```
TOP 3 PROCESS IMPROVEMENTS
════════════════════════════════════════
1. <recommendation>
   Pattern: <what Pareto data shows>
   Action: <specific, actionable step>
   Impact: <expected benefit>

2. <recommendation>
   ...

3. <recommendation>
   ...
```

## References
- Improvement log: `data/reports/improvement_log.json`
- Current state: `docs/agent/current_state.md`
- CLAUDE.md constraints: `CLAUDE.md` (Key Constraints section)
