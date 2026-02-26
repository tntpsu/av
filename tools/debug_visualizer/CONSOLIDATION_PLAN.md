# PhilViz — Consolidation & Roadmap Plan

**Last updated:** 2026-02-23
**Maintained by:** Agent + Philip

---

## Vision: Best-in-Class Analyze & Triage Tool

PhilViz should answer three questions after every drive, without manual grep or source diving:

1. **How did each layer perform?** — Per-layer health timeline, not just per-frame raw values.
2. **Where did the problem start, and who caused it?** — Backward blame trace from symptom to source layer, with stale propagation timeline.
3. **What should I do about it?** — Structured triage report mapping failure patterns to specific config levers and code locations.

This is a combined **analyze tool** (understand performance across layers and time) and **triage tool** (point to the responsible component/code and give a recommended next action).

---

## Completed Phases

### ✅ Phase 1 — Frame-Level Diagnostics

| Feature | Status |
|---|---|
| Frame replay with camera + overlays | Done |
| Polynomial inspector (recorded vs. re-run detection) | Done |
| On-demand debug overlay generation (edges, yellow mask) | Done |
| Ground truth comparison overlays | Done |
| Raw data side panel (150+ fields per frame) | Done |

### ✅ Phase 2 — Recording-Level Analysis

| Feature | Status |
|---|---|
| Recording summary metrics tab | Done |
| 18 issue types auto-detected per frame | Done |
| Issue navigation (jump to problematic frames) | Done |
| Signal chain tab (Perception → Trajectory → Control per frame) | Done |
| Trajectory waterfall diagnostics | Done |
| Control waterfall diagnostics | Done |
| A/B config comparison tab | Done |
| Comfort gate bundle display | Done |
| Causal timeline (issue → downstream consequence) | Done |
| Replay badge awareness (perc lock, traj lock keys) | Done |

**Current state:** PhilViz correctly reports *what* happened. It does not yet explain *why* or *who* is responsible.

---

## Architecture Reference (read before touching any file)

```
tools/debug_visualizer/
├── server.py               # Flask backend — 3,675 lines, 20+ routes
├── index.html              # 939 lines — tab buttons + pane structure
├── visualizer.js           # 8,927 lines — Visualizer class, all tab logic
├── data_loader.js          # 311 lines — DataLoader class, all API calls
├── overlay_renderer.js     # 667 lines — canvas overlay rendering
├── style.css               # Styles
└── backend/
    ├── summary_analyzer.py    # 17-line shim → delegates to tools/drive_summary_core.py
    ├── issue_detector.py      # ~1,500 lines — IssueDetector class, 18 issue types
    └── diagnostics.py         # ~3,000 lines — trajectory/steering/signal-chain analysis
```

### Exact patterns to follow (do not deviate)

**Flask route (server.py):**
```python
@app.route('/api/recording/<path:filename>/layer-health')
def get_layer_health(filename):
    from urllib.parse import unquote
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404
    try:
        from backend.layer_health import LayerHealthAnalyzer
        result = LayerHealthAnalyzer(filepath).compute()
        return jsonify(numpy_to_list(result))
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
```

**HDF5 access (backend/*.py):**
```python
with h5py.File(recording_path, 'r') as f:
    has_field = "group/field" in f          # existence check — always do this
    arr = np.array(f["group/field"][:])     # load full array
    val = safe_float(f["group/scalar"][0])  # safe scalar (handles NaN/Inf)
```

**HTML tab button + pane (index.html):**
```html
<!-- In the tab-buttons div: -->
<button class="tab-btn" data-tab="layers">Layers</button>
<!-- In the main-content div: -->
<div class="tab-pane" id="layers-tab">
    <div id="layers-content"></div>
</div>
```

**JavaScript tab load function (visualizer.js):**
```javascript
async loadLayers() {
    if (!this.currentRecording) return;
    const content = document.getElementById('layers-content');
    content.innerHTML = '<p style="color:#888;text-align:center;padding:2rem;">Loading...</p>';
    try {
        const response = await fetch(`/api/recording/${this.currentRecording}/layer-health`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (data.error) { content.innerHTML = `<p style="color:#ff6b6b;">${data.error}</p>`; return; }
        content.innerHTML = this.renderLayersHtml(data);
    } catch (e) {
        content.innerHTML = `<p style="color:#ff6b6b;">Error: ${this.escapeHtml(e.message)}</p>`;
    }
}
```

**switchTab() addition (visualizer.js ~line 7611):**
```javascript
// Add inside the if/else chain in switchTab():
} else if (tabName === 'layers') {
    this.loadLayers();
} else if (tabName === 'triage') {
    this.loadTriage();
}
```

**JavaScript API call (no DataLoader needed for new endpoints — call fetch directly in visualizer.js):**
```javascript
const params = new URLSearchParams({ frame: frameIdx, metric: 'lateral_error' });
const response = await fetch(`/api/recording/${this.currentRecording}/blame-trace?${params}`);
const data = await response.json();
```

---

## Phase 3 — Layer Attribution Engine

**Goal:** Visible per-layer health across the full recording as a color-coded timeline.
**User value:** "How did each layer perform?" answered in one tab, no manual metric hunting.
**Depends on:** Nothing (first phase, standalone).

---

### 3.1 New file: `backend/layer_health.py`

**Create this file from scratch.** Model the class structure on `diagnostics.py`:
- `__init__` accepts `recording_path: Path`
- `compute()` opens HDF5 once, loads all needed arrays, returns dict

```python
"""
Layer health scoring for PhilViz Phase 3.

Computes per-frame health scores (0.0–1.0) for each stack layer
(Perception, Trajectory, Control) based on HDF5 signal quality.

Scoring is intentionally simple: weighted linear combination of
normalized signals, clamped to [0, 1]. No ML required.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import h5py


# ── Thresholds ────────────────────────────────────────────────────────────────
# Tune these before tuning weights. They define "bad" for each signal.
PERCEPTION_CONF_FLOOR   = 0.1    # confidence below this → hard fallback territory
TRAJ_REF_ERROR_MAX_M    = 2.0    # lateral ref error > this → full penalty
CTRL_LATERAL_ERROR_MAX  = 2.0    # lateral error > this → full penalty (= safety.max_lateral_error)
CTRL_JERK_GATE          = 6.0    # commanded_jerk_p95 comfort gate (m/s³)
LOOKAHEAD_MIN_M         = 5.0    # lookahead below this is suspiciously short
LOOKAHEAD_MAX_M         = 25.0   # lookahead above this is unusually long


@dataclass
class LayerHealthFrame:
    frame_idx:           int
    timestamp:           float
    perception_score:    float
    perception_flags:    list[str]
    trajectory_score:    float
    trajectory_flags:    list[str]
    control_score:       float
    control_flags:       list[str]


class LayerHealthAnalyzer:
    def __init__(self, recording_path: Path) -> None:
        self.path = recording_path

    def compute(self) -> dict:
        """
        Load all needed HDF5 arrays once, compute per-frame scores,
        return JSON-ready dict.
        """
        with h5py.File(self.path, 'r') as f:
            n_frames = self._frame_count(f)
            frames = []
            for i in range(n_frames):
                pf = self._score_perception(f, i)
                tf = self._score_trajectory(f, i)
                cf = self._score_control(f, i)
                frames.append({
                    "frame_idx":         i,
                    "timestamp":         self._scalar(f, "vehicle/timestamp", i, default=float(i) / 20.0),
                    "perception_score":  round(pf["score"], 3),
                    "perception_flags":  pf["flags"],
                    "trajectory_score":  round(tf["score"], 3),
                    "trajectory_flags":  tf["flags"],
                    "control_score":     round(cf["score"], 3),
                    "control_flags":     cf["flags"],
                })

        summary = self._summarize(frames)
        return {
            "frame_count": n_frames,
            "layers": ["perception", "trajectory", "control"],
            "frames": frames,
            "summary": summary,
        }

    # ── Per-layer scorers ────────────────────────────────────────────────────

    def _score_perception(self, f: h5py.File, i: int) -> dict:
        score = 1.0
        flags = []

        # Stale frame (weight 0.30) — hard cap
        stale = bool(self._scalar(f, "perception/stale_frame", i, default=0))
        if stale:
            score -= 0.30
            flags.append("stale")

        # Detection confidence (weight 0.35)
        conf = self._scalar(f, "perception/confidence", i, default=0.5)
        conf_norm = max(0.0, min(1.0, (conf - PERCEPTION_CONF_FLOOR) / (1.0 - PERCEPTION_CONF_FLOOR)))
        score -= 0.35 * (1.0 - conf_norm)
        if conf < PERCEPTION_CONF_FLOOR + 0.1:
            flags.append("low_confidence")

        # Num lanes detected (weight 0.20): 2=ideal, 1=partial, 0=failed
        n_lanes = int(self._scalar(f, "perception/num_lanes_detected", i, default=2))
        lane_score = {2: 1.0, 1: 0.5, 0: 0.0}.get(n_lanes, 0.5)
        score -= 0.20 * (1.0 - lane_score)
        if n_lanes == 1:
            flags.append("single_lane")
        elif n_lanes == 0:
            flags.append("no_lanes")

        # Lane center gate fired (weight 0.15)
        gate_fired = bool(self._scalar(f, "perception/lane_gate_fired", i, default=0))
        if gate_fired:
            score -= 0.15
            flags.append("gate_fired")

        return {"score": max(0.0, min(1.0, score)), "flags": flags}

    def _score_trajectory(self, f: h5py.File, i: int) -> dict:
        score = 1.0
        flags = []

        # Lateral ref error (weight 0.40)
        ref_err = abs(self._scalar(f, "trajectory/lateral_ref_error", i, default=0.0))
        ref_norm = max(0.0, min(1.0, ref_err / TRAJ_REF_ERROR_MAX_M))
        score -= 0.40 * ref_norm
        if ref_err > 0.3:
            flags.append("ref_error_high")

        # Ref rate clamp active (weight 0.25)
        clamped = bool(self._scalar(f, "trajectory/ref_x_rate_clamped", i, default=0))
        if clamped:
            score -= 0.25
            flags.append("ref_rate_clamped")

        # Lookahead validity (weight 0.20)
        lookahead = self._scalar(f, "trajectory/lookahead_distance", i, default=10.0)
        if lookahead < LOOKAHEAD_MIN_M or lookahead > LOOKAHEAD_MAX_M:
            score -= 0.20
            flags.append("lookahead_out_of_range")

        # Curvature context active — informational only (no score impact)
        curv_active = bool(self._scalar(f, "trajectory/curve_anticipation_active", i, default=0))
        if curv_active:
            flags.append("curve_anticipation")

        return {"score": max(0.0, min(1.0, score)), "flags": flags}

    def _score_control(self, f: h5py.File, i: int) -> dict:
        score = 1.0
        flags = []

        # Lateral error (weight 0.40)
        lat_err = abs(self._scalar(f, "control/lateral_error", i, default=0.0))
        lat_norm = max(0.0, min(1.0, lat_err / CTRL_LATERAL_ERROR_MAX))
        score -= 0.40 * lat_norm
        if lat_err > 0.4:
            flags.append("lateral_error_high")

        # Commanded jerk (weight 0.30)
        jerk = abs(self._scalar(f, "control/commanded_jerk", i, default=0.0))
        jerk_norm = max(0.0, min(1.0, jerk / CTRL_JERK_GATE))
        score -= 0.30 * jerk_norm
        if jerk > CTRL_JERK_GATE * 0.75:
            flags.append("jerk_spike")

        # Jerk limiter active (weight 0.10)
        jerk_active = bool(self._scalar(f, "control/jerk_limiter_active", i, default=0))
        if jerk_active:
            score -= 0.10
            flags.append("jerk_limiter_active")

        # Accel smoothness (weight 0.20): rolling std over ±2 frames, normalized
        # (computed at summarize time — here just read raw accel)
        accel = abs(self._scalar(f, "control/acceleration", i, default=0.0))
        if accel > 3.0:   # above comfort gate → flag
            flags.append("accel_high")

        return {"score": max(0.0, min(1.0, score)), "flags": flags}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _scalar(self, f: h5py.File, path: str, i: int, default: float = 0.0) -> float:
        """Safely read scalar at index i from an HDF5 dataset."""
        if path not in f:
            return default
        try:
            val = float(f[path][i])
            if not np.isfinite(val):
                return default
            return val
        except (IndexError, ValueError):
            return default

    def _frame_count(self, f: h5py.File) -> int:
        """Infer frame count from the first available dataset."""
        for key in ("vehicle/timestamp", "control/steering", "perception/confidence"):
            if key in f:
                return len(f[key])
        raise ValueError("Cannot determine frame count — no known dataset found in HDF5.")

    def _summarize(self, frames: list[dict]) -> dict:
        """Aggregate per-layer stats across all frames."""
        summary = {}
        for layer in ("perception", "trajectory", "control"):
            scores = [fr[f"{layer}_score"] for fr in frames]
            if not scores:
                summary[layer] = {}
                continue
            arr = np.array(scores)
            summary[layer] = {
                "mean_score":     round(float(arr.mean()), 3),
                "min_score":      round(float(arr.min()), 3),
                "pct_green":      round(float((arr >= 0.8).mean()), 3),
                "pct_yellow":     round(float(((arr >= 0.5) & (arr < 0.8)).mean()), 3),
                "pct_red":        round(float((arr < 0.5).mean()), 3),
                "pct_unhealthy":  round(float((arr < 0.8).mean()), 3),
            }
        return summary
```

**HDF5 fields used — check existence before reading:**

| Field | Where recorded | Fallback if missing |
|---|---|---|
| `perception/stale_frame` | av_stack.py → recorder | default 0 |
| `perception/confidence` | perception/inference.py → recorder | default 0.5 |
| `perception/num_lanes_detected` | av_stack.py → recorder | default 2 |
| `perception/lane_gate_fired` | av_stack.py EMA gating → recorder | default 0 |
| `trajectory/lateral_ref_error` | trajectory/inference.py → recorder | default 0.0 |
| `trajectory/ref_x_rate_clamped` | trajectory/inference.py → recorder | default 0 |
| `trajectory/lookahead_distance` | trajectory/inference.py → recorder | default 10.0 |
| `trajectory/curve_anticipation_active` | trajectory/inference.py → recorder | default 0 |
| `control/lateral_error` | pid_controller.py → recorder | default 0.0 |
| `control/commanded_jerk` | pid_controller.py → recorder | default 0.0 |
| `control/jerk_limiter_active` | pid_controller.py → recorder | default 0 |
| `control/acceleration` | vehicle state → recorder | default 0.0 |
| `vehicle/timestamp` | av_stack.py → recorder | default frame/20.0 |

**⚠ If a field is not in the HDF5 (older recordings):** `_scalar()` returns the default — the score degrades gracefully to neutral (0.5 penalty contribution → moderate score). Never crash on missing fields.

---

### 3.2 Edit: `server.py` — add route

**Where to add:** After the `/diagnostics` route (~line 2797). Search for:
```python
@app.route('/api/recording/<path:filename>/diagnostics')
```
Add immediately after its function body ends:

```python
@app.route('/api/recording/<path:filename>/layer-health')
def get_layer_health(filename):
    """Compute per-frame Perception / Trajectory / Control health scores."""
    from urllib.parse import unquote
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404
    try:
        from backend.layer_health import LayerHealthAnalyzer
        result = LayerHealthAnalyzer(filepath).compute()
        return jsonify(numpy_to_list(result))
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
```

---

### 3.3 Edit: `index.html` — add Layers tab

**Where to add the button:** In the `<div class="tab-buttons">` block. Search for:
```html
<button class="tab-btn" data-tab="diagnostics">Diagnostics</button>
```
Add immediately after:
```html
<button class="tab-btn" data-tab="layers">Layers</button>
```

**Where to add the pane:** Search for `<div class="tab-pane" id="diagnostics-tab">`. Add immediately after the closing `</div>` of that pane:
```html
<div class="tab-pane" id="layers-tab">
    <div id="layers-content"></div>
</div>
```

---

### 3.4 Edit: `visualizer.js` — add Layers tab logic

**Step 1 — Wire up switchTab().** Search for `} else if (tabName === 'diagnostics') {`. Add after its block:
```javascript
} else if (tabName === 'layers') {
    this.loadLayers();
}
```

**Step 2 — Add loadLayers().** Add near `loadDiagnostics()` (~line 2362):
```javascript
async loadLayers() {
    if (!this.currentRecording) return;
    const content = document.getElementById('layers-content');
    if (!content) return;
    content.innerHTML = '<p style="color:#888;text-align:center;padding:2rem;">Computing layer health…</p>';
    try {
        const response = await fetch(`/api/recording/${this.currentRecording}/layer-health`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (data.error) {
            content.innerHTML = `<p style="color:#ff6b6b;padding:1rem;">Error: ${this.escapeHtml(data.error)}</p>`;
            return;
        }
        content.innerHTML = this.renderLayersHtml(data);
        this._initLayersCanvases(data);
    } catch (e) {
        content.innerHTML = `<p style="color:#ff6b6b;">Error: ${this.escapeHtml(e.message)}</p>`;
    }
}
```

**Step 3 — Add renderLayersHtml().** Returns the HTML scaffolding (canvases + summary table):
```javascript
renderLayersHtml(data) {
    const { summary, frame_count } = data;
    const layers = [
        { key: 'perception', label: 'Perception', color: '#7ecfff' },
        { key: 'trajectory', label: 'Trajectory', color: '#a8e6a3' },
        { key: 'control',    label: 'Control',    color: '#f0b27a' },
    ];

    // Summary table
    let summaryHtml = '<table class="layers-summary-table"><thead><tr>' +
        '<th>Layer</th><th>Mean Score</th><th>🟢 Green</th><th>🟡 Yellow</th><th>🔴 Red</th>' +
        '</tr></thead><tbody>';
    for (const { key, label } of layers) {
        const s = summary[key] || {};
        summaryHtml += `<tr>
            <td>${label}</td>
            <td>${((s.mean_score || 0) * 100).toFixed(1)}%</td>
            <td>${((s.pct_green || 0) * 100).toFixed(1)}%</td>
            <td>${((s.pct_yellow || 0) * 100).toFixed(1)}%</td>
            <td>${((s.pct_red || 0) * 100).toFixed(1)}%</td>
        </tr>`;
    }
    summaryHtml += '</tbody></table>';

    // Canvas rows
    let canvasHtml = '';
    for (const { key, label } of layers) {
        canvasHtml += `
            <div class="layer-row">
                <div class="layer-label">${label}</div>
                <canvas id="layer-canvas-${key}" class="layer-canvas"
                        width="${frame_count}" height="40"
                        data-layer="${key}" title="Click to navigate to frame"></canvas>
            </div>`;
    }

    return `
        <div class="layers-container">
            <h3>Layer Health Timeline <span style="color:#888;font-size:0.85em;">(${frame_count} frames)</span></h3>
            ${summaryHtml}
            <div class="layer-timeline" id="layer-timeline">
                ${canvasHtml}
                <div class="layer-axis">
                    <span>Frame 0</span>
                    <span>Frame ${Math.floor(frame_count / 2)}</span>
                    <span>Frame ${frame_count}</span>
                </div>
                <div id="layer-hover-info" class="layer-hover-info"></div>
            </div>
        </div>`;
}
```

**Step 4 — Add _initLayersCanvases().** Draws the colored bars and wires click + hover:
```javascript
_initLayersCanvases(data) {
    const { frames } = data;
    const layers = ['perception', 'trajectory', 'control'];
    const GREEN = '#4caf50', YELLOW = '#ff9800', RED = '#f44336';

    for (const layer of layers) {
        const canvas = document.getElementById(`layer-canvas-${layer}`);
        if (!canvas) continue;
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        frames.forEach((fr, i) => {
            const score = fr[`${layer}_score`];
            ctx.fillStyle = score >= 0.8 ? GREEN : score >= 0.5 ? YELLOW : RED;
            ctx.fillRect(i, 0, 1, h);
        });

        // Click → navigate to frame
        canvas.addEventListener('click', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const frameIdx = Math.floor((x / rect.width) * frames.length);
            if (frameIdx >= 0 && frameIdx < frames.length) {
                this.goToFrame(frameIdx);        // existing method
                this.switchTab('replay');        // existing method
            }
        });

        // Hover → show score + flags
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const frameIdx = Math.floor((x / rect.width) * frames.length);
            const fr = frames[frameIdx];
            if (!fr) return;
            const score = fr[`${layer}_score`];
            const flags = fr[`${layer}_flags`].join(', ') || '—';
            const info = document.getElementById('layer-hover-info');
            if (info) {
                info.textContent = `Frame ${frameIdx} | ${layer}: ${(score * 100).toFixed(1)}% | ${flags}`;
                info.style.left = `${e.clientX - canvas.getBoundingClientRect().left}px`;
            }
        });
    }
}
```

**Step 5 — Add CSS to `style.css`:**
```css
.layers-container { padding: 1rem; }
.layer-row { display: flex; align-items: center; margin-bottom: 6px; }
.layer-label { width: 100px; font-size: 0.85em; color: #ccc; flex-shrink: 0; }
.layer-canvas { flex: 1; height: 40px; cursor: pointer; border: 1px solid #333; }
.layer-axis { display: flex; justify-content: space-between; font-size: 0.75em; color: #666; margin-top: 4px; }
.layer-hover-info { position: absolute; background: #222; color: #eee; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; pointer-events: none; white-space: nowrap; }
.layers-summary-table { width: 100%; border-collapse: collapse; margin-bottom: 1rem; font-size: 0.9em; }
.layers-summary-table th, .layers-summary-table td { padding: 6px 10px; border: 1px solid #333; text-align: center; }
.layers-summary-table th { background: #222; color: #ccc; }
```

---

### 3.5 Testing Phase 3

**Unit test — backend only:**
```bash
# Create a minimal synthetic HDF5 and verify compute() runs without error
python - <<'EOF'
import h5py, numpy as np, tempfile
from pathlib import Path
from backend.layer_health import LayerHealthAnalyzer

with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
    path = Path(f.name)

with h5py.File(path, 'w') as f:
    N = 100
    f.create_dataset("vehicle/timestamp", data=np.arange(N, dtype=float) * 0.05)
    f.create_dataset("perception/confidence", data=np.random.uniform(0.3, 1.0, N))
    f.create_dataset("perception/num_lanes_detected", data=np.full(N, 2))
    f.create_dataset("control/lateral_error", data=np.random.uniform(0, 0.3, N))
    f.create_dataset("control/commanded_jerk", data=np.random.uniform(0, 1.0, N))

result = LayerHealthAnalyzer(path).compute()
assert result["frame_count"] == N
assert len(result["frames"]) == N
assert all(0 <= fr["perception_score"] <= 1 for fr in result["frames"])
print("✓ Phase 3 backend unit test passed")
EOF
```

**API test — server running:**
```bash
# Start server, then:
curl -s "http://localhost:5001/api/recording/recording_20260222_202957.h5/layer-health" | python -m json.tool | head -40
# Expect: frame_count, layers, frames array, summary dict
```

**Visual test — browser:**
1. Load any s_loop recording
2. Click "Layers" tab
3. Verify: three colored bars render, mostly green for a good recording
4. Click any red segment → should jump to Replay tab at that frame
5. Hover → info banner shows score + flags

**Regression:** No existing tests should break. Run:
```bash
pytest tests/ -v -x -q
```

---

## Phase 4 — Root Cause Tracer

**Goal:** Backward trace from a metric spike to the originating layer. Stale propagation timeline.
**User value:** "Where did the problem start?" answered with a propagation arrow diagram.
**Depends on:** Phase 3 (uses `LayerHealthAnalyzer` output as input to trace).

---

### 4.1 New file: `backend/blame_tracer.py`

```python
"""
Blame tracer for PhilViz Phase 4.

Given a target frame and metric, walks backward through layer health
scores to find the earliest degradation event and assigns primary blame
to the originating layer.

Also finds stale propagation events: stale perception → trajectory drift
→ control error, with lag estimates.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import h5py


BASELINE_WINDOW   = 20    # frames before target to compute baseline
LOOKBACK          = 20    # frames to search backward from target
DEGRADATION_DELTA = 0.20  # score drop threshold to count as "degradation"
MIN_CAUSAL_LAG    = 2     # upstream degradation must precede target by at least this many frames


@dataclass
class BlameLink:
    frame_idx:  int
    layer:      str       # "perception" | "trajectory" | "control"
    signal:     str       # e.g. "stale_frame", "lateral_ref_error"
    value:      float
    baseline:   float
    delta:      float     # value - baseline (positive = degraded)


@dataclass
class BlameChain:
    target_frame:   int
    target_metric:  str
    chain:          list[BlameLink]
    primary_blame:  str       # "perception" | "trajectory" | "control" | "unknown"
    lag_frames:     int
    confidence:     float     # 0–1; lower if chain is broken or signal is noisy


@dataclass
class StalePropagationEvent:
    stale_start_frame:      int
    stale_duration:         int
    trajectory_lag_frames:  int
    control_lag_frames:     int
    max_trajectory_delta:   float
    max_control_delta:      float
    propagation_timeline:   list[dict]   # [{frame, perc_stale, traj_delta, ctrl_delta}]


class BlameTracer:
    def __init__(self, recording_path: Path) -> None:
        self.path = recording_path
        self._health_cache: Optional[list[dict]] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def trace_blame(
        self,
        target_frame: int,
        metric: str = "lateral_error",
        lookback: int = LOOKBACK,
    ) -> dict:
        """
        Walk backward from target_frame to find the originating layer.
        Returns a JSON-ready dict.
        """
        health = self._get_health()
        if not health or target_frame >= len(health):
            return {"error": f"Frame {target_frame} out of range"}

        n = len(health)
        start = max(0, target_frame - lookback)
        baseline_start = max(0, target_frame - lookback - BASELINE_WINDOW)
        baseline_end   = max(0, target_frame - lookback)

        # Compute baselines per layer
        baselines = {}
        for layer in ("perception", "trajectory", "control"):
            key = f"{layer}_score"
            window = [health[i][key] for i in range(baseline_start, baseline_end) if i < n]
            baselines[layer] = float(np.mean(window)) if window else 0.8

        # Walk backward — find first degradation per layer
        first_degradation: dict[str, int] = {}
        for i in range(target_frame, max(start - 1, -1), -1):
            for layer in ("perception", "trajectory", "control"):
                key = f"{layer}_score"
                score = health[i][key]
                baseline = baselines[layer]
                if (baseline - score) >= DEGRADATION_DELTA and layer not in first_degradation:
                    first_degradation[layer] = i

        # Build chain from earliest degradation to target
        chain: list[BlameLink] = []
        for layer in ("perception", "trajectory", "control"):
            if layer in first_degradation:
                fi = first_degradation[layer]
                score = health[fi][f"{layer}_score"]
                baseline = baselines[layer]
                flags = health[fi].get(f"{layer}_flags", [])
                signal = flags[0] if flags else f"{layer}_score"
                chain.append(BlameLink(
                    frame_idx=fi,
                    layer=layer,
                    signal=signal,
                    value=round(score, 3),
                    baseline=round(baseline, 3),
                    delta=round(baseline - score, 3),
                ))

        chain.sort(key=lambda x: x.frame_idx)

        # Primary blame = earliest degrading layer
        primary_blame = "unknown"
        lag_frames = 0
        if chain:
            earliest = chain[0]
            if target_frame - earliest.frame_idx >= MIN_CAUSAL_LAG:
                primary_blame = earliest.layer
                lag_frames = target_frame - earliest.frame_idx

        # Confidence: higher if chain is clean (all three layers degrade in order)
        confidence = self._compute_confidence(chain, target_frame)

        return {
            "target_frame":  target_frame,
            "target_metric": metric,
            "primary_blame": primary_blame,
            "lag_frames":    lag_frames,
            "confidence":    round(confidence, 2),
            "chain": [
                {
                    "frame_idx": lnk.frame_idx,
                    "layer":     lnk.layer,
                    "signal":    lnk.signal,
                    "value":     lnk.value,
                    "baseline":  lnk.baseline,
                    "delta":     lnk.delta,
                }
                for lnk in chain
            ],
        }

    def find_stale_propagation(self, max_lookahead: int = 15) -> dict:
        """
        Find stale perception events and measure downstream cascade
        on trajectory and control for up to max_lookahead frames.
        Returns JSON-ready dict.
        """
        health = self._get_health()
        if not health:
            return {"event_count": 0, "events": []}

        # Find stale events: groups of consecutive stale frames
        stale_flags = [("stale" in fr.get("perception_flags", [])) for fr in health]
        events = []
        i = 0
        while i < len(stale_flags):
            if not stale_flags[i]:
                i += 1
                continue

            # Event start
            start = i
            while i < len(stale_flags) and stale_flags[i]:
                i += 1
            duration = i - start

            # Measure downstream cascade
            baseline_perc = np.mean([health[j]["perception_score"]
                                     for j in range(max(0, start - 10), start)]) if start >= 1 else 0.8
            baseline_traj = np.mean([health[j]["trajectory_score"]
                                     for j in range(max(0, start - 10), start)]) if start >= 1 else 0.8
            baseline_ctrl = np.mean([health[j]["control_score"]
                                     for j in range(max(0, start - 10), start)]) if start >= 1 else 0.8

            timeline = []
            max_traj_delta = 0.0
            max_ctrl_delta = 0.0
            traj_lag = 0
            ctrl_lag = 0

            for offset in range(max_lookahead):
                fi = start + offset
                if fi >= len(health):
                    break
                fr = health[fi]
                traj_delta = max(0.0, baseline_traj - fr["trajectory_score"])
                ctrl_delta = max(0.0, baseline_ctrl - fr["control_score"])
                timeline.append({
                    "frame":       fi,
                    "perc_stale":  stale_flags[fi] if fi < len(stale_flags) else False,
                    "traj_delta":  round(traj_delta, 3),
                    "ctrl_delta":  round(ctrl_delta, 3),
                })
                if traj_delta > max_traj_delta:
                    max_traj_delta = traj_delta
                    traj_lag = offset
                if ctrl_delta > max_ctrl_delta:
                    max_ctrl_delta = ctrl_delta
                    ctrl_lag = offset

            events.append({
                "stale_start_frame":     start,
                "stale_duration":        duration,
                "trajectory_lag_frames": traj_lag,
                "control_lag_frames":    ctrl_lag,
                "max_trajectory_delta":  round(max_traj_delta, 3),
                "max_control_delta":     round(max_ctrl_delta, 3),
                "propagation_timeline":  timeline,
            })

        return {"event_count": len(events), "events": events}

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _get_health(self) -> list[dict]:
        """Lazy-load layer health (cache within this instance)."""
        if self._health_cache is None:
            from backend.layer_health import LayerHealthAnalyzer
            result = LayerHealthAnalyzer(self.path).compute()
            self._health_cache = result["frames"]
        return self._health_cache

    def _compute_confidence(self, chain: list[BlameLink], target_frame: int) -> float:
        """Estimate confidence: 1.0 if all three layers degrade in causal order."""
        if not chain:
            return 0.0
        layers_in_order = [lnk.layer for lnk in chain]
        canonical = ["perception", "trajectory", "control"]
        # Count how many layers appear in canonical order
        present = [l for l in canonical if l in layers_in_order]
        order_score = len(present) / 3.0
        # Reduce confidence if lag is very short (may be coincidence)
        lag = target_frame - chain[0].frame_idx if chain else 0
        lag_penalty = 0.2 if lag < 3 else 0.0
        return max(0.0, min(1.0, order_score - lag_penalty + 0.1 * len(chain)))
```

---

### 4.2 Edit: `server.py` — add two routes

**Add after the `/layer-health` route from Phase 3:**

```python
@app.route('/api/recording/<path:filename>/blame-trace')
def get_blame_trace(filename):
    """Backward blame trace from target_frame to originating layer."""
    from urllib.parse import unquote
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404
    try:
        target_frame = int(request.args.get('frame', 0))
        metric = request.args.get('metric', 'lateral_error')
        from backend.blame_tracer import BlameTracer
        result = BlameTracer(filepath).trace_blame(target_frame, metric)
        return jsonify(numpy_to_list(result))
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/recording/<path:filename>/stale-propagation')
def get_stale_propagation(filename):
    """Find stale perception events and their downstream cascade."""
    from urllib.parse import unquote
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404
    try:
        from backend.blame_tracer import BlameTracer
        result = BlameTracer(filepath).find_stale_propagation()
        return jsonify(numpy_to_list(result))
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
```

---

### 4.3 Edit: `index.html` — Chain tab additions

**No new tab needed** — additions go inside the existing `#chain-tab` pane.

Find the existing `<div class="tab-pane" id="chain-tab">` and add inside it:
```html
<div id="chain-blame-panel" class="blame-panel" style="display:none;">
    <h4>Blame Trace</h4>
    <div id="blame-chain-display"></div>
    <button id="blame-jump-source-btn" class="btn btn-secondary" style="margin-top:0.5rem;">
        Jump to Source Frame
    </button>
</div>
<div id="chain-stale-events" class="stale-events-panel" style="display:none;">
    <h4>Stale Propagation Events</h4>
    <div id="stale-events-table"></div>
</div>
```

---

### 4.4 Edit: `visualizer.js` — Blame Panel logic

**Add method `showBlamePanel(frameIdx, metric)`:**
```javascript
async showBlamePanel(frameIdx, metric = 'lateral_error') {
    const panel = document.getElementById('chain-blame-panel');
    const display = document.getElementById('blame-chain-display');
    if (!panel || !display) return;
    panel.style.display = 'block';
    display.innerHTML = '<em>Tracing blame…</em>';
    try {
        const params = new URLSearchParams({ frame: frameIdx, metric });
        const resp = await fetch(
            `/api/recording/${this.currentRecording}/blame-trace?${params}`
        );
        const data = await resp.json();
        if (data.error) { display.innerHTML = `<span style="color:#ff6b6b;">${data.error}</span>`; return; }

        const badge = {
            perception: '#7ecfff', trajectory: '#a8e6a3', control: '#f0b27a', unknown: '#888'
        }[data.primary_blame] || '#888';

        let html = `<div class="blame-summary">
            Primary blame: <span style="color:${badge};font-weight:bold;">${data.primary_blame}</span>
            &nbsp;(${(data.confidence * 100).toFixed(0)}% confidence, ${data.lag_frames} frame lag)
        </div>`;

        // Arrow chain
        html += '<div class="blame-chain-arrows">';
        for (const lnk of data.chain) {
            html += `<div class="blame-link">
                <span class="blame-frame">Frame ${lnk.frame_idx}</span>
                <span class="blame-layer">[${lnk.layer}]</span>
                <span class="blame-signal">${lnk.signal}</span>
                <span class="blame-delta">Δ${lnk.delta.toFixed(3)}</span>
                <span class="blame-arrow">→</span>
            </div>`;
        }
        html += `<div class="blame-link blame-target">
            <span class="blame-frame">Frame ${data.target_frame}</span>
            <span class="blame-layer">[${data.target_metric}]</span>
            <span class="blame-signal">spike detected</span>
        </div></div>`;

        display.innerHTML = html;

        // Wire jump button
        const jumpBtn = document.getElementById('blame-jump-source-btn');
        if (jumpBtn && data.chain.length > 0) {
            jumpBtn.onclick = () => {
                this.goToFrame(data.chain[0].frame_idx);
                this.switchTab('replay');
            };
            jumpBtn.style.display = 'block';
        }
    } catch (e) {
        display.innerHTML = `<span style="color:#ff6b6b;">${this.escapeHtml(e.message)}</span>`;
    }
}
```

**Add method `loadStaleEvents()`:**
```javascript
async loadStaleEvents() {
    if (!this.currentRecording) return;
    const container = document.getElementById('stale-events-table');
    const panel = document.getElementById('chain-stale-events');
    if (!container || !panel) return;
    panel.style.display = 'block';
    container.innerHTML = '<em>Loading stale events…</em>';
    try {
        const resp = await fetch(
            `/api/recording/${this.currentRecording}/stale-propagation`
        );
        const data = await resp.json();
        if (!data.events || data.events.length === 0) {
            container.innerHTML = '<em style="color:#888;">No stale propagation events found.</em>';
            return;
        }
        let html = '<table class="stale-events-table"><thead><tr>' +
            '<th>Start Frame</th><th>Duration</th><th>Traj Lag</th><th>Ctrl Lag</th>' +
            '<th>Max Traj Δ</th><th>Max Ctrl Δ</th><th></th>' +
            '</tr></thead><tbody>';
        for (const ev of data.events) {
            html += `<tr>
                <td>${ev.stale_start_frame}</td>
                <td>${ev.stale_duration}f</td>
                <td>${ev.trajectory_lag_frames}f</td>
                <td>${ev.control_lag_frames}f</td>
                <td>${ev.max_trajectory_delta.toFixed(3)}</td>
                <td>${ev.max_control_delta.toFixed(3)}</td>
                <td><button class="btn-small" onclick="visualizer.goToFrame(${ev.stale_start_frame})">→ Frame</button></td>
            </tr>`;
        }
        html += '</tbody></table>';
        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = `<span style="color:#ff6b6b;">${this.escapeHtml(e.message)}</span>`;
    }
}
```

**Where to call `loadStaleEvents()`:** Add to `loadChain()` (the existing function that loads chain tab data) — call it at the end after the main chain data loads.

**Where to add "Trace Blame" buttons:** Inside `renderChainHtml()` (find the function that renders per-frame chain rows). After each row that has a low health score, add:
```javascript
// Inside the per-frame loop in renderChainHtml():
const hasSpike = (frameData.perception_score < 0.6 || frameData.control_score < 0.6);
if (hasSpike) {
    rowHtml += `<button class="btn-tiny blame-btn"
        onclick="visualizer.showBlamePanel(${frameData.frame_idx})">Trace Blame</button>`;
}
```

---

### 4.5 CSS additions for Phase 4 (add to `style.css`)

```css
.blame-panel { background: #1a1a1a; border: 1px solid #333; padding: 1rem; margin-top: 1rem; border-radius: 4px; }
.blame-summary { margin-bottom: 0.75rem; font-size: 0.95em; }
.blame-chain-arrows { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.blame-link { display: flex; gap: 6px; align-items: center; background: #222; padding: 4px 8px; border-radius: 4px; font-size: 0.82em; }
.blame-link.blame-target { background: #3a1a1a; border: 1px solid #f44336; }
.blame-frame { color: #888; }
.blame-layer { color: #7ecfff; font-weight: bold; }
.blame-signal { color: #eee; }
.blame-delta { color: #ff9800; }
.blame-arrow { color: #555; font-size: 1.2em; }
.btn-tiny { font-size: 0.72em; padding: 2px 6px; background: #2a2a2a; border: 1px solid #444; color: #ccc; cursor: pointer; border-radius: 3px; }
.btn-tiny:hover { background: #3a3a3a; }
.stale-events-panel { margin-top: 1rem; }
.stale-events-table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
.stale-events-table th, .stale-events-table td { padding: 5px 8px; border: 1px solid #333; text-align: center; }
.stale-events-table th { background: #222; color: #aaa; }
```

---

### 4.6 Testing Phase 4

**Unit test — blame tracer:**
```bash
python - <<'EOF'
import h5py, numpy as np, tempfile
from pathlib import Path
from backend.layer_health import LayerHealthAnalyzer
from backend.blame_tracer import BlameTracer

with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
    path = Path(f.name)

N = 100
with h5py.File(path, 'w') as f:
    ts = np.arange(N, dtype=float) * 0.05
    f.create_dataset("vehicle/timestamp", data=ts)
    conf = np.ones(N) * 0.9
    conf[40:48] = 0.05    # stale event at frame 40
    f.create_dataset("perception/confidence", data=conf)
    f.create_dataset("perception/num_lanes_detected", data=np.full(N, 2))
    f.create_dataset("perception/stale_frame", data=(conf < 0.1).astype(int))
    lat_err = np.ones(N) * 0.1
    lat_err[45:55] = 0.5  # control error peaks 5 frames after stale
    f.create_dataset("control/lateral_error", data=lat_err)
    f.create_dataset("control/commanded_jerk", data=np.zeros(N))

tracer = BlameTracer(path)
blame = tracer.trace_blame(target_frame=50, metric='lateral_error')
print(f"Primary blame: {blame['primary_blame']} (lag={blame['lag_frames']}f, conf={blame['confidence']})")
assert blame['primary_blame'] == 'perception', f"Expected perception, got {blame['primary_blame']}"

stale = tracer.find_stale_propagation()
print(f"Stale events: {stale['event_count']}")
assert stale['event_count'] >= 1
print("✓ Phase 4 backend unit tests passed")
EOF
```

**API test:**
```bash
curl -s "http://localhost:5001/api/recording/recording_20260222_202957.h5/blame-trace?frame=412&metric=lateral_error" | python -m json.tool
curl -s "http://localhost:5001/api/recording/recording_20260222_202957.h5/stale-propagation" | python -m json.tool
```

**Visual test:**
1. Load a recording with known stale perception (s_loop with stale frames)
2. Click Chain tab → scroll to a frame where a spike is visible
3. Click "Trace Blame" → Blame Panel appears with propagation arrows
4. Verify: source frame is ≥ 2 frames before spike frame
5. Click "Stale Events" panel → table lists stale events with lag columns
6. Click "→ Frame" in a stale event row → navigates to that frame

---

## Phase 5 — Triage Report + Code Pointers

**Goal:** Session-level attribution and action checklist. Maps failure patterns to code locations.
**User value:** "What should I do about it?" answered with ordered, config-lever-specific actions.
**Depends on:** Phase 3 (layer health) + Phase 4 (stale propagation events).

---

### 5.1 New file: `backend/triage_engine.py`

```python
"""
Triage engine for PhilViz Phase 5.

Matches failure patterns against a library of known issue signatures,
computes per-layer attribution, and generates an ordered action checklist.

Pattern library: maps (signal conditions) → (code pointer, config lever, fix hint).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import h5py


# ── Pattern Library ───────────────────────────────────────────────────────────
# Each pattern: id, name, severity, signature_fn(metrics) → bool, code_pointer,
# config_lever, fix_hint.
# severity: "safety" > "instability" > "comfort"

PATTERNS = [
    {
        "id": "safety_ool_event",
        "name": "Out-of-lane event",
        "severity": "safety",
        "code_pointer": "av_stack.py:check_safety_limits() (~line 4100)",
        "config_lever": "safety.emergency_stop_error",
        "fix_hint": "Review lateral tracking quality. If consistent OOL on curves: reduce target speed or increase lookahead.",
        "check": lambda m: m.get("out_of_lane_events", 0) >= 1,
    },
    {
        "id": "perc_stale_cascade",
        "name": "Perception stale cascade",
        "severity": "instability",
        "code_pointer": "av_stack.py:apply_lane_ema_gating() (~line 2341)",
        "config_lever": "perception.low_visibility_fallback_max_consecutive_frames",
        "fix_hint": "Reduce stale TTL from 8 → 5 frames. Or raise model_fallback_confidence_hard_threshold.",
        "check": lambda m: m.get("stale_rate", 0) > 0.15 and m.get("lateral_p95", 0) > 0.3,
    },
    {
        "id": "perc_low_confidence",
        "name": "Persistent low detection confidence",
        "severity": "instability",
        "code_pointer": "perception/inference.py:detect() — confidence scoring",
        "config_lever": "perception.model_fallback_confidence_hard_threshold",
        "fix_hint": "Check camera exposure / track lighting. If systematic: retrain segmentation model.",
        "check": lambda m: m.get("mean_perception_confidence", 1.0) < 0.4,
    },
    {
        "id": "perc_single_lane_only",
        "name": "Single-lane detection only (>30% frames)",
        "severity": "instability",
        "code_pointer": "av_stack.py:blend_lane_pair_with_previous() (~line 2500)",
        "config_lever": "perception.lane_center_gate_m",
        "fix_hint": "Reduce lane_center_gate_m (e.g. 0.3 → 0.2) to allow wider lane center updates.",
        "check": lambda m: m.get("single_lane_rate", 0) > 0.30,
    },
    {
        "id": "traj_ref_rate_clamped",
        "name": "Trajectory jitter (ref rate clamp active >20% frames)",
        "severity": "instability",
        "code_pointer": "trajectory/inference.py:compute_reference_point() (~line 445)",
        "config_lever": "trajectory.ref_x_rate_limit",
        "fix_hint": "Increase ref_x_rate_limit (0.22 → 0.30) to reduce clamp frequency. Check perception noise first.",
        "check": lambda m: m.get("ref_rate_clamp_rate", 0) > 0.20,
    },
    {
        "id": "traj_overcorrection",
        "name": "Trajectory overcorrection (clamp + high lateral error)",
        "severity": "instability",
        "code_pointer": "trajectory/inference.py:compute_curve_feedforward() (~line 510)",
        "config_lever": "trajectory.curve_feedforward_gain",
        "fix_hint": "Reduce curve_feedforward_gain. Clamping + high error together suggest feedforward is overcorrecting.",
        "check": lambda m: m.get("ref_rate_clamp_rate", 0) > 0.15 and m.get("lateral_p95", 0) > 0.3,
    },
    {
        "id": "traj_short_lookahead",
        "name": "Lookahead too short for current speed",
        "severity": "comfort",
        "code_pointer": "trajectory/inference.py:get_lookahead_distance() (~line 380)",
        "config_lever": "trajectory.reference_lookahead_speed_table",
        "fix_hint": "Add/raise speed table entry for current target speed. Short lookahead at speed causes oscillation.",
        "check": lambda m: m.get("short_lookahead_rate", 0) > 0.05,
    },
    {
        "id": "ctrl_jerk_spike",
        "name": "Longitudinal jerk spikes (P95 > 4.5 m/s³)",
        "severity": "comfort",
        "code_pointer": "control/pid_controller.py:apply_jerk_limit() (~line 1800)",
        "config_lever": "control.longitudinal.jerk_cooldown_frames",
        "fix_hint": "Increase jerk_cooldown_frames (8 → 12) or reduce max_jerk_max (4.0 → 3.0).",
        "check": lambda m: m.get("commanded_jerk_p95", 0) > 4.5,
    },
    {
        "id": "ctrl_throttle_surge",
        "name": "Throttle surge (rate > limit >5% frames)",
        "severity": "comfort",
        "code_pointer": "control/pid_controller.py:apply_throttle_rate_limit() (~line 2100)",
        "config_lever": "control.longitudinal.throttle_rate_limit",
        "fix_hint": "Reduce throttle_rate_limit (0.04 → 0.03) for smoother acceleration.",
        "check": lambda m: m.get("throttle_surge_rate", 0) > 0.05,
    },
    {
        "id": "ctrl_steering_jerk",
        "name": "Steering jerk near or above cap",
        "severity": "comfort",
        "code_pointer": "control/pid_controller.py:apply_steering_jerk_limit() (~line 1650)",
        "config_lever": "control.lateral.pp_max_steering_jerk",
        "fix_hint": "If near cap (>17): system is healthy, cap working. If above 20: reduce pp_feedback_gain.",
        "check": lambda m: m.get("steering_jerk_max", 0) > 17.0,
    },
    {
        "id": "ctrl_lateral_oscillation",
        "name": "Lateral oscillation (positive osc_slope)",
        "severity": "instability",
        "code_pointer": "control/pid_controller.py:pure_pursuit_steer() (~line 1400)",
        "config_lever": "control.lateral.pp_feedback_gain",
        "fix_hint": "Reduce pp_feedback_gain (0.10 → 0.08). If persists: increase reference_smoothing (0.75 → 0.80).",
        "check": lambda m: m.get("osc_slope", -1) > 0.0,
    },
    {
        "id": "traj_curvature_error",
        "name": "Curvature measurement outliers (>2% frames)",
        "severity": "comfort",
        "code_pointer": "trajectory/utils.py:estimate_curvature() (~line 120)",
        "config_lever": "trajectory.curve_context_curvature_gain",
        "fix_hint": "High curvature outlier rate suggests noisy perception input. Check lane polynomial quality.",
        "check": lambda m: m.get("curvature_outlier_rate", 0) > 0.02,
    },
]

SEVERITY_ORDER = {"safety": 0, "instability": 1, "comfort": 2}


@dataclass
class PatternMatch:
    pattern_id:        str
    name:              str
    severity:          str
    occurrences:       int
    pct_of_recording:  float
    code_pointer:      str
    config_lever:      str
    fix_hint:          str
    example_frames:    list[int]


class TriageEngine:
    def __init__(self, recording_path: Path) -> None:
        self.path = recording_path

    def generate_triage(self) -> dict:
        """Compute full triage report. Returns JSON-ready dict."""
        metrics = self._extract_aggregate_metrics()
        health_summary, frame_health = self._get_layer_health()
        stale_events = self._get_stale_events()

        matched = self._match_patterns(metrics)
        attribution = self._compute_attribution(health_summary, stale_events)
        checklist = self._build_checklist(matched)
        overall = self._overall_health(health_summary)

        return {
            "recording_id":    self.path.stem,
            "total_frames":    metrics.get("total_frames", 0),
            "attribution":     attribution,
            "overall_health":  round(overall, 3),
            "matched_patterns": [
                {
                    "pattern_id":       m.pattern_id,
                    "name":             m.name,
                    "severity":         m.severity,
                    "occurrences":      m.occurrences,
                    "pct_of_recording": round(m.pct_of_recording, 4),
                    "code_pointer":     m.code_pointer,
                    "config_lever":     m.config_lever,
                    "fix_hint":         m.fix_hint,
                    "example_frames":   m.example_frames,
                }
                for m in matched
            ],
            "action_checklist": checklist,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _extract_aggregate_metrics(self) -> dict:
        """Read HDF5 and compute aggregate metrics needed by pattern checks."""
        m: dict = {}
        with h5py.File(self.path, 'r') as f:
            def arr(key, default=None):
                if key not in f:
                    return default
                return np.array(f[key][:])

            n = self._frame_count(f)
            m["total_frames"] = n

            # Perception
            conf = arr("perception/confidence")
            stale = arr("perception/stale_frame")
            n_lanes = arr("perception/num_lanes_detected")
            if conf is not None:
                m["mean_perception_confidence"] = float(np.mean(conf))
            if stale is not None:
                m["stale_rate"] = float(np.mean(stale.astype(bool)))
            if n_lanes is not None:
                m["single_lane_rate"] = float(np.mean(n_lanes == 1))

            # Trajectory
            clamped = arr("trajectory/ref_x_rate_clamped")
            lookahead = arr("trajectory/lookahead_distance")
            if clamped is not None:
                m["ref_rate_clamp_rate"] = float(np.mean(clamped.astype(bool)))
            if lookahead is not None:
                m["short_lookahead_rate"] = float(np.mean(lookahead < 8.0))

            # Control
            lat_err = arr("control/lateral_error")
            jerk = arr("control/commanded_jerk")
            steer_jerk = arr("control/steering_jerk")
            accel = arr("control/acceleration")
            throttle = arr("control/throttle_command")
            ool = arr("safety/out_of_lane")
            if lat_err is not None:
                m["lateral_p95"] = float(np.percentile(np.abs(lat_err), 95))
            if jerk is not None:
                m["commanded_jerk_p95"] = float(np.percentile(np.abs(jerk), 95))
            if steer_jerk is not None:
                m["steering_jerk_max"] = float(np.max(np.abs(steer_jerk)))
            if ool is not None:
                m["out_of_lane_events"] = int(np.sum(np.diff(ool.astype(int)) > 0))

            # Oscillation slope (simple linear regression on |lateral_error|)
            if lat_err is not None and len(lat_err) > 10:
                x = np.arange(len(lat_err))
                slope = np.polyfit(x, np.abs(lat_err), 1)[0]
                m["osc_slope"] = float(slope)

        return m

    def _match_patterns(self, metrics: dict) -> list[PatternMatch]:
        """Check all patterns against extracted metrics. Return sorted matches."""
        matches = []
        for pat in PATTERNS:
            try:
                fired = pat["check"](metrics)
            except Exception:
                fired = False
            if not fired:
                continue
            # Occurrences: use total_frames × relevant rate if available, else 1
            n = metrics.get("total_frames", 1)
            rate_key_map = {
                "perc_stale_cascade":    "stale_rate",
                "perc_low_confidence":   "stale_rate",
                "traj_ref_rate_clamped": "ref_rate_clamp_rate",
                "traj_short_lookahead":  "short_lookahead_rate",
            }
            rate = metrics.get(rate_key_map.get(pat["id"], ""), 0.0) or (1 / n)
            matches.append(PatternMatch(
                pattern_id=pat["id"],
                name=pat["name"],
                severity=pat["severity"],
                occurrences=max(1, int(n * rate)),
                pct_of_recording=rate,
                code_pointer=pat["code_pointer"],
                config_lever=pat["config_lever"],
                fix_hint=pat["fix_hint"],
                example_frames=[],   # Phase 5b enhancement: find actual example frames
            ))
        # Sort: safety first, then instability, then comfort; within tier by occurrences
        matches.sort(key=lambda m: (SEVERITY_ORDER[m.severity], -m.occurrences))
        return matches

    def _build_checklist(self, matches: list[PatternMatch]) -> list[str]:
        """Build ordered action checklist from matched patterns."""
        items = []
        for i, m in enumerate(matches, 1):
            items.append(
                f"{i}. [{m.config_lever}] {m.fix_hint}"
            )
        return items

    def _compute_attribution(self, health_summary: dict, stale_events: dict) -> dict:
        """
        Attribution = per-layer fraction of 'unhealthy' frames.
        Normalized to sum to 1.0.
        """
        raw = {}
        for layer in ("perception", "trajectory", "control"):
            raw[layer] = health_summary.get(layer, {}).get("pct_unhealthy", 0.0)
        total = sum(raw.values())
        if total < 1e-9:
            return {"perception": 0.33, "trajectory": 0.33, "control": 0.34}
        return {k: round(v / total, 3) for k, v in raw.items()}

    def _overall_health(self, health_summary: dict) -> float:
        scores = [health_summary.get(layer, {}).get("mean_score", 0.8)
                  for layer in ("perception", "trajectory", "control")]
        return float(np.mean(scores))

    def _get_layer_health(self):
        from backend.layer_health import LayerHealthAnalyzer
        result = LayerHealthAnalyzer(self.path).compute()
        return result["summary"], result["frames"]

    def _get_stale_events(self):
        from backend.blame_tracer import BlameTracer
        return BlameTracer(self.path).find_stale_propagation()

    def _frame_count(self, f: h5py.File) -> int:
        for key in ("vehicle/timestamp", "control/steering", "perception/confidence"):
            if key in f:
                return len(f[key])
        return 0
```

---

### 5.2 Edit: `server.py` — add triage route

```python
@app.route('/api/recording/<path:filename>/triage-report')
def get_triage_report(filename):
    """Generate session-level triage report with pattern attribution and action checklist."""
    from urllib.parse import unquote
    filename = unquote(filename)
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": f"Recording not found: {filename}"}), 404
    try:
        from backend.triage_engine import TriageEngine
        result = TriageEngine(filepath).generate_triage()
        return jsonify(numpy_to_list(result))
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
```

---

### 5.3 Edit: `index.html` — add Triage tab

**Tab button** (add after the Layers button from Phase 3):
```html
<button class="tab-btn" data-tab="triage">Triage</button>
```

**Tab pane** (add after `#layers-tab`):
```html
<div class="tab-pane" id="triage-tab">
    <div id="triage-content"></div>
</div>
```

---

### 5.4 Edit: `visualizer.js` — Triage tab logic

**Wire switchTab():**
```javascript
} else if (tabName === 'triage') {
    this.loadTriage();
}
```

**Add loadTriage():**
```javascript
async loadTriage() {
    if (!this.currentRecording) return;
    const content = document.getElementById('triage-content');
    if (!content) return;
    content.innerHTML = '<p style="color:#888;text-align:center;padding:2rem;">Generating triage report…</p>';
    try {
        const response = await fetch(`/api/recording/${this.currentRecording}/triage-report`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (data.error) {
            content.innerHTML = `<p style="color:#ff6b6b;">${this.escapeHtml(data.error)}</p>`;
            return;
        }
        content.innerHTML = this.renderTriageHtml(data);
        this._drawAttributionPie(data.attribution);
        this._wireTriageExport(data);
    } catch (e) {
        content.innerHTML = `<p style="color:#ff6b6b;">Error: ${this.escapeHtml(e.message)}</p>`;
    }
}
```

**Add renderTriageHtml():**
```javascript
renderTriageHtml(data) {
    const { attribution, overall_health, matched_patterns, action_checklist, recording_id, total_frames } = data;
    const healthPct = (overall_health * 100).toFixed(1);
    const sev_color = { safety: '#f44336', instability: '#ff9800', comfort: '#4caf50' };

    // Patterns table
    let patternsHtml = '';
    if (!matched_patterns.length) {
        patternsHtml = '<p style="color:#4caf50;">✓ No significant failure patterns detected.</p>';
    } else {
        patternsHtml = '<table class="triage-pattern-table"><thead><tr>' +
            '<th>Pattern</th><th>Severity</th><th>Occurrences</th><th>Code Pointer</th><th>Config Lever</th><th></th>' +
            '</tr></thead><tbody>';
        for (const pat of matched_patterns) {
            patternsHtml += `<tr>
                <td title="${this.escapeHtml(pat.fix_hint)}">${this.escapeHtml(pat.name)}</td>
                <td><span class="sev-badge" style="color:${sev_color[pat.severity]||'#ccc'};">${pat.severity}</span></td>
                <td>${pat.occurrences} (${(pat.pct_of_recording * 100).toFixed(1)}%)</td>
                <td class="code-pointer">${this.escapeHtml(pat.code_pointer)}</td>
                <td class="config-lever">${this.escapeHtml(pat.config_lever)}</td>
                <td>${pat.example_frames.length ?
                    `<button class="btn-small" onclick="visualizer.goToFrame(${pat.example_frames[0]})">→ Frame</button>` :
                    ''}</td>
            </tr>`;
        }
        patternsHtml += '</tbody></table>';
    }

    // Action checklist
    let checklistHtml = action_checklist.map(item =>
        `<li><label><input type="checkbox"> ${this.escapeHtml(item)}</label></li>`
    ).join('');

    return `
        <div class="triage-container">
            <div class="triage-header">
                <h3>Triage Report — ${this.escapeHtml(recording_id)}</h3>
                <span style="color:#888;font-size:0.85em;">${total_frames} frames</span>
                <button id="triage-export-btn" class="btn btn-secondary" style="float:right;">Export JSON</button>
            </div>

            <div class="triage-top-row">
                <div class="triage-attribution">
                    <h4>Error Attribution</h4>
                    <canvas id="attribution-pie" width="200" height="200"></canvas>
                    <div class="attribution-legend">
                        <span style="color:#7ecfff;">■ Perception ${(attribution.perception * 100).toFixed(0)}%</span>
                        <span style="color:#a8e6a3;">■ Trajectory ${(attribution.trajectory * 100).toFixed(0)}%</span>
                        <span style="color:#f0b27a;">■ Control ${(attribution.control * 100).toFixed(0)}%</span>
                    </div>
                </div>
                <div class="triage-health">
                    <h4>Overall Health</h4>
                    <div class="health-meter">
                        <div class="health-bar" style="width:${healthPct}%;background:${overall_health > 0.8 ? '#4caf50' : overall_health > 0.6 ? '#ff9800' : '#f44336'};"></div>
                    </div>
                    <span class="health-value">${healthPct}%</span>
                </div>
            </div>

            <h4>Detected Patterns (${matched_patterns.length})</h4>
            ${patternsHtml}

            <h4>Recommended Actions</h4>
            <ul class="triage-checklist">${checklistHtml}</ul>
        </div>`;
}
```

**Add _drawAttributionPie():**
```javascript
_drawAttributionPie(attribution) {
    const canvas = document.getElementById('attribution-pie');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const cx = 100, cy = 100, r = 80;
    const colors = { perception: '#7ecfff', trajectory: '#a8e6a3', control: '#f0b27a' };
    let startAngle = -Math.PI / 2;
    for (const [layer, fraction] of Object.entries(attribution)) {
        const sliceAngle = fraction * 2 * Math.PI;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r, startAngle, startAngle + sliceAngle);
        ctx.closePath();
        ctx.fillStyle = colors[layer] || '#888';
        ctx.fill();
        startAngle += sliceAngle;
    }
}
```

**Add _wireTriageExport():**
```javascript
_wireTriageExport(data) {
    const btn = document.getElementById('triage-export-btn');
    if (!btn) return;
    btn.onclick = () => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `triage_${data.recording_id}.json`;
        a.click();
        URL.revokeObjectURL(url);
    };
}
```

---

### 5.5 CSS additions for Phase 5 (add to `style.css`)

```css
.triage-container { padding: 1rem; }
.triage-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
.triage-top-row { display: flex; gap: 2rem; margin-bottom: 1.5rem; }
.triage-attribution { text-align: center; }
.attribution-legend { display: flex; flex-direction: column; gap: 4px; font-size: 0.85em; margin-top: 0.5rem; }
.triage-health { flex: 1; display: flex; flex-direction: column; justify-content: center; }
.health-meter { background: #333; border-radius: 4px; height: 24px; width: 100%; overflow: hidden; }
.health-bar { height: 100%; border-radius: 4px; transition: width 0.3s; }
.health-value { font-size: 1.4em; font-weight: bold; margin-top: 0.5rem; display: block; }
.triage-pattern-table { width: 100%; border-collapse: collapse; font-size: 0.83em; margin-bottom: 1.5rem; }
.triage-pattern-table th, .triage-pattern-table td { padding: 6px 8px; border: 1px solid #333; vertical-align: top; }
.triage-pattern-table th { background: #222; color: #aaa; }
.code-pointer { font-family: monospace; font-size: 0.78em; color: #a8e6a3; }
.config-lever { font-family: monospace; font-size: 0.78em; color: #7ecfff; }
.sev-badge { font-weight: bold; font-size: 0.85em; }
.triage-checklist { list-style: none; padding: 0; }
.triage-checklist li { padding: 6px 0; border-bottom: 1px solid #222; font-size: 0.88em; }
.triage-checklist label { cursor: pointer; display: flex; gap: 8px; align-items: flex-start; }
```

---

### 5.6 Testing Phase 5

**Unit test — triage engine:**
```bash
python - <<'EOF'
import h5py, numpy as np, tempfile
from pathlib import Path
from backend.triage_engine import TriageEngine

with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
    path = Path(f.name)

N = 500
with h5py.File(path, 'w') as f:
    f.create_dataset("vehicle/timestamp", data=np.arange(N) * 0.05)
    # Stale cascade: 25% stale rate → should trigger perc_stale_cascade
    stale = np.zeros(N)
    stale[50:175] = 1   # 25% stale
    f.create_dataset("perception/stale_frame", data=stale)
    f.create_dataset("perception/confidence", data=np.where(stale, 0.05, 0.9))
    f.create_dataset("perception/num_lanes_detected", data=np.full(N, 2))
    f.create_dataset("control/lateral_error", data=np.random.uniform(0, 0.35, N))
    f.create_dataset("control/commanded_jerk", data=np.random.uniform(0, 1.0, N))

report = TriageEngine(path).generate_triage()
assert report["total_frames"] == N
matched_ids = [p["pattern_id"] for p in report["matched_patterns"]]
assert "perc_stale_cascade" in matched_ids, f"Expected perc_stale_cascade in {matched_ids}"
assert len(report["action_checklist"]) > 0
print(f"Attribution: {report['attribution']}")
print(f"Patterns: {matched_ids}")
print(f"Checklist: {report['action_checklist'][0]}")
print("✓ Phase 5 backend unit tests passed")
EOF
```

**API test:**
```bash
curl -s "http://localhost:5001/api/recording/recording_20260222_202957.h5/triage-report" | python -m json.tool
# Expect: attribution dict, matched_patterns list, action_checklist list
```

**Visual test:**
1. Load a recording
2. Click "Triage" tab
3. Verify: pie chart renders with three colored slices
4. Verify: health bar renders, percentage makes sense
5. If patterns detected: verify code pointer is readable, "→ Frame" jumps to that frame
6. Click "Export JSON" → file downloads with correct filename

**End-to-end test on known-good s_loop recording:**
```bash
# Should show low error attribution, no safety patterns
curl -s "http://localhost:5001/api/recording/recording_20260222_202957.h5/triage-report" \
  | python -c "import sys,json; d=json.load(sys.stdin); print('Overall health:', d['overall_health']); \
    print('Safety patterns:', [p['pattern_id'] for p in d['matched_patterns'] if p['severity']=='safety'])"
# Expected: overall_health > 0.8, safety patterns = []
```

---

## Dependency Order

```
Phase 3 (layer_health.py)
       ↓
Phase 4 (blame_tracer.py — imports LayerHealthAnalyzer)
       ↓
Phase 5 (triage_engine.py — imports LayerHealthAnalyzer + BlameTracer)
```

Always implement and verify Phase 3 before starting Phase 4. Always verify Phase 4 before starting Phase 5.

---

## Task Registry

| Task ID | Phase | File(s) | Status |
|---|---|---|---|
| T-012a | 3 | `backend/layer_health.py` (new) + `server.py` (route) | Pending |
| T-012b | 3 | `index.html` (tab), `visualizer.js` (loadLayers, renderLayersHtml, _initLayersCanvases), `style.css` | Pending |
| T-012c | 4 | `backend/blame_tracer.py` (new) + `server.py` (2 routes) | Pending |
| T-012d | 4 | `index.html` (blame panel, stale events panel), `visualizer.js` (showBlamePanel, loadStaleEvents, Trace Blame buttons), `style.css` | Pending |
| T-012e | 5 | `backend/triage_engine.py` (new) + `server.py` (route) | Pending |
| T-012f | 5 | `index.html` (tab), `visualizer.js` (loadTriage, renderTriageHtml, _drawAttributionPie, _wireTriageExport), `style.css` | Pending |

---

## Retired Scope (from old Phase 3 placeholder)

The following items were listed as "Phase 3" in the original README but are **retired**:

- ~~Perception Replay~~ — superseded by Phase 3/4 attribution + existing perception-locked replay in `tools/analyze/`
- ~~Calibration Assistant~~ — not on current roadmap

---

## Cross-Cutting Notes

### HDF5 field availability

All scorers must handle missing fields gracefully — older recordings may not have every field.
Use the `_scalar()` / `arr()` pattern with defaults. Never crash on missing fields.

### Performance budget

- Phase 3 (`/layer-health`): target < 2s for 1200-frame recording
- Phase 4 (`/blame-trace`): target < 0.5s (single frame lookup + health data)
- Phase 4 (`/stale-propagation`): target < 1s (iterates once over health data)
- Phase 5 (`/triage-report`): target < 3s (calls Phase 3 + 4 + HDF5 aggregate scan)

Use numpy array operations over the full recording — no per-frame Python loops.

### No server-side caching needed

All routes in server.py are stateless — no `@lru_cache` or in-memory dicts.
The JavaScript `DataLoader` Map-based cache handles repeated calls.
For triage-report (slowest), consider adding a simple dict cache keyed by `recording_path.stat().st_mtime`
if latency is unacceptable.
