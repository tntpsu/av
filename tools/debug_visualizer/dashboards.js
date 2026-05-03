// PhilViz Dashboards — fetch all 3 nightly results in one call,
// render as mobile-friendly cards with status dots, gate badges, and
// per-track / per-scenario lines. Auto-refreshes every 60s.
(function () {
    "use strict";

    const $ = (id) => document.getElementById(id);

    function statusBadge(gate) {
        const cls = gate === "PASS" ? "ok" : gate === "FAIL" ? "fail" : "warn";
        return `<span class="phv-badge ${cls}">${gate || "?"}</span>`;
    }

    function relTime(ts) {
        if (!ts) return "—";
        // ts is ISO; convert to relative
        const t = new Date(ts).getTime() / 1000;
        const sec = Math.max(0, ((Date.now() / 1000) - t) | 0);
        if (sec < 60) return `${sec}s ago`;
        if (sec < 3600) return `${(sec / 60) | 0}m ago`;
        if (sec < 86400) return `${(sec / 3600) | 0}h ago`;
        return `${(sec / 86400) | 0}d ago`;
    }

    function fmtDelta(d) {
        if (d == null) return "—";
        const sign = d > 0 ? "+" : "";
        const cls = d < -1.0 ? "neg" : d > 0 ? "pos" : "zero";
        return `<span class="phv-delta ${cls}">${sign}${d.toFixed(1)}</span>`;
    }

    function renderFixTests(d) {
        const s = d.summary || {};
        $("phv-card-fix-tests-status").innerHTML = statusBadge(
            s.real_breaks > 0 ? "FAIL" : (s.flaky > 0 ? "WARN" : "PASS")
        );
        const flakyHTML = (d.flaky_list || []).slice(0, 6).map((f) =>
            `<li class="phv-mono">${escapeHtml(f.test)}</li>`
        ).join("");
        $("phv-card-fix-tests-body").innerHTML = `
            <div class="phv-stat-row">
                <div class="phv-stat"><span class="phv-stat-label">Total</span><span class="phv-stat-val">${s.total ?? "—"}</span></div>
                <div class="phv-stat"><span class="phv-stat-label">Fixed</span><span class="phv-stat-val">${s.fixed ?? "—"}</span></div>
                <div class="phv-stat"><span class="phv-stat-label">Real breaks</span><span class="phv-stat-val ${s.real_breaks > 0 ? 'fail' : ''}">${s.real_breaks ?? "—"}</span></div>
                <div class="phv-stat"><span class="phv-stat-label">Flaky</span><span class="phv-stat-val ${s.flaky > 0 ? 'warn' : ''}">${s.flaky ?? "—"}</span></div>
            </div>
            <div class="phv-stat-row">
                <div class="phv-stat-narrow"><span class="phv-stat-label">Delivery</span><span class="phv-stat-val phv-mono">${s.delivery || "—"}</span></div>
                <div class="phv-stat-narrow"><span class="phv-stat-label">Started</span><span class="phv-stat-val">${relTime(d.started_at)}</span></div>
            </div>
            ${flakyHTML ? `<details class="phv-details"><summary>Flaky tests (${(d.flaky_list || []).length})</summary><ul class="phv-list-tight">${flakyHTML}</ul></details>` : ""}
        `;
    }

    function renderSweep(d) {
        const s = d.summary || {};
        $("phv-card-sweep-status").innerHTML = statusBadge(s.gate);
        const tracksHTML = (d.tracks || []).map((t) => `
            <div class="phv-track-row ${t.flag ? 'flagged' : ''}">
                <a class="phv-track-name phv-mono" href="/skills?tracks=${encodeURIComponent(t.name)}" title="Run a skill on this track">${escapeHtml(t.name)}</a>
                <span class="phv-track-score">${t.score?.toFixed(1) ?? "—"}</span>
                <span class="phv-track-baseline">${t.baseline?.toFixed(1) ?? "—"}</span>
                ${fmtDelta(t.delta)}
                ${t.flag ? `<span class="phv-track-flag" title="FLAG=${escapeHtml(t.flag)}">⚑</span>` : `<span class="phv-track-flag-spacer"></span>`}
            </div>
        `).join("");
        $("phv-card-sweep-body").innerHTML = `
            <div class="phv-stat-row">
                <div class="phv-stat"><span class="phv-stat-label">Tracks</span><span class="phv-stat-val">${s.passed ?? 0}/6</span></div>
                <div class="phv-stat"><span class="phv-stat-label">Regressions</span><span class="phv-stat-val ${s.regressions > 0 ? 'fail' : ''}">${s.regressions ?? 0}</span></div>
                <div class="phv-stat"><span class="phv-stat-label">Flags</span><span class="phv-stat-val ${s.flags > 0 ? 'warn' : ''}">${s.flags ?? 0}</span></div>
                <div class="phv-stat"><span class="phv-stat-label">Started</span><span class="phv-stat-val">${relTime(d.started_at)}</span></div>
            </div>
            <div class="phv-track-table">
                <div class="phv-track-row phv-track-header">
                    <span>Track</span><span>Score</span><span>Base</span><span>Δ</span><span></span>
                </div>
                ${tracksHTML || `<div class="phv-empty">No tracks recorded</div>`}
            </div>
        `;
    }

    function renderAccSweep(d) {
        const s = d.summary || {};
        $("phv-card-acc-sweep-status").innerHTML = statusBadge(s.gate);
        const verdictColor = (v) =>
            v === "PASS" ? "ok" : v === "FAIL" ? "fail" : v === "WARN" ? "warn" :
            v === "SKIPPED" ? "skip" : v === "AMBIGUOUS" ? "warn" : "neutral";
        const scenariosHTML = (d.scenarios || []).map((sc) => `
            <div class="phv-scenario-row">
                <span class="phv-status-dot ${verdictColor(sc.verdict)}" title="${sc.verdict}"></span>
                <a class="phv-scenario-name phv-mono" href="/skills?tracks=${encodeURIComponent(sc.name)}" title="Run a skill on this scenario">${escapeHtml(sc.name)}</a>
                <span class="phv-scenario-verdict ${verdictColor(sc.verdict)}">${sc.verdict}</span>
                <span class="phv-scenario-reason">${escapeHtml(sc.reason)}</span>
            </div>
        `).join("");
        $("phv-card-acc-sweep-body").innerHTML = `
            <div class="phv-stat-row">
                <div class="phv-stat"><span class="phv-stat-label">Total</span><span class="phv-stat-val">${s.total ?? 0}</span></div>
                <div class="phv-stat"><span class="phv-stat-label">Pass</span><span class="phv-stat-val ok">${s.PASS ?? 0}</span></div>
                <div class="phv-stat"><span class="phv-stat-label">Fail</span><span class="phv-stat-val ${s.FAIL > 0 ? 'fail' : ''}">${s.FAIL ?? 0}</span></div>
                <div class="phv-stat"><span class="phv-stat-label">Skipped</span><span class="phv-stat-val">${s.SKIPPED ?? 0}</span></div>
            </div>
            ${scenariosHTML ? `<div class="phv-scenario-table">${scenariosHTML}</div>` : `<div class="phv-empty">No ACC sweep data yet — first run lands at 4 AM tonight</div>`}
        `;
    }

    function escapeHtml(s) {
        if (!s) return "";
        return String(s).replace(/[&<>"']/g, (c) =>
            ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
        );
    }

    async function refresh() {
        try {
            const r = await fetch("/api/dashboards/all");
            const d = await r.json();
            renderFixTests(d.fix_tests || {});
            renderSweep(d.sweep || {});
            renderAccSweep(d.acc_sweep || {});
        } catch (e) {
            ["fix-tests", "sweep", "acc-sweep"].forEach((k) => {
                const body = $(`phv-card-${k}-body`);
                if (body) body.innerHTML = `<div class="phv-empty">Error loading: ${escapeHtml(e.message)}</div>`;
            });
        }
    }

    refresh();
    setInterval(refresh, 60000);
})();
