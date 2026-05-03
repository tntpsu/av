// PhilViz Tracks page (v2 — browse-with-actions, per Option B redesign).
// Each row gets inline action buttons for the actual workflows. Multi-select
// is intentionally gone — that pattern lives inside the Skills page now,
// where it pairs with /sweep specifically.

(function () {
    "use strict";

    const $ = (id) => document.getElementById(id);

    function relTime(ts) {
        if (!ts) return "(no recording)";
        const sec = Math.max(0, ((Date.now() / 1000) - ts) | 0);
        if (sec < 86400) return `${(sec / 3600) | 0}h ago`;
        const days = (sec / 86400) | 0;
        if (days <= 30) return `${days}d ago`;
        return `${(days / 30) | 0}mo ago`;
    }

    function escapeHtml(s) {
        if (!s) return "";
        return String(s).replace(/[&<>"']/g, (c) =>
            ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
        );
    }

    function trackYamlPath(t) {
        return t.kind === "scenario" ? `tracks/scenarios/${t.name}.yml` : `tracks/${t.name}.yml`;
    }

    function renderRow(t) {
        const rec = t.latest_recording;
        const recAge = rec ? relTime(rec.mtime) : "(no recording)";
        const recName = rec ? rec.name : null;
        const yamlPath = trackYamlPath(t);

        // Build action buttons. Some are disabled when no recording exists.
        const loadAction = recName
            ? `<a class="phv-row-action" href="/?recording=${encodeURIComponent(recName)}" title="Open ${escapeHtml(recName)} in the Analysis page">
                 <span class="phv-action-icon">📺</span>
                 <span class="phv-action-label">Load latest</span>
               </a>`
            : `<span class="phv-row-action disabled" title="No recording exists for this track yet — run /e2e to create one">
                 <span class="phv-action-icon">📺</span>
                 <span class="phv-action-label">Load latest</span>
               </span>`;
        const diagnoseAction = recName
            ? `<a class="phv-row-action" href="/skills?skill=diagnose&args=${encodeURIComponent(recName)}" title="Run /diagnose on the latest recording">
                 <span class="phv-action-icon">🔍</span>
                 <span class="phv-action-label">Diagnose</span>
               </a>`
            : `<span class="phv-row-action disabled" title="No recording — run /e2e first">
                 <span class="phv-action-icon">🔍</span>
                 <span class="phv-action-label">Diagnose</span>
               </span>`;
        const e2eAction = `<a class="phv-row-action" href="/skills?skill=e2e&args=${encodeURIComponent(yamlPath)}" title="Run a fresh Unity drive on this track">
                <span class="phv-action-icon">🎬</span>
                <span class="phv-action-label">Run /e2e</span>
            </a>`;
        const scoresAction = `<a class="phv-row-action" href="/skills?skill=scores&args=${encodeURIComponent(t.name)}" title="Show the scoreboard for this track">
                <span class="phv-action-icon">📊</span>
                <span class="phv-action-label">Scores</span>
            </a>`;

        return `
            <article class="phv-tracks-row-v2">
                <header class="phv-tracks-row-header">
                    <span class="phv-tracks-row-name phv-mono">${escapeHtml(t.name)}</span>
                    <span class="phv-tracks-row-meta ${rec ? "" : "muted"}" title="${escapeHtml(recName || "")}">
                        latest: ${recAge}
                    </span>
                </header>
                <div class="phv-tracks-actions-grid">
                    ${loadAction}
                    ${diagnoseAction}
                    ${e2eAction}
                    ${scoresAction}
                </div>
            </article>`;
    }

    function renderList(container, items) {
        if (!items.length) {
            container.innerHTML = `<div class="phv-empty">(none)</div>`;
            return;
        }
        container.innerHTML = items.map(renderRow).join("");
    }

    async function load() {
        try {
            const r = await fetch("/api/tracks/with-metadata");
            const d = await r.json();
            renderList($("phv-base-tracks"), d.tracks || []);
            renderList($("phv-scenarios"), d.scenarios || []);
        } catch (e) {
            $("phv-base-tracks").innerHTML = `<div class="phv-empty">Error: ${escapeHtml(e.message)}</div>`;
        }
    }

    load();
})();
