// PhilViz Tracks page — multi-select picker that hands selection off to /skills
// via ?tracks= URL param. Replaces having to type track names by hand.
(function () {
    "use strict";

    const $ = (id) => document.getElementById(id);
    const selected = new Set();

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

    function renderList(container, items) {
        if (!items.length) {
            container.innerHTML = `<div class="phv-empty">(none)</div>`;
            return;
        }
        container.innerHTML = items.map((t) => {
            const rec = t.latest_recording;
            const recAge = rec ? relTime(rec.mtime) : "(no recording)";
            const recName = rec ? rec.name : "—";
            const recCls = rec ? "" : "muted";
            return `
                <label class="phv-tracks-row" data-name="${escapeHtml(t.name)}">
                    <input type="checkbox" data-name="${escapeHtml(t.name)}">
                    <div class="phv-tracks-row-content">
                        <span class="phv-tracks-row-name phv-mono">${escapeHtml(t.name)}</span>
                        <span class="phv-tracks-row-meta ${recCls}" title="${escapeHtml(recName)}">latest: ${recAge}</span>
                    </div>
                </label>
            `;
        }).join("");
        container.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
            cb.addEventListener("change", () => {
                const n = cb.getAttribute("data-name");
                if (cb.checked) selected.add(n); else selected.delete(n);
                updateActions();
            });
        });
    }

    function updateActions() {
        const n = selected.size;
        $("phv-tracks-selected-count").textContent = `${n} selected`;
        const btn = $("phv-tracks-run-btn");
        btn.disabled = n === 0;
        btn.textContent = n === 0 ? "⚡ Select tracks to run a skill"
                                  : `⚡ Run skill on ${n} track${n === 1 ? "" : "s"} →`;
    }

    function goToSkills() {
        if (selected.size === 0) return;
        const tracks = Array.from(selected).join(" ");
        window.location.href = `/skills?tracks=${encodeURIComponent(tracks)}`;
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

    $("phv-tracks-run-btn").addEventListener("click", goToSkills);
    load();
})();
