// Pattern 3 — context-aware ⚡ FAB on the Analysis page.
// Tap the floating button → drawer slides up listing skills relevant to
// recordings (/diagnose, /trace, /scores, /validate, /e2e). If a recording
// is currently loaded (selected in the dropdown), the skill link pre-fills
// the recording filename in the args via /skills?args=<filename>.

(function () {
    "use strict";

    // Skills that meaningfully target a single recording.
    const RECORDING_SKILLS = [
        { name: "diagnose", desc: "Run targeted diagnostic on this recording" },
        { name: "trace", desc: "Generate a trace script around an event in this recording" },
        { name: "scores", desc: "Show scoreboard for this recording's track" },
        { name: "validate", desc: "Run acceptance gate checks on this recording" },
        { name: "e2e", desc: "End-to-end re-run + analyze on the same scenario" },
    ];

    function getSelectedRecording() {
        const sel = document.getElementById("recording-select");
        if (!sel || !sel.value) return null;
        return sel.value;
    }

    function init() {
        const fab = document.getElementById("phv-context-fab");
        const drawer = document.getElementById("phv-context-drawer");
        const closeBtn = document.getElementById("phv-context-close");
        const banner = document.getElementById("phv-context-banner");
        const rows = document.getElementById("phv-context-skill-rows");
        if (!fab || !drawer || !rows) return;

        function open() {
            const rec = getSelectedRecording();
            if (rec) {
                banner.innerHTML = `Recording: <code>${rec}</code> — args will pre-fill this filename.`;
            } else {
                banner.textContent = "No recording loaded — skills will run with no specific target.";
            }
            rows.innerHTML = RECORDING_SKILLS.map((s) => {
                const argsParam = rec ? `&args=${encodeURIComponent(rec)}` : "";
                const url = `/skills?skill=${encodeURIComponent(s.name)}${argsParam}`;
                return `<a class="phv-context-skill-row" href="${url}">
                    <span class="phv-context-skill-name">/${s.name}</span>
                    <span class="phv-context-skill-desc">${s.desc}</span>
                </a>`;
            }).join("");
            drawer.classList.add("open");
        }
        function close() { drawer.classList.remove("open"); }

        fab.addEventListener("click", open);
        if (closeBtn) closeBtn.addEventListener("click", close);
        // Tap outside drawer (on the FAB area) closes — implemented via Esc only for now.
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && drawer.classList.contains("open")) close();
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
