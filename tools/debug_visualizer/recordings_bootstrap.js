// Bootstrap script — populates the recording-select dropdown directly,
// independently of the Visualizer class. This is defensive: if visualizer.js
// errors during construction (e.g., from a missing element or a bad cast),
// the dropdown still shows recordings so the user isn't stranded.
//
// Loads BEFORE visualizer.js. Visualizer.loadRecordings() will overwrite
// what we put here when it eventually runs — that's fine, idempotent.

(function () {
    "use strict";

    function init() {
        const sel = document.getElementById("recording-select");
        if (!sel) return;
        // Clear placeholder if user just landed
        fetch("/api/recordings")
            .then((r) => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            })
            .then((recordings) => {
                if (!Array.isArray(recordings)) {
                    sel.title = "API returned non-array";
                    return;
                }
                // Preserve existing selected value if any
                const currentValue = sel.value || "";
                sel.innerHTML = '<option value="">Select recording…</option>';
                if (recordings.length === 0) {
                    const opt = document.createElement("option");
                    opt.value = "";
                    opt.disabled = true;
                    opt.textContent = "No recordings on disk";
                    sel.appendChild(opt);
                    return;
                }
                recordings.forEach((rec) => {
                    const opt = document.createElement("option");
                    opt.value = rec.filename || "";
                    const prov = rec.recording_provenance || {};
                    const trackId = (prov.track_id || rec.track_id || "").trim();
                    const sha = (prov.git_sha_short || rec.git_sha_short || "").trim();
                    const tag =
                        trackId && trackId !== "unknown"
                            ? ` [${trackId}]`
                            : "";
                    const shaTag =
                        sha && sha !== "unknown" ? ` ${sha}` : "";
                    opt.textContent = `${rec.display_name || rec.filename}${tag}${shaTag}`;
                    sel.appendChild(opt);
                });
                if (currentValue) sel.value = currentValue;
                // Tiny diagnostic so we know this fired (visible only by hover):
                sel.title = `${recordings.length} recordings loaded by bootstrap @ ${new Date().toLocaleTimeString()}`;
            })
            .catch((err) => {
                sel.title = `Bootstrap error: ${err.message}`;
                const opt = document.createElement("option");
                opt.value = "";
                opt.disabled = true;
                opt.textContent = `(error: ${err.message})`;
                sel.appendChild(opt);
            });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
