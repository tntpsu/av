// Shared cross-site nav drawer for PhilViz pages.
// Used by both index.html (Analysis) and skills.html (Skills).
// Reads sites from /api/sites/list, renders a slide-in drawer triggered
// by the hamburger button (#phv-hamburger).

(function () {
    "use strict";

    function init() {
        const drawer = document.getElementById("phv-drawer");
        const backdrop = document.getElementById("phv-drawer-backdrop");
        const closeBtn = document.getElementById("phv-drawer-close");
        const list = document.getElementById("phv-sites-list");
        const burgers = document.querySelectorAll("#phv-hamburger, .phv-hamburger");
        if (!drawer || !backdrop || !list) return;

        let loaded = false;

        function open() {
            drawer.classList.add("open");
            backdrop.classList.add("open");
            drawer.setAttribute("aria-hidden", "false");
            if (!loaded) loadSites();
        }
        function close() {
            drawer.classList.remove("open");
            backdrop.classList.remove("open");
            drawer.setAttribute("aria-hidden", "true");
        }

        burgers.forEach((b) => b.addEventListener("click", open));
        if (closeBtn) closeBtn.addEventListener("click", close);
        backdrop.addEventListener("click", close);
        // Esc closes the drawer
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && drawer.classList.contains("open")) close();
        });

        async function loadSites() {
            try {
                const r = await fetch("/api/sites/list");
                const sites = await r.json();
                if (!Array.isArray(sites)) {
                    list.innerHTML = `<div class="phv-empty">${sites.error || "(no sites)"}</div>`;
                    return;
                }
                list.innerHTML = sites
                    .map((s) => {
                        const icon = s.icon || "🔗";
                        const name = (s.name || s.url || "").replace(/[<>&]/g, "");
                        const url = (s.url || "#").replace(/"/g, "&quot;");
                        const ext = url.startsWith("http") && !url.startsWith(location.origin)
                            ? ` target="_blank" rel="noopener"`
                            : "";
                        return `<a class="phv-site-row" href="${url}"${ext}>
                            <span class="phv-site-icon">${icon}</span>
                            <span class="phv-site-name">${name}</span>
                        </a>`;
                    })
                    .join("");
                loaded = true;
            } catch (e) {
                list.innerHTML = `<div class="phv-empty">Error loading sites: ${e.message}</div>`;
            }
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
