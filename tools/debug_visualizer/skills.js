// PhilViz Skills page — minimal client.
// Lists available skills, allows running them via /api/skills/run,
// streams output via SSE, supports cancel, lists recent jobs.

(function () {
    "use strict";

    const els = {
        list: document.getElementById("phv-skill-list-items"),
        jobs: document.getElementById("phv-jobs-list"),
        emptyState: document.getElementById("phv-empty-state"),
        runPanel: document.getElementById("phv-run-panel"),
        skillName: document.getElementById("phv-skill-name"),
        skillDesc: document.getElementById("phv-skill-description"),
        argsHint: document.getElementById("phv-args-hint"),
        argsInput: document.getElementById("phv-args-input"),
        runBtn: document.getElementById("phv-run-btn"),
        cancelBtn: document.getElementById("phv-cancel-btn"),
        jobStatus: document.getElementById("phv-job-status"),
        outputWrap: document.getElementById("phv-output-wrapper"),
        outputMeta: document.getElementById("phv-output-meta"),
        output: document.getElementById("phv-output"),
        backBtn: document.getElementById("phv-back-btn"),
    };

    let selectedSkill = null;
    let activeJobId = null;
    let activeStream = null;
    let lineCount = 0;

    function showRunPanel() {
        els.emptyState.style.display = "none";
        els.runPanel.style.display = "flex";
    }

    function hideRunPanel() {
        els.runPanel.style.display = "none";
        els.emptyState.style.display = "flex";
    }

    function selectSkill(skill) {
        selectedSkill = skill;
        els.skillName.textContent = "/" + skill.name;
        els.skillDesc.textContent = skill.description || "(no description)";
        els.argsHint.textContent = skill.takes_args ? "(this skill uses $ARGUMENTS)" : "";
        els.argsInput.value = "";
        els.outputWrap.style.display = "none";
        els.output.textContent = "";
        els.jobStatus.textContent = "";
        els.runBtn.disabled = false;
        els.cancelBtn.style.display = "none";
        showRunPanel();
        // Highlight the selected row
        document.querySelectorAll(".phv-skill-row.selected").forEach((r) => {
            r.classList.remove("selected");
        });
        const row = document.querySelector(`.phv-skill-row[data-skill="${skill.name}"]`);
        if (row) row.classList.add("selected");
    }

    function renderSkills(skills) {
        if (!skills.length) {
            els.list.innerHTML = `<div class="phv-empty">No skills found in .claude/commands/</div>`;
            return;
        }
        els.list.innerHTML = skills
            .map(
                (s) => `
            <div class="phv-skill-row" data-skill="${s.name}">
                <div class="phv-skill-row-name">/${s.name}</div>
                <div class="phv-skill-row-desc">${escapeHtml(s.description || "")}</div>
            </div>`
            )
            .join("");
        els.list.querySelectorAll(".phv-skill-row").forEach((row) => {
            row.addEventListener("click", () => {
                const name = row.getAttribute("data-skill");
                const skill = skills.find((s) => s.name === name);
                if (skill) selectSkill(skill);
            });
        });
    }

    function renderJobs(jobs) {
        if (!jobs.length) {
            els.jobs.innerHTML = `<div class="phv-empty">No runs yet</div>`;
            return;
        }
        els.jobs.innerHTML = jobs
            .map((j) => {
                const ago = relativeTime(j.started_at);
                const cls =
                    j.status === "running"
                        ? "running"
                        : j.status === "completed"
                        ? "ok"
                        : "fail";
                return `
                <div class="phv-job-row" data-job-id="${j.job_id}" data-skill="${j.skill}">
                    <span class="phv-job-status-dot ${cls}"></span>
                    <span class="phv-job-skill">/${j.skill}</span>
                    <span class="phv-job-meta">${j.status} · ${ago}</span>
                </div>`;
            })
            .join("");
        els.jobs.querySelectorAll(".phv-job-row").forEach((row) => {
            row.addEventListener("click", () => {
                const jobId = row.getAttribute("data-job-id");
                attachToJob(jobId, row.getAttribute("data-skill"));
            });
        });
    }

    function attachToJob(jobId, skillName) {
        // Re-attach to a job that was started earlier (e.g., user reloaded).
        if (activeStream) {
            activeStream.close();
            activeStream = null;
        }
        activeJobId = jobId;
        lineCount = 0;
        els.skillName.textContent = "/" + (skillName || "?");
        els.outputWrap.style.display = "block";
        els.output.textContent = "(reconnecting…)\n";
        els.runBtn.disabled = true;
        els.cancelBtn.style.display = "inline-block";
        els.jobStatus.textContent = "streaming";
        showRunPanel();
        startStream(jobId);
    }

    function startStream(jobId) {
        const url = `/api/skills/stream/${jobId}?last=${lineCount}`;
        activeStream = new EventSource(url);
        activeStream.onmessage = (e) => {
            els.output.textContent += e.data + "\n";
            els.output.scrollTop = els.output.scrollHeight;
            lineCount++;
            els.outputMeta.textContent = `${lineCount} lines`;
        };
        activeStream.addEventListener("done", () => {
            els.jobStatus.textContent = "complete";
            els.runBtn.disabled = false;
            els.cancelBtn.style.display = "none";
            activeStream.close();
            activeStream = null;
            activeJobId = null;
            refreshJobs();
        });
        activeStream.onerror = () => {
            els.jobStatus.textContent = "stream error (will auto-retry)";
        };
    }

    async function runSkill() {
        if (!selectedSkill) return;
        els.runBtn.disabled = true;
        els.jobStatus.textContent = "starting…";
        els.output.textContent = "";
        lineCount = 0;
        try {
            const res = await fetch("/api/skills/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    skill: selectedSkill.name,
                    args: els.argsInput.value,
                }),
            });
            const data = await res.json();
            if (!res.ok) {
                els.jobStatus.textContent = "error: " + (data.error || res.statusText);
                els.runBtn.disabled = false;
                return;
            }
            activeJobId = data.job_id;
            els.outputWrap.style.display = "block";
            els.outputMeta.textContent = `job_id ${activeJobId}`;
            els.cancelBtn.style.display = "inline-block";
            els.jobStatus.textContent = "running";
            startStream(activeJobId);
            refreshJobs();
        } catch (e) {
            els.jobStatus.textContent = "error: " + e.message;
            els.runBtn.disabled = false;
        }
    }

    async function cancelJob() {
        if (!activeJobId) return;
        await fetch(`/api/skills/cancel/${activeJobId}`, { method: "POST" });
        els.jobStatus.textContent = "cancel requested";
    }

    async function refreshSkills() {
        try {
            const res = await fetch("/api/skills/list");
            const skills = await res.json();
            renderSkills(skills);
        } catch (e) {
            els.list.innerHTML = `<div class="phv-empty">Error loading skills: ${e.message}</div>`;
        }
    }

    async function refreshJobs() {
        try {
            const res = await fetch("/api/skills/jobs");
            const jobs = await res.json();
            renderJobs(jobs);
        } catch (e) {
            // silent
        }
    }

    // ---- helpers ----
    function escapeHtml(s) {
        return s.replace(/[&<>"']/g, (c) =>
            ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
        );
    }

    function relativeTime(ts) {
        const sec = Math.max(0, (Date.now() / 1000 - ts) | 0);
        if (sec < 60) return `${sec}s ago`;
        if (sec < 3600) return `${(sec / 60) | 0}m ago`;
        if (sec < 86400) return `${(sec / 3600) | 0}h ago`;
        return `${(sec / 86400) | 0}d ago`;
    }

    // ---- wire up ----
    els.runBtn.addEventListener("click", runSkill);
    els.cancelBtn.addEventListener("click", cancelJob);
    els.backBtn.addEventListener("click", hideRunPanel);

    refreshSkills();
    refreshJobs();
    setInterval(refreshJobs, 5000);
})();
