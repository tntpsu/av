/**
 * Main visualizer application.
 * Orchestrates frame loading, display, and user interactions.
 */

class Visualizer {
    constructor() {
        this.dataLoader = new DataLoader();
        this.overlayRenderer = new OverlayRenderer(document.getElementById('overlay-canvas'));
        this.debugOverlayRenderer = null; // Will be set up for debug canvas
        
        this.currentFrameIndex = 0;
        this.frameCount = 0;
        this.currentRecording = null;  // Track current recording filename
        this.currentRecordingMeta = null;
        this.isPlaying = false;
        this.playSpeed = 1.0;
        this.playInterval = null;
        
        this.currentFrameData = null;
        this.previousPerceptionData = null;  // Track previous frame's perception data for change calculations
        this.currentLongitudinalMetrics = null;
        this.currentImage = null;
        this.currentTopdownImage = null;
        this.lastValidY8m = undefined;  // Cache last valid camera_8m_screen_y value
        this.groundTruthDistance = 7;  // Tunable distance for ground truth conversion (calibrated to account for camera pitch/height)
        this.generatedDebugCache = {};  // Cache for on-demand generated debug images
        this.lastUnityTime = null;
        this.lastUnityFrameIndex = null;
        this.lastUnityFrameCount = null;
        this.lastCameraTimestamp = null;
        this.availableSignals = [];
        this.chart = null;
        this.chartXAxisKey = null;
        this.chartTimeSeries = null;
        this.chartUsesTime = false;
        this.chartCursorValue = null;
        this.chartCursorIndex = null;
        this.chartCurvatureSeries = null;
        this.chartSeriesData = null;
        this.chartSeriesNames = [];
        this.chartXMin = null;
        this.chartXMax = null;
        this.chartSavedViews = this.loadChartViews();
        this.lastChartViewName = this.loadLastChartViewName();
        this.cameraGridHeightKey = 'debugVisualizer.cameraGridHeight';
        this.panelSplitLeftWidthKey = 'debugVisualizer.panelSplitLeftWidthPx';
        this.checkboxStateKey = 'debugVisualizer.checkboxState';
        this.topdownOrthoHalfSize = 12.0;
        this.topdownAvailable = true;
        this.showInformationalIssues = false;
        this.curveEntryStartDistanceM = 34.0;
        this.curveEntryWindowDistanceM = 8.0;
        this.distanceFromStartSeries = null;
        this.topdownSmoothedTrajectory = null;
        this.topdownSmoothedFrameIndex = null;
        this.topdownSmoothingAlpha = 0.45;
        this.topdownCalibratedProjectionReady = false;
        this.topdownProjectionToggleTouched = false;
        this.projectionDiagnostics = {};
        this.distanceScaleStartOffsetMeters = 2.0;
        this.projectionNearFieldBlendEnabled = true;
        this.projectionNearFieldGroundYOffsetMeters = -0.3;
        this.projectionNearFieldBlendDistanceMeters = 10.0;
        this.projectionNearFieldSettingsKey = 'debugVisualizer.projectionNearFieldSettings';
        
        this.setupEventListeners();
        this.loadRecordings();
    }

    escapeHtml(value) {
        return String(value ?? '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    }

    restoreCheckboxState() {
        try {
            const raw = localStorage.getItem(this.checkboxStateKey);
            if (!raw) return;
            const saved = JSON.parse(raw);
            if (!saved || typeof saved !== 'object') return;
            document.querySelectorAll('input[type="checkbox"][id]').forEach((el) => {
                if (Object.prototype.hasOwnProperty.call(saved, el.id)) {
                    el.checked = Boolean(saved[el.id]);
                }
            });
        } catch (error) {
            console.warn('Could not restore checkbox state:', error);
        }
    }

    persistCheckboxState() {
        try {
            const state = {};
            document.querySelectorAll('input[type="checkbox"][id]').forEach((el) => {
                state[el.id] = Boolean(el.checked);
            });
            localStorage.setItem(this.checkboxStateKey, JSON.stringify(state));
        } catch (error) {
            console.warn('Could not persist checkbox state:', error);
        }
    }

    restoreProjectionNearFieldSettings() {
        try {
            const raw = localStorage.getItem(this.projectionNearFieldSettingsKey);
            if (!raw) return;
            const saved = JSON.parse(raw);
            if (!saved || typeof saved !== 'object') return;
            if (Object.prototype.hasOwnProperty.call(saved, 'enabled')) {
                this.projectionNearFieldBlendEnabled = Boolean(saved.enabled);
            }
            if (Object.prototype.hasOwnProperty.call(saved, 'offset')) {
                const v = Number(saved.offset);
                if (Number.isFinite(v)) this.projectionNearFieldGroundYOffsetMeters = v;
            }
            if (Object.prototype.hasOwnProperty.call(saved, 'distance')) {
                const v = Number(saved.distance);
                if (Number.isFinite(v) && v > 0.1) this.projectionNearFieldBlendDistanceMeters = v;
            }
        } catch (error) {
            console.warn('Could not restore projection near-field settings:', error);
        }
    }

    persistProjectionNearFieldSettings() {
        try {
            localStorage.setItem(this.projectionNearFieldSettingsKey, JSON.stringify({
                enabled: Boolean(this.projectionNearFieldBlendEnabled),
                offset: Number(this.projectionNearFieldGroundYOffsetMeters),
                distance: Number(this.projectionNearFieldBlendDistanceMeters),
            }));
        } catch (error) {
            console.warn('Could not persist projection near-field settings:', error);
        }
    }

    ensureCheckboxTooltips() {
        // Provide at least a basic hover tooltip for any checkbox missing one.
        document.querySelectorAll('label').forEach((label) => {
            const cb = label.querySelector('input[type="checkbox"][id]');
            if (!cb) return;
            const text = (label.textContent || '').replace(/\s+/g, ' ').trim();
            if (text && !label.getAttribute('title')) {
                label.setAttribute('title', text);
            }
            if (text && !cb.getAttribute('title')) {
                cb.setAttribute('title', label.getAttribute('title') || text);
            }
        });
    }

    computeTurnSign(points, xKey = 'x', yKey = 'y') {
        if (!Array.isArray(points) || points.length < 2) return 'unknown';
        const valid = points
            .map((p) => ({ x: Number(p?.[xKey]), y: Number(p?.[yKey]) }))
            .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
            .sort((a, b) => a.y - b.y);
        if (valid.length < 2) return 'unknown';
        const first = valid[0];
        const last = valid[Math.min(valid.length - 1, Math.max(1, Math.floor(valid.length * 0.4)))];
        const dx = last.x - first.x;
        if (Math.abs(dx) < 1e-3) return 'straight';
        return dx > 0 ? 'right' : 'left';
    }

    getDistanceScaleSegments(stepMeters = 5, maxMeters = 30) {
        const traj = this.currentFrameData?.trajectory;
        if (!traj) return [];
        const sourcePoints = (Array.isArray(traj.oracle_points) && traj.oracle_points.length > 1)
            ? traj.oracle_points
            : this.getDisplayTrajectoryPoints(traj.trajectory_points || []);

        const center = this.toForwardMonotonicPath(sourcePoints);
        if (center.length < 2) return [];

        const gt = this.currentFrameData?.ground_truth || {};
        const left = Number(gt.left_lane_line_x ?? gt.left_lane_x);
        const right = Number(gt.right_lane_line_x ?? gt.right_lane_x);
        const laneWidth = (
            Number.isFinite(left) && Number.isFinite(right) && (right - left) > 1.0 && (right - left) < 8.0
        ) ? (right - left) : 3.6;
        const halfWidth = laneWidth * 0.5;

        const cumulativeS = [0];
        for (let i = 1; i < center.length; i++) {
            const dx = center[i].x - center[i - 1].x;
            const dy = center[i].y - center[i - 1].y;
            cumulativeS.push(cumulativeS[i - 1] + Math.hypot(dx, dy));
        }
        const totalS = cumulativeS[cumulativeS.length - 1];
        if (!Number.isFinite(totalS) || totalS <= 0) return [];

        const sampleAtDistance = (sTarget) => {
            const s = Math.max(0, Math.min(totalS, sTarget));
            let i = 1;
            while (i < cumulativeS.length && cumulativeS[i] < s) i++;
            const i1 = Math.min(i, center.length - 1);
            const i0 = Math.max(0, i1 - 1);
            const segLen = Math.max(1e-6, cumulativeS[i1] - cumulativeS[i0]);
            const t = (s - cumulativeS[i0]) / segLen;

            const cx = center[i0].x + (center[i1].x - center[i0].x) * t;
            const cy = center[i0].y + (center[i1].y - center[i0].y) * t;
            const tx = center[i1].x - center[i0].x;
            const ty = center[i1].y - center[i0].y;
            const tNorm = Math.hypot(tx, ty) || 1.0;
            const nx = ty / tNorm; // right normal in vehicle frame
            const ny = -tx / tNorm;

            const rx = cx + nx * halfWidth;
            const ry = cy + ny * halfWidth;
            const tickHalf = 0.22;
            return {
                s,
                a: { x: rx - nx * tickHalf, y: ry - ny * tickHalf },
                b: { x: rx + nx * tickHalf, y: ry + ny * tickHalf },
            };
        };

        const segments = [];
        const startOffset = Math.max(0, Number(this.distanceScaleStartOffsetMeters) || 0);
        for (let displayS = 0; displayS <= maxMeters + 1e-6; displayS += stepMeters) {
            const sourceS = startOffset + displayS;
            if (sourceS <= totalS + 1e-6) {
                const seg = sampleAtDistance(sourceS);
                seg.s = displayS;
                segments.push(seg);
            }
        }
        return segments;
    }

    drawMainDistanceScaleFromWorldFiducials(stepMeters = 5, maxMeters = 30) {
        const vehicle = this.currentFrameData?.vehicle || {};
        const anchorsRaw = Array.isArray(vehicle.right_lane_fiducials_world_points)
            ? vehicle.right_lane_fiducials_world_points
            : [];
        const anchors = anchorsRaw
            .map((p) => ({ x: Number(p?.x), y: Number(p?.y), z: Number(p?.z) }))
            .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z));
        if (anchors.length < 2) return false;

        const spacingMeters = Math.max(0.5, Number(vehicle.right_lane_fiducials_spacing_meters) || 5.0);
        const startOffset = Math.max(0, Number(this.distanceScaleStartOffsetMeters) || 0);
        const maxSourceS = (anchors.length - 1) * spacingMeters;
        let drawn = 0;

        const sampleAnchorAtDistance = (sourceS) => {
            const s = Math.max(0, Math.min(maxSourceS, sourceS));
            const idxFloat = s / spacingMeters;
            const i0 = Math.max(0, Math.min(anchors.length - 1, Math.floor(idxFloat)));
            const i1 = Math.max(0, Math.min(anchors.length - 1, Math.ceil(idxFloat)));
            const t = i1 === i0 ? 0 : (idxFloat - i0);
            const p0 = anchors[i0];
            const p1 = anchors[i1];
            const x = p0.x + (p1.x - p0.x) * t;
            const y = p0.y + (p1.y - p0.y) * t;
            const z = p0.z + (p1.z - p0.z) * t;
            const t0 = anchors[Math.max(0, i0 - 1)];
            const t1 = anchors[Math.min(anchors.length - 1, i1 + 1)];
            let tx = t1.x - t0.x;
            let tz = t1.z - t0.z;
            const norm = Math.hypot(tx, tz);
            if (!Number.isFinite(norm) || norm < 1e-3) {
                tx = p1.x - p0.x;
                tz = p1.z - p0.z;
            }
            const tn = Math.hypot(tx, tz) || 1.0;
            const nx = tz / tn;   // right normal in world XZ plane
            const nz = -tx / tn;
            return { x, y, z, nx, nz };
        };

        for (let displayS = 0; displayS <= maxMeters + 1e-6; displayS += stepMeters) {
            const sourceS = startOffset + displayS;
            if (sourceS > maxSourceS + 1e-6) continue;
            const p = sampleAnchorAtDistance(sourceS);
            const tickHalfMeters = 0.22;
            const aWorld = { x: p.x - p.nx * tickHalfMeters, y: p.y, z: p.z - p.nz * tickHalfMeters };
            const bWorld = { x: p.x + p.nx * tickHalfMeters, y: p.y, z: p.z + p.nz * tickHalfMeters };
            const projected = this.projectWorldPointsToImage([aWorld, bWorld]);
            if (!projected || projected.length < 2) continue;
            this.overlayRenderer.drawImagePath(projected, '#ffffff', displayS === 0 ? 3 : 2);
            drawn += 1;
        }

        return drawn > 0;
    }

    drawMainDistanceScale() {
        const showScale = Boolean(document.getElementById('toggle-distance-scale')?.checked);
        if (!showScale) return;
        if (Boolean(this.currentOverlaySnapRisk)) return;
        if (this.drawMainDistanceScaleFromWorldFiducials(5, 30)) return;
        const segments = this.getDistanceScaleSegments(5, 30);
        if (!segments.length) return;
        for (const seg of segments) {
            const projected = this.projectTrajectoryToImage([seg.a, seg.b]);
            if (!projected || projected.length < 2) continue;
            this.overlayRenderer.drawImagePath(projected, '#ffffff', seg.s === 0 ? 3 : 2);
        }
    }


    async buildFailureFrameStateCard(summary) {
        const failureDetected = summary?.executive_summary?.failure_detected;
        const failureFrame = summary?.executive_summary?.failure_frame;
        if (!failureDetected || failureFrame === null || failureFrame === undefined) {
            return null;
        }

        try {
            const frameData = await this.dataLoader.loadFrameData(Number(failureFrame));
            const perception = frameData?.perception || {};
            const control = frameData?.control || {};
            const trajectory = frameData?.trajectory || {};
            const groundTruth = frameData?.ground_truth || {};

            const left = Number(perception.left_lane_line_x);
            const right = Number(perception.right_lane_line_x);
            const hasLeft = Number.isFinite(left);
            const hasRight = Number.isFinite(right);
            const perceptionCenter = hasLeft && hasRight ? (left + right) / 2.0 : null;
            const gtCenter = Number.isFinite(Number(groundTruth.lane_center_x))
                ? Number(groundTruth.lane_center_x)
                : null;
            const mismatch = (perceptionCenter !== null && gtCenter !== null)
                ? Math.abs(perceptionCenter - gtCenter)
                : null;

            return {
                failureFrame: Number(failureFrame),
                perception: {
                    numLanesDetected: perception.num_lanes_detected,
                    leftLaneX: hasLeft ? left : null,
                    rightLaneX: hasRight ? right : null,
                    centerX: perceptionCenter,
                    usingStaleData: Boolean(perception.using_stale_data),
                    staleReason: perception.stale_data_reason || control.stale_reason || null,
                },
                groundTruth: {
                    laneCenterX: gtCenter,
                    pathCurvature: Number.isFinite(Number(groundTruth.path_curvature))
                        ? Number(groundTruth.path_curvature)
                        : null,
                },
                trajectory: {
                    refX: Number.isFinite(Number(trajectory?.reference_point?.x))
                        ? Number(trajectory.reference_point.x)
                        : null,
                    refHeading: Number.isFinite(Number(trajectory?.reference_point?.heading))
                        ? Number(trajectory.reference_point.heading)
                        : null,
                },
                control: {
                    lateralError: Number.isFinite(Number(control.lateral_error))
                        ? Number(control.lateral_error)
                        : null,
                    steering: Number.isFinite(Number(control.steering))
                        ? Number(control.steering)
                        : null,
                },
                gtVsPerceptionCenterMismatchM: mismatch,
            };
        } catch (error) {
            console.warn('Could not build failure-frame state card:', error);
            return null;
        }
    }

    setupEventListeners() {
        // Restore persisted checkbox preferences before wiring listeners.
        this.restoreCheckboxState();
        this.restoreProjectionNearFieldSettings();
        this.ensureCheckboxTooltips();

        // Recording selection
        document.getElementById('load-btn').addEventListener('click', () => this.loadSelectedRecording());
        const recordingSelect = document.getElementById('recording-select');
        if (recordingSelect) {
            recordingSelect.addEventListener('focus', () => this.loadRecordings());
            recordingSelect.addEventListener('click', () => this.loadRecordings());
        }
        
        // Frame navigation
        document.getElementById('frame-slider').addEventListener('input', (e) => {
            this.goToFrame(parseInt(e.target.value));
        });
        document.getElementById('prev-frame-btn').addEventListener('click', () => this.prevFrame());
        document.getElementById('next-frame-btn').addEventListener('click', () => this.nextFrame());
        document.getElementById('play-pause-btn').addEventListener('click', () => this.togglePlay());
        
        // Speed control
        document.getElementById('speed-slider').addEventListener('input', (e) => {
            this.playSpeed = parseFloat(e.target.value);
            document.getElementById('speed-value').textContent = `${this.playSpeed}x`;
            if (this.isPlaying) {
                this.startPlayback();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') this.prevFrame();
            if (e.key === 'ArrowRight') this.nextFrame();
            if (e.key === ' ') {
                e.preventDefault();
                this.togglePlay();
            }
        });
        
        // Polynomial Inspector
        const polyInspectorBtn = document.getElementById('polynomial-inspector-btn');
        if (polyInspectorBtn) {
            polyInspectorBtn.addEventListener('click', () => this.analyzePolynomialFitting());
        }
        
        // Generate Debug Overlays button
        const generateDebugBtn = document.getElementById('generate-debug-btn');
        if (generateDebugBtn) {
            generateDebugBtn.addEventListener('click', () => this.generateDebugOverlays());
        }
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                this.switchTab(tab);
            });
        });
        
        // Overlay toggles
        document.getElementById('toggle-lanes').addEventListener('change', () => this.updateOverlays());
        const fitPointsToggle = document.getElementById('toggle-fit-points');
        if (fitPointsToggle) {
            fitPointsToggle.addEventListener('change', () => this.updateOverlays());
        }
        document.getElementById('toggle-trajectory').addEventListener('change', () => this.updateOverlays());
        const oracleTrajectoryToggle = document.getElementById('toggle-oracle-trajectory');
        if (oracleTrajectoryToggle) {
            oracleTrajectoryToggle.addEventListener('change', () => this.updateOverlays());
        }
        const distanceScaleToggle = document.getElementById('toggle-distance-scale');
        if (distanceScaleToggle) {
            distanceScaleToggle.addEventListener('change', () => this.updateOverlays());
        }
        const rightLaneFiducialsToggle = document.getElementById('toggle-right-lane-fiducials');
        if (rightLaneFiducialsToggle) {
            rightLaneFiducialsToggle.addEventListener('change', () => this.updateOverlays());
        }
        const nearfieldBlendToggle = document.getElementById('toggle-main-nearfield-blend');
        if (nearfieldBlendToggle) {
            nearfieldBlendToggle.checked = Boolean(this.projectionNearFieldBlendEnabled);
            nearfieldBlendToggle.addEventListener('change', () => {
                this.projectionNearFieldBlendEnabled = Boolean(nearfieldBlendToggle.checked);
                this.persistProjectionNearFieldSettings();
                this.updateOverlays();
            });
        }
        const nearfieldOffsetSlider = document.getElementById('nearfield-y-offset-slider');
        const nearfieldOffsetValue = document.getElementById('nearfield-y-offset-value');
        if (nearfieldOffsetSlider) {
            nearfieldOffsetSlider.value = String(this.projectionNearFieldGroundYOffsetMeters);
            if (nearfieldOffsetValue) {
                nearfieldOffsetValue.textContent = this.projectionNearFieldGroundYOffsetMeters.toFixed(2);
            }
            nearfieldOffsetSlider.addEventListener('input', () => {
                const v = Number(nearfieldOffsetSlider.value);
                if (Number.isFinite(v)) {
                    this.projectionNearFieldGroundYOffsetMeters = v;
                    if (nearfieldOffsetValue) nearfieldOffsetValue.textContent = v.toFixed(2);
                    this.persistProjectionNearFieldSettings();
                    this.updateOverlays();
                }
            });
        }
        const nearfieldBlendDistanceSlider = document.getElementById('nearfield-blend-distance-slider');
        const nearfieldBlendDistanceValue = document.getElementById('nearfield-blend-distance-value');
        if (nearfieldBlendDistanceSlider) {
            nearfieldBlendDistanceSlider.value = String(this.projectionNearFieldBlendDistanceMeters);
            if (nearfieldBlendDistanceValue) {
                nearfieldBlendDistanceValue.textContent = this.projectionNearFieldBlendDistanceMeters.toFixed(1);
            }
            nearfieldBlendDistanceSlider.addEventListener('input', () => {
                const v = Number(nearfieldBlendDistanceSlider.value);
                if (Number.isFinite(v) && v > 0.1) {
                    this.projectionNearFieldBlendDistanceMeters = v;
                    if (nearfieldBlendDistanceValue) nearfieldBlendDistanceValue.textContent = v.toFixed(1);
                    this.persistProjectionNearFieldSettings();
                    this.updateOverlays();
                }
            });
        }
        const plannerOnlyTrajectoryToggle = document.getElementById('toggle-planner-only-trajectory');
        if (plannerOnlyTrajectoryToggle) {
            plannerOnlyTrajectoryToggle.addEventListener('change', () => this.updateOverlays());
        }
        const topdownSmoothToggle = document.getElementById('toggle-topdown-pose-smooth');
        if (topdownSmoothToggle) {
            topdownSmoothToggle.addEventListener('change', () => {
                this.topdownSmoothedTrajectory = null;
                this.topdownSmoothedFrameIndex = null;
                this.updateTopdownOverlay();
            });
        }
        const topdownCalibratedToggle = document.getElementById('toggle-topdown-calibrated-projection');
        if (topdownCalibratedToggle) {
            topdownCalibratedToggle.addEventListener('change', () => {
                this.topdownProjectionToggleTouched = true;
                this.updateTopdownOverlay();
            });
        }
        document.getElementById('toggle-reference').addEventListener('change', () => this.updateOverlays());
        document.getElementById('toggle-ground-truth').addEventListener('change', () => this.updateOverlays());
        const roiToggle = document.getElementById('toggle-perception-roi');
        if (roiToggle) {
            roiToggle.addEventListener('change', () => this.updateOverlays());
        }
        const segFitRoiToggle = document.getElementById('toggle-seg-fit-roi');
        if (segFitRoiToggle) {
            segFitRoiToggle.addEventListener('change', () => this.updateOverlays());
        }
        
        // Ground truth distance dropdown
        const gtDistanceSelect = document.getElementById('gt-distance-select');
        gtDistanceSelect.addEventListener('change', (e) => {
            this.groundTruthDistance = parseFloat(e.target.value);
            this.updateOverlays(); // Redraw overlays with new distance
        });
        
        // Frame jump input
        const frameJumpInput = document.getElementById('frame-jump-input');
        const frameJumpBtn = document.getElementById('frame-jump-btn');
        frameJumpBtn.addEventListener('click', () => {
            const frameNum = parseInt(frameJumpInput.value);
            if (!isNaN(frameNum) && frameNum >= 0 && frameNum < this.frameCount) {
                this.goToFrame(frameNum);
                frameJumpInput.value = ''; // Clear input after jumping
            } else {
                alert(`Invalid frame number. Please enter a number between 0 and ${this.frameCount - 1}`);
            }
        });
        // Also allow Enter key to jump
        frameJumpInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                frameJumpBtn.click();
            }
        });
        
        // Debug overlay toggles
        const combinedToggle = document.getElementById('toggle-combined');
        if (combinedToggle) {
            combinedToggle.addEventListener('change', () => this.updateDebugOverlays());
        }
        const edgesToggle = document.getElementById('toggle-edges');
        if (edgesToggle) {
            edgesToggle.addEventListener('change', () => this.updateDebugOverlays());
        }
        const yellowMaskToggle = document.getElementById('toggle-yellow-mask');
        if (yellowMaskToggle) {
            yellowMaskToggle.addEventListener('change', () => this.updateDebugOverlays());
        }
        const histogramToggle = document.getElementById('toggle-histogram');
        if (histogramToggle) {
            histogramToggle.addEventListener('change', () => this.updateDebugOverlays());
        }
        document.getElementById('toggle-seg-mask').addEventListener('change', () => this.updateOverlays());
        document.getElementById('toggle-seg-fit-points').addEventListener('change', () => this.updateOverlays());

        // Chart controls
        const chartPlotBtn = document.getElementById('chart-plot-btn');
        if (chartPlotBtn) {
            chartPlotBtn.addEventListener('click', () => this.plotSelectedSignals());
        }
        const chartClearBtn = document.getElementById('chart-clear-btn');
        if (chartClearBtn) {
            chartClearBtn.addEventListener('click', () => this.clearChart());
        }
        const chartSaveBtn = document.getElementById('chart-save-btn');
        if (chartSaveBtn) {
            chartSaveBtn.addEventListener('click', () => this.saveChartView());
        }
        const chartLoadBtn = document.getElementById('chart-load-btn');
        if (chartLoadBtn) {
            chartLoadBtn.addEventListener('click', () => this.loadSelectedChartView());
        }
        const chartSavedViewsSelect = document.getElementById('chart-saved-views-select');
        if (chartSavedViewsSelect) {
            chartSavedViewsSelect.addEventListener('change', () => this.loadSelectedChartView());
        }
        const chartOverwriteBtn = document.getElementById('chart-overwrite-btn');
        if (chartOverwriteBtn) {
            chartOverwriteBtn.addEventListener('click', () => this.overwriteSelectedChartView());
        }
        const chartDeleteBtn = document.getElementById('chart-delete-btn');
        if (chartDeleteBtn) {
            chartDeleteBtn.addEventListener('click', () => this.deleteSelectedChartView());
        }
        const chartZoomInBtn = document.getElementById('chart-zoom-in-btn');
        if (chartZoomInBtn) {
            chartZoomInBtn.addEventListener('click', () => this.zoomChart(0.5));
        }
        const chartZoomOutBtn = document.getElementById('chart-zoom-out-btn');
        if (chartZoomOutBtn) {
            chartZoomOutBtn.addEventListener('click', () => this.zoomChart(2.0));
        }
        const chartZoomResetBtn = document.getElementById('chart-zoom-reset-btn');
        if (chartZoomResetBtn) {
            chartZoomResetBtn.addEventListener('click', () => this.resetChartZoom());
        }
        const chartSearch = document.getElementById('chart-signal-search');
        if (chartSearch) {
            chartSearch.addEventListener('input', () => this.renderSignalList());
        }
        const quickApplyViewBtn = document.getElementById('quick-apply-view-btn');
        if (quickApplyViewBtn) {
            quickApplyViewBtn.addEventListener('click', () => this.applyQuickSelectedChartView());
        }
        const quickSavedViewsSelect = document.getElementById('quick-saved-views-select');
        if (quickSavedViewsSelect) {
            quickSavedViewsSelect.addEventListener('change', () => this.applyQuickSelectedChartView());
        }

        const legendToggleBtn = document.getElementById('legend-toggle-btn');
        if (legendToggleBtn) {
            legendToggleBtn.addEventListener('click', () => this.toggleLegend());
        }

        this.setupPanelResize();
        this.setupCameraGridResize();

        // CV tools visibility toggle
        const cvToolsToggle = document.getElementById('toggle-cv-tools');
        const cvDebugTools = document.getElementById('cv-debug-tools');
        const cvDataOverlays = document.getElementById('cv-data-overlays');
        const cvDebugOverlayControls = document.getElementById('cv-debug-overlay-controls');
        const setCvToolsVisible = (show) => {
            if (cvDebugTools) {
                cvDebugTools.style.display = show ? 'block' : 'none';
            }
            if (cvDataOverlays) {
                cvDataOverlays.style.display = show ? 'block' : 'none';
            }
            if (cvDebugOverlayControls) {
                cvDebugOverlayControls.style.display = show ? 'block' : 'none';
            }
            if (!show) {
                const combined = document.getElementById('toggle-combined');
                const edges = document.getElementById('toggle-edges');
                const yellow = document.getElementById('toggle-yellow-mask');
                const fitPoints = document.getElementById('toggle-fit-points');
                const histogram = document.getElementById('toggle-histogram');
                if (combined) combined.checked = false;
                if (edges) edges.checked = false;
                if (yellow) yellow.checked = false;
                if (fitPoints) fitPoints.checked = false;
                if (histogram) histogram.checked = false;
                this.updateDebugOverlays();
                this.updateOverlays();
            }
        };
        if (cvToolsToggle) {
            cvToolsToggle.checked = false;
            setCvToolsVisible(false);
            cvToolsToggle.addEventListener('change', (e) => {
                setCvToolsVisible(e.target.checked);
            });
        }
        
        // Opacity control
        const opacitySlider = document.getElementById('opacity-slider');
        if (opacitySlider) {
            opacitySlider.addEventListener('input', (e) => {
                const opacity = parseInt(e.target.value) / 100;
                const opacityValue = document.getElementById('opacity-value');
                if (opacityValue) {
                    opacityValue.textContent = e.target.value;
                }
                const debugCanvas = document.getElementById('debug-overlay-canvas');
                if (debugCanvas) {
                    debugCanvas.style.opacity = opacity;
                }
            });
        }
        
        // Export buttons
        document.getElementById('export-frame-btn').addEventListener('click', () => this.exportFrame());
        document.getElementById('export-video-btn').addEventListener('click', () => this.exportVideo());
        
        // Analyze to Failure checkbox
        const analyzeToFailureCheckbox = document.getElementById('analyze-to-failure');
        if (analyzeToFailureCheckbox) {
            analyzeToFailureCheckbox.addEventListener('change', () => {
                if (this.currentRecording) {
                    // Reload summary, issues, and diagnostics when checkbox changes
                    this.loadSummary();
                    this.loadIssues();
                    this.loadDiagnostics();
                }
            });
        }
        const entryStartDistanceInput = document.getElementById('diag-entry-start-distance');
        const entryWindowDistanceInput = document.getElementById('diag-entry-window-distance');
        [entryStartDistanceInput, entryWindowDistanceInput].forEach((input) => {
            if (!input) return;
            input.addEventListener('change', () => {
                if (this.currentRecording) {
                    this.loadSummary();
                    this.loadDiagnostics();
                }
            });
        });

        // Persist all checkbox preferences so reload/open keeps current selections.
        document.querySelectorAll('input[type="checkbox"][id]').forEach((el) => {
            el.addEventListener('change', () => this.persistCheckboxState());
        });
    }

    async loadRecordings() {
        try {
            const recordings = await this.dataLoader.loadRecordings();
            
            const select = document.getElementById('recording-select');
            if (!select) {
                console.error('Recording select element not found!');
                return;
            }
            
            const currentValue = select.value || this.currentRecording || '';
            select.innerHTML = '<option value="">Select recording...</option>';
            
            if (!recordings || recordings.length === 0) {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'No recordings available';
                option.disabled = true;
                select.appendChild(option);
                return;
            }
            
            recordings.forEach(rec => {
                const option = document.createElement('option');
                option.value = rec.filename;
                option.textContent = rec.filename;
                select.appendChild(option);
            });

            if (currentValue && recordings.some(rec => rec.filename === currentValue)) {
                select.value = currentValue;
            }
        } catch (error) {
            console.error('Error in loadRecordings:', error);
            const select = document.getElementById('recording-select');
            if (select) {
                select.innerHTML = '<option value="">Error loading recordings</option>';
            }
        }
    }

    async loadSelectedRecording() {
        const select = document.getElementById('recording-select');
        const filename = select.value;
        if (!filename) return;
        
        try {
            this.currentRecording = filename;  // Store current recording filename
            this.frameCount = await this.dataLoader.loadRecording(filename);
            try {
                this.currentRecordingMeta = await this.dataLoader.loadRecordingMeta(filename);
            } catch (e) {
                this.currentRecordingMeta = null;
            }
            this.updateRecordingMetaBadges();
            await this.loadDistanceFromStartSeries();
            this.topdownAvailable = Boolean(this.currentRecordingMeta?.topdown_available ?? true);
            this.setTopdownAvailability(this.topdownAvailable);
            document.getElementById('frame-count').textContent = this.frameCount;
            document.getElementById('frame-slider').max = this.frameCount - 1;
            this.currentFrameIndex = 0;
            this.lastUnityTime = null;
            this.lastUnityFrameIndex = null;
            this.lastUnityFrameCount = null;
            this.lastCameraTimestamp = null;
            this.topdownSmoothedTrajectory = null;
            this.topdownSmoothedFrameIndex = null;
            await this.goToFrame(0);
            await this.loadSummary();  // Load summary when recording is loaded
            await this.loadIssues();  // Load issues when recording is loaded
            await this.loadDiagnostics();  // Load diagnostics when recording is loaded
            await this.loadSignalsList();
            this.updateQuickChartValuesTable();
        } catch (error) {
            console.error('Error loading recording:', error);
            alert('Failed to load recording: ' + error.message);
        }
    }

    humanizeRecordingType(value) {
        const v = String(value || 'unknown');
        return v.replaceAll('_', ' ');
    }

    updateRecordingMetaBadges() {
        const typeBadge = document.getElementById('recording-type-badge');
        const topdownBadge = document.getElementById('topdown-badge');
        const meta = this.currentRecordingMeta || {};
        if (typeBadge) {
            const parts = [`type: ${this.humanizeRecordingType(meta.recording_type || 'unknown')}`];
            if (meta.source_recording) parts.push(`src: ${meta.source_recording}`);
            if (meta.lock_source_recording) parts.push(`lock: ${meta.lock_source_recording}`);
            if (meta.control_lock_source_recording) parts.push(`ctrl lock: ${meta.control_lock_source_recording}`);
            typeBadge.textContent = parts.join(' | ');
            typeBadge.style.display = 'inline-flex';
            typeBadge.title = parts.join('\n');
        }
        if (topdownBadge) {
            const hasTop = Boolean(meta.topdown_available);
            topdownBadge.textContent = hasTop ? 'top-down: available' : 'top-down: unavailable';
            topdownBadge.style.color = hasTop ? '#4caf50' : '#ffb74d';
            topdownBadge.style.display = 'inline-flex';
        }
    }

    async loadDistanceFromStartSeries() {
        this.distanceFromStartSeries = null;
        if (!this.currentRecording) return;
        try {
            const data = await this.dataLoader.loadTimeSeries(['derived/distance_m'], 'vehicle/timestamps');
            this.distanceFromStartSeries = data?.signals?.['derived/distance_m'] || null;
        } catch (error) {
            console.warn('Could not load derived distance series:', error);
            this.distanceFromStartSeries = null;
        }
    }
    
    async loadSummary() {
        if (!this.currentRecording) return;
        
        const summaryContent = document.getElementById('summary-content');
        if (!summaryContent) return;
        
        summaryContent.innerHTML = '<p style="color: #888; text-align: center; padding: 2rem;">Loading summary...</p>';
        
        try {
            // Check if "Analyze to Failure" checkbox is checked
            const analyzeToFailure = document.getElementById('analyze-to-failure')?.checked || false;
            const url = `/api/recording/${this.currentRecording}/summary${analyzeToFailure ? '?analyze_to_failure=true' : ''}`;
            const response = await fetch(url);
            
            // Also load diagnostics for quick diagnosis (respect analyze_to_failure checkbox)
            let diagnostics = null;
            let topdownDiagnostics = null;
            try {
                const entryStartDistance = parseFloat(document.getElementById('diag-entry-start-distance')?.value || `${this.curveEntryStartDistanceM}`);
                const entryWindowDistance = parseFloat(document.getElementById('diag-entry-window-distance')?.value || `${this.curveEntryWindowDistanceM}`);
                const diagParams = new URLSearchParams();
                if (analyzeToFailure) diagParams.set('analyze_to_failure', 'true');
                if (Number.isFinite(entryStartDistance)) diagParams.set('curve_entry_start_distance_m', String(entryStartDistance));
                if (Number.isFinite(entryWindowDistance)) diagParams.set('curve_entry_window_distance_m', String(entryWindowDistance));
                const diagUrl = `/api/recording/${this.currentRecording}/diagnostics?${diagParams.toString()}`;
                const diagResponse = await fetch(diagUrl);
                if (diagResponse.ok) {
                    diagnostics = await diagResponse.json();
                }
            } catch (e) {
                console.warn('Could not load diagnostics:', e);
            }
            try {
                const tdResp = await fetch(`/api/recording/${this.currentRecording}/topdown-diagnostics`);
                if (tdResp.ok) {
                    topdownDiagnostics = await tdResp.json();
                }
            } catch (e) {
                console.warn('Could not load top-down diagnostics:', e);
            }
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }
            
            const summary = await response.json();
            this.summaryConfig = summary.config || null;
            
            if (summary.error) {
                summaryContent.innerHTML = `<p style="color: #ff6b6b; text-align: center; padding: 2rem;">Error: ${summary.error}</p>`;
                return;
            }

            const failureStateCard = await this.buildFailureFrameStateCard(summary);
            
            // Build summary HTML
            let html = '<div style="padding: 1rem;">';
            
            // Perception Questions Runner
            html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
            html += '<h3 style="margin-top: 0; color: #4a90e2;">Perception Questions (Q1-Q8)</h3>';
            html += '<div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.75rem;">';
            html += '<button id="run-perception-questions-btn" style="padding: 0.45rem 0.9rem; background: #4a90e2; color: white; border: none; border-radius: 4px; cursor: pointer;">Run Q Script</button>';
            html += '<span id="run-perception-questions-status" style="color: #a0a0a0; font-size: 0.9rem;">Run analyzer for this recording from PhilViz.</span>';
            html += '</div>';
            html += '<div id="run-perception-questions-parsed" style="margin-bottom: 0.5rem; color: #d0d0d0; font-size: 0.9rem;"></div>';
            html += '<pre id="run-perception-questions-output" style="display: none; white-space: pre-wrap; max-height: 260px; overflow-y: auto; padding: 0.75rem; border-radius: 4px; background: #1a1a1a; color: #d8d8d8; font-size: 0.82rem;"></pre>';
            html += '</div>';

            // Executive Summary
            html += '<div style="background: #2a2a2a; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;">';
            html += '<h2 style="margin-top: 0; color: #4a90e2;">Executive Summary</h2>';
            html += `<div style="font-size: 2rem; font-weight: bold; color: ${summary.executive_summary.overall_score >= 80 ? '#4caf50' : summary.executive_summary.overall_score >= 60 ? '#ffa500' : '#ff6b6b'}; margin: 1rem 0;">`;
            html += `Overall Score: ${summary.executive_summary.overall_score.toFixed(1)}/100</div>`;
            
            // Score Breakdown
            if (summary.executive_summary.score_breakdown) {
                const breakdown = summary.executive_summary.score_breakdown;
                html += '<div style="margin-top: 1rem; padding: 0.75rem; background: #1a1a1a; border-radius: 4px; font-size: 0.85rem;">';
                html += '<strong style="color: #4a90e2;">Score Breakdown:</strong><br/>';
                html += `<span style="color: #888;">Base: 100.0</span><br/>`;
                
                const penalties = [
                    { name: 'Lateral Error RMSE', value: breakdown.lateral_error_penalty, max: 30 },
                    { name: 'Steering Jerk', value: breakdown.steering_jerk_penalty, max: 20 },
                    { name: 'Lane Detection', value: breakdown.lane_detection_penalty, max: 20 },
                    { name: 'Stale Data', value: breakdown.stale_data_penalty, max: 15 },
                    { name: 'Perception Instability', value: breakdown.perception_instability_penalty || 0, max: 20 },
                    { name: 'Out-of-Lane', value: breakdown.out_of_lane_penalty, max: 15 }
                ];
                
                penalties.forEach(penalty => {
                    if (penalty.value > 0.01) {
                        const color = penalty.value > penalty.max * 0.7 ? '#ff6b6b' : penalty.value > penalty.max * 0.4 ? '#ffa500' : '#ffa500';
                        html += `<span style="color: ${color};">  -${penalty.value.toFixed(1)}</span> <span style="color: #888;">${penalty.name}</span><br/>`;
                    }
                });
                
                html += '</div>';
            }
            
            // Show analysis scope indicator
            let scope_indicator = '';
            if (summary.executive_summary.analyzed_to_failure && summary.executive_summary.failure_detected) {
                scope_indicator = `<span style="color: #4caf50; font-weight: bold;">📊 Metrics calculated up to failure point (frame ${summary.executive_summary.failure_frame})</span>`;
            } else if (summary.executive_summary.failure_detected) {
                scope_indicator = `<span style="color: #ffa500;">📊 Metrics calculated for full drive (failure occurred at frame ${summary.executive_summary.failure_frame})</span>`;
            } else {
                scope_indicator = `<span style="color: #a0a0a0;">📊 Metrics calculated for full drive</span>`;
            }
            
            const centeredness = summary.path_tracking?.time_in_lane_centered;
            const centerednessText = centeredness !== undefined && centeredness !== null
                ? ` | Centeredness (±0.5m): ${centeredness.toFixed(1)}%`
                : '';
            html += `<div style="color: #a0a0a0; margin-bottom: 1rem;">Drive Duration: ${summary.executive_summary.drive_duration.toFixed(1)}s | Frames: ${summary.executive_summary.total_frames} | Success Rate: ${summary.executive_summary.success_rate.toFixed(1)}%${centerednessText}</div>`;
            
            html += '<h3 style="margin: 1rem 0 0.5rem; color: #9bdcff;">System Health</h3>';
            // Unity timing health
            const unityGapMax = summary.system_health?.unity_time_gap_max ?? null;
            const unityGapCount = summary.system_health?.unity_time_gap_count ?? null;
            if (unityGapMax !== null && unityGapCount !== null) {
                const gapColor = unityGapCount > 0 || unityGapMax > 0.2 ? '#ff6b6b' : '#4caf50';
                const gapStatus = unityGapCount > 0 || unityGapMax > 0.2 ? '⚠️ Hitch Detected' : '✓ No Hitch';
                html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
                html += '<h3 style="margin-top: 0; color: #4a90e2;">Unity Timing Health</h3>';
                html += '<table style="width: 100%; color: #e0e0e0;">';
                html += `<tr><td>Status:</td><td style="text-align: right; color: ${gapColor};">${gapStatus}</td></tr>`;
                html += `<tr><td>Max Unity Time Gap:</td><td style="text-align: right; color: ${gapColor};">${unityGapMax.toFixed(3)}s</td></tr>`;
                html += `<tr><td>Gaps > 0.2s:</td><td style="text-align: right; color: ${gapColor};">${unityGapCount}</td></tr>`;
                html += '</table></div>';
            }

            html += `<div style="margin-bottom: 1rem; padding: 0.5rem; background: #2a2a2a; border-radius: 4px;">${scope_indicator}</div>`;
            
            // Show failure detection info if applicable
            if (summary.executive_summary.failure_detected) {
                html += `<div style="background: #3a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #ff6b6b; margin-top: 1rem;">`;
                html += `<strong style="color: #ff6b6b;">⚠️ Failure Detected:</strong> `;
                html += `Car went out of lane at frame ${summary.executive_summary.failure_frame} and stayed out (using ${summary.executive_summary.failure_detection_source || 'unknown'} data). `;
                if (summary.executive_summary.analyzed_to_failure) {
                    html += `<span style="color: #4caf50;">All metrics below are calculated only up to this point.</span>`;
                } else {
                    html += `<span style="color: #ffa500;">Check "Analyze to Failure" to see metrics for only the good portion of the drive.</span>`;
                }
                html += `</div>`;
            }

            if (failureStateCard) {
                const p = failureStateCard.perception;
                const gt = failureStateCard.groundTruth;
                const tr = failureStateCard.trajectory;
                const ctrl = failureStateCard.control;
                const staleColor = p.usingStaleData ? '#ff6b6b' : '#4caf50';
                html += '<div style="margin-top: 1rem; background: #1f2430; border: 1px solid #3b4252; padding: 1rem; border-radius: 8px;">';
                html += '<h3 style="margin-top: 0; color: #9bdcff;">Failure-Frame Upstream State</h3>';
                html += '<table style="width: 100%; color: #e0e0e0; font-size: 0.9rem;">';
                html += `<tr><td>Failure Frame</td><td style="text-align: right;"><strong>${failureStateCard.failureFrame}</strong></td></tr>`;
                html += `<tr><td>Lanes Detected</td><td style="text-align: right;">${p.numLanesDetected ?? '-'}</td></tr>`;
                html += `<tr><td>Perception Left/Right</td><td style="text-align: right;">${p.leftLaneX !== null ? p.leftLaneX.toFixed(3) : '-'} / ${p.rightLaneX !== null ? p.rightLaneX.toFixed(3) : '-'}</td></tr>`;
                html += `<tr><td>Perception Center vs GT Center</td><td style="text-align: right;">${p.centerX !== null ? p.centerX.toFixed(3) : '-'} / ${gt.laneCenterX !== null ? gt.laneCenterX.toFixed(3) : '-'}</td></tr>`;
                html += `<tr><td>GT-Perception Center Mismatch</td><td style="text-align: right;">${failureStateCard.gtVsPerceptionCenterMismatchM !== null ? failureStateCard.gtVsPerceptionCenterMismatchM.toFixed(3) + 'm' : '-'}</td></tr>`;
                html += `<tr><td>Using Stale Perception</td><td style="text-align: right; color: ${staleColor};">${p.usingStaleData ? 'YES' : 'NO'}</td></tr>`;
                html += `<tr><td>Stale Reason</td><td style="text-align: right;">${this.escapeHtml(p.staleReason || '-')}</td></tr>`;
                html += `<tr><td>Trajectory Ref X / Heading</td><td style="text-align: right;">${tr.refX !== null ? tr.refX.toFixed(3) : '-'} / ${tr.refHeading !== null ? tr.refHeading.toFixed(3) : '-'}</td></tr>`;
                html += `<tr><td>Control Lateral Error / Steering</td><td style="text-align: right;">${ctrl.lateralError !== null ? ctrl.lateralError.toFixed(3) : '-'} / ${ctrl.steering !== null ? ctrl.steering.toFixed(3) : '-'}</td></tr>`;
                html += '</table>';
                html += `<div style="margin-top: 0.75rem;"><button style="padding: 0.3rem 0.8rem; background: #4a90e2; color: white; border: none; border-radius: 4px; cursor: pointer;" onclick="window.visualizer.jumpToFrame(${failureStateCard.failureFrame})">Jump to Failure Frame →</button></div>`;
                html += '</div>';
            }
            
            if (summary.executive_summary.key_issues && summary.executive_summary.key_issues.length > 0) {
                html += '<div style="margin-top: 1rem;"><strong style="color: #ff6b6b;">Key Issues:</strong><ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #ff6b6b;">';
                summary.executive_summary.key_issues.forEach(issue => {
                    html += `<li>${issue}</li>`;
                });
                html += '</ul></div>';
            }
            html += '</div>';
            
            // Quick Diagnosis Section (if diagnostics available)
            if (diagnostics && !diagnostics.error && diagnostics.diagnosis) {
                const diag = diagnostics.diagnosis;
                const primaryIssue = diag.primary_issue || 'unknown';
                const trajScore = diag.trajectory_score || 0;
                const ctrlScore = diag.control_score || 0;
                
                // Determine severity based on actual scores, not just which is lower
                let diagnosisColor = '#4caf50';
                let diagnosisIcon = '✓';
                let diagnosisText = 'System appears to be working correctly';
                
                if (primaryIssue === 'trajectory') {
                    // Use score to determine severity
                    if (trajScore < 70) {
                        diagnosisColor = '#ff6b6b';  // Red for poor
                        diagnosisIcon = '⚠️';
                        diagnosisText = 'Trajectory planning needs attention';
                    } else if (trajScore < 80) {
                        diagnosisColor = '#ffa500';  // Orange for acceptable
                        diagnosisIcon = '⚠️';
                        diagnosisText = 'Trajectory planning may need improvement';
                    } else {
                        // Score is good, but still flagged as primary issue (shouldn't happen with new logic)
                        diagnosisColor = '#4caf50';
                        diagnosisIcon = '✓';
                        diagnosisText = 'System appears to be working correctly';
                    }
                } else if (primaryIssue === 'control') {
                    // Use score to determine severity
                    if (ctrlScore < 70) {
                        diagnosisColor = '#ff6b6b';  // Red for poor
                        diagnosisIcon = '⚠️';
                        diagnosisText = 'Control system needs tuning';
                    } else if (ctrlScore < 80) {
                        diagnosisColor = '#ffa500';  // Orange for acceptable
                        diagnosisIcon = '⚠️';
                        diagnosisText = 'Control system may need tuning';
                    } else {
                        // Score is good, but still flagged as primary issue (shouldn't happen with new logic)
                        diagnosisColor = '#4caf50';
                        diagnosisIcon = '✓';
                        diagnosisText = 'System appears to be working correctly';
                    }
                }
                
                html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid ' + diagnosisColor + ';">';
                html += '<h3 style="margin-top: 0; color: #4a90e2;">Quick Diagnosis</h3>';
                html += `<div style="font-size: 1.1rem; font-weight: bold; color: ${diagnosisColor}; margin: 0.5rem 0;">${diagnosisIcon} ${diagnosisText}</div>`;
                html += '<div style="color: #a0a0a0; margin: 0.5rem 0; font-size: 0.9rem;">';
                html += `Trajectory Quality: <span style="color: ${trajScore >= 80 ? '#4caf50' : trajScore >= 60 ? '#ffa500' : '#ff6b6b'}">${trajScore.toFixed(1)}%</span> | `;
                html += `Control Quality: <span style="color: ${ctrlScore >= 80 ? '#4caf50' : ctrlScore >= 60 ? '#ffa500' : '#ff6b6b'}">${ctrlScore.toFixed(1)}%</span>`;
                html += '</div>';
                
                if (diag.recommendations && diag.recommendations.length > 0) {
                    html += '<div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #444;">';
                    html += '<strong style="color: #ffa500; font-size: 0.9rem;">Top Recommendations:</strong><ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #e0e0e0; font-size: 0.9rem;">';
                    diag.recommendations.slice(0, 3).forEach(rec => {
                        html += `<li style="margin-bottom: 0.25rem;">${rec}</li>`;
                    });
                    html += '</ul></div>';
                }
                
                html += '<div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #444;">';
                html += '<button onclick="window.visualizer.switchTab(\'diagnostics\')" style="padding: 0.5rem 1rem; background: #4a90e2; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">View Full Diagnostics →</button>';
                html += '</div>';
                html += '</div>';
            }

            // Top-down timing/projection diagnostics (instrumentation-only)
            if (topdownDiagnostics && !topdownDiagnostics.error) {
                const dtTraj = topdownDiagnostics.dt_topdown_traj || {};
                const dtUnity = topdownDiagnostics.dt_topdown_unity || {};
                const syncQuality = topdownDiagnostics.sync_quality || 'unknown';
                const qualityColor = syncQuality === 'good' ? '#4caf50' : (syncQuality === 'warn' ? '#ffa500' : '#ff6b6b');
                const missingTopdownProj = Array.isArray(topdownDiagnostics.topdown_projection_fields_missing)
                    ? topdownDiagnostics.topdown_projection_fields_missing.length
                    : 0;
                const idxDeltaTraj = topdownDiagnostics.topdown_traj_index_delta || {};
                const idxDeltaUnity = topdownDiagnostics.topdown_unity_index_delta || {};
                const mismatchSuspected = !!topdownDiagnostics.timestamp_domain_mismatch_suspected;
                const streamFrontUnity = topdownDiagnostics.stream_front_unity_dt_ms_stats || {};
                const streamTopdownUnity = topdownDiagnostics.stream_topdown_unity_dt_ms_stats || {};
                const streamTopdownFront = topdownDiagnostics.stream_topdown_front_dt_ms_stats || {};
                const streamTopdownFrontFrame = topdownDiagnostics.stream_topdown_front_frame_id_delta_stats || {};
                const streamFrontAge = topdownDiagnostics.stream_front_latest_age_ms_stats || {};
                const streamTopdownAge = topdownDiagnostics.stream_topdown_latest_age_ms_stats || {};
                const streamFrontQueue = topdownDiagnostics.stream_front_queue_depth_stats || {};
                const streamTopdownQueue = topdownDiagnostics.stream_topdown_queue_depth_stats || {};
                const streamFrontTsRealtime = topdownDiagnostics.stream_front_timestamp_minus_realtime_ms_stats || {};
                const streamTopdownTsRealtime = topdownDiagnostics.stream_topdown_timestamp_minus_realtime_ms_stats || {};
                const topdownOrthoStats = topdownDiagnostics.topdown_orthographic_size_stats || {};
                const topdownMppStats = topdownDiagnostics.topdown_meters_per_pixel_stats || {};
                const topdownForwardYStats = topdownDiagnostics.topdown_forward_y_stats || {};
                const topdownReady = !!topdownDiagnostics.topdown_calibrated_projection_ready;
                this.topdownCalibratedProjectionReady = topdownReady;
                const topdownCalibratedToggle = document.getElementById('toggle-topdown-calibrated-projection');
                if (topdownCalibratedToggle && !this.topdownProjectionToggleTouched) {
                    // Default to calibrated projection only when diagnostics say it is safe.
                    topdownCalibratedToggle.checked = topdownReady;
                }
                html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #4a90e2;">';
                html += '<h3 style="margin-top: 0; color: #4a90e2;">Top-Down Overlay Trust (Instrumentation)</h3>';
                html += '<table style="width: 100%; color: #e0e0e0;">';
                html += `<tr><td>Top-Down Sync Quality:</td><td style="text-align: right; color: ${qualityColor};">${syncQuality.toUpperCase()}</td></tr>`;
                html += `<tr><td>Top-Down ↔ Trajectory Δt (P95):</td><td style="text-align: right;">${dtTraj.p95_ms !== null && dtTraj.p95_ms !== undefined ? dtTraj.p95_ms.toFixed(1) + ' ms' : '-'}</td></tr>`;
                html += `<tr><td>Top-Down ↔ Trajectory Δt (Max):</td><td style="text-align: right;">${dtTraj.max_ms !== null && dtTraj.max_ms !== undefined ? dtTraj.max_ms.toFixed(1) + ' ms' : '-'}</td></tr>`;
                html += `<tr><td>Top-Down ↔ Unity Δt (P95):</td><td style="text-align: right;">${dtUnity.p95_ms !== null && dtUnity.p95_ms !== undefined ? dtUnity.p95_ms.toFixed(1) + ' ms' : '-'}</td></tr>`;
                html += `<tr><td>Top-Down ↔ Unity Δt (Max):</td><td style="text-align: right;">${dtUnity.max_ms !== null && dtUnity.max_ms !== undefined ? dtUnity.max_ms.toFixed(1) + ' ms' : '-'}</td></tr>`;
                html += `<tr><td>Top-Down ↔ Trajectory Index Δ (P95 abs):</td><td style="text-align: right;">${idxDeltaTraj.p95_abs !== null && idxDeltaTraj.p95_abs !== undefined ? idxDeltaTraj.p95_abs.toFixed(1) + ' frames' : '-'}</td></tr>`;
                html += `<tr><td>Top-Down ↔ Unity Index Δ (P95 abs):</td><td style="text-align: right;">${idxDeltaUnity.p95_abs !== null && idxDeltaUnity.p95_abs !== undefined ? idxDeltaUnity.p95_abs.toFixed(1) + ' frames' : '-'}</td></tr>`;
                html += `<tr><td>Consume Lag Front↔Unity (P95 abs):</td><td style="text-align: right;">${streamFrontUnity.p95_abs !== null && streamFrontUnity.p95_abs !== undefined ? streamFrontUnity.p95_abs.toFixed(1) + ' ms' : '-'}</td></tr>`;
                html += `<tr><td>Consume Lag TopDown↔Unity (P95 abs):</td><td style="text-align: right;">${streamTopdownUnity.p95_abs !== null && streamTopdownUnity.p95_abs !== undefined ? streamTopdownUnity.p95_abs.toFixed(1) + ' ms' : '-'}</td></tr>`;
                html += `<tr><td>Consume Lag TopDown↔Front (P95 abs):</td><td style="text-align: right;">${streamTopdownFront.p95_abs !== null && streamTopdownFront.p95_abs !== undefined ? streamTopdownFront.p95_abs.toFixed(1) + ' ms' : '-'}</td></tr>`;
                html += `<tr><td>Consume Frame Δ TopDown-Front (P95 abs):</td><td style="text-align: right;">${streamTopdownFrontFrame.p95_abs !== null && streamTopdownFrontFrame.p95_abs !== undefined ? streamTopdownFrontFrame.p95_abs.toFixed(1) + ' frames' : '-'}</td></tr>`;
                html += `<tr><td>Bridge Front Latest Age (P95 abs):</td><td style="text-align: right;">${streamFrontAge.p95_abs !== null && streamFrontAge.p95_abs !== undefined ? streamFrontAge.p95_abs.toFixed(1) + ' ms' : '-'}</td></tr>`;
                html += `<tr><td>Bridge TopDown Latest Age (P95 abs):</td><td style="text-align: right;">${streamTopdownAge.p95_abs !== null && streamTopdownAge.p95_abs !== undefined ? streamTopdownAge.p95_abs.toFixed(1) + ' ms' : '-'}</td></tr>`;
                html += `<tr><td>Bridge Front Queue Depth (mean):</td><td style="text-align: right;">${streamFrontQueue.mean !== null && streamFrontQueue.mean !== undefined ? streamFrontQueue.mean.toFixed(1) : '-'}</td></tr>`;
                html += `<tr><td>Bridge TopDown Queue Depth (mean):</td><td style="text-align: right;">${streamTopdownQueue.mean !== null && streamTopdownQueue.mean !== undefined ? streamTopdownQueue.mean.toFixed(1) : '-'}</td></tr>`;
                html += `<tr><td>Front (timestamp - realtime) P95 abs:</td><td style="text-align: right;">${streamFrontTsRealtime.p95_abs !== null && streamFrontTsRealtime.p95_abs !== undefined ? streamFrontTsRealtime.p95_abs.toFixed(1) + ' ms' : '-'}</td></tr>`;
                html += `<tr><td>TopDown (timestamp - realtime) P95 abs:</td><td style="text-align: right;">${streamTopdownTsRealtime.p95_abs !== null && streamTopdownTsRealtime.p95_abs !== undefined ? streamTopdownTsRealtime.p95_abs.toFixed(1) + ' ms' : '-'}</td></tr>`;
                html += `<tr><td>Top-Down Projection Fields Missing:</td><td style="text-align: right; color: ${missingTopdownProj > 0 ? '#ffa500' : '#4caf50'};">${missingTopdownProj}</td></tr>`;
                html += `<tr><td>Top-Down Ortho Size (mean):</td><td style="text-align: right;">${topdownOrthoStats.mean !== null && topdownOrthoStats.mean !== undefined ? topdownOrthoStats.mean.toFixed(2) : '-'}</td></tr>`;
                html += `<tr><td>Top-Down Scale (m/px mean):</td><td style="text-align: right;">${topdownMppStats.mean !== null && topdownMppStats.mean !== undefined ? topdownMppStats.mean.toFixed(4) : '-'}</td></tr>`;
                html += `<tr><td>Top-Down Forward Y (mean):</td><td style="text-align: right;">${topdownForwardYStats.mean !== null && topdownForwardYStats.mean !== undefined ? topdownForwardYStats.mean.toFixed(3) : '-'}</td></tr>`;
                html += `<tr><td>Calibrated Projection Ready:</td><td style="text-align: right; color: ${topdownReady ? '#4caf50' : '#ffa500'};">${topdownReady ? 'YES' : 'NO'}</td></tr>`;
                html += `<tr><td>Timestamp Domain Mismatch Suspected:</td><td style="text-align: right; color: ${mismatchSuspected ? '#ffa500' : '#4caf50'};">${mismatchSuspected ? 'YES' : 'NO'}</td></tr>`;
                html += '</table>';
                html += '<div style="margin-top: 0.5rem; color: #a0a0a0; font-size: 0.85rem;">';
                html += 'Use this card to separate timing mismatch from projection/calibration gaps. No control behavior is affected.';
                html += '</div></div>';
            }
            
            // Helper function for color coding
            const getColorForValue = (value, thresholds) => {
                if (value <= thresholds.good) return '#4caf50';  // Green
                if (value <= thresholds.acceptable) return '#ffa500';  // Orange
                return '#ff6b6b';  // Red
            };
            
            html += '<h3 style="margin: 1rem 0 0.5rem; color: #9bdcff;">Perception</h3>';
            // Perception Quality
            html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
            html += '<h3 style="margin-top: 0; color: #4a90e2;">Perception Quality</h3>';
            html += '<table style="width: 100%; color: #e0e0e0;">';
            const laneDetColor = summary.perception_quality.lane_detection_rate >= 90 ? '#4caf50' : summary.perception_quality.lane_detection_rate >= 70 ? '#ffa500' : '#ff6b6b';
            const staleColor = summary.perception_quality.stale_perception_rate < 10 ? '#4caf50' : summary.perception_quality.stale_perception_rate < 20 ? '#ffa500' : '#ff6b6b';
            const stabilityScore = summary.perception_quality.perception_stability_score;
            const stabilityColor = stabilityScore !== undefined
                ? (stabilityScore >= 80 ? '#4caf50' : stabilityScore >= 60 ? '#ffa500' : '#ff6b6b')
                : '#888';
            const instabilityColor = summary.perception_quality.perception_instability_detected === 0 ? '#4caf50' : '#ff6b6b';
            html += `<tr><td>Lane Detection Rate:</td><td style="text-align: right; color: ${laneDetColor};">${summary.perception_quality.lane_detection_rate.toFixed(1)}%</td></tr>`;
            html += `<tr><td>Confidence (Mean):</td><td style="text-align: right;">${summary.perception_quality.perception_confidence_mean.toFixed(3)}</td></tr>`;
            html += `<tr><td>Jumps Detected:</td><td style="text-align: right;">${summary.perception_quality.perception_jumps_detected || 0}</td></tr>`;
            if (summary.perception_quality.perception_instability_detected !== undefined) {
                html += `<tr><td>Instability Events:</td><td style="text-align: right; color: ${instabilityColor};">${summary.perception_quality.perception_instability_detected}</td></tr>`;
            }
            html += `<tr><td>Stale Data Rate:</td><td style="text-align: right; color: ${staleColor};">${summary.perception_quality.stale_perception_rate.toFixed(1)}%</td></tr>`;
            if (stabilityScore !== undefined) {
                html += `<tr><td>Stability Score:</td><td style="text-align: right; color: ${stabilityColor};">${stabilityScore.toFixed(1)}%</td></tr>`;
            }
            if (summary.perception_quality.lane_line_jitter_p95 !== undefined) {
                const jitterColor = getColorForValue(
                    summary.perception_quality.lane_line_jitter_p95,
                    { good: 0.3, acceptable: 0.6 }
                );
                html += `<tr><td>Lane Line Jitter (P95):</td><td style="text-align: right; color: ${jitterColor};">${summary.perception_quality.lane_line_jitter_p95.toFixed(3)}m</td></tr>`;
            }
            if (summary.perception_quality.reference_jitter_p95 !== undefined) {
                const refJitterColor = getColorForValue(
                    summary.perception_quality.reference_jitter_p95,
                    { good: 0.15, acceptable: 0.25 }
                );
                html += `<tr><td>Reference Jitter (P95):</td><td style="text-align: right; color: ${refJitterColor};">${summary.perception_quality.reference_jitter_p95.toFixed(3)}m</td></tr>`;
            }
            if (summary.perception_quality.single_lane_rate !== undefined) {
                const singleLaneColor = summary.perception_quality.single_lane_rate < 5 ? '#4caf50'
                    : summary.perception_quality.single_lane_rate < 15 ? '#ffa500' : '#ff6b6b';
                html += `<tr><td>Single Lane Rate:</td><td style="text-align: right; color: ${singleLaneColor};">${summary.perception_quality.single_lane_rate.toFixed(1)}%</td></tr>`;
            }
            if (summary.perception_quality.right_lane_low_visibility_rate !== undefined) {
                const rightLowColor = summary.perception_quality.right_lane_low_visibility_rate < 5 ? '#4caf50'
                    : summary.perception_quality.right_lane_low_visibility_rate < 15 ? '#ffa500' : '#ff6b6b';
                html += `<tr><td>Right Lane Low Visibility:</td><td style="text-align: right; color: ${rightLowColor};">${summary.perception_quality.right_lane_low_visibility_rate.toFixed(1)}%</td></tr>`;
            }
            if (summary.perception_quality.right_lane_edge_contact_rate !== undefined) {
                const edgeColor = summary.perception_quality.right_lane_edge_contact_rate < 20 ? '#4caf50'
                    : summary.perception_quality.right_lane_edge_contact_rate < 50 ? '#ffa500' : '#ffb74d';
                html += `<tr><td>Right Lane Edge Contact:</td><td style="text-align: right; color: ${edgeColor};">${summary.perception_quality.right_lane_edge_contact_rate.toFixed(1)}%</td></tr>`;
            }
            html += '</table></div>';

            html += '<h3 style="margin: 1rem 0 0.5rem; color: #9bdcff;">Lateral</h3>';
            // Trajectory Quality
            html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
            html += '<h3 style="margin-top: 0; color: #4a90e2;">Trajectory Quality</h3>';
            html += '<table style="width: 100%; color: #e0e0e0;">';
            const availColor = summary.trajectory_quality.trajectory_availability >= 95 ? '#4caf50' : summary.trajectory_quality.trajectory_availability >= 90 ? '#ffa500' : '#ff6b6b';
            html += `<tr><td>Availability:</td><td style="text-align: right; color: ${availColor};">${summary.trajectory_quality.trajectory_availability.toFixed(1)}%</td></tr>`;
            const refAccColor = getColorForValue(summary.trajectory_quality.ref_point_accuracy_rmse, { good: 0.1, acceptable: 0.2 });
            html += `<tr><td>Reference Point Accuracy (RMSE):</td><td style="text-align: right; color: ${refAccColor};">${summary.trajectory_quality.ref_point_accuracy_rmse.toFixed(3)}m</td></tr>`;
            html += '</table></div>';

            // Path Tracking
            html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
            html += '<h3 style="margin-top: 0; color: #4a90e2;">Path Tracking</h3>';
            html += '<table style="width: 100%; color: #e0e0e0;">';
            const latErrColor = getColorForValue(summary.path_tracking.lateral_error_rmse, { good: 0.2, acceptable: 0.4 });
            html += `<tr><td>Lateral Error (RMSE):</td><td style="text-align: right; color: ${latErrColor};">${summary.path_tracking.lateral_error_rmse.toFixed(3)}m</td></tr>`;
            const latMaxColor = getColorForValue(summary.path_tracking.lateral_error_max, { good: 0.5, acceptable: 1.0 });
            html += `<tr><td>Lateral Error (Max):</td><td style="text-align: right; color: ${latMaxColor};">${summary.path_tracking.lateral_error_max.toFixed(3)}m</td></tr>`;
            const latP95Color = getColorForValue(summary.path_tracking.lateral_error_p95, { good: 0.4, acceptable: 0.8 });
            html += `<tr><td>Lateral Error (P95):</td><td style="text-align: right; color: ${latP95Color};">${summary.path_tracking.lateral_error_p95.toFixed(3)}m</td></tr>`;
            const headingErrDeg = summary.path_tracking.heading_error_rmse * 180 / Math.PI;
            const headingColor = getColorForValue(headingErrDeg, { good: 10, acceptable: 20 });
            html += `<tr><td>Heading Error (RMSE):</td><td style="text-align: right; color: ${headingColor};">${headingErrDeg.toFixed(1)}°</td></tr>`;
            const timeInLaneColor = summary.path_tracking.time_in_lane >= 90 ? '#4caf50' : summary.path_tracking.time_in_lane >= 70 ? '#ffa500' : '#ff6b6b';
            html += `<tr><td>Time in Lane:</td><td style="text-align: right; color: ${timeInLaneColor};">${summary.path_tracking.time_in_lane.toFixed(1)}%</td></tr>`;
            html += '</table></div>';

            // Control Smoothness
            html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
            html += '<h3 style="margin-top: 0; color: #4a90e2;">Control Smoothness</h3>';
            html += '<table style="width: 100%; color: #e0e0e0;">';
            const jerkColor = getColorForValue(summary.control_smoothness.steering_jerk_max, { good: 0.5, acceptable: 1.0 });
            html += `<tr><td>Steering Jerk (Max):</td><td style="text-align: right; color: ${jerkColor};">${summary.control_smoothness.steering_jerk_max.toFixed(3)}/s²</td></tr>`;
            const rateColor = getColorForValue(summary.control_smoothness.steering_rate_max, { good: 2.0, acceptable: 4.0 });
            html += `<tr><td>Steering Rate (Max):</td><td style="text-align: right; color: ${rateColor};">${summary.control_smoothness.steering_rate_max.toFixed(3)}/s</td></tr>`;
            const smoothnessColor = summary.control_smoothness.steering_smoothness >= 2.0 ? '#4caf50' : summary.control_smoothness.steering_smoothness >= 1.0 ? '#ffa500' : '#ff6b6b';
            html += `<tr><td>Steering Smoothness:</td><td style="text-align: right; color: ${smoothnessColor};">${summary.control_smoothness.steering_smoothness.toFixed(2)}</td></tr>`;
            const oscColor = getColorForValue(summary.control_smoothness.oscillation_frequency, { good: 1.0, acceptable: 2.0 });
            html += `<tr><td>Oscillation Frequency:</td><td style="text-align: right; color: ${oscColor};">${summary.control_smoothness.oscillation_frequency.toFixed(2)}Hz</td></tr>`;
            html += '</table></div>';

            // Control stability (straight-line oscillation + adaptive deadband)
            const controlStability = summary.control_stability || null;
            if (controlStability) {
                const oscMean = controlStability.straight_oscillation_mean ?? null;
                const oscMax = controlStability.straight_oscillation_max ?? null;
                const straightFrac = controlStability.straight_fraction ?? null;
                const deadbandMean = controlStability.tuned_deadband_mean ?? null;
                const deadbandMax = controlStability.tuned_deadband_max ?? null;
                const smoothingMean = controlStability.tuned_smoothing_mean ?? null;
                if (oscMean !== null && straightFrac !== null) {
                    const stabilityColor = oscMean > 0.2 ? '#ff6b6b' : oscMean > 0.1 ? '#ffa500' : '#4caf50';
                    html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
                    html += '<h3 style="margin-top: 0; color: #4a90e2;">Control Stability (Straight)</h3>';
                    html += '<table style="width: 100%; color: #e0e0e0;">';
                    html += `<tr><td>Straight Coverage:</td><td style="text-align: right;">${straightFrac.toFixed(1)}%</td></tr>`;
                    html += `<tr><td>Oscillation Mean:</td><td style="text-align: right; color: ${stabilityColor};">${oscMean.toFixed(3)}</td></tr>`;
                    html += `<tr><td>Oscillation Max:</td><td style="text-align: right;">${oscMax !== null ? oscMax.toFixed(3) : '-'}</td></tr>`;
                    html += `<tr><td>Tuned Deadband (Mean):</td><td style="text-align: right;">${deadbandMean !== null ? deadbandMean.toFixed(3) : '-'}</td></tr>`;
                    html += `<tr><td>Tuned Deadband (Max):</td><td style="text-align: right;">${deadbandMax !== null ? deadbandMax.toFixed(3) : '-'}</td></tr>`;
                    html += `<tr><td>Smoothing α (Mean):</td><td style="text-align: right;">${smoothingMean !== null ? smoothingMean.toFixed(3) : '-'}</td></tr>`;
                    if (controlStability.straight_sign_mismatch_rate !== undefined) {
                        const mismatchRate = controlStability.straight_sign_mismatch_rate;
                        const mismatchEvents = controlStability.straight_sign_mismatch_events ?? 0;
                        const mismatchColor = mismatchRate > 15 ? '#ff6b6b' : mismatchRate > 5 ? '#ffa500' : '#4caf50';
                        html += `<tr><td>Straight Sign Mismatch Rate:</td><td style="text-align: right; color: ${mismatchColor};">${mismatchRate.toFixed(1)}%</td></tr>`;
                        html += `<tr><td>Straight Sign Mismatch Events:</td><td style="text-align: right;">${mismatchEvents}</td></tr>`;
                    }
                    html += '</table></div>';

                }
            }
            
            html += '<h3 style="margin: 1rem 0 0.5rem; color: #9bdcff;">Longitudinal</h3>';
            // Speed control
            const speedControl = summary.speed_control || null;
            if (speedControl) {
                const speedRmse = speedControl.speed_error_rmse ?? null;
                const speedMean = speedControl.speed_error_mean ?? null;
                const speedMax = speedControl.speed_error_max ?? null;
                const overspeedRate = speedControl.speed_overspeed_rate ?? null;
                const speedLimitZeroRate = speedControl.speed_limit_zero_rate ?? null;
                const surgeCount = speedControl.speed_surge_count ?? null;
                const surgeAvgDrop = speedControl.speed_surge_avg_drop ?? null;
                const surgeP95Drop = speedControl.speed_surge_p95_drop ?? null;
                if (speedRmse !== null) {
                    const speedColor = speedRmse > 2.0 ? '#ff6b6b' : '#4caf50';
                    html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
                    html += '<h3 style="margin-top: 0; color: #4a90e2;">Speed Control</h3>';
                    html += '<table style="width: 100%; color: #e0e0e0;">';
                    html += `<tr><td>Speed Error (RMSE):</td><td style="text-align: right; color: ${speedColor};">${speedRmse.toFixed(2)} m/s</td></tr>`;
                    html += `<tr><td>Speed Error (Mean):</td><td style="text-align: right;">${speedMean !== null ? speedMean.toFixed(2) : '-'}</td></tr>`;
                    html += `<tr><td>Speed Error (Max):</td><td style="text-align: right;">${speedMax !== null ? speedMax.toFixed(2) : '-'}</td></tr>`;
                    html += `<tr><td>Overspeed Rate (>0.5 m/s):</td><td style="text-align: right;">${overspeedRate !== null ? overspeedRate.toFixed(1) : '-'}%</td></tr>`;
                    html += `<tr><td>Speed Limit Missing:</td><td style="text-align: right;">${speedLimitZeroRate !== null ? speedLimitZeroRate.toFixed(1) : '-'}%</td></tr>`;
                    if (surgeCount !== null) {
                        html += `<tr><td>Straight Surge Drops (>=1.0 m/s):</td><td style="text-align: right;">${surgeCount}</td></tr>`;
                        html += `<tr><td>Straight Surge Drop (Avg):</td><td style="text-align: right;">${surgeAvgDrop !== null ? surgeAvgDrop.toFixed(2) : '-'}</td></tr>`;
                        html += `<tr><td>Straight Surge Drop (P95):</td><td style="text-align: right;">${surgeP95Drop !== null ? surgeP95Drop.toFixed(2) : '-'}</td></tr>`;
                    }
                    html += '</table></div>';
                }
            }
            // Comfort
            const comfort = summary.comfort || null;
            if (comfort) {
                const comfortColor = (comfort.acceleration_p95 > 2.5 || comfort.jerk_p95 > 5.0) ? '#ff6b6b' : '#4caf50';
                const latAccelP95 = comfort.lateral_accel_p95 ?? null;
                const latJerkP95 = comfort.lateral_jerk_p95 ?? null;
                const accelP95Filtered = comfort.acceleration_p95_filtered ?? null;
                const jerkP95Filtered = comfort.jerk_p95_filtered ?? null;
                html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
                html += '<h3 style="margin-top: 0; color: #4a90e2;">Comfort</h3>';
                html += '<table style="width: 100%; color: #e0e0e0;">';
                html += `<tr><td>Accel P95:</td><td style="text-align: right; color: ${comfortColor};">${comfort.acceleration_p95.toFixed(2)} m/s²</td></tr>`;
                if (accelP95Filtered !== null) {
                    html += `<tr><td>Accel P95 (Filt):</td><td style="text-align: right;">${accelP95Filtered.toFixed(2)} m/s²</td></tr>`;
                }
                html += `<tr><td>Jerk P95:</td><td style="text-align: right; color: ${comfortColor};">${comfort.jerk_p95.toFixed(2)} m/s³</td></tr>`;
                if (jerkP95Filtered !== null) {
                    html += `<tr><td>Jerk P95 (Filt):</td><td style="text-align: right;">${jerkP95Filtered.toFixed(2)} m/s³</td></tr>`;
                }
                html += `<tr><td>Lat Accel P95:</td><td style="text-align: right;">${latAccelP95 !== null ? latAccelP95.toFixed(2) : '-'} m/s²</td></tr>`;
                html += `<tr><td>Lat Jerk P95:</td><td style="text-align: right;">${latJerkP95 !== null ? latJerkP95.toFixed(2) : '-'} m/s³</td></tr>`;
                html += `<tr><td>Steering Jerk (Max):</td><td style="text-align: right;">${comfort.steering_jerk_max.toFixed(2)}</td></tr>`;
                html += '</table></div>';
            }

            // Safety
            html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
            html += '<h3 style="margin-top: 0; color: #4a90e2;">Safety</h3>';
            html += '<table style="width: 100%; color: #e0e0e0; margin-bottom: 1rem;">';
            html += `<tr><td>Out-of-Lane Events:</td><td style="text-align: right; color: ${summary.safety.out_of_lane_events === 0 ? '#4caf50' : '#ff6b6b'};">${summary.safety.out_of_lane_events}</td></tr>`;
            html += `<tr><td>Out-of-Lane Time:</td><td style="text-align: right; color: ${summary.safety.out_of_lane_time < 5 ? '#4caf50' : summary.safety.out_of_lane_time < 10 ? '#ffa500' : '#ff6b6b'};">${summary.safety.out_of_lane_time.toFixed(1)}%</td></tr>`;
            html += '</table>';
            
            // Out-of-Lane Events List
            if (summary.safety.out_of_lane_events_list && summary.safety.out_of_lane_events_list.length > 0) {
                html += '<div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #444;">';
                html += '<h4 style="margin-top: 0; margin-bottom: 0.75rem; color: #ffa500; font-size: 0.9rem;">Out-of-Lane Events (click to jump to frame):</h4>';
                html += '<div style="max-height: 300px; overflow-y: auto;">';
                html += '<table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">';
                html += '<thead><tr style="color: #888; border-bottom: 1px solid #444;">';
                html += '<th style="text-align: left; padding: 0.25rem 0.5rem;">#</th>';
                html += '<th style="text-align: left; padding: 0.25rem 0.5rem;">Frames</th>';
                html += '<th style="text-align: right; padding: 0.25rem 0.5rem;">Duration</th>';
                html += '<th style="text-align: right; padding: 0.25rem 0.5rem;">Max Error</th>';
                html += '<th style="text-align: left; padding: 0.25rem 0.5rem;">Source</th>';
                html += '</tr></thead><tbody>';
                
                summary.safety.out_of_lane_events_list.forEach((event, idx) => {
                    const isFailureEvent = summary.executive_summary.failure_detected && 
                                         event.start_frame >= (summary.executive_summary.failure_frame || 0);
                    const rowColor = isFailureEvent ? '#3a1a1a' : '#2a2a2a';
                    const textColor = isFailureEvent ? '#ff6b6b' : '#e0e0e0';
                    
                    html += `<tr style="background: ${rowColor}; cursor: pointer; border-bottom: 1px solid #333;" `;
                    html += `onclick="window.visualizer.goToFrame(${event.start_frame});" `;
                    html += `onmouseover="this.style.background='#3a3a3a';" `;
                    html += `onmouseout="this.style.background='${rowColor}';" `;
                    html += `title="Click to jump to frame ${event.start_frame}">`;
                    html += `<td style="padding: 0.25rem 0.5rem; color: ${textColor};">${idx + 1}</td>`;
                    html += `<td style="padding: 0.25rem 0.5rem; color: ${textColor}; font-weight: ${isFailureEvent ? 'bold' : 'normal'};">`;
                    html += `${event.start_frame}${event.end_frame !== event.start_frame ? `-${event.end_frame}` : ''}`;
                    if (isFailureEvent) {
                        html += ' <span style="color: #ff6b6b; font-size: 0.75em;">(FAILURE)</span>';
                    }
                    html += `</td>`;
                    html += `<td style="text-align: right; padding: 0.25rem 0.5rem; color: ${textColor};">${event.duration_seconds.toFixed(2)}s</td>`;
                    html += `<td style="text-align: right; padding: 0.25rem 0.5rem; color: ${textColor};">${event.max_error.toFixed(3)}m</td>`;
                    html += `<td style="padding: 0.25rem 0.5rem; color: ${textColor}; font-size: 0.75em;">${event.error_source || 'unknown'}</td>`;
                    html += `</tr>`;
                });
                
                html += '</tbody></table>';
                html += '</div>';
                html += '</div>';
            }
            html += '</div>';
            
            // Recommendations
            if (summary.recommendations && summary.recommendations.length > 0) {
                html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
                html += '<h3 style="margin-top: 0; color: #4a90e2;">Recommendations</h3>';
                html += '<ul style="margin: 0; padding-left: 1.5rem; color: #e0e0e0;">';
                summary.recommendations.forEach(rec => {
                    html += `<li style="margin-bottom: 0.5rem;">${rec}</li>`;
                });
                html += '</ul></div>';
            }
            
            html += '</div>';
            summaryContent.innerHTML = html;
            this.bindPerceptionQuestionsRunner();
            
        } catch (error) {
            console.error('Error loading summary:', error);
            summaryContent.innerHTML = `<p style="color: #ff6b6b; text-align: center; padding: 2rem;">Error loading summary: ${error.message}</p>`;
        }
    }

    bindPerceptionQuestionsRunner() {
        const runBtn = document.getElementById('run-perception-questions-btn');
        if (!runBtn) return;
        runBtn.onclick = () => this.runPerceptionQuestionsFromSummary();
    }

    async runPerceptionQuestionsFromSummary() {
        if (!this.currentRecording) return;
        const runBtn = document.getElementById('run-perception-questions-btn');
        const statusEl = document.getElementById('run-perception-questions-status');
        const parsedEl = document.getElementById('run-perception-questions-parsed');
        const outputEl = document.getElementById('run-perception-questions-output');
        if (!runBtn || !statusEl || !parsedEl || !outputEl) return;

        runBtn.disabled = true;
        runBtn.style.opacity = '0.7';
        runBtn.style.cursor = 'not-allowed';
        statusEl.textContent = 'Running analyzer...';
        statusEl.style.color = '#ffa500';
        parsedEl.innerHTML = '';
        outputEl.style.display = 'none';
        outputEl.textContent = '';

        try {
            const response = await fetch(
                `/api/recording/${this.currentRecording}/run-perception-questions`,
                { method: 'POST' }
            );
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || `HTTP ${response.status}`);
            }

            const questions = data.questions || {};
            const ordered = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8'];
            const items = ordered
                .filter((q) => questions[q] !== undefined)
                .map((q) => `${q.toUpperCase()}: ${this.escapeHtml(questions[q])}`);
            parsedEl.innerHTML = items.length > 0
                ? `<strong>Parsed Results:</strong> ${items.join(' | ')}`
                : '<strong>Parsed Results:</strong> unavailable';

            outputEl.textContent = data.output || '(no output)';
            outputEl.style.display = 'block';
            statusEl.textContent = `Completed (exit code ${data.return_code})`;
            statusEl.style.color = data.ok ? '#4caf50' : '#ff6b6b';
        } catch (error) {
            statusEl.textContent = `Failed: ${error.message}`;
            statusEl.style.color = '#ff6b6b';
        } finally {
            runBtn.disabled = false;
            runBtn.style.opacity = '1';
            runBtn.style.cursor = 'pointer';
        }
    }
    
    async loadIssues() {
        if (!this.currentRecording) return;
        
        const issuesContent = document.getElementById('issues-content');
        if (!issuesContent) return;
        
        // Check if "Analyze to Failure" checkbox is checked
        const analyzeToFailure = document.getElementById('analyze-to-failure')?.checked || false;
        
        issuesContent.innerHTML = '<p style="color: #888; text-align: center; padding: 2rem;">Loading issues...</p>';
        
        try {
            const url = `/api/recording/${this.currentRecording}/issues${analyzeToFailure ? '?analyze_to_failure=true' : ''}`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const issuesData = await response.json();
            
            if (issuesData.error) {
                issuesContent.innerHTML = `<p style="color: #ff6b6b; text-align: center; padding: 2rem;">Error: ${issuesData.error}</p>`;
                return;
            }
            
            // Build issues HTML
            let html = '<div style="padding: 1rem;">';
            
            // Summary
            if (issuesData.summary) {
                const summary = issuesData.summary;
                const lowCount = Number(summary?.by_severity?.low || 0);
                const totalCount = Number(summary?.total_issues || 0);
                const actionableCount = Math.max(0, totalCount - lowCount);
                html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
                html += '<h3 style="margin-top: 0; color: #4a90e2;">Issues Summary</h3>';
                html += `<div style="color: #e0e0e0; margin-bottom: 0.5rem;">Total Issues: <strong style="color: ${summary.total_issues > 0 ? '#ff6b6b' : '#4caf50'}">${summary.total_issues}</strong></div>`;
                html += `<div style="color: #9fb3c8; margin-bottom: 0.25rem; font-size: 0.9rem;">Actionable by default (critical/high/medium): <strong>${actionableCount}</strong></div>`;
                
                if (summary.by_severity) {
                    html += '<div style="margin-top: 0.5rem; font-size: 0.9rem;">';
                    html += `<span style="color: #ff6b6b;">Critical: ${summary.by_severity.critical || 0}</span> | `;
                    html += `<span style="color: #ffa500;">High: ${summary.by_severity.high || 0}</span> | `;
                    html += `<span style="color: #ffa500;">Medium: ${summary.by_severity.medium || 0}</span> | `;
                    html += `<span style="color: #888;">Low: ${summary.by_severity.low || 0}</span>`;
                    html += '</div>';
                }
                
                if (summary.by_type && Object.keys(summary.by_type).length > 0) {
                    html += '<div style="margin-top: 0.5rem; font-size: 0.9rem; color: #888;">By Type: ';
                    const typeList = Object.entries(summary.by_type).map(([type, count]) => `${type}: ${count}`).join(', ');
                    html += typeList;
                    html += '</div>';
                }
                if (summary.road_departure_start_frame !== null && summary.road_departure_start_frame !== undefined) {
                    html += `<div style="margin-top: 0.5rem; font-size: 0.9rem; color: #ffb74d;">Road departure starts: frame <strong>${summary.road_departure_start_frame}</strong></div>`;
                }
                if (summary.centerline_cross_start_frame !== null && summary.centerline_cross_start_frame !== undefined) {
                    html += `<div style="margin-top: 0.25rem; font-size: 0.9rem; color: #ef5350;">Centerline crossed: frame <strong>${summary.centerline_cross_start_frame}</strong></div>`;
                }
                
                html += '</div>';
            }

            // Filters and visibility controls (show before timeline for discoverability)
            html += '<div style="margin-bottom: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">';
            html += '<button class="issue-filter-btn active" data-filter="all" style="padding: 0.5rem 1rem; background: #4a90e2; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">All</button>';
            html += '<button class="issue-filter-btn" data-filter="extreme_coefficients" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Extreme Coefficients</button>';
            html += '<button class="issue-filter-btn" data-filter="perception_instability" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Perception Instability</button>';
            html += '<button class="issue-filter-btn" data-filter="right_lane_low_visibility" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Right Lane Low Visibility</button>';
            html += '<button class="issue-filter-btn" data-filter="right_lane_edge_contact" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Right Lane Edge Contact</button>';
            html += '<button class="issue-filter-btn" data-filter="high_lateral_error" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">High Lateral Error</button>';
            html += '<button class="issue-filter-btn" data-filter="negative_control_correlation" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Negative Correlation</button>';
            html += '<button class="issue-filter-btn" data-filter="perception_failure" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Perception Failure</button>';
            html += '<button class="issue-filter-btn" data-filter="straight_sign_mismatch" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Straight Sign Mismatch</button>';
            html += '<button class="issue-filter-btn" data-filter="centerline_cross" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Centerline Cross</button>';
            html += '<button class="issue-filter-btn" data-filter="out_of_lane" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Out of Lane</button>';
            html += '<button class="issue-filter-btn" data-filter="emergency_stop" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Emergency Stop</button>';
            html += '<button class="issue-filter-btn" data-filter="heading_jump" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Heading Jump</button>';
            html += '</div>';
            html += '<div style="margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem; color: #9fb3c8; font-size: 0.9rem;">';
            html += `<label style="display:flex; align-items:center; gap:0.4rem; cursor:pointer;"><input type="checkbox" id="toggle-show-informational-issues" ${this.showInformationalIssues ? 'checked' : ''}> Show informational/low-severity issues</label>`;
            html += '</div>';

            // Causal timeline (respect informational visibility toggle)
            const timelineEvents = Array.isArray(issuesData.causal_timeline)
                ? issuesData.causal_timeline.filter((event) =>
                    this.showInformationalIssues || String(event?.severity || '').toLowerCase() !== 'low'
                )
                : [];
            if (timelineEvents.length > 0) {
                const phaseColor = {
                    perception: '#4fc3f7',
                    trajectory: '#ba68c8',
                    control: '#ffb74d',
                    downstream: '#ef5350',
                };
                const phaseIcon = {
                    perception: '👁️',
                    trajectory: '🛣️',
                    control: '🎛️',
                    downstream: '🚨',
                };
                html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
                html += '<h3 style="margin-top: 0; color: #4a90e2;">Causal Event Timeline</h3>';
                html += '<div style="display: flex; flex-direction: column; gap: 0.5rem;">';
                timelineEvents.forEach((event, idx) => {
                    const p = event.phase || 'downstream';
                    const color = phaseColor[p] || '#888';
                    const icon = phaseIcon[p] || '⚠️';
                    const frame = Number(event.frame || 0);
                    html += `<div style="display: flex; gap: 0.6rem; align-items: center; background: #1f1f1f; border-left: 3px solid ${color}; padding: 0.5rem 0.7rem; border-radius: 4px;">`;
                    html += `<div style="color: ${color}; min-width: 84px; font-size: 0.82rem; font-weight: bold;">${icon} ${this.escapeHtml(p)}</div>`;
                    html += `<div style="color: #cfd8dc; font-size: 0.82rem; min-width: 72px;">F${frame}</div>`;
                    html += `<div style="color: #e0e0e0; font-size: 0.86rem; flex: 1;">${this.escapeHtml(event.type || 'event')}: ${this.escapeHtml(event.description || '')}</div>`;
                    html += `<button style="padding: 0.2rem 0.6rem; background: #4a90e2; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.8rem;" onclick="window.visualizer.jumpToFrame(${frame})">Jump</button>`;
                    html += `</div>`;
                });
                html += '</div>';
                html += '</div>';
            }
            
            // Issues list
            if (issuesData.issues && issuesData.issues.length > 0) {
                html += '<div id="issues-list" style="display: flex; flex-direction: column; gap: 0.75rem;">';
                
                issuesData.issues.forEach((issue, idx) => {
                    const severityColor = {
                        'critical': '#ff6b6b',
                        'high': '#ff6b6b',
                        'medium': '#ffa500',
                        'low': '#ffa500'
                    }[issue.severity] || '#888';
                    
                    const typeIcon = {
                        'extreme_coefficients': '📐',
                        'perception_instability': '📊',
                        'right_lane_low_visibility': '👀',
                        'right_lane_edge_contact': '🧭',
                        'high_lateral_error': '📏',
                        'negative_control_correlation': '↔️',
                        'perception_failure': '👁️',
                        'straight_sign_mismatch': '↩️',
                        'centerline_cross': '🚧',
                        'out_of_lane': '🚫',
                        'emergency_stop': '🛑',
                        'heading_jump': '🔄'
                    }[issue.type] || '⚠️';
                    
                    html += `<div class="issue-item" data-issue-type="${issue.type}" data-issue-severity="${issue.severity || 'unknown'}" style="background: #2a2a2a; padding: 1rem; border-radius: 8px; border-left: 4px solid ${severityColor}; cursor: pointer;" onclick="window.visualizer.jumpToFrame(${issue.frame})">`;
                    html += `<div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">`;
                    html += `<div style="flex: 1;">`;
                    html += `<div style="font-weight: bold; color: ${severityColor}; margin-bottom: 0.25rem;">${typeIcon} Frame ${issue.frame}: ${issue.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>`;
                    html += `<div style="color: #e0e0e0; font-size: 0.9rem;">${issue.description}</div>`;
                    if (issue.type === 'right_lane_low_visibility') {
                        html += '<div style="margin-top: 0.35rem; color: #9fb3c8; font-size: 0.82rem;">Inspect with overlays: <strong>Model Perception Fit Points</strong> + <strong>Segmentation Fit Range</strong> (and CV ROI only if debugging CV fallback), then compare with Trajectory tab `perception_center_x` and Diagnostics GT-vs-Perception hotspots.</div>';
                    }
                    html += `</div>`;
                    html += `<button style="padding: 0.25rem 0.75rem; background: #4a90e2; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem; white-space: nowrap;" onclick="event.stopPropagation(); window.visualizer.jumpToFrame(${issue.frame})">Jump →</button>`;
                    html += `</div>`;
                    html += `</div>`;
                });
                
                html += '</div>';
            } else {
                html += '<div style="background: #2a2a2a; padding: 2rem; border-radius: 8px; text-align: center; color: #4caf50;">';
                html += '<div style="font-size: 2rem; margin-bottom: 0.5rem;">✓</div>';
                html += '<div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 0.25rem;">No Issues Detected</div>';
                html += '<div style="color: #888; font-size: 0.9rem;">Recording appears to be clean!</div>';
                html += '</div>';
            }
            
            html += '</div>';
            issuesContent.innerHTML = html;
            
            // Setup filter buttons
            const filterButtons = issuesContent.querySelectorAll('.issue-filter-btn');
            const issueItems = issuesContent.querySelectorAll('.issue-item');
            const informationalToggle = issuesContent.querySelector('#toggle-show-informational-issues');
            let activeFilter = 'all';

            const applyIssueVisibility = () => {
                const showInformational = Boolean(this.showInformationalIssues);
                issueItems.forEach(item => {
                    const typeMatch = activeFilter === 'all' || item.dataset.issueType === activeFilter;
                    const sev = String(item.dataset.issueSeverity || '').toLowerCase();
                    const isInformational = sev === 'low';
                    const severityMatch = showInformational || !isInformational;
                    item.style.display = (typeMatch && severityMatch) ? 'block' : 'none';
                });
            };

            filterButtons.forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    activeFilter = btn.dataset.filter;
                    
                    // Update button states
                    filterButtons.forEach(b => {
                        if (b === btn) {
                            b.classList.add('active');
                            b.style.background = '#4a90e2';
                        } else {
                            b.classList.remove('active');
                            b.style.background = '#555';
                        }
                    });
                    
                    applyIssueVisibility();
                });
            });
            if (informationalToggle) {
                informationalToggle.addEventListener('change', () => {
                    this.showInformationalIssues = Boolean(informationalToggle.checked);
                    applyIssueVisibility();
                });
            }
            applyIssueVisibility();
            
        } catch (error) {
            console.error('Error loading issues:', error);
            issuesContent.innerHTML = `<p style="color: #ff6b6b; text-align: center; padding: 2rem;">Error loading issues: ${error.message}</p>`;
        }
    }
    
    async loadDiagnostics() {
        if (!this.currentRecording) return;
        
        const diagnosticsContent = document.getElementById('diagnostics-content');
        if (!diagnosticsContent) return;
        
        // Check if "Analyze to Failure" checkbox is checked
        const analyzeToFailure = document.getElementById('analyze-to-failure')?.checked || false;
        const entryStartDistance = parseFloat(document.getElementById('diag-entry-start-distance')?.value || `${this.curveEntryStartDistanceM}`);
        const entryWindowDistance = parseFloat(document.getElementById('diag-entry-window-distance')?.value || `${this.curveEntryWindowDistanceM}`);
        
        diagnosticsContent.innerHTML = '<p style="color: #888; text-align: center; padding: 2rem;">Loading diagnostics...</p>';
        
        try {
            const params = new URLSearchParams();
            if (analyzeToFailure) {
                params.set('analyze_to_failure', 'true');
            }
            if (Number.isFinite(entryStartDistance)) {
                this.curveEntryStartDistanceM = entryStartDistance;
                params.set('curve_entry_start_distance_m', String(entryStartDistance));
            }
            if (Number.isFinite(entryWindowDistance)) {
                this.curveEntryWindowDistanceM = entryWindowDistance;
                params.set('curve_entry_window_distance_m', String(entryWindowDistance));
            }
            const url = `/api/recording/${this.currentRecording}/diagnostics?${params.toString()}`;
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const diagnostics = await response.json();
            
            if (diagnostics.error) {
                diagnosticsContent.innerHTML = `<p style="color: #ff6b6b; text-align: center; padding: 2rem;">Error: ${diagnostics.error}</p>`;
                return;
            }
            
            const fmtOpt = (value, digits = 3) => (value === null || value === undefined
                ? '-'
                : value.toFixed(digits));
            
            // Build diagnostics HTML
            let html = '<div style="padding: 1rem;">';
            
            // Diagnosis Summary
            html += '<div style="background: #2a2a2a; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;">';
            html += '<h2 style="margin-top: 0; color: #4a90e2;">Diagnosis Summary</h2>';
            
            const primaryIssue = diagnostics.diagnosis?.primary_issue || 'unknown';
            const trajScore = diagnostics.diagnosis?.trajectory_score || 0;
            const ctrlScore = diagnostics.diagnosis?.control_score || 0;
            
            // Determine severity based on actual scores, not just which is lower
            let diagnosisColor = '#4caf50';
            let diagnosisText = 'System appears to be working correctly';
            
            if (primaryIssue === 'trajectory') {
                // Use score to determine severity
                if (trajScore < 70) {
                    diagnosisColor = '#ff6b6b';  // Red for poor
                    diagnosisText = '⚠️ Trajectory planning needs attention';
                } else if (trajScore < 80) {
                    diagnosisColor = '#ffa500';  // Orange for acceptable
                    diagnosisText = '⚠️ Trajectory planning may need improvement';
                } else {
                    diagnosisColor = '#4caf50';
                    diagnosisText = 'System appears to be working correctly';
                }
            } else if (primaryIssue === 'control') {
                // Use score to determine severity
                if (ctrlScore < 70) {
                    diagnosisColor = '#ff6b6b';  // Red for poor
                    diagnosisText = '⚠️ Control system needs tuning';
                } else if (ctrlScore < 80) {
                    diagnosisColor = '#ffa500';  // Orange for acceptable
                    diagnosisText = '⚠️ Control system may need tuning';
                } else {
                    diagnosisColor = '#4caf50';
                    diagnosisText = 'System appears to be working correctly';
                }
            }
            
            html += `<div style="font-size: 1.2rem; font-weight: bold; color: ${diagnosisColor}; margin: 1rem 0;">${diagnosisText}</div>`;
            html += `<div style="color: #a0a0a0; margin-bottom: 1rem;">Trajectory Quality: ${trajScore.toFixed(1)}% | Control Quality: ${ctrlScore.toFixed(1)}%</div>`;
            
            // Recommendations
            if (diagnostics.diagnosis?.recommendations && diagnostics.diagnosis.recommendations.length > 0) {
                html += '<div style="margin-top: 1rem;"><strong style="color: #ffa500;">Recommendations:</strong><ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #e0e0e0;">';
                diagnostics.diagnosis.recommendations.forEach(rec => {
                    html += `<li style="margin-bottom: 0.5rem;">${rec}</li>`;
                });
                html += '</ul></div>';
            }
            html += '</div>';
            
            // Trajectory Analysis
            if (diagnostics.trajectory_analysis) {
                const traj = diagnostics.trajectory_analysis;
                const trajColor = traj.quality_score >= 80 ? '#4caf50' : traj.quality_score >= 60 ? '#ffa500' : '#ff6b6b';
                
                html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
                html += '<h3 style="margin-top: 0; color: #4a90e2;">Trajectory Quality</h3>';
                html += `<div style="font-size: 1.5rem; font-weight: bold; color: ${trajColor}; margin-bottom: 1rem;">${traj.quality_score.toFixed(1)}%</div>`;
                
                html += '<table style="width: 100%; color: #e0e0e0; margin-bottom: 1rem;">';
                if (traj.reference_point_stats) {
                    html += `<tr><td>Mean Reference X:</td><td style="text-align: right;">${traj.reference_point_stats.mean_abs.toFixed(3)}m</td></tr>`;
                    html += `<tr><td>Max Reference X:</td><td style="text-align: right;">${traj.reference_point_stats.max_abs.toFixed(3)}m</td></tr>`;
                    html += `<tr><td>Mean Change:</td><td style="text-align: right;">${traj.reference_point_stats.mean_change.toFixed(3)}m</td></tr>`;
                }
                if (traj.accuracy_vs_ground_truth && traj.accuracy_vs_ground_truth.rmse !== null) {
                    const accColor = traj.accuracy_vs_ground_truth.rmse < 0.2 ? '#4caf50' : traj.accuracy_vs_ground_truth.rmse < 0.5 ? '#ffa500' : '#ff6b6b';
                    html += `<tr><td>Accuracy vs Ground Truth (RMSE):</td><td style="text-align: right; color: ${accColor};">${traj.accuracy_vs_ground_truth.rmse.toFixed(3)}m</td></tr>`;
                }
                html += '</table>';

                if (traj.perception_trajectory_attribution) {
                    const attr = traj.perception_trajectory_attribution;
                    const label = attr.attribution_label || 'unknown';
                    const labelColor = label.includes('perception-driven')
                        ? '#ffb74d'
                        : (label.includes('trajectory-logic') ? '#ef5350' : '#4caf50');
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #9b59b6; margin-top: 1rem;">';
                    html += '<strong style="color: #d2a8ff;">Perception → Trajectory Attribution</strong><br/>';
                    html += `<div style="margin-top: 0.4rem; color: ${labelColor};"><strong>${this.escapeHtml(label)}</strong></div>`;
                    html += '<table style="width: 100%; color: #e0e0e0; margin-top: 0.5rem;">';
                    html += `<tr><td>Ref vs Perception RMSE:</td><td style="text-align: right;">${fmtOpt(attr.ref_vs_perception_rmse, 3)}m</td></tr>`;
                    html += `<tr><td>Ref vs Perception Bias:</td><td style="text-align: right;">${fmtOpt(attr.ref_vs_perception_bias, 3)}m</td></tr>`;
                    html += `<tr><td>Ref vs Perception Corr:</td><td style="text-align: right;">${fmtOpt(attr.ref_vs_perception_correlation, 3)}</td></tr>`;
                    html += `<tr><td>Best Lag:</td><td style="text-align: right;">${attr.best_lag_frames === null || attr.best_lag_frames === undefined ? '-' : `${attr.best_lag_frames} frames`}</td></tr>`;
                    html += `<tr><td>Lag Correlation:</td><td style="text-align: right;">${fmtOpt(attr.best_lag_correlation, 3)}</td></tr>`;
                    html += `<tr><td>Perception→Ref Gain:</td><td style="text-align: right;">${fmtOpt(attr.gain_perception_to_ref, 3)}</td></tr>`;
                    html += `<tr><td>GT vs Perception RMSE:</td><td style="text-align: right;">${fmtOpt(attr.gt_perception_rmse, 3)}m</td></tr>`;
                    html += `<tr><td>GT vs Trajectory RMSE:</td><td style="text-align: right;">${fmtOpt(attr.gt_ref_rmse, 3)}m</td></tr>`;
                    html += `<tr><td>Stale Perception Rate:</td><td style="text-align: right;">${fmtOpt(attr.stale_perception_rate_pct, 1)}%</td></tr>`;
                    html += '</table></div>';
                }

                if (traj.perception_trajectory_hotspots && traj.perception_trajectory_hotspots.length > 0) {
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #9b59b6; margin-top: 1rem;">';
                    html += '<strong style="color: #d2a8ff;">Perception → Trajectory Hotspots</strong><br/>';
                    html += '<div style="color: #888; font-size: 0.85rem; margin-bottom: 0.5rem;">Frames where perception-vs-GT error is highest, side-by-side with trajectory.</div>';
                    html += '<div style="display: flex; flex-direction: column; gap: 0.5rem;">';
                    traj.perception_trajectory_hotspots.forEach((spot) => {
                        html += '<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: #2a2a2a; border-radius: 6px;">';
                        html += '<div style="color: #e0e0e0; font-size: 0.9rem;">';
                        html += `<strong>Frame ${spot.frame}</strong> · t=${spot.time.toFixed(2)}s · ${spot.segment}<br/>`;
                        html += `perc_vs_gt=${fmtOpt(spot.perception_vs_gt, 3)}m · ref_vs_gt=${fmtOpt(spot.ref_vs_gt, 3)}m · ref_vs_perc=${fmtOpt(spot.ref_vs_perception, 3)}m<br/>`;
                        html += `gt_center=${fmtOpt(spot.gt_center_x, 3)} · perc_center=${fmtOpt(spot.perception_center_x, 3)} · ref_x=${fmtOpt(spot.ref_x, 3)} · curv=${fmtOpt(spot.curvature, 4)}`;
                        html += '</div>';
                        html += `<button style="padding: 0.25rem 0.75rem; background: #7b61ff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem;" onclick="window.visualizer.jumpToFrame(${spot.frame})">Jump →</button>`;
                        html += '</div>';
                    });
                    html += '</div></div>';
                }
                
                if (traj.issues && traj.issues.length > 0) {
                    html += '<div style="background: #3a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #ff6b6b; margin-top: 1rem;">';
                    html += '<strong style="color: #ff6b6b;">Issues:</strong><ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #ff6b6b;">';
                    traj.issues.forEach(issue => {
                        html += `<li>${issue}</li>`;
                    });
                    html += '</ul></div>';
                }
                
                if (traj.warnings && traj.warnings.length > 0) {
                    html += '<div style="background: #3a2a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #ffa500; margin-top: 1rem;">';
                    html += '<strong style="color: #ffa500;">Warnings:</strong><ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #ffa500;">';
                    traj.warnings.forEach(warning => {
                        html += `<li>${warning}</li>`;
                    });
                    html += '</ul></div>';
                }
                
                html += '</div>';
            }
            
            // Control Analysis
            if (diagnostics.control_analysis) {
                const ctrl = diagnostics.control_analysis;
                const ctrlColor = ctrl.quality_score >= 80 ? '#4caf50' : ctrl.quality_score >= 60 ? '#ffa500' : '#ff6b6b';
                
                html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
                html += '<h3 style="margin-top: 0; color: #4a90e2;">Control Quality</h3>';
                html += `<div style="font-size: 1.5rem; font-weight: bold; color: ${ctrlColor}; margin-bottom: 1rem;">${ctrl.quality_score.toFixed(1)}%</div>`;
                
                // Add comparison note if RMSE is available
                if (ctrl.lateral_error && ctrl.lateral_error.rmse !== null && ctrl.lateral_error.rmse !== undefined) {
                    const rmseColor = ctrl.lateral_error.rmse < 0.2 ? '#4caf50' : ctrl.lateral_error.rmse < 0.4 ? '#ffa500' : '#ff6b6b';
                    html += `<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem; border-left: 3px solid ${rmseColor};">`;
                    html += `<strong style="color: ${rmseColor};">⚠️ Path Tracking Performance:</strong><br/>`;
                    html += `<span style="color: #e0e0e0;">Lateral Error RMSE: <strong style="color: ${rmseColor};">${ctrl.lateral_error.rmse.toFixed(3)}m</strong></span><br/>`;
                    html += `<small style="color: #888;">This matches the "Path Tracking" metric in Summary tab. `;
                    html += `RMSE ${ctrl.lateral_error.rmse >= 0.4 ? 'exceeds' : ctrl.lateral_error.rmse >= 0.2 ? 'is at' : 'is below'} the acceptable threshold (0.4m).</small>`;
                    html += '</div>';
                }
                
                html += '<table style="width: 100%; color: #e0e0e0; margin-bottom: 1rem;">';
                if (ctrl.steering_stats && ctrl.steering_stats.mean_abs !== null) {
                    html += `<tr><td>Mean Steering:</td><td style="text-align: right;">${ctrl.steering_stats.mean_abs.toFixed(3)}</td></tr>`;
                    html += `<tr><td>Max Steering:</td><td style="text-align: right;">${ctrl.steering_stats.max_abs.toFixed(3)}</td></tr>`;
                    html += `<tr><td>Mean Change:</td><td style="text-align: right;">${ctrl.steering_stats.mean_change.toFixed(3)}</td></tr>`;
                }
                if (ctrl.steering_correlation !== null && ctrl.steering_correlation !== undefined) {
                    const corrColor = ctrl.steering_correlation > 0.7 ? '#4caf50' : ctrl.steering_correlation > 0.3 ? '#ffa500' : '#ff6b6b';
                    const corrScopeMap = {
                        straight: ' (straight)',
                        straight_low_curvature: ' (straight, low curvature)',
                        low_curvature: ' (low curvature)'
                    };
                    const corrScope = corrScopeMap[ctrl.steering_correlation_scope] || '';
                    html += `<tr><td>Steering Correlation${corrScope}:</td><td style="text-align: right; color: ${corrColor};">${ctrl.steering_correlation.toFixed(3)}</td></tr>`;
                }
                if (ctrl.lateral_error) {
                    // Display RMSE if available (matches Path Tracking), otherwise fall back to mean
                    if (ctrl.lateral_error.rmse !== null && ctrl.lateral_error.rmse !== undefined) {
                        const latErrColor = ctrl.lateral_error.rmse < 0.2 ? '#4caf50' : ctrl.lateral_error.rmse < 0.4 ? '#ffa500' : '#ff6b6b';
                        html += `<tr><td>Lateral Error (RMSE):</td><td style="text-align: right; color: ${latErrColor};">${ctrl.lateral_error.rmse.toFixed(3)}m</td></tr>`;
                    }
                    if (ctrl.lateral_error.mean !== null && ctrl.lateral_error.mean !== undefined) {
                        const latMeanColor = ctrl.lateral_error.mean < 0.2 ? '#4caf50' : ctrl.lateral_error.mean < 0.4 ? '#ffa500' : '#ff6b6b';
                        html += `<tr><td>Lateral Error (Mean):</td><td style="text-align: right; color: ${latMeanColor};">${ctrl.lateral_error.mean.toFixed(3)}m</td></tr>`;
                    }
                    if (ctrl.lateral_error.max !== null && ctrl.lateral_error.max !== undefined) {
                        html += `<tr><td>Lateral Error (Max):</td><td style="text-align: right;">${ctrl.lateral_error.max.toFixed(3)}m</td></tr>`;
                    }
                }
                html += '</table>';

                if (ctrl.steering_limiter_analysis && ctrl.steering_limiter_analysis.available) {
                    const limiter = ctrl.steering_limiter_analysis;
                    const dominant = limiter.dominant_stage || 'none';
                    const dominantPct = limiter.dominant_stage_pct || 0;
                    const domColor = dominant === 'hard_clip' || dominant === 'jerk_limit'
                        ? '#ff6b6b'
                        : (dominant === 'rate_limit' || dominant === 'smoothing' ? '#ffa500' : '#4caf50');
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; margin-top: 1rem; border-left: 3px solid #7b61ff;">';
                    html += '<strong style="color: #b39bff;">Steering Limiter Root Cause</strong><br/>';
                    html += `<div style="margin-top: 0.35rem; color: ${domColor};">Dominant stage: <strong>${dominant}</strong> (${dominantPct.toFixed(1)}% of limited frames)</div>`;
                    html += `<div style="margin-top: 0.35rem; color: #e0e0e0;">Limited frames: ${limiter.total_limited_frames}</div>`;
                    html += '<table style="width: 100%; color: #e0e0e0; margin-top: 0.5rem;">';
                    html += `<tr><td>Rate Delta Mean:</td><td style="text-align: right;">${(limiter.rate_delta_mean || 0).toFixed(4)}</td></tr>`;
                    html += `<tr><td>Jerk Delta Mean:</td><td style="text-align: right;">${(limiter.jerk_delta_mean || 0).toFixed(4)}</td></tr>`;
                    html += `<tr><td>Hard Clip Delta Mean:</td><td style="text-align: right;">${(limiter.hard_clip_delta_mean || 0).toFixed(4)}</td></tr>`;
                    html += `<tr><td>Smoothing Delta Mean:</td><td style="text-align: right;">${(limiter.smoothing_delta_mean || 0).toFixed(4)}</td></tr>`;
                    html += '</table>';

                    // Graphical steering waterfall chart (Stage 6)
                    const stageSeries = [
                        { key: 'rate_limit', label: 'Rate Limit', mean: limiter.rate_delta_mean || 0, color: '#ffb74d' },
                        { key: 'jerk_limit', label: 'Jerk Limit', mean: limiter.jerk_delta_mean || 0, color: '#ef5350' },
                        { key: 'hard_clip', label: 'Hard Clip', mean: limiter.hard_clip_delta_mean || 0, color: '#ab47bc' },
                        { key: 'smoothing', label: 'Smoothing', mean: limiter.smoothing_delta_mean || 0, color: '#42a5f5' },
                    ];
                    const totalMeanDelta = stageSeries.reduce((s, x) => s + Math.max(0, x.mean), 0);
                    const counts = limiter.dominant_counts || {};
                    const totalLimited = Math.max(1, limiter.total_limited_frames || 0);

                    html += '<div style="margin-top: 0.8rem; background: #20242e; border: 1px solid #3b4252; border-radius: 6px; padding: 0.7rem;">';
                    html += '<div style="color: #d2a8ff; font-weight: bold; margin-bottom: 0.4rem;">Graphical Steering Waterfall</div>';
                    html += '<div style="color: #888; font-size: 0.82rem; margin-bottom: 0.55rem;">Bar length = mean steering reduction at each limiter stage. Dashed line markers show dominant-frame share per stage.</div>';

                    stageSeries.forEach((stage) => {
                        const widthPct = totalMeanDelta > 1e-9 ? (100.0 * Math.max(0, stage.mean) / totalMeanDelta) : 0.0;
                        const sharePct = 100.0 * ((counts[stage.key] || 0) / totalLimited);
                        html += '<div style="display: grid; grid-template-columns: 120px 1fr 92px; gap: 0.5rem; align-items: center; margin: 0.35rem 0;">';
                        html += `<div style="color: #cfd8dc; font-size: 0.85rem;">${stage.label}</div>`;
                        html += '<div style="position: relative; height: 12px; background: #131722; border-radius: 6px; overflow: hidden;">';
                        html += `<div style="position:absolute; left:0; top:0; bottom:0; width:${widthPct.toFixed(1)}%; background:${stage.color};"></div>`;
                        html += `<div style="position:absolute; left:${sharePct.toFixed(1)}%; top:0; bottom:0; border-left: 1px dashed #ffffffcc;"></div>`;
                        html += '</div>';
                        html += `<div style="color: #cfd8dc; font-size: 0.82rem; text-align: right;">Δ${(stage.mean || 0).toFixed(4)} · ${sharePct.toFixed(1)}%</div>`;
                        html += '</div>';
                    });

                    html += '<div style="color: #7f8fa6; font-size: 0.8rem; margin-top: 0.35rem;">';
                    html += `Total mean shaping Δ: ${totalMeanDelta.toFixed(4)} · Limited frames: ${limiter.total_limited_frames || 0}`;
                    html += '</div>';
                    html += '</div>';

                    if (limiter.phase_breakdown) {
                        const pb = limiter.phase_breakdown;
                        const phaseRows = [
                            ['pre_curve', pb.pre_curve],
                            ['curve_entry', pb.curve_entry],
                            ['curve_maintain', pb.curve_maintain],
                            ['overall', pb.overall],
                        ];
                        html += `<div style="margin-top: 0.5rem; color: #e0e0e0; font-size: 0.9rem;">`;
                        html += `Curve start frame: ${pb.curve_start_frame ?? '-'} (${pb.curve_start_source || 'unknown'})`;
                        html += `</div>`;
                        html += '<table style="width: 100%; color: #e0e0e0; margin-top: 0.5rem; font-size: 0.9rem;">';
                        html += '<tr><th style="text-align:left;">Phase</th><th style="text-align:right;">Limited</th><th style="text-align:right;">Dominant</th><th style="text-align:right;">Dominant %</th></tr>';
                        phaseRows.forEach(([name, info]) => {
                            if (!info) return;
                            html += `<tr><td>${name}</td><td style="text-align:right;">${info.limited_frames}/${info.frames}</td><td style="text-align:right;">${info.dominant_stage || 'none'}</td><td style="text-align:right;">${(info.dominant_pct || 0).toFixed(1)}%</td></tr>`;
                        });
                        html += '</table>';
                        if (pb.curve_entry && pb.curve_entry.limited_frames > 0) {
                            html += `<div style="margin-top: 0.35rem; color: #b39bff;">Curve-entry dominant limiter: <strong>${pb.curve_entry.dominant_stage}</strong> (${(pb.curve_entry.dominant_pct || 0).toFixed(1)}%)</div>`;
                        }
                    }
                    html += '</div>';
                }

                if (ctrl.curve_entry_feasibility) {
                    const feas = ctrl.curve_entry_feasibility;
                    const cls = feas.primary_classification || 'mixed-or-unclear';
                    const clsColorMap = {
                        'speed-limited': '#ff6b6b',
                        'steering-authority-limited': '#ff6b6b',
                        'perception-limited': '#ffa500',
                        'mixed-or-unclear': '#4a90e2',
                    };
                    const clsColor = clsColorMap[cls] || '#4a90e2';
                    const speedFeas = feas.speed_feasibility || {};
                    const auth = feas.steering_authority || {};
                    const deltas = feas.limiter_deltas_entry_mean || {};
                    const speedLimitedPct = speedFeas.speed_limited_pct || 0;

                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; margin-top: 1rem; border-left: 3px solid #00bcd4;">';
                    html += '<strong style="color: #00e5ff;">Curve Entry Feasibility (Full Story)</strong><br/>';
                    html += `<div style="margin-top: 0.35rem; color: ${clsColor};">Primary classification: <strong>${cls}</strong></div>`;
                    html += `<div style="margin-top: 0.35rem; color: #e0e0e0;">Window: frames ${feas.entry_start_frame ?? '-'}-${feas.entry_end_frame ?? '-'} `
                        + `(${feas.entry_frames ?? 0} frames), curve start ${feas.curve_start_frame ?? '-'} (${feas.curve_start_source || 'unknown'})</div>`;
                    html += `<div style="margin-top: 0.2rem; color: #9ad6db; font-size: 0.9rem;">Distance window target: `
                        + `${fmtOpt(feas.entry_start_distance_target_m, 2)}m + ${fmtOpt(feas.entry_window_distance_target_m, 2)}m `
                        + `(used ${fmtOpt(feas.entry_start_distance_used_m, 2)}m to ${fmtOpt(feas.entry_end_distance_used_m, 2)}m)</div>`;

                    html += '<table style="width: 100%; color: #e0e0e0; margin-top: 0.5rem;">';
                    html += `<tr><td>v²·κ Mean / Max (budget ${fmtOpt(speedFeas.ay_budget, 2)}):</td><td style="text-align: right;">${fmtOpt(speedFeas.ay_mean, 3)} / ${fmtOpt(speedFeas.ay_max, 3)}</td></tr>`;
                    html += `<tr><td>Speed-Limited Entry Frames:</td><td style="text-align: right; color: ${speedLimitedPct >= 30 ? '#ff6b6b' : '#4caf50'};">${fmtOpt(speedLimitedPct, 1)}%</td></tr>`;
                    html += `<tr><td>Steering |pre| / |final| Mean:</td><td style="text-align: right;">${fmtOpt(auth.pre_abs_mean, 3)} / ${fmtOpt(auth.final_abs_mean, 3)}</td></tr>`;
                    html += `<tr><td>Authority Gap Mean:</td><td style="text-align: right;">${fmtOpt(auth.authority_gap_mean, 3)}</td></tr>`;
                    html += `<tr><td>Transfer Ratio Mean:</td><td style="text-align: right;">${fmtOpt(auth.transfer_ratio_mean, 3)}</td></tr>`;
                    html += `<tr><td>Stale Perception During Entry:</td><td style="text-align: right; color: ${(feas.stale_perception_pct || 0) >= 50 ? '#ff6b6b' : '#4caf50'};">${fmtOpt(feas.stale_perception_pct, 1)}%</td></tr>`;
                    html += `<tr><td>Limiter Δ Mean (rate/jerk/clip/smooth):</td><td style="text-align: right;">${fmtOpt(deltas.rate, 3)} / ${fmtOpt(deltas.jerk, 3)} / ${fmtOpt(deltas.hard_clip, 3)} / ${fmtOpt(deltas.smoothing, 3)}</td></tr>`;
                    html += '</table>';
                    html += '</div>';
                }

                // Control quality breakdown (penalty drivers)
                const breakdown = [];
                if (ctrl.steering_correlation !== null && ctrl.steering_correlation !== undefined) {
                    if (ctrl.steering_correlation < 0.3) {
                        breakdown.push(`Low steering correlation (corr=${ctrl.steering_correlation.toFixed(3)} < 0.30)`);
                    } else if (ctrl.steering_correlation < 0.5) {
                        breakdown.push(`Moderate steering correlation (corr=${ctrl.steering_correlation.toFixed(3)})`);
                    }
                }
                if (ctrl.lateral_error && ctrl.lateral_error.rmse !== null && ctrl.lateral_error.rmse !== undefined) {
                    if (ctrl.lateral_error.rmse > 0.4) {
                        breakdown.push(`High lateral RMSE (${ctrl.lateral_error.rmse.toFixed(3)}m > 0.40m)`);
                    } else if (ctrl.lateral_error.rmse > 0.2) {
                        breakdown.push(`Moderate lateral RMSE (${ctrl.lateral_error.rmse.toFixed(3)}m)`);
                    }
                }
                if (ctrl.steering_stats && ctrl.steering_stats.mean_abs !== null && ctrl.steering_stats.mean_abs !== undefined) {
                    if (ctrl.steering_stats.mean_abs < 0.05) {
                        breakdown.push(`Very small mean steering (${ctrl.steering_stats.mean_abs.toFixed(3)})`);
                    } else if (ctrl.steering_stats.mean_abs < 0.1) {
                        breakdown.push(`Small mean steering (${ctrl.steering_stats.mean_abs.toFixed(3)})`);
                    }
                }
                if (ctrl.steering_stats && ctrl.steering_stats.mean_change !== null && ctrl.steering_stats.mean_change !== undefined) {
                    if (ctrl.steering_stats.mean_change < 0.01) {
                        breakdown.push(`Steering barely changes (mean Δ=${ctrl.steering_stats.mean_change.toFixed(3)})`);
                    }
                }
                html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; margin-top: 1rem; border-left: 3px solid #4a90e2;">';
                html += '<strong style="color: #4a90e2;">Control Quality Breakdown</strong><br/>';
                if (breakdown.length > 0) {
                    html += '<ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #e0e0e0;">';
                    breakdown.forEach(item => {
                        html += `<li>${item}</li>`;
                    });
                    html += '</ul>';
                } else {
                    html += '<div style="color: #4caf50; margin-top: 0.5rem;">No control penalties detected.</div>';
                }
                html += '</div>';
                
                if (ctrl.issues && ctrl.issues.length > 0) {
                    html += '<div style="background: #3a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #ff6b6b; margin-top: 1rem;">';
                    html += '<strong style="color: #ff6b6b;">Issues:</strong><ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #ff6b6b;">';
                    ctrl.issues.forEach(issue => {
                        html += `<li>${issue}</li>`;
                    });
                    html += '</ul></div>';
                }
                
                if (ctrl.warnings && ctrl.warnings.length > 0) {
                    html += '<div style="background: #3a2a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #ffa500; margin-top: 1rem;">';
                    html += '<strong style="color: #ffa500;">Warnings:</strong><ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #ffa500;">';
                    ctrl.warnings.forEach(warning => {
                        html += `<li>${warning}</li>`;
                    });
                    html += '</ul></div>';
                }

                if (ctrl.lateral_error_hotspots && ctrl.lateral_error_hotspots.length > 0) {
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #4a90e2; margin-top: 1rem;">';
                    html += '<strong style="color: #4a90e2;">Lateral Error Hotspots</strong><br/>';
                    html += '<div style="color: #888; font-size: 0.85rem; margin-bottom: 0.5rem;">Top frames by |lateral error| with context.</div>';
                    html += '<div style="display: flex; flex-direction: column; gap: 0.5rem;">';
                    ctrl.lateral_error_hotspots.forEach((spot) => {
                        const mismatch = spot.sign_mismatch ? '⚠️ sign mismatch' : '';
                        const seg = spot.segment ? spot.segment : 'unknown';
                        const straight = spot.is_straight === null || spot.is_straight === undefined
                            ? 'n/a'
                            : (spot.is_straight ? 'yes' : 'no');
                        html += '<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: #2a2a2a; border-radius: 6px;">';
                        html += '<div style="color: #e0e0e0; font-size: 0.9rem;">';
                        html += `<strong>Frame ${spot.frame}</strong> · t=${spot.time.toFixed(2)}s · |err|=${Math.abs(spot.lateral_error).toFixed(3)}m · ${seg} · straight=${straight}<br/>`;
                        html += `ref_x=${spot.ref_x.toFixed(3)} · steer=${spot.steering.toFixed(3)} · heading=${(spot.heading_error * 180 / Math.PI).toFixed(2)}°`;
                        html += ` · curv=${spot.curvature.toFixed(4)} · gt_curv=${spot.gt_curvature.toFixed(4)} ${mismatch}`;
                        html += '</div>';
                        html += `<button style="padding: 0.25rem 0.75rem; background: #4a90e2; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem;" onclick="window.visualizer.jumpToFrame(${spot.frame})">Jump →</button>`;
                        html += '</div>';
                    });
                    html += '</div></div>';
                } else {
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #4a90e2; margin-top: 1rem;">';
                    html += '<strong style="color: #4a90e2;">Lateral Error Hotspots</strong><br/>';
                    html += '<div style="color: #4caf50; font-size: 0.9rem;">No significant hotspots detected.</div>';
                    html += '</div>';
                }
                
                if (ctrl.lateral_error_disagreement_hotspots && ctrl.lateral_error_disagreement_hotspots.length > 0) {
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #4a90e2; margin-top: 1rem;">';
                    html += '<strong style="color: #4a90e2;">GT vs Control Disagreement</strong><br/>';
                    html += '<div style="color: #888; font-size: 0.85rem; margin-bottom: 0.5rem;">Top frames by |control - GT| lateral error.</div>';
                    html += '<div style="display: flex; flex-direction: column; gap: 0.5rem;">';
                    ctrl.lateral_error_disagreement_hotspots.forEach((spot) => {
                        const straight = spot.is_straight === null || spot.is_straight === undefined
                            ? 'n/a'
                            : (spot.is_straight ? 'yes' : 'no');
                        html += '<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: #2a2a2a; border-radius: 6px;">';
                        html += '<div style="color: #e0e0e0; font-size: 0.9rem;">';
                        html += `<strong>Frame ${spot.frame}</strong> · t=${spot.time.toFixed(2)}s · diff=${spot.diff.toFixed(3)}m · ${spot.segment} · straight=${straight}<br/>`;
                        html += `ctrl=${spot.lateral_error.toFixed(3)}m · gt=${spot.gt_error.toFixed(3)}m · steer=${fmtOpt(spot.steering, 3)} · curv=${fmtOpt(spot.curvature, 4)} · gt_curv=${fmtOpt(spot.gt_curvature, 4)}`;
                        html += '</div>';
                        html += `<button style="padding: 0.25rem 0.75rem; background: #4a90e2; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem;" onclick="window.visualizer.jumpToFrame(${spot.frame})">Jump →</button>`;
                        html += '</div>';
                    });
                    html += '</div></div>';
                } else {
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #4a90e2; margin-top: 1rem;">';
                    html += '<strong style="color: #4a90e2;">GT vs Control Disagreement</strong><br/>';
                    html += '<div style="color: #4caf50; font-size: 0.9rem;">No significant disagreements detected.</div>';
                    html += '</div>';
                }

                if (ctrl.gt_perception_hotspots && ctrl.gt_perception_hotspots.length > 0) {
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #9b59b6; margin-top: 1rem;">';
                    html += '<strong style="color: #d2a8ff;">GT vs Perception Lateral Error (Side-by-Side)</strong><br/>';
                    html += '<div style="color: #888; font-size: 0.85rem; margin-bottom: 0.5rem;">Top frames by |perception_error - gt_error|.</div>';
                    html += '<div style="display: flex; flex-direction: column; gap: 0.5rem;">';
                    ctrl.gt_perception_hotspots.forEach((spot) => {
                        const straight = spot.is_straight === null || spot.is_straight === undefined
                            ? 'n/a'
                            : (spot.is_straight ? 'yes' : 'no');
                        html += '<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: #2a2a2a; border-radius: 6px;">';
                        html += '<div style="color: #e0e0e0; font-size: 0.9rem;">';
                        html += `<strong>Frame ${spot.frame}</strong> · t=${spot.time.toFixed(2)}s · diff=${fmtOpt(spot.diff, 3)}m · ${spot.segment} · straight=${straight}<br/>`;
                        html += `gt_err=${fmtOpt(spot.gt_error, 3)}m · perc_err=${fmtOpt(spot.perception_error, 3)}m · ctrl_err=${fmtOpt(spot.control_lateral_error, 3)}m<br/>`;
                        html += `gt_center=${fmtOpt(spot.gt_center_x, 3)} · perc_center=${fmtOpt(spot.perception_center_x, 3)} · curv=${fmtOpt(spot.curvature, 4)} · gt_curv=${fmtOpt(spot.gt_curvature, 4)}`;
                        html += '</div>';
                        html += `<button style="padding: 0.25rem 0.75rem; background: #7b61ff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem;" onclick="window.visualizer.jumpToFrame(${spot.frame})">Jump →</button>`;
                        html += '</div>';
                    });
                    html += '</div></div>';
                } else {
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #9b59b6; margin-top: 1rem;">';
                    html += '<strong style="color: #d2a8ff;">GT vs Perception Lateral Error (Side-by-Side)</strong><br/>';
                    html += '<div style="color: #4caf50; font-size: 0.9rem;">No significant GT-perception disagreement hotspots detected.</div>';
                    html += '</div>';
                }
                
                if (ctrl.accel_hotspots && ctrl.accel_hotspots.length > 0) {
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #4a90e2; margin-top: 1rem;">';
                    html += '<strong style="color: #4a90e2;">Longitudinal Accel Hotspots</strong><br/>';
                    html += '<div style="color: #888; font-size: 0.85rem; margin-bottom: 0.5rem;">Top frames by |accel| with context.</div>';
                    html += '<div style="display: flex; flex-direction: column; gap: 0.5rem;">';
                    ctrl.accel_hotspots.forEach((spot) => {
                        html += '<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: #2a2a2a; border-radius: 6px;">';
                        html += '<div style="color: #e0e0e0; font-size: 0.9rem;">';
                        html += `<strong>Frame ${spot.frame}</strong> · t=${spot.time.toFixed(2)}s · accel=${spot.accel.toFixed(2)} m/s² · jerk=${spot.jerk.toFixed(2)} m/s³ · speed=${spot.speed.toFixed(2)} m/s · ${spot.segment}<br/>`;
                        html += `thr=${fmtOpt(spot.throttle, 2)} · brk=${fmtOpt(spot.brake, 2)} · tgt=${fmtOpt(spot.target_speed, 2)} · steer=${fmtOpt(spot.steering, 3)} · curv=${fmtOpt(spot.curvature, 4)} · gt_curv=${fmtOpt(spot.gt_curvature, 4)}`;
                        html += '</div>';
                        html += `<button style="padding: 0.25rem 0.75rem; background: #4a90e2; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem;" onclick="window.visualizer.jumpToFrame(${spot.frame})">Jump →</button>`;
                        html += '</div>';
                    });
                    html += '</div></div>';
                } else {
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #4a90e2; margin-top: 1rem;">';
                    html += '<strong style="color: #4a90e2;">Longitudinal Accel Hotspots</strong><br/>';
                    html += '<div style="color: #4caf50; font-size: 0.9rem;">No significant hotspots detected.</div>';
                    html += '</div>';
                }
                
                if (ctrl.jerk_hotspots && ctrl.jerk_hotspots.length > 0) {
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #4a90e2; margin-top: 1rem;">';
                    html += '<strong style="color: #4a90e2;">Longitudinal Jerk Hotspots</strong><br/>';
                    html += '<div style="color: #888; font-size: 0.85rem; margin-bottom: 0.5rem;">Top frames by |jerk| with context.</div>';
                    html += '<div style="display: flex; flex-direction: column; gap: 0.5rem;">';
                    ctrl.jerk_hotspots.forEach((spot) => {
                        html += '<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: #2a2a2a; border-radius: 6px;">';
                        html += '<div style="color: #e0e0e0; font-size: 0.9rem;">';
                        html += `<strong>Frame ${spot.frame}</strong> · t=${spot.time.toFixed(2)}s · jerk=${spot.jerk.toFixed(2)} m/s³ · accel=${spot.accel.toFixed(2)} m/s² · speed=${spot.speed.toFixed(2)} m/s · ${spot.segment}<br/>`;
                        html += `thr=${fmtOpt(spot.throttle, 2)} · brk=${fmtOpt(spot.brake, 2)} · tgt=${fmtOpt(spot.target_speed, 2)} · steer=${fmtOpt(spot.steering, 3)} · curv=${fmtOpt(spot.curvature, 4)} · gt_curv=${fmtOpt(spot.gt_curvature, 4)}`;
                        html += '</div>';
                        html += `<button style="padding: 0.25rem 0.75rem; background: #4a90e2; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem;" onclick="window.visualizer.jumpToFrame(${spot.frame})">Jump →</button>`;
                        html += '</div>';
                    });
                    html += '</div></div>';
                } else {
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #4a90e2; margin-top: 1rem;">';
                    html += '<strong style="color: #4a90e2;">Longitudinal Jerk Hotspots</strong><br/>';
                    html += '<div style="color: #4caf50; font-size: 0.9rem;">No significant hotspots detected.</div>';
                    html += '</div>';
                }

                if (ctrl.steering_limiter_hotspots && ctrl.steering_limiter_hotspots.length > 0) {
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #7b61ff; margin-top: 1rem;">';
                    html += '<strong style="color: #b39bff;">Steering Limiter Hotspots</strong><br/>';
                    html += '<div style="color: #888; font-size: 0.85rem; margin-bottom: 0.5rem;">Top frames where command shaping most reduced steering authority.</div>';
                    html += '<div style="display: flex; flex-direction: column; gap: 0.5rem;">';
                    ctrl.steering_limiter_hotspots.forEach((spot) => {
                        html += '<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: #2a2a2a; border-radius: 6px;">';
                        html += '<div style="color: #e0e0e0; font-size: 0.9rem;">';
                        html += `<strong>Frame ${spot.frame}</strong> · t=${spot.time.toFixed(2)}s · dominant=${spot.dominant_stage} · Δ=${spot.dominant_delta.toFixed(4)}<br/>`;
                        html += `rate=${spot.rate_delta.toFixed(4)} · jerk=${spot.jerk_delta.toFixed(4)} · hard=${spot.hard_clip_delta.toFixed(4)} · smooth=${spot.smoothing_delta.toFixed(4)}`;
                        if (spot.lateral_error !== null && spot.lateral_error !== undefined) {
                            html += ` · lat_err=${spot.lateral_error.toFixed(3)}m`;
                        }
                        html += ` · steer=${spot.steering.toFixed(3)} · curv=${spot.curvature.toFixed(4)}`;
                        html += '</div>';
                        html += `<button style="padding: 0.25rem 0.75rem; background: #4a90e2; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem;" onclick="window.visualizer.jumpToFrame(${spot.frame})">Jump →</button>`;
                        html += '</div>';
                    });
                    html += '</div></div>';
                }

                if (ctrl.turn_bias) {
                    const turnBias = ctrl.turn_bias;
                    const align = ctrl.alignment_summary || null;
                    html += '<div style="background: #1a1a1a; padding: 0.75rem; border-radius: 4px; border-left: 3px solid #7b61ff; margin-top: 1rem;">';
                    html += '<strong style="color: #b39bff;">Turn Bias (Road-Frame)</strong><br/>';
                    html += `<div style="color: #888; font-size: 0.85rem; margin-bottom: 0.5rem;">Curve threshold |curv| ≥ ${turnBias.curve_threshold.toFixed(4)}.</div>`;

                    const left = turnBias.left_turn || {};
                    const right = turnBias.right_turn || {};
                    html += '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">';
                    html += '<div style="background: #2a2a2a; padding: 0.5rem; border-radius: 6px;">';
                    html += '<strong style="color: #b39bff;">Left Turns</strong><br/>';
                    html += `frames=${left.frames ?? 0} · mean=${fmtOpt(left.mean, 3)}m · p95|=${fmtOpt(left.p95_abs, 3)}m · outside=${fmtOpt(left.outside_rate, 1)}%`;
                    html += '</div>';
                    html += '<div style="background: #2a2a2a; padding: 0.5rem; border-radius: 6px;">';
                    html += '<strong style="color: #b39bff;">Right Turns</strong><br/>';
                    html += `frames=${right.frames ?? 0} · mean=${fmtOpt(right.mean, 3)}m · p95|=${fmtOpt(right.p95_abs, 3)}m · outside=${fmtOpt(right.outside_rate, 1)}%`;
                    html += '</div>';
                    html += '</div>';

                    if (align) {
                        html += '<div style="margin-top: 0.5rem; color: #e0e0e0; font-size: 0.9rem;">';
                        html += `Perception vs GT center: mean=${fmtOpt(align.perception_vs_gt_mean, 3)}m · `;
                        html += `p95|=${fmtOpt(align.perception_vs_gt_p95_abs, 3)}m · rmse=${fmtOpt(align.perception_vs_gt_rmse, 3)}m`;
                        if (align.road_frame_lane_center_error_mean !== undefined && align.road_frame_lane_center_error_mean !== null) {
                            html += '<br/>';
                            html += `Road-frame lane-center error: mean=${fmtOpt(align.road_frame_lane_center_error_mean, 3)}m · `;
                            html += `p95|=${fmtOpt(align.road_frame_lane_center_error_p95_abs, 3)}m · rmse=${fmtOpt(align.road_frame_lane_center_error_rmse, 3)}m`;
                        }
                        if (align.vehicle_frame_lookahead_offset_mean !== undefined && align.vehicle_frame_lookahead_offset_mean !== null) {
                            html += '<br/>';
                            html += `Vehicle-frame lookahead offset: mean=${fmtOpt(align.vehicle_frame_lookahead_offset_mean, 3)}m · `;
                            html += `p95|=${fmtOpt(align.vehicle_frame_lookahead_offset_p95_abs, 3)}m`;
                        }
                        html += '</div>';
                    }

                    const renderTurnList = (title, items) => {
                        if (!items || items.length === 0) {
                            html += `<div style="margin-top: 0.5rem; color: #4caf50; font-size: 0.9rem;">${title}: none</div>`;
                            return;
                        }
                        html += `<div style="margin-top: 0.75rem; color: #e0e0e0; font-size: 0.9rem;"><strong>${title}</strong></div>`;
                        html += '<div style="display: flex; flex-direction: column; gap: 0.5rem;">';
                        items.forEach((spot) => {
                            html += '<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: #2a2a2a; border-radius: 6px;">';
                            html += '<div style="color: #e0e0e0; font-size: 0.9rem;">';
                            const outside = spot.outside ? 'outside' : 'inside';
                            html += `<strong>Frame ${spot.frame}</strong> · t=${spot.time.toFixed(2)}s · offset=${spot.road_offset.toFixed(3)}m · ${outside}<br/>`;
                            html += `curv=${spot.curvature.toFixed(4)} · steer=${fmtOpt(spot.steering, 3)} · hdgΔ=${fmtOpt(spot.heading_delta_deg, 2)}° · `;
                            html += `gt_center=${fmtOpt(spot.gt_center, 3)} · p_center=${fmtOpt(spot.perception_center, 3)}`;
                            if (spot.lane_center_error !== undefined && spot.lane_center_error !== null) {
                                html += ` · lane_err=${fmtOpt(spot.lane_center_error, 3)}`;
                            }
                            html += '</div>';
                            html += `<button style="padding: 0.25rem 0.75rem; background: #7b61ff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem;" onclick="window.visualizer.jumpToFrame(${spot.frame})">Jump →</button>`;
                            html += '</div>';
                        });
                        html += '</div>';
                    };

                    renderTurnList('Top Left-Turn Offsets', turnBias.top_left);
                    renderTurnList('Top Right-Turn Offsets', turnBias.top_right);
                    html += '</div>';
                }
                
                html += '</div>';
            }
            
            html += '</div>';
            diagnosticsContent.innerHTML = html;
            
        } catch (error) {
            console.error('Error loading diagnostics:', error);
            diagnosticsContent.innerHTML = `<p style="color: #ff6b6b; text-align: center; padding: 2rem;">Error loading diagnostics: ${error.message}</p>`;
        }
    }

    async goToFrame(frameIndex) {
        if (frameIndex < 0 || frameIndex >= this.frameCount) return;
        
        this.currentFrameIndex = frameIndex;
        document.getElementById('frame-slider').value = frameIndex;
        document.getElementById('frame-number').textContent = frameIndex;
        const frameDistanceElem = document.getElementById('frame-distance-from-start');
        if (frameDistanceElem) {
            const d = this.distanceFromStartSeries && frameIndex < this.distanceFromStartSeries.length
                ? this.distanceFromStartSeries[frameIndex]
                : null;
            frameDistanceElem.textContent = (d !== null && d !== undefined && Number.isFinite(d))
                ? d.toFixed(2)
                : '-';
        }
        
        try {
            // Store previous frame's perception data before loading new frame (for change calculations)
            if (this.currentFrameData && this.currentFrameData.perception) {
                this.previousPerceptionData = {
                    left_lane_line_x: this.currentFrameData.perception.left_lane_line_x !== undefined ? 
                        this.currentFrameData.perception.left_lane_line_x : 
                        (this.currentFrameData.perception.left_lane_x !== undefined ? this.currentFrameData.perception.left_lane_x : null),
                    right_lane_line_x: this.currentFrameData.perception.right_lane_line_x !== undefined ? 
                        this.currentFrameData.perception.right_lane_line_x : 
                        (this.currentFrameData.perception.right_lane_x !== undefined ? this.currentFrameData.perception.right_lane_x : null)
                };
            } else {
                this.previousPerceptionData = null;
            }
            
            // Load frame data
            this.currentFrameData = await this.dataLoader.loadFrameData(frameIndex);
            const prevFrameData = frameIndex > 0
                ? await this.dataLoader.loadFrameData(frameIndex - 1)
                : null;
            const prevPrevFrameData = frameIndex > 1
                ? await this.dataLoader.loadFrameData(frameIndex - 2)
                : null;
            this.currentLongitudinalMetrics = this.computeLongitudinalMetrics(
                this.currentFrameData,
                prevFrameData,
                prevPrevFrameData
            );
            
            // Load camera images
            const imageDataUrl = await this.dataLoader.loadFrameImage(frameIndex, 'front_center');
            await this.loadImage(imageDataUrl, 'camera-canvas');
            if (this.topdownAvailable) {
                try {
                    const topdownUrl = await this.dataLoader.loadFrameImage(frameIndex, 'top_down');
                    await this.loadImage(topdownUrl, 'topdown-canvas');
                    this.setTopdownAvailability(true);
                    this.updateTopdownOverlay();
                } catch (error) {
                    if (error && error.status === 404) {
                        this.topdownAvailable = false;
                        if (this.currentRecordingMeta) {
                            this.currentRecordingMeta.topdown_available = false;
                            this.updateRecordingMetaBadges();
                        }
                        this.setTopdownAvailability(false);
                    } else {
                        console.error('Error loading top-down image:', error);
                    }
                    this.clearCanvas('topdown-canvas');
                    this.currentTopdownImage = null;
                }
            }
            
            // Update displays
            this.updateDataPanel();
            this.updateOverlays();
            this.updateDebugOverlays();
            this.updateTopdownOverlay();
            this.updateChartCursor();
            this.updateQuickChartValuesTable();
            
            // Update timestamp
            if (this.currentFrameData.camera) {
                document.getElementById('frame-time').textContent = 
                    this.currentFrameData.camera.timestamp.toFixed(2);
            }
        } catch (error) {
            console.error('Error loading frame:', error);
        }
    }

    async loadImage(imageDataUrl, canvasId = 'camera-canvas') {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.getElementById(canvasId);
                if (!canvas) {
                    resolve();
                    return;
                }
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                this.currentImage = img;
                if (canvasId === 'topdown-canvas') {
                    this.currentTopdownImage = img;
                }
                resolve();
            };
            img.onerror = reject;
            img.src = imageDataUrl;
        });
    }

    clearCanvas(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            return;
        }
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    setTopdownAvailability(isAvailable) {
        const topdownContainer = document.getElementById('topdown-container');
        const quickPanelTitle = document.querySelector('.quick-debug-panel h3');
        if (!topdownContainer) {
            return;
        }
        if (isAvailable) {
            topdownContainer.style.display = '';
            if (quickPanelTitle) {
                quickPanelTitle.textContent = 'Quick Chart Focus';
            }
        } else {
            topdownContainer.style.display = 'none';
            if (quickPanelTitle) {
                const recType = this.currentRecordingMeta?.recording_type || 'unknown';
                quickPanelTitle.textContent = `Quick Chart Focus (Top-Down unavailable: ${this.humanizeRecordingType(recType)})`;
            }
        }
    }

    updateDataPanel() {
        if (!this.currentFrameData) return;
        
        // Update all tabs (including new "All Data" tab)
        this.updatePerceptionData();
        this.updateTrajectoryData();
        this.updateProjectionData();
        this.updateControlData();
        this.updateVehicleData();
        this.updateGroundTruthData();
    }

    quatMultiply(a, b) {
        return [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
        ];
    }

    quatRotateVec(v, q) {
        const vq = [v.x, v.y, v.z, 0];
        const qConj = [-q[0], -q[1], -q[2], q[3]];
        const res = this.quatMultiply(this.quatMultiply(q, vq), qConj);
        return { x: res[0], y: res[1], z: res[2] };
    }

    projectTrajectoryToImage(trajPoints) {
        if (!this.currentFrameData || !this.currentFrameData.vehicle) {
            return null;
        }
        const vehicle = this.currentFrameData.vehicle;
        if (!vehicle.rotation || !vehicle.position) {
            return null;
        }
        if (vehicle.camera_pos_x === null || vehicle.camera_forward_x === null) {
            return null;
        }

        const q = vehicle.rotation;
        if (!Array.isArray(q) || q.length < 4) {
            return null;
        }
        const camPosWorld = {
            x: vehicle.camera_pos_x,
            y: vehicle.camera_pos_y,
            z: vehicle.camera_pos_z
        };
        const vehPosWorld = {
            x: vehicle.position[0],
            y: vehicle.position[1],
            z: vehicle.position[2]
        };
        const camForwardWorld = {
            x: vehicle.camera_forward_x,
            y: vehicle.camera_forward_y,
            z: vehicle.camera_forward_z
        };

        // Build camera basis in WORLD frame and project world points into camera frame.
        // Trajectory points are in vehicle frame: x=right, y=forward on road plane.
        const forwardNorm = Math.hypot(camForwardWorld.x, camForwardWorld.y, camForwardWorld.z) || 1.0;
        const forward = {
            x: camForwardWorld.x / forwardNorm,
            y: camForwardWorld.y / forwardNorm,
            z: camForwardWorld.z / forwardNorm
        };

        // Unity world up is +Y. Build right-handed camera basis in world frame.
        const worldUp = { x: 0, y: 1, z: 0 };
        // Right = up x forward (not forward x up), otherwise X can be mirrored.
        const right = {
            x: worldUp.y * forward.z - worldUp.z * forward.y,
            y: worldUp.z * forward.x - worldUp.x * forward.z,
            z: worldUp.x * forward.y - worldUp.y * forward.x
        };
        const rightNorm = Math.hypot(right.x, right.y, right.z) || 1.0;
        right.x /= rightNorm;
        right.y /= rightNorm;
        right.z /= rightNorm;
        // Up = forward x right to keep basis orthonormal.
        const up = {
            x: forward.y * right.z - forward.z * right.y,
            y: forward.z * right.x - forward.x * right.z,
            z: forward.x * right.y - forward.y * right.x
        };

        const width = this.overlayRenderer.imageWidth;
        const height = this.overlayRenderer.imageHeight;
        const cx = width / 2.0;
        const cy = height / 2.0;
        const hfov = vehicle.camera_horizontal_fov && vehicle.camera_horizontal_fov > 0
            ? vehicle.camera_horizontal_fov
            : this.overlayRenderer.cameraFov;
        const vfov = vehicle.camera_field_of_view && vehicle.camera_field_of_view > 0
            ? vehicle.camera_field_of_view
            : (2.0 * Math.atan(Math.tan((hfov * Math.PI) / 360.0) * (height / width)) * (180 / Math.PI));

        const fx = (width / 2.0) / Math.tan((hfov * Math.PI) / 360.0);
        const fy = (height / 2.0) / Math.tan((vfov * Math.PI) / 360.0);

        const vehForwardRaw = this.quatRotateVec({ x: 0, y: 0, z: 1 }, q);
        const vehRightRaw = this.quatRotateVec({ x: 1, y: 0, z: 0 }, q);
        const forward2DNorm = Math.hypot(vehForwardRaw.x, vehForwardRaw.z) || 1.0;
        const vehForward2D = {
            x: vehForwardRaw.x / forward2DNorm,
            y: 0.0,
            z: vehForwardRaw.z / forward2DNorm
        };
        let vehRight2DNorm = Math.hypot(vehRightRaw.x, vehRightRaw.z);
        let vehRight2D;
        if (vehRight2DNorm < 1e-6) {
            // Fallback right vector derived from flattened forward vector.
            vehRight2D = {
                x: vehForward2D.z,
                y: 0.0,
                z: -vehForward2D.x
            };
            vehRight2DNorm = Math.hypot(vehRight2D.x, vehRight2D.z) || 1.0;
            vehRight2D.x /= vehRight2DNorm;
            vehRight2D.z /= vehRight2DNorm;
        } else {
            vehRight2D = {
                x: vehRightRaw.x / vehRight2DNorm,
                y: 0.0,
                z: vehRightRaw.z / vehRight2DNorm
            };
        }
        const groundY = Number.isFinite(Number(vehicle.road_center_at_car_y))
            ? Number(vehicle.road_center_at_car_y)
            : (Number.isFinite(Number(vehPosWorld.y)) ? Number(vehPosWorld.y) : 0.0);
        const localToWorldGround = (xLocal, yLocal) => {
            const yForward = Number.isFinite(Number(yLocal)) ? Number(yLocal) : 0.0;
            const blendDen = Math.max(1e-3, Number(this.projectionNearFieldBlendDistanceMeters) || 10.0);
            const nearWeightRaw = Math.max(0.0, Math.min(1.0, 1.0 - (Math.max(0.0, yForward) / blendDen)));
            const nearWeight = this.projectionNearFieldBlendEnabled ? nearWeightRaw : 0.0;
            const blendedGroundY = groundY + ((Number(this.projectionNearFieldGroundYOffsetMeters) || 0.0) * nearWeight);
            return {
                x: vehPosWorld.x + (vehRight2D.x * xLocal) + (vehForward2D.x * yForward),
                y: blendedGroundY,
                z: vehPosWorld.z + (vehRight2D.z * xLocal) + (vehForward2D.z * yForward)
            };
        };

        const points = [];
        let sanityProjectedRightPx = null;
        let sanityProjectedLeftPx = null;
        let firstVisibleSrcY = null;
        for (const point of trajPoints) {
            if (point.y < 0) {
                continue;
            }
            const localX = Number(point.x);
            const localY = Number(point.y);
            if (!Number.isFinite(localX) || !Number.isFinite(localY)) {
                continue;
            }
            // Keep overlays on road plane: ignore body pitch/roll when mapping local x/y to world.
            const worldPoint = localToWorldGround(localX, localY);
            const p = {
                x: worldPoint.x - camPosWorld.x,
                y: worldPoint.y - camPosWorld.y,
                z: worldPoint.z - camPosWorld.z
            };
            const xCam = p.x * right.x + p.y * right.y + p.z * right.z;
            const yCam = p.x * up.x + p.y * up.y + p.z * up.z;
            const zCam = p.x * forward.x + p.y * forward.y + p.z * forward.z;
            if (zCam <= 0.1) {
                continue;
            }
            const xImg = cx + (xCam / zCam) * fx;
            const yImg = cy - (yCam / zCam) * fy;
            points.push({ x: xImg, y: yImg, srcX: Number(point.x), srcY: Number(point.y) });
            if (firstVisibleSrcY === null) {
                firstVisibleSrcY = Number(point.y);
            }
        }

        // Projection sanity check: at the same forward distance, +x should land right of -x.
        // This catches accidental handedness/sign flips in camera basis.
        const sanityDistance = 8.0;
        const sanityOffset = 1.0;
        const sanityProject = (xLocal, zLocal) => {
            const worldPoint = localToWorldGround(xLocal, zLocal);
            const p = {
                x: worldPoint.x - camPosWorld.x,
                y: worldPoint.y - camPosWorld.y,
                z: worldPoint.z - camPosWorld.z
            };
            const xCam = p.x * right.x + p.y * right.y + p.z * right.z;
            const yCam = p.x * up.x + p.y * up.y + p.z * up.z;
            const zCam = p.x * forward.x + p.y * forward.y + p.z * forward.z;
            if (zCam <= 0.1) return null;
            return { x: cx + (xCam / zCam) * fx, y: cy - (yCam / zCam) * fy };
        };
        const sanityRight = sanityProject(+sanityOffset, sanityDistance);
        const sanityLeft = sanityProject(-sanityOffset, sanityDistance);
        if (sanityRight && sanityLeft) {
            sanityProjectedRightPx = sanityRight.x;
            sanityProjectedLeftPx = sanityLeft.x;
            if (
                this._projectionSanityLoggedFrame !== this.currentFrameIndex &&
                sanityProjectedRightPx <= sanityProjectedLeftPx
            ) {
                console.warn(
                    '[Projection sanity] possible mirrored X in main camera projection',
                    {
                        frameIndex: this.currentFrameIndex,
                        sanityProjectedRightPx,
                        sanityProjectedLeftPx
                    }
                );
                this._projectionSanityLoggedFrame = this.currentFrameIndex;
            }
        }

        points._diag = {
            main_first_visible_src_y_m: firstVisibleSrcY,
            main_mirror_sanity: (
                Number.isFinite(sanityProjectedRightPx) &&
                Number.isFinite(sanityProjectedLeftPx) &&
                sanityProjectedRightPx > sanityProjectedLeftPx
            ) ? 'ok' : 'check',
            main_nearfield_blend: this.projectionNearFieldBlendEnabled ? 'on' : 'off',
            main_nearfield_y_offset_m: Number(this.projectionNearFieldGroundYOffsetMeters),
            main_nearfield_blend_distance_m: Number(this.projectionNearFieldBlendDistanceMeters),
        };

        return points;
    }

    projectWorldPointsToImage(worldPoints) {
        if (!this.currentFrameData || !this.currentFrameData.vehicle || !Array.isArray(worldPoints)) {
            return null;
        }
        const vehicle = this.currentFrameData.vehicle;
        if (!vehicle.position || vehicle.camera_pos_x === null || vehicle.camera_forward_x === null) {
            return null;
        }
        const camPosWorld = {
            x: vehicle.camera_pos_x,
            y: vehicle.camera_pos_y,
            z: vehicle.camera_pos_z
        };
        const camForwardWorld = {
            x: vehicle.camera_forward_x,
            y: vehicle.camera_forward_y,
            z: vehicle.camera_forward_z
        };
        const forwardNorm = Math.hypot(camForwardWorld.x, camForwardWorld.y, camForwardWorld.z) || 1.0;
        const forward = {
            x: camForwardWorld.x / forwardNorm,
            y: camForwardWorld.y / forwardNorm,
            z: camForwardWorld.z / forwardNorm
        };
        const worldUp = { x: 0, y: 1, z: 0 };
        const right = {
            x: worldUp.y * forward.z - worldUp.z * forward.y,
            y: worldUp.z * forward.x - worldUp.x * forward.z,
            z: worldUp.x * forward.y - worldUp.y * forward.x
        };
        const rightNorm = Math.hypot(right.x, right.y, right.z) || 1.0;
        right.x /= rightNorm;
        right.y /= rightNorm;
        right.z /= rightNorm;
        const up = {
            x: forward.y * right.z - forward.z * right.y,
            y: forward.z * right.x - forward.x * right.z,
            z: forward.x * right.y - forward.y * right.x
        };

        const width = this.overlayRenderer.imageWidth;
        const height = this.overlayRenderer.imageHeight;
        const cx = width / 2.0;
        const cy = height / 2.0;
        const hfov = vehicle.camera_horizontal_fov && vehicle.camera_horizontal_fov > 0
            ? vehicle.camera_horizontal_fov
            : this.overlayRenderer.cameraFov;
        const vfov = vehicle.camera_field_of_view && vehicle.camera_field_of_view > 0
            ? vehicle.camera_field_of_view
            : (2.0 * Math.atan(Math.tan((hfov * Math.PI) / 360.0) * (height / width)) * (180 / Math.PI));
        const fx = (width / 2.0) / Math.tan((hfov * Math.PI) / 360.0);
        const fy = (height / 2.0) / Math.tan((vfov * Math.PI) / 360.0);

        const points = [];
        for (const point of worldPoints) {
            const wx = Number(point?.x);
            const wy = Number(point?.y);
            const wz = Number(point?.z);
            if (!Number.isFinite(wx) || !Number.isFinite(wy) || !Number.isFinite(wz)) {
                continue;
            }
            const p = {
                x: wx - camPosWorld.x,
                y: wy - camPosWorld.y,
                z: wz - camPosWorld.z
            };
            const xCam = p.x * right.x + p.y * right.y + p.z * right.z;
            const yCam = p.x * up.x + p.y * up.y + p.z * up.z;
            const zCam = p.x * forward.x + p.y * forward.y + p.z * forward.z;
            if (zCam <= 0.1) {
                points.push(null);
                continue;
            }
            const xImg = cx + (xCam / zCam) * fx;
            const yImg = cy - (yCam / zCam) * fy;
            points.push({ x: xImg, y: yImg });
        }
        return points;
    }

    getDisplayTrajectoryPoints(trajPoints) {
        if (!Array.isArray(trajPoints) || trajPoints.length === 0) return [];
        const plannerOnly = Boolean(document.getElementById('toggle-planner-only-trajectory')?.checked);
        if (!plannerOnly || trajPoints.length < 2) return trajPoints;
        const rp = this.currentFrameData?.trajectory?.reference_point;
        if (!rp || !Number.isFinite(Number(rp.x)) || !Number.isFinite(Number(rp.y))) {
            return trajPoints;
        }
        const p0 = trajPoints[0];
        const p0x = Number(p0?.x);
        const p0y = Number(p0?.y);
        if (!Number.isFinite(p0x) || !Number.isFinite(p0y)) {
            return trajPoints;
        }
        // Drop only if first point is effectively the prepended reference point.
        if (Math.abs(p0x - Number(rp.x)) < 1e-6 && Math.abs(p0y - Number(rp.y)) < 1e-6) {
            return trajPoints.slice(1);
        }
        return trajPoints;
    }

    toForwardMonotonicPath(points) {
        if (!Array.isArray(points) || points.length === 0) return [];
        const validForward = points
            .map((p) => ({ x: Number(p?.x), y: Number(p?.y) }))
            .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && p.y >= 0);
        if (validForward.length === 0) return [];

        let minYIdx = 0;
        let minY = Number.POSITIVE_INFINITY;
        for (let i = 0; i < validForward.length; i++) {
            if (validForward[i].y < minY) {
                minY = validForward[i].y;
                minYIdx = i;
            }
        }

        const out = [];
        let lastY = null;
        for (let i = minYIdx; i < validForward.length; i++) {
            const p = validForward[i];
            if (lastY !== null && p.y + 1e-6 < lastY) {
                continue;
            }
            out.push(p);
            lastY = p.y;
        }
        return out;
    }

    sampleLateralAtForwardDistance(pathPoints, forwardMeters) {
        if (!Array.isArray(pathPoints) || pathPoints.length < 2 || !Number.isFinite(forwardMeters)) {
            return null;
        }
        const pts = pathPoints
            .map((p) => ({ x: Number(p?.x), y: Number(p?.y) }))
            .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
            .sort((a, b) => a.y - b.y);
        if (pts.length < 2) return null;
        const yTarget = Number(forwardMeters);
        if (yTarget < pts[0].y || yTarget > pts[pts.length - 1].y) return null;
        for (let i = 1; i < pts.length; i++) {
            const p0 = pts[i - 1];
            const p1 = pts[i];
            if (p1.y < yTarget) continue;
            const dy = p1.y - p0.y;
            if (Math.abs(dy) < 1e-6) return p1.x;
            const t = (yTarget - p0.y) / dy;
            return p0.x + (p1.x - p0.x) * t;
        }
        return null;
    }

    samplePathShapeAtForwardDistance(pathPoints, forwardMeters, halfWindowMeters = 0.5) {
        if (!Array.isArray(pathPoints) || pathPoints.length < 3 || !Number.isFinite(forwardMeters)) {
            return null;
        }
        const h = Number(halfWindowMeters);
        if (!Number.isFinite(h) || h <= 0) return null;
        const y0 = Number(forwardMeters) - h;
        const y1 = Number(forwardMeters);
        const y2 = Number(forwardMeters) + h;
        const x0 = this.sampleLateralAtForwardDistance(pathPoints, y0);
        const x1 = this.sampleLateralAtForwardDistance(pathPoints, y1);
        const x2 = this.sampleLateralAtForwardDistance(pathPoints, y2);
        if (!Number.isFinite(x0) || !Number.isFinite(x1) || !Number.isFinite(x2)) {
            return null;
        }

        const dxdy = (x2 - x0) / (2.0 * h);
        const d2xdy2 = (x2 - (2.0 * x1) + x0) / (h * h);
        const headingRad = Math.atan2(dxdy, 1.0);
        const denom = Math.pow(1.0 + (dxdy * dxdy), 1.5);
        const curvature = Math.abs(denom) > 1e-8 ? (d2xdy2 / denom) : null;
        return {
            headingRad,
            curvature,
        };
    }

    computeLateralErrorMetrics(plannerPath, oraclePath) {
        const lookaheads = [5, 10, 15];
        const out = {
            source: 'unavailable',
            err5m: null,
            err10m: null,
            err15m: null,
        };
        if (!Array.isArray(plannerPath) || plannerPath.length < 2) {
            return out;
        }

        let baselinePath = null;
        if (Array.isArray(oraclePath) && oraclePath.length >= 2) {
            baselinePath = oraclePath;
            out.source = 'oracle';
        } else {
            const gt = this.currentFrameData?.ground_truth || {};
            const left = Number(gt.left_lane_line_x ?? gt.left_lane_x);
            const right = Number(gt.right_lane_line_x ?? gt.right_lane_x);
            const center = Number(gt.lane_center_x);
            const centerX = Number.isFinite(center)
                ? center
                : (Number.isFinite(left) && Number.isFinite(right) ? (left + right) * 0.5 : null);
            if (centerX !== null) {
                baselinePath = lookaheads.map((y) => ({ x: centerX, y }));
                out.source = 'gt_lane_center_constant_x';
            }
        }
        if (!baselinePath || baselinePath.length < 2) return out;

        const plannerMonotonic = this.toForwardMonotonicPath(plannerPath);
        const baselineMonotonic = this.toForwardMonotonicPath(baselinePath);
        if (plannerMonotonic.length < 2 || baselineMonotonic.length < 2) return out;

        const errors = lookaheads.map((y) => {
            const px = this.sampleLateralAtForwardDistance(plannerMonotonic, y);
            const bx = this.sampleLateralAtForwardDistance(baselineMonotonic, y);
            if (!Number.isFinite(px) || !Number.isFinite(bx)) return null;
            return px - bx;
        });
        [out.err5m, out.err10m, out.err15m] = errors;
        return out;
    }

    computeTurnStrengthMismatchMetrics(plannerPath, oraclePath) {
        const out = {
            source: 'unavailable',
            headingDelta5mDeg: null,
            headingDelta10mDeg: null,
            headingDelta15mDeg: null,
            curvatureRatio5m: null,
            curvatureRatio10m: null,
            curvatureRatio15m: null,
        };
        if (!Array.isArray(plannerPath) || plannerPath.length < 3 || !Array.isArray(oraclePath) || oraclePath.length < 3) {
            return out;
        }
        const lookaheads = [5, 10, 15];
        const plannerMonotonic = this.toForwardMonotonicPath(plannerPath);
        const oracleMonotonic = this.toForwardMonotonicPath(oraclePath);
        if (plannerMonotonic.length < 3 || oracleMonotonic.length < 3) return out;

        const headingDeltas = [];
        const curvatureRatios = [];
        const wrapAngleRad = (a) => {
            let out = Number(a);
            if (!Number.isFinite(out)) return null;
            while (out > Math.PI) out -= (2.0 * Math.PI);
            while (out < -Math.PI) out += (2.0 * Math.PI);
            return out;
        };
        for (const y of lookaheads) {
            const plannerShape = this.samplePathShapeAtForwardDistance(plannerMonotonic, y);
            const oracleShape = this.samplePathShapeAtForwardDistance(oracleMonotonic, y);
            if (!plannerShape || !oracleShape) {
                headingDeltas.push(null);
                curvatureRatios.push(null);
                continue;
            }
            const dhWrapped = wrapAngleRad(Number(plannerShape.headingRad) - Number(oracleShape.headingRad));
            const dh = dhWrapped === null ? null : (dhWrapped * (180 / Math.PI));
            headingDeltas.push(Number.isFinite(dh) ? dh : null);

            const kp = Number(plannerShape.curvature);
            const ko = Number(oracleShape.curvature);
            if (Number.isFinite(kp) && Number.isFinite(ko) && Math.abs(ko) >= 1e-4) {
                curvatureRatios.push(kp / ko);
            } else {
                curvatureRatios.push(null);
            }
        }
        [out.headingDelta5mDeg, out.headingDelta10mDeg, out.headingDelta15mDeg] = headingDeltas;
        [out.curvatureRatio5m, out.curvatureRatio10m, out.curvatureRatio15m] = curvatureRatios;
        out.source = 'planner_vs_oracle';
        return out;
    }

    computeTrajectorySuppressionWaterfall(plannerPath, oraclePath, turnMismatch) {
        const out = {
            source: 'proxy_from_current_frame',
            headingZeroGate: null,
            smallHeadingGate: null,
            multiLookaheadActive: null,
            xClipCount: null,
            heavyXClipping: null,
            preclipXAbsMax: null,
            preclipXAbsP95: null,
            frontFrameIdDelta: null,
            frontUnityDtMs: null,
            overlaySnapRisk: null,
            controlCurvVsOracleRatio10m: null,
            underTurnFlag10m: null,
            syncOverallStatus: null,
            syncTrajStatus: null,
            syncTrajReason: null,
            syncControlStatus: null,
            syncControlReason: null,
            syncWindowMs: null,
            syncDtCamTrajMs: null,
            syncDtCamControlMs: null,
            syncDtCamVehicleMs: null,
            frontTsReused: null,
            frontTsNonMonotonic: null,
            frontIdReused: null,
            frontNegativeDelta: null,
            frontClockJump: null,
            topTsReused: null,
            topTsNonMonotonic: null,
            topIdReused: null,
            topNegativeDelta: null,
            topClockJump: null,
            contractMisalignedRisk: null,
            cadenceRisk: null,
            cadencePolicy: null,
            cadenceFrameDeltaRiskThreshold: null,
            cadenceUnityDtRiskThresholdMs: null,
        };
        const p = this.currentFrameData?.perception || {};
        const t = this.currentFrameData?.trajectory || {};
        const c = this.currentFrameData?.control || {};
        const v = this.currentFrameData?.vehicle || {};

        const coeffs = Array.isArray(p?.lane_line_coefficients) ? p.lane_line_coefficients : null;
        if (coeffs && coeffs.length >= 2) {
            const leftA = Number(Array.isArray(coeffs[0]) ? coeffs[0][0] : null);
            const rightA = Number(Array.isArray(coeffs[1]) ? coeffs[1][0] : null);
            if (Number.isFinite(leftA) && Number.isFinite(rightA)) {
                const centerA = 0.5 * (leftA + rightA);
                out.headingZeroGate = Math.abs(centerA) < 0.01;
            }
        }

        const headingRad = Number(t?.reference_point?.heading ?? t?.reference_point_heading);
        if (Number.isFinite(headingRad)) {
            out.smallHeadingGate = Math.abs(headingRad) < (Math.PI / 180.0);
        }

        const method = String(t?.reference_point_method ?? '');
        if (method) {
            out.multiLookaheadActive = (method === 'multi_lookahead_heading_blend');
        }
        const xClipCount = Number(t?.diag_x_clip_count);
        if (Number.isFinite(xClipCount)) {
            out.xClipCount = xClipCount;
            out.heavyXClipping = xClipCount >= 8.0;
        }
        const preclipXAbsMax = Number(t?.diag_preclip_x_abs_max);
        if (Number.isFinite(preclipXAbsMax)) {
            out.preclipXAbsMax = preclipXAbsMax;
        }
        const preclipXAbsP95 = Number(t?.diag_preclip_x_abs_p95);
        if (Number.isFinite(preclipXAbsP95)) {
            out.preclipXAbsP95 = preclipXAbsP95;
        }
        const frontFrameIdDelta = Number(v?.stream_front_frame_id_delta);
        if (Number.isFinite(frontFrameIdDelta)) {
            out.frontFrameIdDelta = frontFrameIdDelta;
        }
        const frontUnityDtMs = Number(v?.stream_front_unity_dt_ms);
        if (Number.isFinite(frontUnityDtMs)) {
            out.frontUnityDtMs = frontUnityDtMs;
        }
        const sync = this.currentFrameData?.sync || {};
        const syncOverallStatus = String(sync?.overall_alignment_status || '');
        const syncTrajStatus = String(sync?.trajectory_alignment_status || '');
        const syncControlStatus = String(sync?.control_alignment_status || '');
        out.syncOverallStatus = syncOverallStatus || null;
        out.syncTrajStatus = syncTrajStatus || null;
        out.syncTrajReason = String(sync?.trajectory_alignment_reason || '') || null;
        out.syncControlStatus = syncControlStatus || null;
        out.syncControlReason = String(sync?.control_alignment_reason || '') || null;
        out.syncWindowMs = Number(sync?.alignment_window_ms);
        out.syncDtCamTrajMs = Number(sync?.dt_cam_traj_ms);
        out.syncDtCamControlMs = Number(sync?.dt_cam_control_ms);
        out.syncDtCamVehicleMs = Number(sync?.dt_cam_vehicle_ms);
        const syncPolicy = String(this.currentRecordingMeta?.metadata?.stream_sync_policy || '').toLowerCase();
        const frameDeltaRiskThreshold = (syncPolicy === 'latest') ? 3.0 : 2.0;
        const unityDtRiskThresholdMs = 20.0;
        out.cadencePolicy = syncPolicy || 'unknown';
        out.cadenceFrameDeltaRiskThreshold = frameDeltaRiskThreshold;
        out.cadenceUnityDtRiskThresholdMs = unityDtRiskThresholdMs;
        const hasFrameDeltaRisk = Number.isFinite(frontFrameIdDelta) && frontFrameIdDelta >= frameDeltaRiskThreshold;
        const hasUnityDtRisk = Number.isFinite(frontUnityDtMs) && Math.abs(frontUnityDtMs) >= unityDtRiskThresholdMs;
        const hasSyncAlignRisk = (
            syncTrajStatus.toLowerCase() === 'misaligned' ||
            syncTrajStatus.toLowerCase() === 'missing' ||
            syncControlStatus.toLowerCase() === 'misaligned' ||
            syncControlStatus.toLowerCase() === 'missing'
        );
        out.contractMisalignedRisk = hasSyncAlignRisk;
        out.cadenceRisk = hasFrameDeltaRisk || hasUnityDtRisk;
        out.overlaySnapRisk = hasSyncAlignRisk;

        out.frontTsReused = Number(v?.stream_front_timestamp_reused);
        out.frontTsNonMonotonic = Number(v?.stream_front_timestamp_non_monotonic);
        out.frontIdReused = Number(v?.stream_front_frame_id_reused);
        out.frontNegativeDelta = Number(v?.stream_front_negative_frame_delta);
        out.frontClockJump = Number(v?.stream_front_clock_jump);
        out.topTsReused = Number(v?.stream_topdown_timestamp_reused);
        out.topTsNonMonotonic = Number(v?.stream_topdown_timestamp_non_monotonic);
        out.topIdReused = Number(v?.stream_topdown_frame_id_reused);
        out.topNegativeDelta = Number(v?.stream_topdown_negative_frame_delta);
        out.topClockJump = Number(v?.stream_topdown_clock_jump);

        const controlCurv = Number(c?.path_curvature_input);
        const oracleMonotonic = this.toForwardMonotonicPath(Array.isArray(oraclePath) ? oraclePath : []);
        const oracleShape10 = this.samplePathShapeAtForwardDistance(oracleMonotonic, 10.0);
        if (Number.isFinite(controlCurv) && oracleShape10 && Number.isFinite(Number(oracleShape10.curvature))) {
            const oracleCurv10 = Number(oracleShape10.curvature);
            if (Math.abs(oracleCurv10) >= 1e-4) {
                out.controlCurvVsOracleRatio10m = Math.abs(controlCurv) / Math.abs(oracleCurv10);
            }
        }

        const ratio10 = Number(turnMismatch?.curvatureRatio10m);
        if (Number.isFinite(ratio10)) {
            out.underTurnFlag10m = ratio10 < 0.6;
        }
        return out;
    }

    getRightLaneFiducialDiagnostics() {
        const out = {
            source: 'unavailable',
            err5m: null,
            err10m: null,
            err15m: null,
            meanErr: null,
            maxErr: null,
            pairs: [],
        };
        const v = this.currentFrameData?.vehicle || {};
        const vehicleWorldPoints = Array.isArray(v.right_lane_fiducials_world_points)
            ? v.right_lane_fiducials_world_points
            : [];
        const vehicleTruePoints = Array.isArray(v.right_lane_fiducials_vehicle_true_points)
            ? v.right_lane_fiducials_vehicle_true_points
            : [];
        const vehicleMonotonicPoints = Array.isArray(v.right_lane_fiducials_vehicle_monotonic_points)
            ? v.right_lane_fiducials_vehicle_monotonic_points
            : [];
        const vehicleLegacyPoints = Array.isArray(v.right_lane_fiducials_vehicle_points)
            ? v.right_lane_fiducials_vehicle_points
            : [];
        const vehiclePoints = vehicleTruePoints.length > 0
            ? vehicleTruePoints
            : (vehicleLegacyPoints.length > 0 ? vehicleLegacyPoints : vehicleMonotonicPoints);
        const screenPoints = Array.isArray(v.right_lane_fiducials_screen_points)
            ? v.right_lane_fiducials_screen_points
            : [];
        const spacing = Number(v.right_lane_fiducials_spacing_meters);
        if (vehiclePoints.length < 2 || screenPoints.length < 2 || !Number.isFinite(spacing) || spacing <= 0) {
            return out;
        }

        let sum = 0;
        let count = 0;
        let maxErr = null;
        for (let i = 0; i < Math.min(vehiclePoints.length, screenPoints.length); i++) {
            const vp = vehiclePoints[i];
            const sp = screenPoints[i];
            let proj = null;
            if (vehicleWorldPoints.length > i) {
                const worldProjected = this.projectWorldPointsToImage([vehicleWorldPoints[i]]);
                proj = (worldProjected && worldProjected.length > 0) ? worldProjected[0] : null;
            } else {
                const projected = this.projectTrajectoryToImage([vp]);
                proj = (projected && projected.length > 0) ? projected[0] : null;
            }
            const validTruth = Boolean(sp && (sp.valid ?? true) && Number.isFinite(Number(sp.x)) && Number.isFinite(Number(sp.y)) && Number(sp.x) >= 0 && Number(sp.y) >= 0);
            const validProj = Boolean(proj && Number.isFinite(Number(proj.x)) && Number.isFinite(Number(proj.y)));
            const distance = i * spacing;
            let errPx = null;
            if (validTruth && validProj) {
                errPx = Math.hypot(Number(proj.x) - Number(sp.x), Number(proj.y) - Number(sp.y));
                sum += errPx;
                count += 1;
                maxErr = (maxErr === null) ? errPx : Math.max(maxErr, errPx);
            }
            out.pairs.push({ distance, vehicle: vp, truth: sp, projected: proj, errPx });
        }
        const vehicleSource = vehicleWorldPoints.length > 0
            ? 'world_xyz'
            : (vehicleTruePoints.length > 0
                ? 'vehicle_true_xy'
                : (vehicleLegacyPoints.length > 0 ? 'vehicle_xy_legacy' : 'vehicle_monotonic_xy'));
        out.source = `unity_world_to_screen + ${vehicleSource} (${count}/${out.pairs.length} valid)`;
        out.meanErr = count > 0 ? (sum / count) : null;
        out.maxErr = maxErr;

        const readAt = (targetM) => {
            const idx = Math.round(targetM / spacing);
            if (idx < 0 || idx >= out.pairs.length) return null;
            return out.pairs[idx].errPx;
        };
        out.err5m = readAt(5);
        out.err10m = readAt(10);
        out.err15m = readAt(15);
        return out;
    }

    drawRightLaneFiducialsOverlay(fidDiag) {
        if (!fidDiag || !Array.isArray(fidDiag.pairs) || fidDiag.pairs.length === 0) return;
        const ctx = this.overlayRenderer.ctx;
        ctx.save();
        for (const p of fidDiag.pairs) {
            const tx = Number(p?.truth?.x);
            const ty = Number(p?.truth?.y);
            const tValid = Boolean(p?.truth?.valid) && Number.isFinite(tx) && Number.isFinite(ty) && tx >= 0 && ty >= 0;
            const px = Number(p?.projected?.x);
            const py = Number(p?.projected?.y);
            const pValid = Number.isFinite(px) && Number.isFinite(py);

            if (tValid) {
                ctx.strokeStyle = '#00ffff';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(tx - 6, ty);
                ctx.lineTo(tx + 6, ty);
                ctx.moveTo(tx, ty - 6);
                ctx.lineTo(tx, ty + 6);
                ctx.stroke();
            }
            if (pValid) {
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(px, py, 4, 0, Math.PI * 2);
                ctx.stroke();
            }
            if (tValid && pValid) {
                ctx.strokeStyle = 'rgba(255,0,0,0.5)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(tx, ty);
                ctx.lineTo(px, py);
                ctx.stroke();
            }
        }
        ctx.restore();
    }

    updateTopdownOverlay() {
        const canvas = document.getElementById('topdown-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (this.currentTopdownImage) {
            ctx.drawImage(this.currentTopdownImage, 0, 0);
        }
        if (!this.currentFrameData) return;
        const showTrajectory = document.getElementById('toggle-trajectory')?.checked;
        const showOracle = document.getElementById('toggle-oracle-trajectory')?.checked;
        const showDistanceScale = document.getElementById('toggle-distance-scale')?.checked;
        if (!showTrajectory && !showOracle && !showDistanceScale) return;
        const rawTrajectory = showTrajectory
            ? (this.currentFrameData.trajectory?.trajectory_points || [])
            : [];
        const trajectory = this.getDisplayTrajectoryPoints(rawTrajectory);
        const oracleTrajectory = this.currentFrameData.trajectory?.oracle_points || [];
        if (!trajectory.length && !oracleTrajectory.length && !showDistanceScale) return;

        // #region agent log
        if (this._agentTopdownLastLoggedFrame !== this.currentFrameIndex) {
            fetch('http://127.0.0.1:7244/ingest/4b1a3fa9-dffd-42fc-a8f8-c8a4d88904be',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({runId:'pre-fix-topdown-v1',hypothesisId:'H1',location:'visualizer.js:updateTopdownOverlay:entry',message:'Topdown trajectory render entry',data:{frameIndex:this.currentFrameIndex,trajectoryCount:Array.isArray(trajectory)?trajectory.length:0,rawTrajectoryCount:Array.isArray(rawTrajectory)?rawTrajectory.length:0,topdownAvailable:this.topdownAvailable,recording:this.currentRecording||null},timestamp:Date.now()})}).catch(()=>{});
        }
        // #endregion

        let nonFiniteCount = 0;
        let negativeYCount = 0;
        let monotonicBreaks = 0;
        let prevY = null;
        const samplePoints = [];
        for (let i = 0; i < trajectory.length; i++) {
            const p = trajectory[i];
            const x = Number(p?.x);
            const y = Number(p?.y);
            if (!Number.isFinite(x) || !Number.isFinite(y)) {
                nonFiniteCount += 1;
                continue;
            }
            if (y < 0) {
                negativeYCount += 1;
            }
            if (prevY !== null && y + 1e-6 < prevY) {
                monotonicBreaks += 1;
            }
            prevY = y;
            if (samplePoints.length < 8) {
                samplePoints.push({ i, x, y });
            }
        }

        // #region agent log
        if (this._agentTopdownLastLoggedFrame !== this.currentFrameIndex) {
            fetch('http://127.0.0.1:7244/ingest/4b1a3fa9-dffd-42fc-a8f8-c8a4d88904be',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({runId:'pre-fix-topdown-v1',hypothesisId:'H2',location:'visualizer.js:updateTopdownOverlay:raw-shape',message:'Raw trajectory shape metrics',data:{frameIndex:this.currentFrameIndex,nonFiniteCount,negativeYCount,monotonicBreaks,samplePoints},timestamp:Date.now()})}).catch(()=>{});
        }
        // #endregion

        // Normalize top-down render path to the forward monotonic segment.
        // Recorded trajectory can contain a prefixed far lookahead point before near points.
        const validForwardPoints = trajectory.filter((p) =>
            Number.isFinite(Number(p?.x)) &&
            Number.isFinite(Number(p?.y)) &&
            Number(p.y) >= 0
        );
        const hasPlannerPath = validForwardPoints.length > 0;
        if (!hasPlannerPath && !oracleTrajectory.length && !showDistanceScale) return;
        let minYIdx = 0;
        let minY = Number.POSITIVE_INFINITY;
        for (let i = 0; i < validForwardPoints.length; i++) {
            const y = Number(validForwardPoints[i].y);
            if (y < minY) {
                minY = y;
                minYIdx = i;
            }
        }
        const renderPoints = [];
        let renderBreaks = 0;
        let lastYForRender = null;
        for (let i = minYIdx; i < validForwardPoints.length; i++) {
            const pt = validForwardPoints[i];
            const y = Number(pt.y);
            if (lastYForRender !== null && y + 1e-6 < lastYForRender) {
                renderBreaks += 1;
                continue;
            }
            renderPoints.push(pt);
            lastYForRender = y;
        }

        const displayRenderPoints = this.getTopdownDisplayRenderPoints(renderPoints);

        // #region agent log
        if (this._agentTopdownLastLoggedFrame !== this.currentFrameIndex) {
            fetch('http://127.0.0.1:7244/ingest/4b1a3fa9-dffd-42fc-a8f8-c8a4d88904be',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({runId:'post-fix-topdown-v2',hypothesisId:'H5',location:'visualizer.js:updateTopdownOverlay:normalized-shape',message:'Normalized trajectory path metrics',data:{frameIndex:this.currentFrameIndex,rawCount:rawTrajectory.length,displayCount:trajectory.length,validForwardCount:validForwardPoints.length,minYIdx,minY,normalizedCount:renderPoints.length,renderBreaks},timestamp:Date.now()})}).catch(()=>{});
        }
        // #endregion

        const toggleEl = document.getElementById('toggle-topdown-calibrated-projection');
        const useCalibratedTopdownProjection = Boolean(
            (toggleEl && toggleEl.checked) || (!this.topdownProjectionToggleTouched && this.topdownCalibratedProjectionReady)
        );
        const topdownOrthoSize = Number(this.currentFrameData?.vehicle?.topdown_camera_orthographic_size);
        const calibratedOrthoValid = Number.isFinite(topdownOrthoSize) && topdownOrthoSize > 0.01;
        const activeOrthoHalfSize = (
            useCalibratedTopdownProjection && calibratedOrthoValid
                ? topdownOrthoSize
                : this.topdownOrthoHalfSize
        );
        const pixelsPerMeter = canvas.height / (activeOrthoHalfSize * 2.0);
        const cx = canvas.width / 2.0;
        const cy = canvas.height / 2.0;

        // #region agent log
        if (this._agentTopdownLastLoggedFrame !== this.currentFrameIndex) {
            const camTs = Number(this.currentFrameData?.camera?.timestamp);
            const trajTs = Number(this.currentFrameData?.trajectory?.timestamp);
            const tsDeltaMs = (Number.isFinite(camTs) && Number.isFinite(trajTs))
                ? (trajTs - camTs) * 1000.0
                : null;
            fetch('http://127.0.0.1:7244/ingest/4b1a3fa9-dffd-42fc-a8f8-c8a4d88904be',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({runId:'post-fix-topdown-v3',hypothesisId:'H8',location:'visualizer.js:updateTopdownOverlay:timestamp-alignment',message:'Camera/trajectory timestamp alignment',data:{frameIndex:this.currentFrameIndex,cameraTs:Number.isFinite(camTs)?camTs:null,trajectoryTs:Number.isFinite(trajTs)?trajTs:null,tsDeltaMs},timestamp:Date.now()})}).catch(()=>{});
        }
        // #endregion

        ctx.save();
        ctx.strokeStyle = '#ff00ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        let first = true;
        let prevPx = null;
        let prevPy = null;
        let drawnCount = 0;
        let skippedNegativeY = 0;
        let skippedNonFinite = 0;
        let discontinuityBreaks = 0;
        let pxMin = Number.POSITIVE_INFINITY;
        let pxMax = Number.NEGATIVE_INFINITY;
        let pyMin = Number.POSITIVE_INFINITY;
        let pyMax = Number.NEGATIVE_INFINITY;
        for (const point of displayRenderPoints) {
            if (!Number.isFinite(point?.x) || !Number.isFinite(point?.y)) {
                skippedNonFinite += 1;
                continue;
            }
            if (point.y < 0) {
                skippedNegativeY += 1;
                continue;
            }
            const px = cx + (point.x * pixelsPerMeter);
            const py = cy - (point.y * pixelsPerMeter);
            pxMin = Math.min(pxMin, px);
            pxMax = Math.max(pxMax, px);
            pyMin = Math.min(pyMin, py);
            pyMax = Math.max(pyMax, py);
            if (first || prevPx === null || prevPy === null) {
                ctx.moveTo(px, py);
                first = false;
            } else {
                ctx.lineTo(px, py);
            }
            drawnCount += 1;
            prevPx = px;
            prevPy = py;
        }
        ctx.stroke();

        if (showOracle && oracleTrajectory.length > 0) {
            ctx.strokeStyle = '#66ff66';
            ctx.lineWidth = 2;
            ctx.beginPath();
            let oracleFirst = true;
            for (const point of oracleTrajectory) {
                const ox = Number(point?.x);
                const oy = Number(point?.y);
                if (!Number.isFinite(ox) || !Number.isFinite(oy) || oy < 0) {
                    continue;
                }
                const px = cx + (ox * pixelsPerMeter);
                const py = cy - (oy * pixelsPerMeter);
                if (oracleFirst) {
                    ctx.moveTo(px, py);
                    oracleFirst = false;
                } else {
                    ctx.lineTo(px, py);
                }
            }
            if (!oracleFirst) {
                ctx.stroke();
            }
        }

        if (showDistanceScale) {
            const scaleSegments = this.getDistanceScaleSegments(5, 30);
            if (scaleSegments.length > 0) {
                ctx.strokeStyle = '#ffffff';
                for (const seg of scaleSegments) {
                    const ax = Number(seg?.a?.x);
                    const ay = Number(seg?.a?.y);
                    const bx = Number(seg?.b?.x);
                    const by = Number(seg?.b?.y);
                    if (!Number.isFinite(ax) || !Number.isFinite(ay) || !Number.isFinite(bx) || !Number.isFinite(by)) {
                        continue;
                    }
                    let x0 = cx + (ax * pixelsPerMeter);
                    let y0 = cy - (ay * pixelsPerMeter);
                    let x1 = cx + (bx * pixelsPerMeter);
                    let y1 = cy - (by * pixelsPerMeter);
                    const topdownTickScale = 1.7;
                    const mx = (x0 + x1) * 0.5;
                    const my = (y0 + y1) * 0.5;
                    x0 = mx + (x0 - mx) * topdownTickScale;
                    y0 = my + (y0 - my) * topdownTickScale;
                    x1 = mx + (x1 - mx) * topdownTickScale;
                    y1 = my + (y1 - my) * topdownTickScale;
                    ctx.lineWidth = seg.s === 0 ? 4 : 3;
                    ctx.beginPath();
                    ctx.moveTo(x0, y0);
                    ctx.lineTo(x1, y1);
                    ctx.stroke();
                }
            }
        }

        ctx.fillStyle = '#ff3333';
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // #region agent log
        if (this._agentTopdownLastLoggedFrame !== this.currentFrameIndex) {
            let signFlips = 0;
            let prevDxSign = null;
            let maxAbsDxDy = 0;
            for (let i = 1; i < displayRenderPoints.length; i++) {
                const dx = Number(displayRenderPoints[i].x) - Number(displayRenderPoints[i - 1].x);
                const dy = Number(displayRenderPoints[i].y) - Number(displayRenderPoints[i - 1].y);
                if (Math.abs(dy) < 1e-6) continue;
                const s = Math.sign(dx / dy);
                if (prevDxSign !== null && s !== 0 && s !== prevDxSign) {
                    signFlips += 1;
                }
                if (s !== 0) prevDxSign = s;
                maxAbsDxDy = Math.max(maxAbsDxDy, Math.abs(dx / dy));
            }
            fetch('http://127.0.0.1:7244/ingest/4b1a3fa9-dffd-42fc-a8f8-c8a4d88904be',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({runId:'post-fix-topdown-v3',hypothesisId:'H6',location:'visualizer.js:updateTopdownOverlay:drawn',message:'Topdown normalized path geometry',data:{frameIndex:this.currentFrameIndex,drawnCount,skippedNegativeY,skippedNonFinite,discontinuityBreaks,bounds:{pxMin:Number.isFinite(pxMin)?pxMin:null,pxMax:Number.isFinite(pxMax)?pxMax:null,pyMin:Number.isFinite(pyMin)?pyMin:null,pyMax:Number.isFinite(pyMax)?pyMax:null},pixelsPerMeter,cx,cy,signFlips,maxAbsDxDy,firstPoint:displayRenderPoints[0]||null,lastPoint:displayRenderPoints[displayRenderPoints.length-1]||null,oracleCount:Array.isArray(oracleTrajectory)?oracleTrajectory.length:0,displaySmoothing:Boolean(document.getElementById('toggle-topdown-pose-smooth')?.checked)},timestamp:Date.now()})}).catch(()=>{});
            this._agentTopdownLastLoggedFrame = this.currentFrameIndex;
        }
        // #endregion

        this.projectionDiagnostics.topdown_turn_sign = this.computeTurnSign(displayRenderPoints, 'x', 'y');
        this.projectionDiagnostics.topdown_calibrated = useCalibratedTopdownProjection ? 'on' : 'off';
        this.projectionDiagnostics.topdown_smooth = Boolean(document.getElementById('toggle-topdown-pose-smooth')?.checked) ? 'on' : 'off';
        this.projectionDiagnostics.planner_only = Boolean(document.getElementById('toggle-planner-only-trajectory')?.checked) ? 'on' : 'off';

    }

    getTopdownDisplayRenderPoints(renderPoints) {
        const smoothEnabled = Boolean(document.getElementById('toggle-topdown-pose-smooth')?.checked);
        if (!smoothEnabled || !Array.isArray(renderPoints) || renderPoints.length === 0) {
            this.topdownSmoothedTrajectory = null;
            this.topdownSmoothedFrameIndex = null;
            return renderPoints;
        }

        const isSequential = (
            this.topdownSmoothedFrameIndex !== null &&
            this.currentFrameIndex === this.topdownSmoothedFrameIndex + 1
        );
        const sameLength = (
            Array.isArray(this.topdownSmoothedTrajectory) &&
            this.topdownSmoothedTrajectory.length === renderPoints.length
        );

        if (!isSequential || !sameLength) {
            this.topdownSmoothedTrajectory = renderPoints.map((p) => ({ ...p }));
            this.topdownSmoothedFrameIndex = this.currentFrameIndex;
            return this.topdownSmoothedTrajectory;
        }

        const alpha = this.topdownSmoothingAlpha;
        for (let i = 0; i < renderPoints.length; i++) {
            const prev = this.topdownSmoothedTrajectory[i];
            const cur = renderPoints[i];
            prev.x = (1 - alpha) * prev.x + alpha * cur.x;
            prev.y = (1 - alpha) * prev.y + alpha * cur.y;
        }
        this.topdownSmoothedFrameIndex = this.currentFrameIndex;
        return this.topdownSmoothedTrajectory;
    }

    updatePerceptionData() {
        if (!this.currentFrameData || !this.currentFrameData.perception) return;
        
        const p = this.currentFrameData.perception;
        
        // Update both individual tab and all-data tab
        const updateField = (id, value) => {
            const elem = document.getElementById(id);
            const allElem = document.getElementById('all-' + id);
            if (elem) elem.textContent = value;
            if (allElem) allElem.textContent = value;
        };
        
        updateField('perception-method', p.detection_method || '-');
        updateField('perception-confidence', p.confidence !== undefined ? p.confidence.toFixed(3) : '-');
        updateField('perception-num-lanes', p.num_lanes_detected || '-');

        const method = (p.detection_method || '').toLowerCase();
        const isCv = method === 'cv';
        const isSeg = method === 'segmentation';
        const hasModelFitPoints = Boolean(
            (Array.isArray(p.fit_points_left) && p.fit_points_left.length > 0) ||
            (Array.isArray(p.fit_points_right) && p.fit_points_right.length > 0)
        );

        // Disable fit points toggle when not using CV detection
        const fitToggle = document.getElementById('toggle-fit-points');
        if (fitToggle) {
            const fitLabel = fitToggle.closest('label');
            fitToggle.disabled = !isCv;
            if (!isCv) {
                fitToggle.checked = false;
            }
            if (fitLabel) {
                fitLabel.style.opacity = isCv ? '1.0' : '0.4';
            }
        }

        const segMaskToggle = document.getElementById('toggle-seg-mask');
        if (segMaskToggle) {
            const segLabel = segMaskToggle.closest('label');
            segMaskToggle.disabled = !isSeg;
            if (!isSeg) {
                segMaskToggle.checked = false;
            }
            if (segLabel) {
                segLabel.style.opacity = isSeg ? '1.0' : '0.4';
            }
        }
        const segPointsToggle = document.getElementById('toggle-seg-fit-points');
        if (segPointsToggle) {
            const segLabel = segPointsToggle.closest('label');
            segPointsToggle.disabled = !(isSeg || hasModelFitPoints);
            if (!(isSeg || hasModelFitPoints)) {
                segPointsToggle.checked = false;
            }
            if (segLabel) {
                segLabel.style.opacity = (isSeg || hasModelFitPoints) ? '1.0' : '0.4';
            }
        }
        const left_lane_line_x = p.left_lane_line_x !== undefined ? p.left_lane_line_x : (p.left_lane_x !== undefined ? p.left_lane_x : undefined);  // Backward compatibility
        const right_lane_line_x = p.right_lane_line_x !== undefined ? p.right_lane_line_x : (p.right_lane_x !== undefined ? p.right_lane_x : undefined);  // Backward compatibility
        
        updateField('perception-left-x', left_lane_line_x !== undefined ? `${left_lane_line_x.toFixed(3)}m` : '-');
        updateField('perception-right-x', right_lane_line_x !== undefined ? `${right_lane_line_x.toFixed(3)}m` : '-');
        
        if (left_lane_line_x !== undefined && right_lane_line_x !== undefined) {
            // FIXED: Calculate width using simple distance scaling
            // The recorded left_lane_line_x and right_lane_line_x are at 8.0m
            // Width scales linearly with distance (for same camera FOV)
            // If lines align at tunableDistance, use that for width calculation
            const recordedDistance = this.currentFrameData.vehicle && this.currentFrameData.vehicle.ground_truth_lookahead_distance
                ? this.currentFrameData.vehicle.ground_truth_lookahead_distance
                : 8.0;
            const tunableDistance = this.groundTruthDistance; // Distance from slider (where lines align)
            
            // Calculate width at recorded distance (raw perception width)
            const widthAtRecorded = right_lane_line_x - left_lane_line_x;
            updateField('perception-width', `${widthAtRecorded.toFixed(3)}m`);
        } else {
            updateField('perception-width', '-');
        }
        
        // NEW: Display stale data diagnostic fields
        const usingStale = p.using_stale_data !== undefined ? p.using_stale_data : false;
        
        // NEW: Show actual detected values when using stale data
        const actual_left = p.actual_detected_left_lane_x !== undefined && p.actual_detected_left_lane_x !== null ? p.actual_detected_left_lane_x : null;
        const actual_right = p.actual_detected_right_lane_x !== undefined && p.actual_detected_right_lane_x !== null ? p.actual_detected_right_lane_x : null;
        
        const showActual = usingStale && (actual_left !== null || actual_right !== null);
        
        // Show/hide actual detected rows
        const actualLeftRowIds = ['perception-actual-left-row', 'all-perception-actual-left-row'];
        const actualRightRowIds = ['perception-actual-right-row', 'all-perception-actual-right-row'];
        
        actualLeftRowIds.forEach(id => {
            const elem = document.getElementById(id);
            if (elem) {
                elem.style.display = (showActual && actual_left !== null) ? 'table-row' : 'none';
            }
        });
        
        actualRightRowIds.forEach(id => {
            const elem = document.getElementById(id);
            if (elem) {
                elem.style.display = (showActual && actual_right !== null) ? 'table-row' : 'none';
            }
        });
        
        // Update actual detected values
        if (showActual) {
            if (actual_left !== null) {
                updateField('perception-actual-left-x', `${actual_left.toFixed(3)}m`);
                updateField('all-perception-actual-left-x', `${actual_left.toFixed(3)}m`);
            }
            if (actual_right !== null) {
                updateField('perception-actual-right-x', `${actual_right.toFixed(3)}m`);
                updateField('all-perception-actual-right-x', `${actual_right.toFixed(3)}m`);
            }
        }
        updateField('perception-using-stale', usingStale ? 'YES ⚠️' : 'NO');
        updateField('all-perception-using-stale', usingStale ? 'YES ⚠️' : 'NO');
        
        const staleReason = p.stale_data_reason || '-';
        updateField('perception-stale-reason', staleReason);
        updateField('all-perception-stale-reason', staleReason);
        
        const rejectReason = p.reject_reason || '-';
        updateField('perception-reject-reason', rejectReason);
        updateField('all-perception-reject-reason', rejectReason);
        
        // NEW: Dynamic visibility based on stale_reason
        const isInstability = staleReason === 'perception_instability';
        const isJumpDetection = staleReason === 'jump_detection';
        
        // Show/hide instability diagnostic rows (width_change, center_shift)
        const instabilityRowIds = ['perception-instability-row', 'perception-instability-row-2', 
                                   'all-perception-instability-row', 'all-perception-instability-row-2'];
        instabilityRowIds.forEach(id => {
            const elem = document.getElementById(id);
            if (elem) {
                elem.style.display = isInstability ? 'table-row' : 'none';
            }
        });
        
        // Show/hide jump detection rows (left_jump, right_jump, jump_threshold)
        const jumpRowIds = ['perception-jump-row', 'perception-jump-row-2', 'perception-jump-row-3',
                            'all-perception-jump-row', 'all-perception-jump-row-2', 'all-perception-jump-row-3'];
        jumpRowIds.forEach(id => {
            const elem = document.getElementById(id);
            if (elem) {
                elem.style.display = isJumpDetection ? 'table-row' : 'none';
            }
        });
        
        // Calculate and display width_change and center_shift (only if instability)
        // NEW: Use stored values if available (from new diagnostic fields)
        if (isInstability) {
            // Prefer stored values (actual detected values and calculated changes from AV stack)
            const stored_width_change = p.instability_width_change !== undefined && p.instability_width_change !== null ? p.instability_width_change : null;
            const stored_center_shift = p.instability_center_shift !== undefined && p.instability_center_shift !== null ? p.instability_center_shift : null;
            
            if (stored_width_change !== null && stored_center_shift !== null) {
                // Use stored values (from AV stack) - these are the actual changes that triggered instability
                updateField('perception-width-change', `${stored_width_change.toFixed(3)}m`);
                updateField('all-perception-width-change', `${stored_width_change.toFixed(3)}m`);
                updateField('perception-center-shift', `${stored_center_shift.toFixed(3)}m`);
                updateField('all-perception-center-shift', `${stored_center_shift.toFixed(3)}m`);
            } else {
                // Fallback: Calculate from previous frame (for old recordings without diagnostic fields)
                if (this.previousPerceptionData && 
                    this.previousPerceptionData.left_lane_line_x !== null && 
                    this.previousPerceptionData.right_lane_line_x !== null &&
                    left_lane_line_x !== undefined && 
                    right_lane_line_x !== undefined) {
                    
                    const prev_width = this.previousPerceptionData.right_lane_line_x - this.previousPerceptionData.left_lane_line_x;
                    const prev_center = (this.previousPerceptionData.left_lane_line_x + this.previousPerceptionData.right_lane_line_x) / 2.0;
                    const current_width = right_lane_line_x - left_lane_line_x;
                    const current_center = (left_lane_line_x + right_lane_line_x) / 2.0;
                    const width_change = Math.abs(current_width - prev_width);
                    const center_shift = Math.abs(current_center - prev_center);
                    
                    const displayText = width_change < 0.001 ? '0.000m (fallback)' : `${width_change.toFixed(3)}m`;
                    const displayTextCenter = center_shift < 0.001 ? '0.000m (fallback)' : `${center_shift.toFixed(3)}m`;
                    
                    updateField('perception-width-change', displayText);
                    updateField('all-perception-width-change', displayText);
                    updateField('perception-center-shift', displayTextCenter);
                    updateField('all-perception-center-shift', displayTextCenter);
                } else {
                    updateField('perception-width-change', '-');
                    updateField('all-perception-width-change', '-');
                    updateField('perception-center-shift', '-');
                    updateField('all-perception-center-shift', '-');
                }
            }
            
            // Apply red color for instability
            const widthChangeElem = document.getElementById('perception-width-change');
            const allWidthChangeElem = document.getElementById('all-perception-width-change');
            const centerShiftElem = document.getElementById('perception-center-shift');
            const allCenterShiftElem = document.getElementById('all-perception-center-shift');
            
            if (widthChangeElem) widthChangeElem.style.color = '#ff6b6b';
            if (allWidthChangeElem) allWidthChangeElem.style.color = '#ff6b6b';
            if (centerShiftElem) centerShiftElem.style.color = '#ff6b6b';
            if (allCenterShiftElem) allCenterShiftElem.style.color = '#ff6b6b';
        } else {
            updateField('perception-width-change', '-');
            updateField('all-perception-width-change', '-');
            updateField('perception-center-shift', '-');
            updateField('all-perception-center-shift', '-');
        }
        
        // Jump detection fields (only if jump_detection)
        if (isJumpDetection) {
            const leftJump = p.left_jump_magnitude !== undefined ? p.left_jump_magnitude : null;
            updateField('perception-left-jump', leftJump !== null ? `${leftJump.toFixed(3)}m` : '-');
            updateField('all-perception-left-jump', leftJump !== null ? `${leftJump.toFixed(3)}m` : '-');
            
            const rightJump = p.right_jump_magnitude !== undefined ? p.right_jump_magnitude : null;
            updateField('perception-right-jump', rightJump !== null ? `${rightJump.toFixed(3)}m` : '-');
            updateField('all-perception-right-jump', rightJump !== null ? `${rightJump.toFixed(3)}m` : '-');
            
            const jumpThreshold = p.jump_threshold !== undefined ? p.jump_threshold : null;
            updateField('perception-jump-threshold', jumpThreshold !== null ? `${jumpThreshold.toFixed(2)}m` : '-');
            updateField('all-perception-jump-threshold', jumpThreshold !== null ? `${jumpThreshold.toFixed(2)}m` : '-');
        } else {
            updateField('perception-left-jump', '-');
            updateField('all-perception-left-jump', '-');
            updateField('perception-right-jump', '-');
            updateField('all-perception-right-jump', '-');
            updateField('perception-jump-threshold', '-');
            updateField('all-perception-jump-threshold', '-');
        }
        
        // NEW: Display perception health metrics
        const consecutiveBad = p.consecutive_bad_detection_frames !== undefined ? p.consecutive_bad_detection_frames : 0;
        const healthScore = p.perception_health_score !== undefined ? p.perception_health_score : 1.0;
        const healthStatus = p.perception_health_status || 'healthy';
        const badEvents = p.perception_bad_events || '-';
        const badEventsRecent = p.perception_bad_events_recent || '-';
        const clampEvents = p.perception_clamp_events || '-';
        const timestampFrozen = p.perception_timestamp_frozen ? 'YES ⚠️' : 'NO';
        
        // Color code health status
        let healthColor = '#4caf50'; // green
        if (healthStatus === 'degraded') healthColor = '#ff9800'; // orange
        else if (healthStatus === 'poor') healthColor = '#ff5722'; // deep orange
        else if (healthStatus === 'critical') healthColor = '#f44336'; // red
        
        updateField('perception-consecutive-bad', consecutiveBad.toString());
        updateField('all-perception-consecutive-bad', consecutiveBad.toString());
        
        updateField('perception-health-score', `${(healthScore * 100).toFixed(1)}%`);
        updateField('all-perception-health-score', `${(healthScore * 100).toFixed(1)}%`);
        
        const statusElem = document.getElementById('perception-health-status');
        const allStatusElem = document.getElementById('all-perception-health-status');
        if (statusElem) {
            statusElem.textContent = healthStatus;
            statusElem.style.color = healthColor;
        }
        if (allStatusElem) {
            allStatusElem.textContent = healthStatus;
            allStatusElem.style.color = healthColor;
        }
        updateField('perception-health-bad-events', badEvents);
        updateField('all-perception-health-bad-events', badEvents);
        updateField('perception-health-bad-events-recent', badEventsRecent);
        updateField('all-perception-health-bad-events-recent', badEventsRecent);
        updateField('perception-clamp-events', clampEvents);
        updateField('all-perception-clamp-events', clampEvents);
        updateField('perception-timestamp-frozen', timestampFrozen);
        updateField('all-perception-timestamp-frozen', timestampFrozen);
    }

    updateTrajectoryData() {
        if (!this.currentFrameData || !this.currentFrameData.trajectory) return;
        
        const t = this.currentFrameData.trajectory;
        const rp = t.reference_point;
        
        const updateField = (id, value) => {
            const elem = document.getElementById(id);
            const allElem = document.getElementById('all-' + id);
            if (elem) elem.textContent = value;
            if (allElem) allElem.textContent = value;
        };
        
        if (rp) {
            updateField('trajectory-x', rp.x !== undefined ? `${rp.x.toFixed(3)}m` : '-');
            updateField('trajectory-y', rp.y !== undefined ? `${rp.y.toFixed(3)}m` : '-');
            updateField('trajectory-heading', rp.heading !== undefined ? `${(rp.heading * 180 / Math.PI).toFixed(2)}°` : '-');
            updateField('trajectory-velocity', rp.velocity !== undefined ? `${rp.velocity.toFixed(2)}m/s` : '-');
            
            if (t.reference_point_raw) {
                const rpr = t.reference_point_raw;
                updateField('trajectory-raw-x', rpr.x !== undefined ? `${rpr.x.toFixed(3)}m` : '-');
                updateField('trajectory-raw-y', rpr.y !== undefined ? `${rpr.y.toFixed(3)}m` : '-');
                updateField('trajectory-raw-heading', rpr.heading !== undefined ? `${(rpr.heading * 180 / Math.PI).toFixed(2)}°` : '-');
            }
            
            // NEW: Display debug information
            const method = t.reference_point_method || t.reference_point?.method || 'unknown';
            updateField('trajectory-method', method);
            
            const perceptionCenterX = t.perception_center_x !== undefined ? t.perception_center_x : 
                                     (t.reference_point?.perception_center_x !== undefined ? t.reference_point.perception_center_x : null);
            if (perceptionCenterX !== null && perceptionCenterX !== undefined) {
                updateField('trajectory-perception-center', `${perceptionCenterX.toFixed(3)}m`);
                
                // Calculate difference between trajectory and perception
                if (rp && rp.x !== undefined) {
                    const diff = rp.x - perceptionCenterX;
                    const diffStr = `${diff.toFixed(3)}m ${diff > 0 ? '(traj right)' : '(traj left)'}`;
                    updateField('trajectory-vs-perception', diffStr);
                } else {
                    updateField('trajectory-vs-perception', '-');
                }
            } else {
                updateField('trajectory-perception-center', '-');
                updateField('trajectory-vs-perception', '-');
            }
        }
    }

    updateProjectionData() {
        const updateField = (id, value) => {
            const elem = document.getElementById(id);
            if (elem) elem.textContent = value;
        };
        const d = this.projectionDiagnostics || {};
        updateField(
            'projection-main-first-visible-y',
            Number.isFinite(Number(d.main_first_visible_src_y_m))
                ? `${Number(d.main_first_visible_src_y_m).toFixed(2)} m`
                : '-'
        );
        updateField('projection-main-mirror-sanity', d.main_mirror_sanity || '-');
        updateField('projection-main-nearfield-blend', d.main_nearfield_blend || '-');
        updateField(
            'projection-main-nearfield-y-offset',
            Number.isFinite(Number(d.main_nearfield_y_offset_m))
                ? `${Number(d.main_nearfield_y_offset_m).toFixed(2)} m`
                : '-'
        );
        updateField(
            'projection-main-nearfield-blend-distance',
            Number.isFinite(Number(d.main_nearfield_blend_distance_m))
                ? `${Number(d.main_nearfield_blend_distance_m).toFixed(1)} m`
                : '-'
        );
        updateField('projection-source-turn-sign', d.source_turn_sign || '-');
        updateField('projection-main-turn-sign', d.main_turn_sign || '-');
        const fmtErr = (v) => {
            if (!Number.isFinite(Number(v))) return '-';
            const n = Number(v);
            const side = n > 0 ? 'right' : (n < 0 ? 'left' : 'center');
            return `${Math.abs(n).toFixed(3)} m (${side})`;
        };
        updateField('projection-lateral-error-5m', fmtErr(d.lateral_error_5m));
        updateField('projection-lateral-error-10m', fmtErr(d.lateral_error_10m));
        updateField('projection-lateral-error-15m', fmtErr(d.lateral_error_15m));
        updateField('projection-lateral-error-source', d.lateral_error_source || '-');
        const fmtDeg = (v) => {
            if (!Number.isFinite(Number(v))) return '-';
            const n = Number(v);
            const side = n > 0 ? 'right' : (n < 0 ? 'left' : 'aligned');
            return `${Math.abs(n).toFixed(2)} deg (${side})`;
        };
        const fmtRatio = (v) => {
            if (!Number.isFinite(Number(v))) return '-';
            return `${Number(v).toFixed(2)}x`;
        };
        updateField('projection-heading-delta-5m', fmtDeg(d.heading_delta_5m_deg));
        updateField('projection-heading-delta-10m', fmtDeg(d.heading_delta_10m_deg));
        updateField('projection-heading-delta-15m', fmtDeg(d.heading_delta_15m_deg));
        updateField('projection-curvature-ratio-5m', fmtRatio(d.curvature_ratio_5m));
        updateField('projection-curvature-ratio-10m', fmtRatio(d.curvature_ratio_10m));
        updateField('projection-curvature-ratio-15m', fmtRatio(d.curvature_ratio_15m));
        updateField('projection-turn-mismatch-source', d.turn_mismatch_source || '-');
        const fmtBool = (v) => {
            if (v === true) return 'on';
            if (v === false) return 'off';
            return '-';
        };
        const fmtFlag = (v) => {
            if (v === true || Number(v) > 0.5) return 'on';
            if (v === false || (Number.isFinite(Number(v)) && Number(v) <= 0.5)) return 'off';
            return '-';
        };
        updateField('projection-traj-heading-zero-gate', fmtBool(d.traj_heading_zero_gate));
        updateField('projection-traj-small-heading-gate', fmtBool(d.traj_small_heading_gate));
        updateField('projection-traj-multilookahead-active', fmtBool(d.traj_multilookahead_active));
        updateField(
            'projection-traj-x-clip-count',
            Number.isFinite(Number(d.traj_x_clip_count)) ? Number(d.traj_x_clip_count).toFixed(0) : '-'
        );
        updateField('projection-traj-heavy-x-clipping', fmtBool(d.traj_heavy_x_clipping));
        updateField(
            'projection-traj-preclip-abs-max',
            Number.isFinite(Number(d.traj_preclip_abs_max))
                ? `${Number(d.traj_preclip_abs_max).toFixed(2)} m`
                : '-'
        );
        updateField(
            'projection-traj-preclip-abs-p95',
            Number.isFinite(Number(d.traj_preclip_abs_p95))
                ? `${Number(d.traj_preclip_abs_p95).toFixed(2)} m`
                : '-'
        );
        updateField(
            'projection-traj-front-frame-delta',
            Number.isFinite(Number(d.traj_front_frame_delta))
                ? Number(d.traj_front_frame_delta).toFixed(0)
                : '-'
        );
        updateField(
            'projection-traj-front-unity-dt-ms',
            Number.isFinite(Number(d.traj_front_unity_dt_ms))
                ? Number(d.traj_front_unity_dt_ms).toFixed(2)
                : '-'
        );
        updateField('projection-traj-overlay-snap-risk', fmtBool(d.traj_overlay_snap_risk));
        updateField('projection-traj-contract-misaligned-risk', fmtBool(d.traj_contract_misaligned_risk));
        updateField('projection-traj-cadence-risk', fmtBool(d.traj_cadence_risk));
        const cadencePolicy = String(d.traj_cadence_policy || 'unknown').toLowerCase();
        const cadencePolicyLabel = cadencePolicy || 'unknown';
        const cadenceFrameDeltaThreshold = Number(d.traj_cadence_frame_delta_threshold);
        const cadenceUnityDtThresholdMs = Number(d.traj_cadence_unity_dt_threshold_ms);
        if (Number.isFinite(cadenceFrameDeltaThreshold) && Number.isFinite(cadenceUnityDtThresholdMs)) {
            updateField(
                'projection-traj-cadence-policy',
                `${cadencePolicyLabel} (frameΔ>=${cadenceFrameDeltaThreshold.toFixed(0)}, |dt|>=${cadenceUnityDtThresholdMs.toFixed(0)}ms)`
            );
        } else {
            updateField('projection-traj-cadence-policy', '-');
        }
        updateField(
            'projection-traj-control-curv-ratio-10m',
            Number.isFinite(Number(d.traj_control_curv_ratio_10m))
                ? `${Number(d.traj_control_curv_ratio_10m).toFixed(2)}x`
                : '-'
        );
        updateField('projection-traj-underturn-10m-flag', fmtBool(d.traj_underturn_10m_flag));
        updateField('projection-traj-waterfall-source', d.traj_waterfall_source || '-');
        updateField('projection-sync-overall-status', d.sync_overall_status || '-');
        updateField('projection-sync-traj-status', d.sync_traj_status || '-');
        updateField('projection-sync-control-status', d.sync_control_status || '-');
        updateField(
            'projection-sync-window-ms',
            Number.isFinite(Number(d.sync_window_ms)) ? Number(d.sync_window_ms).toFixed(1) : '-'
        );
        updateField(
            'projection-sync-cam-traj-dt-ms',
            Number.isFinite(Number(d.sync_dt_cam_traj_ms)) ? Number(d.sync_dt_cam_traj_ms).toFixed(2) : '-'
        );
        updateField(
            'projection-sync-cam-control-dt-ms',
            Number.isFinite(Number(d.sync_dt_cam_control_ms)) ? Number(d.sync_dt_cam_control_ms).toFixed(2) : '-'
        );
        updateField(
            'projection-sync-cam-vehicle-dt-ms',
            Number.isFinite(Number(d.sync_dt_cam_vehicle_ms)) ? Number(d.sync_dt_cam_vehicle_ms).toFixed(2) : '-'
        );
        updateField('projection-sync-traj-reason', d.sync_traj_reason || '-');
        updateField('projection-sync-control-reason', d.sync_control_reason || '-');
        updateField('projection-clock-front-ts-reused', fmtFlag(d.clock_front_ts_reused));
        updateField('projection-clock-front-ts-nonmono', fmtFlag(d.clock_front_ts_nonmonotonic));
        updateField('projection-clock-front-id-reused', fmtFlag(d.clock_front_id_reused));
        updateField('projection-clock-front-negative-delta', fmtFlag(d.clock_front_negative_delta));
        updateField('projection-clock-front-jump', fmtFlag(d.clock_front_jump));
        updateField('projection-clock-top-ts-reused', fmtFlag(d.clock_top_ts_reused));
        updateField('projection-clock-top-ts-nonmono', fmtFlag(d.clock_top_ts_nonmonotonic));
        updateField('projection-clock-top-id-reused', fmtFlag(d.clock_top_id_reused));
        updateField('projection-clock-top-negative-delta', fmtFlag(d.clock_top_negative_delta));
        updateField('projection-clock-top-jump', fmtFlag(d.clock_top_jump));
        const fmtPx = (v) => Number.isFinite(Number(v)) ? `${Number(v).toFixed(1)} px` : '-';
        updateField('projection-right-fiducial-err-5m', fmtPx(d.right_fiducial_err_5m));
        updateField('projection-right-fiducial-err-10m', fmtPx(d.right_fiducial_err_10m));
        updateField('projection-right-fiducial-err-15m', fmtPx(d.right_fiducial_err_15m));
        updateField('projection-right-fiducial-err-mean', fmtPx(d.right_fiducial_err_mean));
        updateField('projection-right-fiducial-err-max', fmtPx(d.right_fiducial_err_max));
        updateField('projection-right-fiducial-source', d.right_fiducial_source || '-');
        updateField('projection-topdown-turn-sign', d.topdown_turn_sign || '-');
        updateField('projection-topdown-calibrated', d.topdown_calibrated ?? '-');
        updateField('projection-topdown-smooth', d.topdown_smooth ?? '-');
        updateField('projection-planner-only', d.planner_only ?? '-');
    }

    computeLongitudinalMetrics(currentFrame, prevFrame, prevPrevFrame) {
        const currVehicle = currentFrame ? currentFrame.vehicle : null;
        const prevVehicle = prevFrame ? prevFrame.vehicle : null;
        const prevPrevVehicle = prevPrevFrame ? prevPrevFrame.vehicle : null;
        if (!currVehicle || currVehicle.speed === undefined || currVehicle.speed === null) {
            return { accel: null, jerk: null };
        }

        const currTime = currVehicle.timestamp ?? currentFrame?.camera?.timestamp ?? null;
        const prevTime = prevVehicle?.timestamp ?? prevFrame?.camera?.timestamp ?? null;
        const prevPrevTime = prevPrevVehicle?.timestamp ?? prevPrevFrame?.camera?.timestamp ?? null;

        let accel = null;
        let jerk = null;
        if (prevVehicle && prevVehicle.speed !== undefined && prevVehicle.speed !== null && currTime !== null && prevTime !== null) {
            const dt = currTime - prevTime;
            if (dt > 1e-6) {
                accel = (currVehicle.speed - prevVehicle.speed) / dt;
            }
        }
        if (
            accel !== null
            && prevPrevVehicle
            && prevPrevVehicle.speed !== undefined
            && prevPrevVehicle.speed !== null
            && prevTime !== null
            && prevPrevTime !== null
        ) {
            const dtPrev = prevTime - prevPrevTime;
            const dtCurr = currTime !== null && prevTime !== null ? (currTime - prevTime) : null;
            if (dtPrev > 1e-6 && dtCurr !== null && dtCurr > 1e-6) {
                const prevAccel = (prevVehicle.speed - prevPrevVehicle.speed) / dtPrev;
                jerk = (accel - prevAccel) / dtCurr;
            }
        }

        return { accel, jerk };
    }

    updateControlData() {
        if (!this.currentFrameData || !this.currentFrameData.control) return;
        
        const c = this.currentFrameData.control;
        const t = this.currentFrameData.trajectory || null;
        const ref = t ? t.reference_point : null;
        
        const updateField = (id, value) => {
            const elem = document.getElementById(id);
            const allElem = document.getElementById('all-' + id);
            if (elem) elem.textContent = value;
            if (allElem) allElem.textContent = value;
        };
        
        updateField('control-steering', c.steering !== undefined ? c.steering.toFixed(4) : '-');
        updateField('control-throttle', c.throttle !== undefined ? c.throttle.toFixed(4) : '-');
        updateField('control-brake', c.brake !== undefined ? c.brake.toFixed(4) : '-');
        const longitudinal = this.currentLongitudinalMetrics || { accel: null, jerk: null };
        updateField(
            'control-accel',
            longitudinal.accel !== null && longitudinal.accel !== undefined
                ? `${longitudinal.accel.toFixed(2)} m/s²`
                : '-'
        );
        updateField(
            'control-jerk',
            longitudinal.jerk !== null && longitudinal.jerk !== undefined
                ? `${longitudinal.jerk.toFixed(2)} m/s³`
                : '-'
        );
        updateField('control-ref-x', ref && ref.x !== undefined ? `${ref.x.toFixed(3)}m` : '-');
        updateField('control-ref-y', ref && ref.y !== undefined ? `${ref.y.toFixed(3)}m` : '-');
        updateField(
            'control-ref-heading',
            ref && ref.heading !== undefined ? `${(ref.heading * 180 / Math.PI).toFixed(2)}°` : '-'
        );
        updateField(
            'control-ref-speed',
            ref && ref.velocity !== undefined ? `${ref.velocity.toFixed(2)} m/s` : '-'
        );
        updateField('control-lateral-error', c.lateral_error !== undefined ? `${c.lateral_error.toFixed(3)}m` : '-');
        updateField('control-heading-error', c.heading_error !== undefined ? `${(c.heading_error * 180 / Math.PI).toFixed(2)}°` : '-');
        updateField('control-total-error', c.total_error !== undefined ? c.total_error.toFixed(4) : '-');
        updateField('control-total-error-scaled', c.total_error_scaled !== undefined ? c.total_error_scaled.toFixed(4) : '-');
        updateField('control-feedforward-steering', c.feedforward_steering !== undefined ? c.feedforward_steering.toFixed(4) : '-');
        updateField('control-feedback-steering', c.feedback_steering !== undefined ? c.feedback_steering.toFixed(4) : '-');
        updateField(
            'control-straight-sign-flip-override',
            c.straight_sign_flip_override_active !== undefined
                ? (c.straight_sign_flip_override_active ? 'YES' : 'NO')
                : '-'
        );
        updateField('control-pid-integral', c.pid_integral !== undefined ? c.pid_integral.toFixed(4) : '-');
        updateField('control-pid-derivative', c.pid_derivative !== undefined ? c.pid_derivative.toFixed(4) : '-');
        updateField('control-pid-error', c.pid_error !== undefined ? c.pid_error.toFixed(4) : '-');
        updateField(
            'control-target-speed-raw',
            c.target_speed_raw !== undefined ? `${c.target_speed_raw.toFixed(2)} m/s` : '-'
        );
        updateField(
            'control-target-speed-post-limits',
            c.target_speed_post_limits !== undefined ? `${c.target_speed_post_limits.toFixed(2)} m/s` : '-'
        );
        updateField(
            'control-target-speed-planned',
            c.target_speed_planned !== undefined ? `${c.target_speed_planned.toFixed(2)} m/s` : '-'
        );
        updateField(
            'control-target-speed-final',
            c.target_speed_final !== undefined ? `${c.target_speed_final.toFixed(2)} m/s` : '-'
        );
        updateField(
            'control-target-speed-slew-active',
            c.target_speed_slew_active !== undefined ? (c.target_speed_slew_active ? 'YES' : 'NO') : '-'
        );
        updateField(
            'control-target-speed-ramp-active',
            c.target_speed_ramp_active !== undefined ? (c.target_speed_ramp_active ? 'YES' : 'NO') : '-'
        );
        updateField(
            'control-launch-throttle-cap',
            c.launch_throttle_cap !== undefined ? c.launch_throttle_cap.toFixed(3) : '-'
        );
        updateField(
            'control-launch-throttle-cap-active',
            c.launch_throttle_cap_active !== undefined ? (c.launch_throttle_cap_active ? 'YES' : 'NO') : '-'
        );
        updateField('control-is-straight', c.is_straight !== undefined ? (c.is_straight ? 'YES' : 'NO') : '-');
        updateField('control-straight-oscillation-rate', c.straight_oscillation_rate !== undefined ? c.straight_oscillation_rate.toFixed(3) : '-');
        updateField('control-tuned-deadband', c.tuned_deadband !== undefined ? c.tuned_deadband.toFixed(3) : '-');
        updateField('control-tuned-smoothing-alpha', c.tuned_error_smoothing_alpha !== undefined ? c.tuned_error_smoothing_alpha.toFixed(3) : '-');
        
        // NEW: Display stale perception diagnostic fields
        const usingStalePerception = c.using_stale_perception !== undefined ? c.using_stale_perception : false;
        updateField('control-using-stale', usingStalePerception ? 'YES ⚠️' : 'NO');
        updateField('all-control-using-stale', usingStalePerception ? 'YES ⚠️' : 'NO');
        
        const stalePerceptionReason = c.stale_perception_reason || '-';
        updateField('control-stale-reason', stalePerceptionReason);
        updateField('all-control-stale-reason', stalePerceptionReason);

        // Steering limiter waterfall diagnostics
        const fmtSteer = (v) => (v !== undefined && v !== null ? Number(v).toFixed(4) : '-');
        const fmtDelta = (v) => (v !== undefined && v !== null ? Number(v).toFixed(4) : '-');
        const fmtFlag = (v) => (v ? 'YES ⚠️' : 'NO');
        updateField('control-steering-pre-rate', fmtSteer(c.steering_pre_rate_limit));
        updateField('control-steering-post-rate', fmtSteer(c.steering_post_rate_limit));
        updateField('control-steering-post-jerk', fmtSteer(c.steering_post_jerk_limit));
        updateField('control-steering-post-sign-flip', fmtSteer(c.steering_post_sign_flip));
        updateField('control-steering-post-hard-clip', fmtSteer(c.steering_post_hard_clip));
        updateField('control-steering-post-smoothing', fmtSteer(c.steering_post_smoothing));
        updateField('control-steering-rate-limited', fmtFlag(!!c.steering_rate_limited_active));
        updateField('control-steering-jerk-limited', fmtFlag(!!c.steering_jerk_limited_active));
        updateField('control-steering-hard-clipped', fmtFlag(!!c.steering_hard_clip_active));
        updateField('control-steering-smoothing-active', fmtFlag(!!c.steering_smoothing_active));
        updateField('control-steering-rate-delta', fmtDelta(c.steering_rate_limited_delta));
        updateField('control-steering-jerk-delta', fmtDelta(c.steering_jerk_limited_delta));
        updateField('control-steering-hard-clip-delta', fmtDelta(c.steering_hard_clip_delta));
        updateField('control-steering-smoothing-delta', fmtDelta(c.steering_smoothing_delta));
    }

    updateVehicleData() {
        if (!this.currentFrameData || !this.currentFrameData.vehicle) return;
        
        const v = this.currentFrameData.vehicle;
        const mpsToMph = (mps) => mps * 2.236936;
        
        const updateField = (id, value) => {
            const elem = document.getElementById(id);
            const allElem = document.getElementById('all-' + id);
            if (elem) elem.textContent = value;
            if (allElem) allElem.textContent = value;
        };
        
        // NEW: Display camera calibration data
        const camera8mY = v.camera_8m_screen_y;
        if (camera8mY !== undefined && camera8mY !== null) {
            // Handle both -1.0 (invalid) and 0.0 (Unity not calculating) - both mean "not available"
            // CRITICAL: If value is 0.0 or -1.0, check if we have a cached valid value
            let displayValue;
            let isValid = false;
            
            if (camera8mY > 0) {
                // Valid value from Unity
                displayValue = `${camera8mY.toFixed(1)}px`;
                isValid = true;
                // Cache it for future frames
                this.lastValidY8m = camera8mY;
            } else if (this.lastValidY8m !== undefined && this.lastValidY8m > 0) {
                // Invalid value (0.0 or -1.0) but we have a cached valid value
                displayValue = `${this.lastValidY8m.toFixed(1)}px (cached)`;
                isValid = true;
            } else {
                // No valid value and no cache
                displayValue = (camera8mY === -1.0) ? '-1.0 (not calculated)' : '0.0 (Unity not calculating)';
                isValid = false;
            }
            
            updateField('vehicle-camera-8m-y', displayValue);
            
            const camera8mStatusElem = document.getElementById('vehicle-camera-8m-status');
            if (camera8mStatusElem) {
                if (isValid) {
                    camera8mStatusElem.textContent = '✅ Valid (from Unity or cached)';
                    camera8mStatusElem.style.color = '#00ff00';
                } else {
                    camera8mStatusElem.textContent = '❌ Not available (camera not found or old recording)';
                    camera8mStatusElem.style.color = '#ff0000';
                }
            }
        } else {
            // No data in frame, but check cache
            if (this.lastValidY8m !== undefined && this.lastValidY8m > 0) {
                updateField('vehicle-camera-8m-y', `${this.lastValidY8m.toFixed(1)}px (cached)`);
                const camera8mStatusElem = document.getElementById('vehicle-camera-8m-status');
                if (camera8mStatusElem) {
                    camera8mStatusElem.textContent = '✅ Valid (cached)';
                    camera8mStatusElem.style.color = '#00ff00';
                }
            } else {
                updateField('vehicle-camera-8m-y', '- (no data)');
            }
        }
        
        // NEW: Display camera FOV values from Unity
        const cameraFieldOfView = v.camera_field_of_view;
        const cameraHorizontalFOV = v.camera_horizontal_fov;
        if (cameraFieldOfView !== undefined && cameraFieldOfView !== null && cameraFieldOfView > 0) {
            updateField('vehicle-camera-fov-vertical', `${cameraFieldOfView.toFixed(2)}°`);
        } else {
            updateField('vehicle-camera-fov-vertical', '- (not available)');
        }
        if (cameraHorizontalFOV !== undefined && cameraHorizontalFOV !== null && cameraHorizontalFOV > 0) {
            updateField('vehicle-camera-fov-horizontal', `${cameraHorizontalFOV.toFixed(2)}°`);
        } else {
            updateField('vehicle-camera-fov-horizontal', '- (not available)');
        }

        const configFov = this.summaryConfig?.camera_fov;
        if (configFov !== undefined && configFov !== null && configFov > 0) {
            updateField('vehicle-camera-fov-config', `${configFov.toFixed(2)}°`);
        } else {
            updateField('vehicle-camera-fov-config', '-');
        }
        const configCameraHeight = this.summaryConfig?.camera_height;
        if (configCameraHeight !== undefined && configCameraHeight !== null && configCameraHeight > 0) {
            updateField('vehicle-camera-height-config', `${configCameraHeight.toFixed(2)}m`);
        } else {
            updateField('vehicle-camera-height-config', '-');
        }
        
        // NEW: Display camera position and forward direction from Unity
        const cameraPosX = v.camera_pos_x;
        const cameraPosY = v.camera_pos_y;
        const cameraPosZ = v.camera_pos_z;
        const cameraForwardX = v.camera_forward_x;
        const cameraForwardY = v.camera_forward_y;
        const cameraForwardZ = v.camera_forward_z;
        
        if (cameraPosX !== undefined && cameraPosX !== null && (cameraPosX !== 0.0 || cameraPosY !== 0.0 || cameraPosZ !== 0.0)) {
            updateField('vehicle-camera-pos-x', `${cameraPosX.toFixed(3)}m`);
            updateField('vehicle-camera-pos-y', `${cameraPosY.toFixed(3)}m`);
            updateField('vehicle-camera-pos-z', `${cameraPosZ.toFixed(3)}m`);
        } else {
            updateField('vehicle-camera-pos-x', '- (not available)');
            updateField('vehicle-camera-pos-y', '- (not available)');
            updateField('vehicle-camera-pos-z', '- (not available)');
        }
        
        if (cameraForwardX !== undefined && cameraForwardX !== null && (cameraForwardX !== 0.0 || cameraForwardY !== 0.0 || cameraForwardZ !== 0.0)) {
            updateField('vehicle-camera-forward-x', `${cameraForwardX.toFixed(3)}`);
            updateField('vehicle-camera-forward-y', `${cameraForwardY.toFixed(3)}`);
            updateField('vehicle-camera-forward-z', `${cameraForwardZ.toFixed(3)}`);
        } else {
            updateField('vehicle-camera-forward-x', '- (not available)');
            updateField('vehicle-camera-forward-y', '- (not available)');
            updateField('vehicle-camera-forward-z', '- (not available)');
        }

        // NEW: Camera frame metadata (camera is the reference timeline)
        const cameraInfo = this.currentFrameData.camera || {};
        const topdownInfo = this.currentFrameData.camera_topdown || {};
        const trajectoryInfo = this.currentFrameData.trajectory || {};
        const cameraTimestamp = cameraInfo.timestamp;
        const topdownTimestamp = topdownInfo.timestamp;
        const cameraFrameId = cameraInfo.frame_id;
        const topdownFrameId = topdownInfo.frame_id;
        const trajectoryTimestamp = trajectoryInfo.timestamp;
        updateField(
            'vehicle-camera-timestamp',
            cameraTimestamp !== undefined && cameraTimestamp !== null ? cameraTimestamp.toFixed(3) : '-'
        );
        updateField(
            'vehicle-topdown-timestamp',
            topdownTimestamp !== undefined && topdownTimestamp !== null ? topdownTimestamp.toFixed(3) : '-'
        );
        updateField('vehicle-camera-frame-id', cameraFrameId !== undefined && cameraFrameId !== null ? cameraFrameId : '-');
        updateField('vehicle-topdown-frame-id', topdownFrameId !== undefined && topdownFrameId !== null ? topdownFrameId : '-');
        
        // Flag when camera timestamp is reused across sequential frames
        let cameraTimestampReusedDisplay = '-';
        let cameraTimestampReused = false;
        if (cameraTimestamp !== undefined && cameraTimestamp !== null) {
            if (this.lastCameraTimestamp !== null && this.lastUnityFrameIndex === this.currentFrameIndex - 1) {
                cameraTimestampReused = cameraTimestamp === this.lastCameraTimestamp;
                cameraTimestampReusedDisplay = cameraTimestampReused ? 'Yes' : 'No';
            } else {
                cameraTimestampReusedDisplay = 'No';
            }
            this.lastCameraTimestamp = cameraTimestamp;
        }
        updateField('vehicle-camera-timestamp-reused', cameraTimestampReusedDisplay);
        const camReuseElem = document.getElementById('vehicle-camera-timestamp-reused');
        const camReuseAllElem = document.getElementById('all-vehicle-camera-timestamp-reused');
        if (camReuseElem) camReuseElem.style.color = cameraTimestampReused ? '#ff6b6b' : '';
        if (camReuseAllElem) camReuseAllElem.style.color = cameraTimestampReused ? '#ff6b6b' : '';

        // NEW: Unity timing diagnostics
        const unityTime = v.unity_time;
        const unityFrame = v.unity_frame_count;
        const unityDeltaTime = v.unity_delta_time;
        const unitySmoothDeltaTime = v.unity_smooth_delta_time;
        const unityUnscaledDeltaTime = v.unity_unscaled_delta_time;
        const unityTimeScale = v.unity_time_scale;

        updateField('vehicle-unity-time', unityTime !== undefined && unityTime !== null ? unityTime.toFixed(3) : '-');
        updateField('vehicle-unity-frame', unityFrame !== undefined && unityFrame !== null ? unityFrame : '-');
        updateField('vehicle-unity-delta-time', unityDeltaTime !== undefined && unityDeltaTime !== null ? unityDeltaTime.toFixed(4) : '-');
        updateField('vehicle-unity-smooth-delta-time', unitySmoothDeltaTime !== undefined && unitySmoothDeltaTime !== null ? unitySmoothDeltaTime.toFixed(4) : '-');
        updateField('vehicle-unity-unscaled-delta-time', unityUnscaledDeltaTime !== undefined && unityUnscaledDeltaTime !== null ? unityUnscaledDeltaTime.toFixed(4) : '-');
        updateField('vehicle-unity-time-scale', unityTimeScale !== undefined && unityTimeScale !== null ? unityTimeScale.toFixed(2) : '-');

        // Top-down sync diagnostics (frame-level)
        let topdownTrajDeltaMs = null;
        if (topdownTimestamp !== undefined && topdownTimestamp !== null &&
            trajectoryTimestamp !== undefined && trajectoryTimestamp !== null) {
            topdownTrajDeltaMs = (trajectoryTimestamp - topdownTimestamp) * 1000.0;
        }
        let topdownUnityDeltaMs = null;
        if (topdownTimestamp !== undefined && topdownTimestamp !== null &&
            unityTime !== undefined && unityTime !== null) {
            topdownUnityDeltaMs = (unityTime - topdownTimestamp) * 1000.0;
        }
        const fmtDelta = (vMs) => (vMs === null || !Number.isFinite(vMs)) ? '-' : `${vMs.toFixed(1)} ms`;
        updateField('vehicle-topdown-traj-delta', fmtDelta(topdownTrajDeltaMs));
        updateField('vehicle-topdown-unity-delta', fmtDelta(topdownUnityDeltaMs));

        let syncTrust = 'UNKNOWN';
        let syncTrustColor = '#888';
        if (topdownTrajDeltaMs !== null && Number.isFinite(topdownTrajDeltaMs)) {
            const absDt = Math.abs(topdownTrajDeltaMs);
            if (absDt <= 33.5) {
                syncTrust = 'GOOD';
                syncTrustColor = '#4caf50';
            } else if (absDt <= 66.5) {
                syncTrust = 'WARN';
                syncTrustColor = '#ffa500';
            } else {
                syncTrust = 'POOR';
                syncTrustColor = '#ff6b6b';
            }
        }
        updateField('vehicle-topdown-sync-trust', syncTrust);
        const trustElem = document.getElementById('vehicle-topdown-sync-trust');
        const trustAllElem = document.getElementById('all-vehicle-topdown-sync-trust');
        if (trustElem) trustElem.style.color = syncTrustColor;
        if (trustAllElem) trustAllElem.style.color = syncTrustColor;

        // Compute time gap only for sequential frames
        let unityTimeGapDisplay = '-';
        if (unityTime !== undefined && unityTime !== null) {
            if (this.lastUnityTime !== null && this.lastUnityFrameIndex === this.currentFrameIndex - 1) {
                const gap = unityTime - this.lastUnityTime;
                unityTimeGapDisplay = `${gap.toFixed(3)}s`;
            }
            this.lastUnityTime = unityTime;
            this.lastUnityFrameIndex = this.currentFrameIndex;
        }
        updateField('vehicle-unity-time-gap', unityTimeGapDisplay);

        // Flag when Unity frame/time is reused across sequential camera frames
        let unityFrameReusedDisplay = '-';
        let unityFrameReused = false;
        if (unityFrame !== undefined && unityFrame !== null) {
            if (this.lastUnityFrameCount !== null && this.lastUnityFrameIndex === this.currentFrameIndex - 1) {
                unityFrameReused = unityFrame === this.lastUnityFrameCount;
                unityFrameReusedDisplay = unityFrameReused ? 'Yes' : 'No';
            } else {
                unityFrameReusedDisplay = 'No';
            }
            this.lastUnityFrameCount = unityFrame;
        }
        updateField('vehicle-unity-frame-reused', unityFrameReusedDisplay);
        const reuseElem = document.getElementById('vehicle-unity-frame-reused');
        const reuseAllElem = document.getElementById('all-vehicle-unity-frame-reused');
        if (reuseElem) reuseElem.style.color = unityFrameReused ? '#ff6b6b' : '';
        if (reuseAllElem) reuseAllElem.style.color = unityFrameReused ? '#ff6b6b' : '';

        // Steering debug (Unity-side)
        const steeringInput = v.steering_input;
        const desiredSteerAngle = v.desired_steer_angle ?? v.steering_angle;
        const steeringAngleActual = v.steering_angle_actual;
        updateField(
            'vehicle-steering-input',
            steeringInput !== undefined && steeringInput !== null ? steeringInput.toFixed(3) : '-'
        );
        updateField(
            'vehicle-desired-steer-angle',
            desiredSteerAngle !== undefined && desiredSteerAngle !== null ? desiredSteerAngle.toFixed(3) : '-'
        );
        updateField(
            'vehicle-steering-angle-actual',
            steeringAngleActual !== undefined && steeringAngleActual !== null
                ? steeringAngleActual.toFixed(3)
                : '-'
        );
        
        // NEW: Display road center debug information
        const roadCenterAtCarX = v.road_center_at_car_x;
        const roadCenterAtCarY = v.road_center_at_car_y;
        const roadCenterAtCarZ = v.road_center_at_car_z;
        const roadCenterAtLookaheadX = v.road_center_at_lookahead_x;
        const roadCenterAtLookaheadY = v.road_center_at_lookahead_y;
        const roadCenterAtLookaheadZ = v.road_center_at_lookahead_z;
        const roadCenterReferenceT = v.road_center_reference_t;
        
        if (roadCenterAtCarX !== undefined && roadCenterAtCarX !== null) {
            updateField('vehicle-road-center-at-car-x', `${roadCenterAtCarX.toFixed(3)}m`);
            updateField('vehicle-road-center-at-car-y', `${roadCenterAtCarY.toFixed(3)}m`);
            updateField('vehicle-road-center-at-car-z', `${roadCenterAtCarZ.toFixed(3)}m`);
            updateField('vehicle-road-center-at-lookahead-x', `${roadCenterAtLookaheadX.toFixed(3)}m`);
            updateField('vehicle-road-center-at-lookahead-y', `${roadCenterAtLookaheadY.toFixed(3)}m`);
            updateField('vehicle-road-center-at-lookahead-z', `${roadCenterAtLookaheadZ.toFixed(3)}m`);
            updateField('vehicle-road-center-reference-t', `${roadCenterReferenceT.toFixed(4)}`);
            
            // Calculate offset (car position - road center)
            const carPos = v.position;
            if (carPos && carPos.length >= 3) {
                const offsetX = carPos[0] - roadCenterAtCarX;
                const offsetZ = carPos[2] - roadCenterAtCarZ;
                updateField('vehicle-road-center-offset-x', `${offsetX.toFixed(3)}m`);
                updateField('vehicle-road-center-offset-z', `${offsetZ.toFixed(3)}m`);
            } else {
                updateField('vehicle-road-center-offset-x', '- (no car position)');
                updateField('vehicle-road-center-offset-z', '- (no car position)');
            }
            
            // Also update "All Data" tab
            updateField('all-road-center-at-car-x', `${roadCenterAtCarX.toFixed(3)}m`);
            updateField('all-road-center-at-car-y', `${roadCenterAtCarY.toFixed(3)}m`);
            updateField('all-road-center-at-car-z', `${roadCenterAtCarZ.toFixed(3)}m`);
            updateField('all-road-center-at-lookahead-x', `${roadCenterAtLookaheadX.toFixed(3)}m`);
            updateField('all-road-center-at-lookahead-y', `${roadCenterAtLookaheadY.toFixed(3)}m`);
            updateField('all-road-center-at-lookahead-z', `${roadCenterAtLookaheadZ.toFixed(3)}m`);
            updateField('all-road-center-reference-t', `${roadCenterReferenceT.toFixed(4)}`);
            if (carPos && carPos.length >= 3) {
                const offsetX = carPos[0] - roadCenterAtCarX;
                const offsetZ = carPos[2] - roadCenterAtCarZ;
                updateField('all-road-center-offset-x', `${offsetX.toFixed(3)}m`);
                updateField('all-road-center-offset-z', `${offsetZ.toFixed(3)}m`);
            } else {
                updateField('all-road-center-offset-x', '- (no car position)');
                updateField('all-road-center-offset-z', '- (no car position)');
            }
        } else {
            updateField('vehicle-road-center-at-car-x', '- (not available)');
            updateField('vehicle-road-center-at-car-y', '- (not available)');
            updateField('vehicle-road-center-at-car-z', '- (not available)');
            updateField('vehicle-road-center-at-lookahead-x', '- (not available)');
            updateField('vehicle-road-center-at-lookahead-y', '- (not available)');
            updateField('vehicle-road-center-at-lookahead-z', '- (not available)');
            updateField('vehicle-road-center-reference-t', '- (not available)');
            updateField('vehicle-road-center-offset-x', '- (not available)');
            updateField('vehicle-road-center-offset-z', '- (not available)');
            updateField('all-road-center-at-car-x', '- (not available)');
            updateField('all-road-center-at-car-y', '- (not available)');
            updateField('all-road-center-at-car-z', '- (not available)');
            updateField('all-road-center-at-lookahead-x', '- (not available)');
            updateField('all-road-center-at-lookahead-y', '- (not available)');
            updateField('all-road-center-at-lookahead-z', '- (not available)');
            updateField('all-road-center-reference-t', '- (not available)');
            updateField('all-road-center-offset-x', '- (not available)');
            updateField('all-road-center-offset-z', '- (not available)');
        }
        
        if (v.position) {
            updateField('vehicle-x', `${v.position[0].toFixed(3)}m`);
            updateField('vehicle-y', `${v.position[1].toFixed(3)}m`);
            updateField('vehicle-z', `${v.position[2].toFixed(3)}m`);
        }
        if (v.speed !== undefined && v.speed !== null) {
            const speedMph = mpsToMph(v.speed);
            updateField('vehicle-speed', `${v.speed.toFixed(2)} m/s (${speedMph.toFixed(1)} mph)`);
        } else {
            updateField('vehicle-speed', '-');
        }
        const d = this.distanceFromStartSeries && this.currentFrameIndex < this.distanceFromStartSeries.length
            ? this.distanceFromStartSeries[this.currentFrameIndex]
            : null;
        updateField(
            'vehicle-distance-from-start',
            (d !== null && d !== undefined && Number.isFinite(d)) ? `${d.toFixed(2)} m` : '-'
        );
        const longitudinal = this.currentLongitudinalMetrics || { accel: null, jerk: null };
        updateField(
            'vehicle-accel',
            longitudinal.accel !== null && longitudinal.accel !== undefined
                ? `${longitudinal.accel.toFixed(2)} m/s²`
                : '-'
        );
        updateField(
            'vehicle-jerk',
            longitudinal.jerk !== null && longitudinal.jerk !== undefined
                ? `${longitudinal.jerk.toFixed(2)} m/s³`
                : '-'
        );
        if (v.speed_limit !== undefined && v.speed_limit !== null && v.speed_limit > 0) {
            const limitMph = mpsToMph(v.speed_limit);
            updateField('vehicle-speed-limit', `${v.speed_limit.toFixed(2)} m/s (${limitMph.toFixed(1)} mph)`);
        } else {
            updateField('vehicle-speed-limit', '-');
        }
        if (v.speed_limit_preview !== undefined && v.speed_limit_preview !== null && v.speed_limit_preview > 0) {
            const previewMph = mpsToMph(v.speed_limit_preview);
            updateField(
                'vehicle-speed-limit-preview',
                `${v.speed_limit_preview.toFixed(2)} m/s (${previewMph.toFixed(1)} mph)`
            );
        } else {
            updateField('vehicle-speed-limit-preview', '-');
        }
        if (v.speed_limit_preview_distance !== undefined && v.speed_limit_preview_distance !== null && v.speed_limit_preview_distance > 0) {
            updateField('vehicle-speed-limit-preview-distance', `${v.speed_limit_preview_distance.toFixed(1)} m`);
        } else {
            updateField('vehicle-speed-limit-preview-distance', '-');
        }
        
        // Calculate heading from quaternion
        if (v.rotation) {
            const q = v.rotation;
            const heading = Math.atan2(2*(q[3]*q[2] + q[0]*q[1]), 1 - 2*(q[1]*q[1] + q[2]*q[2]));
            updateField('vehicle-heading', `${(heading * 180 / Math.PI).toFixed(2)}°`);
        }
        updateField('vehicle-steering', v.steering_angle !== undefined ? `${(v.steering_angle * 180 / Math.PI).toFixed(2)}°` : '-');
    }

    updateGroundTruthData() {
        if (!this.currentFrameData || !this.currentFrameData.ground_truth) return;
        
        const gt = this.currentFrameData.ground_truth;
        const maxLateralAccel = 2.5;
        const minCurveSpeed = 2.0;
        const mpsToMph = (mps) => mps * 2.236936;
        
        const updateField = (id, value) => {
            const elem = document.getElementById(id);
            const allElem = document.getElementById('all-' + id);
            if (elem) elem.textContent = value;
            if (allElem) allElem.textContent = value;
        };
        
        const gt_left_lane_line_x = gt.left_lane_line_x !== undefined ? gt.left_lane_line_x : (gt.left_lane_x !== undefined ? gt.left_lane_x : undefined);  // Backward compatibility
        const gt_right_lane_line_x = gt.right_lane_line_x !== undefined ? gt.right_lane_line_x : (gt.right_lane_x !== undefined ? gt.right_lane_x : undefined);  // Backward compatibility
        
        updateField('gt-left-x', gt_left_lane_line_x !== undefined ? `${gt_left_lane_line_x.toFixed(3)}m` : '-');
        updateField('gt-right-x', gt_right_lane_line_x !== undefined ? `${gt_right_lane_line_x.toFixed(3)}m` : '-');
        updateField('gt-center-x', gt.lane_center_x !== undefined ? `${gt.lane_center_x.toFixed(3)}m` : '-');
        if (gt.path_curvature !== undefined && gt.path_curvature !== null) {
            updateField('gt-path-curvature', `${gt.path_curvature.toFixed(4)} 1/m`);
            if (Math.abs(gt.path_curvature) > 1e-6) {
                let curveSpeed = Math.sqrt(maxLateralAccel / Math.abs(gt.path_curvature));
                curveSpeed = Math.max(curveSpeed, minCurveSpeed);
                updateField(
                    'gt-curve-speed-limit',
                    `${curveSpeed.toFixed(2)} m/s (${mpsToMph(curveSpeed).toFixed(1)} mph)`
                );
            } else {
                updateField('gt-curve-speed-limit', '-');
            }
        } else {
            updateField('gt-path-curvature', '-');
            updateField('gt-curve-speed-limit', '-');
        }
        
        if (gt_left_lane_line_x !== undefined && gt_right_lane_line_x !== undefined) {
            const width = gt_right_lane_line_x - gt_left_lane_line_x;
            updateField('gt-width', `${width.toFixed(3)}m`);
            
            // Calculate errors if perception data available
            // NOTE: These are PERCEPTION errors (how much perception differs from ground truth)
            if (this.currentFrameData.perception) {
                const p = this.currentFrameData.perception;
                const p_left_lane_line_x = p.left_lane_line_x !== undefined ? p.left_lane_line_x : (p.left_lane_x !== undefined ? p.left_lane_x : undefined);  // Backward compatibility
                const p_right_lane_line_x = p.right_lane_line_x !== undefined ? p.right_lane_line_x : (p.right_lane_x !== undefined ? p.right_lane_x : undefined);  // Backward compatibility
                if (p_left_lane_line_x !== undefined && p_right_lane_line_x !== undefined) {
                    const detectedWidth = p_right_lane_line_x - p_left_lane_line_x;
                    const widthError = Math.abs(detectedWidth - width);
                    updateField('gt-width-error', `${widthError.toFixed(3)}m (perception vs GT)`);
                    
                    const detectedCenter = (p_left_lane_line_x + p_right_lane_line_x) / 2;
                    const centerError = Math.abs(detectedCenter - gt.lane_center_x);
                    updateField('gt-center-error', `${centerError.toFixed(3)}m (perception vs GT)`);
                } else {
                    updateField('gt-width-error', '- (no perception data)');
                    updateField('gt-center-error', '- (no perception data)');
                }
            } else {
                updateField('gt-width-error', '- (no perception data)');
                updateField('gt-center-error', '- (no perception data)');
            }
        }
    }


    updateOverlays() {
        this.overlayRenderer.clear();
        this.projectionDiagnostics = {};
        let plannerPathForMetrics = [];
        let oraclePathForMetrics = [];
        
        if (!this.currentFrameData) {
            console.log('[OVERLAY] No frame data available');
            return;
        }
        
        console.log('[OVERLAY] Updating overlays:', {
            hasPerception: !!this.currentFrameData.perception,
            hasGroundTruth: !!this.currentFrameData.ground_truth,
            hasTrajectory: !!this.currentFrameData.trajectory,
            toggleGroundTruth: document.getElementById('toggle-ground-truth').checked,
            toggleLanes: document.getElementById('toggle-lanes').checked
        });
        
        // NEW: Get actual 8m position from Unity camera calibration (if available)
        // This is the TRUE y pixel where 8m on the ground appears, from Unity's WorldToScreenPoint
        // CRITICAL: Cache the last valid value so black line persists across frames
        // (Unity only calculates it every 60 frames, but we want to show it on all frames)
        let y8mActual = null;
        if (this.currentFrameData.vehicle && 
            this.currentFrameData.vehicle.camera_8m_screen_y !== undefined) {
            const camera8mY = this.currentFrameData.vehicle.camera_8m_screen_y;
            if (camera8mY > 0) {
                // Valid value - use it and cache it
                y8mActual = camera8mY;
                this.lastValidY8m = camera8mY;  // Cache for future frames
            } else if (this.lastValidY8m !== undefined && this.lastValidY8m > 0) {
                // Invalid value (-1.0) but we have a cached valid value - use cached
                y8mActual = this.lastValidY8m;
            }
        } else if (this.lastValidY8m !== undefined && this.lastValidY8m > 0) {
            // No vehicle data but we have cached value - use cached
            y8mActual = this.lastValidY8m;
        }

        if (y8mActual !== null && y8mActual > 0) {
            const baseDistance = 8.0 * ((this.overlayRenderer.imageHeight - y8mActual) / this.overlayRenderer.imageHeight);
            this.overlayRenderer.setBaseDistance(baseDistance);
        }

        let yLookaheadActual = null;
        if (this.currentFrameData.vehicle &&
            this.currentFrameData.vehicle.camera_lookahead_screen_y !== undefined) {
            const lookaheadY = this.currentFrameData.vehicle.camera_lookahead_screen_y;
            if (lookaheadY > 0) {
                yLookaheadActual = lookaheadY;
                this.lastValidYLookahead = lookaheadY;
            } else if (this.lastValidYLookahead !== undefined && this.lastValidYLookahead > 0) {
                yLookaheadActual = this.lastValidYLookahead;
            }
        } else if (this.lastValidYLookahead !== undefined && this.lastValidYLookahead > 0) {
            yLookaheadActual = this.lastValidYLookahead;
        }
        
        // NEW: Update overlay renderer with Unity's actual horizontal FOV from recording
        // This ensures visualizer uses the same FOV that Unity actually uses (not hardcoded 110°)
        if (this.currentFrameData.vehicle && 
            this.currentFrameData.vehicle.camera_horizontal_fov !== undefined &&
            this.currentFrameData.vehicle.camera_horizontal_fov > 0) {
            const unityHorizontalFOV = this.currentFrameData.vehicle.camera_horizontal_fov;
            this.overlayRenderer.setCameraFov(unityHorizontalFOV);
        }

        const vehicle = this.currentFrameData?.vehicle || {};
        const streamFrontFrameDelta = Number(vehicle.stream_front_frame_id_delta);
        const streamFrontUnityDtMs = Number(vehicle.stream_front_unity_dt_ms);
        const syncPolicy = String(this.currentRecordingMeta?.metadata?.stream_sync_policy || '').toLowerCase();
        const frameDeltaRiskThreshold = (syncPolicy === 'latest') ? 3.0 : 2.0;
        const unityDtRiskThresholdMs = 20.0;
        const frameDeltaRisk = Number.isFinite(streamFrontFrameDelta) && streamFrontFrameDelta >= frameDeltaRiskThreshold;
        const unityDtRisk = Number.isFinite(streamFrontUnityDtMs) && Math.abs(streamFrontUnityDtMs) >= unityDtRiskThresholdMs;
        const cadenceRisk = frameDeltaRisk || unityDtRisk;
        const sync = this.currentFrameData?.sync || {};
        const syncTrajStatus = String(sync?.trajectory_alignment_status || '').toLowerCase();
        const syncControlStatus = String(sync?.control_alignment_status || '').toLowerCase();
        const hasSyncContract = Boolean(syncTrajStatus || syncControlStatus);
        const overlayAlignmentRisk = (
            syncTrajStatus === 'misaligned' ||
            syncTrajStatus === 'missing' ||
            syncControlStatus === 'misaligned' ||
            syncControlStatus === 'missing'
        );
        const overlayDegradeRisk = hasSyncContract ? overlayAlignmentRisk : (cadenceRisk || overlayAlignmentRisk);
        this.currentOverlaySnapRisk = overlayDegradeRisk;
        
        // Draw black reference line FIRST (shows where 8m actually appears)
        // This line is always drawn, using cached value if current frame doesn't have it
        if (y8mActual !== null && y8mActual > 0) {
            this.overlayRenderer.drawReferenceLine(y8mActual);
        } else {
            // Fallback: Draw at default 350px if no valid 8m position available
            this.overlayRenderer.drawReferenceLine(350);
        }
        
        // Draw segmentation mask (model output) before lane overlays
        if (document.getElementById('toggle-seg-mask').checked) {
            const p = this.currentFrameData.perception;
            const method = p && p.detection_method ? String(p.detection_method).toLowerCase() : '';
            if (method === 'segmentation' && p && p.segmentation_mask_png) {
                const maskData = this.overlayRenderer.drawSegmentationMaskFromPng(
                    p.segmentation_mask_png,
                    '#ffd400',
                    '#00bfff',
                    0.75,
                    () => this.updateOverlays()
                );
                if (maskData) {
                    this.overlayRenderer.drawSegmentationMaskImageData(maskData);
                }
            }
        }

        // Draw ground truth FIRST (green lines) - background layer
        if (document.getElementById('toggle-ground-truth').checked) {
            if (this.currentFrameData.ground_truth) {
                const gt = this.currentFrameData.ground_truth;
                // Extract ground truth lane line positions (with backward compatibility)
                const gt_left_lane_line_x = gt.left_lane_line_x !== undefined ? gt.left_lane_line_x : (gt.left_lane_x !== undefined ? gt.left_lane_x : undefined);
                const gt_right_lane_line_x = gt.right_lane_line_x !== undefined ? gt.right_lane_line_x : (gt.right_lane_x !== undefined ? gt.right_lane_x : undefined);
                
                if (gt_left_lane_line_x !== undefined && gt_right_lane_line_x !== undefined) {
                    // Draw ground truth in green (solid lines)
                    // FIXED: Always use tunable slider distance (user can override actual distance)
                    // The slider allows manual tuning to align green lines with red lines
                    // If user wants to use actual distance, they can set slider to match it
                    this.overlayRenderer.drawGroundTruth(gt_left_lane_line_x, gt_right_lane_line_x, this.groundTruthDistance, y8mActual);
                }
            }
        }
        
        // Draw fit points (points used for polynomial fitting) - NEW
        const fitPointsToggle = document.getElementById('toggle-fit-points');
        if (fitPointsToggle && fitPointsToggle.checked) {
            // First try to use fit_points from generate-debug cache (on-demand, always current code)
            let fit_points_left = null;
            let fit_points_right = null;
            
            if (this.generatedFitPointsCache && this.generatedFitPointsCache[this.currentFrameIndex]) {
                const cached = this.generatedFitPointsCache[this.currentFrameIndex];
                fit_points_left = cached.left;
                fit_points_right = cached.right;
            }
            
            // Fallback to stored fit_points from recording (if available)
            if (!fit_points_left && !fit_points_right && this.currentFrameData.perception) {
                const p = this.currentFrameData.perception;
                fit_points_left = p.fit_points_left;
                fit_points_right = p.fit_points_right;
            }
            
            // Draw points if available
            const cfgMin = Number.isFinite(Number(this.summaryConfig?.segmentation_fit_min_row_ratio))
                ? Number(this.summaryConfig.segmentation_fit_min_row_ratio)
                : 0.45;
            const cfgMax = Number.isFinite(Number(this.summaryConfig?.segmentation_fit_max_row_ratio))
                ? Number(this.summaryConfig.segmentation_fit_max_row_ratio)
                : 0.85;
            const leftFiltered = this.filterFitPointsToRoadRegion(fit_points_left, cfgMin, cfgMax);
            const rightFiltered = this.filterFitPointsToRoadRegion(fit_points_right, cfgMin, cfgMax);
            if (leftFiltered.length > 0) {
                this.overlayRenderer.drawFitPoints(leftFiltered, '#ff00ff', 3);  // Magenta for left lane
            }
            if (rightFiltered.length > 0) {
                this.overlayRenderer.drawFitPoints(rightFiltered, '#00ffff', 3);  // Cyan for right lane
            }
        }

        const segFitPointsToggle = document.getElementById('toggle-seg-fit-points');
        if (segFitPointsToggle && segFitPointsToggle.checked) {
            const p = this.currentFrameData.perception;
            const cfgMin = Number.isFinite(Number(this.summaryConfig?.segmentation_fit_min_row_ratio))
                ? Number(this.summaryConfig.segmentation_fit_min_row_ratio)
                : 0.45;
            const cfgMax = Number.isFinite(Number(this.summaryConfig?.segmentation_fit_max_row_ratio))
                ? Number(this.summaryConfig.segmentation_fit_max_row_ratio)
                : 0.85;
            const leftFiltered = this.filterFitPointsToRoadRegion(p?.fit_points_left, cfgMin, cfgMax);
            const rightFiltered = this.filterFitPointsToRoadRegion(p?.fit_points_right, cfgMin, cfgMax);
            if (leftFiltered.length > 0) {
                this.overlayRenderer.drawFitPoints(leftFiltered, '#ffd400', 3);  // Yellow for left lane
            }
            if (rightFiltered.length > 0) {
                this.overlayRenderer.drawFitPoints(rightFiltered, '#00bfff', 3);  // Blue for right lane
            }
        }
        
        // Draw perception ROI (CV mask region)
        const roiToggle = document.getElementById('toggle-perception-roi');
        if (roiToggle && roiToggle.checked) {
            const w = this.overlayRenderer.imageWidth;
            const h = this.overlayRenderer.imageHeight;
            const yTop = h * 0.18;
            const yBottom = h * 0.80;
            this.overlayRenderer.drawRoiRect(0, yTop, w, yBottom, '#ffd400', 2, true);
        }
        const segFitRoiToggle = document.getElementById('toggle-seg-fit-roi');
        if (segFitRoiToggle && segFitRoiToggle.checked) {
            const minRatio = this.summaryConfig?.segmentation_fit_min_row_ratio;
            const maxRatio = this.summaryConfig?.segmentation_fit_max_row_ratio;
            const h = this.overlayRenderer.imageHeight;
            const w = this.overlayRenderer.imageWidth;
            if (maxRatio !== undefined && maxRatio !== null && maxRatio > 0) {
                const yMin = h * ((minRatio !== undefined && minRatio !== null) ? minRatio : 0.45);
                const yMax = h * maxRatio;
                this.overlayRenderer.drawRoiRect(0, yMin, w, yMax, '#00e5ff', 2, true);
                this.overlayRenderer.drawHorizontalLine(yMin, '#00e5ff', 2, true);
                this.overlayRenderer.drawHorizontalLine(yMax, '#00e5ff', 2, true);
            }
        }

        // Draw detected lane lines SECOND (red vertical lines) - overlay layer
        // This allows easy comparison: Green = GT, Red = Detected positions
        // FIXED: If we have polynomial coefficients, use them to draw red lines at y=350px
        // This ensures red lines match orange curves at the black line
        if (document.getElementById('toggle-lanes').checked) {
            if (this.currentFrameData.perception) {
                const p = this.currentFrameData.perception;
                const isCv = (p.detection_method || '').toLowerCase() === 'cv';
                // Extract perception lane line positions (with backward compatibility)
                const left_lane_line_x = p.left_lane_line_x !== undefined ? p.left_lane_line_x : (p.left_lane_x !== undefined ? p.left_lane_x : undefined);
                const right_lane_line_x = p.right_lane_line_x !== undefined ? p.right_lane_line_x : (p.right_lane_x !== undefined ? p.right_lane_x : undefined);
                const perceptionDistance = this.currentFrameData.vehicle && this.currentFrameData.vehicle.ground_truth_lookahead_distance
                    ? this.currentFrameData.vehicle.ground_truth_lookahead_distance
                    : 8.0;
                
                // Prefer polynomial coefficients only for CV (matches orange curves exactly)
                if (isCv && p.lane_line_coefficients && Array.isArray(p.lane_line_coefficients) && p.lane_line_coefficients.length >= 2) {
                    // NEW: Use actual 8m position so red lines align with black line
                    this.overlayRenderer.drawLaneLinesFromCoefficients(p.lane_line_coefficients, 350, '#ff0000', y8mActual);
                } else if (left_lane_line_x !== undefined && right_lane_line_x !== undefined) {
                    // Fallback: Use vehicle coords if coefficients not available
                    // FIXED: Keep red lines at fixed 8.0m (don't move visually)
                    // Width calculation will be updated separately using tunable distance
                    this.overlayRenderer.drawLaneLinesFromVehicleCoords(
                        left_lane_line_x, right_lane_line_x, perceptionDistance, '#ff0000', yLookaheadActual || y8mActual
                    );
                }

                // Draw perceived center line (cyan) when available, regardless of curves/lines source
                const perceptionCenterX = p.perception_lane_center_x !== undefined
                    ? p.perception_lane_center_x
                    : (
                        left_lane_line_x !== undefined && right_lane_line_x !== undefined
                            ? (left_lane_line_x + right_lane_line_x) / 2.0
                            : undefined
                    );
                const perceptionLookahead = p.perception_lookahead_distance
                    ? p.perception_lookahead_distance
                    : perceptionDistance;
                if (perceptionCenterX !== undefined) {
                    this.overlayRenderer.drawCenterLineFromVehicleCoords(
                        perceptionCenterX, perceptionLookahead, '#00ffff', yLookaheadActual || y8mActual
                    );
                }
            }
        }
        
        // Draw trajectory (if available)
        if (document.getElementById('toggle-trajectory').checked) {
            if (this.currentFrameData.trajectory && this.currentFrameData.trajectory.trajectory_points) {
                let trajectoryMinY = Math.floor(this.overlayRenderer.imageHeight * 0.2);
                if (y8mActual !== null && y8mActual > 0) {
                    const pxPerMeter = (this.overlayRenderer.imageHeight - y8mActual) / 8.0;
                    if (pxPerMeter > 0) {
                        const metersToShow = 17.5;
                        const yForMeters = this.overlayRenderer.imageHeight - (metersToShow * pxPerMeter);
                        trajectoryMinY = Math.max(0, Math.floor(yForMeters));
                    }
                }

                const rawTrajPoints = this.currentFrameData.trajectory.trajectory_points;
                const trajPoints = this.getDisplayTrajectoryPoints(rawTrajPoints);
                const mainRenderTraj = this.toForwardMonotonicPath(trajPoints);
                plannerPathForMetrics = mainRenderTraj;
                this.projectionDiagnostics.source_turn_sign = this.computeTurnSign(mainRenderTraj, 'x', 'y');
                if (mainRenderTraj.length) {
                    const projected = this.projectTrajectoryToImage(mainRenderTraj);
                    if (projected && projected.length > 1) {
                        const clipped = [];
                        let prev = null;
                        for (const pt of projected) {
                            if (!Number.isFinite(pt?.x) || !Number.isFinite(pt?.y)) continue;
                            if (pt.y < trajectoryMinY || pt.y > this.overlayRenderer.imageHeight) continue;
                            if (prev) {
                                const jumpPx = Math.hypot(pt.x - prev.x, pt.y - prev.y);
                                if (jumpPx > 140) {
                                    prev = pt;
                                    continue;
                                }
                            }
                            clipped.push(pt);
                            prev = pt;
                        }
                        if (overlayDegradeRisk) {
                            this.overlayRenderer.drawImagePoints(clipped, '#ff00ff', 2);
                        } else {
                            this.overlayRenderer.drawImagePath(clipped, '#ff00ff', 2);
                        }
                        const diag = projected._diag || {};
                        this.projectionDiagnostics.main_first_visible_src_y_m = diag.main_first_visible_src_y_m;
                        this.projectionDiagnostics.main_mirror_sanity = diag.main_mirror_sanity || '-';
                        this.projectionDiagnostics.main_nearfield_blend = diag.main_nearfield_blend || '-';
                        this.projectionDiagnostics.main_nearfield_y_offset_m = diag.main_nearfield_y_offset_m;
                        this.projectionDiagnostics.main_nearfield_blend_distance_m = diag.main_nearfield_blend_distance_m;
                        this.projectionDiagnostics.main_turn_sign = this.computeTurnSign(projected, 'srcX', 'srcY');
                    } else {
                        const sortedPoints = [...mainRenderTraj]
                            .filter(point => point.y >= 0)
                            .sort((a, b) => (a.y || 0) - (b.y || 0));
                        this.overlayRenderer.drawTrajectory(
                            sortedPoints,
                            '#ff00ff',
                            2,
                            trajectoryMinY,
                            this.overlayRenderer.imageHeight
                        );
                        this.projectionDiagnostics.main_first_visible_src_y_m = null;
                        this.projectionDiagnostics.main_mirror_sanity = 'fallback';
                        this.projectionDiagnostics.main_nearfield_blend = this.projectionNearFieldBlendEnabled ? 'on' : 'off';
                        this.projectionDiagnostics.main_nearfield_y_offset_m = Number(this.projectionNearFieldGroundYOffsetMeters);
                        this.projectionDiagnostics.main_nearfield_blend_distance_m = Number(this.projectionNearFieldBlendDistanceMeters);
                        this.projectionDiagnostics.main_turn_sign = this.computeTurnSign(mainRenderTraj, 'x', 'y');
                    }
                }
            }
        }

        // Draw oracle trajectory (display only)
        if (document.getElementById('toggle-oracle-trajectory')?.checked) {
            const oracleScreenPoints = this.currentFrameData?.vehicle?.oracle_trajectory_screen_points;
            let drewScreenOracle = false;
            if (Array.isArray(oracleScreenPoints) && oracleScreenPoints.length > 1) {
                const validOracleScreenPoints = oracleScreenPoints
                    .map((p) => ({ x: Number(p?.x), y: Number(p?.y), valid: Boolean(p?.valid) }))
                    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && p.valid);
                if (validOracleScreenPoints.length > 1) {
                    if (overlayDegradeRisk) {
                        this.overlayRenderer.drawImagePoints(validOracleScreenPoints, '#66ff66', 2);
                    } else {
                        this.overlayRenderer.drawImagePath(validOracleScreenPoints, '#66ff66', 2);
                    }
                    drewScreenOracle = true;
                }
            }
            const oracleWorldPoints = this.currentFrameData?.vehicle?.oracle_trajectory_world_points;
            let drewWorldOracle = false;
            if (!drewScreenOracle && Array.isArray(oracleWorldPoints) && oracleWorldPoints.length > 1) {
                const projectedWorldOracle = this.projectWorldPointsToImage(oracleWorldPoints);
                if (projectedWorldOracle && projectedWorldOracle.length > 1) {
                    if (overlayDegradeRisk) {
                        this.overlayRenderer.drawImagePoints(projectedWorldOracle, '#66ff66', 2);
                    } else {
                        this.overlayRenderer.drawImagePath(projectedWorldOracle, '#66ff66', 2);
                    }
                    drewWorldOracle = true;
                }
            }
            const oraclePoints = this.currentFrameData?.trajectory?.oracle_points;
            const renderOracle = (Array.isArray(oraclePoints) && oraclePoints.length > 0)
                ? this.toForwardMonotonicPath(oraclePoints)
                : [];
            if (renderOracle.length > 0) {
                oraclePathForMetrics = renderOracle;
            }
            if (!drewScreenOracle && !drewWorldOracle && renderOracle.length > 0) {
                const projectedOracle = this.projectTrajectoryToImage(renderOracle);
                if (projectedOracle && projectedOracle.length > 1) {
                    if (overlayDegradeRisk) {
                        this.overlayRenderer.drawImagePoints(projectedOracle, '#66ff66', 2);
                    } else {
                        this.overlayRenderer.drawImagePath(projectedOracle, '#66ff66', 2);
                    }
                } else {
                    const sortedOracle = [...renderOracle]
                        .filter(point => point.y >= 0)
                        .sort((a, b) => (a.y || 0) - (b.y || 0));
                    this.overlayRenderer.drawTrajectory(
                        sortedOracle,
                        '#66ff66',
                        2,
                        null,
                        this.overlayRenderer.imageHeight
                    );
                }
            }
        }
        const lateralMetrics = this.computeLateralErrorMetrics(plannerPathForMetrics, oraclePathForMetrics);
        this.projectionDiagnostics.lateral_error_5m = lateralMetrics.err5m;
        this.projectionDiagnostics.lateral_error_10m = lateralMetrics.err10m;
        this.projectionDiagnostics.lateral_error_15m = lateralMetrics.err15m;
        this.projectionDiagnostics.lateral_error_source = lateralMetrics.source;
        const turnMismatch = this.computeTurnStrengthMismatchMetrics(plannerPathForMetrics, oraclePathForMetrics);
        this.projectionDiagnostics.heading_delta_5m_deg = turnMismatch.headingDelta5mDeg;
        this.projectionDiagnostics.heading_delta_10m_deg = turnMismatch.headingDelta10mDeg;
        this.projectionDiagnostics.heading_delta_15m_deg = turnMismatch.headingDelta15mDeg;
        this.projectionDiagnostics.curvature_ratio_5m = turnMismatch.curvatureRatio5m;
        this.projectionDiagnostics.curvature_ratio_10m = turnMismatch.curvatureRatio10m;
        this.projectionDiagnostics.curvature_ratio_15m = turnMismatch.curvatureRatio15m;
        this.projectionDiagnostics.turn_mismatch_source = turnMismatch.source;
        const trajWaterfall = this.computeTrajectorySuppressionWaterfall(plannerPathForMetrics, oraclePathForMetrics, turnMismatch);
        this.projectionDiagnostics.traj_heading_zero_gate = trajWaterfall.headingZeroGate;
        this.projectionDiagnostics.traj_small_heading_gate = trajWaterfall.smallHeadingGate;
        this.projectionDiagnostics.traj_multilookahead_active = trajWaterfall.multiLookaheadActive;
        this.projectionDiagnostics.traj_x_clip_count = trajWaterfall.xClipCount;
        this.projectionDiagnostics.traj_heavy_x_clipping = trajWaterfall.heavyXClipping;
        this.projectionDiagnostics.traj_preclip_abs_max = trajWaterfall.preclipXAbsMax;
        this.projectionDiagnostics.traj_preclip_abs_p95 = trajWaterfall.preclipXAbsP95;
        this.projectionDiagnostics.traj_front_frame_delta = trajWaterfall.frontFrameIdDelta;
        this.projectionDiagnostics.traj_front_unity_dt_ms = trajWaterfall.frontUnityDtMs;
        this.projectionDiagnostics.traj_overlay_snap_risk = trajWaterfall.overlaySnapRisk;
        this.projectionDiagnostics.traj_contract_misaligned_risk = trajWaterfall.contractMisalignedRisk;
        this.projectionDiagnostics.traj_cadence_risk = trajWaterfall.cadenceRisk;
        this.projectionDiagnostics.traj_cadence_policy = trajWaterfall.cadencePolicy;
        this.projectionDiagnostics.traj_cadence_frame_delta_threshold = trajWaterfall.cadenceFrameDeltaRiskThreshold;
        this.projectionDiagnostics.traj_cadence_unity_dt_threshold_ms = trajWaterfall.cadenceUnityDtRiskThresholdMs;
        this.projectionDiagnostics.traj_control_curv_ratio_10m = trajWaterfall.controlCurvVsOracleRatio10m;
        this.projectionDiagnostics.traj_underturn_10m_flag = trajWaterfall.underTurnFlag10m;
        this.projectionDiagnostics.traj_waterfall_source = trajWaterfall.source;
        this.projectionDiagnostics.sync_overall_status = trajWaterfall.syncOverallStatus;
        this.projectionDiagnostics.sync_traj_status = trajWaterfall.syncTrajStatus;
        this.projectionDiagnostics.sync_traj_reason = trajWaterfall.syncTrajReason;
        this.projectionDiagnostics.sync_control_status = trajWaterfall.syncControlStatus;
        this.projectionDiagnostics.sync_control_reason = trajWaterfall.syncControlReason;
        this.projectionDiagnostics.sync_window_ms = trajWaterfall.syncWindowMs;
        this.projectionDiagnostics.sync_dt_cam_traj_ms = trajWaterfall.syncDtCamTrajMs;
        this.projectionDiagnostics.sync_dt_cam_control_ms = trajWaterfall.syncDtCamControlMs;
        this.projectionDiagnostics.sync_dt_cam_vehicle_ms = trajWaterfall.syncDtCamVehicleMs;
        this.projectionDiagnostics.clock_front_ts_reused = trajWaterfall.frontTsReused;
        this.projectionDiagnostics.clock_front_ts_nonmonotonic = trajWaterfall.frontTsNonMonotonic;
        this.projectionDiagnostics.clock_front_id_reused = trajWaterfall.frontIdReused;
        this.projectionDiagnostics.clock_front_negative_delta = trajWaterfall.frontNegativeDelta;
        this.projectionDiagnostics.clock_front_jump = trajWaterfall.frontClockJump;
        this.projectionDiagnostics.clock_top_ts_reused = trajWaterfall.topTsReused;
        this.projectionDiagnostics.clock_top_ts_nonmonotonic = trajWaterfall.topTsNonMonotonic;
        this.projectionDiagnostics.clock_top_id_reused = trajWaterfall.topIdReused;
        this.projectionDiagnostics.clock_top_negative_delta = trajWaterfall.topNegativeDelta;
        this.projectionDiagnostics.clock_top_jump = trajWaterfall.topClockJump;
        this.projectionDiagnostics.main_nearfield_blend = this.projectionNearFieldBlendEnabled ? 'on' : 'off';
        this.projectionDiagnostics.main_nearfield_y_offset_m = Number(this.projectionNearFieldGroundYOffsetMeters);
        this.projectionDiagnostics.main_nearfield_blend_distance_m = Number(this.projectionNearFieldBlendDistanceMeters);
        const rightFidDiag = this.getRightLaneFiducialDiagnostics();
        this.projectionDiagnostics.right_fiducial_err_5m = rightFidDiag.err5m;
        this.projectionDiagnostics.right_fiducial_err_10m = rightFidDiag.err10m;
        this.projectionDiagnostics.right_fiducial_err_15m = rightFidDiag.err15m;
        this.projectionDiagnostics.right_fiducial_err_mean = rightFidDiag.meanErr;
        this.projectionDiagnostics.right_fiducial_err_max = rightFidDiag.maxErr;
        this.projectionDiagnostics.right_fiducial_source = rightFidDiag.source;
        if (document.getElementById('toggle-right-lane-fiducials')?.checked) {
            this.drawRightLaneFiducialsOverlay(rightFidDiag);
        }
        this.drawMainDistanceScale();
        
        // Draw reference point
        if (document.getElementById('toggle-reference').checked) {
            if (this.currentFrameData.trajectory && this.currentFrameData.trajectory.reference_point) {
                // NEW: Pass actual 8m position so red dot aligns with black line
                this.overlayRenderer.drawReferencePoint(
                    this.currentFrameData.trajectory.reference_point,
                    '#ff0000',
                    8,
                    y8mActual  // Use actual 8m position from Unity
                );
                if (this.currentFrameData.trajectory.reference_point_raw) {
                    this.overlayRenderer.drawReferencePoint(
                        this.currentFrameData.trajectory.reference_point_raw,
                        '#ffa500',
                        6,
                        y8mActual
                    );
                }
            }
        }

        this.updateTopdownOverlay();
        this.updateProjectionData();
    }

    async updateDebugOverlays() {
        const canvas = document.getElementById('debug-overlay-canvas');
        if (!canvas) {
            return;
        }
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!this.currentFrameData) {
            console.warn('No current frame data available');
            return;
        }
        
        const frameId = this.dataLoader.formatFrameId(this.currentFrameIndex);
        
        // Load and display debug images based on toggles
        const types = [
            { id: 'toggle-combined', name: 'combined' },
            { id: 'toggle-edges', name: 'edges' },
            { id: 'toggle-yellow-mask', name: 'yellow_mask' },
            { id: 'toggle-histogram', name: 'histogram', prefix: 'line_histogram_' }
        ];
        
        // Collect all checked types
        const checkedTypes = types.filter(type => {
            const elem = document.getElementById(type.id);
            return elem && elem.checked;
        });
        
        if (checkedTypes.length === 0) return; // Nothing to show
        
        // Check if we have on-demand generated debug images
        const cacheKey = `${this.currentRecording}_${this.currentFrameIndex}`;
        const hasGenerated = this.generatedDebugCache && this.generatedDebugCache[cacheKey];
        
        // Load all images first, then composite them
        const images = [];
        const missingImages = [];
        for (const type of checkedTypes) {
            try {
                let imageUrl = null;
                
                // First try on-demand generated images
                if (hasGenerated && type.name !== 'histogram') {
                    const generated = this.generatedDebugCache[cacheKey];
                    if (generated[type.name]) {
                        imageUrl = generated[type.name];
                    }
                }
                
                // Fallback to saved debug images
                if (!imageUrl) {
                    imageUrl = await this.dataLoader.loadDebugImage(this.currentFrameIndex, type.name);
                }
                
                if (imageUrl) {
                    const img = new Image();
                    await new Promise((resolve, reject) => {
                        img.onload = resolve;
                        img.onerror = () => {
                            console.warn(`Debug image ${type.name} failed to load for frame ${this.currentFrameIndex}`);
                            missingImages.push(type.name);
                            resolve(); // Don't reject, just continue
                        };
                        img.src = imageUrl;
                    });
                    if (img.complete && img.naturalWidth > 0) {
                        images.push({img, type: type.name});
                    }
                } else {
                    missingImages.push(type.name);
                }
            } catch (error) {
                console.error(`Error loading debug image ${type.name}:`, error);
                missingImages.push(type.name);
            }
        }
        
        // Show message if images are missing (only if not using generated images)
        if (missingImages.length > 0 && checkedTypes.length > 0 && !hasGenerated) {
            const nearestFrame = Math.floor(this.currentFrameIndex / 30) * 30;
            console.info(`Debug images not available for frame ${this.currentFrameIndex}. ` +
                        `Debug images are saved every 30 frames. Nearest frame with debug images: ${nearestFrame}. ` +
                        `Click "Generate Debug Overlays" to create them on-demand.`);
        }
        
        // Composite all images (draw them on top of each other)
        // If only one image, draw it directly; otherwise blend them
        if (images.length === 1) {
            ctx.globalAlpha = 1.0;
            ctx.drawImage(images[0].img, 0, 0, canvas.width, canvas.height);
        } else if (images.length > 1) {
            // Draw first image at full opacity
            ctx.globalAlpha = 1.0;
            ctx.drawImage(images[0].img, 0, 0, canvas.width, canvas.height);
            // Draw remaining images with reduced opacity for blending
            ctx.globalAlpha = 0.5;
            for (let i = 1; i < images.length; i++) {
                ctx.drawImage(images[i].img, 0, 0, canvas.width, canvas.height);
            }
            ctx.globalAlpha = 1.0; // Reset
        } else {
            console.warn('No debug images to display');
        }
    }

    filterFitPointsToRoadRegion(points, minRowRatio = 0.45, maxRowRatio = 0.85) {
        if (!Array.isArray(points) || points.length === 0) {
            return [];
        }
        const minY = this.overlayRenderer.imageHeight * minRowRatio;
        const maxY = this.overlayRenderer.imageHeight * maxRowRatio;
        return points.filter((pt) =>
            Array.isArray(pt) &&
            pt.length >= 2 &&
            Number.isFinite(pt[0]) &&
            Number.isFinite(pt[1]) &&
            pt[1] >= minY &&
            pt[1] <= maxY
        );
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.tab === tabName) {
                btn.classList.add('active');
            }
        });
        
        // Update tab panes
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        // Handle "all-data" tab name mapping
        const tabId = tabName === 'all-data' ? 'all-data-tab' : `${tabName}-tab`;
        const targetPane = document.getElementById(tabId);
        if (targetPane) {
            targetPane.classList.add('active');
        }
        
        // Load data for tabs that need it when switched to
        if (this.currentRecording) {
            if (tabName === 'summary') {
                this.loadSummary();
            } else if (tabName === 'issues') {
                this.loadIssues();
            } else if (tabName === 'diagnostics') {
                this.loadDiagnostics();
            }
        }
    }

    async jumpToFrame(frameIndex) {
        // Alias for goToFrame - used by click handlers in HTML
        await this.goToFrame(frameIndex);
    }

    prevFrame() {
        if (this.currentFrameIndex > 0) {
            this.goToFrame(this.currentFrameIndex - 1);
        }
    }

    nextFrame() {
        if (this.currentFrameIndex < this.frameCount - 1) {
            this.goToFrame(this.currentFrameIndex + 1);
        }
    }

    togglePlay() {
        if (this.isPlaying) {
            this.stopPlayback();
        } else {
            this.startPlayback();
        }
    }

    startPlayback() {
        this.isPlaying = true;
        document.getElementById('play-pause-btn').textContent = '⏸ Pause';
        const interval = 1000 / (30 * this.playSpeed); // 30 FPS base
        this.playInterval = setInterval(() => {
            if (this.currentFrameIndex < this.frameCount - 1) {
                this.nextFrame();
            } else {
                this.stopPlayback();
            }
        }, interval);
    }

    stopPlayback() {
        this.isPlaying = false;
        document.getElementById('play-pause-btn').textContent = '▶ Play';
        if (this.playInterval) {
            clearInterval(this.playInterval);
            this.playInterval = null;
        }
    }

    exportFrame() {
        // Create a composite canvas with all overlays
        const composite = document.createElement('canvas');
        composite.width = 640;
        composite.height = 480;
        const ctx = composite.getContext('2d');
        
        // Draw camera frame
        const cameraCanvas = document.getElementById('camera-canvas');
        ctx.drawImage(cameraCanvas, 0, 0);
        
        // Draw overlays
        const overlayCanvas = document.getElementById('overlay-canvas');
        ctx.drawImage(overlayCanvas, 0, 0);
        
        // Draw debug overlays
        const debugCanvas = document.getElementById('debug-overlay-canvas');
        ctx.globalAlpha = parseFloat(document.getElementById('debug-overlay-canvas').style.opacity || 0.5);
        ctx.drawImage(debugCanvas, 0, 0);
        ctx.globalAlpha = 1.0;
        
        // Download
        composite.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `frame_${String(this.currentFrameIndex).padStart(6, '0')}.png`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    async exportVideo() {
        alert('Video export not yet implemented. This would require server-side processing with FFmpeg.');
        // TODO: Implement video export using MediaRecorder API or server-side FFmpeg
    }
    
    async analyzePolynomialFitting() {
        if (!this.currentRecording || this.currentFrameIndex === undefined) {
            alert('Please load a recording and select a frame first.');
            return;
        }
        
        const btn = document.getElementById('polynomial-inspector-btn');
        const resultsDiv = document.getElementById('polynomial-inspector-results');
        const contentDiv = document.getElementById('polynomial-inspector-content');
        
        // Show loading state
        btn.disabled = true;
        btn.textContent = 'Analyzing...';
        resultsDiv.style.display = 'block';
        contentDiv.innerHTML = '<div style="color: #888;">Loading analysis...</div>';
        
        try {
            const response = await fetch(
                `/api/recording/${this.currentRecording}/frame/${this.currentFrameIndex}/polynomial-analysis`
            );
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }
            
            const analysis = await response.json();
            
            // Format and display results
            let html = '<div style="line-height: 1.6;">';
            
            html += `<h4 style="margin-top: 0; color: #4a90e2;">Frame ${analysis.frame_index} Analysis</h4>`;
            
            // Show comparison: Recorded vs Re-run
            if (analysis.recorded) {
                html += `<div style="margin-bottom: 1rem; padding: 0.75rem; background: #2a2a2a; border-radius: 4px; border-left: 3px solid #4a90e2;">`;
                html += `<strong style="color: #4a90e2;">📼 RECORDED (Original Run):</strong><br/>`;
                html += `<span style="font-size: 0.9rem;">Lanes Detected: <strong>${analysis.recorded.num_lanes_detected || 0}</strong></span>`;
                if (analysis.recorded.using_stale_data) {
                    html += ` | <span style="color: #ff6b6b;">⚠️ Using Stale Data</span>`;
                    if (analysis.recorded.stale_data_reason) {
                        html += ` (${analysis.recorded.stale_data_reason})`;
                    }
                }
                html += `</div>`;
            }
            
            html += `<div style="margin-bottom: 1rem; padding: 0.75rem; background: #2a2a2a; border-radius: 4px; border-left: 3px solid #ffa500;">`;
            html += `<strong style="color: #ffa500;">🔄 RE-RUN (Current Code):</strong><br/>`;
            html += `<span style="font-size: 0.9rem;">Lanes Detected: <strong>${analysis.num_lanes_detected || 0}</strong></span>`;
            if (analysis.recorded && analysis.recorded.num_lanes_detected !== analysis.num_lanes_detected) {
                html += ` <span style="color: #ff6b6b;">(Different from recorded!)</span>`;
            }
            html += `</div>`;
            
            if (analysis.debug_info) {
                const di = analysis.debug_info;
                html += `<p><strong>Line Segments:</strong> ${di.num_lines_detected || 0} total `;
                html += `(${di.left_lines_count || 0} left, ${di.right_lines_count || 0} right)</p>`;
                
                // Show polynomial evaluations
                for (const lane of ['left', 'right']) {
                    const laneIdx = lane === 'left' ? 0 : 1;
                    const coeffs = analysis.lane_coefficients[laneIdx];
                    
                    if (coeffs) {
                        html += `<div style="margin-top: 1rem; padding: 0.75rem; background: #1a1a1a; border-radius: 4px;">`;
                        html += `<strong style="color: #4a90e2;">${lane.toUpperCase()} LANE</strong>`;
                        html += `<div style="margin-top: 0.5rem; font-family: monospace; font-size: 0.85rem;">`;
                        html += `Coefficients: a=${coeffs[0].toFixed(6)}, b=${coeffs[1].toFixed(6)}, c=${coeffs[2].toFixed(6)}`;
                        html += `</div>`;
                        
                        // Point information
                        if (di[`${lane}_points_count`]) {
                            html += `<div style="margin-top: 0.5rem; font-size: 0.9rem;">`;
                            html += `Points used: ${di[`${lane}_points_count`]}`;
                            if (di[`${lane}_y_range`]) {
                                const yRange = di[`${lane}_y_range`];
                                const imageHeight = 480; // Assuming standard height
                                const extrapolation = imageHeight - yRange[1];
                                html += ` | Y range: [${yRange[0].toFixed(0)}, ${yRange[1].toFixed(0)}]px`;
                                if (extrapolation > 0) {
                                    const color = extrapolation > 100 ? '#ff6b6b' : '#ffa500';
                                    html += ` | <span style="color: ${color};">Extrapolation: ${extrapolation.toFixed(0)}px</span>`;
                                }
                            }
                            html += `</div>`;
                        }
                        
                        // Polynomial evaluation
                        if (di[`${lane}_polynomial_evaluation`]) {
                            html += `<div style="margin-top: 0.5rem; font-size: 0.85rem;">`;
                            html += `<strong>Polynomial Evaluation:</strong><br/>`;
                            di[`${lane}_polynomial_evaluation`].forEach(evaluation => {
                                const status = evaluation.in_bounds ? '✅' : '❌';
                                html += `  y=${evaluation.y.toString().padStart(3)}px: x=${evaluation.x.toFixed(1)}px ${status}<br/>`;
                            });
                            html += `</div>`;
                        }
                        
                        html += `</div>`;
                    }
                }
                
                // Validation failures - only show if there are actual failures
                const hasFailures = di.validation_failures && Object.keys(di.validation_failures).length > 0;
                const failureCount = hasFailures ? Object.values(di.validation_failures).reduce((sum, failures) => sum + failures.length, 0) : 0;
                
                if (failureCount > 0) {
                    html += `<div style="margin-top: 1rem; padding: 0.75rem; background: #3a1a1a; border-radius: 4px; color: #ff6b6b;">`;
                    html += `<strong>⚠️ Validation Failures (Why Re-run Rejected):</strong><br/>`;
                    for (const [lane, failures] of Object.entries(di.validation_failures)) {
                        if (failures.length > 0) {
                            html += `<strong>${lane}:</strong> ${failures.join(', ')}<br/>`;
                        }
                    }
                    html += `<div style="margin-top: 0.5rem; font-size: 0.85rem; color: #aaa;">`;
                    html += `💡 <em>These detections were rejected by current validation rules. `;
                    html += `The original recording may have used these (with stale data flag) or had different validation.</em></div>`;
                    html += `</div>`;
                } else {
                    // Show success message when no failures
                    html += `<div style="margin-top: 1rem; padding: 0.75rem; background: #1a3a1a; border-radius: 4px; color: #4caf50;">`;
                    html += `<strong>✅ Validation Passed:</strong><br/>`;
                    html += `<span style="font-size: 0.9rem;">All detections passed validation rules. `;
                    html += `Both lanes were accepted by current code (same as recorded).</span>`;
                    html += `</div>`;
                }
            }
            
            // Show recorded coefficients if available and different
            if (analysis.recorded && analysis.recorded.lane_coefficients) {
                html += `<div style="margin-top: 1rem; padding: 0.75rem; background: #1a2a1a; border-radius: 4px; border-left: 3px solid #4a90e2;">`;
                html += `<strong style="color: #4a90e2;">📼 Recorded Coefficients:</strong><br/>`;
                for (let i = 0; i < 2; i++) {
                    const laneName = i === 0 ? 'left' : 'right';
                    const recordedCoeffs = analysis.recorded.lane_coefficients[i];
                    if (recordedCoeffs) {
                        html += `<div style="margin-top: 0.5rem; font-family: monospace; font-size: 0.85rem;">`;
                        html += `<strong>${laneName.toUpperCase()}:</strong> a=${recordedCoeffs[0].toFixed(6)}, b=${recordedCoeffs[1].toFixed(6)}, c=${recordedCoeffs[2].toFixed(6)}`;
                        html += `</div>`;
                    }
                }
                html += `</div>`;
            }
            
            // NEW: Full System Validation (what av_stack.py would do)
            if (analysis.full_system_validation) {
                const fsv = analysis.full_system_validation;
                const borderColor = fsv.would_reject ? '#ff6b6b' : '#4caf50';
                const bgColor = fsv.would_reject ? '#3a1a1a' : '#1a3a1a';
                const textColor = fsv.would_reject ? '#ff6b6b' : '#4caf50';
                
                html += `<div style="margin-top: 1.5rem; padding: 0.75rem; background: ${bgColor}; border-radius: 4px; border-left: 3px solid ${borderColor};">`;
                html += `<strong style="color: ${textColor}; font-size: 1.1rem;">${fsv.would_reject ? '❌' : '✅'} FULL SYSTEM VALIDATION (av_stack.py):</strong><br/>`;
                html += `<span style="font-size: 0.9rem;">${fsv.would_reject ? 'Would REJECT' : 'Would ACCEPT'} these detections</span><br/><br/>`;
                
                // Polynomial x-value validation
                if (fsv.polynomial_x_validation) {
                    html += `<strong style="color: #ffa500;">1. Polynomial X-Value Validation:</strong><br/>`;
                    for (const [lane, validation] of Object.entries(fsv.polynomial_x_validation)) {
                        const status = validation.passed ? '✅' : '❌';
                        const color = validation.passed ? '#4caf50' : '#ff6b6b';
                        html += `<span style="color: ${color};">  ${status} ${lane.toUpperCase()} lane: `;
                        if (validation.passed) {
                            html += `Passed (all x-values within reasonable range)</span><br/>`;
                        } else {
                            html += `FAILED - Extreme x-values detected:</span><br/>`;
                            validation.extreme_positions.forEach(pos => {
                                html += `<span style="margin-left: 1rem; font-size: 0.85rem; color: #ff6b6b;">`;
                                html += `  • y=${pos.y}px: ${pos.reason}</span><br/>`;
                            });
                        }
                    }
                    html += `<br/>`;
                }
                
                // Lane width validation
                if (fsv.lane_width_validation) {
                    const widthVal = fsv.lane_width_validation;
                    html += `<strong style="color: #ffa500;">2. Lane Width Validation:</strong><br/>`;
                    if (widthVal.passed !== null) {
                        const status = widthVal.passed ? '✅' : '❌';
                        const color = widthVal.passed ? '#4caf50' : '#ff6b6b';
                        html += `<span style="color: ${color};">  ${status} Width: ${widthVal.width_meters.toFixed(3)}m `;
                        if (widthVal.passed) {
                            html += `(within range [${widthVal.expected_range[0]}, ${widthVal.expected_range[1]}]m)</span><br/>`;
                        } else {
                            html += `(OUTSIDE range [${widthVal.expected_range[0]}, ${widthVal.expected_range[1]}]m)</span><br/>`;
                        }
                        html += `<span style="margin-left: 1rem; font-size: 0.85rem; color: #aaa;">`;
                        html += `Left: ${widthVal.left_x_vehicle.toFixed(3)}m, Right: ${widthVal.right_x_vehicle.toFixed(3)}m</span><br/>`;
                    } else {
                        html += `<span style="color: #ffa500;">  ⚠️ Could not validate (error: ${widthVal.error})</span><br/>`;
                    }
                }
                
                // Rejection reasons and recommendations
                if (fsv.would_reject && fsv.rejection_reasons.length > 0) {
                    html += `<br/><strong style="color: #ff6b6b;">Rejection Reasons:</strong><br/>`;
                    fsv.rejection_reasons.forEach(reason => {
                        html += `<span style="color: #ff6b6b;">  • ${reason}</span><br/>`;
                    });
                    html += `<br/><strong style="color: #ffa500;">💡 Recommendations:</strong><br/>`;
                    if (fsv.rejection_reasons.includes('left_extreme_coefficients') || 
                        fsv.rejection_reasons.includes('right_extreme_coefficients')) {
                        html += `<span style="font-size: 0.9rem;">  • Polynomial produces extreme x-values → Improve point detection/fitting</span><br/>`;
                        html += `<span style="font-size: 0.9rem;">  • Points may be too sparse or from upcoming curves</span><br/>`;
                        html += `<span style="font-size: 0.9rem;">  • Consider stricter curve filtering or better polynomial constraints</span><br/>`;
                    }
                    if (fsv.rejection_reasons.includes('invalid_width')) {
                        html += `<span style="font-size: 0.9rem;">  • Lane width invalid → Check coordinate conversion or detection accuracy</span><br/>`;
                    }
                } else if (!fsv.would_reject) {
                    html += `<br/><span style="color: #4caf50;">✅ All validations passed - detections would be accepted by full system</span>`;
                }
                
                html += `</div>`;
            }
            
            html += '</div>';
            contentDiv.innerHTML = html;
            
        } catch (error) {
            contentDiv.innerHTML = `<div style="color: #ff6b6b;">Error: ${error.message}</div>`;
        } finally {
            btn.disabled = false;
            btn.textContent = 'Analyze Current Frame';
        }
    }
    
    async generateDebugOverlays() {
        if (!this.currentRecording || this.currentFrameIndex === undefined) {
            alert('Please load a recording and select a frame first.');
            return;
        }
        
        const btn = document.getElementById('generate-debug-btn');
        if (!btn) {
            console.error('Generate debug button not found');
            return;
        }
        
        btn.disabled = true;
        btn.textContent = 'Generating...';
        
        try {
            const response = await fetch(
                `/api/recording/${this.currentRecording}/frame/${this.currentFrameIndex}/generate-debug`
            );
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }
            
            const data = await response.json();
            
            // Store generated debug images in cache
            if (!this.generatedDebugCache) {
                this.generatedDebugCache = {};
            }
            
            // Store fit_points in cache if available
            if (data.fit_points_left || data.fit_points_right) {
                if (!this.generatedFitPointsCache) {
                    this.generatedFitPointsCache = {};
                }
                this.generatedFitPointsCache[this.currentFrameIndex] = {
                    left: data.fit_points_left || null,
                    right: data.fit_points_right || null
                };
            }
            
            // Update overlays to show new fit points if checkbox is checked
            const fitPointsToggle = document.getElementById('toggle-fit-points');
            if (fitPointsToggle && fitPointsToggle.checked) {
                this.updateOverlays();
            }
            const cacheKey = `${this.currentRecording}_${this.currentFrameIndex}`;
            this.generatedDebugCache[cacheKey] = {
                edges: data.edges,
                yellow_mask: data.yellow_mask,
                combined: data.combined
            };
            
            // Store fit_points in cache if available
            if (data.fit_points_left || data.fit_points_right) {
                if (!this.generatedFitPointsCache) {
                    this.generatedFitPointsCache = {};
                }
                this.generatedFitPointsCache[this.currentFrameIndex] = {
                    left: data.fit_points_left || null,
                    right: data.fit_points_right || null
                };
            }
            
            // Auto-check the checkboxes
            const edgesToggle = document.getElementById('toggle-edges');
            const yellowToggle = document.getElementById('toggle-yellow-mask');
            const combinedToggle = document.getElementById('toggle-combined');
            if (edgesToggle) edgesToggle.checked = true;
            if (yellowToggle) yellowToggle.checked = true;
            if (combinedToggle) combinedToggle.checked = true;
            
            // Update debug overlays display
            await this.updateDebugOverlays();
            
            // Update overlays to show fit points if checkbox is checked
            if (fitPointsToggle && fitPointsToggle.checked) {
                this.updateOverlays();
            }
            
            // Show success message
            btn.textContent = '✅ Generated!';
            setTimeout(() => {
                btn.textContent = 'Generate Debug Overlays (Current Frame)';
            }, 2000);
            
        } catch (error) {
            console.error('Error generating debug overlays:', error);
            alert(`Error generating debug overlays: ${error.message}`);
            btn.textContent = 'Generate Debug Overlays (Current Frame)';
        } finally {
            btn.disabled = false;
        }
    }

    async loadSignalsList() {
        try {
            const signals = await this.dataLoader.loadSignals();
            this.availableSignals = signals;
            this.renderSignalList();
            this.populateXAxisOptions();
            this.populateCurvatureOptions();
            this.renderSavedViews();
            this.applyLastChartView();
        } catch (error) {
            console.error('Error loading signals:', error);
        }
    }

    populateXAxisOptions() {
        const select = document.getElementById('chart-x-axis');
        if (!select) return;
        const signalNames = this.availableSignals.map(s => s.name);
        select.innerHTML = '';
        for (const name of signalNames) {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            select.appendChild(option);
        }
        if (!signalNames.includes(this.chartXAxisKey)) {
            const defaultKey = signalNames.find(name => name.includes('timestamps')) || signalNames[0] || '';
            this.chartXAxisKey = defaultKey;
        }
        if (this.chartXAxisKey) {
            select.value = this.chartXAxisKey;
        }
    }

    populateCurvatureOptions() {
        const select = document.getElementById('chart-curvature-signal');
        if (!select) return;
        const signalNames = this.availableSignals.map(s => s.name);
        select.innerHTML = '';
        const autoOption = document.createElement('option');
        autoOption.value = '';
        autoOption.textContent = 'Auto (curvature)';
        select.appendChild(autoOption);
        for (const name of signalNames) {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            select.appendChild(option);
        }
    }

    renderSignalList() {
        const container = document.getElementById('chart-signal-list');
        if (!container) return;
        const query = (document.getElementById('chart-signal-search')?.value || '').toLowerCase();
        const signals = this.availableSignals
            .map(s => s.name)
            .filter(name => !name.includes('timestamps'));
        container.innerHTML = '';
        for (const name of signals) {
            if (query && !name.toLowerCase().includes(query)) continue;
            const item = document.createElement('label');
            item.className = 'chart-signal-item';
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = name;
            item.appendChild(checkbox);
            const text = document.createElement('span');
            text.textContent = name;
            item.appendChild(text);
            container.appendChild(item);
        }
    }

    getSelectedSignals() {
        const container = document.getElementById('chart-signal-list');
        if (!container) return [];
        const selected = [];
        container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            if (cb.checked) {
                selected.push(cb.value);
            }
        });
        return selected;
    }

    async plotSelectedSignals() {
        const signals = this.getSelectedSignals();
        const xAxisKey = document.getElementById('chart-x-axis')?.value || '';
        if (!signals.length) {
            alert('Select at least one signal to plot.');
            return;
        }
        try {
            const data = await this.dataLoader.loadTimeSeries(signals, xAxisKey);
            await this.loadCurvatureSeries(xAxisKey, data.time?.length || data.signals[signals[0]].length);
            this.renderChart(data);
        } catch (error) {
            console.error('Error plotting signals:', error);
            alert(`Failed to plot signals: ${error.message}`);
        }
    }

    renderChart(data) {
        const ctx = document.getElementById('chart-canvas');
        if (!ctx) return;
        const labels = data.time || data.signals[Object.keys(data.signals)[0]].map((_, i) => i);
        const usesTime = !!data.time;
        this.chartTimeSeries = data.time || null;
        this.chartUsesTime = usesTime;
        this.chartSeriesData = data.signals || {};
        this.chartSeriesNames = Object.keys(this.chartSeriesData);
        if (labels.length) {
            this.chartXMin = labels[0];
            this.chartXMax = labels[labels.length - 1];
        } else {
            this.chartXMin = null;
            this.chartXMax = null;
        }
        const palette = [
            '#4a90e2', '#e94f37', '#8bc34a', '#f4b400',
            '#9c27b0', '#00bcd4', '#ff9800', '#cddc39',
        ];
        const datasets = Object.entries(data.signals).map(([name, values], idx) => {
            const seriesData = usesTime
                ? values.map((value, i) => ({ x: data.time[i], y: value }))
                : values;
            return {
                label: name,
                data: seriesData,
                borderColor: palette[idx % palette.length],
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.2,
            };
        });

        if (this.chart) {
            const xType = this.chart.options.scales?.x?.type || 'category';
            if ((usesTime && xType !== 'linear') || (!usesTime && xType !== 'category')) {
                this.chart.destroy();
                this.chart = null;
            } else {
                this.chart.data.labels = labels;
                this.chart.data.datasets = datasets;
                this.chart.options.scales.x.min = this.chartXMin;
                this.chart.options.scales.x.max = this.chartXMax;
                this.chart.update();
                this.updateQuickChartValuesTable();
                return;
            }
        }
        const cursorPlugin = {
            id: 'frameCursor',
            afterDraw: (chart) => {
                if (this.chartCursorValue === null && this.chartCursorIndex === null) {
                    return;
                }
                const xScale = chart.scales.x;
                const value = this.chartUsesTime ? this.chartCursorValue : this.chartCursorIndex;
                if (value === null || value === undefined) {
                    return;
                }
                const x = xScale.getPixelForValue(value);
                if (!Number.isFinite(x)) {
                    return;
                }
                const ctx2 = chart.ctx;
                ctx2.save();
                ctx2.strokeStyle = '#ffffff';
                ctx2.lineWidth = 1;
                ctx2.beginPath();
                ctx2.moveTo(x, chart.chartArea.top);
                ctx2.lineTo(x, chart.chartArea.bottom);
                ctx2.stroke();
                ctx2.restore();
            },
        };
        const curveShadePlugin = {
            id: 'curveShade',
            beforeDatasetsDraw: (chart) => {
                if (!this.chartCurvatureSeries || !this.isCurveShadingEnabled()) {
                    return;
                }
                const xScale = chart.scales.x;
                const ctx2 = chart.ctx;
                const light = this.getCurveThresholdLight();
                const dark = this.getCurveThresholdDark();
                const timeSeries = this.chartTimeSeries;
                const series = this.chartCurvatureSeries;
                const isTime = this.chartUsesTime;

                let current = null;
                const ranges = [];
                for (let i = 0; i < series.length; i++) {
                    const value = Math.abs(series[i]);
                    let level = 0;
                    if (value >= dark) {
                        level = 2;
                    } else if (value >= light) {
                        level = 1;
                    }
                    const xVal = isTime && timeSeries ? timeSeries[i] : i;
                    if (level === 0) {
                        if (current) {
                            current.end = xVal;
                            ranges.push(current);
                            current = null;
                        }
                        continue;
                    }
                    if (!current || current.level !== level) {
                        if (current) {
                            current.end = xVal;
                            ranges.push(current);
                        }
                        current = { start: xVal, end: xVal, level };
                    } else {
                        current.end = xVal;
                    }
                }
                if (current) {
                    ranges.push(current);
                }

                ctx2.save();
                for (const range of ranges) {
                    const color = range.level === 2 ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.06)';
                    const xStart = xScale.getPixelForValue(range.start);
                    const xEnd = xScale.getPixelForValue(range.end);
                    ctx2.fillStyle = color;
                    ctx2.fillRect(xStart, chart.chartArea.top, xEnd - xStart, chart.chartArea.bottom - chart.chartArea.top);
                }
                ctx2.restore();
            },
        };
        this.chart = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#e0e0e0' } },
                },
                scales: {
                    x: {
                        type: usesTime ? 'linear' : 'category',
                        ticks: { color: '#cfcfcf' },
                        grid: { color: '#333' },
                    },
                    y: { ticks: { color: '#cfcfcf' }, grid: { color: '#333' } },
                },
            },
            plugins: [curveShadePlugin, cursorPlugin],
        });
        this.updateChartCursor();
        this.updateQuickChartValuesTable();
    }

    async loadCurvatureSeries(xAxisKey, expectedLength) {
        this.chartCurvatureSeries = null;
        if (!this.isCurveShadingEnabled()) {
            return;
        }
        const curvatureKey = this.getCurvatureSignalKey();
        if (!curvatureKey) {
            return;
        }
        try {
            const data = await this.dataLoader.loadTimeSeries([curvatureKey], xAxisKey);
            const series = data.signals[curvatureKey] || [];
            this.chartCurvatureSeries = series.slice(0, expectedLength);
        } catch (error) {
            console.warn('Failed to load curvature series:', error);
        }
    }

    isCurveShadingEnabled() {
        return !!document.getElementById('chart-shade-curves')?.checked;
    }

    getCurvatureSignalKey() {
        const select = document.getElementById('chart-curvature-signal');
        const chosen = select?.value || '';
        if (chosen) {
            return chosen;
        }
        const candidates = [
            'ground_truth/path_curvature',
            'vehicle/ground_truth_path_curvature',
            'control/path_curvature_input',
            'trajectory/curvature',
        ];
        return candidates.find(name => this.availableSignals.find(s => s.name === name));
    }

    getCurveThresholdLight() {
        const value = parseFloat(document.getElementById('chart-curve-threshold-light')?.value);
        return Number.isFinite(value) ? value : 0.01;
    }

    getCurveThresholdDark() {
        const value = parseFloat(document.getElementById('chart-curve-threshold-dark')?.value);
        return Number.isFinite(value) ? value : 0.02;
    }

    updateChartCursor() {
        if (!this.chart) return;
        if (this.chartUsesTime && this.chartTimeSeries) {
            const idx = Math.min(this.currentFrameIndex, this.chartTimeSeries.length - 1);
            this.chartCursorIndex = idx;
            this.chartCursorValue = this.chartTimeSeries[idx];
        } else {
            this.chartCursorIndex = this.currentFrameIndex;
            this.chartCursorValue = null;
        }
        this.chart.update('none');
        this.updateQuickChartValuesTable();
    }

    clearChart() {
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }
        this.chartSeriesData = null;
        this.chartSeriesNames = [];
        this.chartXMin = null;
        this.chartXMax = null;
        this.updateQuickChartValuesTable();
    }

    loadChartViews() {
        try {
            const raw = localStorage.getItem('av_debug_chart_views');
            const defaults = this.getDefaultChartViews();
            if (raw) {
                const saved = JSON.parse(raw);
                const merged = [...saved];
                for (const view of defaults) {
                    if (!merged.some(existing => existing.name === view.name)) {
                        merged.push(view);
                    }
                }
                return merged;
            }
            return defaults;
        } catch (error) {
            return this.getDefaultChartViews();
        }
    }

    getDefaultChartViews() {
        return [
            {
                name: 'Steering Plan vs Command',
                signals: [
                    'ground_truth/path_curvature',
                    'control/path_curvature_input',
                    'control/steering'
                ],
                timeKey: 'vehicle/timestamps'
            },
            {
                name: 'Perception Wiggle Debug',
                signals: [
                    'perception/left_lane_line_x',
                    'perception/right_lane_line_x',
                    'trajectory/reference_point_x',
                    'vehicle/road_frame_lane_center_offset',
                    'control/steering'
                ],
                timeKey: 'vehicle/timestamps'
            },
            {
                name: 'Oscillation Lead-In',
                signals: [
                    'trajectory/reference_point_x',
                    'trajectory/reference_point_raw_x',
                    'derived/expected_right_center',
                    'perception/right_lane_line_x',
                    'perception/left_lane_line_x',
                    'control/lateral_error',
                    'control/heading_error',
                    'control/steering',
                    'vehicle/road_frame_lane_center_offset'
                ],
                timeKey: 'vehicle/timestamps'
            },
            {
                name: 'Steering Errors vs Command',
                signals: [
                    'control/lateral_error',
                    'control/heading_error',
                    'control/total_error',
                    'control/steering'
                ],
                timeKey: 'vehicle/timestamps'
            },
            {
                name: 'Steering Limiter Waterfall',
                signals: [
                    'control/steering_pre_rate_limit',
                    'control/steering_post_rate_limit',
                    'control/steering_post_jerk_limit',
                    'control/steering_post_hard_clip',
                    'control/steering_post_smoothing',
                    'control/steering'
                ],
                timeKey: 'vehicle/timestamps'
            },
            {
                name: 'Steering Limiter Deltas',
                signals: [
                    'control/steering_rate_limited_delta',
                    'control/steering_jerk_limited_delta',
                    'control/steering_hard_clip_delta',
                    'control/steering_smoothing_delta',
                    'control/feedback_steering',
                    'control/feedforward_steering'
                ],
                timeKey: 'vehicle/timestamps'
            },
            {
                name: 'Trajectory Reference vs Command',
                signals: [
                    'trajectory/reference_point_x',
                    'trajectory/reference_point_heading',
                    'trajectory/reference_point_velocity',
                    'control/steering'
                ],
                timeKey: 'vehicle/timestamps'
            }
        ];
    }

    loadLastChartViewName() {
        try {
            return localStorage.getItem('av_debug_chart_view_last') || '';
        } catch (error) {
            return '';
        }
    }

    saveLastChartViewName(name) {
        this.lastChartViewName = name || '';
        localStorage.setItem('av_debug_chart_view_last', this.lastChartViewName);
    }

    saveChartViews() {
        localStorage.setItem('av_debug_chart_views', JSON.stringify(this.chartSavedViews));
    }

    saveChartView() {
        const name = document.getElementById('chart-view-name')?.value?.trim();
        if (!name) {
            alert('Enter a view name.');
            return;
        }
        const signals = this.getSelectedSignals();
        if (!signals.length) {
            alert('Select signals before saving.');
            return;
        }
        const timeKey = document.getElementById('chart-x-axis')?.value || '';
        this.chartSavedViews = this.chartSavedViews.filter(v => v.name !== name);
        this.chartSavedViews.push({ name, signals, timeKey });
        this.saveChartViews();
        this.renderSavedViews();
        document.getElementById('chart-view-name').value = '';
    }

    renderSavedViews() {
        const select = document.getElementById('chart-saved-views-select');
        const quickSelect = document.getElementById('quick-saved-views-select');
        const currentValue = select ? select.value : '';
        const quickCurrentValue = quickSelect ? quickSelect.value : '';
        const fillSelect = (el, placeholderText) => {
            if (!el) return;
            el.innerHTML = '';
            const placeholder = document.createElement('option');
            placeholder.value = '';
            placeholder.textContent = this.chartSavedViews.length ? placeholderText : 'No saved views';
            el.appendChild(placeholder);
            for (const view of this.chartSavedViews) {
                const option = document.createElement('option');
                option.value = view.name;
                option.textContent = view.name;
                el.appendChild(option);
            }
        };
        fillSelect(select, 'Select a view');
        fillSelect(quickSelect, 'Quick select view');
        const preferred = currentValue || quickCurrentValue || this.lastChartViewName;
        if (preferred && this.chartSavedViews.some(v => v.name === preferred)) {
            if (select) select.value = preferred;
            if (quickSelect) quickSelect.value = preferred;
        }
    }

    loadSelectedChartView() {
        const select = document.getElementById('chart-saved-views-select');
        if (!select || !select.value) return;
        const quickSelect = document.getElementById('quick-saved-views-select');
        if (quickSelect) {
            quickSelect.value = select.value;
        }
        const view = this.chartSavedViews.find(v => v.name === select.value);
        if (view) {
            this.applyChartView(view);
        }
    }

    deleteSelectedChartView() {
        const select = document.getElementById('chart-saved-views-select');
        if (!select || !select.value) return;
        this.deleteChartView(select.value);
    }

    overwriteSelectedChartView() {
        const select = document.getElementById('chart-saved-views-select');
        if (!select || !select.value) return;
        const name = select.value;
        const signals = this.getSelectedSignals();
        if (!signals.length) {
            alert('Select signals before overwriting.');
            return;
        }
        const timeKey = document.getElementById('chart-x-axis')?.value || '';
        this.chartSavedViews = this.chartSavedViews.filter(v => v.name !== name);
        this.chartSavedViews.push({ name, signals, timeKey });
        this.saveChartViews();
        this.renderSavedViews();
        this.saveLastChartViewName(name);
    }

    applyChartView(view) {
        const select = document.getElementById('chart-x-axis');
        if (select && view.timeKey) {
            select.value = view.timeKey;
        }
        this.renderSignalList();
        const container = document.getElementById('chart-signal-list');
        if (container) {
            container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                cb.checked = view.signals.includes(cb.value);
            });
        }
        this.plotSelectedSignals();
        this.saveLastChartViewName(view.name);
    }

    deleteChartView(name) {
        this.chartSavedViews = this.chartSavedViews.filter(v => v.name !== name);
        if (name === this.lastChartViewName) {
            this.saveLastChartViewName('');
        }
        this.saveChartViews();
        this.renderSavedViews();
    }

    applyLastChartView() {
        if (!this.lastChartViewName) return;
        const view = this.chartSavedViews.find(v => v.name === this.lastChartViewName);
        if (view) {
            this.applyChartView(view);
        }
    }

    applyQuickSelectedChartView() {
        const quickSelect = document.getElementById('quick-saved-views-select');
        const mainSelect = document.getElementById('chart-saved-views-select');
        if (!quickSelect || !quickSelect.value) {
            return;
        }
        if (mainSelect) {
            mainSelect.value = quickSelect.value;
        }
        const view = this.chartSavedViews.find(v => v.name === quickSelect.value);
        if (view) {
            this.applyChartView(view);
        }
    }

    updateQuickChartValuesTable() {
        const body = document.getElementById('quick-chart-values-body');
        if (!body) return;

        if (!this.chartSeriesData || !this.chartSeriesNames.length) {
            body.innerHTML = '<tr><td colspan="2" style="text-align: center; color: #888;">Plot a chart view to see values</td></tr>';
            return;
        }

        const frameIndex = this.currentFrameIndex;
        const rows = [];
        for (const signal of this.chartSeriesNames) {
            const series = this.chartSeriesData[signal] || [];
            const value = frameIndex < series.length ? series[frameIndex] : null;
            rows.push(
                `<tr><td>${signal}</td><td>${this.formatQuickChartValue(value)}</td></tr>`
            );
        }
        body.innerHTML = rows.length
            ? rows.join('')
            : '<tr><td colspan="2" style="text-align: center; color: #888;">No chart data available</td></tr>';
    }

    formatQuickChartValue(value) {
        if (value === null || value === undefined || Number.isNaN(value)) {
            return '-';
        }
        if (!Number.isFinite(value)) {
            return String(value);
        }
        const abs = Math.abs(value);
        if (abs >= 1000) return value.toFixed(2);
        if (abs >= 100) return value.toFixed(3);
        if (abs >= 1) return value.toFixed(4);
        return value.toFixed(6);
    }

    zoomChart(factor) {
        if (!this.chart || this.chartXMin === null || this.chartXMax === null) {
            return;
        }
        const xScaleOpts = this.chart.options?.scales?.x;
        if (!xScaleOpts) {
            return;
        }
        const fullMin = this.chartXMin;
        const fullMax = this.chartXMax;
        let currentMin = xScaleOpts.min ?? fullMin;
        let currentMax = xScaleOpts.max ?? fullMax;
        if (!Number.isFinite(currentMin) || !Number.isFinite(currentMax)) {
            currentMin = fullMin;
            currentMax = fullMax;
        }
        const fullRange = fullMax - fullMin;
        if (fullRange <= 0) {
            return;
        }
        const center = this.chartUsesTime
            ? (this.chartCursorValue ?? ((currentMin + currentMax) / 2))
            : (this.chartCursorIndex ?? ((currentMin + currentMax) / 2));
        const minRange = this.chartUsesTime ? Math.max(fullRange / 1000, 1e-6) : 1;
        let nextRange = (currentMax - currentMin) * factor;
        if (!Number.isFinite(nextRange) || nextRange <= 0) {
            return;
        }
        nextRange = Math.min(fullRange, Math.max(minRange, nextRange));

        let nextMin = center - (nextRange / 2);
        let nextMax = center + (nextRange / 2);
        if (nextMin < fullMin) {
            nextMax += (fullMin - nextMin);
            nextMin = fullMin;
        }
        if (nextMax > fullMax) {
            nextMin -= (nextMax - fullMax);
            nextMax = fullMax;
        }
        nextMin = Math.max(fullMin, nextMin);
        nextMax = Math.min(fullMax, nextMax);
        xScaleOpts.min = nextMin;
        xScaleOpts.max = nextMax;
        this.chart.update('none');
    }

    resetChartZoom() {
        if (!this.chart || this.chartXMin === null || this.chartXMax === null) {
            return;
        }
        const xScaleOpts = this.chart.options?.scales?.x;
        if (!xScaleOpts) {
            return;
        }
        xScaleOpts.min = this.chartXMin;
        xScaleOpts.max = this.chartXMax;
        this.chart.update('none');
    }

    toggleLegend() {
        const panel = document.getElementById('legend-panel');
        const btn = document.getElementById('legend-toggle-btn');
        if (!panel || !btn) return;
        panel.classList.toggle('collapsed');
        btn.textContent = panel.classList.contains('collapsed') ? 'Legend' : 'Collapse';
    }

    setupPanelResize() {
        const handle = document.getElementById('panel-resize-handle');
        const cameraPanel = document.getElementById('camera-panel');
        const dataPanel = document.getElementById('data-panel');
        if (!handle || !cameraPanel || !dataPanel) return;

        const applySplit = (requestedLeftWidth) => {
            const main = document.getElementById('main-content');
            if (!main) return;
            const rect = main.getBoundingClientRect();
            if (!rect || rect.width <= 0) return;
            const minLeft = 520;
            const minRight = 320;
            const maxLeft = rect.width - minRight - 6;
            const clamped = Math.max(minLeft, Math.min(maxLeft, requestedLeftWidth));
            cameraPanel.style.flex = 'none';
            cameraPanel.style.width = `${clamped}px`;
            dataPanel.style.width = `${rect.width - clamped - 6}px`;
        };

        const savedLeft = localStorage.getItem(this.panelSplitLeftWidthKey);
        if (savedLeft) {
            const parsed = parseFloat(savedLeft);
            if (Number.isFinite(parsed) && parsed >= 320 && parsed <= 4000) {
                // Delay to ensure layout has settled before applying persisted split.
                requestAnimationFrame(() => applySplit(parsed));
            } else {
                localStorage.removeItem(this.panelSplitLeftWidthKey);
            }
        }

        let dragging = false;
        handle.addEventListener('mousedown', (event) => {
            dragging = true;
            document.body.style.cursor = 'col-resize';
            event.preventDefault();
        });
        document.addEventListener('mousemove', (event) => {
            if (!dragging) return;
            const main = document.getElementById('main-content');
            if (!main) return;
            const rect = main.getBoundingClientRect();
            const minLeft = 520;
            const minRight = 320;
            const maxLeft = rect.width - minRight - 6;
            const leftWidth = event.clientX - rect.left;
            const clamped = Math.max(minLeft, Math.min(maxLeft, leftWidth));
            cameraPanel.style.flex = 'none';
            cameraPanel.style.width = `${clamped}px`;
            dataPanel.style.width = `${rect.width - clamped - 6}px`;
        });
        document.addEventListener('mouseup', () => {
            if (dragging) {
                dragging = false;
                document.body.style.cursor = 'default';
                const leftWidth = cameraPanel.getBoundingClientRect().width;
                if (Number.isFinite(leftWidth) && leftWidth > 0) {
                    localStorage.setItem(this.panelSplitLeftWidthKey, String(Math.round(leftWidth)));
                }
            }
        });

        // Keep persisted split valid across browser resizes.
        window.addEventListener('resize', () => {
            const saved = localStorage.getItem(this.panelSplitLeftWidthKey);
            if (!saved) return;
            const parsed = parseFloat(saved);
            if (Number.isFinite(parsed)) {
                applySplit(parsed);
            }
        });
    }

    setupCameraGridResize() {
        const handle = document.getElementById('camera-grid-resize-handle');
        const grid = document.getElementById('camera-grid');
        const wrapper = document.getElementById('camera-grid-wrapper');
        if (!handle || !grid) {
            return;
        }

        const baseRect = grid.getBoundingClientRect();
        const baseHeight = baseRect.height || 1;
        const baseWidth = baseRect.width || 1;
        this.cameraGridBaseHeight = baseHeight;
        this.cameraGridBaseWidth = baseWidth;

        const applyHeight = (height) => {
            const clampedHeight = Math.max(200, height);
            const scale = clampedHeight / this.cameraGridBaseHeight;
            if (wrapper) {
                wrapper.style.height = `${clampedHeight}px`;
            }
            grid.style.width = `${this.cameraGridBaseWidth}px`;
            grid.style.height = `${this.cameraGridBaseHeight}px`;
            grid.style.transform = `scale(${scale})`;
        };

        const savedHeight = localStorage.getItem(this.cameraGridHeightKey);
        if (savedHeight) {
            const parsed = parseFloat(savedHeight);
            if (Number.isFinite(parsed) && parsed >= 300 && parsed <= 1400) {
                applyHeight(parsed);
            } else {
                localStorage.removeItem(this.cameraGridHeightKey);
                if (wrapper) wrapper.style.height = '60vh';
                grid.style.width = '';
                grid.style.height = '';
                grid.style.transform = '';
            }
        } else {
            if (wrapper) wrapper.style.height = '60vh';
            grid.style.width = '';
            grid.style.height = '';
            grid.style.transform = '';
        }

        let isResizing = false;
        let startY = 0;
        let startHeight = 0;

        const onMouseMove = (e) => {
            if (!isResizing) return;
            const delta = e.clientY - startY;
            const newHeight = Math.max(200, startHeight + delta);
            applyHeight(newHeight);
        };

        const onMouseUp = () => {
            if (!isResizing) return;
            isResizing = false;
            const currentHeight = wrapper
                ? wrapper.getBoundingClientRect().height
                : grid.getBoundingClientRect().height;
            localStorage.setItem(this.cameraGridHeightKey, String(Math.round(currentHeight)));
            document.body.style.cursor = '';
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };

        handle.addEventListener('mousedown', (e) => {
            isResizing = true;
            startY = e.clientY;
            startHeight = wrapper
                ? wrapper.getBoundingClientRect().height
                : grid.getBoundingClientRect().height;
            document.body.style.cursor = 'row-resize';
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });
    }
}

// Initialize visualizer when page loads
let visualizer;
document.addEventListener('DOMContentLoaded', () => {
    visualizer = new Visualizer();
    window.visualizer = visualizer;  // Make accessible globally for onclick handlers
});

