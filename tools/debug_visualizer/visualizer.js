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
        this.isPlaying = false;
        this.playSpeed = 1.0;
        this.playInterval = null;
        
        this.currentFrameData = null;
        this.previousPerceptionData = null;  // Track previous frame's perception data for change calculations
        this.currentImage = null;
        this.lastValidY8m = undefined;  // Cache last valid camera_8m_screen_y value
        this.groundTruthDistance = 7;  // Tunable distance for ground truth conversion (calibrated to account for camera pitch/height)
        this.generatedDebugCache = {};  // Cache for on-demand generated debug images
        this.lastUnityTime = null;
        this.lastUnityFrameIndex = null;
        
        this.setupEventListeners();
        this.loadRecordings();
    }

    setupEventListeners() {
        // Recording selection
        document.getElementById('load-btn').addEventListener('click', () => this.loadSelectedRecording());
        
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
        document.getElementById('toggle-detected-curves').addEventListener('change', () => this.updateOverlays());
        document.getElementById('toggle-fit-points').addEventListener('change', () => this.updateOverlays());
        document.getElementById('toggle-trajectory').addEventListener('change', () => this.updateOverlays());
        document.getElementById('toggle-reference').addEventListener('change', () => this.updateOverlays());
        document.getElementById('toggle-ground-truth').addEventListener('change', () => this.updateOverlays());
        
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
        document.getElementById('toggle-combined').addEventListener('change', () => this.updateDebugOverlays());
        document.getElementById('toggle-edges').addEventListener('change', () => this.updateDebugOverlays());
        document.getElementById('toggle-yellow-mask').addEventListener('change', () => this.updateDebugOverlays());
        document.getElementById('toggle-histogram').addEventListener('change', () => this.updateDebugOverlays());
        
        // Opacity control
        document.getElementById('opacity-slider').addEventListener('input', (e) => {
            const opacity = parseInt(e.target.value) / 100;
            document.getElementById('opacity-value').textContent = e.target.value;
            document.getElementById('debug-overlay-canvas').style.opacity = opacity;
        });
        
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
    }

    async loadRecordings() {
        try {
            console.log('Loading recordings...');
            const recordings = await this.dataLoader.loadRecordings();
            console.log('Recordings loaded:', recordings);
            
            const select = document.getElementById('recording-select');
            if (!select) {
                console.error('Recording select element not found!');
                return;
            }
            
            select.innerHTML = '<option value="">Select recording...</option>';
            
            if (!recordings || recordings.length === 0) {
                console.warn('No recordings found');
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
            
            console.log(`Added ${recordings.length} recordings to dropdown`);
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
            document.getElementById('frame-count').textContent = this.frameCount;
            document.getElementById('frame-slider').max = this.frameCount - 1;
            this.currentFrameIndex = 0;
            await this.goToFrame(0);
            await this.loadSummary();  // Load summary when recording is loaded
            await this.loadIssues();  // Load issues when recording is loaded
            await this.loadDiagnostics();  // Load diagnostics when recording is loaded
        } catch (error) {
            console.error('Error loading recording:', error);
            alert('Failed to load recording: ' + error.message);
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
            try {
                const diagUrl = `/api/recording/${this.currentRecording}/diagnostics${analyzeToFailure ? '?analyze_to_failure=true' : ''}`;
                const diagResponse = await fetch(diagUrl);
                if (diagResponse.ok) {
                    diagnostics = await diagResponse.json();
                }
            } catch (e) {
                console.warn('Could not load diagnostics:', e);
            }
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }
            
            const summary = await response.json();
            
            if (summary.error) {
                summaryContent.innerHTML = `<p style="color: #ff6b6b; text-align: center; padding: 2rem;">Error: ${summary.error}</p>`;
                return;
            }
            
            // Build summary HTML
            let html = '<div style="padding: 1rem;">';
            
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
            
            // Unity timing health
            const unityGapMax = summary.system_health?.unity_time_gap_max ?? null;
            const unityGapCount = summary.system_health?.unity_time_gap_count ?? null;
            if (unityGapMax !== null && unityGapCount !== null) {
                const gapColor = unityGapCount > 0 || unityGapMax > 0.2 ? '#ff6b6b' : '#4caf50';
                const gapStatus = unityGapCount > 0 || unityGapMax > 0.2 ? '⚠️ Hitch Detected' : '✓ No Hitch';
                html += '<div style="margin-bottom: 1rem; padding: 0.75rem; background: #1f1f1f; border-radius: 4px;">';
                html += `<strong style="color: ${gapColor};">Unity Timing Health:</strong> `;
                html += `<span style="color: ${gapColor};">${gapStatus}</span><br/>`;
                html += `<span style="color: #888;">Max Unity Time Gap: ${unityGapMax.toFixed(3)}s | Gaps > 0.2s: ${unityGapCount}</span>`;
                html += '</div>';
            }

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
                    html += '<div style="margin-bottom: 1rem; padding: 0.75rem; background: #1f1f1f; border-radius: 4px;">';
                    html += `<strong style="color: ${stabilityColor};">Control Stability (Straight):</strong><br/>`;
                    html += `<span style="color: #888;">Straight Coverage: ${straightFrac.toFixed(1)}% | Oscillation Mean: ${oscMean.toFixed(3)} | Max: ${oscMax !== null ? oscMax.toFixed(3) : '-'}</span><br/>`;
                    html += `<span style="color: #888;">Tuned Deadband: ${deadbandMean !== null ? deadbandMean.toFixed(3) : '-'} (max ${deadbandMax !== null ? deadbandMax.toFixed(3) : '-'}) | Smoothing α: ${smoothingMean !== null ? smoothingMean.toFixed(3) : '-'}</span>`;
                    html += '</div>';
                }
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
            
            // Helper function for color coding
            const getColorForValue = (value, thresholds) => {
                if (value <= thresholds.good) return '#4caf50';  // Green
                if (value <= thresholds.acceptable) return '#ffa500';  // Orange
                return '#ff6b6b';  // Red
            };
            
            // Path Tracking
            html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
            html += '<h3 style="margin-top: 0; color: #4a90e2;">Path Tracking</h3>';
            html += '<table style="width: 100%; color: #e0e0e0;">';
            
            // Lateral Error RMSE: Good <0.2m, Acceptable <0.4m
            const latErrColor = getColorForValue(summary.path_tracking.lateral_error_rmse, { good: 0.2, acceptable: 0.4 });
            html += `<tr><td>Lateral Error (RMSE):</td><td style="text-align: right; color: ${latErrColor};">${summary.path_tracking.lateral_error_rmse.toFixed(3)}m</td></tr>`;
            
            // Lateral Error Max: Good <0.5m, Acceptable <1.0m
            const latMaxColor = getColorForValue(summary.path_tracking.lateral_error_max, { good: 0.5, acceptable: 1.0 });
            html += `<tr><td>Lateral Error (Max):</td><td style="text-align: right; color: ${latMaxColor};">${summary.path_tracking.lateral_error_max.toFixed(3)}m</td></tr>`;
            
            // Lateral Error P95: Good <0.4m, Acceptable <0.8m
            const latP95Color = getColorForValue(summary.path_tracking.lateral_error_p95, { good: 0.4, acceptable: 0.8 });
            html += `<tr><td>Lateral Error (P95):</td><td style="text-align: right; color: ${latP95Color};">${summary.path_tracking.lateral_error_p95.toFixed(3)}m</td></tr>`;
            
            // Heading Error RMSE: Good <10°, Acceptable <20°
            const headingErrDeg = summary.path_tracking.heading_error_rmse * 180 / Math.PI;
            const headingColor = getColorForValue(headingErrDeg, { good: 10, acceptable: 20 });
            html += `<tr><td>Heading Error (RMSE):</td><td style="text-align: right; color: ${headingColor};">${headingErrDeg.toFixed(1)}°</td></tr>`;
            
            // Time in Lane: Good >90%, Acceptable >70%
            const timeInLaneColor = summary.path_tracking.time_in_lane >= 90 ? '#4caf50' : summary.path_tracking.time_in_lane >= 70 ? '#ffa500' : '#ff6b6b';
            html += `<tr><td>Time in Lane:</td><td style="text-align: right; color: ${timeInLaneColor};">${summary.path_tracking.time_in_lane.toFixed(1)}%</td></tr>`;
            html += '</table></div>';
            
            // Control Smoothness
            html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
            html += '<h3 style="margin-top: 0; color: #4a90e2;">Control Smoothness</h3>';
            html += '<table style="width: 100%; color: #e0e0e0;">';
            
            // Steering Jerk: Good <0.5/s², Acceptable <1.0/s² (lower is better)
            const jerkColor = getColorForValue(summary.control_smoothness.steering_jerk_max, { good: 0.5, acceptable: 1.0 });
            html += `<tr><td>Steering Jerk (Max):</td><td style="text-align: right; color: ${jerkColor};">${summary.control_smoothness.steering_jerk_max.toFixed(3)}/s²</td></tr>`;
            
            // Steering Rate: Good <2.0/s, Acceptable <4.0/s (lower is better)
            const rateColor = getColorForValue(summary.control_smoothness.steering_rate_max, { good: 2.0, acceptable: 4.0 });
            html += `<tr><td>Steering Rate (Max):</td><td style="text-align: right; color: ${rateColor};">${summary.control_smoothness.steering_rate_max.toFixed(3)}/s</td></tr>`;
            
            // Steering Smoothness: Good >2.0, Acceptable >1.0 (higher is better - inverse)
            const smoothnessColor = summary.control_smoothness.steering_smoothness >= 2.0 ? '#4caf50' : summary.control_smoothness.steering_smoothness >= 1.0 ? '#ffa500' : '#ff6b6b';
            html += `<tr><td>Steering Smoothness:</td><td style="text-align: right; color: ${smoothnessColor};">${summary.control_smoothness.steering_smoothness.toFixed(2)}</td></tr>`;
            
            // Oscillation Frequency: Good <1.0Hz, Acceptable <2.0Hz (lower is better)
            const oscColor = getColorForValue(summary.control_smoothness.oscillation_frequency, { good: 1.0, acceptable: 2.0 });
            html += `<tr><td>Oscillation Frequency:</td><td style="text-align: right; color: ${oscColor};">${summary.control_smoothness.oscillation_frequency.toFixed(2)}Hz</td></tr>`;
            html += '</table></div>';
            
            // Perception Quality
            html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
            html += '<h3 style="margin-top: 0; color: #4a90e2;">Perception Quality</h3>';
            html += '<table style="width: 100%; color: #e0e0e0;">';
            html += `<tr><td>Lane Detection Rate:</td><td style="text-align: right; color: ${summary.perception_quality.lane_detection_rate >= 90 ? '#4caf50' : summary.perception_quality.lane_detection_rate >= 70 ? '#ffa500' : '#ff6b6b'};">${summary.perception_quality.lane_detection_rate.toFixed(1)}%</td></tr>`;
            html += `<tr><td>Confidence (Mean):</td><td style="text-align: right;">${summary.perception_quality.perception_confidence_mean.toFixed(3)}</td></tr>`;
            html += `<tr><td>Jumps Detected:</td><td style="text-align: right;">${summary.perception_quality.perception_jumps_detected || 0}</td></tr>`;
            if (summary.perception_quality.perception_instability_detected !== undefined) {
                html += `<tr><td>Instability Events:</td><td style="text-align: right; color: ${summary.perception_quality.perception_instability_detected === 0 ? '#4caf50' : '#ff6b6b'};">${summary.perception_quality.perception_instability_detected}</td></tr>`;
            }
            html += `<tr><td>Stale Data Rate:</td><td style="text-align: right; color: ${summary.perception_quality.stale_perception_rate < 10 ? '#4caf50' : summary.perception_quality.stale_perception_rate < 20 ? '#ffa500' : '#ff6b6b'};">${summary.perception_quality.stale_perception_rate.toFixed(1)}%</td></tr>`;
            if (summary.perception_quality.perception_stability_score !== undefined) {
                html += `<tr><td>Stability Score:</td><td style="text-align: right; color: ${summary.perception_quality.perception_stability_score >= 80 ? '#4caf50' : summary.perception_quality.perception_stability_score >= 60 ? '#ffa500' : '#ff6b6b'};">${summary.perception_quality.perception_stability_score.toFixed(1)}%</td></tr>`;
            }
            html += '</table></div>';
            
            // Trajectory Quality
            html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
            html += '<h3 style="margin-top: 0; color: #4a90e2;">Trajectory Quality</h3>';
            html += '<table style="width: 100%; color: #e0e0e0;">';
            
            // Availability: Good >95%, Acceptable >90%
            const availColor = summary.trajectory_quality.trajectory_availability >= 95 ? '#4caf50' : summary.trajectory_quality.trajectory_availability >= 90 ? '#ffa500' : '#ff6b6b';
            html += `<tr><td>Availability:</td><td style="text-align: right; color: ${availColor};">${summary.trajectory_quality.trajectory_availability.toFixed(1)}%</td></tr>`;
            
            // Reference Point Accuracy: Good <0.1m, Acceptable <0.2m (lower is better)
            const refAccColor = getColorForValue(summary.trajectory_quality.ref_point_accuracy_rmse, { good: 0.1, acceptable: 0.2 });
            html += `<tr><td>Reference Point Accuracy (RMSE):</td><td style="text-align: right; color: ${refAccColor};">${summary.trajectory_quality.ref_point_accuracy_rmse.toFixed(3)}m</td></tr>`;
            html += '</table></div>';
            
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
            
        } catch (error) {
            console.error('Error loading summary:', error);
            summaryContent.innerHTML = `<p style="color: #ff6b6b; text-align: center; padding: 2rem;">Error loading summary: ${error.message}</p>`;
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
                html += '<div style="background: #2a2a2a; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">';
                html += '<h3 style="margin-top: 0; color: #4a90e2;">Issues Summary</h3>';
                html += `<div style="color: #e0e0e0; margin-bottom: 0.5rem;">Total Issues: <strong style="color: ${summary.total_issues > 0 ? '#ff6b6b' : '#4caf50'}">${summary.total_issues}</strong></div>`;
                
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
                
                html += '</div>';
            }
            
            // Filter buttons
            html += '<div style="margin-bottom: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">';
            html += '<button class="issue-filter-btn active" data-filter="all" style="padding: 0.5rem 1rem; background: #4a90e2; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">All</button>';
            html += '<button class="issue-filter-btn" data-filter="extreme_coefficients" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Extreme Coefficients</button>';
            html += '<button class="issue-filter-btn" data-filter="perception_instability" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Perception Instability</button>';
            html += '<button class="issue-filter-btn" data-filter="high_lateral_error" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">High Lateral Error</button>';
            html += '<button class="issue-filter-btn" data-filter="perception_failure" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Perception Failure</button>';
            html += '<button class="issue-filter-btn" data-filter="out_of_lane" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Out of Lane</button>';
            html += '<button class="issue-filter-btn" data-filter="emergency_stop" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Emergency Stop</button>';
            html += '<button class="issue-filter-btn" data-filter="heading_jump" style="padding: 0.5rem 1rem; background: #555; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Heading Jump</button>';
            html += '</div>';
            
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
                        'high_lateral_error': '📏',
                        'perception_failure': '👁️',
                        'out_of_lane': '🚫',
                        'emergency_stop': '🛑',
                        'heading_jump': '🔄'
                    }[issue.type] || '⚠️';
                    
                    html += `<div class="issue-item" data-issue-type="${issue.type}" style="background: #2a2a2a; padding: 1rem; border-radius: 8px; border-left: 4px solid ${severityColor}; cursor: pointer;" onclick="window.visualizer.jumpToFrame(${issue.frame})">`;
                    html += `<div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">`;
                    html += `<div style="flex: 1;">`;
                    html += `<div style="font-weight: bold; color: ${severityColor}; margin-bottom: 0.25rem;">${typeIcon} Frame ${issue.frame}: ${issue.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>`;
                    html += `<div style="color: #e0e0e0; font-size: 0.9rem;">${issue.description}</div>`;
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
            filterButtons.forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const filter = btn.dataset.filter;
                    
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
                    
                    // Filter issues
                    const issueItems = issuesContent.querySelectorAll('.issue-item');
                    issueItems.forEach(item => {
                        if (filter === 'all' || item.dataset.issueType === filter) {
                            item.style.display = 'block';
                        } else {
                            item.style.display = 'none';
                        }
                    });
                });
            });
            
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
        
        diagnosticsContent.innerHTML = '<p style="color: #888; text-align: center; padding: 2rem;">Loading diagnostics...</p>';
        
        try {
            const url = `/api/recording/${this.currentRecording}/diagnostics${analyzeToFailure ? '?analyze_to_failure=true' : ''}`;
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const diagnostics = await response.json();
            
            if (diagnostics.error) {
                diagnosticsContent.innerHTML = `<p style="color: #ff6b6b; text-align: center; padding: 2rem;">Error: ${diagnostics.error}</p>`;
                return;
            }
            
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
                    html += `<tr><td>Steering Correlation:</td><td style="text-align: right; color: ${corrColor};">${ctrl.steering_correlation.toFixed(3)}</td></tr>`;
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
            
            // Load camera image
            const imageDataUrl = await this.dataLoader.loadFrameImage(frameIndex);
            await this.loadImage(imageDataUrl);
            
            // Update displays
            this.updateDataPanel();
            this.updateOverlays();
            this.updateDebugOverlays();
            
            // Update timestamp
            if (this.currentFrameData.camera) {
                document.getElementById('frame-time').textContent = 
                    this.currentFrameData.camera.timestamp.toFixed(2);
            }
        } catch (error) {
            console.error('Error loading frame:', error);
        }
    }

    async loadImage(imageDataUrl) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.getElementById('camera-canvas');
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                this.currentImage = img;
                resolve();
            };
            img.onerror = reject;
            img.src = imageDataUrl;
        });
    }

    updateDataPanel() {
        if (!this.currentFrameData) return;
        
        // Update all tabs (including new "All Data" tab)
        this.updatePerceptionData();
        this.updateTrajectoryData();
        this.updateControlData();
        this.updateVehicleData();
        this.updateGroundTruthData();
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

        // Disable fit points toggle when not using CV detection
        const fitToggle = document.getElementById('toggle-fit-points');
        if (fitToggle) {
            const fitLabel = fitToggle.closest('label');
            const isCv = (p.detection_method || '').toLowerCase() === 'cv';
            fitToggle.disabled = !isCv;
            if (!isCv) {
                fitToggle.checked = false;
            }
            if (fitLabel) {
                fitLabel.style.opacity = isCv ? '1.0' : '0.4';
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
            const recordedDistance = 8.0; // Distance at which perception was measured
            const tunableDistance = this.groundTruthDistance; // Distance from slider (where lines align)
            
            // Calculate width at recorded distance
            const widthAtRecorded = right_lane_line_x - left_lane_line_x;
            
            // Scale width to tunable distance (width scales linearly with distance)
            // This matches what's visually displayed when lines align
            const width = widthAtRecorded * (tunableDistance / recordedDistance);
            updateField('perception-width', `${width.toFixed(3)}m`);
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

    updateControlData() {
        if (!this.currentFrameData || !this.currentFrameData.control) return;
        
        const c = this.currentFrameData.control;
        
        const updateField = (id, value) => {
            const elem = document.getElementById(id);
            const allElem = document.getElementById('all-' + id);
            if (elem) elem.textContent = value;
            if (allElem) allElem.textContent = value;
        };
        
        updateField('control-steering', c.steering !== undefined ? c.steering.toFixed(4) : '-');
        updateField('control-throttle', c.throttle !== undefined ? c.throttle.toFixed(4) : '-');
        updateField('control-brake', c.brake !== undefined ? c.brake.toFixed(4) : '-');
        updateField('control-lateral-error', c.lateral_error !== undefined ? `${c.lateral_error.toFixed(3)}m` : '-');
        updateField('control-heading-error', c.heading_error !== undefined ? `${(c.heading_error * 180 / Math.PI).toFixed(2)}°` : '-');
        updateField('control-total-error', c.total_error !== undefined ? c.total_error.toFixed(4) : '-');
        updateField('control-pid-integral', c.pid_integral !== undefined ? c.pid_integral.toFixed(4) : '-');
        updateField('control-pid-derivative', c.pid_derivative !== undefined ? c.pid_derivative.toFixed(4) : '-');
        updateField('control-pid-error', c.pid_error !== undefined ? c.pid_error.toFixed(4) : '-');
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
    }

    updateVehicleData() {
        if (!this.currentFrameData || !this.currentFrameData.vehicle) return;
        
        const v = this.currentFrameData.vehicle;
        
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
        updateField('vehicle-speed', v.speed !== undefined ? `${v.speed.toFixed(2)}m/s` : '-');
        
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
        
        if (!this.currentFrameData) {
            console.log('[OVERLAY] No frame data available');
            return;
        }
        
        console.log('[OVERLAY] Updating overlays:', {
            hasPerception: !!this.currentFrameData.perception,
            hasGroundTruth: !!this.currentFrameData.ground_truth,
            hasTrajectory: !!this.currentFrameData.trajectory,
            toggleGroundTruth: document.getElementById('toggle-ground-truth').checked,
            toggleLanes: document.getElementById('toggle-lanes').checked,
            toggleCurves: document.getElementById('toggle-detected-curves').checked
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
        
        // NEW: Update overlay renderer with Unity's actual horizontal FOV from recording
        // This ensures visualizer uses the same FOV that Unity actually uses (not hardcoded 110°)
        if (this.currentFrameData.vehicle && 
            this.currentFrameData.vehicle.camera_horizontal_fov !== undefined &&
            this.currentFrameData.vehicle.camera_horizontal_fov > 0) {
            const unityHorizontalFOV = this.currentFrameData.vehicle.camera_horizontal_fov;
            this.overlayRenderer.setCameraFov(unityHorizontalFOV);
        }
        
        // Draw black reference line FIRST (shows where 8m actually appears)
        // This line is always drawn, using cached value if current frame doesn't have it
        if (y8mActual !== null && y8mActual > 0) {
            this.overlayRenderer.drawReferenceLine(y8mActual);
        } else {
            // Fallback: Draw at default 350px if no valid 8m position available
            this.overlayRenderer.drawReferenceLine(350);
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
        
        // Draw detected lane curves FIRST (orange curves) - shows actual polynomial curves in image space
        if (document.getElementById('toggle-detected-curves').checked) {
            // Prefer newly generated coefficients from debug overlays (if available)
            let laneCoeffs = null;
            if (this.generatedCoefficientsCache && this.generatedCoefficientsCache[this.currentFrameIndex]) {
                laneCoeffs = this.generatedCoefficientsCache[this.currentFrameIndex];
            } else if (this.currentFrameData.perception && this.currentFrameData.perception.lane_line_coefficients) {
                laneCoeffs = this.currentFrameData.perception.lane_line_coefficients;
            }
            
            if (laneCoeffs) {
                // Filter out None/null coefficients (only draw valid lanes)
                const validCoeffs = laneCoeffs.filter(c => c !== null && c !== undefined && Array.isArray(c) && c.length >= 2);
                if (validCoeffs.length > 0) {
                    // Draw detected curves in orange to distinguish from the vertical lines
                    this.overlayRenderer.drawDetectedLaneCurves(validCoeffs, '#ff8800');
                } else {
                    console.log('[OVERLAY] No valid lane coefficients to draw (all None or invalid)');
                }
            } else {
                console.log('[OVERLAY] No lane_line_coefficients available:', {
                    hasPerception: !!this.currentFrameData.perception,
                    hasCoeffs: !!(this.currentFrameData.perception && this.currentFrameData.perception.lane_line_coefficients)
                });
            }
        }
        
        // Draw fit points (points used for polynomial fitting) - NEW
        if (document.getElementById('toggle-fit-points').checked) {
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
            if (fit_points_left && Array.isArray(fit_points_left) && fit_points_left.length > 0) {
                this.overlayRenderer.drawFitPoints(fit_points_left, '#ff00ff', 3);  // Magenta for left lane
            }
            if (fit_points_right && Array.isArray(fit_points_right) && fit_points_right.length > 0) {
                this.overlayRenderer.drawFitPoints(fit_points_right, '#00ffff', 3);  // Cyan for right lane
            }
        }
        
        // Draw detected lane lines SECOND (red vertical lines) - overlay layer
        // This allows easy comparison: Green = GT, Red = Detected positions
        // FIXED: If we have polynomial coefficients, use them to draw red lines at y=350px
        // This ensures red lines match orange curves at the black line
        if (document.getElementById('toggle-lanes').checked) {
            if (this.currentFrameData.perception) {
                const p = this.currentFrameData.perception;
                // Extract perception lane line positions (with backward compatibility)
                const left_lane_line_x = p.left_lane_line_x !== undefined ? p.left_lane_line_x : (p.left_lane_x !== undefined ? p.left_lane_x : undefined);
                const right_lane_line_x = p.right_lane_line_x !== undefined ? p.right_lane_line_x : (p.right_lane_x !== undefined ? p.right_lane_x : undefined);
                
                // Prefer polynomial coefficients if available (matches orange curves exactly)
                if (p.lane_line_coefficients && Array.isArray(p.lane_line_coefficients) && p.lane_line_coefficients.length >= 2) {
                    // NEW: Use actual 8m position so red lines align with black line
                    this.overlayRenderer.drawLaneLinesFromCoefficients(p.lane_line_coefficients, 350, '#ff0000', y8mActual);
                } else if (left_lane_line_x !== undefined && right_lane_line_x !== undefined) {
                    // Fallback: Use vehicle coords if coefficients not available
                    // FIXED: Keep red lines at fixed 8.0m (don't move visually)
                    // Width calculation will be updated separately using tunable distance
                    this.overlayRenderer.drawLaneLinesFromVehicleCoords(
                        left_lane_line_x, right_lane_line_x, 8.0, '#ff0000', y8mActual
                    );
                }
            }
        }
        
        // Draw trajectory (if available)
        if (document.getElementById('toggle-trajectory').checked) {
            if (this.currentFrameData.trajectory && this.currentFrameData.trajectory.trajectory_points) {
                const trajPoints = this.currentFrameData.trajectory.trajectory_points;
                // Convert to format expected by drawTrajectory: array of {x, y, heading}
                this.overlayRenderer.drawTrajectory(trajPoints, '#ff00ff', 2);
            }
        }
        
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
            }
        }
    }

    async updateDebugOverlays() {
        const canvas = document.getElementById('debug-overlay-canvas');
        if (!canvas) {
            console.warn('Debug overlay canvas not found');
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
        const checkedTypes = types.filter(type => document.getElementById(type.id).checked);
        
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
            
            // Store lane_line_coefficients in cache (for orange curves)
            if (data.lane_line_coefficients) {
                if (!this.generatedCoefficientsCache) {
                    this.generatedCoefficientsCache = {};
                }
                this.generatedCoefficientsCache[this.currentFrameIndex] = data.lane_line_coefficients;
            }
            
            // Update overlays to show new fit points and curves if checkboxes are checked
            if (document.getElementById('toggle-fit-points').checked || document.getElementById('toggle-detected-curves').checked) {
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
            document.getElementById('toggle-edges').checked = true;
            document.getElementById('toggle-yellow-mask').checked = true;
            document.getElementById('toggle-combined').checked = true;
            
            // Update debug overlays display
            await this.updateDebugOverlays();
            
            // Update overlays to show fit points if checkbox is checked
            if (document.getElementById('toggle-fit-points').checked) {
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
}

// Initialize visualizer when page loads
let visualizer;
document.addEventListener('DOMContentLoaded', () => {
    visualizer = new Visualizer();
    window.visualizer = visualizer;  // Make accessible globally for onclick handlers
});

