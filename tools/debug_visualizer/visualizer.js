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
        this.isPlaying = false;
        this.playSpeed = 1.0;
        this.playInterval = null;
        
        this.currentFrameData = null;
        this.currentImage = null;
        this.lastValidY8m = undefined;  // Cache last valid camera_8m_screen_y value
        this.groundTruthDistance = 7;  // Tunable distance for ground truth conversion (calibrated to account for camera pitch/height)
        
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
        document.getElementById('toggle-trajectory').addEventListener('change', () => this.updateOverlays());
        document.getElementById('toggle-reference').addEventListener('change', () => this.updateOverlays());
        document.getElementById('toggle-ground-truth').addEventListener('change', () => this.updateOverlays());
        
        // Ground truth distance slider
        const gtDistanceSlider = document.getElementById('gt-distance-slider');
        const gtDistanceValue = document.getElementById('gt-distance-value');
        gtDistanceSlider.addEventListener('input', (e) => {
            this.groundTruthDistance = parseFloat(e.target.value);
            gtDistanceValue.textContent = this.groundTruthDistance.toFixed(1);
            this.updateOverlays(); // Redraw overlays with new distance
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
    }

    async loadRecordings() {
        const recordings = await this.dataLoader.loadRecordings();
        const select = document.getElementById('recording-select');
        select.innerHTML = '<option value="">Select recording...</option>';
        recordings.forEach(rec => {
            const option = document.createElement('option');
            option.value = rec.filename;
            option.textContent = rec.filename;
            select.appendChild(option);
        });
    }

    async loadSelectedRecording() {
        const select = document.getElementById('recording-select');
        const filename = select.value;
        if (!filename) return;
        
        try {
            this.frameCount = await this.dataLoader.loadRecording(filename);
            document.getElementById('frame-count').textContent = this.frameCount;
            document.getElementById('frame-slider').max = this.frameCount - 1;
            this.currentFrameIndex = 0;
            await this.goToFrame(0);
        } catch (error) {
            console.error('Error loading recording:', error);
            alert('Failed to load recording: ' + error.message);
        }
    }

    async goToFrame(frameIndex) {
        if (frameIndex < 0 || frameIndex >= this.frameCount) return;
        
        this.currentFrameIndex = frameIndex;
        document.getElementById('frame-slider').value = frameIndex;
        document.getElementById('frame-number').textContent = frameIndex;
        
        try {
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
        updateField('perception-using-stale', usingStale ? 'YES ⚠️' : 'NO');
        updateField('all-perception-using-stale', usingStale ? 'YES ⚠️' : 'NO');
        
        const staleReason = p.stale_data_reason || '-';
        updateField('perception-stale-reason', staleReason);
        updateField('all-perception-stale-reason', staleReason);
        
        const leftJump = p.left_jump_magnitude !== undefined ? p.left_jump_magnitude : null;
        updateField('perception-left-jump', leftJump !== null ? `${leftJump.toFixed(3)}m` : '-');
        updateField('all-perception-left-jump', leftJump !== null ? `${leftJump.toFixed(3)}m` : '-');
        
        const rightJump = p.right_jump_magnitude !== undefined ? p.right_jump_magnitude : null;
        updateField('perception-right-jump', rightJump !== null ? `${rightJump.toFixed(3)}m` : '-');
        updateField('all-perception-right-jump', rightJump !== null ? `${rightJump.toFixed(3)}m` : '-');
        
        const jumpThreshold = p.jump_threshold !== undefined ? p.jump_threshold : null;
        updateField('perception-jump-threshold', jumpThreshold !== null ? `${jumpThreshold.toFixed(2)}m` : '-');
        updateField('all-perception-jump-threshold', jumpThreshold !== null ? `${jumpThreshold.toFixed(2)}m` : '-');
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
            if (this.currentFrameData.perception && this.currentFrameData.perception.lane_line_coefficients) {
                const laneCoeffs = this.currentFrameData.perception.lane_line_coefficients;
                // Draw detected curves in orange to distinguish from the vertical lines
                this.overlayRenderer.drawDetectedLaneCurves(laneCoeffs, '#ff8800');
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
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!this.currentFrameData) return;
        
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
        
        // Load all images first, then composite them
        const images = [];
        for (const type of checkedTypes) {
            try {
                const imageUrl = await this.dataLoader.loadDebugImage(this.currentFrameIndex, type.name);
                if (imageUrl) {
                    const img = new Image();
                    await new Promise((resolve, reject) => {
                        img.onload = resolve;
                        img.onerror = reject;
                        img.src = imageUrl;
                    });
                    images.push(img);
                }
            } catch (error) {
                console.error(`Error loading debug image ${type.name}:`, error);
            }
        }
        
        // Composite all images (draw them on top of each other)
        // If only one image, draw it directly; otherwise blend them
        if (images.length === 1) {
            ctx.drawImage(images[0], 0, 0, canvas.width, canvas.height);
        } else if (images.length > 1) {
            // Draw first image at full opacity
            ctx.globalAlpha = 1.0;
            ctx.drawImage(images[0], 0, 0, canvas.width, canvas.height);
            // Draw remaining images with reduced opacity for blending
            ctx.globalAlpha = 0.5;
            for (let i = 1; i < images.length; i++) {
                ctx.drawImage(images[i], 0, 0, canvas.width, canvas.height);
            }
            ctx.globalAlpha = 1.0; // Reset
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
}

// Initialize visualizer when page loads
let visualizer;
document.addEventListener('DOMContentLoaded', () => {
    visualizer = new Visualizer();
});

