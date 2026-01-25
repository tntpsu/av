/**
 * Overlay renderer for debug visualizer.
 * Renders lane lines, trajectory, and reference point on canvas.
 */

class OverlayRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.imageWidth = 640;
        this.imageHeight = 480;
        
        // Camera parameters (should match trajectory planner)
        // CRITICAL: Use Unity's actual horizontal FOV from recording (not hardcoded)
        // Default to 110° if not provided, but will be updated from recording data
        // Unity Inspector shows "Field of View Axis: Horizontal" with value 110°
        // But Unity's actual horizontal FOV might be different (e.g., 101.33° if vertical=84.93°)
        this.cameraFov = 110.0; // degrees (HORIZONTAL FOV - default, will be updated from recording)
        this.cameraHeight = 1.2; // meters (matches Unity camera height above ground)
        this.baseDistance = 1.5; // meters at bottom of image (tunable for perspective model)
    }
    
    /**
     * Update camera FOV from Unity's actual calculated horizontal FOV.
     * This ensures visualizer matches Unity's actual FOV, not the Inspector value.
     */
    setCameraFov(horizontalFov) {
        if (horizontalFov > 0 && horizontalFov < 180) {
            this.cameraFov = horizontalFov;
        }
    }
    
    /**
     * Update base distance for perspective model.
     * This controls how distance maps to Y pixel position.
     */
    setBaseDistance(baseDistance) {
        if (baseDistance > 0 && baseDistance < 10) {
            this.baseDistance = baseDistance;
        }
    }

    /**
     * Clear the overlay canvas.
     */
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /**
     * Convert vehicle coordinates to image coordinates.
     * Simplified version - assumes evaluation at a specific distance.
     */
    vehicleToImage(xVehicle, yVehicle, distance = 8.0) {
        // CRITICAL FIX: Use cameraFov directly as horizontal FOV
        // Unity Inspector shows "Horizontal: 110°", and Unity calculates horizontal = 110.00°
        // Config value (110°) matches Unity's horizontal FOV, so use it directly
        const horizontalFovRad = (this.cameraFov * Math.PI) / 180; // Use directly as horizontal FOV
        
        // Calculate width at distance
        const widthAtDistance = 2.0 * distance * Math.tan(horizontalFovRad / 2);
        const pixelToMeter = widthAtDistance / this.imageWidth;
        
        // Convert x (lateral)
        const xCenter = this.imageWidth / 2;
        const xImage = xCenter + (xVehicle / pixelToMeter);
        
        // Convert y (forward) - simplified perspective model
        // Use tunable baseDistance (can be adjusted via slider)
        const yNormalized = this.baseDistance / distance;
        const yFromBottom = yNormalized * this.imageHeight;
        const yImage = this.imageHeight - yFromBottom;
        
        return { x: xImage, y: yImage };
    }

    /**
     * Draw points used for polynomial fitting.
     * @param {Array} points - Array of [x, y] points in image coordinates
     * @param {string} color - Color for the points (default: different colors for left/right)
     * @param {number} pointSize - Size of each point (default: 3)
     */
    drawFitPoints(points, color = '#ff00ff', pointSize = 3) {
        if (!points || points.length === 0) return;
        
        this.ctx.fillStyle = color;
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 1;
        
        for (const point of points) {
            if (point && point.length >= 2) {
                const x = point[0];
                const y = point[1];
                
                // Only draw if point is within reasonable bounds
                if (x >= -this.imageWidth && x <= this.imageWidth * 2 && 
                    y >= 0 && y <= this.imageHeight) {
                    // Draw filled circle
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, pointSize, 0, 2 * Math.PI);
                    this.ctx.fill();
                    
                    // Draw outline for better visibility
                    this.ctx.stroke();
                }
            }
        }
    }

    /**
     * Draw lane line from polynomial coefficients.
     */
    drawLaneLine(coeffs, color = '#00ff00', lineWidth = 2) {
        if (!coeffs || coeffs.length < 2) return;
        
        // NEW: Validate polynomial coefficients to detect extreme values
        // Extreme coefficients can cause curves to go way outside image bounds
        // Check if coefficients are reasonable (within expected range for lane detection)
        if (coeffs.length >= 3) {
            const a = coeffs[0];
            const b = coeffs[1];
            const c = coeffs[2];
            
            // Evaluate at middle and bottom of image (where we have actual detection points)
            // Don't check top - extrapolation at top can be extreme on curves, which is OK
            const yMid = this.imageHeight / 2;
            const yBottom = this.imageHeight - 1;
            
            const xMid = a * yMid * yMid + b * yMid + c;
            const xBottom = a * yBottom * yBottom + b * yBottom + c;
            
            // Check if polynomial produces extreme values outside reasonable bounds
            // CRITICAL: Match the validation logic in av_stack.py
            // Only check where we actually use the polynomial (lookahead distance + bottom)
            // Use same thresholds as validation: 2.5x image width (more lenient for curves)
            const maxReasonableX = this.imageWidth * 2.5;  // Match av_stack.py validation
            const minReasonableX = -this.imageWidth * 1.5;  // Match av_stack.py validation
            
            // Only check middle and bottom (where we have actual detection points)
            // Don't check top of image - extrapolation there can be extreme on curves, which is OK
            if (xMid < minReasonableX || xMid > maxReasonableX ||
                xBottom < minReasonableX || xBottom > maxReasonableX) {
                // Extreme coefficients detected - log warning and use dashed line to indicate issue
                console.warn(`[OverlayRenderer] Extreme lane coefficients detected: a=${a.toFixed(6)}, b=${b.toFixed(6)}, c=${c.toFixed(6)}. ` +
                           `X values: mid=${xMid.toFixed(1)}, bottom=${xBottom.toFixed(1)} (range: ${minReasonableX.toFixed(0)} to ${maxReasonableX.toFixed(0)})`);
                // Use dashed line and different color to indicate invalid coefficients
                this.ctx.setLineDash([5, 5]);
                color = '#ff0000'; // Red for invalid
            } else {
                this.ctx.setLineDash([]); // Solid line for valid
            }
        }
        
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = lineWidth;
        this.ctx.beginPath();
        
        // Evaluate polynomial at multiple y positions
        // FIXED: Only draw curve within reasonable bounds to avoid showing extreme extrapolation
        const step = 10; // pixels
        let firstPoint = true;
        let hasExtremeValues = false;
        
        // Determine valid y range where polynomial produces reasonable x values
        // Only draw where x is within reasonable bounds (not extreme)
        const maxReasonableX = this.imageWidth * 2.5;
        const minReasonableX = -this.imageWidth * 1.5;
        let validYStart = null;
        let validYEnd = null;
        
        // Find valid y range
        for (let y = 0; y <= this.imageHeight; y += step) {
            let x = 0;
            if (coeffs.length >= 3) {
                x = coeffs[0] * y * y + coeffs[1] * y + coeffs[2];
            } else if (coeffs.length >= 2) {
                x = coeffs[0] * y + coeffs[1];
            } else {
                x = coeffs[0];
            }
            
            if (x >= minReasonableX && x <= maxReasonableX) {
                if (validYStart === null) {
                    validYStart = y;
                }
                validYEnd = y;
            } else {
                hasExtremeValues = true;
            }
        }
        
        // Only draw if we have a valid range
        if (validYStart !== null && validYEnd !== null && validYEnd > validYStart) {
            // Draw only within valid range
            for (let y = validYStart; y <= validYEnd; y += step) {
                let x = 0;
                if (coeffs.length >= 3) {
                    x = coeffs[0] * y * y + coeffs[1] * y + coeffs[2];
                } else if (coeffs.length >= 2) {
                    x = coeffs[0] * y + coeffs[1];
                } else {
                    x = coeffs[0];
                }
                
                // Clamp to image bounds for display
                x = Math.max(0, Math.min(this.imageWidth - 1, x));
                
                if (firstPoint) {
                    this.ctx.moveTo(x, y);
                    firstPoint = false;
                } else {
                    this.ctx.lineTo(x, y);
                }
            }
            
            this.ctx.stroke();
            
            // Log if we had to clip due to extreme values
            if (hasExtremeValues) {
                console.warn(`[OverlayRenderer] Clipped lane curve drawing: valid range y=${validYStart}-${validYEnd}px (extreme values outside this range)`);
            }
        } else {
            // No valid range - polynomial is extreme everywhere
            console.warn(`[OverlayRenderer] Skipping lane curve drawing: polynomial produces extreme values at all y positions`);
        }
        
        this.ctx.setLineDash([]); // Reset to solid for next draw
    }

    /**
     * Draw lane lines from vehicle coordinates.
     * Assumes left_lane_line_x and right_lane_line_x are at a specific distance.
     * 
     * @param {number} leftLaneLineX - Left lane line (painted marking) x position in vehicle coords (meters)
     * @param {number} rightLaneLineX - Right lane line (painted marking) x position in vehicle coords (meters)
     * @param {number} distance - Lookahead distance (meters)
     * @param {string} color - Line color (hex or CSS color), default red for detected lanes
     */
    drawLaneLinesFromVehicleCoords(leftLaneLineX, rightLaneLineX, distance = 8.0, color = '#ff0000', yLookaheadOverride = null) {
        if (leftLaneLineX === null || rightLaneLineX === null || leftLaneLineX === undefined || rightLaneLineX === undefined) {
            console.log('[OVERLAY] Skipping lane lines: leftLaneLineX=', leftLaneLineX, 'rightLaneLineX=', rightLaneLineX);
            return;
        }
        
        // Calculate y position for the lookahead distance (where measurements are taken)
        // NEW: Use actual 8m position from Unity if available, otherwise fallback to 350px
        // yLookaheadOverride comes from camera_8m_screen_y (actual position where 8m appears)
        const yLookahead = yLookaheadOverride !== null && yLookaheadOverride > 0 ? yLookaheadOverride : 350;
        
        // CRITICAL FIX: Calculate x positions at yLookahead, not using vehicleToImage's y calculation
        // vehicleToImage calculates y from distance, but we want x positions at y=350px specifically
        // So we calculate x directly from vehicle coords at the specified y position
        // CRITICAL: Use cameraFov directly as horizontal FOV (matches Unity Inspector "Horizontal: 110°")
        const horizontalFovRad = (this.cameraFov * Math.PI) / 180; // Use directly as horizontal FOV
        
        // Calculate width at the evaluation distance
        // NOTE: This formula assumes camera is looking straight ahead horizontally
        // If camera is tilted down, the effective FOV at ground level changes
        // The actual issue might be that we need to account for camera pitch angle
        const widthAtDistance = 2.0 * distance * Math.tan(horizontalFovRad / 2);
        const pixelToMeter = widthAtDistance / this.imageWidth;
        
        // Convert x (lateral) positions - center of image is x=0 in vehicle coords
        const xCenter = this.imageWidth / 2;
        const leftX = xCenter + (leftLaneLineX / pixelToMeter);
        const rightX = xCenter + (rightLaneLineX / pixelToMeter);
        
        // NOTE: Black reference line is now drawn separately in updateOverlays()
        // to ensure it's always visible and uses the cached y8mActual value
        // We don't draw it here anymore to avoid duplicate lines
        
        // Draw as vertical lines (simplified - actual lanes curve)
        // Draw from bottom 1/3 to bottom (where lanes are visible)
        const yStart = Math.floor(this.imageHeight * 0.33);
        const yEnd = this.imageHeight;
        
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([]); // Solid lines
        this.ctx.beginPath();
        this.ctx.moveTo(leftX, yStart);
        this.ctx.lineTo(leftX, yEnd);
        this.ctx.stroke();
        
        this.ctx.beginPath();
        this.ctx.moveTo(rightX, yStart);
        this.ctx.lineTo(rightX, yEnd);
        this.ctx.stroke();
    }

    /**
     * Draw trajectory points.
     */
    drawTrajectory(trajectoryPoints, color = '#ff00ff', lineWidth = 2) {
        if (!trajectoryPoints || trajectoryPoints.length === 0) return;
        
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = lineWidth;
        this.ctx.beginPath();
        
        let firstPoint = true;
        for (const point of trajectoryPoints) {
            const pos = this.vehicleToImage(point.x, point.y, point.y);
            if (firstPoint) {
                this.ctx.moveTo(pos.x, pos.y);
                firstPoint = false;
            } else {
                this.ctx.lineTo(pos.x, pos.y);
            }
        }
        
        this.ctx.stroke();
    }

    /**
     * Draw black reference line at specified y pixel (shows where 8m appears).
     */
    drawReferenceLine(yPixel) {
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]); // Dashed line
        this.ctx.beginPath();
        this.ctx.moveTo(0, yPixel);
        this.ctx.lineTo(this.imageWidth, yPixel);
        this.ctx.stroke();
        this.ctx.setLineDash([]); // Reset to solid
    }

    /**
     * Draw reference point.
     */
    drawReferencePoint(refPoint, color = '#ff0000', size = 8, yLookaheadOverride = null) {
        if (!refPoint || refPoint.x === undefined || refPoint.y === undefined) return;
        
        // NEW: Use actual 8m position from Unity if available (more accurate)
        // Otherwise fall back to simplified model
        const distance = refPoint.y; // Forward distance (lookahead, typically 8m)
        let pos;
        if (yLookaheadOverride !== null && yLookaheadOverride > 0 && Math.abs(distance - 8.0) < 1.5) {
            // Use actual 8m position from Unity (matches black line)
            // CHANGED: Allow distances from 6.5m to 9.5m (was 7.9-8.1m) to handle config variations
            const horizontalFovRad = (this.cameraFov * Math.PI) / 180;
            const widthAtDistance = 2.0 * distance * Math.tan(horizontalFovRad / 2);
            const pixelToMeter = widthAtDistance / this.imageWidth;
            const xCenter = this.imageWidth / 2;
            const xImage = xCenter + (refPoint.x / pixelToMeter);
            const yImage = yLookaheadOverride; // Use actual 8m position from Unity
            pos = { x: xImage, y: yImage };
        } else {
            // Fallback to simplified model
            pos = this.vehicleToImage(refPoint.x, refPoint.y, refPoint.y);
        }
        
        // Draw circle
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.arc(pos.x, pos.y, size, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Draw heading arrow
        if (refPoint.heading !== undefined) {
            const heading = refPoint.heading;
            const arrowLength = 30;
            const arrowX = pos.x + Math.sin(heading) * arrowLength;
            const arrowY = pos.y - Math.cos(heading) * arrowLength;
            
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(pos.x, pos.y);
            this.ctx.lineTo(arrowX, arrowY);
            this.ctx.stroke();
            
            // Draw arrowhead
            const angle = Math.atan2(arrowY - pos.y, arrowX - pos.x);
            const arrowSize = 8;
            this.ctx.beginPath();
            this.ctx.moveTo(arrowX, arrowY);
            this.ctx.lineTo(
                arrowX - arrowSize * Math.cos(angle - Math.PI / 6),
                arrowY - arrowSize * Math.sin(angle - Math.PI / 6)
            );
            this.ctx.lineTo(
                arrowX - arrowSize * Math.cos(angle + Math.PI / 6),
                arrowY - arrowSize * Math.sin(angle + Math.PI / 6)
            );
            this.ctx.closePath();
            this.ctx.fill();
        }
    }

    /**
     * Draw ground truth lane lines.
     * 
     * @param {number} leftLaneLineX - Left lane line (painted marking) x position in vehicle coords (meters)
     * @param {number} rightLaneLineX - Right lane line (painted marking) x position in vehicle coords (meters)
     * @param {number} distance - Lookahead distance (meters)
     */
    drawGroundTruth(leftLaneLineX, rightLaneLineX, distance = 8.0, yLookaheadOverride = null) {
        if (leftLaneLineX === null || rightLaneLineX === null || leftLaneLineX === undefined || rightLaneLineX === undefined) {
            console.log('[OVERLAY] Skipping ground truth: leftLaneLineX=', leftLaneLineX, 'rightLaneLineX=', rightLaneLineX);
            return;
        }
        
        // Draw ground truth in green (solid lines) to match our visualization tool
        // This allows easy comparison: Green = GT, Red = Detected
        // NEW: Pass yLookaheadOverride to use actual 8m position from Unity
        this.drawLaneLinesFromVehicleCoords(leftLaneLineX, rightLaneLineX, distance, '#00ff00', yLookaheadOverride);
    }

    /**
     * Draw detected lane curves from polynomial coefficients.
     * These show the actual curves detected by the lane detection algorithm in image space.
     * 
     * @param {Array<Array<number>>} laneCoeffsList - List of coefficient arrays, one per lane
     * @param {string} color - Line color (hex or CSS color), default orange for detected curves
     */
    drawDetectedLaneCurves(laneCoeffsList, color = '#ff8800') {
        if (!laneCoeffsList || laneCoeffsList.length === 0) return;
        
        // Draw each lane curve
        for (const coeffs of laneCoeffsList) {
            if (coeffs && coeffs.length >= 2) {
                this.drawLaneLine(coeffs, color, 2);
            }
        }
    }

    /**
     * Draw red vertical lines directly from polynomial coefficients at y=350px.
     * This ensures red lines match orange curves at the black reference line.
     * 
     * @param {Array<Array<number>>} laneCoeffsList - List of coefficient arrays, one per lane
     * @param {number} yLookahead - Y pixel position to evaluate (default 350px)
     * @param {string} color - Line color (default red)
     */
    drawLaneLinesFromCoefficients(laneCoeffsList, yLookahead = 350, color = '#ff0000', yLookaheadOverride = null) {
        // NEW: Use actual 8m position from Unity if available, otherwise use provided default
        if (yLookaheadOverride !== null && yLookaheadOverride > 0) {
            yLookahead = yLookaheadOverride;
        }
        if (!laneCoeffsList || !Array.isArray(laneCoeffsList) || laneCoeffsList.length < 2) {
            console.warn(`[OverlayRenderer] drawLaneLinesFromCoefficients: Invalid coefficients. Got:`, laneCoeffsList);
            return;
        }
        
        const yStart = Math.floor(this.imageHeight * 0.33);
        const yEnd = this.imageHeight;
        const imageCenter = this.imageWidth / 2;
        
        // Evaluate polynomials at yLookahead to get x positions for all lanes
        const lanePositions = [];
        for (let i = 0; i < laneCoeffsList.length; i++) {
            const laneCoeffs = laneCoeffsList[i];
            if (!laneCoeffs || !Array.isArray(laneCoeffs) || laneCoeffs.length < 3) {
                console.warn(`[OverlayRenderer] drawLaneLinesFromCoefficients: Lane ${i} has invalid coefficients:`, laneCoeffs);
                continue;
            }
            
            // Evaluate polynomial: x = a*y^2 + b*y + c (quadratic)
            const x = laneCoeffs[0] * yLookahead * yLookahead + laneCoeffs[1] * yLookahead + laneCoeffs[2];
            
            // Clamp to image bounds for drawing
            const clampedX = Math.max(0, Math.min(this.imageWidth - 1, x));
            lanePositions.push(clampedX);
        }
        
        // Sort by x position and assign: leftmost = left lane, rightmost = right lane
        // This handles cases where both lanes might be on the same side of image center
        lanePositions.sort((a, b) => a - b);
        const leftX = lanePositions.length >= 1 ? lanePositions[0] : null;
        const rightX = lanePositions.length >= 2 ? lanePositions[lanePositions.length - 1] : null;
        
        // Draw red vertical lines at the calculated x positions
        if (leftX !== null && rightX !== null) {
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 3;
            this.ctx.setLineDash([]); // Solid lines
            this.ctx.beginPath();
            this.ctx.moveTo(leftX, yStart);
            this.ctx.lineTo(leftX, yEnd);
            this.ctx.stroke();
            this.ctx.beginPath();
            this.ctx.moveTo(rightX, yStart);
            this.ctx.lineTo(rightX, yEnd);
            this.ctx.stroke();
        } else {
            console.warn(`[OverlayRenderer] drawLaneLinesFromCoefficients: Could not determine both lane positions. leftX=${leftX}, rightX=${rightX}, yLookahead=${yLookahead}, numLanes=${laneCoeffsList.length}`);
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = OverlayRenderer;
}

