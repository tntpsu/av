/**
 * Data loader for debug visualizer.
 * Handles loading frame data from server API and debug visualization images.
 */

const API_BASE = 'http://localhost:5001/api';

class DataLoader {
    constructor() {
        this.currentRecording = null;
        this.frameCount = 0;
        this.frameCache = new Map();
        this.debugImageCache = new Map();
    }

    /**
     * Load list of available recordings.
     */
    async loadRecordings() {
        try {
            const response = await fetch(`${API_BASE}/recordings`);
            if (!response.ok) {
                return [];
            }
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error loading recordings:', error);
            return [];
        }
    }

    /**
     * Load recording metadata (frame count).
     */
    async loadRecording(filename) {
        try {
            const response = await fetch(`${API_BASE}/recording/${filename}/frames`);
            if (!response.ok) throw new Error('Failed to load recording');
            const data = await response.json();
            this.currentRecording = filename;
            this.frameCount = data.frame_count;
            return data.frame_count;
        } catch (error) {
            console.error('Error loading recording:', error);
            throw error;
        }
    }

    /**
     * Load recording metadata.
     */
    async loadRecordingMeta(filename = null) {
        const target = filename || this.currentRecording;
        if (!target) {
            throw new Error('No recording specified');
        }
        try {
            const response = await fetch(`${API_BASE}/recording/${target}/meta`, { cache: 'no-store' });
            if (!response.ok) throw new Error('Failed to load recording metadata');
            return await response.json();
        } catch (error) {
            console.error('Error loading recording metadata:', error);
            throw error;
        }
    }

    /**
     * Load run-level trajectory layer localization summary.
     */
    async loadTrajectoryLayerLocalization(filename = null, clipLimitM = null) {
        const target = filename || this.currentRecording;
        if (!target) {
            throw new Error('No recording specified');
        }
        const params = new URLSearchParams();
        if (Number.isFinite(Number(clipLimitM))) {
            params.set('clip_limit_m', String(Number(clipLimitM)));
        }
        const suffix = params.toString() ? `?${params.toString()}` : '';
        const response = await fetch(`${API_BASE}/recording/${target}/trajectory-layer-localization${suffix}`, { cache: 'no-store' });
        if (!response.ok) throw new Error('Failed to load trajectory layer localization');
        return response.json();
    }

    /**
     * Load available numeric signals for the current recording.
     */
    async loadSignals() {
        if (!this.currentRecording) {
            throw new Error('No recording loaded');
        }
        try {
            const response = await fetch(`${API_BASE}/recording/${this.currentRecording}/signals`);
            if (!response.ok) throw new Error('Failed to load signals');
            const data = await response.json();
            return data.signals || [];
        } catch (error) {
            console.error('Error loading signals:', error);
            throw error;
        }
    }

    /**
     * Load time-series data for selected signals.
     */
    async loadTimeSeries(signals, timeKey = null) {
        if (!this.currentRecording) {
            throw new Error('No recording loaded');
        }
        const params = new URLSearchParams();
        params.set('signals', signals.join(','));
        if (timeKey) {
            params.set('time', timeKey);
        }
        const response = await fetch(`${API_BASE}/recording/${this.currentRecording}/timeseries?${params.toString()}`);
        if (!response.ok) throw new Error('Failed to load time series');
        return response.json();
    }

    /**
     * Load frame data for a specific frame index.
     */
    async loadFrameData(frameIndex) {
        if (!this.currentRecording) {
            throw new Error('No recording loaded');
        }

        // Check cache
        const cacheKey = `${this.currentRecording}_${frameIndex}`;
        if (this.frameCache.has(cacheKey)) {
            return this.frameCache.get(cacheKey);
        }

        try {
            const response = await fetch(`${API_BASE}/recording/${this.currentRecording}/frame/${frameIndex}`);
            if (!response.ok) throw new Error(`Failed to load frame ${frameIndex}`);
            const data = await response.json();
            
            // Cache the data
            this.frameCache.set(cacheKey, data);
            
            // Limit cache size
            if (this.frameCache.size > 50) {
                const firstKey = this.frameCache.keys().next().value;
                this.frameCache.delete(firstKey);
            }
            
            return data;
        } catch (error) {
            console.error(`Error loading frame ${frameIndex}:`, error);
            throw error;
        }
    }

    /**
     * Load camera frame image as base64.
     */
    async loadFrameImage(frameIndex, cameraId = 'front_center') {
        if (!this.currentRecording) {
            throw new Error('No recording loaded');
        }

        try {
            const params = new URLSearchParams();
            if (cameraId) {
                params.set('camera_id', cameraId);
            }
            params.set('format', 'png');
            const response = await fetch(
                `${API_BASE}/recording/${this.currentRecording}/frame/${frameIndex}/image?${params.toString()}`,
                { cache: 'no-store' }
            );
            if (!response.ok) {
                const error = new Error(`Failed to load frame image ${frameIndex}`);
                error.status = response.status;
                error.cameraId = cameraId;
                throw error;
            }
            const blob = await response.blob();
            if (!blob || blob.size === 0) {
                throw new Error(`Empty image payload for frame ${frameIndex}, camera ${cameraId}`);
            }
            const imageUrl = URL.createObjectURL(blob);
            return imageUrl;
        } catch (error) {
            // Fallback to JSON data URL path when blob decoding path fails in-browser.
            if (!(error && error.status === 404)) {
                try {
                    const fallbackParams = new URLSearchParams();
                    if (cameraId) {
                        fallbackParams.set('camera_id', cameraId);
                    }
                    const fallbackResp = await fetch(
                        `${API_BASE}/recording/${this.currentRecording}/frame/${frameIndex}/image?${fallbackParams.toString()}`,
                        { cache: 'no-store' }
                    );
                    if (!fallbackResp.ok) {
                        const fallbackErr = new Error(`Failed fallback frame image load ${frameIndex}`);
                        fallbackErr.status = fallbackResp.status;
                        fallbackErr.cameraId = cameraId;
                        throw fallbackErr;
                    }
                    const data = await fallbackResp.json();
                    if (!data || typeof data.image !== 'string' || !data.image.startsWith('data:image/')) {
                        throw new Error(`Invalid fallback image payload for frame ${frameIndex}, camera ${cameraId}`);
                    }
                    return data.image;
                } catch (fallbackError) {
                    console.error(`Error loading frame image ${frameIndex} (fallback also failed):`, fallbackError);
                    throw fallbackError;
                }
            }
            console.error(`Error loading frame image ${frameIndex}:`, error);
            throw error;
        }
    }

    /**
     * Load debug visualization image.
     */
    async loadDebugImage(frameIndex, type = 'combined') {
        const frameId = `frame_${String(frameIndex).padStart(6, '0')}`;
        let imageName;
        
        // Handle special case for histogram
        if (type === 'histogram') {
            imageName = `line_histogram_${String(frameIndex).padStart(6, '0')}`;
        } else {
            imageName = type === 'combined' ? frameId : `${frameId}_${type}`;
        }
        
        const cacheKey = `${this.currentRecording}_${imageName}`;

        // Check cache
        if (this.debugImageCache.has(cacheKey)) {
            return this.debugImageCache.get(cacheKey);
        }

        try {
            const response = await fetch(`${API_BASE}/debug/${imageName}.png`);
            if (!response.ok) {
                // Image might not exist, return null
                return null;
            }
            
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            
            // Cache the URL
            this.debugImageCache.set(cacheKey, url);
            
            return url;
        } catch (error) {
            console.error(`Error loading debug image ${imageName}:`, error);
            return null;
        }
    }

    /**
     * Format frame number for debug image filename.
     */
    formatFrameId(frameIndex) {
        return `frame_${String(frameIndex).padStart(6, '0')}`;
    }

    /**
     * Clear caches.
     */
    clearCache() {
        // Revoke object URLs
        for (const url of this.debugImageCache.values()) {
            URL.revokeObjectURL(url);
        }
        this.frameCache.clear();
        this.debugImageCache.clear();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DataLoader;
}

