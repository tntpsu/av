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
            console.log(`Fetching recordings from ${API_BASE}/recordings`);
            const response = await fetch(`${API_BASE}/recordings`);
            console.log('Response status:', response.status, response.statusText);
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Failed to load recordings:', response.status, errorText);
                return [];
            }
            const data = await response.json();
            console.log('Recordings data received:', data);
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
    async loadFrameImage(frameIndex) {
        if (!this.currentRecording) {
            throw new Error('No recording loaded');
        }

        try {
            const response = await fetch(`${API_BASE}/recording/${this.currentRecording}/frame/${frameIndex}/image`);
            if (!response.ok) throw new Error(`Failed to load frame image ${frameIndex}`);
            const data = await response.json();
            return data.image; // Base64 data URL
        } catch (error) {
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

