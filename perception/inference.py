"""
Perception inference pipeline for lane detection.
"""

import torch
import numpy as np
import cv2
from typing import List, Optional, Tuple
from pathlib import Path
import logging

from .models.lane_detection import LaneDetectionModel, SimpleLaneDetector, load_pretrained_model

logger = logging.getLogger(__name__)


class LaneDetectionInference:
    """Lane detection inference pipeline."""
    
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True, 
                 fallback_to_cv: bool = True):
        """
        Initialize lane detection inference.
        
        Args:
            model_path: Path to trained model checkpoint
            use_gpu: Whether to use GPU if available
            fallback_to_cv: Fallback to traditional CV if model fails
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.fallback_to_cv = fallback_to_cv
        
        # Track if model is trained (has checkpoint)
        self.model_trained = model_path is not None and Path(model_path).exists()
        
        # Load model
        if self.model_trained:
            self.model = load_pretrained_model(model_path)
            logger.info(f"Loaded trained model from {model_path}")
        else:
            self.model = LaneDetectionModel()
            logger.info("Using untrained model - will force CV fallback")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Fallback detector
        if fallback_to_cv:
            self.cv_detector = SimpleLaneDetector()
        
        # Track last detection method for recording
        self.last_detection_method = "ml"  # Default to ML
        
        # Image preprocessing parameters
        self.input_height = 320
        self.input_width = 800
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input RGB image
        
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Resize
        img_resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def postprocess(self, lane_coeffs: List[Optional[np.ndarray]], 
                   original_shape: Tuple[int, int]) -> List[Optional[np.ndarray]]:
        """
        Postprocess lane detection results.
        
        Args:
            lane_coeffs: List of polynomial coefficients
            original_shape: Original image shape (height, width)
        
        Returns:
            Processed lane coefficients
        """
        # Scale coefficients to original image size
        scale_h = original_shape[0] / self.input_height
        scale_w = original_shape[1] / self.input_width
        
        processed_lanes = []
        for coeffs in lane_coeffs:
            if coeffs is None:
                processed_lanes.append(None)
                continue
            
            # Adjust coefficients for scaling
            # For polynomial y = ax^2 + bx + c, we need to scale
            # This is approximate - full scaling would require recomputing
            scaled_coeffs = coeffs.copy()
            scaled_coeffs[0] *= (scale_w / (scale_h ** 2))  # x^2 term
            scaled_coeffs[1] *= (scale_w / scale_h)  # x term
            scaled_coeffs[2] *= scale_w  # constant term
            
            processed_lanes.append(scaled_coeffs)
        
        return processed_lanes
    
    def detect(self, image: np.ndarray) -> Tuple[List[Optional[np.ndarray]], float]:
        """
        Detect lane lines in image.
        
        Args:
            image: Input RGB image
        
        Returns:
            Tuple of (lane_coefficients, confidence)
        """
        original_shape = image.shape[:2]
        
        # If model is untrained, skip ML inference and go straight to CV fallback
        if not self.model_trained and self.fallback_to_cv:
            logger.info("Model untrained, using CV fallback")
            self.last_detection_method = "cv"
            try:
                cv_lanes = self.cv_detector.detect(image, return_debug=False)
                cv_lanes_detected = sum(1 for c in cv_lanes if c is not None)
                logger.info(f"CV fallback: detected {cv_lanes_detected} lanes")
                return cv_lanes, 0.5
            except Exception as cv_error:
                logger.error(f"CV fallback failed: {cv_error}")
                return [None, None], 0.0
        
        try:
            # Preprocess
            img_tensor = self.preprocess(image)
            
            # Inference
            with torch.no_grad():
                cls_logits, exist_logits = self.model(img_tensor)
            
            # Decode lanes
            lane_coeffs = self.model.decode_lanes(cls_logits, exist_logits)
            
            # Postprocess
            lane_coeffs = self.postprocess(lane_coeffs, original_shape)
            
            # Calculate confidence (average of existence probabilities)
            exist_probs = torch.sigmoid(exist_logits)
            confidence = exist_probs.mean().item()
            
            # Count detected lanes
            lanes_detected = sum(1 for c in lane_coeffs if c is not None)
            
            # Check if we should use CV fallback:
            # 1. Model confidence is very low (< 0.1), OR
            # 2. No lanes detected but confidence is moderate (untrained model giving false confidence)
            if (confidence < 0.1) or (lanes_detected == 0 and confidence < 0.6):
                # Model likely untrained or not working, try CV fallback
                if self.fallback_to_cv:
                    try:
                        logger.info(f"ML model detected {lanes_detected} lanes (conf={confidence:.2f}), trying CV fallback")
                        cv_lanes = self.cv_detector.detect(image, return_debug=False)
                        cv_lanes_detected = sum(1 for c in cv_lanes if c is not None)
                        if cv_lanes_detected > 0:
                            logger.info(f"CV fallback: detected {cv_lanes_detected} lanes (ML detected {lanes_detected})")
                            self.last_detection_method = "cv"
                            return cv_lanes, 0.5
                        else:
                            logger.debug("CV fallback found no lanes - may need parameter tuning")
                    except Exception as cv_error:
                        logger.error(f"CV fallback failed: {cv_error}")
            
            self.last_detection_method = "ml"
            return lane_coeffs, confidence
        
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to traditional CV
            if self.fallback_to_cv:
                try:
                    logger.info("Falling back to CV due to model inference error")
                    lane_coeffs = self.cv_detector.detect(image, return_debug=False)
                    cv_lanes_detected = sum(1 for c in lane_coeffs if c is not None)
                    logger.info(f"CV fallback detected {cv_lanes_detected} lanes")
                    self.last_detection_method = "cv"
                    return lane_coeffs, 0.5  # Lower confidence for CV fallback
                except Exception as cv_error:
                    logger.error(f"CV fallback also failed: {cv_error}")
            
            self.last_detection_method = "ml"  # Even if failed, we tried ML
            return [None, None], 0.0
    
    def visualize(self, image: np.ndarray, lane_coeffs: List[Optional[np.ndarray]], 
                  color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3) -> np.ndarray:
        """
        Visualize detected lane lines on image.
        
        Args:
            image: Input RGB image
            lane_coeffs: Lane polynomial coefficients
            color: Lane line color (BGR)
            thickness: Line thickness
        
        Returns:
            Image with lane lines drawn
        """
        vis_image = image.copy()
        h, w = image.shape[:2]
        
        for coeffs in lane_coeffs:
            if coeffs is None:
                continue
            
            # Generate points along lane
            y_points = np.linspace(h // 2, h, 50)
            x_points = np.polyval(coeffs, y_points)
            
            # Filter valid points
            valid_mask = (x_points >= 0) & (x_points < w)
            x_points = x_points[valid_mask]
            y_points = y_points[valid_mask]
            
            if len(x_points) < 2:
                continue
            
            # Draw lane line
            points = np.array([x_points, y_points], dtype=np.int32).T
            cv2.polylines(vis_image, [points], isClosed=False, color=color, thickness=thickness)
        
        return vis_image


if __name__ == "__main__":
    # Test inference
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m perception.inference <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    detector = LaneDetectionInference()
    lane_coeffs, confidence = detector.detect(image_rgb)
    
    print(f"Detected lanes with confidence: {confidence:.2f}")
    for i, coeffs in enumerate(lane_coeffs):
        if coeffs is not None:
            print(f"Lane {i}: {coeffs}")
    
    # Visualize
    vis_image = detector.visualize(image_rgb, lane_coeffs)
    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("Lane Detection", vis_image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

