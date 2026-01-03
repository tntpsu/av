"""
Vehicle dynamics model (bicycle model).
Used for control and simulation.
"""

import numpy as np
from typing import Tuple


class BicycleModel:
    """
    Bicycle model for vehicle dynamics.
    Simplified 2D model assuming no roll or pitch.
    """
    
    def __init__(self, wheelbase: float = 2.5, max_steering_angle: float = 0.5):
        """
        Initialize bicycle model.
        
        Args:
            wheelbase: Distance between front and rear axles (meters)
            max_steering_angle: Maximum steering angle (radians)
        """
        self.wheelbase = wheelbase
        self.max_steering_angle = max_steering_angle
    
    def update(self, x: float, y: float, heading: float, velocity: float,
               steering_angle: float, dt: float) -> Tuple[float, float, float]:
        """
        Update vehicle state using bicycle model.
        
        Args:
            x: Current x position
            y: Current y position
            heading: Current heading (radians)
            velocity: Current velocity (m/s)
            steering_angle: Steering angle (radians, clamped)
            dt: Time step (seconds)
        
        Returns:
            New (x, y, heading)
        """
        # Clamp steering angle
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        # Bicycle model equations
        # Assuming rear-wheel drive
        if abs(steering_angle) < 1e-6:
            # Straight line motion
            dx = velocity * np.cos(heading) * dt
            dy = velocity * np.sin(heading) * dt
            dheading = 0.0
        else:
            # Turning motion
            turning_radius = self.wheelbase / np.tan(steering_angle)
            angular_velocity = velocity / turning_radius
            
            dx = velocity * np.cos(heading) * dt
            dy = velocity * np.sin(heading) * dt
            dheading = angular_velocity * dt
        
        new_x = x + dx
        new_y = y + dy
        new_heading = heading + dheading
        
        # Normalize heading to [-pi, pi]
        new_heading = np.arctan2(np.sin(new_heading), np.cos(new_heading))
        
        return new_x, new_y, new_heading
    
    def compute_curvature(self, steering_angle: float) -> float:
        """
        Compute curvature from steering angle.
        
        Args:
            steering_angle: Steering angle (radians)
        
        Returns:
            Curvature (1/m)
        """
        if abs(steering_angle) < 1e-6:
            return 0.0
        
        turning_radius = self.wheelbase / np.tan(steering_angle)
        return 1.0 / turning_radius

