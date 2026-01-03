"""
Data loading utilities for training.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import List, Iterator, Tuple, Optional


def load_h5_recording(recording_path: str) -> dict:
    """
    Load data from HDF5 recording file.
    
    Args:
        recording_path: Path to HDF5 file
    
    Returns:
        Dictionary with loaded data
    """
    data = {}
    
    with h5py.File(recording_path, 'r') as f:
        # Load camera frames
        if 'camera/images' in f:
            data['images'] = f['camera/images'][:]
            data['image_timestamps'] = f['camera/timestamps'][:]
        
        # Load vehicle states
        if 'vehicle/timestamps' in f:
            data['vehicle_timestamps'] = f['vehicle/timestamps'][:]
            data['vehicle_positions'] = f['vehicle/position'][:]
            data['vehicle_speeds'] = f['vehicle/speed'][:]
            data['vehicle_steering'] = f['vehicle/steering_angle'][:]
        
        # Load control commands
        if 'control/timestamps' in f:
            data['control_timestamps'] = f['control/timestamps'][:]
            data['control_steering'] = f['control/steering'][:]
            data['control_throttle'] = f['control/throttle'][:]
            data['control_brake'] = f['control/brake'][:]
    
    return data


def find_recordings(data_dir: str) -> List[Path]:
    """
    Find all recording files in directory.
    
    Args:
        data_dir: Directory to search
    
    Returns:
        List of recording file paths
    """
    data_path = Path(data_dir)
    recordings = list(data_path.glob("*.h5"))
    return recordings


def synchronize_data(images: np.ndarray, image_timestamps: np.ndarray,
                    vehicle_timestamps: np.ndarray, vehicle_data: np.ndarray) -> List[dict]:
    """
    Synchronize image and vehicle data by timestamp.
    
    Args:
        images: Image array
        image_timestamps: Image timestamps
        vehicle_timestamps: Vehicle timestamps
        vehicle_data: Vehicle data array
    
    Returns:
        List of synchronized samples
    """
    synchronized = []
    
    for i, img_ts in enumerate(image_timestamps):
        # Find closest vehicle timestamp
        closest_idx = np.argmin(np.abs(vehicle_timestamps - img_ts))
        
        synchronized.append({
            'image': images[i],
            'image_timestamp': img_ts,
            'vehicle_data': vehicle_data[closest_idx],
            'vehicle_timestamp': vehicle_timestamps[closest_idx]
        })
    
    return synchronized

