"""
Dataset classes for lane detection training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List
import h5py


class LaneDetectionDataset(Dataset):
    """Dataset for lane detection training."""
    
    def __init__(self, data_dir: str, transform: Optional[callable] = None, 
                 augment: bool = True):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing training data (HDF5 files or images)
            transform: Optional transform function
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        
        # Load data files
        self.data_files = self._find_data_files()
        self.samples = self._load_samples()
    
    def _find_data_files(self) -> List[Path]:
        """Find all data files in directory."""
        files = []
        
        # Look for HDF5 files
        h5_files = list(self.data_dir.glob("*.h5"))
        files.extend(h5_files)
        
        # Look for image directories
        img_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        files.extend(img_dirs)
        
        return files
    
    def _load_samples(self) -> List[dict]:
        """Load sample metadata."""
        samples = []
        
        for data_file in self.data_files:
            if data_file.suffix == '.h5':
                # Load from HDF5
                samples.extend(self._load_from_h5(data_file))
            elif data_file.is_dir():
                # Load from image directory
                samples.extend(self._load_from_dir(data_file))
        
        return samples
    
    def _load_from_h5(self, h5_file: Path) -> List[dict]:
        """Load samples from HDF5 file."""
        samples = []
        
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'camera/images' not in f:
                    return samples
                
                images = f['camera/images']
                timestamps = f['camera/timestamps']
                
                for i in range(len(images)):
                    samples.append({
                        'image_path': str(h5_file),
                        'image_idx': i,
                        'type': 'h5'
                    })
        except Exception as e:
            print(f"Error loading {h5_file}: {e}")
        
        return samples
    
    def _load_from_dir(self, img_dir: Path) -> List[dict]:
        """Load samples from image directory."""
        samples = []
        
        # Look for images
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in img_extensions:
            for img_file in img_dir.glob(f"*{ext}"):
                # Look for corresponding label file
                label_file = img_file.with_suffix('.txt')
                if label_file.exists():
                    samples.append({
                        'image_path': str(img_file),
                        'label_path': str(label_file),
                        'type': 'image'
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Tuple of (image, classification_target, existence_target)
        """
        sample = self.samples[idx]
        
        # Load image
        if sample['type'] == 'h5':
            image = self._load_image_from_h5(sample['image_path'], sample['image_idx'])
            # For now, create dummy labels (should be replaced with actual labels)
            cls_target, exist_target = self._create_dummy_labels()
        else:
            image = cv2.imread(sample['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cls_target, exist_target = self._load_labels(sample.get('label_path'))
        
        # Apply augmentation
        if self.augment:
            image, cls_target, exist_target = self._augment(image, cls_target, exist_target)
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, cls_target, exist_target
    
    def _load_image_from_h5(self, h5_path: str, idx: int) -> np.ndarray:
        """Load image from HDF5 file."""
        with h5py.File(h5_path, 'r') as f:
            image = f['camera/images'][idx]
            return image
    
    def _load_labels(self, label_path: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load labels from file."""
        # TODO: Implement actual label loading
        # For now, return dummy labels
        return self._create_dummy_labels()
    
    def _create_dummy_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create dummy labels for testing."""
        num_lanes = 4
        num_gridding = 200
        
        # Random classification target
        cls_target = torch.zeros(num_lanes, num_gridding)
        
        # Random existence target
        exist_target = torch.zeros(num_lanes)
        
        return cls_target, exist_target
    
    def _augment(self, image: np.ndarray, cls_target: torch.Tensor, 
                 exist_target: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Apply data augmentation."""
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.7, 1.3)
            image = (image * brightness_factor).clip(0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if np.random.rand() > 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean = image.mean()
            image = ((image - mean) * contrast_factor + mean).clip(0, 255).astype(np.uint8)
        
        # Random horizontal flip (with label adjustment)
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            # Flip lane positions in labels (would need proper implementation)
        
        return image, cls_target, exist_target

