import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio.v3 as iio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hdr_qa.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class HDRImageConfig:
    """Configuration for HDR image processing"""
    ref_folder: str
    dist_folder: str
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_samples: Optional[int] = None
    early_stop_patience: int = 10
    num_epochs: int = 50
    val_split: float = 0.2

class HDRFileError(Exception):
    """Custom exception for HDR file processing errors"""
    pass

class HDRImageReader:
    """EXR reader with multiple fallback methods and clear error reporting"""
    
    @staticmethod
    def read_exr(path: str) -> np.ndarray:
        """Read EXR file with comprehensive error handling"""
        path = str(Path(path).resolve())
        
        try:
            print(f"Reading EXR file: {path}")
            # Read image using imageio
            img = iio.imread(path)
            print(f"Image shape: {img.shape}")
            
            # Convert to float32 and normalize
            img = img.astype(np.float32)
            img = img / (np.max(img) + 1e-7)
            
            return img
            
        except Exception as ex:
            print(f"Error reading EXR file: {str(ex)}")
            raise HDRFileError(f"Could not read EXR file {path}") from ex

class HDRImageDataset(Dataset):
    """Dataset for paired HDR image quality assessment"""
    
    def __init__(self, config: HDRImageConfig, transform=None, verify: bool = True):
        self.config = config
        self.transform = transform
        self.image_pairs = []
        self.reader = HDRImageReader()
        self.cache = {}  # Cache for loaded images
        
        self._validate_folders()
        self._discover_image_pairs()
        
        if verify and not self._verify_dataset():
            raise ValueError("Dataset verification failed")
    
    def _validate_folders(self):
        """Validate input folders structure"""
        for folder in [self.config.ref_folder, self.config.dist_folder]:
            if not Path(folder).is_dir():
                raise ValueError(f"Folder does not exist: {folder}")
            if not any(f.suffix.lower() == '.exr' for f in Path(folder).iterdir()):
                raise ValueError(f"No EXR files found in: {folder}")
    
    def _discover_image_pairs(self):
        """Match reference images with their distorted versions"""
        ref_files = sorted(Path(self.config.ref_folder).glob('*.exr'))
        dist_files = sorted(Path(self.config.dist_folder).glob('*.exr'))
        
        dist_map = {}
        for df in dist_files:
            base_name = df.stem.split('_')[0]
            dist_map.setdefault(base_name, []).append(df)
        
        for rf in ref_files:
            base_name = rf.stem
            if base_name in dist_map:
                self.image_pairs.extend((rf, df) for df in dist_map[base_name])
        
        if not self.image_pairs:
            raise ValueError("No matching image pairs found")
        
        if self.config.max_samples:
            self.image_pairs = self.image_pairs[:self.config.max_samples]
        
        logger.info(f"Created {len(self.image_pairs)} image pairs")
    
    def _verify_dataset(self) -> bool:
        """Verify sample images can be loaded"""
        sample_size = min(3, len(self.image_pairs))
        logger.info(f"Verifying first {sample_size} image pairs...")
        
        for i in range(sample_size):
            try:
                ref_path, dist_path = self.image_pairs[i]
                self.reader.read_exr(str(ref_path))
                self.reader.read_exr(str(dist_path))
                logger.info(f"Verified pair {i+1}")
            except Exception as e:
                logger.error(f"Failed to verify pair {i+1}: {str(e)}")
                return False
        return True
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ref_path, dist_path = self.image_pairs[idx]
        
        # Check cache first
        ref_key = str(ref_path)
        dist_key = str(dist_path)
        
        if ref_key in self.cache:
            ref_img = self.cache[ref_key]
        else:
            ref_img = self.reader.read_exr(str(ref_path))
            self.cache[ref_key] = ref_img
        
        if dist_key in self.cache:
            dist_img = self.cache[dist_key]
        else:
            dist_img = self.reader.read_exr(str(dist_path))
            self.cache[dist_key] = dist_img
        
        # Convert to tensors
        ref_tensor = torch.from_numpy(ref_img).permute(2, 0, 1).float()
        dist_tensor = torch.from_numpy(dist_img).permute(2, 0, 1).float()
        
        if self.transform:
            ref_tensor = self.transform(ref_tensor)
            dist_tensor = self.transform(dist_tensor)
            
        return ref_tensor, dist_tensor

class HDRQualityModel(nn.Module):
    """CNN model for HDR image quality assessment"""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.regressor(x)

class Trainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_loader, val_loader=None, num_epochs=50):
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model('best_model.pth')
                
                print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}')
        
        # Plot training history
        self._plot_history()
    
    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (ref_imgs, dist_imgs) in enumerate(train_loader):
            ref_imgs = ref_imgs.to(self.device)
            dist_imgs = dist_imgs.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            quality_scores = self.model(dist_imgs)
            
            # Calculate loss (using reference images as ground truth quality)
            # We'll use the mean pixel value as a simple quality metric
            ref_quality = ref_imgs.mean(dim=[2, 3]).view(-1, 1)
            
            # Ensure batch sizes match
            if quality_scores.size(0) != ref_quality.size(0):
                # If sizes don't match, use the smaller size
                min_size = min(quality_scores.size(0), ref_quality.size(0))
                quality_scores = quality_scores[:min_size]
                ref_quality = ref_quality[:min_size]
            
            loss = self.criterion(quality_scores, ref_quality)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for ref_imgs, dist_imgs in val_loader:
                ref_imgs = ref_imgs.to(self.device)
                dist_imgs = dist_imgs.to(self.device)
                
                quality_scores = self.model(dist_imgs)
                ref_quality = ref_imgs.mean(dim=[2, 3]).view(-1, 1)
                
                # Ensure batch sizes match
                if quality_scores.size(0) != ref_quality.size(0):
                    # If sizes don't match, use the smaller size
                    min_size = min(quality_scores.size(0), ref_quality.size(0))
                    quality_scores = quality_scores[:min_size]
                    ref_quality = ref_quality[:min_size]
                
                loss = self.criterion(quality_scores, ref_quality)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _plot_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig('training_history.png')
        plt.close()
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']

