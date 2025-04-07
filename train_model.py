import torch
from torch.utils.data import random_split, DataLoader
from backend.models import ImageQualityModel
from backend.trainer import Trainer, HDRImageDataset, HDRImageConfig
import os
import sys
import logging

# Configure logging (similar to trainer.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hdr_qa.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create configuration
    config = HDRImageConfig(
        ref_folder=r'C:\Users\Xain M-k\Desktop\my AI\narwaria\reference',
        dist_folder=r'C:\Users\Xain M-k\Desktop\my AI\narwaria\distorted',
        batch_size=2,  # Smaller batch size
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=10,  # Fewer epochs
        val_split=0.2,
        max_samples=20  # Limit the number of samples for faster testing
    )
    
    # Create dataset
    dataset = HDRImageDataset(config=config)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Initialize model and trainer
    model = ImageQualityModel()
    trainer = Trainer(model, device=device)
    
    # Train the model
    trainer.train(train_loader, val_loader, num_epochs=config.num_epochs)
    
    # Save the final model
    os.makedirs('results', exist_ok=True)
    trainer.save_model('results/quality_model.pth')

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    main()