from backend.utils import process_folder
from backend.models import ImageQualityModel
import torch
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

# Paths
ref_folder = r'C:\Users\Xain M-k\Desktop\my AI\narwaria\reference'
dist_folder = r'C:\Users\Xain M-k\Desktop\my AI\narwaria\distorted'
output_csv = r'C:\Users\Xain M-k\Desktop\my AI\results\narwaria_scores.csv'
model_path = r'C:\Users\Xain M-k\Desktop\my AI\results\quality_model.pth'

# Ensure folders exist
for folder in [ref_folder, dist_folder]:
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Process folders and generate results with traditional metrics
print(f"Processing images from {ref_folder} and {dist_folder}")
print(f"Results will be saved to {output_csv}")

# Get traditional metrics
process_folder(ref_folder, dist_folder, output_csv)

# Load and use deep learning model if it exists
if os.path.exists(model_path):
    print("\nUsing deep learning model for additional predictions...")
    model = ImageQualityModel()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    
    # Read existing results
    results_df = pd.read_csv(output_csv)
    
    # Add deep learning predictions
    dl_predictions = []
    
    for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Making DL predictions"):
        # Load and preprocess image
        dist_path = os.path.join(dist_folder, row['Distorted'])
        dist_img = cv2.imread(dist_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)
        dist_img = dist_img.astype(np.float32) / dist_img.max()
        dist_img = torch.from_numpy(dist_img).permute(2, 0, 1).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            pred = model(dist_img)
            dl_predictions.append(pred.item())
    
    # Add predictions to results
    results_df['DL_Quality_Score'] = dl_predictions
    
    # Save updated results
    results_df.to_csv(output_csv, index=False)
    print("Added deep learning predictions to results")
else:
    print("\nDeep learning model not found. Please run train_model.py first to train the model.")

print("Processing complete!")