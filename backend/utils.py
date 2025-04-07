import numpy as np
import torch
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from piq import psnr
from imquality import brisque
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import logging
from tqdm import tqdm
import scipy.stats as stats
from scipy.ndimage import gaussian_filter

# Enable OpenEXR support in OpenCV


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hdr_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def read_hdr_image(path):
    """Read HDR/EXR image and return as RGB float32 array"""
    try:
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            if np.any(img < 0):
                img = np.clip(img, 0, None)
            return img
        else:
            raise ValueError(f"Failed to read image: {path}")
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        raise

def normalize_image(img):
    """Normalize image to [0,1] range"""
    max_val = img.max()
    if max_val > 1.0:
        return img / max_val
    return img

def tone_map(img, method='reinhard'):
    """Tone mapping operators for HDR images"""
    if method == 'reinhard':
        # Reinhard tone mapping
        img_tone = img / (1.0 + img)
        return np.clip(img_tone, 0, 1)
    elif method == 'gamma':
        # Gamma correction
        gamma = 2.2
        img_tone = np.power(img, 1/gamma)
        return np.clip(img_tone, 0, 1)
    elif method == 'log':
        # Logarithmic tone mapping
        img_tone = np.log1p(img) / np.log1p(img.max())
        return np.clip(img_tone, 0, 1)
    else:
        raise ValueError(f"Unknown tone mapping method: {method}")

def calculate_ssim(ref_img, dist_img):
    """Calculate SSIM by converting to grayscale first"""
    ref_gray = rgb2gray(ref_img)
    dist_gray = rgb2gray(dist_img)
    return ssim(ref_gray, dist_gray, data_range=1.0)

def compute_pu_mssim(ref, dist):
    """Compute PU-MSSIM with tone mapping"""
    ref_tm = tone_map(ref)
    dist_tm = tone_map(dist)
    ref_gray = rgb2gray(ref_tm)
    dist_gray = rgb2gray(dist_tm)
    return ssim(ref_gray, dist_gray, data_range=1.0)

def compute_hdr_vdp(ref, dist):
    """Compute HDR-VDP (simplified version)"""
    # Convert to luminance
    ref_lum = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    dist_lum = 0.2126 * dist[:,:,0] + 0.7152 * dist[:,:,1] + 0.0722 * dist[:,:,2]
    
    # Apply tone mapping
    ref_tm = tone_map(ref_lum)
    dist_tm = tone_map(dist_lum)
    
    # Compute difference
    diff = np.abs(ref_tm - dist_tm)
    
    # Apply visibility threshold
    threshold = 0.01
    visible_diff = np.mean(diff > threshold)
    
    return 1.0 - visible_diff  # Higher score means better quality

def compute_hdr_psnr(ref, dist):
    """Compute HDR-PSNR"""
    # Convert to luminance
    ref_lum = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    dist_lum = 0.2126 * dist[:,:,0] + 0.7152 * dist[:,:,1] + 0.0722 * dist[:,:,2]
    
    # Apply tone mapping
    ref_tm = tone_map(ref_lum)
    dist_tm = tone_map(dist_lum)
    
    # Compute MSE
    mse = np.mean((ref_tm - dist_tm) ** 2)
    
    # Compute PSNR
    if mse == 0:
        return 100.0
    else:
        return 20 * np.log10(1.0 / np.sqrt(mse))

def compute_hdr_ssim(ref, dist):
    """Compute HDR-SSIM"""
    # Convert to luminance
    ref_lum = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    dist_lum = 0.2126 * dist[:,:,0] + 0.7152 * dist[:,:,1] + 0.0722 * dist[:,:,2]
    
    # Apply tone mapping
    ref_tm = tone_map(ref_lum)
    dist_tm = tone_map(dist_lum)
    
    # Compute SSIM
    return ssim(ref_tm, dist_tm, data_range=1.0)

def compute_hdr_mssim(ref, dist):
    """Compute HDR-MSSIM (Multi-Scale SSIM)"""
    # Convert to luminance
    ref_lum = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    dist_lum = 0.2126 * dist[:,:,0] + 0.7152 * dist[:,:,1] + 0.0722 * dist[:,:,2]
    
    # Apply tone mapping
    ref_tm = tone_map(ref_lum)
    dist_tm = tone_map(dist_lum)
    
    # Compute MSSIM (simplified version)
    scales = [1.0, 0.5, 0.25]
    mssim_values = []
    
    for scale in scales:
        if scale != 1.0:
            ref_scaled = cv2.resize(ref_tm, None, fx=scale, fy=scale)
            dist_scaled = cv2.resize(dist_tm, None, fx=scale, fy=scale)
        else:
            ref_scaled = ref_tm
            dist_scaled = dist_tm
        
        mssim_values.append(ssim(ref_scaled, dist_scaled, data_range=1.0))
    
    # Weighted average of SSIM at different scales
    weights = [0.5, 0.3, 0.2]
    return np.sum(np.array(mssim_values) * np.array(weights))

def compute_hdr_psnr_xyb(ref, dist):
    """Compute HDR-PSNR in XYB color space"""
    # Convert to XYB color space (simplified)
    ref_xyb = np.zeros_like(ref)
    dist_xyb = np.zeros_like(dist)
    
    # X channel (luminance)
    ref_xyb[:,:,0] = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    dist_xyb[:,:,0] = 0.2126 * dist[:,:,0] + 0.7152 * dist[:,:,1] + 0.0722 * dist[:,:,2]
    
    # Y channel (green-red)
    ref_xyb[:,:,1] = 0.5 * (ref[:,:,1] - ref[:,:,0])
    dist_xyb[:,:,1] = 0.5 * (dist[:,:,1] - dist[:,:,0])
    
    # B channel (blue-yellow)
    ref_xyb[:,:,2] = 0.25 * (ref[:,:,2] - ref[:,:,0])
    dist_xyb[:,:,2] = 0.25 * (dist[:,:,2] - dist[:,:,0])
    
    # Apply tone mapping
    ref_tm = tone_map(ref_xyb)
    dist_tm = tone_map(dist_xyb)
    
    # Compute PSNR for each channel
    psnr_values = []
    for i in range(3):
        mse = np.mean((ref_tm[:,:,i] - dist_tm[:,:,i]) ** 2)
        if mse == 0:
            psnr_values.append(100.0)
        else:
            psnr_values.append(20 * np.log10(1.0 / np.sqrt(mse)))
    
    # Weighted average of PSNR for each channel
    weights = [0.6, 0.2, 0.2]  # More weight to luminance
    return np.sum(np.array(psnr_values) * np.array(weights))

def compute_hdr_ssim_xyb(ref, dist):
    """Compute HDR-SSIM in XYB color space"""
    # Convert to XYB color space (simplified)
    ref_xyb = np.zeros_like(ref)
    dist_xyb = np.zeros_like(dist)
    
    # X channel (luminance)
    ref_xyb[:,:,0] = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    dist_xyb[:,:,0] = 0.2126 * dist[:,:,0] + 0.7152 * dist[:,:,1] + 0.0722 * dist[:,:,2]
    
    # Y channel (green-red)
    ref_xyb[:,:,1] = 0.5 * (ref[:,:,1] - ref[:,:,0])
    dist_xyb[:,:,1] = 0.5 * (dist[:,:,1] - dist[:,:,0])
    
    # B channel (blue-yellow)
    ref_xyb[:,:,2] = 0.25 * (ref[:,:,2] - ref[:,:,0])
    dist_xyb[:,:,2] = 0.25 * (dist[:,:,2] - dist[:,:,0])
    
    # Apply tone mapping
    ref_tm = tone_map(ref_xyb)
    dist_tm = tone_map(dist_xyb)
    
    # Compute SSIM for each channel
    ssim_values = []
    for i in range(3):
        ssim_values.append(ssim(ref_tm[:,:,i], dist_tm[:,:,i], data_range=1.0))
    
    # Weighted average of SSIM for each channel
    weights = [0.6, 0.2, 0.2]  # More weight to luminance
    return np.sum(np.array(ssim_values) * np.array(weights))

def compute_hdr_psnr_xyb_log(ref, dist):
    """Compute HDR-PSNR in log space"""
    # Convert to XYB color space (simplified)
    ref_xyb = np.zeros_like(ref)
    dist_xyb = np.zeros_like(dist)
    
    # X channel (luminance)
    ref_xyb[:,:,0] = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    dist_xyb[:,:,0] = 0.2126 * dist[:,:,0] + 0.7152 * dist[:,:,1] + 0.0722 * dist[:,:,2]
    
    # Y channel (green-red)
    ref_xyb[:,:,1] = 0.5 * (ref[:,:,1] - ref[:,:,0])
    dist_xyb[:,:,1] = 0.5 * (dist[:,:,1] - dist[:,:,0])
    
    # B channel (blue-yellow)
    ref_xyb[:,:,2] = 0.25 * (ref[:,:,2] - ref[:,:,0])
    dist_xyb[:,:,2] = 0.25 * (dist[:,:,2] - dist[:,:,0])
    
    # Apply log transformation
    ref_log = np.log1p(ref_xyb)
    dist_log = np.log1p(dist_xyb)
    
    # Normalize
    ref_log = ref_log / ref_log.max()
    dist_log = dist_log / dist_log.max()
    
    # Compute PSNR for each channel
    psnr_values = []
    for i in range(3):
        mse = np.mean((ref_log[:,:,i] - dist_log[:,:,i]) ** 2)
        if mse == 0:
            psnr_values.append(100.0)
        else:
            psnr_values.append(20 * np.log10(1.0 / np.sqrt(mse)))
    
    # Weighted average of PSNR for each channel
    weights = [0.6, 0.2, 0.2]  # More weight to luminance
    return np.sum(np.array(psnr_values) * np.array(weights))

def compute_hdr_ssim_xyb_log(ref, dist):
    """Compute HDR-SSIM in log space"""
    # Convert to XYB color space (simplified)
    ref_xyb = np.zeros_like(ref)
    dist_xyb = np.zeros_like(dist)
    
    # X channel (luminance)
    ref_xyb[:,:,0] = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    dist_xyb[:,:,0] = 0.2126 * dist[:,:,0] + 0.7152 * dist[:,:,1] + 0.0722 * dist[:,:,2]
    
    # Y channel (green-red)
    ref_xyb[:,:,1] = 0.5 * (ref[:,:,1] - ref[:,:,0])
    dist_xyb[:,:,1] = 0.5 * (dist[:,:,1] - dist[:,:,0])
    
    # B channel (blue-yellow)
    ref_xyb[:,:,2] = 0.25 * (ref[:,:,2] - ref[:,:,0])
    dist_xyb[:,:,2] = 0.25 * (dist[:,:,2] - dist[:,:,0])
    
    # Apply log transformation
    ref_log = np.log1p(ref_xyb)
    dist_log = np.log1p(dist_xyb)
    
    # Normalize
    ref_log = ref_log / ref_log.max()
    dist_log = dist_log / dist_log.max()
    
    # Compute SSIM for each channel
    ssim_values = []
    for i in range(3):
        ssim_values.append(ssim(ref_log[:,:,i], dist_log[:,:,i], data_range=1.0))
    
    # Weighted average of SSIM for each channel
    weights = [0.6, 0.2, 0.2]  # More weight to luminance
    return np.sum(np.array(ssim_values) * np.array(weights))

def compute_hdr_psnr_xyb_log_tone(ref, dist):
    """Compute HDR-PSNR in log space with tone mapping"""
    # Convert to XYB color space (simplified)
    ref_xyb = np.zeros_like(ref)
    dist_xyb = np.zeros_like(dist)
    
    # X channel (luminance)
    ref_xyb[:,:,0] = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    dist_xyb[:,:,0] = 0.2126 * dist[:,:,0] + 0.7152 * dist[:,:,1] + 0.0722 * dist[:,:,2]
    
    # Y channel (green-red)
    ref_xyb[:,:,1] = 0.5 * (ref[:,:,1] - ref[:,:,0])
    dist_xyb[:,:,1] = 0.5 * (dist[:,:,1] - dist[:,:,0])
    
    # B channel (blue-yellow)
    ref_xyb[:,:,2] = 0.25 * (ref[:,:,2] - ref[:,:,0])
    dist_xyb[:,:,2] = 0.25 * (dist[:,:,2] - dist[:,:,0])
    
    # Apply log transformation
    ref_log = np.log1p(ref_xyb)
    dist_log = np.log1p(dist_xyb)
    
    # Normalize
    ref_log = ref_log / ref_log.max()
    dist_log = dist_log / dist_log.max()
    
    # Apply tone mapping
    ref_tm = tone_map(ref_log)
    dist_tm = tone_map(dist_log)
    
    # Compute PSNR for each channel
    psnr_values = []
    for i in range(3):
        mse = np.mean((ref_tm[:,:,i] - dist_tm[:,:,i]) ** 2)
        if mse == 0:
            psnr_values.append(100.0)
        else:
            psnr_values.append(20 * np.log10(1.0 / np.sqrt(mse)))
    
    # Weighted average of PSNR for each channel
    weights = [0.6, 0.2, 0.2]  # More weight to luminance
    return np.sum(np.array(psnr_values) * np.array(weights))

def compute_hdr_ssim_xyb_log_tone(ref, dist):
    """Compute HDR-SSIM in log space with tone mapping"""
    # Convert to XYB color space (simplified)
    ref_xyb = np.zeros_like(ref)
    dist_xyb = np.zeros_like(dist)
    
    # X channel (luminance)
    ref_xyb[:,:,0] = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    dist_xyb[:,:,0] = 0.2126 * dist[:,:,0] + 0.7152 * dist[:,:,1] + 0.0722 * dist[:,:,2]
    
    # Y channel (green-red)
    ref_xyb[:,:,1] = 0.5 * (ref[:,:,1] - ref[:,:,0])
    dist_xyb[:,:,1] = 0.5 * (dist[:,:,1] - dist[:,:,0])
    
    # B channel (blue-yellow)
    ref_xyb[:,:,2] = 0.25 * (ref[:,:,2] - ref[:,:,0])
    dist_xyb[:,:,2] = 0.25 * (dist[:,:,2] - dist[:,:,0])
    
    # Apply log transformation
    ref_log = np.log1p(ref_xyb)
    dist_log = np.log1p(dist_xyb)
    
    # Normalize
    ref_log = ref_log / ref_log.max()
    dist_log = dist_log / dist_log.max()
    
    # Apply tone mapping
    ref_tm = tone_map(ref_log)
    dist_tm = tone_map(dist_log)
    
    # Compute SSIM for each channel
    ssim_values = []
    for i in range(3):
        ssim_values.append(ssim(ref_tm[:,:,i], dist_tm[:,:,i], data_range=1.0))
    
    # Weighted average of SSIM for each channel
    weights = [0.6, 0.2, 0.2]  # More weight to luminance
    return np.sum(np.array(ssim_values) * np.array(weights))

def compute_hdr_psnr_xyb_log_tone_reinhard(ref, dist):
    """Compute HDR-PSNR in log space with Reinhard tone mapping"""
    # Convert to XYB color space (simplified)
    ref_xyb = np.zeros_like(ref)
    dist_xyb = np.zeros_like(dist)
    
    # X channel (luminance)
    ref_xyb[:,:,0] = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    dist_xyb[:,:,0] = 0.2126 * dist[:,:,0] + 0.7152 * dist[:,:,1] + 0.0722 * dist[:,:,2]
    
    # Y channel (green-red)
    ref_xyb[:,:,1] = 0.5 * (ref[:,:,1] - ref[:,:,0])
    dist_xyb[:,:,1] = 0.5 * (dist[:,:,1] - dist[:,:,0])
    
    # B channel (blue-yellow)
    ref_xyb[:,:,2] = 0.25 * (ref[:,:,2] - ref[:,:,0])
    dist_xyb[:,:,2] = 0.25 * (dist[:,:,2] - dist[:,:,0])
    
    # Apply log transformation
    ref_log = np.log1p(ref_xyb)
    dist_log = np.log1p(dist_xyb)
    
    # Normalize
    ref_log = ref_log / ref_log.max()
    dist_log = dist_log / dist_log.max()
    
    # Apply Reinhard tone mapping
    ref_tm = tone_map(ref_log, method='reinhard')
    dist_tm = tone_map(dist_log, method='reinhard')
    
    # Compute PSNR for each channel
    psnr_values = []
    for i in range(3):
        mse = np.mean((ref_tm[:,:,i] - dist_tm[:,:,i]) ** 2)
        if mse == 0:
            psnr_values.append(100.0)
        else:
            psnr_values.append(20 * np.log10(1.0 / np.sqrt(mse)))
    
    # Weighted average of PSNR for each channel
    weights = [0.6, 0.2, 0.2]  # More weight to luminance
    return np.sum(np.array(psnr_values) * np.array(weights))

def compute_hdr_ssim_xyb_log_tone_reinhard(ref, dist):
    """Compute HDR-SSIM in log space with Reinhard tone mapping"""
    # Convert to XYB color space (simplified)
    ref_xyb = np.zeros_like(ref)
    dist_xyb = np.zeros_like(dist)
    
    # X channel (luminance)
    ref_xyb[:,:,0] = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    dist_xyb[:,:,0] = 0.2126 * dist[:,:,0] + 0.7152 * dist[:,:,1] + 0.0722 * dist[:,:,2]
    
    # Y channel (green-red)
    ref_xyb[:,:,1] = 0.5 * (ref[:,:,1] - ref[:,:,0])
    dist_xyb[:,:,1] = 0.5 * (dist[:,:,1] - dist[:,:,0])
    
    # B channel (blue-yellow)
    ref_xyb[:,:,2] = 0.25 * (ref[:,:,2] - ref[:,:,0])
    dist_xyb[:,:,2] = 0.25 * (dist[:,:,2] - dist[:,:,0])
    
    # Apply log transformation
    ref_log = np.log1p(ref_xyb)
    dist_log = np.log1p(dist_xyb)
    
    # Normalize
    ref_log = ref_log / ref_log.max()
    dist_log = dist_log / dist_log.max()
    
    # Apply Reinhard tone mapping
    ref_tm = tone_map(ref_log, method='reinhard')
    dist_tm = tone_map(dist_log, method='reinhard')
    
    # Compute SSIM for each channel
    ssim_values = []
    for i in range(3):
        ssim_values.append(ssim(ref_tm[:,:,i], dist_tm[:,:,i], data_range=1.0))
    
    # Weighted average of SSIM for each channel
    weights = [0.6, 0.2, 0.2]  # More weight to luminance
    return np.sum(np.array(ssim_values) * np.array(weights))

def compute_scores(ref_path, dist_path):
    """
    Compute image quality scores between reference and distorted HDR images
    
    Args:
        ref_path (str): Path to reference HDR/EXR image
        dist_path (str): Path to distorted HDR/EXR image
        
    Returns:
        tuple: (psnr_score, ssim_score, pu_mssim_score, dist_brisque, hdr_vdp, hdr_psnr, hdr_ssim, hdr_mssim, 
                hdr_psnr_xyb, hdr_ssim_xyb, hdr_psnr_xyb_log, hdr_ssim_xyb_log, hdr_psnr_xyb_log_tone, 
                hdr_ssim_xyb_log_tone, hdr_psnr_xyb_log_tone_reinhard, hdr_ssim_xyb_log_tone_reinhard)
    """
    # Read and normalize images
    ref_img = normalize_image(read_hdr_image(ref_path))
    dist_img = normalize_image(read_hdr_image(dist_path))
    
    # Convert to tensors for PSNR calculation
    ref_tensor = torch.tensor(ref_img).permute(2, 0, 1).unsqueeze(0).float()
    dist_tensor = torch.tensor(dist_img).permute(2, 0, 1).unsqueeze(0).float()
    
    # Calculate traditional metrics
    psnr_score = psnr(ref_tensor, dist_tensor).item()
    ssim_score = calculate_ssim(ref_img, dist_img)
    pu_mssim_score = compute_pu_mssim(ref_img, dist_img)
    
    # Convert to uint8 for BRISQUE as it expects 0-255 values
    dist_uint8 = (dist_img * 255).astype(np.uint8)
    dist_brisque = brisque.score(dist_uint8)
    
    # Calculate HDR-specific metrics
    hdr_vdp = compute_hdr_vdp(ref_img, dist_img)
    hdr_psnr = compute_hdr_psnr(ref_img, dist_img)
    hdr_ssim = compute_hdr_ssim(ref_img, dist_img)
    hdr_mssim = compute_hdr_mssim(ref_img, dist_img)
    hdr_psnr_xyb = compute_hdr_psnr_xyb(ref_img, dist_img)
    hdr_ssim_xyb = compute_hdr_ssim_xyb(ref_img, dist_img)
    hdr_psnr_xyb_log = compute_hdr_psnr_xyb_log(ref_img, dist_img)
    hdr_ssim_xyb_log = compute_hdr_ssim_xyb_log(ref_img, dist_img)
    hdr_psnr_xyb_log_tone = compute_hdr_psnr_xyb_log_tone(ref_img, dist_img)
    hdr_ssim_xyb_log_tone = compute_hdr_ssim_xyb_log_tone(ref_img, dist_img)
    hdr_psnr_xyb_log_tone_reinhard = compute_hdr_psnr_xyb_log_tone_reinhard(ref_img, dist_img)
    hdr_ssim_xyb_log_tone_reinhard = compute_hdr_ssim_xyb_log_tone_reinhard(ref_img, dist_img)
    
    return (psnr_score, ssim_score, pu_mssim_score, dist_brisque, hdr_vdp, hdr_psnr, hdr_ssim, hdr_mssim, 
            hdr_psnr_xyb, hdr_ssim_xyb, hdr_psnr_xyb_log, hdr_ssim_xyb_log, hdr_psnr_xyb_log_tone, 
            hdr_ssim_xyb_log_tone, hdr_psnr_xyb_log_tone_reinhard, hdr_ssim_xyb_log_tone_reinhard)

def find_reference_files(ref_folder):
    """Create mapping from base names to reference files"""
    ref_files = {}
    for f in os.listdir(ref_folder):
        if f.endswith('.exr') or f.endswith('.hdr'):
            # Extract base name without extension
            base = os.path.splitext(f)[0]
            ref_files[base] = f
    return ref_files

def save_results(results, output_csv):
    """Save results to CSV file"""
    if results:
        pd.DataFrame(results).to_csv(output_csv, index=False)
        logger.info(f"Saved {len(results)} results to {output_csv}")
    else:
        with open(output_csv, 'w') as f:
            f.write("Reference,Distorted,PSNR,SSIM,PU-MSSIM,BRISQUE,HDR-VDP,HDR-PSNR,HDR-SSIM,HDR-MSSIM,HDR-PSNR-XYB,HDR-SSIM-XYB,HDR-PSNR-XYB-LOG,HDR-SSIM-XYB-LOG,HDR-PSNR-XYB-LOG-TONE,HDR-SSIM-XYB-LOG-TONE,HDR-PSNR-XYB-LOG-TONE-REINHARD,HDR-SSIM-XYB-LOG-TONE-REINHARD\n")
        logger.warning("No results to save. Empty CSV created.")

def process_folder(ref_folder, dist_folder, output_csv):
    """Process reference and distorted folders to compute metrics"""
    results = []
    
    # Get reference files
    ref_files = find_reference_files(ref_folder)
    
    # Get distorted files
    dist_files = [f for f in os.listdir(dist_folder) if f.endswith('.exr') or f.endswith('.hdr')]
    logger.info(f"Processing {len(dist_files)} distorted images...")
    
    for dist_file in tqdm(dist_files, desc="Processing files"):
        # Extract base number from distorted filename (improved matching)
        # This handles both Number_X.exr and just Number.exr formats
        base_number = os.path.splitext(dist_file)[0].split('_')[0]
        
        # Look for exact matching reference or base name match
        ref_file = None
        
        # First try to find exact match by base name
        exact_match = base_number + os.path.splitext(dist_file)[1]
        if exact_match in ref_files.values():
            ref_file = exact_match
        else:
            # Try to find matching base name in references
            for ref_base, ref_name in ref_files.items():
                if ref_base == base_number or ref_base.startswith(base_number + '_'):
                    ref_file = ref_name
                    break
        
        if not ref_file:
            logger.warning(f"No reference found for {dist_file} (base: {base_number})")
            continue
        
        ref_path = os.path.join(ref_folder, ref_file)
        dist_path = os.path.join(dist_folder, dist_file)
        
        try:
            # Calculate metrics using the compute_scores function
            (psnr_score, ssim_score, pu_mssim_score, dist_brisque, hdr_vdp, hdr_psnr, hdr_ssim, hdr_mssim, 
             hdr_psnr_xyb, hdr_ssim_xyb, hdr_psnr_xyb_log, hdr_ssim_xyb_log, hdr_psnr_xyb_log_tone, 
             hdr_ssim_xyb_log_tone, hdr_psnr_xyb_log_tone_reinhard, hdr_ssim_xyb_log_tone_reinhard) = compute_scores(ref_path, dist_path)
            
            results.append({
                'Reference': ref_file,
                'Distorted': dist_file,
                'PSNR': psnr_score,
                'SSIM': ssim_score,
                'PU-MSSIM': pu_mssim_score,
                'BRISQUE': dist_brisque,
                'HDR-VDP': hdr_vdp,
                'HDR-PSNR': hdr_psnr,
                'HDR-SSIM': hdr_ssim,
                'HDR-MSSIM': hdr_mssim,
                'HDR-PSNR-XYB': hdr_psnr_xyb,
                'HDR-SSIM-XYB': hdr_ssim_xyb,
                'HDR-PSNR-XYB-LOG': hdr_psnr_xyb_log,
                'HDR-SSIM-XYB-LOG': hdr_ssim_xyb_log,
                'HDR-PSNR-XYB-LOG-TONE': hdr_psnr_xyb_log_tone,
                'HDR-SSIM-XYB-LOG-TONE': hdr_ssim_xyb_log_tone,
                'HDR-PSNR-XYB-LOG-TONE-REINHARD': hdr_psnr_xyb_log_tone_reinhard,
                'HDR-SSIM-XYB-LOG-TONE-REINHARD': hdr_ssim_xyb_log_tone_reinhard
            })
            
            # Print progress updates periodically
            if len(results) % 10 == 0:
                logger.info(f"Processed {len(results)} pairs")
                
        except Exception as e:
            logger.error(f"Failed to process {dist_file}: {e}")
            logger.exception(e)  # Log the full stack trace
    
    save_results(results, output_csv)
    logger.info(f"All processing complete. Results saved to {output_csv}")