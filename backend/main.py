from utils import compute_scores

# Provide your image paths here:
ref_image_path = r'proj/nar/reference/sample_ref.exr'      # Change to your file
dist_image_path = r'proj/nar/distorted/sample_dist.exr'    # Change to your file

# Compute scores
psnr_score, ssim_score, pu_mssim_score, dist_brisque = compute_scores(ref_image_path, dist_image_path)

# Print results
print("Scores for image pair:")
print(f"PSNR: {psnr_score:.4f}")
print(f"SSIM: {ssim_score:.4f}")
print(f"PU-MSSIM: {pu_mssim_score:.4f}")
print(f"BRISQUE: {dist_brisque:.4f}")