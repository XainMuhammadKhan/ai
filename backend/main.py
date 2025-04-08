import os
from utils import compute_scores, find_reference_files

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_folder = os.path.join(base_dir, 'narwaria', 'reference')
    dist_folder = os.path.join(base_dir, 'narwaria', 'distorted')
    
    # Find all reference files
    ref_files = find_reference_files(ref_folder)
    print(f"Found {len(ref_files)} reference files")
    
    # Process each reference file
    for ref_base, ref_file in ref_files.items():
        ref_path = os.path.join(ref_folder, ref_file)
        print(f"\nProcessing reference file: {ref_file}")
        
        # Find corresponding distorted files
        dist_files = [f for f in os.listdir(dist_folder) if f.startswith(ref_base)]
        print(f"Found {len(dist_files)} distorted files for {ref_base}")
        
        for dist_file in dist_files:
            dist_path = os.path.join(dist_folder, dist_file)
            print(f"\nComparing with: {dist_file}")
            
            try:
                # Compute scores
                psnr_score, ssim_score, pu_mssim_score, brisque_score = compute_scores(ref_path, dist_path)
                
                # Print results
                print(f"PSNR: {psnr_score:.2f}")
                print(f"SSIM: {ssim_score:.4f}")
                print(f"PU-MSSIM: {pu_mssim_score:.4f}")
                print(f"BRISQUE: {brisque_score:.2f}")
                
            except Exception as e:
                print(f"Error processing {dist_file}: {str(e)}")

if __name__ == "__main__":
    main()