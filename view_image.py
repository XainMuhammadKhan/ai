import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import matplotlib.pyplot as plt

folder_path = r"C:\Users\Xain M-k\Desktop\my AI\Narwaria\reference"
exr_files = [f for f in os.listdir(folder_path) if f.endswith(".exr")]

for file_name in exr_files:
    file_path = os.path.join(folder_path, file_name)
    print("Reading:", file_path)
    img = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    if img is None:
        print(f"❌ Failed to read {file_name}")
        continue

    # Clean HDR image
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # OpenCV Reinhard tone mapping
    tonemap = cv2.createTonemapReinhard( gamma=1.2,        # Darker contrast
    intensity=0,   # Reduce brightness
    light_adapt=0.8,  # Less light adaptation
    color_adapt=0.5 )
    img_ldr = tonemap.process(img.astype(np.float32))  # Tone map the HDR image

    # Convert to 8-bit image
    img_ldr_8bit = np.clip(img_ldr * 255, 0, 255).astype(np.uint8)

    # Save result
    output_path = os.path.join(folder_path, f"{file_name}_tonemapped.png")
    cv2.imwrite(output_path, img_ldr_8bit)

    # Display
    plt.imshow(cv2.cvtColor(img_ldr_8bit, cv2.COLOR_BGR2RGB))
    plt.title(file_name)
    plt.axis('off')
    plt.show()
