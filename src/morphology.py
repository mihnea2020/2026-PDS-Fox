import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage import morphology, measure
from skimage.filters import gaussian, threshold_otsu

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMGS_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MASKS_DIR = os.path.join(PROJECT_ROOT, "data", "masks")
os.makedirs(MASKS_DIR, exist_ok=True)

valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
filenames = sorted(f for f in os.listdir(IMGS_DIR) if f.lower().endswith(valid_ext))

if not filenames:
    print("No images found.")
    exit()

print(f"Found {len(filenames)} images.\n")

for fname in filenames:
    img_path = os.path.join(IMGS_DIR, fname)

    #Load image
    img = imread(img_path)

    h, w = img.shape[:2]
    img_area = h * w

    brush_radius = max(2, min(15, int(min(h, w) * 0.003)))

    r_open = min(7, max(3, int(min(h, w) * 0.008)))
    r_close = min(10, max(3, int(min(h, w) * 0.015)))
    
    disk_open = morphology.disk(r_open)
    disk_close = morphology.disk(r_close)

    max_hole_size = min(500, max(50, int(img_area * 0.0005)))

    im256 = rgb2gray(img) * 256
    blurred_im = gaussian(im256, sigma=2)

    thresh_dark = np.percentile(blurred_im, 5)
    mask_dark = blurred_im < thresh_dark

    thresh_bright = np.percentile(blurred_im, 95)
    mask_bright = blurred_im > thresh_bright
    
    hsv = rgb2hsv(img)
    sat = gaussian(hsv[:, :, 1], sigma=2)
    thresh_sat = np.percentile(sat, 97)
    mask_sat = sat > thresh_sat

    lab = rgb2lab(img)
    a = gaussian(lab[:, :, 1], sigma=2)
    thresh_a = np.percentile(a, 95)
    mask_red = a > thresh_a

    # mask = np.maximum(mask_dark, mask_bright)    
    # mask = np.maximum(mask, mask_sat)
    # mask = np.maximum(mask, mask_red)

    mask = np.zeros((h, w), dtype=bool)
    mask = np.logical_or(mask,  mask_dark)
    mask = np.logical_or(mask, mask_bright)
    mask = np.logical_or(mask, mask_sat)
    mask = np.logical_or(mask, mask_red)

    mask = morphology.opening(mask, disk_open)   # remove specks
    mask = morphology.closing(mask, disk_close)  # fill gaps

    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)

    if regions:
        regions = [r for r in regions if r.area > 0.001 * img_area]
        
        if regions:
            cy, cx = h / 2, w / 2
            diag_sq = h**2 + w**2
            
            def score(r):
                dist_sq = (r.centroid[0] - cy)**2 + (r.centroid[1] - cx)**2
                return r.area / img_area - 5.0 * (dist_sq / diag_sq)
            
            best = max(regions, key=score)
            mask = labeled == best.label
        else:
            mask = np.zeros((h, w), dtype=bool)
    else:
        mask = np.zeros((h, w), dtype=bool)
        
    mask = morphology.remove_small_holes(mask, max_size=max_hole_size)

    # --- 8. Save ---
    mask_uint8 = (mask * 255).astype(np.uint8)
    out_name = fname.replace("processed_", "", 1)
    imsave(os.path.join(MASKS_DIR, out_name), mask_uint8, check_contrast=False)

    print(f"  masked {fname}  ({h}×{w}, brush={brush_radius}, hole={max_hole_size})")

print(f"\nDone. {len(filenames)} masks saved to {MASKS_DIR}")