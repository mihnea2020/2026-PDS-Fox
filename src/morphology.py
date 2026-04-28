import os
import numpy as np
import cv2

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
    # Load image
    img_path = os.path.join(IMGS_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # Skin subtraction in LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Sample the border to learn normal skin color
    border = 20
    top    = lab[:border, :, :]
    bottom = lab[-border:, :, :]
    left   = lab[:, :border, :]
    right  = lab[:, -border:, :]
    skin = np.vstack([top.reshape(-1, 3), bottom.reshape(-1, 3), 
                      left.reshape(-1, 3), right.reshape(-1, 3)])
    skin_mean = np.mean(skin, axis=0)
    skin_std  = np.std(skin, axis=0) + 1e-6

    # Compute how different each pixel is from normal skin
    diff = np.zeros((h, w), dtype=np.float32)
    for c in range(3):
        diff += ((lab[:, :, c].astype(np.float32) - skin_mean[c]) / skin_std[c]) ** 2
    diff = np.sqrt(diff)
    
    # Blur to kill hair noise
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    
    # Normalize to 0-255 for thresholding
    diff_u8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold the difference map
    _, mask = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # If Otsu grabbed >70% of the image, fall back to stricter percentile
    if cv2.countNonZero(mask) / (h * w) > 0.7:
        thresh = np.percentile(diff, 80)
        _, mask = cv2.threshold(diff_u8, thresh, 255, cv2.THRESH_BINARY)

    # Clean up
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Pick the blob closest to center
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    best_label = -1
    best_score = -1
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # Reject specks and whole-image blobs
        if area < 100 or area > (h * w * 0.7):
            continue
        
        # Distance from image center
        bx, by = centroids[i]
        dist = np.sqrt((bx - cx)**2 + (by - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        norm_dist = dist / max_dist
        
        # Penalty if blob touches image border
        border_touch = sum([
            x < 5, y < 5, 
            (x + bw) > (w - 5), 
            (y + bh) > (h - 5)
        ])
        
        #Score: big + centered + not touching border
        score = area * ((1 - norm_dist) ** 4) * (0.4 ** border_touch)
        
        if score > best_score:
            best_score = score
            best_label = i

    if best_label > 0:
        mask = np.uint8(labels == best_label) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        print(f"  {fname}: NOTHING FOUND")

    # Save
    out_name = fname.replace("processed_", "", 1)
    cv2.imwrite(os.path.join(MASKS_DIR, out_name), mask)
    ratio = cv2.countNonZero(mask) / (h * w)
    print(f"  {fname}: mask = {ratio:.1%}")

print(f"\nDone. {len(filenames)} masks saved to {MASKS_DIR}")