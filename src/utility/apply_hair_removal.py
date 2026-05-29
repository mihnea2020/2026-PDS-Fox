import sys
from pathlib import Path
import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utility.hair_util import removeHair

def main():
    imgs_dir = Path("data/imgs")
    out_dir = Path("data/imgs_hair_removed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    img_paths = sorted(imgs_dir.glob("*.png"))
    print(f"Found {len(img_paths)} images to process.")
    
    for idx, path in enumerate(img_paths, 1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, _, img_out = removeHair(img_rgb, img_gray, kernel_size=20, threshold_dark=20)
        
        img_out_bgr = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / path.name), img_out_bgr)
        
        if idx % 20 == 0 or idx == len(img_paths):
            print(f"Processed {idx}/{len(img_paths)} images...")

if __name__ == "__main__":
    main()
