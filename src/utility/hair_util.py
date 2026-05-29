import numpy as np
from scipy import ndimage
from skimage import color, exposure, filters, morphology, util

try:
    import cv2
except ImportError:
    cv2 = None


def hair_mask(img, lesion=None):
    gray = exposure.equalize_adapthist(color.rgb2gray(img.astype(float) / 255.0), clip_limit=0.03)
    
    if cv2 is None:
        edges = filters.sobel(gray)
        hair = edges > np.percentile(edges, 97)
    else:
        u8 = util.img_as_ubyte(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        black = cv2.morphologyEx(u8, cv2.MORPH_BLACKHAT, kernel)
        white = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, kernel)
        
        sx = cv2.Sobel(u8, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(u8, cv2.CV_64F, 0, 1, ksize=3)
        sob = (255 * np.sqrt(sx**2 + sy**2) / max(np.sqrt(sx**2 + sy**2).max(), 1)).astype(np.uint8)
        
        lap = np.abs(cv2.Laplacian(u8, cv2.CV_64F))
        lap = (255 * lap / max(lap.max(), 1)).astype(np.uint8)
        
        combined = np.maximum(np.maximum(black, white), ((sob.astype(float) + lap.astype(float)) / 2).astype(np.uint8))
        hair = cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -2) > 0
        
    hair = morphology.closing(hair, morphology.disk(1))
    hair = morphology.remove_small_objects(hair, max_size=12)
    
    if lesion is not None:
        hair = np.logical_and(hair, morphology.dilation(lesion.astype(bool), morphology.disk(5)))
        
    return hair.astype(bool)


def hair_coverage(img, mask):
    area = mask.sum()
    if area == 0:
        return np.nan
    return float(np.logical_and(hair_mask(img, mask), mask).sum() / area)


def remove_hair(img, mask, cov):
    if cv2 is None or (not np.isnan(cov) and cov < 0.005):
        return img.copy()
        
    k = 15 if np.isnan(cov) or cov < 0.035 else 25
    gray = util.img_as_ubyte(exposure.equalize_adapthist(color.rgb2gray(img.astype(float) / 255.0), clip_limit=0.03))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    combined = np.maximum(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel), cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel))
    
    _, m = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((m > 0) & morphology.dilation(mask.astype(bool), morphology.disk(8))).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    return cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)


def removeHair(img_org, img_gray, kernel_size=20, threshold_dark=20,
               threshold_light=20, radius=2, work_size=1024):
    if cv2 is None:
        return None, None, img_org.copy()
        
    h, w = img_org.shape[:2]
    max_dim = max(h, w)
    scale = work_size / max_dim

    # Resize to working resolution for detection
    new_w, new_h = int(w * scale), int(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img_work = cv2.resize(img_org, (new_w, new_h), interpolation=interp)
    gray_work = cv2.resize(img_gray, (new_w, new_h), interpolation=interp)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    blackhat = cv2.morphologyEx(gray_work, cv2.MORPH_BLACKHAT, kernel)
    _, thresh_dark = cv2.threshold(blackhat, threshold_dark, 255, cv2.THRESH_BINARY)

    tophat = cv2.morphologyEx(gray_work, cv2.MORPH_TOPHAT, kernel)
    _, thresh_light = cv2.threshold(tophat, threshold_light, 255, cv2.THRESH_BINARY)

    thresh_work = cv2.bitwise_or(thresh_dark, thresh_light)

    # Resize mask back to original
    if scale != 1.0:
        thresh = cv2.resize(thresh_work, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        thresh = thresh_work

    img_out = cv2.inpaint(img_org, thresh, radius, cv2.INPAINT_TELEA)
    return blackhat, thresh, img_out