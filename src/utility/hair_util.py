import cv2
import numpy as np


def removeHair(img_org, img_gray, kernel_size=20, threshold_dark=20, 
               threshold_light=20, radius=2, work_size=1024):
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