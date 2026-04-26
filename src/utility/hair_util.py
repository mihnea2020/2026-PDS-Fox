import cv2
import numpy as np


def removeHair(img_org, img_gray, kernel_size=25, threshold_dark=10, threshold_light=25, radius=3):
    # kernel for the morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    # perform the blackHat filtering on the grayscale image to find the dark hair contours
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair contours in preparation for the inpainting algorithm
    _, thresh_dark = cv2.threshold(blackhat, threshold_dark, 255, cv2.THRESH_BINARY)

    # TopHat: detects light/white hairs on darker skin (higher threshold to avoid false positives)
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    _, thresh_light = cv2.threshold(tophat, threshold_light, 255, cv2.THRESH_BINARY)

    # Combine both masks to handle all hair colours
    thresh = cv2.bitwise_or(thresh_dark, thresh_light)

    # inpaint the original image depending on the mask
    img_out = cv2.inpaint(img_org, thresh, radius, cv2.INPAINT_TELEA)

    return blackhat, thresh, img_out