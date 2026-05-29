from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import color, exposure, measure, morphology, transform, util
import imageio.v2 as imageio

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
TILE_SIZE = 256


def read_image(path):
    img = imageio.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img if img.dtype == np.uint8 else util.img_as_ubyte(img)


def find_mask(masks_dir, stem):
    for name in [f"{stem}.png", f"{stem}_mask.png",
                 f"{stem}_segmentation.png", f"mask_{stem}.png"]:
        p = masks_dir / name
        if p.exists():
            return p
    low = stem.lower()
    for p in masks_dir.iterdir():
        if p.suffix.lower() in IMG_EXT and p.stem.lower() == low:
            return p
    return None


def clean_mask(raw):
    if raw.ndim == 3:
        raw = raw[..., 0]
    binary = (raw > 127).astype(bool) if raw.max() > 1 else (raw > 0).astype(bool)
    binary = ndimage.binary_fill_holes(binary)
    binary = morphology.remove_small_objects(binary, max_size=30)
    binary = morphology.closing(binary, morphology.disk(2))
    binary = morphology.opening(binary, morphology.disk(1))
    labels = measure.label(binary)
    if labels.max() > 0:
        largest = max(measure.regionprops(labels), key=lambda r: r.area).label
        binary = labels == largest
    return binary.astype(bool)


def square_crop(arr, size, interp_order):
    h, w = arr.shape[:2]
    scale = size / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    arr = transform.resize(arr, (nh, nw), order=interp_order,
                           preserve_range=True, anti_aliasing=(interp_order > 0))
    top = (nh - size) // 2
    left = (nw - size) // 2
    return arr[top:top + size, left:left + size]


def normalise_brightness(img):
    hsv = color.rgb2hsv(img.astype(float) / 255.0)
    hsv[..., 2] = exposure.equalize_adapthist(hsv[..., 2], clip_limit=0.03)
    return util.img_as_ubyte(color.hsv2rgb(hsv))


def preprocess_pair(img_path, mask_path, size=TILE_SIZE):
    img = read_image(img_path)
    raw = imageio.imread(str(mask_path))
    if raw.ndim == 3:
        raw = raw[..., 0]
    if raw.shape[:2] != img.shape[:2]:
        raw = transform.resize(raw, img.shape[:2], order=0,
                               preserve_range=True, anti_aliasing=False)
    img_s = square_crop(img, size, interp_order=1).astype(np.uint8)
    mask_s = square_crop(raw, size, interp_order=0).astype(np.uint8)
    mask_c = clean_mask(mask_s)
    img_n = normalise_brightness(img_s)
    return img_n, mask_c


def preprocess_dataset(images_dir, masks_dir, out_images, out_masks,
                       size=TILE_SIZE, verbose=True):
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in images_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXT)
    done, skip = 0, 0
    for img_path in files:
        mask_path = find_mask(masks_dir, img_path.stem)
        if mask_path is None:
            skip += 1
            continue
        try:
            img_out, mask_out = preprocess_pair(img_path, mask_path, size)
        except Exception as exc:
            if verbose:
                print(f"ERROR {img_path.name}: {exc}")
            skip += 1
            continue
        imageio.imwrite(str(out_images / img_path.name), img_out)
        imageio.imwrite(str(out_masks / (img_path.stem + "_mask.png")),
                        (mask_out.astype(np.uint8) * 255))
        done += 1
        if verbose and done % 100 == 0:
            print(f"{done}/{len(files)} done ...")
    if verbose:
        print(f"Done: {done} preprocessed, {skip} skipped.")
    return {"processed": done, "skipped": skip}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir",  type=Path, default=Path("data/images"))
    p.add_argument("--masks-dir",   type=Path, default=Path("data/masks"))
    p.add_argument("--out-images",  type=Path, default=Path("data/preprocessed/images"))
    p.add_argument("--out-masks",   type=Path, default=Path("data/preprocessed/masks"))
    p.add_argument("--target-size", type=int,  default=TILE_SIZE)
    a = p.parse_args()
    preprocess_dataset(a.images_dir, a.masks_dir,
                       a.out_images, a.out_masks, a.target_size)
