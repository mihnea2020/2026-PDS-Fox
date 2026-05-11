from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import color, measure, morphology, transform, util
import imageio.v2 as imageio

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
CANCER = {"BCC", "MEL", "SCC"}

MEL_RGB = np.array([
    [197, 188, 217],
    [ 41,  31,  30],
    [118,  21,  17],
    [163,  82,  16],
    [135,  44,   5],
    [113, 108, 139],
], dtype=float) / 255.0


def _read_image(path):
    img = imageio.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img if img.dtype == np.uint8 else util.img_as_ubyte(img)


def _read_mask(path, shape):
    mask = imageio.imread(str(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != tuple(shape[:2]):
        mask = transform.resize(mask, shape[:2], order=0,
                                preserve_range=True, anti_aliasing=False)
    binary = (mask > 127).astype(bool)
    binary = ndimage.binary_fill_holes(binary)
    binary = morphology.remove_small_objects(binary, max_size=30)
    binary = morphology.closing(binary, morphology.disk(2))
    binary = morphology.opening(binary, morphology.disk(1))
    labels = measure.label(binary)
    if labels.max() > 0:
        largest = max(measure.regionprops(labels), key=lambda r: r.area).label
        binary = labels == largest
    return binary.astype(bool)


def _find_file(folder, img_id):
    stem = Path(img_id).stem
    direct = folder / img_id
    if direct.exists():
        return direct
    for ext in IMG_EXT:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    if not folder.exists():
        return None
    low = stem.lower()
    for p in folder.iterdir():
        if p.suffix.lower() in IMG_EXT and p.stem.lower() == low:
            return p
    return None


def _find_mask(masks_dir, img_id):
    stem = Path(img_id).stem
    for name in [f"{stem}.png", f"{stem}_mask.png",
                 f"{stem}_segmentation.png", f"mask_{stem}.png"]:
        p = masks_dir / name
        if p.exists():
            return p
    return _find_file(masks_dir, img_id)


def mask_area(mask):
    return float(mask.sum())


def asymmetry(mask):
    if mask.sum() == 0:
        return np.nan
    def jd(a, b):
        u = np.logical_or(a, b).sum()
        return 0.0 if u == 0 else 1.0 - np.logical_and(a, b).sum() / u
    return float(1 - (1 - jd(mask, np.fliplr(mask))) *
                     (1 - jd(mask, np.flipud(mask))))


def border_compactness(mask):
    area = float(mask.sum())
    if area == 0:
        return np.nan
    border = np.logical_and(mask, ~ndimage.binary_erosion(mask))
    P = float(border.sum())
    return float((P ** 2) / (4 * math.pi * area))


def border_radial(mask):
    if mask.sum() == 0:
        return np.nan
    border = np.logical_and(mask, ~ndimage.binary_erosion(mask))
    b = np.argwhere(border)
    m = np.argwhere(mask)
    if len(b) == 0 or len(m) == 0:
        return np.nan
    dists = np.sqrt(((b - m.mean(axis=0)) ** 2).sum(axis=1))
    return float(0.0 if dists.mean() == 0 else dists.std() / dists.mean())


def lab_colour_std(img, mask):
    if mask.sum() == 0:
        return np.nan
    lab = color.rgb2lab(img.astype(float) / 255.0)
    return float(lab[mask].std(axis=0).mean())


def hsv_features(img, mask):
    keys = ["mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v",
            "hue_entropy", "melanoma_colour_count"]
    pix = img[mask].astype(float) / 255.0
    if len(pix) == 0:
        return {k: np.nan for k in keys}
    hsv = color.rgb2hsv(pix.reshape(1, -1, 3)).reshape(-1, 3)
    hist, _ = np.histogram(hsv[:, 0], bins=30, range=(0, 1))
    prob = hist / max(hist.sum(), 1)
    prob = prob[prob > 0]
    entropy = float(-(prob * np.log2(prob)).sum())
    ref = color.rgb2hsv(MEL_RGB.reshape(1, -1, 3)).reshape(-1, 3)
    count = 0
    for r in ref:
        hd = np.minimum(np.abs(hsv[:, 0] - r[0]), 1 - np.abs(hsv[:, 0] - r[0]))
        close = (hd < 0.06) & (np.abs(hsv[:, 1] - r[1]) < 0.35) & \
                              (np.abs(hsv[:, 2] - r[2]) < 0.35)
        if close.mean() > 0.01:
            count += 1
    vals = [*hsv.mean(axis=0), *hsv.std(axis=0), entropy, float(count)]
    return dict(zip(keys, [float(v) for v in vals]))


def bleeding_likelihood(img, mask):
    if mask.sum() == 0:
        return np.nan
    pix = img[mask].astype(float) / 255.0
    hsv = color.rgb2hsv(pix.reshape(1, -1, 3)).reshape(-1, 3)
    red = (((hsv[:, 0] < 0.08) | (hsv[:, 0] > 0.92)) &
           (hsv[:, 1] > 0.40) &
           (hsv[:, 2] > 0.15) & (hsv[:, 2] < 0.55))
    return float(red.mean())


def extract_features(img, mask):
    feats = {
        "mask_area":           mask_area(mask),
        "asymmetry":           asymmetry(mask),
        "border_compactness":  border_compactness(mask),
        "border_radial":       border_radial(mask),
        "lab_colour_std":      lab_colour_std(img, mask),
        "bleeding_likelihood": bleeding_likelihood(img, mask),
    }
    feats.update(hsv_features(img, mask))
    return feats


def extract_all(images_dir, masks_dir, metadata_path, out_path, max_images=None):
    meta = pd.read_csv(metadata_path)
    meta = meta.drop(columns=[c for c in meta.columns
                               if str(c).startswith("Unnamed")], errors="ignore")
    meta["diagnostic"] = meta["diagnostic"].astype(str).str.upper().str.strip()
    meta["cancer_label"] = meta["diagnostic"].isin(CANCER).astype(int)
    if max_images is not None:
        meta = meta.head(max_images)

    rows, skipped = [], []
    for _, row in meta.iterrows():
        img_id = str(row["img_id"])
        img_path = _find_file(images_dir, img_id)
        mask_path = _find_mask(masks_dir, img_id)
        if img_path is None or mask_path is None:
            skipped.append({"img_id": img_id,
                            "reason": "missing image" if img_path is None else "missing mask"})
            continue
        try:
            img = _read_image(img_path)
            mask = _read_mask(mask_path, img.shape[:2])
            feat = extract_features(img, mask)
        except Exception as exc:
            skipped.append({"img_id": img_id, "reason": str(exc)})
            continue
        record = {"img_id": img_id, "diagnostic": row["diagnostic"],
                  "cancer_label": int(row["cancer_label"])}
        for col in ["patient_id", "age", "gender", "region",
                    "diameter_1", "diameter_2", "fitspatrick"]:
            if col in meta.columns:
                record[col] = row[col]
        record.update(feat)
        rows.append(record)
        if len(rows) % 100 == 0:
            print(f"{len(rows)} extracted ...")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(
            out_path.with_name(out_path.stem + "_skipped.csv"), index=False)
    print(f"Extracted {len(df)} images -> {out_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path, default=Path("data/preprocessed/images"))
    p.add_argument("--masks-dir",  type=Path, default=Path("data/preprocessed/masks"))
    p.add_argument("--metadata",   type=Path, default=Path("data/metadata.csv"))
    p.add_argument("--out",        type=Path, default=Path("results/features.csv"))
    p.add_argument("--max-images", type=int,  default=None)
    a = p.parse_args()
    extract_all(a.images_dir, a.masks_dir, a.metadata, a.out, a.max_images)
