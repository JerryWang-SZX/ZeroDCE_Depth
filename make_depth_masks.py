#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_depth_masks.py
Read "pre-enhancement" low-light images, use MiDaS(DPT-Hybrid) to generate depth and (optional) consistency confidence,
and save the normalized depth mask.

Usage Example:
  python make_depth_masks.py --input datasets/lowlight_test --out midas_depth \
      --short_side 384 --with_conf

Output Contents (using 0001.jpg as an example):
  midas_depth/
    0001_disp.npy, 0001_disp.png
    0001_Dnorm.npy, 0001_Dnorm.png
    0001_conf.npy, 0001_conf.png   # Only when --with_conf is used
"""
import os, glob, argparse
from pathlib import Path
import numpy as np
import cv2 as cv
import torch
from tqdm import tqdm

# ---------- Utilities ----------
def norm01(x, p1=1, p2=99, eps=1e-6):
    lo, hi = np.percentile(x, p1), np.percentile(x, p2)
    return np.clip((x - lo) / (hi - lo + eps), 0, 1).astype(np.float32)

def vis_color(x01):
    v = (np.clip(x01, 0, 1) * 255).astype(np.uint8)
    return cv.applyColorMap(v, cv.COLORMAP_TURBO)

# ---------- MiDaS ----------
@torch.inference_mode()
def load_midas(device):
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(device).eval()
    tf = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return model, tf

@torch.inference_mode()
def predict_disp(model, tf, bgr, short_side, device):
    h, w = bgr.shape[:2]
    s = short_side / min(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    im = cv.resize(bgr, (nw, nh), interpolation=cv.INTER_CUBIC)
    rgb = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    pred = model(tf(rgb).to(device))
    disp = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(nh, nw),
        mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy().astype(np.float32)
    # Back to original resolution
    disp = cv.resize(disp, (w, h), interpolation=cv.INTER_CUBIC).astype(np.float32)
    return disp

def multi_scale_conf(model, tf, bgr, short_side, device,
                     scales=(1.0, 0.75, 0.5), alpha=6.0, beta=2.0):
    disps = []
    h, w = bgr.shape[:2]
    for s in scales:
        if s == 1.0:
            d = predict_disp(model, tf, bgr, short_side, device)
        else:
            bgr_s = cv.resize(bgr, (int(w * s), int(h * s)), interpolation=cv.INTER_CUBIC)
            d = predict_disp(model, tf, bgr_s, short_side, device)
            d = cv.resize(d, (w, h), interpolation=cv.INTER_CUBIC)
        disps.append(d.astype(np.float32))
    disps = np.stack(disps, 0)
    med = np.median(disps, axis=0)
    mad = np.median(np.abs(disps - med[None, ...]), axis=0)
    mad_n = norm01(mad)
    # Gradient consistency variance
    def grad(x):
        x = norm01(x)
        gx = cv.Sobel(x, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(x, cv.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)
    varg = np.var(np.stack([grad(d) for d in disps], 0), axis=0)
    varg_n = norm01(varg)
    conf = np.exp(-alpha * mad_n) * np.exp(-beta * varg_n)
    return med.astype(np.float32), conf.astype(np.float32)

# ---------- Main Flow ----------
def main(args):
    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images (read common extensions just like lowlight_test.py)
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    if args.recursive:
        for e in exts:
            paths += glob.glob(str(in_dir / "**" / e), recursive=True)
    else:
        for e in exts:
            paths += glob.glob(str(in_dir / e))
    paths = sorted(paths)
    assert paths, f"No images found in {in_dir}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tf = load_midas(device)

    for p in tqdm(paths, desc="MiDaS depth"):
        bgr = cv.imread(p)
        stem = Path(p).stem

        # Depth
        disp = predict_disp(model, tf, bgr, args.short_side, device)
        Dnorm = norm01(disp)  # The "depth mask" we want (larger values for near objects)

        # Save depth and visualization
        np.save(out_dir / f"{stem}_disp.npy", disp)
        np.save(out_dir / f"{stem}_Dnorm.npy", Dnorm)
        cv.imwrite(str(out_dir / f"{stem}_disp.png"), vis_color(norm01(disp)))
        cv.imwrite(str(out_dir / f"{stem}_Dnorm.png"), (Dnorm * 255).astype(np.uint8))

        # Optional: Consistency confidence
        if args.with_conf:
            _, conf = multi_scale_conf(model, tf, bgr, args.short_side, device)
            np.save(out_dir / f"{stem}_conf.npy", conf)
            cv.imwrite(str(out_dir / f"{stem}_conf.png"), (conf * 255).astype(np.uint8))

    print(f"Done. Results saved to: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Make depth masks (MiDaS) for lowlight raw images")
    ap.add_argument("--input", help="Pre-enhancement image directory (original image directory used by lowlight_test.py)", default="data/test_data/real")
    ap.add_argument("--out", help="Save directory", default="midas_depth")
    ap.add_argument("--short_side", type=int, default=384, help="MiDaS short side")
    ap.add_argument("--with_conf", action="store_true", help="Compute and save multi-scale consistency confidence")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    args = ap.parse_args()
    main(args)