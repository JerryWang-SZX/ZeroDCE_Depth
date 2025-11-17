#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_depth_aware_metrics.py
Compares the original ZeroDCE++ vs our depth-gated version, measuring under "brightness-aligned" conditions:
- Far/Mid/Near noise (high-frequency residual variance)
- Edge/Detail preservation (gradient energy)
- Zonal SNR (gradient energy / noise standard deviation)
- Very-low-frequency vignetting/banding (very-low-freq amplitude)
- Depth-intensity consistency (Spearman/Pearson with g_far or logR)

Usage:
python eval_depth_aware_metrics.py \
  --orig  data/test_data/real \
  --base  result_Zero_DCEpp/real \
  --ours  result_Zero_DCEpp_depth_bright \
  --depth midas_depth \
  --ours_debug result_Zero_DCEpp_depth_bright/_debug \
  --out   eval_report

Notes:
- --base refers to the original ZeroDCE++ output directory
- --ours refers to our method's output directory
- --depth is the output of make_depth_masks.py (containing *_Dnorm.npy / optional *_conf.npy)
- --ours_debug directly reads g_far.png if it exists (corresponding to <stem>_g_far.png for each image)
"""

import os, glob, argparse
from pathlib import Path
import numpy as np
import cv2 as cv
import pandas as pd
from scipy.stats import spearmanr, pearsonr

def imread_any(p):
    im = cv.imread(str(p), cv.IMREAD_COLOR)
    assert im is not None, f"read fail: {p}"
    return cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

def to_gray01(img):
    return cv.cvtColor((img*255).astype(np.uint8), cv.COLOR_RGB2GRAY).astype(np.float32)/255.0

def match_meanY(img, tgt_mean=0.25):
    y = to_gray01(img)
    m = float(y.mean()) + 1e-8
    g = tgt_mean / m
    out = np.clip(img * g, 0, 1)
    return out, g

def hi_freq_residual(gray01, sigma=1.0):
    # High-frequency using bilateral filter: original - light bilateral smoothing
    base = cv.bilateralFilter(gray01.astype(np.float32), d=0, sigmaColor=12, sigmaSpace=7)
    res = gray01 - base
    return res

def grad_mag(gray01):
    gx = cv.Sobel(gray01, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray01, cv.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx*gx + gy*gy)

def very_low_freq(gray01, sigma=80):
    # Very-low-frequency: amplitude after large-sigma Gaussian (smaller means less "vignetting/banding")
    base = cv.GaussianBlur(gray01, (0,0), sigmaX=sigma)
    return base

def load_D(depth_dir, stem):
    p = Path(depth_dir)/f"{stem}_Dnorm.npy"
    if p.exists():
        D = np.load(p).astype(np.float32)
    else:
        # fallback to read png
        p2 = Path(depth_dir)/f"{stem}_Dnorm.png"
        im = cv.imread(str(p2), cv.IMREAD_GRAYSCALE)
        assert im is not None, f"no depth for {stem}"
        D = (im.astype(np.float32)/255.0)
    return np.clip(D,0,1)

def load_gate(debug_dir, stem, shape=None):
    if debug_dir is None: return None
    p = Path(debug_dir)/f"{stem}_g_far.png"
    if not p.exists(): return None
    g = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
    if g is None: return None
    g = g.astype(np.float32)/255.0
    if shape is not None and g.shape != shape:
        g = cv.resize(g, (shape[1], shape[0]), interpolation=cv.INTER_CUBIC)
    return np.clip(g,0,1)

def corr(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    try:
        sp = spearmanr(a, b, nan_policy='omit').correlation
    except Exception:
        sp = np.nan
    try:
        pe = pearsonr(a, b)[0]
    except Exception:
        pe = np.nan
    return float(sp), float(pe)

def main(args):
    orig_dir = Path(args.orig)
    base_dir = Path(args.base)
    ours_dir = Path(args.ours)
    depth_dir= Path(args.depth)
    dbg_dir  = Path(args.ours_debug) if args.ours_debug else None
    out_dir  = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    stems = []
    # Collect filenames based on original images
    paths = []
    for e in ("*.png","*.jpg","*.jpeg","*.bmp"):
        paths += glob.glob(str(orig_dir/e))
    paths = sorted(paths)
    assert paths, f"no images in {orig_dir}"

    for p in paths:
        stem = Path(p).stem
        # Read
        I0 = imread_any(p)
        # Find base / ours
        def find_in(d):
            for ext in (".png",".jpg",".jpeg",".bmp"):
                f = d/f"{stem}{ext}"
                if f.exists(): return f
            return None
        fb = find_in(base_dir); fo = find_in(ours_dir)
        if fb is None or fo is None:
            print(f"[skip] missing outputs for {stem}")
            continue
        B  = imread_any(fb)
        O  = imread_any(fo)
        H,W = I0.shape[:2]
        if B.shape[:2] != (H,W): B = cv.resize(B, (W,H), interpolation=cv.INTER_CUBIC)
        if O.shape[:2] != (H,W): O = cv.resize(O, (W,H), interpolation=cv.INTER_CUBIC)

        # Brightness alignment (avoid "darker" having an advantage): pull both to the same meanY (use the mean of both or a fixed 0.25)
        tgt_mean = 0.28
        Bn,_ = match_meanY(B, tgt_mean)
        On,_ = match_meanY(O, tgt_mean)

        # Depth and masks
        D = load_D(depth_dir, stem)
        if D.shape != (H,W): D = cv.resize(D, (W,H), interpolation=cv.INTER_CUBIC)
        near = (D >= 0.6).astype(np.uint8)
        mid  = ((D >= 0.4) & (D < 0.6)).astype(np.uint8)
        far  = (D < 0.4).astype(np.uint8)

        # Noise estimation (high-frequency residual variance), measured in "flat regions": pixels where |∇| < P40
        def region_noise(gray01, region_mask):
            g = grad_mag(gray01)
            th = np.percentile(g[region_mask>0], 40) if region_mask.sum()>0 else np.percentile(g,40)
            flat = ((g <= th) & (region_mask>0))
            res = hi_freq_residual(gray01)
            if flat.sum() < 10: return float('nan')
            return float(np.var(res[flat]))
        # Detail/Edges: mean gradient within the region
        def region_grad(gray01, region_mask):
            g = grad_mag(gray01)
            if region_mask.sum() < 10: return float('nan')
            return float(g[region_mask>0].mean())
        # Very-low-frequency "vignetting" amplitude
        def region_vlf(gray01, region_mask, sigma=80):
            vlf = very_low_freq(gray01, sigma=sigma)
            if region_mask.sum() < 10: return float('nan')
            return float(vlf[region_mask>0].std())

        # Convert to grayscale
        Bgy = to_gray01(Bn); Ogy = to_gray01(On)

        # Noise/Gradient/Very-low-freq
        n_far_B = region_noise(Bgy, far); n_far_O = region_noise(Ogy, far)
        n_near_B= region_noise(Bgy, near); n_near_O= region_noise(Ogy, near)
        g_near_B= region_grad (Bgy, near); g_near_O= region_grad (Ogy, near)
        vlf_B   = region_vlf  (Bgy, (near+mid+far)>0, sigma=80)
        vlf_O   = region_vlf  (Ogy, (near+mid+far)>0, sigma=80)

        # SNR by depth band
        def SNR(g_mean, n_var):
            return float(g_mean / (np.sqrt(n_var)+1e-8)) if np.isfinite(g_mean) and np.isfinite(n_var) else float('nan')
        snr_far_B = SNR(region_grad(Bgy, far), n_far_B)
        snr_far_O = SNR(region_grad(Ogy, far), n_far_O)
        snr_near_B= SNR(region_grad(Bgy, near), n_near_B)
        snr_near_O= SNR(region_grad(Ogy, near), n_near_O)

        # Depth-intensity consistency (prioritize reading g_far; otherwise use logR)
        gfar = load_gate(dbg_dir, stem, shape=(H,W))
        if gfar is not None:
            sp, pe = corr(1.0-D, gfar)  # Far->Large
            depth_int_sp, depth_int_pe = sp, pe
        else:
            logR_B = np.log(np.clip(Bgy / (to_gray01(I0)+1e-6), 1e-6, 1e6))
            logR_O = np.log(np.clip(Ogy / (to_gray01(I0)+1e-6), 1e-6, 1e6))
            depth_int_sp, depth_int_pe = corr(1.0-D, logR_O)  # Use ours' gain

        rows.append(dict(
            id=stem,
            noise_far_base=n_far_B, noise_far_ours=n_far_O,
            noise_near_base=n_near_B, noise_near_ours=n_near_O,
            grad_near_base=g_near_B, grad_near_ours=g_near_O,
            snr_far_base=snr_far_B, snr_far_ours=snr_far_O,
            snr_near_base=snr_near_B, snr_near_ours=snr_near_O,
            vlf_base=vlf_B, vlf_ours=vlf_O,
            depth_int_spearman=depth_int_sp, depth_int_pearson=depth_int_pe
        ))

    df = pd.DataFrame(rows)
    out_csv = out_dir/"depth_gate_eval.csv"
    df.to_csv(out_csv, index=False)
    # Brief summary
    def mean_delta(a,b):  # ours - base
        return float(np.nanmean(df[b] - df[a]))
    summary = {
        "Δ noise_far (ours-base)": mean_delta("noise_far_base","noise_far_ours"),
        "Δ noise_near (ours-base)": mean_delta("noise_near_base","noise_near_ours"),
        "Δ grad_near (ours-base)" : mean_delta("grad_near_base","grad_near_ours"),
        "Δ SNR_far (ours-base)"   : mean_delta("snr_far_base","snr_far_ours"),
        "Δ SNR_near (ours-base)"  : mean_delta("snr_near_base","snr_near_ours"),
        "Δ very_low_freq std"     : mean_delta("vlf_base","vlf_ours"),
        "mean Spearman(depth,intensity)": float(np.nanmean(df["depth_int_spearman"]))
    }
    with open(out_dir/"summary.txt","w") as f:
        for k,v in summary.items():
            f.write(f"{k}: {v:.4f}\n")
    print("Saved:", out_csv, "and summary.txt")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--ours_debug", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args)