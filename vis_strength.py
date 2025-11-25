#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize "enhancement strength mask" and analyze its correlation with depth.

Usage Example:
  python vis_strength.py \
      --orig datasets/lowlight_test \
      --enh  results/lowlight_test_zero_dce++ \
      --depth midas_depth \
      --out vis_strength \
      --exts .png .jpg .jpeg \
      --use_conf
"""

import os, glob, argparse, math, json
from pathlib import Path
import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt

# ---------- Basic Utilities ----------
def imread_float(path):
    im = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.dtype != np.uint8:
        im = im.astype(np.float32)
        if im.max() > 1.0:
            im = np.clip(im / 255.0, 0, 1)
        return im
    return im.astype(np.float32) / 255.0

def to_luma_srgb(bgr):
    # sRGB/BT.709 Luma
    b,g,r = cv.split(bgr)
    return 0.0722*b + 0.7152*g + 0.2126*r

def norm01(x, p1=1, p2=99, eps=1e-6):
    lo, hi = np.percentile(x, p1), np.percentile(x, p2)
    return np.clip((x - lo) / (hi - lo + eps), 0, 1).astype(np.float32)

def vis_colormap(x01):
    v8 = (np.clip(x01, 0, 1) * 255).astype(np.uint8)
    return cv.applyColorMap(v8, cv.COLORMAP_TURBO)

def find_match(path, candidates):
    stem = Path(path).stem
    for c in candidates:
        s = Path(c).stem
        if s == stem or s.startswith(stem + "_"):
            return c
    return None

def load_depth(depth_dir, stem):
    # Priority: *_Dnorm.npy -> *_Dnorm.png -> *_disp.npy -> *_disp.png
    prefer = [
        f"{stem}_Dnorm.npy", f"{stem}_Dnorm.png",
        f"{stem}_disp.npy",  f"{stem}_disp.png",
    ]
    for name in prefer:
        p = depth_dir / name
        if p.exists():
            if p.suffix == ".npy":
                d = np.load(p).astype(np.float32)
                return np.clip(d, 0, 1) if "Dnorm" in name else norm01(d)
            else:
                im = cv.imread(str(p), cv.IMREAD_UNCHANGED)
                if im is None: continue
                if im.ndim == 3:
                    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                im = im.astype(np.float32)
                if im.max() > 1: im /= 255.0
                return np.clip(im, 0, 1) if "Dnorm" in name else norm01(im)
    return None

def safe_resize(x, shape_hw):
    h,w = shape_hw
    if x.ndim == 2:
        return cv.resize(x, (w,h), interpolation=cv.INTER_CUBIC)
    return cv.resize(x, (w,h), interpolation=cv.INTER_CUBIC)

# ---------- Strength Computation and Smoothing ----------
def compute_strength(orig_bgr, enh_bgr, eps=1e-6):
    Yo = to_luma_srgb(orig_bgr)
    Ye = to_luma_srgb(enh_bgr)
    dY   = Ye - Yo                              # Luma difference
    logR = np.log((Ye + eps) / (Yo + eps))      # Exposure ratio (recommended to check this)
    return Yo, Ye, dY.astype(np.float32), logR.astype(np.float32)

def smooth_mask(mask01, guide, sigma_s=12, sigma_r=0.2, iters=2):
    # Edge-preserving smoothing, reduce noise/halos
    m = mask01.copy().astype(np.float32)
    for _ in range(iters):
        m = cv.bilateralFilter(m, d=0,
                               sigmaColor=25*sigma_r*255.0,
                               sigmaSpace=sigma_s)
    g = guide.astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    m = 0.7*m + 0.3*cv.bilateralFilter(g, d=0,
                                       sigmaColor=25*sigma_r*255.0,
                                       sigmaSpace=sigma_s)
    return m

def overlay_heatmap(base_bgr01, heat01, alpha=0.55):
    cm = vis_colormap(heat01)
    cm = cv.cvtColor(cm, cv.COLOR_BGR2RGB)
    base = (np.clip(base_bgr01, 0, 1)*255).astype(np.uint8)
    if base.ndim == 2:
        base = cv.cvtColor(base, cv.COLOR_GRAY2BGR)
    over = cv.addWeighted(base, 1-alpha, cm, alpha, 0)
    return over[..., ::-1]  # -> BGR

# ---------- Statistics ----------
def correlation(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 10:
        return dict(pearson=np.nan, spearman=np.nan)
    pa = (a - a.mean()) / (a.std() + 1e-12)
    pb = (b - b.mean()) / (b.std() + 1e-12)
    pearson = float(np.clip((pa*pb).mean(), -1, 1))
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    spearman = float(np.clip((ra*rb).mean(), -1, 1))
    return dict(pearson=pearson, spearman=spearman)

def save_hist(data, out_png, title):
    plt.figure()
    plt.hist(data.reshape(-1), bins=80)
    plt.title(title); plt.xlabel("Value"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def save_hexbin(x, y, out_png, title):
    plt.figure()
    plt.hexbin(x.reshape(-1), y.reshape(-1), gridsize=50)
    plt.title(title)
    plt.xlabel("Depth (0=far, 1=near)"); plt.ylabel("Strength (log-ratio)")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

# ---------- Main Flow ----------
def main():
    ap = argparse.ArgumentParser(description="Visualize enhancement strength vs depth")
    ap.add_argument("--orig", help="Original image directory", default="data/test_data/real")
    ap.add_argument("--enh", help="Enhanced result directory", default="data/result_Zero_DCE++/real")
    ap.add_argument("--depth", help="Depth/confidence directory (*_Dnorm/disp/conf)", default="midas_depth")
    ap.add_argument("--out", help="Output directory", default="vis_strength")
    ap.add_argument("--exts", nargs="+", default=[".png",".jpg",".jpeg",".bmp"], help="Matching extensions")
    ap.add_argument("--use_conf", action="store_true", help="If *_conf.npy exists, perform weighted correlation")
    args = ap.parse_args()

    orig_dir = Path(args.orig)
    enh_dir  = Path(args.enh)
    depth_dir= Path(args.depth)
    out_dir  = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Collect original images
    paths = []
    for e in args.exts:
        paths += glob.glob(str(orig_dir / f"*{e}"))
    paths = sorted(paths)

    dataset_stats = []
    for p in paths:
        stem = Path(p).stem
        # Find enhanced images
        enh_cands = []
        for e in args.exts:
            enh_cands += glob.glob(str(enh_dir / f"{stem}{e}"))
        enh_p = find_match(p, enh_cands)
        if enh_p is None:
            print(f"[WARN] enhanced file not found for {stem}, skip")
            continue

        # Read images and align
        ori = imread_float(p)
        enh = imread_float(enh_p)
        if ori.shape != enh.shape:
            enh = safe_resize(enh, ori.shape[:2])

        # Compute strength
        Yo, Ye, dY, logR = compute_strength(ori, enh)
        dY_v   = norm01(dY, 1, 99)
        logR_v = norm01(logR, 1, 99)

        # Read depth/confidence
        D = load_depth(depth_dir, stem)
        if D is None:
            print(f"[WARN] depth not found for {stem}, visualizing without depth")
            D = np.zeros_like(Yo)
        conf = None
        if args.use_conf:
            cp = depth_dir / f"{stem}_conf.npy"
            if cp.exists():
                conf = np.load(cp).astype(np.float32)
                if conf.shape != D.shape:
                    conf = safe_resize(conf, D.shape)
            else:
                print(f"[INFO] no conf for {stem}")

        # Smooth strength (for visualization only)
        dY_s   = smooth_mask(dY_v, Yo)
        logR_s = smooth_mask(logR_v, Yo)

        # Export visualization
        im_out_dir = out_dir / stem
        im_out_dir.mkdir(exist=True, parents=True)

        cv.imwrite(str(im_out_dir / "orig.png"), (np.clip(ori,0,1)*255).astype(np.uint8))
        cv.imwrite(str(im_out_dir / "enh.png"),  (np.clip(enh,0,1)*255).astype(np.uint8))
        cv.imwrite(str(im_out_dir / "depth.png"), vis_colormap(norm01(D)))
        cv.imwrite(str(im_out_dir / "dY_heat_raw.png"),    vis_colormap(dY_v))
        cv.imwrite(str(im_out_dir / "dY_heat_smooth.png"), vis_colormap(dY_s))
        cv.imwrite(str(im_out_dir / "logR_heat_raw.png"),    vis_colormap(logR_v))
        cv.imwrite(str(im_out_dir / "logR_heat_smooth.png"), vis_colormap(logR_s))
        cv.imwrite(str(im_out_dir / "overlay_dY.png"),   overlay_heatmap(ori, dY_v))
        cv.imwrite(str(im_out_dir / "overlay_logR.png"), overlay_heatmap(ori, logR_v))

        # Correlation statistics (logR is closer to exposure ratio)
        Dn = np.clip(D.astype(np.float32), 0, 1)
        x = Dn.reshape(-1); y = logR.reshape(-1)

        if conf is not None:
            w = conf.reshape(-1).astype(np.float32)
            w = w / (w.mean() + 1e-6)
            xm = np.average(x, weights=w)
            ym = np.average(y, weights=w)
            xs = x - xm; ys = y - ym
            pearson_w = float(np.sum(w*xs*ys) /
                              (np.sqrt(np.sum(w*xs*xs)) * np.sqrt(np.sum(w*ys*ys)) + 1e-12))
            stats = dict(pearson=pearson_w, **correlation(x, y))
        else:
            stats = correlation(x, y)

        save_hist(logR, im_out_dir / "hist_logR.png", f"{stem} log-ratio")
        save_hexbin(Dn, logR, im_out_dir / "depth_vs_logR_hexbin.png",
                    f"{stem} depth vs strength")

        # Depth binning means
        bins = np.linspace(0,1,11)
        idx = np.digitize(Dn.reshape(-1), bins)-1
        means = [float(np.mean(y[idx == b])) if np.any(idx == b) else np.nan for b in range(10)]
        np.savetxt(im_out_dir / "depth_bin_means.csv", np.array(means), delimiter=",")

        meta = {
            "stem": stem,
            "pearson_depth_vs_logR": stats["pearson"],
            "spearman_depth_vs_logR": stats["spearman"],
            "notes": "logR = log((Y_enh+eps)/(Y_orig+eps)); depth=Dnorm(near=1)."
        }
        with open(im_out_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        dataset_stats.append(meta)

        # Dataset summary
    if dataset_stats:
        ps = np.array([m["pearson_depth_vs_logR"] for m in dataset_stats if np.isfinite(m["pearson_depth_vs_logR"])])
        ss = np.array([m["spearman_depth_vs_logR"] for m in dataset_stats if np.isfinite(m["spearman_depth_vs_logR"])])
        summ = {
            "num_images": len(dataset_stats),
            "mean_pearson": float(np.mean(ps)) if ps.size else np.nan,
            "mean_spearman": float(np.mean(ss)) if ss.size else np.nan,
        }
        with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(summ, f, ensure_ascii=False, indent=2)
        print("Summary:", summ)
    else:
        print("No images processed. Check your paths & suffixes.")

if __name__ == "__main__":
    main()