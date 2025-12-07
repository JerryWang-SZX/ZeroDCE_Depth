#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-DCE++ training with depth-aware losses (gate + noise + vlf), green-block safe.

Usage (Example):
python lowlight_train_depth_clean.py \
  --lowlight_images_path data/train_data \
  --depth_dir train_midas_depth --use_conf \
  --num_epochs 30 --train_batch_size 8 --lr 1e-4 \
  --scale_factor 12 --snapshots_folder snapshots_Depth_DCE_clean \
  --lambda_gate 0.3 --lambda_noise 1.0 --lambda_vlf 0.03 \
  --gamma_i 2.5 --gamma_dn 1.5 --exp_target 0.6 \
  --load_pretrain --pretrain_dir snapshots_Zero_DCE++/Epoch99.pth
"""

import os, argparse, time, json, math, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision

import dataloader_depth_dce as dataloader          # Your dataloader.py (returns CHW, [0,1] tensor)
import model               # Your model.py
import Myloss              # Your Myloss.py

# --------- utils ---------
def crop_to_sf(t, sf):
    if sf <= 1: return t
    h, w = t.shape[-2:]
    return t[..., : (h//sf)*sf, : (w//sf)*sf]

def percentile_norm_per_batch(x, p1=5, p2=95, eps=1e-6):
    B = x.size(0)
    xs = x.view(B, -1)
    lo = torch.quantile(xs, q=p1/100.0, dim=1, keepdim=True).view(B,1,1,1)
    hi = torch.quantile(xs, q=p2/100.0, dim=1, keepdim=True).view(B,1,1,1)
    return ((x - lo) / (hi - lo + eps)).clamp(0, 1)

def rgb_to_Y(x):
    return (0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3])

def grad_mag(x):
    gx = x[:,:,:,1:] - x[:,:,:,:-1]
    gy = x[:,:,1:,:] - x[:,:,:-1,:]
    gx = F.pad(gx, (0,1,0,0)); gy = F.pad(gy, (0,0,0,1))
    return torch.sqrt(gx*gx + gy*gy + 1e-6)

def _gauss1d_kernel(sigma, device, dtype):
    radius = max(1, int(round(3.0*float(sigma))))
    k = 2*radius + 1
    coords = torch.arange(k, device=device, dtype=dtype) - radius
    kernel = torch.exp(-(coords**2) / (2*sigma*sigma)); kernel /= (kernel.sum() + 1e-8)
    return kernel, radius

def _gauss_sep(x, sigma):
    if sigma <= 0: return x
    B,C,H,W = x.shape
    k1d, r = _gauss1d_kernel(sigma, x.device, x.dtype)
    x = F.pad(x, (r,r,r,r), mode='reflect')
    kx = k1d.view(1,1,1,-1).repeat(C,1,1,1)
    ky = k1d.view(1,1,-1,1).repeat(C,1,1,1)
    x = F.conv2d(x, kx, groups=C)
    x = F.conv2d(x, ky, groups=C)
    return x

def gauss_blur(x, sigma):
    if sigma <= 20: return _gauss_sep(x, sigma)
    s = max(2, int(sigma // 8))
    H,W = x.shape[-2:]
    xs = F.interpolate(x, scale_factor=1.0/s, mode='bilinear', align_corners=False)
    ys = _gauss_sep(xs, sigma/s)
    return F.interpolate(ys, size=(H,W), mode='bilinear', align_corners=False)

# --------- depth loader (match by stem) ---------
def _load_np(path):
    return torch.from_numpy(np.load(path).astype(np.float32))

def load_depth_batch(stems, depth_dir, device, use_conf):
    Ds, Cs = [], []
    for s in stems:
        D = None; C = None
        cand = [f"{s}_Dnorm.npy", f"{s}_Dnorm.png", f"{s}_disp.npy", f"{s}_disp.png"]
        for name in cand:
            p = Path(depth_dir)/name
            if p.exists():
                if p.suffix == ".npy":
                    arr = np.load(p).astype(np.float32)
                else:
                    import cv2 as cv
                    im = cv.imread(str(p), cv.IMREAD_UNCHANGED)
                    if im is None: continue
                    if im.ndim==3: im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                    arr = im.astype(np.float32)
                    if arr.max()>1: arr/=255.0
                D = torch.from_numpy(arr).float()
                if "Dnorm" not in name:    # disp -> 0-1
                    lo,hi = np.percentile(arr, 5), np.percentile(arr, 95)
                    D = ((D - lo)/(hi-lo+1e-6)).clamp(0,1)
                break
        if D is None:
            Ds.append(None); Cs.append(None); continue
        if use_conf:
            pconf = Path(depth_dir)/f"{s}_conf.npy"
            if pconf.exists(): C = _load_np(pconf).float()
        Ds.append(D); Cs.append(C)
    # stack tensors (missing ones are None -> later skip)
    return Ds, Cs

# --------- train ---------
def train(cfg):
    # device
    if cfg.device.lower()=="cpu":
        device=torch.device("cpu"); os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.device)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark=True

    sf = int(cfg.scale_factor)
    print(f"[INFO] scale_factor={sf} (train)")

    net = model.enhance_net_nopool(sf).to(device)
    if cfg.load_pretrain:
        print(f"[INFO] load pretrain: {cfg.pretrain_dir}")
        state = torch.load(cfg.pretrain_dir, map_location=device)
        if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
        net.load_state_dict(state, strict=True)

    train_ds = dataloader.lowlight_loader(cfg.lowlight_images_path)   # Returns CHW tensor in [0,1]
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.train_batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )

    L_color = Myloss.L_color()
    L_spa   = Myloss.L_spa()
    L_exp   = Myloss.L_exp(16)
    L_TV    = Myloss.L_TV()

    opt = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    net.train(); os.makedirs(cfg.snapshots_folder, exist_ok=True)

    it_global=0
    for epoch in range(cfg.num_epochs):
        t0 = time.time(); loss_sum=0.0
        for imgs_or_batch in train_loader:
            # --- Compatible with Tensor or dict ---
            if isinstance(imgs_or_batch, dict):
                imgs = imgs_or_batch.get('image')
                assert imgs is not None, "batch dict must contain key 'image'"
                # Retrieve filenames (or paths) of the current batch from dataloader for depth alignment
                stems = imgs_or_batch.get('stems') or imgs_or_batch.get('stem') \
                        or imgs_or_batch.get('paths') or imgs_or_batch.get('path')
                if stems is None:  # Fallback
                    stems = [None] * imgs.size(0)
                # Normalize to stem list
                from pathlib import Path
                norm_stems = []
                if isinstance(stems, (list, tuple)):
                    for s in stems:
                        if s is None:
                            norm_stems.append(None)
                        else:
                            s = str(s)
                            norm_stems.append(Path(s).stem)
                else:
                    # Single string
                    norm_stems = [Path(str(stems)).stem] * imgs.size(0)
                batch_stems = norm_stems
            else:
                # Old version: directly provided CHW Tensor
                imgs = imgs_or_batch
                batch_stems = [None] * imgs.size(0)

            imgs = imgs.to(device).float()
            imgs = crop_to_sf(imgs, sf)
            B, _, H, W = imgs.shape

            # Try to parse stem from original dataloader path (compatibility: skip depth loss if missing)
            stems = getattr(train_ds, "data_list", None)
            batch_stems = None
            if stems is not None:
                # It is hard to get indices directly after DataLoader shuffle;
                # simple way: add an attribute in dataloader.lowlight_loader
                # `last_batch_paths` returns absolute path list of current batch. Fallback here: use sequential names if missing.
                batch_stems = getattr(train_ds, "last_batch_stems", None)
            if batch_stems is None:
                # Temporarily fake stem, load_depth_batch will return all None later (skip depth loss)
                batch_stems = [None]*B

            Ds, Cs = ( [None]*B, [None]*B )
            if cfg.depth_dir is not None and batch_stems[0] is not None:
                Ds, Cs = load_depth_batch(batch_stems, cfg.depth_dir, device, cfg.use_conf)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                enh, A = net(imgs)
                # ------ baseline losses ------
                loss_tv  = 1600.0 * L_TV(A)
                loss_spa = torch.mean(L_spa(enh, imgs))
                loss_col = 5.0 * torch.mean(L_color(enh))
                loss_exp = 10.0 * torch.mean(L_exp(enh, cfg.exp_target))
                L_base   = loss_tv + loss_spa + loss_col + loss_exp

                # ------ depth-aware losses ------
                L_gate  = torch.zeros((), device=device)
                L_noise = torch.zeros((), device=device)
                L_vlf   = torch.zeros((), device=device)

                Y0   = rgb_to_Y(imgs).clamp(1e-6, 1.0)
                Yenh = rgb_to_Y(enh).clamp(1e-6, 1.0)

                # (A) gate: Make gain shape ~ (1-D)^gamma_i
                # Here use log gain -> percentile normalization -> sigmoid to 0-1
                logR  = (Yenh / Y0).log()
                logRn = percentile_norm_per_batch(logR, 5, 95)
                logRs = torch.sigmoid(4.0*(logRn-0.5)) * 0.5 + 0.5

                # (B) noise: Charbonnier high frequency, larger weight for far + flat regions
                G0  = grad_mag(Y0)
                G0n = percentile_norm_per_batch(G0, 5, 95)
                flat = torch.sigmoid(8.0*(0.5 - G0n))
                Y_lp = gauss_blur(Yenh, sigma=1.5)
                Y_hf = Yenh - Y_lp
                noise_charb = torch.sqrt(Y_hf*Y_hf + 1e-3*1e-3)
                noise_norm  = percentile_norm_per_batch(noise_charb, 5, 95)

                # (C) vlf: very low freq std suppresses vignetting
                Y_vlf = gauss_blur(Yenh, sigma=80.0)
                L_vlf = Y_vlf.std()

                # Apply D, C sample by sample
                for b in range(B):
                    if Ds[b] is None: continue
                    D = torch.from_numpy(np.asarray(Ds[b])).to(device).float().view(1,1,*Ds[b].shape)
                    if D.max() > 1: D = D / 255.0
                    D = F.interpolate(D, size=(H,W), mode='bilinear', align_corners=False)
                    D = percentile_norm_per_batch(D, 5, 95)

                    w_far = (1.0 - D).pow(cfg.gamma_dn)
                    if Cs[b] is not None and cfg.use_conf:
                        C = torch.from_numpy(np.asarray(Cs[b])).to(device).float().view(1,1,*Cs[b].shape)
                        if C.max()>1: C = C/255.0
                        C = F.interpolate(C, size=(H,W), mode='bilinear', align_corners=False)
                        C = C.clamp(0,1)
                        w_far = w_far * (0.3 + 0.7*C)

                    g_t = (1.0 - D).pow(cfg.gamma_i)    # Target "bright far, stable near" gating
                    L_gate  = L_gate  + F.smooth_l1_loss(logRs[b:b+1], g_t)

                    L_noise = L_noise + (noise_norm[b:b+1] * flat[b:b+1] * w_far).mean()

                L_gate  = L_gate / max(1, sum([1 for d in Ds if d is not None]))
                L_noise = L_noise/ max(1, sum([1 for d in Ds if d is not None]))

                loss = (L_base
                        + cfg.lambda_gate  * L_gate
                        + cfg.lambda_noise * L_noise
                        + cfg.lambda_vlf   * L_vlf)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
            scaler.step(opt); scaler.update()

            loss_sum += float(loss.detach().cpu())
            if it_global % cfg.display_iter == 0:
                y = enh.detach(); y_min=float(y.min().cpu()); y_max=float(y.max().cpu())
                print(f"[E{epoch} I{it_global:04d}] loss={float(loss):.4f} "
                      f"| base={float(L_base):.4f}, gate={float(L_gate):.4f}, "
                      f"noise={float(L_noise):.4f}, vlf={float(L_vlf):.4f} "
                      f"| y[{y_min:.3f},{y_max:.3f}]")

        ckpt = Path(cfg.snapshots_folder)/f"Epoch{epoch}.pth"
        torch.save(net.state_dict(), str(ckpt))
        with open(str(ckpt)+".meta.json","w",encoding="utf-8") as f:
            json.dump({"scale_factor": sf, "epoch": epoch}, f)
        print(f"[E{epoch}] avg_loss={loss_sum/max(1,len(train_loader)):.4f} "
              f"time={time.time()-t0:.1f}s saved: {ckpt}")

    print("[DONE] Training complete]")

# --------- args ---------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lowlight_images_path', type=str, default="data/train_data")
    ap.add_argument('--depth_dir', type=str, default='train_midas_depth')
    ap.add_argument('--use_conf', action='store_true')

    ap.add_argument('--num_epochs', type=int, default=30)
    ap.add_argument('--train_batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=4)

    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--grad_clip_norm', type=float, default=0.1)
    ap.add_argument('--exp_target', type=float, default=1.2)

    ap.add_argument('--scale_factor', type=int, default=12)
    ap.add_argument('--snapshots_folder', type=str, default="snapshots_Depth_DCE_final")

    ap.add_argument('--load_pretrain', action='store_true')
    ap.add_argument('--pretrain_dir', type=str, default="snapshots_Zero_DCE++/Epoch99.pth")

    ap.add_argument('--device', type=str, default="0")
    ap.add_argument('--display_iter', type=int, default=10)
    ap.add_argument('--use_amp', action='store_true')
    ap.add_argument('--gamma_i', type=float, default=2.5)
    ap.add_argument('--gamma_dn', type=float, default=1.5)

    ap.add_argument('--lambda_gate', type=float, default=0.3)
    ap.add_argument('--lambda_noise', type=float, default=1.0)
    ap.add_argument('--lambda_vlf', type=float, default=0.03)
    return ap

if __name__ == "__main__":
    cfg = build_argparser().parse_args()
    train(cfg)