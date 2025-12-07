# Depth-Gated Low-Light Enhancement (README)

> Drop-in **post-processing adapter** that makes any low-light enhancement **depth-aware**: far regions get stronger denoising & more enhancement; near subjects keep textures. Zero training required. Works out-of-the-box on top of **Zero-DCE++**; easily portable to derain/dehaze.

---

## 1) Quick Start

### A) Prepare environment
- Python ≥ 3.8, PyTorch per your CUDA
- `opencv-python`, `numpy`, `torchvision`, `scipy`, `pandas` #TODO: list the rest of the dependencies
- MiDaS (for offline depth)

### B) Precompute depth (one-time)
```bash
python make_depth_masks.py \
  --dataset_dir bdd100k-night-v3.yolov11 \
  --split test \
  --out midas_depth \
  --short_side 512 --with_conf
```

### C) Produce the vanilla Zero-DCE++ baseline (unchanged)
PyTorch must be compiled with CUDA enabled to run this scripts.

```bash
python lowlight_test.py
```

Keep results under `result_Zero_DCE++/...`

### D) Run our **depth-aware** adapter on top of Zero-DCE++

**Aggressive preset (max far-field suppression):**
```bash
python lowlight_depth.py \
  --input bdd100k-night-v3.yolov11/test/images \
  --output result_Zero_DCE++_depth \
  --weights snapshots_Zero_DCE++/Epoch99.pth \
  --depth_dir midas_depth --use_conf \
  --denoise_far --far_gamma 3.5 \
  --strong 30 30 7 21 --light 0 0 7 21 --blend_back 0.10 \
  --intensity_gate --gamma_i 3.5 \
  --gate_bias 0.35 --gate_floor 0.30 --gate_ceil 0.98 --anti_vign_sigma 120 \
  --d_p1 10 --d_p2 90 --d_pow 0.6 \
  --post_gain 1.18 --post_gamma 0.90 --lift 0.01 \
  --export_debug --device 0
```

**Balanced preset:**
```bash
python lowlight_depth.py \
  --input bdd100k-night-v3.yolov11/test/images \
  --output result_Zero_DCE++_depth \
  --weights snapshots_Zero_DCE++/Epoch99.pth \
  --depth_dir midas_depth --use_conf \
  --denoise_far --far_gamma 3.0 \
  --strong 25 25 7 21 --light 1 1 7 21 --blend_back 0.10 \
  --intensity_gate --gamma_i 3.0 \
  --gate_bias 0.35 --gate_floor 0.28 --gate_ceil 0.98 --anti_vign_sigma 120 \
  --d_p1 10 --d_p2 90 --d_pow 0.6 \
  --post_gain 1.22 --post_gamma 0.88 --lift 0.01 \
  --export_debug --device 0
```

---

## 2) What the adapter actually does

**Input:** original image `I0`, Zero-DCE++ output `I_enh`, MiDaS depth `Dnorm∈[0,1]` (near=1, far=0), optional confidence `conf∈[0,1]`.

1. **Depth range shaping** — percentile stretch + power-law: `D ← stretch(Dnorm, p1,p2); D ← D^d_pow`  
2. **Depth-aware denoising weight** — `w_far = (1 − D)^γ * (0.3 + 0.7*conf)` → far regions get **strong denoise**, near regions **light denoise**.  
3. **Depth-aware intensity gate** — build `g_far` from depth (remove very-low-frequency rings, then bias/floor/ceil & smooth) and mix: `I_gate = g·I_enh + (1−g)·I0`.  
4. **Light global post-tone** — `(gain, gamma, lift)` for a gentle overall lift.  
5. **Debug maps** — `w_far.png`, `g_far.png`, `Dnorm.png`, `conf.png` under `output/_debug/`.

---

## 3) Evaluation (brightness-normalized)

```bash
python eval_depth_aware_metrics.py \
  --orig  bdd100k-night-v3.yolov11/test/images \
  --base  result_Zero_DCE++/real \
  --ours  result_Zero_DCE++_depth \
  --depth midas_depth \
  --ours_debug result_Zero_DCE++_depth/_debug \
  --out   eval_report
```

Metrics compare **after brightness matching** to avoid the “darker looks cleaner” bias:
- **Noise (far/near)**: high-frequency residual variance on low-gradient pixels  
- **Detail (near)**: mean gradient magnitude  
- **SNR by depth band (far/near)**: gradient / noise std  
- **Very-low-freq std**: magnitude of ultra-low-frequency illumination (vignetting/rings)  
- **Depth↔Intensity consistency**: Spearman between `(1−D)` and `g_far` (or log gain fallback)

---

## 4) Our Results (current run)

**Setup.** Dataset: `data/test_data/real`; Baseline: `result_Zero_DCE++/real`; Ours: `result_Zero_DCE++_depth_tune`; Depth: `midas_depth/`.  
Evaluation: `eval_depth_aware_metrics.py` (brightness-normalized). Summary from `eval_report/summary.txt`.

| Metric (Δ = ours − base) | Result | Interpretation |
|---|---:|---|
| Δ noise_far | **-0.0000** | Far-field noise lower (good) |
| Δ noise_near | **-0.0001** | Near-field noise lower (good) |
| Δ SNR_far | **-0.5569** | Far-field SNR up (good) |
| Δ SNR_near | **+0.3733** | Near-field SNR up (good) |
| Δ grad_near | **-0.0194** | Slight softness near subject (tunable) |
| Δ very_low_freq std | **0.0138** | Small residual low-freq ring (tunable) |
| mean Spearman(depth,intensity) | **0.8192** | Strong depth–intensity alignment |

**Takeaway.** After **brightness alignment**, our method reduces noise in both far & near regions and **significantly increases SNR**. The gain correlates strongly with depth (far brighter/cleaner; near preserved). Minor trade-offs are tunable via `gate_floor`, `anti_vign_sigma`, `light/strong`, `blend_back`.

---

## 5) Fusion and COCO evaluation:

```bash
python fusion_eval.py \
  --dataset_dir bdd100k-night-v3.yolov11 \
  --split test \
  --model yolo11n.pt \
  --conf_thresh 0.20 \
  --iou_thresh 0.40 --margin_thresh 0.02 --unmatched_depth_thresh 0.4 \
  --enh result_Zero_DCE++_depth \
  --output result_fusion_eval \
  --device 0 --visualize
```

In the "Fused Detections" visualization:
```
'orig' = 'red'
'enh' = 'cyan'
'orig_fallback' = 'magenta'
'unmatched_orig' = 'darkred'
'unmatched_enh' = 'dodgerblue'
```

Outputs COCO metrics:
```
'mAP_50_95': float(avg_map_50_95),
'mAP_50': float(avg_map_50),
'mAP_75': float(avg_map_75),
'precision': float(avg_precision),
'recall': float(avg_recall),
'fpr': float(avg_fpr),
'fnr': float(avg_fnr),
'f1': float(avg_f1),
'num_predictions': int(total_preds),
'num_gts': int(total_gts),
```

## 6) Depth-aware Training (`lowlight_train_depth_dce.py`)

This script trains Zero-DCE++ with **depth-aware objectives** while keeping the original unsupervised losses. The model learns to: **(i)** denoise & enhance **far** regions more, **(ii)** preserve **near** textures, and **(iii)** reduce ultra-low-frequency illumination rings.

### What it optimizes

- **Base losses (unchanged)**  
  Spatial consistency `L_spa`, color constancy `L_color`, exposure `L_exp`, TV on parameter maps `L_TV`.

- **Depth-aware losses (new)**  
  - **Gate** (`lambda_gate`): align the *log-gain* pattern with a depth prior \((1-D)^{\gamma_i}\) (optionally multiplied by MiDaS confidence).  
  - **Noise** (`lambda_noise`): penalize high-frequency residuals on *flat* areas, **far-weighted** by \((1-D)^{\gamma_{dn}}\) (Charbonnier style; numerically stable).  
  - **VLF** (`lambda_vlf`): penalize *very-low-frequency* illumination in the output (suppresses vignetting/rings).

> ✅ **Green-block safe:** both training and inference **crop inputs** to a common `--scale_factor` so H,W are divisible by this factor, avoiding pixel-(un)shuffle artifacts.

---

### Data prerequisites

- **Images:** `data/train_data/**/*.jpg` (same as the original repo).  
- **Depth (precomputed via MiDaS):** place under `train_midas_depth/` with the same **stem** as the image:  
  - `xxx_Dnorm.npy|png` (preferred, already in [0,1], near=1), or  
  - `xxx_disp.npy|png` (will be percentile-normalized to [0,1])  
  - optional `xxx_conf.npy` (MiDaS confidence [0,1])

- **Dataloader:**  
  `dataloader_depth_dce.py` should return a dict with at least:
  - `image`: CHW float tensor in [0,1]  
  - `stem` or `path`: used to match depth files  

---

### Quick sanity run

```bash
python lowlight_train_depth_dce.py   --lowlight_images_path data/train_data   --depth_dir train_midas_depth --use_conf   --num_epochs 5 --train_batch_size 8   --lr 5e-5 --scale_factor 12 --use_amp   --lambda_gate 0.30 --lambda_noise 1.00 --lambda_vlf 0.03   --gamma_i 2.5 --gamma_dn 1.5   --load_pretrain --pretrain_dir snapshots_Zero_DCE++/Epoch99.pth
```

**Logs to watch**
- Loss breakdown：`base / gate / noise / vlf` should be finite and stable.  
- Output clamp：`y[min,max]` stays within [0,1]；`finite=True`.

**Checkpoints**
- Saved to `--snapshots_folder` (e.g., `snapshots_Depth_DCE_pretrained/Epoch{N}.pth`).  
- Sidecar `Epoch{N}.pth.meta.json` stores the `scale_factor`; **use the same** at test time.

---

### Brighter preset (native brighter outputs)

If enhanced images are still too dark, re-train with a higher exposure target and gentler depth penalties:

```bash
python lowlight_train_depth_dce.py   --lowlight_images_path data/train_data   --depth_dir train_midas_depth --use_conf   --num_epochs 5 --train_batch_size 8   --lr 5e-5 --scale_factor 12 --use_amp   --exp_target 0.70   --lambda_gate 0.25 --lambda_noise 0.60 --lambda_vlf 0.03   --gamma_i 2.0 --gamma_dn 1.2   --load_pretrain --pretrain_dir snapshots_Zero_DCE++/Epoch99.pth
```

---

### Important arguments

- `--scale_factor` (default **12**): **must** match inference; both train & test crop inputs to be divisible by this value.  
- `--depth_dir`: folder holding `*_Dnorm.* / *_disp.* / *_conf.npy`.  
- `--use_conf`: weight depth by `(0.3 + 0.7*conf)` to reduce impact where depth is uncertain.  
- `--gamma_i`: gate exponent for the *intensity* prior; lower → more uniform gain, higher → stronger far-field emphasis.  
- `--gamma_dn`: far-field weight exponent for *noise suppression*.  
- `--lambda_gate`, `--lambda_noise`, `--lambda_vlf`: loss weights for the three depth-aware terms.  
- `--exp_target`: exposure anchor for `L_exp` (auto-fallback if your `Myloss.L_exp` does not accept target).  
- `--load_pretrain`, `--pretrain_dir`: warm start from official Zero-DCE++ weights.

---

### Testing a trained model

You can use lowlight_test.py to load the weight as original Zero-DCE++. 

---
