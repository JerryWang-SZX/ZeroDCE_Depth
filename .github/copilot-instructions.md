# Copilot Instructions for ZeroDCE_Depth

## Project Overview

This is a **depth-aware low-light image enhancement** system that extends Zero-DCE++ with MiDaS-based depth guidance. The core innovation: far regions get aggressive denoising + enhancement; near subjects preserve textures. It's a post-processing adapter requiring no retraining.

## Architecture & Data Flow

### Pipeline Stages
1. **Depth Estimation** (`make_depth_masks.py`): MiDaS(DPT-Hybrid) generates normalized depth `Dnorm ∈ [0,1]` (near=1, far=0) + optional confidence maps for all test images
2. **Zero-DCE++ Inference** (`lowlight_test.py`): Standard baseline enhancement
3. **Depth-Aware Post-Processing** (`lowlight_depth.py`): Applies depth-gated denoising + intensity gating on top of enhanced output
4. **Object Detection Fusion** (`fusion_detections.py`): YOLO detections fused from original + enhanced branches using depth gate confidence
5. **Evaluation** (`eval_depth_aware_metrics.py`): Brightness-normalized metrics (noise, SNR, detail preservation, depth-intensity consistency)

### Output Directories
- `midas_depth/`: `*_Dnorm.npy`, `*_conf.npy` (depth and confidence from MiDaS)
- `result_Zero_DCE++/`: Baseline Zero-DCE++ outputs (reference)
- `result_Zero_DCE++_depth/`: Depth-gated enhanced images + `_debug/` subfolder with `w_far.png`, `g_far.png`, `Dnorm.png`, `conf.png` maps
- `eval_report/`: CSV metrics and summary statistics

## Critical Design Patterns

### Depth-Aware Denoising (`lowlight_depth.py`)
```python
# Percentile + power-law stretch for robust depth remapping
D = _norm01(Dnorm, p1=d_p1, p2=d_p2)  # Percentile clipping
D = D ** d_pow  # Power-law to emphasize far/near regions

# Denoising weight: far regions (D≈0) get w_far≈1.0
w_far = (1 - D) ** far_gamma * (0.3 + 0.7 * conf)
I_denoised = w_far * bilateral_denoise(I_enh) + (1 - w_far) * I_enh
```
**Key parameters**: `d_p1=10`, `d_p2=90`, `d_pow=0.6` are pre-tuned; `far_gamma` controls denoising aggression (2.0-3.5).

### Intensity Gating (Depth-Controlled Brightness)
```python
# Build g_far from depth: remove low-freq rings, then bias/floor/ceil + smooth
g_far = gaussian_blur(D, sigma=anti_vign_sigma)
g_far = clip(g_far * (1 + gate_bias), gate_floor, gate_ceil)
I_gate = g_far * I_enh + (1 - g_far) * I0  # Mix enhanced/original
```
**Tuning knobs**: `gate_bias` (±0.35), `gate_floor` (0.28), `gate_ceil` (0.98) control far/near brightness balance. `anti_vign_sigma=120` removes vignetting rings.

### Brightness-Normalized Metrics (`eval_depth_aware_metrics.py`)
All comparisons **after brightness matching** to avoid "darker=cleaner" bias:
```python
def match_meanY(img, tgt_mean=0.25):
    y = grayscale(img)
    g = tgt_mean / (y.mean() + eps)
    return np.clip(img * g, 0, 1)
```
Metrics: `noise_far/near` (high-freq residual), `detail_near` (gradient energy), `SNR` (gradient/noise), `very_low_freq_std` (vignetting).

## File Responsibility Map

| File | Purpose | Key Outputs |
|------|---------|-------------|
| `model.py` | Zero-DCE++ architecture (CSDN_Tem blocks, enhance_net_nopool) | PyTorch nn.Module |
| `lowlight_train.py` | Training loop with color/spatial/exposure/TV losses | checkpoint.pth |
| `lowlight_test.py` | Baseline inference, saves to `result_Zero_DCE++/` | Enhanced images |
| `make_depth_masks.py` | MiDaS inference + percentile normalization | `*_Dnorm.npy`, `*_conf.npy` |
| `lowlight_depth.py` | Depth-gated post-processing (denoising + gating) | Enhanced + debug maps |
| `fusion_detections.py` | YOLO inference + IoU-based detection fusion | Fused detections + stats |
| `eval_depth_aware_metrics.py` | Brightness-aligned metrics calculation | CSV + summary report |
| `dataloader.py` | Training dataset loader (512×512 crops) | PyTorch DataLoader |
| `Myloss.py` | Custom losses: L_color, L_spa, L_exp, L_TV | Loss functions |

## Command Patterns & Workflows

### Standard Quick Workflow
```bash
# 1. Depth precompute (one-time)
python make_depth_masks.py --dataset_dir bdd100k-night-v3.yolov11 --split test --out midas_depth --with_conf

# 2. Baseline (unchanged)
python lowlight_test.py

# 3. Depth-aware enhancement (balanced preset)
python lowlight_depth.py --input bdd100k-night-v3.yolov11/test/images --output result_Zero_DCE++_depth \
  --weights snapshots_Zero_DCE++/Epoch99.pth --depth_dir midas_depth --use_conf \
  --denoise_far --far_gamma 3.0 --intensity_gate --gamma_i 3.0 \
  --d_p1 10 --d_p2 90 --d_pow 0.6 --gate_bias 0.35 --export_debug

# 4. Evaluate
python eval_depth_aware_metrics.py --orig bdd100k-night-v3.yolov11/test/images \
  --base result_Zero_DCE++/real --ours result_Zero_DCE++_depth --depth midas_depth \
  --out eval_report

# 5. Fusion (optional)
python fusion_detections.py --dataset_dir bdd100k-night-v3.yolov11 --split test \
  --enh result_Zero_DCE++_depth --output result_fusion --visualize
```

### Preset Tuning Principles
- **Aggressive (max far suppression)**: `far_gamma=3.5`, `strong=30,30,7,21`, `light=0,0,7,21`, `gate_floor=0.30`
- **Balanced (default)**: `far_gamma=3.0`, `strong=25,25,7,21`, `light=1,1,7,21`, `gate_floor=0.28`
- **Conservative**: `far_gamma=2.0`, `strong=20,20,7,21`, `light=3,3,7,21`, `gate_floor=0.25`

Parameters are independent: `--post_gain`, `--post_gamma`, `--lift` apply gentle global tone curves after depth gating.

## Common Modifications & Patterns

### Adding a New Enhancement Stage
1. Add argument in argparse (e.g., `--my_flag`)
2. Load depth in `_load_depth_for_stem()` if needed
3. Apply transformation in `process_batch()` between existing stages
4. Export debug map if `args.export_debug` is True: `cv.imwrite(f"{debug_dir}/{stem}_my_map.png", map_vis)`

### Modifying Loss Function
Edit `Myloss.py`:
- `L_color`: RGB channel consistency (prevent color cast)
- `L_spa`: Spatial smoothness on enhanced image gradients
- `L_exp`: Exposure (target mean brightness ≈0.5)
- `L_TV`: Total variation (suppress oscillations)

Combine in `lowlight_train.py` with weights, e.g., `loss = a*L_color + b*L_spa + c*L_exp + d*L_TV`.

### Extending Evaluation Metrics
In `eval_depth_aware_metrics.py`, new metrics follow this pattern:
1. Load brightness-aligned images: `img_base_match, _ = match_meanY(img_base)`
2. Compute metric from images: `metric = compute_my_metric(img_base_match, img_ours_match, depth, ...)`
3. Store in DataFrame: `metrics_list.append({"image": stem, "my_metric": metric})`
4. Export to CSV via pandas

## Dependencies & Environment

**Core packages**: PyTorch (CUDA-enabled), torchvision, opencv-python, numpy, scipy, pandas, Pillow, ultralytics (YOLO), tqdm

**External models**: MiDaS (loaded via `torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")`), YOLO11n (`yolo11n.pt` downloaded automatically)

**GPU requirement**: All inference scripts assume `--device 0` (GPU:0); CPU mode not tested.

## Debugging Tips

- **Depth artifacts**: Check `midas_depth/*_Dnorm.png` visually; if inverted, manually flip with `1 - D`
- **Extreme output**: Debug maps in `result_Zero_DCE++_depth/_debug/` show `w_far.png` (denoising weight) and `g_far.png` (gating). If banding, increase `anti_vign_sigma`
- **Eval fails**: Ensure `result_Zero_DCE++/real/` exists (baseline); run `lowlight_test.py` first
- **Memory issues**: Reduce batch size via `--batch_size` in inference scripts, or lower input resolution
