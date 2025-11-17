# Depth-Gated Low-Light Enhancement (README)

> Drop-in **post-processing adapter** that makes any low-light enhancement **depth-aware**: far regions get stronger denoising & more enhancement; near subjects keep textures. Zero training required. Works out-of-the-box on top of **Zero-DCE++**; easily portable to derain/dehaze.

---

## 1) Repository Layout (ours)

We added four scripts (non-intrusive to the original Zero-DCE++ code):

```
Zero-DCE++/
├── make_depth_masks.py              # (NEW) MiDaS depth/confidence precomputation
├── lowlight_depth.py                # (NEW) Depth-aware post-process for inference
├── vis_strength.py                  # (NEW) Quick correlation check (depth vs gain)
├── eval_depth_aware_metrics.py      # (NEW) Brightness-normalized evaluation suite
└── (original files: lowlight_test.py, lowlight_train.py, model.py, etc.)
```

Outputs:
```
midas_depth/                         # *_Dnorm.npy/png, *_conf.npy (optional)
result_Zero_DCEpp/                   # vanilla Zero-DCE++ baseline (unchanged)
result_Zero_DCEpp_depth_*/           # ours (presets), each with:
    └── _debug/                      # w_far.png, g_far.png, Dnorm.png, conf.png
eval_report/                         # depth_gate_eval.csv + summary.txt
```

---

## 2) Quick Start

### A) Prepare environment
- Python ≥ 3.8, PyTorch per your CUDA
- `opencv-python`, `numpy`, `torchvision`, `scipy`, `pandas`
- MiDaS (for offline depth)

### B) Precompute depth (one-time)
```bash
python make_depth_masks.py \
  --input data/test_data/real \
  --out   midas_depth \
  --short_side 512 --with_conf
```

### C) Produce the vanilla Zero-DCE++ baseline (unchanged)
Keep results under `result_Zero_DCEpp/...`

### D) Run our **depth-aware** adapter on top of Zero-DCE++

**Balanced preset (recommended for report):**
```bash
python lowlight_depth.py \
  --input data/test_data/real \
  --output result_Zero_DCEpp_depth_tune \
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

**Aggressive preset (max far-field suppression):**
```bash
python lowlight_depth.py \
  --input data/test_data/real \
  --output result_Zero_DCEpp_depth_strong \
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

---

## 3) What the adapter actually does

**Input:** original image `I0`, Zero-DCE++ output `I_enh`, MiDaS depth `Dnorm∈[0,1]` (near=1, far=0), optional confidence `conf∈[0,1]`.

1. **Depth range shaping** — percentile stretch + power-law: `D ← stretch(Dnorm, p1,p2); D ← D^d_pow`  
2. **Depth-aware denoising weight** — `w_far = (1 − D)^γ * (0.3 + 0.7*conf)` → far regions get **strong denoise**, near regions **light denoise**.  
3. **Depth-aware intensity gate** — build `g_far` from depth (remove very-low-frequency rings, then bias/floor/ceil & smooth) and mix: `I_gate = g·I_enh + (1−g)·I0`.  
4. **Light global post-tone** — `(gain, gamma, lift)` for a gentle overall lift.  
5. **Debug maps** — `w_far.png`, `g_far.png`, `Dnorm.png`, `conf.png` under `output/_debug/`.

---

## 4) Evaluation (brightness-normalized)

```bash
python eval_depth_aware_metrics.py \
  --orig  data/test_data/real \
  --base  result_Zero_DCEpp/real \
  --ours  result_Zero_DCEpp_depth_tune \
  --depth midas_depth \
  --ours_debug result_Zero_DCEpp_depth_tune/_debug \
  --out   eval_report
```

Metrics compare **after brightness matching** to avoid the “darker looks cleaner” bias:
- **Noise (far/near)**: high-frequency residual variance on low-gradient pixels  
- **Detail (near)**: mean gradient magnitude  
- **SNR by depth band (far/near)**: gradient / noise std  
- **Very-low-freq std**: magnitude of ultra-low-frequency illumination (vignetting/rings)  
- **Depth↔Intensity consistency**: Spearman between `(1−D)` and `g_far` (or log gain fallback)

---

## 5) Our Results (current run)

**Setup.** Dataset: `data/test_data/real`; Baseline: `result_Zero_DCEpp/real`; Ours: `result_Zero_DCEpp_depth_tune`; Depth: `midas_depth/`.  
Evaluation: `eval_depth_aware_metrics.py` (brightness-normalized). Summary from `eval_report/summary.txt`.

| Metric (Δ = ours − base) | Result | Interpretation |
|---|---:|---|
| Δ noise_far | **−0.0006** | Far-field noise lower (good) |
| Δ noise_near | **−0.0005** | Near-field noise lower (good) |
| Δ SNR_far | **+0.4019** | Far-field SNR up (good) |
| Δ SNR_near | **+0.4772** | Near-field SNR up (good) |
| Δ grad_near | **−0.0474** | Slight softness near subject (tunable) |
| Δ very_low_freq std | **+0.0147** | Small residual low-freq ring (tunable) |
| mean Spearman(depth,intensity) | **0.9463** | Strong depth–intensity alignment |

**Takeaway.** After **brightness alignment**, our method reduces noise in both far & near regions and **significantly increases SNR**. The gain correlates strongly with depth (far brighter/cleaner; near preserved). Minor trade-offs are tunable via `gate_floor`, `anti_vign_sigma`, `light/strong`, `blend_back`.