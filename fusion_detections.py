#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion_detections.py

Depth-aware object detection fusion:
1. Run YOLO on original image, detections_orig
2. Run YOLO on enhanced image, detections_enh
3. Load continuous depth gate g_far from detections_enh/_debug/
4. Cross-branch clustering: same class and IoU >= 0.60
5. For each candidate box pair, compute mean gate g_bar inside the box
6. Weighted score: score = confidence*(1 + g_bar)/2
7. Pick higher score; if margin < 0.02, fallback to original branch
8. Export fused detections as JSON and provide statistics
"""

import os
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import time

import numpy as np
import cv2 as cv
from PIL import Image
from tqdm import tqdm

import torch
from ultralytics import YOLO

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# -------------------- Utility Functions --------------------
def load_image(image_path: str) -> np.ndarray:
    im = cv.imread(image_path, cv.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Image not found:{image_path}")
    return cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

def load_depth_gate(depth_debug_dir: Path, stem: str) -> Optional[np.ndarray]:
    candidates = [ # Check .npy first, then .png
        depth_debug_dir / f"{stem}_g_far.npy",
        depth_debug_dir / f"{stem}_g_far.png"
    ]
    for cand in candidates:
        if not cand.exists(): continue
        if cand.suffix == ".npy": # Prioritize .npy
            return np.load(cand).astype(np.float32)
        im = cv.imread(str(cand), cv.IMREAD_GRAYSCALE)
        if im is not None:
            return (im.astype(np.float32) / 255.0)
    return None

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    # Intersection
    x1_i = max(box1[0], box2[0]) # [x1, y1, x2, y2]
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    
    if x2_i < x1_i or y2_i < y1_i: return 0.0
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0: return 0.0
    
    return inter/union

def mean_depth_gate_in_box(gate: np.ndarray, box: np.ndarray) -> float:
    h, w = gate.shape[:2]
    
    # Assume box is in pixel coordinates
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return 0.5  # Default fallback
    
    roi = gate[y1:y2, x1:x2]
    return float(np.mean(roi))

def run_yolo_inference(model: YOLO, image_path: str, conf_thresh: float = 0.45) -> List[Dict]:
    results = model(image_path, conf=conf_thresh, verbose=False)[0]
    
    detections = []
    for det in results.boxes.data:  # tensor of shape [N, 6] (x1, y1, x2, y2, conf, cls)
        x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
        class_id = int(cls_id)
        class_name = results.names[class_id]
        
        detections.append({
            'class_id': class_id,
            'class_name': class_name,
            'confidence': float(conf),
            'box': np.array([x1, y1, x2, y2], dtype=np.float32),
        })
    
    return detections

def find_matching_pairs(detections_orig: List[Dict], detections_enh: List[Dict], threshold: float = 0.60) -> List[Tuple[int, int, float]]:
    # Group detections by class in orig and enh
    orig_by_class = {}
    for i, det in enumerate(detections_orig):
        class_id = det['class_id']
        if class_id not in orig_by_class:
            orig_by_class[class_id] = []
        orig_by_class[class_id].append((i, det))
    
    enh_by_class = {}
    for i, det in enumerate(detections_enh):
        class_id = det['class_id']
        if class_id not in enh_by_class:
            enh_by_class[class_id] = []
        enh_by_class[class_id].append((i, det))
    
    pairs = [] # (idx_orig, idx_enh, iou)
    matched_orig = set()
    matched_enh = set()
    
    # For all classes in orig and enh
    for class_id in orig_by_class:
        if class_id not in enh_by_class: continue
        
        orig_list = orig_by_class[class_id]
        enh_list = enh_by_class[class_id]
        
        # Append IoUs above the treshold
        ious = []
        for i_orig, (idx_orig, det_orig) in enumerate(orig_list):
            for i_enh, (idx_enh, det_enh) in enumerate(enh_list):
                iou = compute_iou(det_orig['box'], det_enh['box'])
                if iou >= threshold:
                    ious.append((iou, i_orig, i_enh, idx_orig, idx_enh))
        
        # Sort by IoU descending
        ious.sort(reverse=True)
        for iou, i_orig, i_enh, idx_orig, idx_enh in ious:
            if idx_orig not in matched_orig and idx_enh not in matched_enh:
                pairs.append((idx_orig, idx_enh, iou))
                matched_orig.add(idx_orig)
                matched_enh.add(idx_enh)
    
    return pairs

def fuse_detections(
    detections_orig: List[Dict],
    detections_enh: List[Dict],
    gate_enh: Optional[np.ndarray],
    pairs: List[Tuple[int, int, float]],
    fallback_margin_thresh: float = 0.02,
    unmatched_g_bar_thresh: float = 0.7
) -> Tuple[List[Dict], Dict]:
    fused = []
    stats = {
        'total_pairs': len(pairs),
        'chosen_orig': 0,
        'chosen_enh': 0,
        'fallback_orig': 0,
        'unmatched_orig': 0,
        'unmatched_enh': 0,
    }
    
    matched_orig_set = set()
    matched_enh_set = set()
    
    # For all matched pairs in orig and enh
    for idx_orig, idx_enh, iou in pairs:
        matched_orig_set.add(idx_orig)
        matched_enh_set.add(idx_enh)
        
        det_orig = detections_orig[idx_orig].copy()
        det_enh = detections_enh[idx_enh].copy()
        
        # Compute weighted scores
        score_orig = det_orig['confidence'] * (1.0 + 0.5) / 2.0 # 0.5 g_bar for orig
        g_bar_enh = mean_depth_gate_in_box(gate_enh, det_enh['box']) if gate_enh is not None else 0.5
        score_enh = det_enh['confidence'] * (1.0 + g_bar_enh) / 2.0
        
        # Pick higher score
        if score_enh > score_orig:
            margin = (score_enh - score_orig) / (score_enh + 1e-8)
            if margin < fallback_margin_thresh:
                # Fallback to original
                det_orig['source'] = 'orig_fallback'
                det_orig['pair_iou'] = iou
                det_orig['gate_enh'] = g_bar_enh
                fused.append(det_orig) # Fused detection is orig
                stats['fallback_orig'] += 1
            else:
                # Use enhanced
                det_enh['source'] = 'enh'
                det_enh['pair_iou'] = iou
                det_enh['gate_enh'] = g_bar_enh
                fused.append(det_enh) # Fused detection is enh
                stats['chosen_enh'] += 1
        else:
            det_orig['source'] = 'orig'
            det_orig['pair_iou'] = iou
            det_orig['gate_enh'] = g_bar_enh
            fused.append(det_orig) # Fused detection is orig
            stats['chosen_orig'] += 1
    
    # All unmatched detections from orig
    for i, det in enumerate(detections_orig):
        if i not in matched_orig_set:
            det = det.copy()
            det['source'] = 'unmatched_orig'
            det['pair_iou'] = None
            det['gate_enh'] = None
            fused.append(det)
            stats['unmatched_orig'] += 1
    
    # Add unmatched detections from enh
    if gate_enh is not None:
        for i, det in enumerate(detections_enh):
            if i not in matched_enh_set:
                g_bar = mean_depth_gate_in_box(gate_enh, det['box'])
                if g_bar > unmatched_g_bar_thresh:  # Only add g_bar is above threshold
                    det = det.copy()
                    det['source'] = 'unmatched_enh_highgate'
                    det['pair_iou'] = None
                    det['gate_enh'] = g_bar
                    fused.append(det)
                    stats['unmatched_enh'] += 1
    
    return fused, stats

def visualize_fused_detections(image_orig: np.ndarray, detections_orig: List[Dict], image_enh: np.ndarray, detections_enh: List[Dict], fused: List[Dict], output_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))

    axes[0].imshow(image_orig)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for det in detections_orig:
        x1, y1, x2, y2 = det['box']
        w_box = x2 - x1
        h_box = y2 - y1

        color = 'red'

        rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor=color, facecolor='none')
        axes[0].add_patch(rect)
        
        label = f"{det['class_name']} {det['confidence']:.2f}"
        axes[0].text(x1, y1 - 5, label, color=color, fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
    
    axes[1].imshow(image_enh)
    axes[1].set_title("Enhanced Image")
    axes[1].axis('off')

    for det in detections_enh:
        x1, y1, x2, y2 = det['box']
        w_box = x2 - x1
        h_box = y2 - y1

        color = 'green'

        rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor=color, facecolor='none')
        axes[1].add_patch(rect)
        
        label = f"{det['class_name']} {det['confidence']:.2f}"
        axes[1].text(x1, y1 - 5, label, color=color, fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

    axes[2].imshow(image_enh)
    axes[2].set_title("Fused Detections on Enhanced")
    axes[2].axis('off')
    
    # Overlay fused detections on enhanced image
    for det in fused:
        x1, y1, x2, y2 = det['box']
        w_box = x2 - x1
        h_box = y2 - y1
        
        source = det['source']
        if source == 'orig': color = 'red' # detect from orig
        elif source == 'enh': color = 'green' # detect from enh
        elif source == 'orig_fallback': color = 'blue' # detect fallback to orig
        else: color = 'yellow' # unmatched
        
        rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor=color, facecolor='none')
        axes[2].add_patch(rect)
        
        label = f"{det['class_name']} {det['confidence']:.2f}"
        axes[2].text(x1, y1 - 5, label, color=color, fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def save_detections_json(fused: List[Dict], output_path: str):
    # We can change the format and tracked statistics as needed
    data = []
    for det in fused:
        box_list = [float(x) for x in det.get('box').tolist()]

        det_json = {
            'class_id': int(det.get('class_id')) if det.get('class_id') is not None else None,
            'class_name': str(det.get('class_name')) if det.get('class_name') is not None else None,
            'confidence': float(det.get('confidence')) if det.get('confidence') is not None else None,
            'box_xyxy': box_list,
            'source': str(det.get('source')) if det.get('source') is not None else None,
            'pair_iou': float(det.get('pair_iou')) if det.get('pair_iou') is not None else None,
            'gate_enh': float(det.get('gate_enh')) if det.get('gate_enh') is not None else None,
        }
        data.append(det_json)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# -------------------- Main Function --------------------
def main(args):
    # Setup device
    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    print(f"Using device:{device}")
    
    # Load YOLO model
    print(f"YOLO model:{args.model}")
    yolo = YOLO(args.model)
    yolo.to(device)
    
    # Setup dir paths
    base = Path(args.dataset_dir)
    orig_dir = base / args.split / 'images'
    enh_dir = Path(args.enh)
    depth_debug_dir = enh_dir / '_debug' # Depth debug in enh_dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images in orig dir
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(str(orig_dir / '**' / pattern), recursive=True))
    
    if not image_files:
        print(f"No images found in {orig_dir}")
        return
    
    image_files.sort()
    print(f"{len(image_files)} images in {orig_dir}")
    
    # Process each image
    all_fused = {}
    all_stats = {
        'total_images': len(image_files),
        'total_pairs': 0,
        'chosen_orig': 0,
        'chosen_enh': 0,
        'fallback_orig': 0,
        'unmatched_orig': 0,
        'unmatched_enh': 0,
    }
    
    start_time = time.time()
    
    for image_path in tqdm(image_files, desc="Processing images"):
        image_path = Path(image_path)
        stem = image_path.stem
        rel_path = image_path.relative_to(orig_dir)
        
        # Get corresponding enhanced image path
        enh_image_path = enh_dir / rel_path
        if not enh_image_path.exists():
            print(f"Warning: No enhanced image for {image_path}!, skipping")
            continue
        
        # Load depth gate
        gate_enh = None
        if depth_debug_dir is not None:
            gate_enh = load_depth_gate(depth_debug_dir, stem)
        
        # YOLO on orig and enh
        detections_orig = run_yolo_inference(yolo, str(image_path), args.conf_thresh)
        detections_enh = run_yolo_inference(yolo, str(enh_image_path), args.conf_thresh)
        
        # Find matching pairs
        pairs = find_matching_pairs(detections_orig, detections_enh, args.iou_thresh)
        
        # Fuse detections
        fused, stats = fuse_detections(detections_orig, detections_enh, gate_enh, pairs, args.margin_thresh, args.unmatched_thresh)
        all_fused[stem] = fused
        
        # Update total stats
        all_stats['total_pairs'] += stats['total_pairs']
        all_stats['chosen_orig'] += stats['chosen_orig']
        all_stats['chosen_enh'] += stats['chosen_enh']
        all_stats['fallback_orig'] += stats['fallback_orig']
        all_stats['unmatched_orig'] += stats['unmatched_orig']
        all_stats['unmatched_enh'] += stats['unmatched_enh']
        
        # Save JSON
        json_path = output_dir / f"{stem}_fused.json"
        save_detections_json(fused, str(json_path))
        
        # Export visualization image can be slow
        if args.visualize:
            image_orig = load_image(str(image_path))
            image_enh = load_image(str(enh_image_path))
            vis_path = output_dir / f"{stem}_fused.png"
            visualize_fused_detections(image_orig, detections_orig, image_enh, detections_enh, fused, str(vis_path))
    
    elapsed = time.time() - start_time
    
    # Save summary
    summary = {
        'elapsed_seconds': elapsed,
        'images_processed': len(all_fused),
        'statistics': all_stats,
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nElapsed time: {elapsed:.2f}s")
    print(f"Images processed: {len(all_fused)}")
    print(f"Total matched pairs: {all_stats['total_pairs']}")
    print(f"  Chosen from original: {all_stats['chosen_orig']}")
    print(f"  Chosen from enhanced: {all_stats['chosen_enh']}")
    print(f"  Fallback to original: {all_stats['fallback_orig']}")
    print(f"Unmatched detections:")
    print(f"  From original: {all_stats['unmatched_orig']}")
    print(f"  From enhanced (high gate): {all_stats['unmatched_enh']}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth-gated object detection fusion")
    
    # Input/Output
    parser.add_argument('--enh', type=str, required=False, default='result_Zero_DCE++_depth', help='Enhanced images directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory for fused detections')

    # Dataset (BDD100K folder)
    parser.add_argument('--dataset_dir', type=str, default='bdd100k-night-v3.yolov11', help='Dataset root')
    parser.add_argument('--split', type=str, default='test', choices=['test','train','valid'], help='Dataset split under dataset_dir to use')
    
    # YOLO
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--conf_thresh', type=float, default=0.45, help='YOLO confidence threshold')
    
    # Fusion parameters
    parser.add_argument('--iou_thresh', type=float, default=0.60, help='IoU threshold for matching')
    parser.add_argument('--margin_thresh', type=float, default=0.02, help='Score margin threshold for fallback')
    parser.add_argument('--unmatched_thresh', type=float, default=0.7, help='g_bar margin for unmatched detections from enhanced')
    
    # Device
    parser.add_argument('--device', type=int, default=0, help='GPU device ID (or -1 for CPU)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()

    main(args)