#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion_eval.py

Depth-aware object detection fusion with COCO evaluation:
1. Run YOLO on original image -> detections_orig
2. Run YOLO on enhanced image -> detections_enh
3. Fuse detections using depth gate g_far
4. Evaluate COCO metrics for original, enhanced, and fused detections
5. Compare results side-by-side with visualizations

Outputs:
- JSON files with detections (orig, enh, fused)
- Visualization plots (GT, orig, enh, fused)
- Summary statistics with COCO metrics
"""

import os
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

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

def get_bdd100k_class_mapping():
    """Convert common COCO class IDs to BDD100K class IDs."""
    return {
        # COCO ID -> BDD100K ID
        0: 14,   # person
        1: 15,   # bicycle
        2: 12,   # car
        3: 13,   # motorcycle
        5: 11,   # bus
        7: 21,   # truck
        9: 17,   # traffic light
        10: 16,  # fire hydrant
        11: 18,  # stop sign
    }

def run_yolo_inference(model: YOLO, image_path: str, conf_thresh: float, max_box_width_ratio: float = 0.8) -> List[Dict]:
    """Run YOLO inference on a single image."""
    results = model(image_path, conf=conf_thresh, verbose=False)[0]
    
    # Get image dimensions to check box sizes
    img = cv.imread(image_path)
    img_width = img.shape[1] if img is not None else None
    
    class_mapping = get_bdd100k_class_mapping()
    detections = []
    
    for det in results.boxes.data:  # tensor of shape [N, 6] (x1, y1, x2, y2, conf, cls)
        x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
        coco_class_id = int(cls_id)
        bdd100k_class_id = class_mapping.get(coco_class_id, -1)
        
        # Skip if class not in mapping
        if bdd100k_class_id < 0:
            continue
        
        # Skip if box is too wide (likely covers most of image like car hood)
        box_width = x2 - x1
        if img_width is not None and box_width > max_box_width_ratio * img_width:
            continue
        
        class_name = results.names[coco_class_id]
        
        detections.append({
            'class_id': bdd100k_class_id,
            'class_name': class_name,
            'confidence': float(conf),
            'box': np.array([x1, y1, x2, y2], dtype=np.float32),
        })
    
    return detections

def find_matching_pairs(detections_orig: List[Dict], detections_enh: List[Dict], threshold: float) -> List[Tuple[int, int, float]]:
    """Find matching detections between original and enhanced using IoU."""
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
    
    pairs = []
    matched_orig = set()
    matched_enh = set()
    
    for class_id in orig_by_class:
        if class_id not in enh_by_class:
            continue
        
        orig_list = orig_by_class[class_id]
        enh_list = enh_by_class[class_id]
        
        ious = []
        for i_orig, (idx_orig, det_orig) in enumerate(orig_list):
            for i_enh, (idx_enh, det_enh) in enumerate(enh_list):
                iou = compute_iou(det_orig['box'], det_enh['box'])
                if iou >= threshold:
                    ious.append((iou, i_orig, i_enh, idx_orig, idx_enh))
        
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
    fallback_margin_thresh: float,
    unmatched_depth_thresh: float
) -> Tuple[List[Dict], Dict]:
    """Fuse detections from original and enhanced branches."""
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
    
    for idx_orig, idx_enh, iou in pairs:
        matched_orig_set.add(idx_orig)
        matched_enh_set.add(idx_enh)
        
        det_orig = detections_orig[idx_orig].copy()
        det_enh = detections_enh[idx_enh].copy()
        
        g_bar_enh = mean_depth_gate_in_box(gate_enh, det_enh['box']) if gate_enh is not None else 0.5
        score_orig = det_orig['confidence'] * (1.0 + (1.0-g_bar_enh)) / 2.0
        score_enh = det_enh['confidence'] * (1.0 + g_bar_enh) / 2.0
        
        if score_enh > score_orig:
            margin = (score_enh - score_orig) / (score_enh + 1e-8)
            if margin < fallback_margin_thresh:
                det_orig['source'] = 'orig_fallback'
                det_orig['pair_iou'] = iou
                det_orig['gate_enh'] = g_bar_enh
                fused.append(det_orig)
                stats['fallback_orig'] += 1
            else:
                det_enh['source'] = 'enh'
                det_enh['pair_iou'] = iou
                det_enh['gate_enh'] = g_bar_enh
                fused.append(det_enh)
                stats['chosen_enh'] += 1
        else:
            det_orig['source'] = 'orig'
            det_orig['pair_iou'] = iou
            det_orig['gate_enh'] = g_bar_enh
            fused.append(det_orig)
            stats['chosen_orig'] += 1
    
    for i, det in enumerate(detections_orig):
        if i not in matched_orig_set:
            det = det.copy()
            det['source'] = 'unmatched_orig'
            det['pair_iou'] = None
            det['gate_enh'] = None
            fused.append(det)
            stats['unmatched_orig'] += 1
    
    if gate_enh is not None:
        for i, det in enumerate(detections_enh):
            if i not in matched_enh_set:
                g_bar = mean_depth_gate_in_box(gate_enh, det['box'])
                
                if g_bar > unmatched_depth_thresh:
                    det = det.copy()
                    det['source'] = 'unmatched_enh'
                    det['pair_iou'] = None
                    det['gate_enh'] = g_bar
                    fused.append(det)
                    stats['unmatched_enh'] += 1
    
    return fused, stats

def compute_coco_metrics(detections: List[Dict], gt_dets: List[Dict], iou_thresholds: List[float]) -> Dict[str, float]:
    """Compute COCO metrics for a single image."""
    
    gt_by_class = {}
    for gt in gt_dets:
        cid = gt['class_id']
        if cid not in gt_by_class:
            gt_by_class[cid] = []
        gt_by_class[cid].append(gt)
    
    sorted_preds = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    tp = np.zeros((len(sorted_preds), len(iou_thresholds)), dtype=np.uint8)
    fp = np.zeros((len(sorted_preds), len(iou_thresholds)), dtype=np.uint8)
    
    best_ious = []
    best_gt_indices = []
    
    for pred in sorted_preds:
        cid = pred['class_id']
        gts = gt_by_class.get(cid, [])
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gts):
            iou = compute_iou(pred['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        best_ious.append(best_iou)
        best_gt_indices.append(best_gt_idx)
    
    for iou_t_idx, iou_thresh in enumerate(iou_thresholds):
        matched_gt_at_thresh = {}
        
        for pred_idx, (pred, best_iou, best_gt_idx) in enumerate(zip(sorted_preds, best_ious, best_gt_indices)):
            cid = pred['class_id']
            if cid not in matched_gt_at_thresh:
                matched_gt_at_thresh[cid] = set()
            
            if best_iou >= iou_thresh and best_gt_idx >= 0 and best_gt_idx not in matched_gt_at_thresh[cid]:
                tp[pred_idx, iou_t_idx] = 1
                matched_gt_at_thresh[cid].add(best_gt_idx)
            else:
                fp[pred_idx, iou_t_idx] = 1
    
    total_tp = np.cumsum(tp, axis=0)
    total_fp = np.cumsum(fp, axis=0)
    total_num_gt = sum(len(gt_class) for gt_class in gt_by_class.values())
    
    aps = []
    recalls = []
    for iou_t_idx in range(len(iou_thresholds)):
        tp_curve = total_tp[:, iou_t_idx]
        fp_curve = total_fp[:, iou_t_idx]
        
        recall = tp_curve / (total_num_gt + 1e-10)
        precision = tp_curve / (tp_curve + fp_curve + 1e-10)
        
        recall_np = np.concatenate(([0.0], recall, [1.0]))
        precision_np = np.concatenate(([0.0], precision, [0.0]))
        
        for i in range(len(precision_np) - 1, 0, -1):
            precision_np[i - 1] = np.maximum(precision_np[i - 1], precision_np[i])
        
        i = np.where(recall_np[1:] != recall_np[:-1])[0]
        ap = np.sum((recall_np[i + 1] - recall_np[i]) * precision_np[i + 1])
        aps.append(float(ap))
        recalls.append(float(recall[-1]) if len(recall) > 0 else 0.0)
    
    map_50 = aps[0] if len(aps) > 0 else 0.0
    map_50_95 = np.mean(aps) if len(aps) > 0 else 0.0
    recall_avg = np.mean(recalls) if len(recalls) > 0 else 0.0
    
    return {
        'mAP_50_95': float(map_50_95),
        'mAP_50': float(map_50),
        'recall': float(recall_avg),
        'num_predictions': len(sorted_preds),
        'num_gts': total_num_gt,
    }

def load_ground_truth_from_labels(label_path: str, image_width: int, image_height: int) -> List[Dict]:
    """Load ground truth from YOLO label file."""
    gt_dets = []
    
    if not Path(label_path).exists():
        return gt_dets
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            x1 = (x_center - width / 2) * image_width
            y1 = (y_center - height / 2) * image_height
            x2 = (x_center + width / 2) * image_width
            y2 = (y_center + height / 2) * image_height
            
            gt_dets.append({
                'class_id': class_id,
                'box': np.array([x1, y1, x2, y2], dtype=np.float32),
            })
    
    return gt_dets

def visualize_detections(image_gt: np.ndarray, gt_dets: List[Dict],
                        image_orig: np.ndarray, detections_orig: List[Dict],
                        image_enh: np.ndarray, detections_enh: List[Dict],
                        fused: List[Dict], output_path: str):
    """Visualize GT, original, enhanced, and fused detections in 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # GT (top-left)
    axes[0, 0].imshow(image_gt)
    axes[0, 0].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    for gt_box in gt_dets:
        x1, y1, x2, y2 = gt_box['box']
        w_box = x2 - x1
        h_box = y2 - y1
        
        rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor='green', facecolor='none')
        axes[0, 0].add_patch(rect)
        
        label = f"GT:{gt_box.get('class_id', 0)}"
        axes[0, 0].text(x1, y1 - 5, label, color='green', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

    # Original (top-right)
    axes[0, 1].imshow(image_orig)
    axes[0, 1].set_title("Original Detections", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    for det in detections_orig:
        x1, y1, x2, y2 = det['box']
        w_box = x2 - x1
        h_box = y2 - y1

        rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor='red', facecolor='none')
        axes[0, 1].add_patch(rect)
        
        label = f"{det['class_name']} {det['confidence']:.2f}"
        axes[0, 1].text(x1, y1 - 5, label, color='red', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

    # Enhanced (bottom-left)
    axes[1, 0].imshow(image_enh)
    axes[1, 0].set_title("Enhanced Detections", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    for det in detections_enh:
        x1, y1, x2, y2 = det['box']
        w_box = x2 - x1
        h_box = y2 - y1

        rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor='cyan', facecolor='none')
        axes[1, 0].add_patch(rect)
        
        label = f"{det['class_name']} {det['confidence']:.2f}"
        axes[1, 0].text(x1, y1 - 5, label, color='cyan', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

    # Fused (bottom-right)
    axes[1, 1].imshow(image_enh)
    axes[1, 1].set_title("Fused Detections", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    for det in fused:
        x1, y1, x2, y2 = det['box']
        w_box = x2 - x1
        h_box = y2 - y1
        
        source = det['source']
        if source == 'orig': color = 'red'
        elif source == 'enh': color = 'cyan'
        elif source == 'orig_fallback': color = 'darkred'
        elif source == 'unmatched_orig': color = 'magenta'
        elif source == 'unmatched_enh': color = 'green'
        else: color = 'black'
        
        rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor=color, facecolor='none')
        axes[1, 1].add_patch(rect)
        
        label = f"{det['class_name']} {det['confidence']:.2f}"
        axes[1, 1].text(x1, y1 - 5, label, color=color, fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def save_detections_json(detections: List[Dict], output_path: str):
    """Save detections to JSON file."""
    data = []
    for det in detections:
        box_list = [float(x) for x in det.get('box').tolist()]

        det_json = {
            'class_id': int(det.get('class_id')) if det.get('class_id') is not None else None,
            'class_name': str(det.get('class_name')) if det.get('class_name') is not None else None,
            'confidence': float(det.get('confidence')) if det.get('confidence') is not None else None,
            'box_xyxy': box_list,
            'source': str(det.get('source')) if det.get('source') is not None else None,
        }
        data.append(det_json)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# -------------------- Main Function --------------------
def main(args):
    """Main evaluation pipeline."""
    # Setup device
    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    print(f"Using device: {device}")
    
    # Load YOLO model
    print(f"YOLO model: {args.model}")
    yolo = YOLO(args.model)
    yolo.to(device)
    
    # Setup dir paths
    base = Path(args.dataset_dir)
    orig_dir = base / args.split / 'images'
    labels_dir = base / args.split / 'labels'
    enh_dir = Path(args.enh)
    depth_debug_dir = enh_dir / '_debug'
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup visualization directory
    vis_output_dir = None
    if args.visualize:
        vis_output_dir = Path(args.vis_output)
        vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(str(orig_dir / '**' / pattern), recursive=True))
    
    if not image_files:
        print(f"No images found in {orig_dir}")
        return
    
    image_files.sort()
    print(f"Found {len(image_files)} images in {orig_dir}")
    
    # Aggregate metrics across all images
    all_metrics = {
        'orig': [],
        'enh': [],
        'fused': [],
    }
    
    all_fusion_stats = {
        'total_images': len(image_files),
        'total_pairs': 0,
        'chosen_orig': 0,
        'chosen_enh': 0,
        'fallback_orig': 0,
        'unmatched_orig': 0,
        'unmatched_enh': 0,
    }
    
    for image_path in tqdm(image_files, desc="Processing images"):
        image_path = Path(image_path)
        stem = image_path.stem
        rel_path = image_path.relative_to(orig_dir)
        
        # Get corresponding enhanced image and label paths
        enh_image_path = enh_dir / rel_path
        label_path = labels_dir / f"{stem}.txt"
        
        if not enh_image_path.exists():
            print(f"Warning: No enhanced image for {image_path}!, skipping")
            continue
        
        # Load images
        image_orig = load_image(str(image_path))
        image_enh = load_image(str(enh_image_path))
        
        # Load depth gate
        gate_enh = None
        if depth_debug_dir.exists():
            gate_enh = load_depth_gate(depth_debug_dir, stem)
        
        # Get image dimensions for GT loading
        img_h, img_w = image_orig.shape[:2]
        
        # Load ground truth
        gt_dets = load_ground_truth_from_labels(str(label_path), img_w, img_h)
        
        # Run YOLO inference
        detections_orig = run_yolo_inference(yolo, str(image_path), args.conf_thresh, args.max_box_width_ratio)
        detections_enh = run_yolo_inference(yolo, str(enh_image_path), args.conf_thresh, args.max_box_width_ratio)
        
        # Find matching pairs and fuse
        pairs = find_matching_pairs(detections_orig, detections_enh, args.iou_thresh)
        fused, fusion_stats = fuse_detections(detections_orig, detections_enh, gate_enh, pairs, 
                                              args.margin_thresh, args.unmatched_depth_thresh)
        
        # Update fusion stats
        all_fusion_stats['total_pairs'] += fusion_stats['total_pairs']
        all_fusion_stats['chosen_orig'] += fusion_stats['chosen_orig']
        all_fusion_stats['chosen_enh'] += fusion_stats['chosen_enh']
        all_fusion_stats['fallback_orig'] += fusion_stats['fallback_orig']
        all_fusion_stats['unmatched_orig'] += fusion_stats['unmatched_orig']
        all_fusion_stats['unmatched_enh'] += fusion_stats['unmatched_enh']
        
        # Compute COCO metrics
        if len(gt_dets) > 0:  # Only compute if GT exists
            COCO_iou_thresholds = np.arange(0.5, 1.0, 0.05)
            metrics_orig = compute_coco_metrics(detections_orig, gt_dets, COCO_iou_thresholds)
            metrics_enh = compute_coco_metrics(detections_enh, gt_dets, COCO_iou_thresholds)
            metrics_fused = compute_coco_metrics(fused, gt_dets, COCO_iou_thresholds)
            
            all_metrics['orig'].append(metrics_orig)
            all_metrics['enh'].append(metrics_enh)
            all_metrics['fused'].append(metrics_fused)
        
        # Save detections as JSON
        json_orig_path = output_dir / f"{stem}_orig.json"
        json_enh_path = output_dir / f"{stem}_enh.json"
        json_fused_path = output_dir / f"{stem}_fused.json"
        
        save_detections_json(detections_orig, str(json_orig_path))
        save_detections_json(detections_enh, str(json_enh_path))
        save_detections_json(fused, str(json_fused_path))
        
        # Save visualizations if requested
        if vis_output_dir is not None:
            vis_path = vis_output_dir / f"{stem}_comparison.png"
            visualize_detections(image_orig, gt_dets, image_orig, detections_orig, 
                               image_enh, detections_enh, fused, str(vis_path))
    
    # Compute aggregate metrics
    def aggregate_metrics(metrics_list):
        if not metrics_list:
            return {'mAP_50_95': 0.0, 'mAP_50': 0.0, 'recall': 0.0, 'num_predictions': 0, 'num_gts': 0}
        
        avg_map_50_95 = np.mean([m['mAP_50_95'] for m in metrics_list])
        avg_map_50 = np.mean([m['mAP_50'] for m in metrics_list])
        avg_recall = np.mean([m['recall'] for m in metrics_list])
        total_preds = sum(m['num_predictions'] for m in metrics_list)
        total_gts = sum(m['num_gts'] for m in metrics_list)
        
        return {
            'mAP_50_95': float(avg_map_50_95),
            'mAP_50': float(avg_map_50),
            'recall': float(avg_recall),
            'num_predictions': int(total_preds),
            'num_gts': int(total_gts),
        }
    
    agg_orig = aggregate_metrics(all_metrics['orig'])
    agg_enh = aggregate_metrics(all_metrics['enh'])
    agg_fused = aggregate_metrics(all_metrics['fused'])
    
    # Save summary
    summary = {
        'images_processed': len(image_files),
        'fusion_statistics': all_fusion_stats,
        'coco_metrics': {
            'original': agg_orig,
            'enhanced': agg_enh,
            'fused': agg_fused,
        }
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print(f"\nFusion & COCO Evaluation Results:")
    print(f"Images processed: {len(image_files)}")
    print(f"\nFusion Statistics:")
    print(f"  Total matched pairs: {all_fusion_stats['total_pairs']}")
    print(f"    Chosen from original: {all_fusion_stats['chosen_orig']}")
    print(f"    Chosen from enhanced: {all_fusion_stats['chosen_enh']}")
    print(f"    Fallback to original: {all_fusion_stats['fallback_orig']}")
    print(f"  Unmatched from original: {all_fusion_stats['unmatched_orig']}")
    print(f"  Unmatched from enhanced (g_bar > threshold): {all_fusion_stats['unmatched_enh']}")
    
    print(f"\nCOCO Metrics (Average across images):")
    print(f"\nOriginal Branch:")
    print(f"  mAP@[.5:.95]: {agg_orig['mAP_50_95']:.4f}")
    print(f"  mAP@50:       {agg_orig['mAP_50']:.4f}")
    print(f"  Recall:       {agg_orig['recall']:.4f}")
    print(f"  Predictions:  {agg_orig['num_predictions']}")
    print(f"  Ground Truths: {agg_orig['num_gts']}")
    
    print(f"\nEnhanced Branch:")
    print(f"  mAP@[.5:.95]: {agg_enh['mAP_50_95']:.4f}")
    print(f"  mAP@50:       {agg_enh['mAP_50']:.4f}")
    print(f"  Recall:       {agg_enh['recall']:.4f}")
    print(f"  Predictions:  {agg_enh['num_predictions']}")
    print(f"  Ground Truths: {agg_enh['num_gts']}")
    
    print(f"\nFused Branch:")
    print(f"  mAP@[.5:.95]: {agg_fused['mAP_50_95']:.4f}")
    print(f"  mAP@50:       {agg_fused['mAP_50']:.4f}")
    print(f"  Recall:       {agg_fused['recall']:.4f}")
    print(f"  Predictions:  {agg_fused['num_predictions']}")
    print(f"  Ground Truths: {agg_fused['num_gts']}")
    
    print(f"\nComparison (Fused vs Original):")
    print(f"  Δ mAP@50: {agg_fused['mAP_50'] - agg_orig['mAP_50']:+.4f}")
    print(f"  Δ mAP@[.5:.95]: {agg_fused['mAP_50_95'] - agg_orig['mAP_50_95']:+.4f}")
    print(f"  Δ Recall: {agg_fused['recall'] - agg_orig['recall']:+.4f}")
    
    print(f"Results saved to {output_dir}")
    if vis_output_dir: print(f"Visualizations saved to {vis_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth-gated detection fusion with COCO evaluation")
    
    # Input/Output
    parser.add_argument('--enh', type=str, required=False, default='result_Zero_DCE++_depth', 
                       help='Enhanced images directory')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output directory for results (JSON + stats)')
    parser.add_argument('--vis_output', type=str, default='vis_fusion_eval',
                       help='Output directory for visualizations')

    # Dataset (BDD100K folder)
    parser.add_argument('--dataset_dir', type=str, default='bdd100k-night-v3.yolov11', 
                       help='Dataset root')
    parser.add_argument('--split', type=str, default='test', choices=['test','train','valid'], 
                       help='Dataset split to evaluate')
    
    # YOLO
    parser.add_argument('--model', type=str, default='yolo11n.pt', 
                       help='YOLO model path')
    parser.add_argument('--conf_thresh', type=float, default=0.2, 
                       help='YOLO confidence threshold')
    parser.add_argument('--max_box_width_ratio', type=float, default=0.9,
                       help='Maximum allowed box width percent of image width (filters car hoods)')
    
    # Fusion parameters
    parser.add_argument('--iou_thresh', type=float, default=0.60, 
                       help='IoU threshold for matching orig/enh detections')
    parser.add_argument('--margin_thresh', type=float, default=0.02, 
                       help='Score margin threshold for fallback')
    parser.add_argument('--unmatched_depth_thresh', type=float, default=0.4, 
                       help='minimum g_bar for unmatched enh detections')
    
    # Inference
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for YOLO inference')
    parser.add_argument('--device', type=int, default=0, 
                       help='GPU device ID (or -1 for CPU)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', 
                       help='Save comparison visualizations')
    
    args = parser.parse_args()

    main(args)
