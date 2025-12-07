# -*- coding: utf-8 -*-
import os
import glob
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

random.seed(1143)


def _list_images(root: str, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
    root = os.path.abspath(root)
    paths = []
    for ext in exts:
        # Support recursive subdirectories
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    # Deduplicate + Sort
    paths = sorted(list({os.path.abspath(p) for p in paths}))
    return paths


def populate_train_list(lowlight_images_path: str) -> List[str]:
    return _list_images(lowlight_images_path)


def _pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor [3,H,W] in [0,1]."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC RGB
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _resize_np01(arr: np.ndarray, size_hw: Tuple[int, int], mode=Image.BILINEAR) -> np.ndarray:
    """Resize a (H,W) or (H,W,*) float32 array in [0,1] with PIL backend."""
    h, w = size_hw
    if arr.ndim == 2:
        im = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8), mode="L")
        im = im.resize((w, h), mode)
        out = np.asarray(im).astype(np.float32) / 255.0
        return out
    elif arr.ndim == 3:
        # treat last dim as channels (e.g., RGB)
        im = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))
        im = im.resize((w, h), mode)
        out = np.asarray(im).astype(np.float32) / 255.0
        return out
    else:
        raise ValueError(f"Unsupported array shape for resize: {arr.shape}")


def _load_depth_candidates(depth_dir: str, stem: str) -> Optional[np.ndarray]:
    """
    Load depth: priority *_Dnorm.npy -> *_Dnorm.png -> *_disp.npy -> *_disp.png
    Output range normalized to [0,1]; Convention: D=1 Near, D=0 Far.
    """
    if depth_dir is None:
        return None
    cands = [
        os.path.join(depth_dir, f"{stem}_Dnorm.npy"),
        os.path.join(depth_dir, f"{stem}_Dnorm.png"),
        os.path.join(depth_dir, f"{stem}_disp.npy"),
        os.path.join(depth_dir, f"{stem}_disp.png"),
    ]
    for p in cands:
        if os.path.exists(p):
            if p.endswith(".npy"):
                d = np.load(p).astype(np.float32)
                if "Dnorm" in os.path.basename(p):
                    d = np.clip(d, 0.0, 1.0)
                else:
                    # disp normalization
                    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
                    d = (d - lo) / (hi - lo + 1e-6)
                    d = np.clip(d, 0.0, 1.0)
                return d
            else:
                im = Image.open(p)
                if im.mode != "L":
                    im = im.convert("L")
                d = np.asarray(im).astype(np.float32) / 255.0
                if "Dnorm" not in os.path.basename(p):
                    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
                    d = (d - lo) / (hi - lo + 1e-6)
                    d = np.clip(d, 0.0, 1.0)
                return d
    return None


def _load_conf(depth_dir: str, stem: str) -> Optional[np.ndarray]:
    if depth_dir is None:
        return None
    p = os.path.join(depth_dir, f"{stem}_conf.npy")
    if os.path.exists(p):
        c = np.load(p).astype(np.float32)
        c = np.clip(c, 0.0, 1.0)
        return c
    return None


def _load_teacher(teacher_dir: str, stem: str, size_hw: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load teacher image (RGB [0,1]) and teacher gate g_far ([0,1], single channel)
    """
    if teacher_dir is None:
        return None, None

    teach_img_path = os.path.join(teacher_dir, f"{stem}.png")
    teach_gate_path = os.path.join(teacher_dir, "_debug", f"{stem}_g_far.png")

    t_img = None
    t_gate = None

    if os.path.exists(teach_img_path):
        im = Image.open(teach_img_path).convert("RGB")
        im = im.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
        t_img = (np.asarray(im).astype(np.float32) / 255.0)  # HWC RGB in [0,1]

    if os.path.exists(teach_gate_path):
        im = Image.open(teach_gate_path).convert("L")
        im = im.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
        t_gate = (np.asarray(im).astype(np.float32) / 255.0)  # HW in [0,1]

    return t_img, t_gate


class lowlight_loader(data.Dataset):
    """
    Training Dataset:
      - Load low-light image -> [3,H,W] in [0,1]
      - Optionally load depth/confidence (from midas_depth)
      - Optionally load teacher image/gate (from teacher_depthgate_balanced)
    Returns dictionary:
      {
        'image': [3,H,W] float32,
        'depth': [1,H,W] float32 in [0,1] (near=1, far=0) or None (key not returned if missing)
        'conf':  [1,H,W] float32 in [0,1] (optional)
        'teacher_img':  [3,H,W] (optional)
        'teacher_gate': [1,H,W] (optional)
        'stem': str,
        'path': str
      }
    """

    def __init__(
        self,
        lowlight_images_path: str,
        size: int = 512,
        depth_dir: str = "midas_depth",
        teacher_dir: Optional[str] = None,
    ):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = int(size)
        self.depth_dir = depth_dir
        self.teacher_dir = teacher_dir
        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_path = self.data_list[index]
        stem = os.path.splitext(os.path.basename(img_path))[0]

        # Load & Unify size (RGB)
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((self.size, self.size), Image.LANCZOS)
        img_t = _pil_to_tensor01(img)  # [3,H,W]

        H, W = self.size, self.size

        # Depth / Confidence
        out: Dict[str, torch.Tensor] = {
            "image": img_t,
            "stem": stem,
            "path": img_path,
        }

        D = _load_depth_candidates(self.depth_dir, stem) if self.depth_dir else None
        if D is not None:
            if D.shape != (H, W):
                D = _resize_np01(D, (H, W), mode=Image.BILINEAR)
            out["depth"] = torch.from_numpy(D).unsqueeze(0).float()  # [1,H,W]

        C = _load_conf(self.depth_dir, stem) if self.depth_dir else None
        if C is not None:
            if C.shape != (H, W):
                C = _resize_np01(C, (H, W), mode=Image.BILINEAR)
            out["conf"] = torch.from_numpy(C).unsqueeze(0).float()    # [1,H,W]

        # Teacher image/gate (optional)
        if self.teacher_dir is not None:
            t_img, t_gate = _load_teacher(self.teacher_dir, stem, (H, W))
            if t_img is not None:
                out["teacher_img"] = torch.from_numpy(t_img).permute(2, 0, 1).float()  # [3,H,W]
            if t_gate is not None:
                out["teacher_gate"] = torch.from_numpy(t_gate).unsqueeze(0).float()    # [1,H,W]

        return out