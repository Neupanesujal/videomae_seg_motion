"""Utilities for aligning precomputed RAFT motion with VideoMAE tubelets.

These helpers are deliberately independent of the VideoMAE dataset/augmentation
implementation. Integrate them after the dataset exposes its exact sampled RGB
frame indices and after the shared RGB-motion crop is implemented.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import numpy as np


class MotionAlignmentError(RuntimeError):
    """Raised when a requested RGB temporal interval cannot be matched to RAFT."""


def load_motion_npz(npz_path: str | Path):
    """Load compact RAFT archive and validate basic fields.

    Expected keys: motion_magnitude, frame_idx, gap, fps.
    Returns magnitude (T,H,W), frame_idx (T,), gap (scalar), fps (scalar).
    """
    p = Path(npz_path)
    if not p.is_file():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        required = {"motion_magnitude", "frame_idx", "gap", "fps"}
        missing = required.difference(z.files)
        if missing:
            raise MotionAlignmentError(f"{p}: missing NPZ fields {sorted(missing)}")
        magnitude = np.asarray(z["motion_magnitude"], dtype=np.float32)
        frame_idx = np.asarray(z["frame_idx"], dtype=np.int64).reshape(-1)
        gap = int(np.asarray(z["gap"]).item())
        fps = float(np.asarray(z["fps"]).item())
    if magnitude.ndim != 3:
        raise MotionAlignmentError(f"{p}: motion_magnitude must be [T,H,W], got {magnitude.shape}")
    if len(frame_idx) != magnitude.shape[0]:
        raise MotionAlignmentError(f"{p}: frame_idx length {len(frame_idx)} != maps {magnitude.shape[0]}")
    if gap <= 0:
        raise MotionAlignmentError(f"{p}: gap must be positive, got {gap}")
    if not np.isfinite(magnitude).all():
        bad = int((~np.isfinite(magnitude)).sum())
        raise MotionAlignmentError(f"{p}: motion contains {bad} NaN/Inf values")
    if np.any(frame_idx < 0):
        raise MotionAlignmentError(f"{p}: frame_idx contains negative values")
    if len(np.unique(frame_idx)) != len(frame_idx):
        raise MotionAlignmentError(f"{p}: duplicate frame_idx values")
    return magnitude, frame_idx, gap, fps


def match_tubelet_motion(npz_path: str | Path,
                         sampled_rgb_frame_ids: Iterable[int],
                         tubelet_size: int = 2,
                         expected_raft_gap: Optional[int] = None,
                         missing_policy: str = "zero"):
    """Return 1 RAFT map per temporal tubelet based on actual sampled RGB IDs.

    For VideoMAE's 16-frame, tubelet-size-2 clip, this returns 8 maps. The map
    is looked up using the first sampled RGB frame in each tubelet, not by
    assuming a clip offset or slicing the NPZ array.

    missing_policy: 'zero' (fill missing/repeated intervals with zero map) or
    'error'. Zero fallback is explicit and returned in the diagnostics.
    """
    sampled = np.asarray(list(sampled_rgb_frame_ids), dtype=np.int64)
    if sampled.ndim != 1 or len(sampled) == 0 or len(sampled) % tubelet_size:
        raise ValueError(f"Expected a 1-D frame list divisible by tubelet_size={tubelet_size}; got {sampled.shape}")
    if tubelet_size < 1:
        raise ValueError("tubelet_size must be >= 1")
    magnitude, frame_idx, gap, fps = load_motion_npz(npz_path)
    if expected_raft_gap is not None and gap != int(expected_raft_gap):
        raise MotionAlignmentError(f"NPZ gap={gap}, expected {expected_raft_gap}")

    lookup = {int(fi): i for i, fi in enumerate(frame_idx)}
    map_starts = sampled[::tubelet_size]
    selected, diagnostics = [], []
    zero_map = np.zeros(magnitude.shape[1:], dtype=np.float32)
    for t, start in enumerate(map_starts):
        # Repeated RGB indices (near video boundaries) do not represent motion.
        repeated = len(np.unique(sampled[t*tubelet_size:(t+1)*tubelet_size])) < tubelet_size
        idx = lookup.get(int(start))
        if repeated or idx is None:
            reason = "repeated_rgb_frames" if repeated else "missing_raft_start_index"
            if missing_policy == "error":
                raise MotionAlignmentError(f"tubelet {t}: {reason}; start frame={int(start)}")
            if missing_policy != "zero":
                raise ValueError("missing_policy must be 'zero' or 'error'")
            selected.append(zero_map.copy())
            diagnostics.append({"tubelet": t, "start_frame": int(start), "npz_map_index": None,
                              "status": "zero_fallback", "reason": reason})
            continue
        # Ensure the two RGB frames in this tubelet correspond to the RAFT gap.
        rgb_interval = int(sampled[t*tubelet_size + 1] - start)
        if rgb_interval != gap:
            msg = f"RGB tubelet interval {rgb_interval} != RAFT gap {gap}"
            if missing_policy == "error":
                raise MotionAlignmentError(f"tubelet {t}, start={int(start)}: {msg}")
            selected.append(zero_map.copy())
            diagnostics.append({"tubelet": t, "start_frame": int(start), "npz_map_index": int(idx),
                              "status": "zero_fallback", "reason": msg})
            continue
        selected.append(magnitude[idx])
        diagnostics.append({"tubelet": t, "start_frame": int(start), "npz_map_index": int(idx),
                          "status": "matched", "reason": None})
    return np.stack(selected, axis=0), diagnostics, {"gap": gap, "fps": fps, "npz_shape": list(magnitude.shape)}


def normalize_motion(motion: np.ndarray, method: str = "clip_minmax",
                    percentile: float = 99.0, eps: float = 1e-6) -> np.ndarray:
    """Normalize aligned motion scores to [0,1], handling static maps safely.

    clip_minmax: clip at the chosen percentile, then min-max scale globally
    across the supplied clip (all tubelets/spatial positions).
    minmax: ordinary global min-max.
    """
    x = np.asarray(motion, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Motion array contains NaN or Inf")
    x = np.maximum(x, 0)
    if method == "clip_minmax":
        if not (0 < percentile <= 100):
            raise ValueError("percentile must be in (0,100]")
        hi = float(np.percentile(x, percentile))
        x = np.clip(x, 0, hi)
    elif method == "minmax":
        pass
    else:
        raise ValueError("method must be 'clip_minmax' or 'minmax'")
    lo, hi = float(x.min()), float(x.max())
    if hi - lo <= eps:
        return np.zeros_like(x, dtype=np.float32)
    out = (x - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)


def spatial_motion_scores(motion_8x14x14: np.ndarray) -> np.ndarray:
    """Temporal mean used by tube masking; returns [14,14] scores."""
    x = np.asarray(motion_8x14x14, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got {x.shape}")
    return x.mean(axis=0)


def make_motion_guided_tube_mask(motion_8x14x14: np.ndarray, mask_ratio: float = 0.75,
                                beta: float = 2.0, rng=None) -> np.ndarray:
    """Sample one spatial mask biased toward higher motion, repeat over time.

    Returns flattened boolean mask in VideoMAE token order [T,H,W]. Exactly
    int(mask_ratio * H*W) locations are masked in each tube (shared spatially).
    A beta of 0 gives uniform sampling. Uses weighted sampling without
    replacement via numpy's Generator.choice.
    """
    x = np.asarray(motion_8x14x14, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got {x.shape}")
    T, H, W = x.shape
    if not 0 <= mask_ratio < 1:
        raise ValueError("mask_ratio must be in [0,1)")
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    n_mask = int(mask_ratio * H * W)
    scores = x.mean(axis=0).reshape(-1)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.clip(scores, 0, 1)
    if rng is None:
        rng = np.random.default_rng()
    if beta == 0 or scores.max() - scores.min() <= 1e-8:
        probs = np.full(scores.size, 1.0 / scores.size, dtype=np.float64)
    else:
        logits = beta * (scores - scores.max())
        weights = np.exp(logits)
        probs = weights / weights.sum()
    masked_idx = rng.choice(scores.size, size=n_mask, replace=False, p=probs)
    spatial_mask = np.zeros(scores.size, dtype=bool)
    spatial_mask[masked_idx] = True
    return np.tile(spatial_mask.reshape(H, W), (T, 1, 1)).reshape(-1)


def motion_weighted_mse(pred: "torch.Tensor", target: "torch.Tensor",
                        masked_motion: "torch.Tensor", alpha: float = 1.0):
    """Weighted token MSE; tensors expected [B,N,C] and motion [B,N].

    This helper imports torch only when called. With alpha=0 it matches the
    ordinary mean MSE over all elements, up to floating point roundoff.
    """
    import torch
    if pred.shape != target.shape:
        raise ValueError(f"pred {pred.shape} != target {target.shape}")
    if masked_motion.shape != pred.shape[:2]:
        raise ValueError(f"motion shape {masked_motion.shape} must match [B,N]={pred.shape[:2]}")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    per_token = (pred.float() - target.float()).pow(2).mean(dim=-1)
    weights = 1.0 + alpha * masked_motion.float().clamp(0, 1)
    return (per_token * weights).sum() / weights.sum().clamp_min(1e-8)
