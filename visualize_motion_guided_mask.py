#!/usr/bin/env python3

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from test_motion_pretraining_batch import (
    build_production_args,
)

from datasets import build_pretraining_dataset

from motion_utils import (
    spatial_motion_scores,
    make_motion_guided_tube_mask,
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_INDEX = 223
MASK_RATIO = 0.75
BETA = 2.0
NUM_TRIALS = 1000

OUTPUT_DIR = "outputs/motion_masking/motion_mask_visualization_beta2_sample223"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)


# ============================================================
# HELPERS
# ============================================================

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def make_random_tube_mask(T, H, W, mask_ratio, rng):
    """Uniform random tube mask, shared across temporal tubelets."""
    n_mask = int(mask_ratio * H * W)

    idx = rng.choice(
        H * W,
        size=n_mask,
        replace=False,
    )

    spatial_mask = np.zeros(H * W, dtype=bool)
    spatial_mask[idx] = True

    return np.tile(
        spatial_mask.reshape(H, W),
        (T, 1, 1),
    )


def denormalize_video(video):
    """
    Video: [C,T,H,W]

    VideoMAE GroupNormalize typically uses mean=0.5,
    std=0.5 for each RGB channel.
    """
    video = video.astype(np.float32)

    video = video * 0.5 + 0.5

    video = np.clip(video, 0, 1)

    return video


def overlay_mask(frame, mask_14x14):
    """
    Resize patch mask to frame resolution and overlay
    masked regions in red.
    """
    H, W, _ = frame.shape

    mask_large = np.repeat(
        np.repeat(mask_14x14, H // 14, axis=0),
        W // 14,
        axis=1,
    )

    mask_large = mask_large[:H, :W]

    result = frame.copy()

    red = np.zeros_like(result)
    red[..., 0] = 1.0

    result[mask_large] = (
        0.45 * result[mask_large]
        + 0.55 * red[mask_large]
    )

    return result


def summarize_scores(name, motion, mask):
    masked_motion = motion[mask]
    visible_motion = motion[~mask]

    print(f"\n{name}")
    print("-" * 50)
    print(f"Mean motion, all patches     : {motion.mean():.6f}")
    print(f"Mean motion, masked patches  : {masked_motion.mean():.6f}")
    print(f"Mean motion, visible patches : {visible_motion.mean():.6f}")
    print(f"Masked patches               : {mask.sum()}")
    print(f"Visible patches              : {(~mask).sum()}")


# ============================================================
# LOAD SAMPLE
# ============================================================

print("Loading production dataset...")

args = build_production_args()

# Enable motion-guided masking for dataset initialization
args.motion_guided_mask = True
args.motion_weighted_loss = False
args.motion_alpha = 1.0
args.motion_beta = BETA

dataset = build_pretraining_dataset(args)

video, dataset_mask, motion = dataset[SAMPLE_INDEX]

video = to_numpy(video)
motion = to_numpy(motion)

print("\nSample information")
print("=" * 60)
print("Video shape :", video.shape)
print("Mask shape  :", to_numpy(dataset_mask).shape)
print("Motion shape:", motion.shape)

# Expected video: [3,16,224,224]
# Expected motion: [8,14,14]

if video.shape != (3, 16, 224, 224):
    raise RuntimeError(f"Unexpected video shape: {video.shape}")

if motion.shape != (8, 14, 14):
    raise RuntimeError(f"Unexpected motion shape: {motion.shape}")


# ============================================================
# PREPARE VIDEO AND MOTION
# ============================================================

video = denormalize_video(video)

# Temporal mean motion for tube masking
spatial_motion = spatial_motion_scores(motion)

T = motion.shape[0]
H = motion.shape[1]
W = motion.shape[2]

n_mask = int(MASK_RATIO * H * W)

print("\nMotion statistics")
print("=" * 60)
print(f"Motion minimum : {motion.min():.6f}")
print(f"Motion maximum : {motion.max():.6f}")
print(f"Motion mean    : {motion.mean():.6f}")
print(f"Spatial mean   : {spatial_motion.mean():.6f}")
print(f"Mask ratio     : {MASK_RATIO}")
print(f"Masked patches : {n_mask}/{H*W}")
print(f"Beta           : {BETA}")


# ============================================================
# GENERATE MASKS
# ============================================================

# Random uniform mask
random_mask = make_random_tube_mask(
    T, H, W,
    MASK_RATIO,
    rng,
)

# Actual implemented motion-guided mask
guided_mask_flat = make_motion_guided_tube_mask(
    motion,
    mask_ratio=MASK_RATIO,
    beta=BETA,
    rng=rng,
)

guided_mask = guided_mask_flat.reshape(T, H, W)

random_spatial_mask = random_mask[0]
guided_spatial_mask = guided_mask[0]


# ============================================================
# QUANTITATIVE COMPARISON
# ============================================================

print("\nSingle-sample mask comparison")
print("=" * 60)

summarize_scores(
    "RANDOM MASK",
    spatial_motion,
    random_spatial_mask,
)

summarize_scores(
    "MOTION-GUIDED MASK",
    spatial_motion,
    guided_spatial_mask,
)


# ============================================================
# MONTE CARLO: MASKING FREQUENCY
# ============================================================

print(f"\nRunning {NUM_TRIALS} masking trials...")
print("This tests the statistical behavior of the masking strategy.")

guided_frequency = np.zeros(
    (H, W),
    dtype=np.float32,
)

random_frequency = np.zeros(
    (H, W),
    dtype=np.float32,
)

for trial in range(NUM_TRIALS):

    guided = make_motion_guided_tube_mask(
        motion,
        mask_ratio=MASK_RATIO,
        beta=BETA,
        rng=rng,
    ).reshape(T, H, W)[0]

    random = make_random_tube_mask(
        T, H, W,
        MASK_RATIO,
        rng,
    )[0]

    guided_frequency += guided
    random_frequency += random


guided_frequency /= NUM_TRIALS
random_frequency /= NUM_TRIALS

# Correlation between patch motion and masking frequency
guided_corr = np.corrcoef(
    spatial_motion.flatten(),
    guided_frequency.flatten(),
)[0, 1]

random_corr = np.corrcoef(
    spatial_motion.flatten(),
    random_frequency.flatten(),
)[0, 1]

print("\nMonte Carlo results")
print("=" * 60)
print(f"Motion-guided mask frequency correlation: {guided_corr:.4f}")
print(f"Random mask frequency correlation      : {random_corr:.4f}")

print(
    "Mean mask frequency of high-motion patches "
    "(top 25%, guided):",
    guided_frequency[
        spatial_motion >= np.quantile(spatial_motion, 0.75)
    ].mean(),
)

print(
    "Mean mask frequency of low-motion patches "
    "(bottom 25%, guided):",
    guided_frequency[
        spatial_motion <= np.quantile(spatial_motion, 0.25)
    ].mean(),
)


# ============================================================
# VISUALIZATION 1: PATCH-LEVEL MAPS
# ============================================================

fig, axes = plt.subplots(
    1, 5,
    figsize=(22, 5),
)

images = [
    spatial_motion,
    random_spatial_mask.astype(float),
    guided_spatial_mask.astype(float),
    random_frequency,
    guided_frequency,
]

titles = [
    "Mean motion score",
    "Random mask",
    "Motion-guided mask",
    "Random mask frequency",
    "Motion-guided frequency",
]

for ax, img, title in zip(axes, images, titles):

    im = ax.imshow(img, cmap="viridis")

    ax.set_title(title)
    ax.set_xlabel("Patch X")
    ax.set_ylabel("Patch Y")

    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()

path1 = os.path.join(
    OUTPUT_DIR,
    "patch_mask_comparison.png",
)

plt.savefig(path1, dpi=200)
plt.close()

print(f"\nSaved: {path1}")


# ============================================================
# VISUALIZATION 2: VIDEO FRAMES AND MASK OVERLAYS
# ============================================================

# Select frames from different temporal positions
frame_indices = [0, 4, 8, 12]

fig, axes = plt.subplots(
    4, 4,
    figsize=(16, 14),
)

for col, frame_idx in enumerate(frame_indices):

    frame = video[:, frame_idx].transpose(1, 2, 0)

    tubelet_idx = frame_idx // 2

    frame_motion = motion[tubelet_idx]

    random_overlay = overlay_mask(
        frame,
        random_spatial_mask,
    )

    guided_overlay = overlay_mask(
        frame,
        guided_spatial_mask,
    )

    axes[0, col].imshow(frame)
    axes[0, col].set_title(
        f"Original frame {frame_idx}"
    )

    axes[1, col].imshow(
        frame_motion,
        cmap="inferno",
    )
    axes[1, col].set_title(
        f"Motion map | tubelet {tubelet_idx}"
    )

    axes[2, col].imshow(random_overlay)
    axes[2, col].set_title(
        "Random masked regions"
    )

    axes[3, col].imshow(guided_overlay)
    axes[3, col].set_title(
        "Motion-guided masked regions"
    )

    for row in range(4):
        axes[row, col].axis("off")


axes[0, 0].set_ylabel("Original", fontsize=14)
axes[1, 0].set_ylabel("Motion", fontsize=14)
axes[2, 0].set_ylabel("Random mask", fontsize=14)
axes[3, 0].set_ylabel("Guided mask", fontsize=14)

plt.tight_layout()

path2 = os.path.join(
    OUTPUT_DIR,
    "video_mask_overlay_comparison.png",
)

plt.savefig(path2, dpi=200)
plt.close()

print(f"Saved: {path2}")


# ============================================================
# VISUALIZATION 3: MOTION VS MASKING FREQUENCY
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5),
)

axes[0].scatter(
    spatial_motion.flatten(),
    random_frequency.flatten(),
    alpha=0.6,
)

axes[0].set_title(
    "Random masking"
)

axes[0].set_xlabel(
    "Motion score"
)

axes[0].set_ylabel(
    "Masking frequency"
)

axes[0].grid(True)


axes[1].scatter(
    spatial_motion.flatten(),
    guided_frequency.flatten(),
    alpha=0.6,
)

axes[1].set_title(
    "Motion-guided masking"
)

axes[1].set_xlabel(
    "Motion score"
)

axes[1].set_ylabel(
    "Masking frequency"
)

axes[1].grid(True)


plt.tight_layout()

path3 = os.path.join(
    OUTPUT_DIR,
    "motion_vs_masking_frequency.png",
)

plt.savefig(path3, dpi=200)
plt.close()

print(f"Saved: {path3}")


# ============================================================
# SAVE NUMERICAL DATA FOR FUTURE ANALYSIS
# ============================================================

np.savez_compressed(
    os.path.join(
        OUTPUT_DIR,
        "motion_mask_diagnostics.npz",
    ),

    motion_scores=spatial_motion,

    random_mask=random_spatial_mask,

    guided_mask=guided_spatial_mask,

    random_mask_frequency=random_frequency,

    guided_mask_frequency=guided_frequency,

    beta=BETA,

    mask_ratio=MASK_RATIO,

    num_trials=NUM_TRIALS,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

print(f"Output directory: {OUTPUT_DIR}")

print("\nCorrelation summary:")
print(f"Random mask      : {random_corr:.4f}")
print(f"Motion-guided    : {guided_corr:.4f}")

print("\nGenerated files:")
print("1. patch_mask_comparison.png")
print("2. video_mask_overlay_comparison.png")
print("3. motion_vs_masking_frequency.png")
print("4. motion_mask_diagnostics.npz")