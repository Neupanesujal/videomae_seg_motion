#!/usr/bin/env python3

import sys
import gc
import copy
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader, Subset


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

VIDEO_LIST = (
    PROJECT_ROOT
    / "datasets/ucfTrainTestlist/trainlist01.txt"
)

MOTION_ROOT = (
    PROJECT_ROOT
    / "raft/output_datasets_motion/motion_npz"
)

PRETRAINED_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/pretrained/"
    / "videomae_v1_base_kinetics_e1600.pth"
)

SAMPLE_INDEX = 0

BATCH_SIZE = 1

DEVICE = "cuda"


# ============================================================
# EXPERIMENT MODES
# ============================================================

EXPERIMENTS = {
    "B1": {
        "motion_guided_mask": False,
        "motion_weighted_loss": True,
        "alpha": 1.0,
        "beta": 2.0,
    },

    "B2": {
        "motion_guided_mask": True,
        "motion_weighted_loss": False,
        "alpha": 1.0,
        "beta": 2.0,
    },

    "B3": {
        "motion_guided_mask": True,
        "motion_weighted_loss": True,
        "alpha": 1.0,
        "beta": 2.0,
    },
}


# ============================================================
# UTILITIES
# ============================================================

def check_finite_tensor(name, tensor):

    if not torch.isfinite(tensor).all():

        raise RuntimeError(
            f"{name} contains NaN or Inf values"
        )


def check_mask(mask):

    # Convert to NumPy to support dataset masks
    # regardless of whether they are NumPy or Torch.
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = np.asarray(mask)

    mask = mask.reshape(8, 14, 14)

    masked_per_tubelet = mask.sum(axis=(1, 2))

    print(
        "Masked spatial tokens per tubelet:",
        masked_per_tubelet.tolist()
    )

    # 75% of 196 spatial tokens = 147
    expected_masked = 147

    assert np.all(
        masked_per_tubelet == expected_masked
    ), (
        f"Unexpected masked token count: "
        f"{masked_per_tubelet}"
    )

    # Tube masking: same spatial mask across all 8 temporal tubelets
    for t in range(1, 8):

        assert np.array_equal(
            mask[0],
            mask[t]
        ), (
            "Mask is not temporally consistent"
        )

    print(
        "Mask validation passed: "
        "147/196 masked spatial tokens per tubelet, "
        "temporally consistent."
    )


# ============================================================
# PRODUCTION ARGUMENTS
# ============================================================

def build_production_args():

    import run_mae_pretraining

    # Parse the original VideoMAE arguments
    sys.argv = [
        "run_mae_pretraining.py"
    ]

    args = (
        run_mae_pretraining.get_args()
    )

    # --------------------------------------------------------
    # Dataset configuration
    # --------------------------------------------------------

    args.data_path = str(VIDEO_LIST)

    args.batch_size = BATCH_SIZE

    args.num_workers = 0

    args.pin_mem = False

    args.device = DEVICE

    args.num_frames = 16

    args.sampling_rate = 4

    args.input_size = 224

    args.mask_type = "tube"

    args.mask_ratio = 0.75

    args.window_size = (
        8,
        14,
        14,
    )

    args.patch_size = (
        16,
        16,
    )

    # --------------------------------------------------------
    # Checkpoint configuration
    # --------------------------------------------------------

    args.pretrained = str(
        PRETRAINED_CHECKPOINT
    )

    args.resume = ""

    args.auto_resume = False

    args.output_dir = ""

    args.log_dir = ""

    args.epochs = 1

    args.start_epoch = 0

    # --------------------------------------------------------
    # Optimizer configuration
    # --------------------------------------------------------

    args.lr = 3e-4

    args.min_lr = 1e-5

    args.weight_decay = 0.05

    args.weight_decay_end = 0.05

    args.warmup_epochs = 0

    args.warmup_steps = 0

    args.clip_grad = 0

    # --------------------------------------------------------
    # Motion configuration
    # --------------------------------------------------------

    args.motion_enabled = True

    args.motion_root = str(
        MOTION_ROOT
    )

    args.motion_gap = 4

    args.motion_missing_policy = "zero"

    args.motion_percentile = 99.0

    args.motion_alpha = 1.0

    args.motion_beta = 2.0

    # No distributed training
    args.distributed = False

    return args


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(name, config):

    print("\n")
    print("=" * 70)

    print(
        f"TESTING {name}"
    )

    print("=" * 70)

    import run_mae_pretraining

    from datasets import (
        build_pretraining_dataset
    )

    from optim_factory import (
        create_optimizer
    )

    from engine_for_pretraining import (
        train_one_epoch
    )

    from utils import (
        NativeScalerWithGradNormCount
    )

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    args = build_production_args()

    args.motion_guided_mask = (
        config["motion_guided_mask"]
    )

    args.motion_weighted_loss = (
        config["motion_weighted_loss"]
    )

    args.motion_alpha = (
        config["alpha"]
    )

    args.motion_beta = (
        config["beta"]
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = (
        build_pretraining_dataset(args)
    )

    if SAMPLE_INDEX >= len(dataset):

        raise IndexError(
            f"Sample index {SAMPLE_INDEX} "
            f"out of range for dataset "
            f"length {len(dataset)}"
        )

    # Use ONLY one video
    test_dataset = Subset(
        dataset,
        [SAMPLE_INDEX]
    )

    data_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    # --------------------------------------------------------
    # Inspect the real sample
    # --------------------------------------------------------

    sample = dataset[SAMPLE_INDEX]

    if len(sample) != 3:

        raise RuntimeError(
            "Expected dataset output "
            "(videos, mask, motion_maps)"
        )

    videos, mask, motion_maps = sample

    print(
        "\nSample shapes:"
    )

    print(
        "Video:",
        tuple(videos.shape)
    )

    print(
        "Mask:",
        tuple(mask.shape)
    )

    print(
        "Motion:",
        tuple(motion_maps.shape)
    )

    check_mask(mask)

    check_finite_tensor(
        "Motion maps",
        motion_maps
    )

    if motion_maps.min() < 0:

        raise RuntimeError(
            "Motion values below zero"
        )

    if motion_maps.max() > 1:

        raise RuntimeError(
            "Motion values above one"
        )

    print(
        "Motion score range:",
        motion_maps.min().item(),
        motion_maps.max().item()
    )

    print(
        "Motion score mean:",
        motion_maps.mean().item()
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = (
        run_mae_pretraining.get_model(args)
    )

    # Load original pretrained checkpoint
    checkpoint = torch.load(
        args.pretrained,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    del checkpoint

    model.to(DEVICE)

    print(
        "\nPretrained model loaded successfully."
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = create_optimizer(
        args,
        model,
    )

    loss_scaler = (
        NativeScalerWithGradNormCount()
    )

    # --------------------------------------------------------
    # Snapshot selected representative parameters
    # (CPU copies avoid extra GPU memory consumption)
    # --------------------------------------------------------

    selected_names = [
        "mask_token",
        "encoder.patch_embed.proj.weight",
        "encoder.blocks.0.attn.qkv.weight",
        "decoder.blocks.0.attn.qkv.weight",
        "decoder.head.weight",
    ]

    named_params = dict(
        model.named_parameters()
    )

    snapshots = {}

    for name in selected_names:

        if name not in named_params:

            print(
                f"Warning: parameter not found: {name}"
            )

            continue

        p = named_params[name]

        snapshots[name] = (
            p.detach().cpu().clone()
        )

    scale_before = (
        loss_scaler.state_dict()["scale"]
    )

    parameter = next(
        p for p in model.parameters()
        if p.requires_grad
    )

    parameter_before = (
        parameter.detach()
        .clone()
    )

    # --------------------------------------------------------
    # Run production training engine
    # --------------------------------------------------------

    print(
        "\nStarting one production training step..."
    )

    stats = train_one_epoch(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        device=torch.device(DEVICE),
        epoch=0,
        loss_scaler=loss_scaler,
        max_norm=args.clip_grad,
        patch_size=16,
        normlize_target=args.normlize_target,
        log_writer=None,
        start_steps=0,
        lr_schedule_values=None,
        wd_schedule_values=None,
        args=args,
    )

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Verify gradients and optimizer update
    # --------------------------------------------------------

    scale_after = (
        loss_scaler.state_dict()["scale"]
    )

    print(
        "\nAMP scale before:",
        scale_before
    )

    print(
        "AMP scale after:",
        scale_after
    )

    # --------------------------------------------------------
    # Gradient statistics
    # --------------------------------------------------------

    grads = [
        p.grad
        for p in model.parameters()
        if p.grad is not None
    ]

    num_grad_params = len(grads)

    num_nonzero_grads = sum(
        torch.count_nonzero(g).item() > 0
        for g in grads
    )

    all_grads_finite = all(
        torch.isfinite(g).all().item()
        for g in grads
    )

    print(
        "Parameters with gradients:",
        num_grad_params
    )

    print(
        "Parameters with nonzero gradients:",
        num_nonzero_grads
    )

    print(
        "All gradients finite:",
        all_grads_finite
    )

    # --------------------------------------------------------
    # Selected parameter changes
    # --------------------------------------------------------

    changed_params = []

    for name, before in snapshots.items():

        after = (
            named_params[name]
            .detach()
            .cpu()
        )

        delta = (
            after - before
        ).abs().max().item()

        print(
            f"{name}: max_delta={delta:.12e}"
        )

        if delta > 0:

            changed_params.append(name)

    print(
        "\nChanged representative parameters:",
        changed_params
    )

    # --------------------------------------------------------
    # Optimizer state / step verification
    # --------------------------------------------------------

    optimizer_state_entries = len(
        optimizer.state
    )

    adam_steps = []

    for state in optimizer.state.values():

        if "step" in state:

            step = state["step"]

            if torch.is_tensor(step):

                step = step.item()

            adam_steps.append(step)

    print(
        "Optimizer state entries:",
        optimizer_state_entries
    )

    if adam_steps:

        print(
            "Adam step values:",
            sorted(set(adam_steps))
        )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    if not all_grads_finite:

        raise RuntimeError(
            "Non-finite gradients detected"
        )

    if num_nonzero_grads == 0:

        raise RuntimeError(
            "All gradients are zero"
        )

    if not changed_params:

        raise RuntimeError(
            "No representative parameters changed"
        )

    if scale_after < scale_before:

        raise RuntimeError(
            "AMP scaler decreased; likely skipped optimizer step"
        )

    if optimizer_state_entries == 0:

        raise RuntimeError(
            "Optimizer state was not created"
        )

    print(
        "\nOptimizer update verification passed."
    )

    parameter_after = (
        parameter.detach()
    )

    parameter_delta = (
        parameter_after - parameter_before
    ).abs().max().item()

    print(
        "\nMaximum parameter change:",
        parameter_delta
    )

    print(
        "Training statistics:",
        stats
    )

    if parameter_delta <= 0:

        raise RuntimeError(
            "Optimizer did not update parameters"
        )

    if not np.isfinite(
        stats["loss"]
    ):

        raise RuntimeError(
            "Training loss is not finite"
        )

    print(
        f"\n{name} PASSED"
    )

    print(
        "Forward pass: OK"
    )

    print(
        "Loss: OK"
    )

    print(
        "Backward pass: OK"
    )

    print(
        "Optimizer update: OK"
    )

    print(
        "Finite loss: OK"
    )

    print(
        "Parameter change: OK"
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del model
    del optimizer
    del loss_scaler
    del dataset
    del test_dataset
    del data_loader
    del sample
    del videos
    del mask
    del motion_maps

    gc.collect()

    torch.cuda.empty_cache()

    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required for this test"
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "Sample index:",
        SAMPLE_INDEX
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    results = {}

    for name, config in (
        EXPERIMENTS.items()
    ):

        results[name] = (
            run_experiment(
                name,
                config,
            )
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "ALL MOTION INTEGRATION TESTS PASSED"
    )

    print("=" * 70)

    for name, stats in results.items():

        print(
            f"{name}: "
            f"loss={stats['loss']:.6f}"
        )


if __name__ == "__main__":

    main()