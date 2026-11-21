#!/usr/bin/env python3

"""
Multi-batch stability test for motion-aware VideoMAE pretraining.

B1: Random tube masking + motion-weighted reconstruction loss
B2: Motion-guided tube masking + uniform reconstruction loss
B3: Motion-guided tube masking + motion-weighted reconstruction loss

Uses:
    - Real UCF video dataset
    - Offline RAFT motion NPZ maps
    - Original pretrained VideoMAE checkpoint
    - Production model
    - Production optimizer
    - Production AMP scaler
    - Production train_one_epoch()

This is a stability/integration test.
It is NOT a performance evaluation.
"""

import gc
import argparse
import random

import numpy as np
import torch

from torch.utils.data import DataLoader, Subset


# ============================================================
# EXPERIMENT CONFIGURATIONS
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
# SEED
# ============================================================

SEED = 42


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_finite_tensor(name, tensor):

    if not torch.isfinite(tensor).all():

        raise RuntimeError(
            f"Non-finite values detected in {name}"
        )


def get_optimizer_steps(optimizer):

    steps = []

    for state in optimizer.state.values():

        if "step" in state:

            step = state["step"]

            if torch.is_tensor(step):

                step = step.item()

            steps.append(int(step))

    return steps


def snapshot_parameters(model):

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
                f"WARNING: Parameter not found: {name}"
            )

            continue

        snapshots[name] = (
            named_params[name]
            .detach()
            .cpu()
            .clone()
        )

    return snapshots


def check_parameter_updates(model, snapshots):

    named_params = dict(
        model.named_parameters()
    )

    changed = []

    print("\nParameter update verification:")

    for name, before in snapshots.items():

        after = (
            named_params[name]
            .detach()
            .cpu()
        )

        max_delta = (
            after - before
        ).abs().max().item()

        print(
            f"{name}: "
            f"max_delta={max_delta:.12e}"
        )

        if max_delta > 0:

            changed.append(name)

    if not changed:

        raise RuntimeError(
            "No representative parameters were updated."
        )

    print(
        "Updated representative parameters:",
        changed
    )

    return changed


def check_gradients(model):

    grads = [

        p.grad

        for p in model.parameters()

        if p.grad is not None

    ]

    if not grads:

        raise RuntimeError(
            "No gradients found."
        )

    num_grad_params = len(grads)

    num_nonzero_grad_params = sum(

        torch.count_nonzero(g).item() > 0

        for g in grads

    )

    all_grads_finite = all(

        torch.isfinite(g).all().item()

        for g in grads

    )

    print("\nGradient verification:")

    print(
        "Parameters with gradients:",
        num_grad_params
    )

    print(
        "Parameters with nonzero gradients:",
        num_nonzero_grad_params
    )

    print(
        "All gradients finite:",
        all_grads_finite
    )

    if not all_grads_finite:

        raise RuntimeError(
            "Non-finite gradients detected."
        )

    if num_nonzero_grad_params == 0:

        raise RuntimeError(
            "All gradients are zero."
        )

    return (
        num_grad_params,
        num_nonzero_grad_params,
    )


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    name,
    config,
    selected_indices,
    num_batches,
    batch_size,
):

    print("\n")

    print("=" * 80)

    print(
        f"RUNNING MULTI-BATCH TEST: {name}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Set reproducible seed for each experiment
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Production arguments
    # --------------------------------------------------------

    from test_motion_pretraining_batch import (
        build_production_args,
    )

    args = build_production_args()

    args.batch_size = batch_size

    args.num_workers = 0

    args.pin_mem = False

    args.device = "cuda"

    args.distributed = False

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

    args.epochs = 1

    args.start_epoch = 0

    # Preserve the setting used in your existing
    # successful single-batch integration test.
    args.clip_grad = 0

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    from datasets import (
        build_pretraining_dataset,
    )

    dataset = build_pretraining_dataset(args)

    if max(selected_indices) >= len(dataset):

        raise RuntimeError(
            "Selected sample indices exceed dataset size."
        )

    test_dataset = Subset(
        dataset,
        selected_indices,
    )

    data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    if len(data_loader) != num_batches:

        raise RuntimeError(
            f"Expected {num_batches} batches, "
            f"but DataLoader contains {len(data_loader)}."
        )

    print("\nDataset:")

    print(
        "Full dataset size:",
        len(dataset)
    )

    print(
        "Selected video samples:",
        len(selected_indices)
    )

    print(
        "Batch size:",
        batch_size
    )

    print(
        "Number of batches:",
        len(data_loader)
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    import run_mae_pretraining

    model = run_mae_pretraining.get_model(args)

    print(
        "\nLoading pretrained checkpoint:"
    )

    print(
        args.pretrained
    )

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

    model.to("cuda")

    print(
        "Pretrained model loaded successfully."
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    from optim_factory import (
        create_optimizer,
    )

    optimizer = create_optimizer(
        args,
        model,
    )

    # --------------------------------------------------------
    # AMP scaler
    # --------------------------------------------------------

    from utils import (
        NativeScalerWithGradNormCount,
    )

    loss_scaler = (
        NativeScalerWithGradNormCount()
    )

    scale_before = (
        loss_scaler.state_dict()["scale"]
    )

    # --------------------------------------------------------
    # Parameter snapshots
    # --------------------------------------------------------

    snapshots = snapshot_parameters(
        model
    )

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    memory_before = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    print(
        f"\nGPU allocated before training: "
        f"{memory_before:.2f} GB"
    )

    # --------------------------------------------------------
    # Run actual production training engine
    # --------------------------------------------------------

    print(
        f"\nStarting production training "
        f"for {num_batches} batches..."
    )

    from engine_for_pretraining import (
        train_one_epoch,
    )

    stats = train_one_epoch(

        model=model,

        data_loader=data_loader,

        optimizer=optimizer,

        device=torch.device("cuda"),

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
    # Loss verification
    # --------------------------------------------------------

    print("\nTraining statistics:")

    print(stats)

    if "loss" not in stats:

        raise RuntimeError(
            "Training statistics do not contain loss."
        )

    final_loss = float(
        stats["loss"]
    )

    if not np.isfinite(final_loss):

        raise RuntimeError(
            "Training loss is non-finite."
        )

    # --------------------------------------------------------
    # AMP verification
    # --------------------------------------------------------

    scale_after = (
        loss_scaler.state_dict()["scale"]
    )

    print("\nAMP:")

    print(
        "Scale before:",
        scale_before
    )

    print(
        "Scale after:",
        scale_after
    )

    # --------------------------------------------------------
    # Gradient verification
    # --------------------------------------------------------

    (
        num_grad_params,
        num_nonzero_grad_params,
    ) = check_gradients(model)

    # --------------------------------------------------------
    # Parameter updates
    # --------------------------------------------------------

    changed_params = (
        check_parameter_updates(
            model,
            snapshots,
        )
    )

    # --------------------------------------------------------
    # Optimizer state and step verification
    # --------------------------------------------------------

    optimizer_state_entries = len(
        optimizer.state
    )

    optimizer_steps = (
        get_optimizer_steps(
            optimizer
        )
    )

    print("\nOptimizer verification:")

    print(
        "Optimizer state entries:",
        optimizer_state_entries
    )

    print(
        "Optimizer step values:",
        sorted(set(optimizer_steps))
    )

    if optimizer_state_entries == 0:

        raise RuntimeError(
            "Optimizer state was not initialized."
        )

    if not optimizer_steps:

        raise RuntimeError(
            "No optimizer step counters found."
        )

    if min(optimizer_steps) < 1:

        raise RuntimeError(
            "Optimizer did not perform any update."
        )

    # We expect an optimizer update on each batch.
    # If AMP skips an update due to overflow, the test
    # should report that instead of silently passing.

    if min(optimizer_steps) != num_batches:

        raise RuntimeError(
            "Some optimizer updates were skipped. "
            f"Expected {num_batches} updates, "
            f"observed minimum step {min(optimizer_steps)}."
        )

    if max(optimizer_steps) != num_batches:

        raise RuntimeError(
            "Optimizer step count is inconsistent. "
            f"Expected {num_batches}, "
            f"observed maximum step {max(optimizer_steps)}."
        )

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print("\nGPU memory:")

    print(
        f"Peak allocated: "
        f"{peak_memory:.2f} GB"
    )

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    print("\n")

    print("=" * 80)

    print(
        f"{name} MULTI-BATCH TEST PASSED"
    )

    print("=" * 80)

    print(
        "Finite loss:                PASS"
    )

    print(
        "Finite gradients:            PASS"
    )

    print(
        "Nonzero gradients:           PASS"
    )

    print(
        "Representative parameters:   PASS"
    )

    print(
        "Optimizer updates:            PASS"
    )

    print(
        "Motion dataset loading:      PASS"
    )

    print(
        "Completed batches:",
        num_batches,
    )

    print(
        "Final loss:",
        final_loss,
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "experiment": name,

        "loss": final_loss,

        "batches": num_batches,

        "batch_size": batch_size,

        "peak_memory_gb": peak_memory,

        "optimizer_steps": (
            sorted(set(optimizer_steps))
        ),

        "grad_params": num_grad_params,

        "nonzero_grad_params": (
            num_nonzero_grad_params
        ),

        "amp_scale_before": (
            float(scale_before)
        ),

        "amp_scale_after": (
            float(scale_after)
        ),

        "changed_parameters": (
            changed_params
        ),

    }

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del model

    del optimizer

    del loss_scaler

    del dataset

    del test_dataset

    del data_loader

    del args

    gc.collect()

    torch.cuda.empty_cache()

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Multi-batch stability test "
            "for motion-aware VideoMAE."
        )
    )

    parser.add_argument(
        "--batches",
        type=int,
        default=10,
        help="Number of training batches per experiment.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size.",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["B1", "B2", "B3"],
        default=["B1", "B2", "B3"],
        help="Experiments to run.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parsed = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if parsed.batches < 1:

        parser.error(
            "--batches must be at least 1"
        )

    if parsed.batch_size < 1:

        parser.error(
            "--batch-size must be at least 1"
        )

    # --------------------------------------------------------
    # CUDA check
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required for this test."
        )

    set_seed(
        parsed.seed
    )

    print("=" * 80)

    print(
        "MOTION PRETRAINING MULTI-BATCH VALIDATION"
    )

    print("=" * 80)

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "Batch size:",
        parsed.batch_size
    )

    print(
        "Batches per experiment:",
        parsed.batches
    )

    print(
        "Experiments:",
        parsed.experiments
    )

    # --------------------------------------------------------
    # Select same videos for every experiment
    # --------------------------------------------------------

    from test_motion_pretraining_batch import (
        build_production_args,
    )

    from datasets import (
        build_pretraining_dataset,
    )

    args = build_production_args()

    args.motion_guided_mask = False

    args.motion_weighted_loss = False

    args.motion_alpha = 1.0

    args.motion_beta = 2.0

    args.batch_size = parsed.batch_size

    args.num_workers = 0

    args.pin_mem = False

    print(
        "\nInitializing dataset to select test videos..."
    )

    selection_dataset = (
        build_pretraining_dataset(args)
    )

    required_samples = (
        parsed.batches
        * parsed.batch_size
    )

    if len(selection_dataset) < required_samples:

        raise RuntimeError(
            f"Dataset has {len(selection_dataset)} samples, "
            f"but {required_samples} are required."
        )

    rng = np.random.default_rng(
        parsed.seed
    )

    selected_indices = rng.choice(

        len(selection_dataset),

        size=required_samples,

        replace=False,

    ).tolist()

    print(
        "\nSelected video indices:"
    )

    print(
        selected_indices
    )

    print(
        "Selection size:",
        len(selected_indices)
    )

    del selection_dataset

    del args

    gc.collect()

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    results = []

    for name in parsed.experiments:

        result = run_experiment(

            name=name,

            config=EXPERIMENTS[name],

            selected_indices=selected_indices,

            num_batches=parsed.batches,

            batch_size=parsed.batch_size,

        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")

    print("=" * 100)

    print(
        "MULTI-BATCH TEST SUMMARY"
    )

    print("=" * 100)

    header = (

        f"{'Experiment':<12}"

        f"{'Loss':<15}"

        f"{'Batches':<12}"

        f"{'Batch Size':<12}"

        f"{'Peak GPU GB':<15}"

        f"{'Optimizer Steps'}"

    )

    print(header)

    print("-" * 100)

    for result in results:

        print(

            f"{result['experiment']:<12}"

            f"{result['loss']:<15.6f}"

            f"{result['batches']:<12}"

            f"{result['batch_size']:<12}"

            f"{result['peak_memory_gb']:<15.2f}"

            f"{result['optimizer_steps']}"

        )

    print("\n")

    print(
        "ALL REQUESTED EXPERIMENTS PASSED."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()