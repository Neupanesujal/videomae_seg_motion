from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEOMAE_ROOT = PROJECT_ROOT / "VideoMAE"

DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "ucfTrainTestlist"
    / "experiment1_pretrain.txt"
)

PRETRAINED_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "pretrained"
    / "videomae_v1_base_kinetics_e1600.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiment1"
    / "smoke_test"
)


def build_command():
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",

        # Single GPU
        "--nproc_per_node=1",

        str(VIDEOMAE_ROOT / "run_mae_pretraining.py"),

        # --------------------------------------------------
        # Dataset
        # --------------------------------------------------
        "--data_path",
        str(DATA_PATH),

        # --------------------------------------------------
        # Model
        # --------------------------------------------------
        "--model",
        "pretrain_videomae_base_patch16_224",

        # Starting from Kinetics-400 pretrained VideoMAE
        "--pretrained",
        str(PRETRAINED_CHECKPOINT),

        # --------------------------------------------------
        # VideoMAE configuration
        # --------------------------------------------------
        "--mask_type",
        "tube",

        "--mask_ratio",
        "0.75",

        "--decoder_depth",
        "4",

        "--num_frames",
        "16",

        "--sampling_rate",
        "4",

        "--input_size",
        "224",

        # --------------------------------------------------
        # Optimization
        # --------------------------------------------------
        "--opt",
        "adamw",

        "--opt_betas",
        "0.9",
        "0.95",

        "--lr",
        "3e-4",

        "--warmup_epochs",
        "0",

        # --------------------------------------------------
        # Training
        # --------------------------------------------------
        "--batch_size",
        "8",

        "--epochs",
        "4",

        # --------------------------------------------------
        # Checkpointing
        # --------------------------------------------------
        "--save_ckpt_freq",
        "1",

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        "--log_dir",
        str(OUTPUT_DIR),

        "--output_dir",
        str(OUTPUT_DIR),

        # --------------------------------------------------
        # Memory optimization
        # --------------------------------------------------
        "--use_checkpoint",
    ]


def main():

    # ------------------------------------------------------
    # Validate required files
    # ------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Pretraining list not found:\n{DATA_PATH}"
        )

    if not PRETRAINED_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found:\n"
            f"{PRETRAINED_CHECKPOINT}"
        )

    # ------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Build command
    # ------------------------------------------------------

    command = build_command()

    print("=" * 70)
    print("VideoMAE UCF101 PRETRAINING SMOKE TEST")
    print("=" * 70)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {DATA_PATH}")
    print(f"Checkpoint   : {PRETRAINED_CHECKPOINT}")
    print(f"Output       : {OUTPUT_DIR}")

    print()
    print("Configuration:")
    print("  Model          : VideoMAE v1 ViT-Base")
    print("  Pretraining    : Kinetics-400 pretrained")
    print("  Dataset        : UCF101")
    print("  Videos         : 13,320")
    print("  Frames         : 16")
    print("  Sampling rate  : 4")
    print("  Input size     : 224")
    print("  Mask type      : Tube")
    print("  Mask ratio     : 0.75")
    print("  Decoder depth  : 4")
    print("  Batch size     : 1")
    print("  Epochs         : 1")
    print("  GPUs           : 1")

    print()
    print("Command:")
    print(" ".join(command))

    print()
    print("=" * 70)
    print("Starting pretraining...")
    print("=" * 70)
    print()

    # ------------------------------------------------------
    # Run VideoMAE
    # ------------------------------------------------------

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    # ------------------------------------------------------
    # Finished
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(f"Check outputs here:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()