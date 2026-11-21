import sys
import subprocess
from pathlib import Path

from scripts.utils.config import (
    load_experiment_config,
    resolve_path,
)

from scripts.utils.experiment import (
    prepare_experiment_directory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEOMAE_DIR = (
    PROJECT_ROOT / "VideoMAE"
)


def build_command(
    config,
    output_dir,
):

    video = config["video"]
    model = config["model"]
    masking = config["masking"]
    optimizer = config["optimizer"]
    training = config["training"]
    dataset = config["dataset"]
    dataloader = config["dataloader"]

    command = [
        sys.executable,
        str(VIDEOMAE_DIR / "run_mae_pretraining.py"),

        "--batch_size",
        str(training["batch_size"]),

        "--epochs",
        str(training["epochs"]),

        "--save_ckpt_freq",
        str(training["save_ckpt_freq"]),

        "--model",
        model["name"],

        "--decoder_depth",
        str(model["decoder_depth"]),

        "--mask_type",
        masking["type"],

        "--mask_ratio",
        str(masking["ratio"]),

        "--input_size",
        str(video["input_size"]),

        "--drop_path",
        str(model["drop_path"]),

        "--opt",
        optimizer["type"],

        "--opt_eps",
        str(optimizer["opt_eps"]),

        "--weight_decay",
        str(optimizer["weight_decay"]),

        "--lr",
        str(optimizer["lr"]),

        "--min_lr",
        str(optimizer["min_lr"]),

        "--warmup_lr",
        str(optimizer["warmup_lr"]),

        "--warmup_epochs",
        str(optimizer["warmup_epochs"]),

        "--warmup_steps",
        str(optimizer["warmup_steps"]),

        "--data_path",
        str(
            resolve_path(
                dataset["data_path"]
            )
        ),

        "--num_frames",
        str(video["num_frames"]),

        "--sampling_rate",
        str(video["sampling_rate"]),

        "--output_dir",
        str(output_dir),

        "--log_dir",
        str(
            output_dir / "logs"
        ),

        "--device",
        config["device"],

        "--seed",
        str(config["seed"]),

        "--pretrained",
        str(
            resolve_path(
                config["pretrained"]["path"]
            )
        ),

        "--num_workers",
        str(
            dataloader["num_workers"]
        ),
    ]

    if dataloader["pin_mem"]:
        command.append("--pin_mem")

    
    # --------------------------------------------------
    # Optional motion-aware pretraining configuration
    # --------------------------------------------------
    motion = config.get("motion", {})

    if motion.get("enabled", False):
        command.extend([
            "--motion_enabled",
            "--motion_root",
            str(resolve_path(motion.get(
                "root",
                "raft/output_datasets_motion/motion_npz"
            ))),
            "--motion_gap",
            str(motion.get("gap", 4)),
            "--motion_missing_policy",
            str(motion.get("missing_policy", "zero")),
            "--motion_percentile",
            str(motion.get("percentile", 99.0)),
        ])

        if motion.get("guided_mask", False):
            command.append("--motion_guided_mask")

        if motion.get("weighted_loss", False):
            command.append("--motion_weighted_loss")

        command.extend([
            "--motion_alpha",
            str(motion.get("alpha", 1.0)),
            "--motion_beta",
            str(motion.get("beta", 2.0)),
        ])

    if model["use_checkpoint"]:
        command.append(
            "--use_checkpoint"
        )

    if (
        optimizer["opt_betas"]
        is not None
    ):

        command.append(
            "--opt_betas"
        )

        for beta in optimizer["opt_betas"]:
            command.append(
                str(beta)
            )

    return command


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python3 -m scripts.pretrain "
            "<experiment_name>\n\n"
            "Example:\n"
            "  python3 -m scripts.pretrain "
            "exp01_random"
        )

        sys.exit(1)

    experiment_name = sys.argv[1]

    print("=" * 70)
    print(
        f"Loading experiment: "
        f"{experiment_name}"
    )
    print("=" * 70)

    config = load_experiment_config(
        experiment_name
    )

    output_dir = (
        prepare_experiment_directory(
            config
        )
    )

    command = build_command(
        config,
        output_dir,
    )

    print("\nExperiment:")
    print(
        config["experiment"]["name"]
    )

    print("\nOutput directory:")
    print(output_dir)

    print("\nStarting VideoMAE:")
    print(
        " ".join(command)
    )

    print("\n" + "=" * 70)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()