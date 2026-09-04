import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEOMAE_ROOT = PROJECT_ROOT / "VideoMAE"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/pretrained/videomae_v1_base_kinetics_e1600.pth"

sys.path.insert(0, str(VIDEOMAE_ROOT))

import modeling_pretrain  # noqa: E402


def inspect_checkpoint(path):
    print(f"Checkpoint: {path}")
    print(f"Size: {path.stat().st_size / (1024 ** 2):.1f} MB")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    print("\nCheckpoint type:")
    print(type(checkpoint))

    if isinstance(checkpoint, dict):
        print("\nTop-level keys:")
        for key in checkpoint:
            print(f"  {key}")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]

            print(f"\nModel parameters: {len(state_dict)}")

            print("\nFirst 10 parameters:")
            for key, value in list(state_dict.items())[:10]:
                print(f"  {key:50s} {tuple(value.shape)}")

    return checkpoint


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = inspect_checkpoint(CHECKPOINT_PATH)

    print("\nCheckpoint inspection: OK")


if __name__ == "__main__":
    main()