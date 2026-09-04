"""
Reconstruct videos from VideoMAE pretraining checkpoints.

For each checkpoint:
    1. Load a VideoMAE V1 ViT-Base model.
    2. Select deterministic UCF101 videos from distinct classes.
    3. Sample a 16-frame clip.
    4. Apply deterministic random tubelet masking.
    5. Reconstruct masked tubelets using the model.
    6. Save original, masked, and reconstructed videos.

Expected experiment structure:

outputs/
└── experiment1/
    └── exp01_random/
        ├── checkpoints/
        │   ├── checkpoint-19.pth
        │   ├── checkpoint-39.pth
        │   └── ...
        ├── config/
        ├── logs/
        ├── reconstructions/
        └── evaluation/

To run

python3 -m scripts.reconstruct_checkpoints \
    --experiment smoke_test \
    --checkpoint_dir outputs/experiment1/smoke_test \
    --output_dir outputs/experiment1/smoke_test/reconstructions
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEOMAE_ROOT = PROJECT_ROOT / "VideoMAE"

sys.path.insert(0, str(VIDEOMAE_ROOT))

from modeling_pretrain import pretrain_videomae_base_patch16_224


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

NUM_FRAMES = 16
SAMPLING_RATE = 4

INPUT_SIZE = 224

TUBELET_SIZE = 2
PATCH_SIZE = 16

MASK_RATIO = 0.75

NUM_TOKENS = (
    (NUM_FRAMES // TUBELET_SIZE)
    * (INPUT_SIZE // PATCH_SIZE)
    * (INPUT_SIZE // PATCH_SIZE)
)

# ImageNet normalization used by VideoMAE
IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)

# Deterministic mask seed
BASE_MASK_SEED = 1234

# ---------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------

DEFAULT_EXPERIMENT = "exp01_random"

DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "UCF-101"
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def find_checkpoints(checkpoint_dir):
    """
    Find all VideoMAE checkpoints in the experiment checkpoint directory.
    """

    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist:\n"
            f"{checkpoint_dir}"
        )

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*.pth"),
        key=lambda p: int(
            p.stem.split("-")[-1]
        ),
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in:\n"
            f"{checkpoint_dir}"
        )

    return checkpoints


def find_videos(data_root):
    """
    Find all UCF101 videos recursively.

    Expected structure:

    UCF-101/
        ApplyEyeMakeup/
            v_ApplyEyeMakeup_g01_c01.avi
        Archery/
            ...
        ...
    """

    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(
            f"UCF101 data root does not exist:\n"
            f"{data_root}"
        )

    video_extensions = {
        ".avi",
        ".mp4",
        ".mov",
        ".mkv",
    }

    videos = [
        path
        for path in data_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in video_extensions
    ]

    videos = sorted(videos)

    if not videos:
        raise FileNotFoundError(
            f"No video files found under:\n"
            f"{data_root}"
        )

    return videos


def select_fixed_videos(
    videos,
    num_videos=10,
):
    """
    Select deterministic videos from distinct UCF101 classes.

    The selection is based on sorted class names and uses evenly
    spaced classes to provide coverage across the dataset.

    For each selected class, the first sorted video is selected.

    Returns:
        selected_videos: list[Path]
    """

    # Group videos by class.
    class_to_videos = {}

    for video_path in videos:
        class_name = video_path.parent.name

        if class_name not in class_to_videos:
            class_to_videos[class_name] = []

        class_to_videos[class_name].append(video_path)

    # Sort videos within each class.
    for class_name in class_to_videos:
        class_to_videos[class_name] = sorted(
            class_to_videos[class_name]
        )

    classes = sorted(class_to_videos.keys())

    if len(classes) < num_videos:
        raise ValueError(
            f"Requested {num_videos} videos from distinct classes, "
            f"but only {len(classes)} classes were found."
        )

    # Evenly distribute selections across the sorted class list.
    class_indices = np.linspace(
        0,
        len(classes) - 1,
        num=num_videos,
        dtype=int,
    )

    # Remove accidental duplicate indices.
    class_indices = list(dict.fromkeys(class_indices))

    # If rounding somehow produces fewer classes, fill deterministically.
    if len(class_indices) < num_videos:
        for index in range(len(classes)):
            if index not in class_indices:
                class_indices.append(index)

            if len(class_indices) == num_videos:
                break

    class_indices = sorted(class_indices)

    selected_videos = []

    for index in class_indices:
        class_name = classes[index]

        # First sorted video = deterministic selection.
        selected_videos.append(
            class_to_videos[class_name][0]
        )

    return selected_videos


def save_selection_manifest(
    selected_videos,
    output_dir,
    checkpoint_names,
):
    """
    Save the exact videos and reconstruction parameters used.
    """

    manifest = {
        "num_videos": len(selected_videos),
        "videos": [
            {
                "index": index,
                "class": video_path.parent.name,
                "filename": video_path.name,
                "path": str(video_path),
            }
            for index, video_path
            in enumerate(selected_videos)
        ],
        "reconstruction_parameters": {
            "num_frames": NUM_FRAMES,
            "sampling_rate": SAMPLING_RATE,
            "input_size": INPUT_SIZE,
            "tubelet_size": TUBELET_SIZE,
            "patch_size": PATCH_SIZE,
            "mask_ratio": MASK_RATIO,
            "num_tokens": NUM_TOKENS,
            "base_mask_seed": BASE_MASK_SEED,
        },
        "checkpoints": [
            checkpoint.name
            for checkpoint in checkpoint_names
        ],
    }

    manifest_path = output_dir / "selection_manifest.json"

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(
        f"Saved selection manifest:\n"
        f"  {manifest_path}"
    )


# ---------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------

def load_video_frames(video_path):
    """
    Load all frames from a video using OpenCV.

    Returns:
        frames: list of RGB numpy arrays
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )

    frames = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(
            f"No frames could be read from:\n"
            f"{video_path}"
        )

    return frames


def select_frame_indices(
    num_available_frames,
):
    """
    Select 16 frames using VideoMAE-style temporal sampling.

    Desired:
        16 frames
        sampling rate = 4
    """

    required_span = (
        (NUM_FRAMES - 1) * SAMPLING_RATE
    ) + 1

    if num_available_frames >= required_span:

        max_start = (
            num_available_frames
            - required_span
        )

        # Deterministic center start.
        start = max_start // 2

        indices = (
            start
            + np.arange(NUM_FRAMES)
            * SAMPLING_RATE
        )

    else:
        # If the video is too short, sample evenly
        # across the available frames.
        indices = np.linspace(
            0,
            num_available_frames - 1,
            NUM_FRAMES,
            dtype=int,
        )

    return indices.astype(int)


def load_clip(video_path):
    """
    Load a deterministic 16-frame RGB clip.
    """

    frames = load_video_frames(video_path)

    indices = select_frame_indices(
        len(frames)
    )

    clip = np.stack(
        [frames[index] for index in indices],
        axis=0,
    )

    return clip


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def resize_frames(clip):
    """
    Resize frames to 224x224.
    """

    resized = []

    for frame in clip:
        frame = cv2.resize(
            frame,
            (INPUT_SIZE, INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        resized.append(frame)

    return np.stack(
        resized,
        axis=0,
    )


def normalize_clip(clip):
    """
    Apply ImageNet normalization.

    Input:
        [T, H, W, C]
        uint8 RGB

    Output:
        [T, H, W, C]
        float32
    """

    clip = clip.astype(np.float32) / 255.0

    clip = (
        clip - IMAGENET_MEAN
    ) / IMAGENET_STD

    return clip


def preprocess_clip(clip):
    """
    Convert video clip to VideoMAE input format.

    Input:
        [T, H, W, C]

    Output:
        [1, C, T, H, W]
    """

    clip = resize_frames(clip)
    clip = normalize_clip(clip)

    tensor = torch.from_numpy(
        clip
    ).permute(
        3,
        0,
        1,
        2,
    )

    tensor = tensor.unsqueeze(0)

    return tensor.float()


# ---------------------------------------------------------------------
# Tubelet masking
# ---------------------------------------------------------------------

def create_random_tube_mask(
    seed,
    device,
):
    """
    Create a deterministic random tubelet mask.

    VideoMAE token layout:

        temporal tubelets:
            16 / 2 = 8

        spatial patches:
            224 / 16 = 14 x 14

        total:
            8 x 14 x 14 = 1568

    True  = masked
    False = visible
    """

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(seed)

    num_masked = int(
        NUM_TOKENS * MASK_RATIO
    )

    permutation = torch.randperm(
        NUM_TOKENS,
        generator=generator,
    )

    mask = torch.zeros(
        NUM_TOKENS,
        dtype=torch.bool,
    )

    mask[
        permutation[:num_masked]
    ] = True

    return mask.to(device)


# ---------------------------------------------------------------------
# Patchify / unpatchify
# ---------------------------------------------------------------------

def patchify(video):
    """
    Convert video into tubelet tokens.

    Input:
        [B, C, T, H, W]

    Output:
        [B, N, tubelet_size * patch_size * patch_size * C]
    """

    B, C, T, H, W = video.shape

    assert (
        T % TUBELET_SIZE == 0
    )

    assert (
        H % PATCH_SIZE == 0
    )

    assert (
        W % PATCH_SIZE == 0
    )

    t = T // TUBELET_SIZE
    h = H // PATCH_SIZE
    w = W // PATCH_SIZE

    x = video.reshape(
        B,
        C,
        t,
        TUBELET_SIZE,
        h,
        PATCH_SIZE,
        w,
        PATCH_SIZE,
    )

    x = x.permute(
        0,
        2,
        4,
        6,
        3,
        5,
        7,
        1,
    )

    x = x.reshape(
        B,
        t * h * w,
        TUBELET_SIZE
        * PATCH_SIZE
        * PATCH_SIZE
        * C,
    )

    return x


def unpatchify(
    patches,
    T,
    H,
    W,
):
    """
    Convert tubelet tokens back to video.

    Input:
        [B, N, patch_dim]

    Output:
        [B, C, T, H, W]
    """

    B, N, patch_dim = patches.shape

    C = patch_dim // (
        TUBELET_SIZE
        * PATCH_SIZE
        * PATCH_SIZE
    )

    t = T // TUBELET_SIZE
    h = H // PATCH_SIZE
    w = W // PATCH_SIZE

    x = patches.reshape(
        B,
        t,
        h,
        w,
        TUBELET_SIZE,
        PATCH_SIZE,
        PATCH_SIZE,
        C,
    )

    x = x.permute(
        0,
        7,
        1,
        4,
        2,
        5,
        3,
        6,
    )

    x = x.reshape(
        B,
        C,
        T,
        H,
        W,
    )

    return x


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def denormalize_clip(tensor):
    """
    Reverse ImageNet normalization.

    Input:
        [T, H, W, C]
    """

    tensor = tensor.detach().cpu().numpy()

    tensor = (
        tensor * IMAGENET_STD
    ) + IMAGENET_MEAN

    tensor = np.clip(
        tensor,
        0.0,
        1.0,
    )

    tensor = (
        tensor * 255.0
    ).astype(np.uint8)

    return tensor


def apply_mask_to_video(
    video,
    mask,
):
    """
    Zero masked tubelets for visualization.

    Input:
        video:
            [1, C, T, H, W]

        mask:
            [N]
    """

    patches = patchify(video)

    masked_patches = patches.clone()

    masked_patches[
        :, mask, :
    ] = 0.0

    masked_video = unpatchify(
        masked_patches,
        T=NUM_FRAMES,
        H=INPUT_SIZE,
        W=INPUT_SIZE,
    )

    return masked_video


# ---------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------

@torch.no_grad()
def reconstruct_video(
    model,
    video,
    mask,
):
    """
    Reconstruct the masked tubelets.

    The model predicts normalized target patches.

    Visible patches:
        original input patches

    Masked patches:
        model predictions
    """

    model.eval()

    prediction = model(
        video,
        mask.unsqueeze(0),
    )

    # Original patch representation.
    original_patches = patchify(video)

    # Model prediction.
    reconstructed_patches = (
        original_patches.clone()
    )

    # VideoMAE target construction:
    # target patches are normalized independently.
    target = patchify(video)

    mean = target.mean(
        dim=-1,
        keepdim=True,
    )

    var = target.var(
        dim=-1,
        keepdim=True,
    )

    target_normalized = (
        target - mean
    ) / torch.sqrt(
        var + 1e-6
    )

    # Insert predictions for masked tokens.
    reconstructed_patches[
        :, mask, :
    ] = prediction

    # Reverse target normalization for predicted patches.
    reconstructed_patches[
        :, mask, :
    ] = (
        reconstructed_patches[
            :, mask, :
        ]
        * torch.sqrt(
            var[:, mask, :]
            + 1e-6
        )
        + mean[:, mask, :]
    )

    reconstructed_video = unpatchify(
        reconstructed_patches,
        T=NUM_FRAMES,
        H=INPUT_SIZE,
        W=INPUT_SIZE,
    )

    return reconstructed_video


# ---------------------------------------------------------------------
# Video writing
# ---------------------------------------------------------------------

def save_video(
    clip,
    output_path,
    fps=25,
):
    """
    Save RGB uint8 video as MP4.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    T, H, W, C = clip.shape

    assert C == 3

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (W, H),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create video:\n"
            f"{output_path}"
        )

    for frame in clip:

        frame_bgr = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR,
        )

        writer.write(
            frame_bgr
        )

    writer.release()


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_model(
    checkpoint_path,
    device,
):
    """
    Load VideoMAE V1 ViT-Base from checkpoint.
    """

    print(
        f"\nLoading checkpoint:\n"
        f"  {checkpoint_path}"
    )

    model = (
        pretrain_videomae_base_patch16_224(
            decoder_depth=4,
            drop_path_rate=0.0,
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "'model' state dictionary."
        )

    state_dict = checkpoint["model"]

    message = model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        f"Checkpoint loaded successfully."
    )

    print(
        f"  Missing keys: "
        f"{len(message.missing_keys)}"
    )

    print(
        f"  Unexpected keys: "
        f"{len(message.unexpected_keys)}"
    )

    model.to(device)

    model.eval()

    return model


# ---------------------------------------------------------------------
# Main reconstruction
# ---------------------------------------------------------------------

def reconstruct_checkpoint(
    model,
    checkpoint_path,
    selected_videos,
    output_dir,
    device,
):
    """
    Reconstruct all selected videos using one checkpoint.
    """

    checkpoint_name = (
        checkpoint_path.stem
    )

    checkpoint_output_dir = (
        output_dir
        / checkpoint_name
    )

    checkpoint_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"Checkpoint: {checkpoint_name}"
    )

    print(
        f"{'=' * 70}"
    )

    for video_index, video_path in enumerate(
        selected_videos
    ):

        class_name = (
            video_path.parent.name
        )

        video_stem = (
            video_path.stem
        )

        print(
            f"\n[{video_index + 1}/"
            f"{len(selected_videos)}]"
        )

        print(
            f"Class: {class_name}"
        )

        print(
            f"Video: {video_path.name}"
        )

        # -------------------------------------------------------------
        # Load clip
        # -------------------------------------------------------------

        clip = load_clip(
            video_path
        )

        # -------------------------------------------------------------
        # Preprocess
        # -------------------------------------------------------------

        video_tensor = preprocess_clip(
            clip
        ).to(device)

        # -------------------------------------------------------------
        # Create deterministic mask
        # -------------------------------------------------------------

        mask_seed = (
            BASE_MASK_SEED
            + video_index
        )

        mask = create_random_tube_mask(
            seed=mask_seed,
            device=device,
        )

        # -------------------------------------------------------------
        # Masked input
        # -------------------------------------------------------------

        masked_video = (
            apply_mask_to_video(
                video_tensor,
                mask,
            )
        )

        # -------------------------------------------------------------
        # Reconstruction
        # -------------------------------------------------------------

        reconstructed_video = (
            reconstruct_video(
                model,
                video_tensor,
                mask,
            )
        )

        # -------------------------------------------------------------
        # Convert back to RGB uint8
        # -------------------------------------------------------------

        original_vis = denormalize_clip(
            video_tensor[0]
            .permute(1, 2, 3, 0)
        )

        masked_vis = denormalize_clip(
            masked_video[0]
            .permute(1, 2, 3, 0)
        )

        reconstructed_vis = (
            denormalize_clip(
                reconstructed_video[0]
                .permute(1, 2, 3, 0)
            )
        )

        # -------------------------------------------------------------
        # Filename
        # -------------------------------------------------------------

        prefix = (
            f"{class_name}__{video_stem}"
        )

        original_path = (
            checkpoint_output_dir
            / f"{prefix}__original.mp4"
        )

        masked_path = (
            checkpoint_output_dir
            / f"{prefix}__masked.mp4"
        )

        reconstructed_path = (
            checkpoint_output_dir
            / f"{prefix}__reconstructed.mp4"
        )

        # -------------------------------------------------------------
        # Save videos
        # -------------------------------------------------------------

        save_video(
            original_vis,
            original_path,
        )

        save_video(
            masked_vis,
            masked_path,
        )

        save_video(
            reconstructed_vis,
            reconstructed_path,
        )

        print(
            "Saved:"
        )

        print(
            f"  Original:       {original_path.name}"
        )

        print(
            f"  Masked:         {masked_path.name}"
        )

        print(
            f"  Reconstructed:  "
            f"{reconstructed_path.name}"
        )

    print(
        f"\nCompleted checkpoint: "
        f"{checkpoint_name}"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct UCF101 videos "
            "from VideoMAE checkpoints."
        )
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT,
        help=(
            "Experiment name under "
            "outputs/experiment1/"
        ),
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Path to UCF101 video directory.",
    )

    parser.add_argument(
        "--num_videos",
        type=int,
        default=10,
        help=(
            "Number of videos to reconstruct. "
            "Each video comes from a different "
            "UCF101 class."
        ),
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Optional explicit checkpoint "
            "directory."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Optional explicit reconstruction "
            "output directory."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    # -------------------------------------------------------------
    # Resolve experiment paths
    # -------------------------------------------------------------

    experiment_dir = (
        PROJECT_ROOT
        / "outputs"
        / "experiment1"
        / args.experiment
    )

    if args.checkpoint_dir is None:

        checkpoint_dir = (
            experiment_dir
            / "checkpoints"
        )

    else:

        checkpoint_dir = Path(
            args.checkpoint_dir
        )

    if args.output_dir is None:

        output_dir = (
            experiment_dir
            / "reconstructions"
        )

    else:

        output_dir = Path(
            args.output_dir
        )

    data_root = Path(
        args.data_root
    )

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 70
    )

    print(
        "VideoMAE Checkpoint Reconstruction"
    )

    print(
        "=" * 70
    )

    print(
        f"Experiment:       {args.experiment}"
    )

    print(
        f"Checkpoint dir:   {checkpoint_dir}"
    )

    print(
        f"Output dir:       {output_dir}"
    )

    print(
        f"Dataset:          {data_root}"
    )

    print(
        f"Device:           {device}"
    )

    print(
        f"Videos:           {args.num_videos}"
    )

    print(
        f"Mask ratio:       {MASK_RATIO}"
    )

    print(
        f"Num tokens:       {NUM_TOKENS}"
    )

    # -------------------------------------------------------------
    # Find checkpoints
    # -------------------------------------------------------------

    checkpoints = find_checkpoints(
        checkpoint_dir
    )

    print(
        f"\nFound {len(checkpoints)} checkpoint(s):"
    )

    for checkpoint in checkpoints:
        print(
            f"  - {checkpoint.name}"
        )

    # -------------------------------------------------------------
    # Find UCF101 videos
    # -------------------------------------------------------------

    videos = find_videos(
        data_root
    )

    print(
        f"\nFound {len(videos)} videos."
    )

    # -------------------------------------------------------------
    # Select deterministic videos
    # -------------------------------------------------------------

    selected_videos = (
        select_fixed_videos(
            videos,
            num_videos=args.num_videos,
        )
    )

    print(
        "\nSelected videos:"
    )

    for index, video_path in enumerate(
        selected_videos
    ):

        print(
            f"  {index + 1:02d}. "
            f"{video_path.parent.name}/"
            f"{video_path.name}"
        )

    # -------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save manifest
    # -------------------------------------------------------------

    save_selection_manifest(
        selected_videos,
        output_dir,
        checkpoints,
    )

    # -------------------------------------------------------------
    # Process every checkpoint
    # -------------------------------------------------------------

    for checkpoint_path in checkpoints:

        model = load_model(
            checkpoint_path,
            device,
        )

        reconstruct_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            selected_videos=selected_videos,
            output_dir=output_dir,
            device=device,
        )

        # Free GPU memory before loading
        # the next checkpoint.
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Reconstruction completed successfully."
    )

    print(
        f"Results saved to:\n"
        f"  {output_dir}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()