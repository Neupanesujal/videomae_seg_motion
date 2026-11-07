#!/usr/bin/env python3

"""
RAFT Optical Flow Extraction for UCF-101

Pipeline
--------
Video
  ↓
Frame pairs (t -> t + GAP)
  ↓
RAFT dense optical flow
  ↓
Forward-backward consistency filtering
  ↓
RANSAC global camera-motion estimation
  ↓
Camera-motion compensation
  ↓
Temporal median smoothing
  ↓
Noise-floor filtering
  ↓
Clean motion
  ↓
Motion magnitude [N, H, W]
  ↓
Compressed .npz

Saved NPZ contents
------------------
motion_magnitude : [N, H, W] float16  (magnitude of the clean motion)
frame_idx        : [N] int32
gap              : int32
fps              : float32

Output structure
----------------
OUTPUT_ROOT/
│
├── motion_npz/
│   └── UCF-101/
│       ├── Class1/
│       │   ├── video1.npz
│       │   └── video2.npz
│       └── Class2/
│           └── ...
│
└── diagnostics/
    ├── videos/
    │   └── UCF-101/
    │       ├── Class1/
    │       └── Class2/
    │
    ├── logs/
    │   ├── processing.log
    │   └── video_results.csv
    │
    └── summary/
        ├── summary.json
        └── run_history.json

Resume behavior
---------------
If a valid .npz already exists, that video is skipped.

If the program stops:
    - completed videos remain completed
    - the currently running video has no final .npz
    - next execution automatically retries that video
    - the previous interrupted video is recorded

If a video fails:
    - failure is recorded
    - processing continues with the next video
    - rerunning the script retries it

Timing
-------
Records:
    - per-video processing time
    - total accumulated processing time
    - wall-clock time for each run
    - number of completed/skipped/failed/interrupted videos
"""

import os
import sys
import csv
import json
import time
import signal
import logging
import traceback
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch

from torchvision.models.optical_flow import (
    raft_large,
    Raft_Large_Weights,
)


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# INPUT DATASET
# ------------------------------------------------------------

# Change this to your UCF-101 root.
#
# Expected:
#
# UCF-101/
# ├── ApplyEyeMakeup/
# ├── Archery/
# ├── ...
#
INPUT_ROOT = Path(
    "datasets/UCF-101"
)


# ------------------------------------------------------------
# OUTPUT ROOT
# ------------------------------------------------------------

OUTPUT_ROOT = Path(
    "raft/output_datasets_motion"
)

MOTION_ROOT = OUTPUT_ROOT / "motion_npz"

DIAGNOSTICS_ROOT = OUTPUT_ROOT / "diagnostics"

DIAGNOSTIC_VIDEO_ROOT = (
    DIAGNOSTICS_ROOT / "videos"
)

LOG_ROOT = (
    DIAGNOSTICS_ROOT / "logs"
)

SUMMARY_ROOT = (
    DIAGNOSTICS_ROOT / "summary"
)


# ------------------------------------------------------------
# RAFT SETTINGS
# ------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use frame t -> frame t + GAP
GAP = 4

# RAFT iterations.
# torchvision RAFT uses the final prediction by default.
RAFT_ITERS = 12

# Number of frame pairs processed together on the GPU.
# Increase this if GPU memory allows it.
# This changes only the inference batching, not the motion pipeline,
# output format, saving structure, or resume behavior.
RAFT_BATCH_SIZE = 32


# ------------------------------------------------------------
# FORWARD-BACKWARD CONSISTENCY
# ------------------------------------------------------------

# Pixels are considered reliable when the forward flow
# followed by backward flow approximately returns to
# the starting pixel.
FB_THRESHOLD = 1.5


# ------------------------------------------------------------
# CAMERA MOTION ESTIMATION
# ------------------------------------------------------------

# "affine" is the recommended default for your pipeline.
CAMERA_MODEL = "affine"

# Number of pixels sampled for RANSAC.
RANSAC_POINTS = 5000

# Minimum number of points needed.
MIN_RANSAC_POINTS = 50

# Minimum fraction of points agreeing with camera model.
MIN_INLIER_RATIO = 0.30

# RANSAC reprojection threshold in pixels.
RANSAC_REPROJECTION_THRESHOLD = 2.0


# ------------------------------------------------------------
# TEMPORAL SMOOTHING
# ------------------------------------------------------------

MEDIAN_WINDOW = 3


# ------------------------------------------------------------
# NOISE FLOOR
# ------------------------------------------------------------

# Motion smaller than this value is removed.
# Units: pixels/frame.
NOISE_FLOOR = 0.3


# ------------------------------------------------------------
# DIAGNOSTIC VIDEO
# ------------------------------------------------------------

SAVE_DIAGNOSTIC_VIDEOS = True

DIAGNOSTIC_FPS = None

# Maximum number of arrows shown per frame.
MAX_ARROWS = 500

# Arrow spacing.
ARROW_STEP = 16


# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

# float16 substantially reduces storage size.
SAVE_DTYPE = np.float16

# Use compressed NPZ.
COMPRESS_NPZ = True


# ============================================================
# GLOBAL STATE
# ============================================================

CURRENT_VIDEO = None
CURRENT_VIDEO_START = None

STOP_REQUESTED = False


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    MOTION_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    DIAGNOSTIC_VIDEO_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    SUMMARY_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOGGING
# ============================================================

LOG_FILE = LOG_ROOT / "processing.log"


def setup_logging():

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console_handler = logging.StreamHandler(
        sys.stdout
    )

    console_handler.setFormatter(
        formatter
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        mode="a"
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ============================================================
# PATHS
# ============================================================

RESULTS_CSV = LOG_ROOT / "video_results.csv"

STATE_FILE = LOG_ROOT / "run_state.json"

SUMMARY_FILE = SUMMARY_ROOT / "summary.json"

RUN_HISTORY_FILE = SUMMARY_ROOT / "run_history.json"


# ============================================================
# TIME HELPERS
# ============================================================

def now_string():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def seconds_to_hms(seconds):

    seconds = float(seconds)

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = seconds % 60

    return (
        f"{hours:02d}h "
        f"{minutes:02d}m "
        f"{secs:05.2f}s"
    )


# ============================================================
# STATE MANAGEMENT
# ============================================================

def save_state(
    status,
    video=None,
    run_start=None
):

    state = {
        "status": status,
        "video": str(video) if video else None,
        "run_start": run_start,
        "updated_at": now_string(),
    }

    with open(
        STATE_FILE,
        "w"
    ) as f:

        json.dump(
            state,
            f,
            indent=2
        )


def load_state():

    if not STATE_FILE.exists():

        return None

    try:

        with open(
            STATE_FILE,
            "r"
        ) as f:

            return json.load(f)

    except Exception:

        return None


# ============================================================
# CSV RESULTS
# ============================================================

CSV_FIELDS = [
    "timestamp",
    "video",
    "class_name",
    "status",
    "duration_sec",
    "output_path",
    "diagnostic_path",
    "error",
]


def initialize_results_csv():

    if RESULTS_CSV.exists():

        return

    with open(
        RESULTS_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS
        )

        writer.writeheader()


def append_result(
    video,
    class_name,
    status,
    duration_sec=0.0,
    output_path="",
    diagnostic_path="",
    error=""
):

    with open(
        RESULTS_CSV,
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS
        )

        writer.writerow({
            "timestamp": now_string(),
            "video": str(video),
            "class_name": class_name,
            "status": status,
            "duration_sec": f"{duration_sec:.4f}",
            "output_path": str(output_path),
            "diagnostic_path": str(
                diagnostic_path
            ),
            "error": error,
        })


def load_results():

    if not RESULTS_CSV.exists():

        return []

    rows = []

    with open(
        RESULTS_CSV,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append(row)

    return rows


# ============================================================
# NPZ VALIDATION
# ============================================================

def validate_npz(path):

    """
    Check whether an existing NPZ is complete and readable.
    """

    if not path.exists():

        return False

    try:

        with np.load(
            path,
            allow_pickle=False
        ) as data:

            required = {
                "motion_magnitude",
                "frame_idx",
                "gap",
                "fps",
            }

            if not required.issubset(
                set(data.files)
            ):

                return False

            motion_magnitude = data["motion_magnitude"]

            frame_idx = data["frame_idx"]

            # [N, H, W]
            if motion_magnitude.ndim != 3:

                return False

            if motion_magnitude.shape[0] != len(
                frame_idx
            ):

                return False

        return True

    except Exception:

        return False


# ============================================================
# VIDEO DISCOVERY
# ============================================================

VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
}


def discover_videos():

    videos = []

    for path in INPUT_ROOT.rglob("*"):

        if not path.is_file():

            continue

        if path.suffix.lower() not in VIDEO_EXTENSIONS:

            continue

        videos.append(path)

    videos.sort()

    return videos


# ============================================================
# MIRRORED OUTPUT PATH
# ============================================================

def get_relative_video_path(video_path):

    return video_path.relative_to(
        INPUT_ROOT
    )


def get_output_npz_path(video_path):

    relative = get_relative_video_path(
        video_path
    )

    return (
        MOTION_ROOT /
        relative.parent /
        f"{relative.stem}.npz"
    )


def get_diagnostic_video_path(video_path):

    relative = get_relative_video_path(
        video_path
    )

    return (
        DIAGNOSTIC_VIDEO_ROOT /
        relative.parent /
        f"{relative.stem}_motion.mp4"
    )


# ============================================================
# VIDEO LOADING
# ============================================================

def load_video(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps is None or fps <= 0:

        fps = 25.0

    frames = []

    while True:

        ok, frame = cap.read()

        if not ok:

            break

        frames.append(frame)

    cap.release()

    if len(frames) < GAP + 1:

        raise RuntimeError(
            f"Video has too few frames: "
            f"{len(frames)}"
        )

    return frames, float(fps)


# ============================================================
# RAFT MODEL
# ============================================================

def load_raft():

    logging.info(
        f"Loading RAFT on {DEVICE}"
    )

    weights = (
        Raft_Large_Weights.DEFAULT
    )

    model = raft_large(
        weights=weights,
        progress=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    transforms = weights.transforms()

    logging.info(
        "RAFT model loaded."
    )

    return model, transforms


# ============================================================
# FRAME → RAFT INPUT
# ============================================================

def frame_to_tensor(frame):

    # OpenCV BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    tensor = torch.from_numpy(
        rgb
    )

    tensor = tensor.permute(
        2, 0, 1
    )

    tensor = tensor.contiguous()

    return tensor


# ============================================================
# RAFT FLOW
# ============================================================

@torch.inference_mode()
def calculate_raft_flow(
    model,
    transforms,
    frame1,
    frame2
):

    img1 = frame_to_tensor(
        frame1
    )

    img2 = frame_to_tensor(
        frame2
    )

    img1 = img1.unsqueeze(0)

    img2 = img2.unsqueeze(0)

    img1, img2 = transforms(
        img1,
        img2
    )

    img1 = img1.to(
        DEVICE
    )

    img2 = img2.to(
        DEVICE
    )

    predictions = model(
        img1,
        img2
    )

    flow = predictions[-1]

    # [1, 2, H, W]
    flow = flow[0]

    flow = flow.detach()

    flow = flow.cpu().numpy()

    # [2, H, W]
    return flow.astype(
        np.float32
    )


# ============================================================
# BATCHED RAFT FLOW
# ============================================================

@torch.inference_mode()
def calculate_raft_flow_batch(
    model,
    transforms,
    frames1,
    frames2
):
    """
    Calculate RAFT flow for multiple frame pairs at once.

    The original single-pair RAFT calculation is unchanged.
    This function only batches the same operations together on the GPU.

    For each pair:
        frames1[i] -> frames2[i]

    Forward and backward pairs are passed through the model together
    by the caller, so the GPU does fewer model launches.
    """

    if len(frames1) != len(frames2):
        raise ValueError(
            "frames1 and frames2 must have the same length."
        )

    if len(frames1) == 0:
        return []

    tensors1 = []
    tensors2 = []

    for frame1, frame2 in zip(frames1, frames2):

        img1 = frame_to_tensor(frame1)
        img2 = frame_to_tensor(frame2)

        tensors1.append(img1)
        tensors2.append(img2)

    img1 = torch.stack(tensors1, dim=0)
    img2 = torch.stack(tensors2, dim=0)

    img1, img2 = transforms(img1, img2)

    img1 = img1.to(DEVICE)
    img2 = img2.to(DEVICE)

    predictions = model(
        img1,
        img2
    )

    flows = predictions[-1]

    # [B, 2, H, W] -> list of [2, H, W]
    flows = flows.detach().cpu().numpy()

    return [
        flow.astype(np.float32)
        for flow in flows
    ]


# ============================================================
# WARP FLOW
# ============================================================

def warp_flow(
    flow,
    displacement
):

    """
    Warp a flow field using another flow.

    Used for forward-backward consistency.
    """

    _, h, w = flow.shape

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    map_x = (
        grid_x +
        displacement[0]
    )

    map_y = (
        grid_y +
        displacement[1]
    )

    warped_u = cv2.remap(
        flow[0],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    warped_v = cv2.remap(
        flow[1],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return np.stack(
        [warped_u, warped_v],
        axis=0
    )


# ============================================================
# FORWARD-BACKWARD CONSISTENCY
# ============================================================

def forward_backward_mask(
    forward_flow,
    backward_flow,
    threshold=FB_THRESHOLD
):

    warped_backward = warp_flow(
        backward_flow,
        forward_flow
    )

    consistency = (
        forward_flow +
        warped_backward
    )

    error = np.sqrt(
        consistency[0] ** 2 +
        consistency[1] ** 2
    )

    valid = (
        error <= threshold
    )

    return valid, error


# ============================================================
# CAMERA MOTION ESTIMATION
# ============================================================

def estimate_camera_motion(
    flow,
    valid_mask
):

    h, w = flow.shape[1:]

    ys, xs = np.where(
        valid_mask
    )

    if len(xs) < MIN_RANSAC_POINTS:

        return None, None, 0.0

    # Randomly sample points if necessary.
    if len(xs) > RANSAC_POINTS:

        rng = np.random.default_rng(
            0
        )

        indices = rng.choice(
            len(xs),
            RANSAC_POINTS,
            replace=False
        )

        xs = xs[indices]
        ys = ys[indices]

    src = np.stack(
        [xs, ys],
        axis=1
    ).astype(
        np.float32
    )

    u = flow[
        0,
        ys,
        xs
    ]

    v = flow[
        1,
        ys,
        xs
    ]

    dst = np.stack(
        [
            xs + u,
            ys + v
        ],
        axis=1
    ).astype(
        np.float32
    )

    if CAMERA_MODEL == "affine":

        matrix, inlier_mask = (
            cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=(
                    RANSAC_REPROJECTION_THRESHOLD
                ),
                maxIters=2000,
                confidence=0.99,
            )
        )

        if matrix is None:

            return None, None, 0.0

    else:

        raise ValueError(
            f"Unsupported camera model: "
            f"{CAMERA_MODEL}"
        )

    inlier_ratio = float(
        np.mean(
            inlier_mask.astype(
                np.float32
            )
        )
    )

    if (
        inlier_ratio <
        MIN_INLIER_RATIO
    ):

        return (
            None,
            inlier_mask,
            inlier_ratio
        )

    # Calculate global camera displacement
    # at every pixel.
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    ones = np.ones_like(
        grid_x
    )

    points = np.stack(
        [
            grid_x,
            grid_y,
            ones,
        ],
        axis=0
    )

    transformed = np.einsum(
        "ij,jhw->ihw",
        matrix,
        points
    )

    camera_u = (
        transformed[0] -
        grid_x
    )

    camera_v = (
        transformed[1] -
        grid_y
    )

    camera_flow = np.stack(
        [
            camera_u,
            camera_v
        ],
        axis=0
    )

    return (
        camera_flow,
        inlier_mask,
        inlier_ratio
    )


# ============================================================
# CAMERA COMPENSATION
# ============================================================

def compensate_camera_motion(
    flow,
    valid_mask
):

    camera_flow, inliers, inlier_ratio = (
        estimate_camera_motion(
            flow,
            valid_mask
        )
    )

    if camera_flow is None:

        return (
            flow.copy(),
            None,
            False,
            inlier_ratio
        )

    compensated = (
        flow -
        camera_flow
    )

    return (
        compensated,
        camera_flow,
        True,
        inlier_ratio
    )


# ============================================================
# TEMPORAL MEDIAN SMOOTHING
# ============================================================

def temporal_median_smooth(
    flows,
    window=MEDIAN_WINDOW
):

    if len(flows) == 0:

        return flows

    if window <= 1:

        return flows.copy()

    radius = window // 2

    result = np.zeros_like(
        flows,
        dtype=np.float32
    )

    n = len(flows)

    for i in range(n):

        start = max(
            0,
            i - radius
        )

        end = min(
            n,
            i + radius + 1
        )

        result[i] = np.median(
            flows[start:end],
            axis=0
        )

    return result


# ============================================================
# NOISE FLOOR
# ============================================================

def apply_noise_floor(
    flow,
    threshold=NOISE_FLOOR
):

    magnitude = np.sqrt(
        flow[0] ** 2 +
        flow[1] ** 2
    )

    mask = (
        magnitude >= threshold
    )

    cleaned = flow.copy()

    cleaned[
        :,
        ~mask
    ] = 0.0

    return cleaned


# ============================================================
# FLOW MAGNITUDE
# ============================================================

def flow_magnitude(flow):

    return np.sqrt(
        flow[0] ** 2 +
        flow[1] ** 2
    )


# ============================================================
# FLOW COLOR VISUALIZATION
# ============================================================

def flow_to_color(
    flow,
    magnitude_scale
):

    u = flow[0]

    v = flow[1]

    magnitude = np.sqrt(
        u * u +
        v * v
    )

    angle = np.arctan2(
        v,
        u
    )

    hue = (
        angle + np.pi
    ) / (
        2 * np.pi
    )

    hue = (
        hue * 179
    ).astype(
        np.uint8
    )

    saturation = np.full_like(
        hue,
        255,
        dtype=np.uint8
    )

    value = np.clip(
        magnitude /
        max(magnitude_scale, 1e-6) *
        255.0,
        0,
        255
    ).astype(
        np.uint8
    )

    hsv = np.stack(
        [
            hue,
            saturation,
            value
        ],
        axis=2
    )

    bgr = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    return bgr


# ============================================================
# DRAW FLOW ARROWS
# ============================================================

def draw_flow_arrows(
    image,
    flow,
    step=ARROW_STEP
):

    output = image.copy()

    h, w = flow.shape[1:]

    count = 0

    for y in range(
        step // 2,
        h,
        step
    ):

        for x in range(
            step // 2,
            w,
            step
        ):

            u = float(
                flow[0, y, x]
            )

            v = float(
                flow[1, y, x]
            )

            magnitude = np.sqrt(
                u * u +
                v * v
            )

            if magnitude < NOISE_FLOOR:

                continue

            x2 = int(
                round(x + u)
            )

            y2 = int(
                round(y + v)
            )

            x2 = np.clip(
                x2,
                0,
                w - 1
            )

            y2 = np.clip(
                y2,
                0,
                h - 1
            )

            cv2.arrowedLine(
                output,
                (x, y),
                (x2, y2),
                (255, 255, 255),
                1,
                tipLength=0.3,
            )

            count += 1

            if count >= MAX_ARROWS:

                return output

    return output


# ============================================================
# PANEL HELPERS
# ============================================================

def add_title(
    image,
    title
):

    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (
            output.shape[1],
            30
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        title,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


def add_diagnostics(
    image,
    values
):

    output = image.copy()

    y = 25

    for key, value in values.items():

        text = (
            f"{key}: {value}"
        )

        cv2.putText(
            output,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 20

    return output


# ============================================================
# DIAGNOSTIC VIDEO
# ============================================================

def create_diagnostic_video(
    frames,
    raw_flows,
    compensated_flows,
    clean_flows,
    diagnostics,
    fps,
    output_path
):

    if not SAVE_DIAGNOSTIC_VIDEOS:

        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    h, w = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (
            w * 2,
            h * 2
        )
    )

    if not writer.isOpened():

        logging.warning(
            f"Could not create diagnostic video: "
            f"{output_path}"
        )

        return

    # Fixed scale over the complete video.
    all_magnitudes = []

    for flow in raw_flows:

        all_magnitudes.append(
            flow_magnitude(flow)
        )

    if all_magnitudes:

        all_magnitudes = np.concatenate(
            [
                x.reshape(-1)
                for x in all_magnitudes
            ]
        )

        raw_p98 = float(
            np.percentile(
                all_magnitudes,
                98
            )
        )

    else:

        raw_p98 = 1.0

    for i in range(
        len(raw_flows)
    ):

        frame_index = i

        base_frame = frames[
            min(
                frame_index,
                len(frames) - 1
            )
        ]

        raw = raw_flows[i]

        compensated = (
            compensated_flows[i]
        )

        clean = clean_flows[i]

        raw_vis = flow_to_color(
            raw,
            raw_p98
        )

        clean_vis = flow_to_color(
            clean,
            raw_p98
        )

        compensated_vis = flow_to_color(
            compensated,
            raw_p98
        )

        clean_overlay = (
            base_frame.copy()
        )

        clean_overlay = (
            draw_flow_arrows(
                clean_overlay,
                clean
            )
        )

        raw_vis = draw_flow_arrows(
            raw_vis,
            raw
        )

        compensated_vis = (
            draw_flow_arrows(
                compensated_vis,
                compensated
            )
        )

        panel_a = add_title(
            clean_overlay,
            "A: Input + Cleaned Motion"
        )

        panel_b = add_title(
            raw_vis,
            "B: Raw RAFT Flow"
        )

        panel_c = add_title(
            compensated_vis,
            "C: Camera-Compensated Flow"
        )

        diag = diagnostics[i]

        diag_image = np.zeros_like(
            base_frame
        )

        diag_image = add_diagnostics(
            diag_image,
            {
                "Frame": diag["frame_index"],
                "FB valid %":
                    f'{diag["fb_valid_pct"]:.1f}',
                "Inliers %":
                    f'{diag["inlier_ratio"] * 100:.1f}',
                "Camera":
                    diag["camera_compensation"],
                "dx":
                    f'{diag["camera_dx"]:.3f}',
                "dy":
                    f'{diag["camera_dy"]:.3f}',
                "p98 raw":
                    f'{diag["p98_raw"]:.3f}',
                "Active %":
                    f'{diag["active_frac"] * 100:.2f}',
            }
        )

        panel_d = add_title(
            diag_image,
            "D: Diagnostics"
        )

        top = np.hstack(
            [
                panel_a,
                panel_b
            ]
        )

        bottom = np.hstack(
            [
                panel_c,
                panel_d
            ]
        )

        combined = np.vstack(
            [
                top,
                bottom
            ]
        )

        writer.write(
            combined
        )

    writer.release()


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    model,
    transforms
):

    global CURRENT_VIDEO

    CURRENT_VIDEO = video_path

    class_name = (
        video_path.parent.name
    )

    output_npz = get_output_npz_path(
        video_path
    )

    diagnostic_path = (
        get_diagnostic_video_path(
            video_path
        )
    )

    output_npz.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    diagnostic_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # RESUME CHECK
    # --------------------------------------------------------

    if validate_npz(
        output_npz
    ):

        logging.info(
            f"SKIP: {video_path}"
        )

        append_result(
            video_path,
            class_name,
            "skipped",
            0.0,
            output_npz,
            diagnostic_path,
        )

        return "skipped", 0.0

    # --------------------------------------------------------
    # START TIMING
    # --------------------------------------------------------

    start = time.perf_counter()

    logging.info(
        f"START: {video_path}"
    )

    save_state(
        "processing",
        video_path,
        now_string()
    )

    try:

        # ----------------------------------------------------
        # LOAD VIDEO
        # ----------------------------------------------------

        frames, fps = load_video(
            video_path
        )

        num_frames = len(
            frames
        )

        num_pairs = (
            num_frames - GAP
        )

        logging.info(
            f"Frames: {num_frames} | "
            f"Pairs: {num_pairs} | "
            f"FPS: {fps:.2f}"
        )

        raw_flows = []

        compensated_flows = []

        fb_masks = []

        diagnostics = []

        # ----------------------------------------------------
        # FLOW EXTRACTION
        # ----------------------------------------------------

        for batch_start in range(
            0,
            num_pairs,
            RAFT_BATCH_SIZE
        ):

            if STOP_REQUESTED:
                raise KeyboardInterrupt

            batch_end = min(
                batch_start + RAFT_BATCH_SIZE,
                num_pairs
            )

            batch_indices = list(
                range(
                    batch_start,
                    batch_end
                )
            )

            batch_frames1 = [
                frames[i]
                for i in batch_indices
            ]

            batch_frames2 = [
                frames[i + GAP]
                for i in batch_indices
            ]

            # ------------------------------------------------
            # Forward + backward RAFT inference
            #
            # Both directions are combined into one larger
            # batch. This keeps the original forward/backward
            # algorithm while reducing GPU model launches.
            # ------------------------------------------------

            forward_frames1 = batch_frames1
            forward_frames2 = batch_frames2

            backward_frames1 = batch_frames2
            backward_frames2 = batch_frames1

            combined_frames1 = (
                forward_frames1 +
                backward_frames1
            )

            combined_frames2 = (
                forward_frames2 +
                backward_frames2
            )

            combined_flows = calculate_raft_flow_batch(
                model,
                transforms,
                combined_frames1,
                combined_frames2
            )

            batch_count = len(batch_indices)

            forward_flows = (
                combined_flows[:batch_count]
            )

            backward_flows = (
                combined_flows[batch_count:]
            )

            # ------------------------------------------------
            # Preserve the original per-frame-pair processing
            # exactly after RAFT inference.
            # ------------------------------------------------

            for local_index, i in enumerate(
                batch_indices
            ):

                forward = forward_flows[local_index]
                backward = backward_flows[local_index]

                # -----------------------------------------------
                # FB consistency
                # -----------------------------------------------

                valid_mask, fb_error = (
                    forward_backward_mask(
                        forward,
                        backward
                    )
                )

                fb_valid_pct = (
                    float(
                        np.mean(valid_mask)
                    ) * 100.0
                )

                filtered = forward.copy()

                filtered[
                    :,
                    ~valid_mask
                ] = 0.0

                # -----------------------------------------------
                # Camera compensation
                # -----------------------------------------------

                (
                    compensated,
                    camera_flow,
                    comp_on,
                    inlier_ratio,
                ) = compensate_camera_motion(
                    filtered,
                    valid_mask
                )

                # -----------------------------------------------
                # Camera diagnostics
                # -----------------------------------------------

                if camera_flow is not None:

                    camera_dx = float(
                        np.median(
                            camera_flow[0]
                        )
                    )

                    camera_dy = float(
                        np.median(
                            camera_flow[1]
                        )
                    )

                else:

                    camera_dx = 0.0

                    camera_dy = 0.0

                # -----------------------------------------------
                # Raw p98
                # -----------------------------------------------

                raw_magnitude = (
                    flow_magnitude(
                        forward
                    )
                )

                p98_raw = float(
                    np.percentile(
                        raw_magnitude,
                        98
                    )
                )

                # -----------------------------------------------
                # Active fraction before final smoothing
                # -----------------------------------------------

                compensated_magnitude = (
                    flow_magnitude(
                        compensated
                    )
                )

                active_frac = float(
                    np.mean(
                        compensated_magnitude
                        >= NOISE_FLOOR
                    )
                )

                raw_flows.append(
                    forward
                )

                compensated_flows.append(
                    compensated
                )

                fb_masks.append(
                    valid_mask
                )

                diagnostics.append({
                    "frame_index": i,
                    "fb_valid_pct":
                        fb_valid_pct,
                    "inlier_ratio":
                        inlier_ratio,
                    "camera_compensation":
                        "ON" if comp_on else "OFF",
                    "camera_dx":
                        camera_dx,
                    "camera_dy":
                        camera_dy,
                    "p98_raw":
                        p98_raw,
                    "active_frac":
                        active_frac,
                })

            logging.info(
                f"{video_path.name}: "
                f"{batch_end}/{num_pairs} "
                f"flow pairs "
                f"(batch {len(batch_indices)})"
            )

        # ----------------------------------------------------
        # TEMPORAL SMOOTHING
        # ----------------------------------------------------

        compensated_array = np.stack(
            compensated_flows,
            axis=0
        )

        smoothed = (
            temporal_median_smooth(
                compensated_array,
                MEDIAN_WINDOW
            )
        )

        # ----------------------------------------------------
        # NOISE FLOOR
        # ----------------------------------------------------

        clean_flows = []

        for flow in smoothed:

            clean = apply_noise_floor(
                flow,
                NOISE_FLOOR
            )

            clean_flows.append(
                clean
            )

        clean_array = np.stack(
            clean_flows,
            axis=0
        )

        frame_index = np.arange(
            num_pairs,
            dtype=np.int32
        )

        # ----------------------------------------------------
        # MOTION MAGNITUDE  [N, H, W]
        # ----------------------------------------------------

        motion_magnitude = np.sqrt(
            clean_array[:, 0] ** 2 +
            clean_array[:, 1] ** 2
        )

        # ----------------------------------------------------
        # SAVE NPZ ATOMICALLY
        # ----------------------------------------------------

        temp_npz = output_npz.with_suffix(
            ".npz.tmp"
        )

        save_function = (
            np.savez_compressed
            if COMPRESS_NPZ
            else np.savez
        )

        save_function(
            temp_npz,
            motion_magnitude=motion_magnitude.astype(
                SAVE_DTYPE
            ),
            frame_idx=frame_index,
            gap=np.int32(GAP),
            fps=np.float32(fps),
        )

        # np.savez may append ".npz" to the filename.
        generated_temp = Path(
            str(temp_npz) + ".npz"
        )

        if generated_temp.exists():

            temp_npz = generated_temp

        os.replace(
            temp_npz,
            output_npz
        )

        # ----------------------------------------------------
        # VERIFY OUTPUT
        # ----------------------------------------------------

        if not validate_npz(
            output_npz
        ):

            raise RuntimeError(
                "NPZ was written but failed validation."
            )

        # ----------------------------------------------------
        # DIAGNOSTIC VIDEO
        # ----------------------------------------------------

        if SAVE_DIAGNOSTIC_VIDEOS:

            create_diagnostic_video(
                frames=frames,
                raw_flows=raw_flows,
                compensated_flows=compensated_flows,
                clean_flows=clean_flows,
                diagnostics=diagnostics,
                fps=(
                    DIAGNOSTIC_FPS
                    if DIAGNOSTIC_FPS
                    else fps
                ),
                output_path=diagnostic_path,
            )

        # ----------------------------------------------------
        # END TIMING
        # ----------------------------------------------------

        duration = (
            time.perf_counter() -
            start
        )

        logging.info(
            f"COMPLETE: {video_path} | "
            f"{seconds_to_hms(duration)}"
        )

        append_result(
            video_path,
            class_name,
            "completed",
            duration,
            output_npz,
            diagnostic_path,
        )

        save_state(
            "idle"
        )

        CURRENT_VIDEO = None

        return "completed", duration

    except KeyboardInterrupt:

        duration = (
            time.perf_counter() -
            start
        )

        logging.warning(
            f"INTERRUPTED: {video_path}"
        )

        append_result(
            video_path,
            class_name,
            "interrupted",
            duration,
            output_npz,
            diagnostic_path,
            "KeyboardInterrupt",
        )

        save_state(
            "interrupted",
            video_path
        )

        CURRENT_VIDEO = None

        raise

    except Exception as exc:

        duration = (
            time.perf_counter() -
            start
        )

        error_text = (
            f"{type(exc).__name__}: "
            f"{str(exc)}"
        )

        logging.error(
            f"FAILED: {video_path}"
        )

        logging.error(
            traceback.format_exc()
        )

        append_result(
            video_path,
            class_name,
            "failed",
            duration,
            output_npz,
            diagnostic_path,
            error_text,
        )

        save_state(
            "idle"
        )

        CURRENT_VIDEO = None

        return "failed", duration


# ============================================================
# HANDLE CTRL+C
# ============================================================

def signal_handler(
    signum,
    frame
):

    global STOP_REQUESTED

    STOP_REQUESTED = True

    logging.warning(
        "Stop requested. "
        "Finishing current operation..."
    )


# ============================================================
# PREVIOUS INTERRUPTED RUN
# ============================================================

def check_previous_state():

    state = load_state()

    if not state:

        return

    if state.get(
        "status"
    ) != "processing":

        return

    previous_video = state.get(
        "video"
    )

    if not previous_video:

        return

    previous_path = Path(
        previous_video
    )

    output_path = (
        get_output_npz_path(
            previous_path
        )
    )

    if validate_npz(
        output_path
    ):

        return

    logging.warning(
        "Previous run appears to have "
        "stopped while processing:"
    )

    logging.warning(
        f"    {previous_path}"
    )

    logging.warning(
        "The video will be retried."
    )

    append_result(
        previous_path,
        previous_path.parent.name,
        "interrupted",
        0.0,
        output_path,
        get_diagnostic_video_path(
            previous_path
        ),
        "Previous run ended before NPZ completion.",
    )


# ============================================================
# SUMMARY
# ============================================================

def generate_summary(
    run_start,
    run_end,
    run_wall_time
):

    rows = load_results()

    # --------------------------------------------------------
    # Count statuses
    # --------------------------------------------------------

    counts = {
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "interrupted": 0,
    }

    durations = []

    for row in rows:

        status = row.get(
            "status",
            ""
        )

        if status in counts:

            counts[status] += 1

        try:

            duration = float(
                row.get(
                    "duration_sec",
                    0
                )
            )

            if status in {
                "completed",
                "failed",
                "interrupted",
            }:

                durations.append(
                    duration
                )

        except Exception:

            pass

    completed_durations = []

    for row in rows:

        if row.get(
            "status"
        ) != "completed":

            continue

        try:

            completed_durations.append(
                float(
                    row[
                        "duration_sec"
                    ]
                )
            )

        except Exception:

            pass

    total_accumulated = sum(
        durations
    )

    total_completed_time = sum(
        completed_durations
    )

    average_completed = (
        np.mean(
            completed_durations
        )
        if completed_durations
        else 0.0
    )

    # --------------------------------------------------------
    # Summary object
    # --------------------------------------------------------

    summary = {
        "generated_at":
            now_string(),

        "dataset":
            str(INPUT_ROOT),

        "output_root":
            str(OUTPUT_ROOT),

        "run_start":
            run_start,

        "run_end":
            run_end,

        "current_run_wall_time_sec":
            run_wall_time,

        "current_run_wall_time":
            seconds_to_hms(
                run_wall_time
            ),

        "videos_found":
            len(
                discover_videos()
            ),

        "result_records":
            len(rows),

        "status_counts":
            counts,

        "total_accumulated_video_processing_sec":
            total_accumulated,

        "total_accumulated_video_processing_time":
            seconds_to_hms(
                total_accumulated
            ),

        "total_successful_processing_sec":
            total_completed_time,

        "total_successful_processing_time":
            seconds_to_hms(
                total_completed_time
            ),

        "average_completed_video_sec":
            average_completed,

        "average_completed_video_time":
            seconds_to_hms(
                average_completed
            ),

        "parameters": {
            "gap": GAP,
            "raft_batch_size": RAFT_BATCH_SIZE,
            "fb_threshold":
                FB_THRESHOLD,
            "camera_model":
                CAMERA_MODEL,
            "ransac_points":
                RANSAC_POINTS,
            "min_ransac_points":
                MIN_RANSAC_POINTS,
            "min_inlier_ratio":
                MIN_INLIER_RATIO,
            "ransac_reprojection_threshold":
                RANSAC_REPROJECTION_THRESHOLD,
            "median_window":
                MEDIAN_WINDOW,
            "noise_floor":
                NOISE_FLOOR,
            "save_dtype":
                str(SAVE_DTYPE),
            "compressed_npz":
                COMPRESS_NPZ,
        },

        "device":
            DEVICE,
    }

    with open(
        SUMMARY_FILE,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Human-readable terminal summary
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("RAFT MOTION EXTRACTION SUMMARY")
    print("=" * 65)

    print(
        f"Dataset                : "
        f"{INPUT_ROOT}"
    )

    print(
        f"Videos found           : "
        f"{summary['videos_found']}"
    )

    print(
        f"Completed              : "
        f"{counts['completed']}"
    )

    print(
        f"Skipped                : "
        f"{counts['skipped']}"
    )

    print(
        f"Failed                 : "
        f"{counts['failed']}"
    )

    print(
        f"Interrupted             : "
        f"{counts['interrupted']}"
    )

    print()

    print(
        f"Current run wall time  : "
        f"{seconds_to_hms(run_wall_time)}"
    )

    print(
        f"Accumulated processing : "
        f"{seconds_to_hms(total_accumulated)}"
    )

    print(
        f"Successful processing  : "
        f"{seconds_to_hms(total_completed_time)}"
    )

    print(
        f"Average successful     : "
        f"{seconds_to_hms(average_completed)}"
    )

    print()

    print(
        f"Motion NPZ directory   : "
        f"{MOTION_ROOT}"
    )

    print(
        f"Diagnostics directory  : "
        f"{DIAGNOSTICS_ROOT}"
    )

    print(
        f"Summary                : "
        f"{SUMMARY_FILE}"
    )

    print("=" * 65)
    print()


# ============================================================
# RUN HISTORY
# ============================================================

def save_run_history(
    run_start,
    run_end,
    wall_time
):

    history = []

    if RUN_HISTORY_FILE.exists():

        try:

            with open(
                RUN_HISTORY_FILE,
                "r"
            ) as f:

                history = json.load(
                    f
                )

        except Exception:

            history = []

    history.append({
        "run_start":
            run_start,

        "run_end":
            run_end,

        "wall_time_sec":
            wall_time,

        "wall_time":
            seconds_to_hms(
                wall_time
            ),
    })

    with open(
        RUN_HISTORY_FILE,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global STOP_REQUESTED

    create_directories()

    setup_logging()

    initialize_results_csv()

    signal.signal(
        signal.SIGINT,
        signal_handler
    )

    signal.signal(
        signal.SIGTERM,
        signal_handler
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not INPUT_ROOT.exists():

        raise FileNotFoundError(
            f"INPUT_ROOT does not exist:\n"
            f"{INPUT_ROOT}"
        )

    logging.info(
        "=" * 65
    )

    logging.info(
        "RAFT UCF-101 MOTION EXTRACTION"
    )

    logging.info(
        "=" * 65
    )

    logging.info(
        f"Input: {INPUT_ROOT}"
    )

    logging.info(
        f"Output: {OUTPUT_ROOT}"
    )

    logging.info(
        f"Device: {DEVICE}"
    )

    logging.info(
        f"Gap: {GAP}"
    )

    logging.info(
        f"Noise floor: {NOISE_FLOOR}"
    )

    logging.info(
        f"RAFT batch size: {RAFT_BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # Check previous interrupted run
    # --------------------------------------------------------

    check_previous_state()

    # --------------------------------------------------------
    # Find videos
    # --------------------------------------------------------

    videos = discover_videos()

    if not videos:

        logging.warning(
            "No videos found."
        )

        return

    logging.info(
        f"Found {len(videos)} videos."
    )

    # --------------------------------------------------------
    # Start run timer
    # --------------------------------------------------------

    run_start_time = (
        time.perf_counter()
    )

    run_start_string = now_string()

    save_state(
        "running",
        None,
        run_start_string
    )

    # --------------------------------------------------------
    # Load RAFT
    # --------------------------------------------------------

    model, transforms = load_raft()

    # --------------------------------------------------------
    # Process videos
    # --------------------------------------------------------

    try:

        for index, video_path in enumerate(
            videos,
            start=1
        ):

            if STOP_REQUESTED:

                logging.warning(
                    "Stop requested before next video."
                )

                break

            logging.info(
                ""
            )

            logging.info(
                f"[{index}/{len(videos)}] "
                f"{video_path}"
            )

            status, duration = process_video(
                video_path,
                model,
                transforms
            )

            logging.info(
                f"Status: {status} | "
                f"Duration: "
                f"{seconds_to_hms(duration)}"
            )

    except KeyboardInterrupt:

        logging.warning(
            "Processing interrupted by user."
        )

    finally:

        # ----------------------------------------------------
        # Run timing
        # ----------------------------------------------------

        run_end_time = (
            time.perf_counter()
        )

        run_wall_time = (
            run_end_time -
            run_start_time
        )

        run_end_string = now_string()

        save_state(
            "idle"
        )

        # ----------------------------------------------------
        # Save run history
        # ----------------------------------------------------

        save_run_history(
            run_start_string,
            run_end_string,
            run_wall_time
        )

        # ----------------------------------------------------
        # Generate summary
        # ----------------------------------------------------

        generate_summary(
            run_start_string,
            run_end_string,
            run_wall_time
        )

        logging.info(
            "Run finished."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()