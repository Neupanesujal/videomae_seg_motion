
import numpy as np
import csv
import os
import traceback

from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

NPZ_ROOT = Path(
    "raft/output_datasets_motion/motion_npz"
)

OUTPUT_ROOT = Path(
    "raft/output_datasets_motion/motion_quality_analysis"
)

CATEGORY_DIR = OUTPUT_ROOT / "categories"

NOISE_FLOOR = 0.3

# Initial screening thresholds.
# These will be tuned after examining the distributions.

LOW_P95_THRESHOLD = 0.5
LOW_ACTIVE_RATIO = 0.05

SPARSE_ACTIVE_RATIO = 5.0

HIGH_P99_THRESHOLD = 20.0

FRAME_VARIATION_THRESHOLD = 3.0

# Number of concurrent NPZ analysis workers.
# Start with 4. Increase to 8 if storage permits.
NUM_WORKERS = min(8, os.cpu_count() or 4)


# ============================================================
# CATEGORY DEFINITIONS
# ============================================================

CATEGORIES = {
    "NEAR_STATIC": "near_static.txt",
    "SPARSE_MOTION": "sparse_motion.txt",
    "HIGH_MOTION_OUTLIER": "high_motion_outlier.txt",
    "FRAME_INCONSISTENT": "frame_inconsistent.txt",
    "NO_OBVIOUS_STATISTICAL_FLAG": "no_obvious_flag.txt",
}


# ============================================================
# ANALYZE ONE NPZ FILE
# ============================================================

def analyze_npz(npz_path):

    with np.load(
        npz_path,
        allow_pickle=False
    ) as data:

        magnitude = data["motion_magnitude"].astype(
            np.float32,
            copy=False
        )

        frame_idx = data["frame_idx"]
        gap = int(data["gap"])
        fps = float(data["fps"])

        # Copy-independent statistics are calculated
        # while the NPZ file is open.

        if magnitude.ndim != 3:
            raise ValueError(
                f"Expected [N,H,W], got {magnitude.shape}"
            )

        if not np.all(np.isfinite(magnitude)):
            raise ValueError(
                "Non-finite motion values found"
            )

        if np.any(magnitude < 0):
            raise ValueError(
                "Negative motion magnitudes found"
            )

        num_frames, height, width = magnitude.shape

        flat = magnitude.reshape(-1)

        # ----------------------------------------------------
        # GLOBAL MOTION STATISTICS
        # ----------------------------------------------------

        mean_motion = float(np.mean(flat))
        median_motion = float(np.median(flat))

        p5 = float(np.percentile(flat, 5))
        p50 = float(np.percentile(flat, 50))
        p95 = float(np.percentile(flat, 95))
        p99 = float(np.percentile(flat, 99))

        maximum = float(np.max(flat))

        zero_ratio = float(
            np.mean(flat == 0) * 100
        )

        active_ratio = float(
            np.mean(flat >= NOISE_FLOOR) * 100
        )

        # ----------------------------------------------------
        # FRAME-LEVEL STATISTICS
        # ----------------------------------------------------

        frame_means = np.mean(
            magnitude,
            axis=(1, 2)
        )

        frame_p95 = np.percentile(
            magnitude,
            95,
            axis=(1, 2)
        )

        frame_mean_std = float(
            np.std(frame_means)
        )

        frame_mean_cv = float(
            frame_mean_std /
            (np.mean(frame_means) + 1e-8)
        )

        frame_p95_std = float(
            np.std(frame_p95)
        )

        frame_active_ratios = np.mean(
            magnitude >= NOISE_FLOOR,
            axis=(1, 2)
        ) * 100

        inactive_frames = float(
            np.mean(
                frame_active_ratios < 0.01
            ) * 100
        )

        active_frame_ratio = float(
            np.mean(
                frame_active_ratios >= 0.01
            ) * 100
        )

        # ----------------------------------------------------
        # ADDITIONAL TEMPORAL STATISTICS
        # ----------------------------------------------------

        frame_mean_min = float(
            np.min(frame_means)
        )

        frame_mean_max = float(
            np.max(frame_means)
        )

        frame_mean_p95 = float(
            np.percentile(frame_means, 95)
        )

        # ----------------------------------------------------
        # RETURN RESULTS
        # ----------------------------------------------------

        return {
            "video": npz_path.stem,
            "class": npz_path.parent.name,
            "path": str(npz_path.relative_to(NPZ_ROOT)),

            "num_frames": num_frames,
            "height": height,
            "width": width,
            "gap": gap,
            "fps": fps,

            "mean": mean_motion,
            "median": median_motion,
            "p5": p5,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "maximum": maximum,

            "zero_ratio": zero_ratio,
            "active_ratio": active_ratio,

            "frame_mean_std": frame_mean_std,
            "frame_mean_cv": frame_mean_cv,
            "frame_p95_std": frame_p95_std,

            "inactive_frames": inactive_frames,
            "active_frame_ratio": active_frame_ratio,

            "frame_mean_min": frame_mean_min,
            "frame_mean_max": frame_mean_max,
            "frame_mean_p95": frame_mean_p95,
        }


# ============================================================
# VIDEO CATEGORIZATION
# ============================================================

def categorize_video(s):

    categories = []

    # 1. Near-static
    if (
        s["p95"] < LOW_P95_THRESHOLD
        and s["active_ratio"] < LOW_ACTIVE_RATIO
    ):
        categories.append("NEAR_STATIC")

    # 2. Sparse motion
    elif s["active_ratio"] < SPARSE_ACTIVE_RATIO:
        categories.append("SPARSE_MOTION")

    # 3. High magnitude outlier
    if s["p99"] > HIGH_P99_THRESHOLD:
        categories.append("HIGH_MOTION_OUTLIER")

    # 4. Frame-level inconsistency
    if (
        s["frame_mean_cv"] > FRAME_VARIATION_THRESHOLD
        and s["mean"] > 0
    ):
        categories.append("FRAME_INCONSISTENT")

    # 5. No obvious statistical warning
    if not categories:
        categories.append(
            "NO_OBVIOUS_STATISTICAL_FLAG"
        )

    return categories


# ============================================================
# CSV WRITER
# ============================================================

def write_csv(path, rows, fields):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================
# CATEGORY REPORTS
# ============================================================

def write_category_reports(results):

    grouped = defaultdict(list)

    for item in results:

        for category in item["categories"]:

            grouped[category].append(item)

    for category, filename in CATEGORIES.items():

        output_path = CATEGORY_DIR / filename

        videos = grouped[category]

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("=" * 80 + "\n")
            f.write(f"{category}\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"Total videos: {len(videos)}\n\n"
            )

            for item in sorted(
                videos,
                key=lambda x: x["path"]
            ):

                f.write(
                    f"{item['path']}\n"
                )

    return grouped


# ============================================================
# CLASS-LEVEL STATISTICS
# ============================================================

def calculate_class_statistics(results):

    class_groups = defaultdict(list)

    for item in results:
        class_groups[item["class"]].append(item)

    class_rows = []

    for class_name, videos in sorted(
        class_groups.items()
    ):

        class_rows.append({

            "class": class_name,

            "video_count": len(videos),

            "mean_motion": np.mean([
                x["mean"] for x in videos
            ]),

            "median_video_p95": np.median([
                x["p95"] for x in videos
            ]),

            "mean_p95": np.mean([
                x["p95"] for x in videos
            ]),

            "mean_p99": np.mean([
                x["p99"] for x in videos
            ]),

            "mean_active_ratio": np.mean([
                x["active_ratio"] for x in videos
            ]),

            "mean_zero_ratio": np.mean([
                x["zero_ratio"] for x in videos
            ]),

            "near_static_count": sum(
                "NEAR_STATIC" in x["categories"]
                for x in videos
            ),

            "sparse_motion_count": sum(
                "SPARSE_MOTION" in x["categories"]
                for x in videos
            ),

            "high_motion_count": sum(
                "HIGH_MOTION_OUTLIER" in x["categories"]
                for x in videos
            ),

            "inconsistent_count": sum(
                "FRAME_INCONSISTENT" in x["categories"]
                for x in videos
            ),
        })

    return class_rows


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    if not NPZ_ROOT.exists():
        raise FileNotFoundError(
            f"NPZ directory not found: {NPZ_ROOT.resolve()}"
        )

    npz_files = sorted(
        NPZ_ROOT.rglob("*.npz")
    )

    if not npz_files:
        raise RuntimeError(
            "No NPZ files found."
        )

    CATEGORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("RAFT MOTION QUALITY ANALYSIS")
    print("=" * 65)

    print(f"NPZ files: {len(npz_files)}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Output: {OUTPUT_ROOT.resolve()}")
    print()

    results = []
    failed = []

    # --------------------------------------------------------
    # MULTITHREADED ANALYSIS
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=NUM_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                analyze_npz,
                path
            ): path

            for path in npz_files
        }

        completed = 0

        for future in as_completed(futures):

            path = futures[future]

            completed += 1

            try:

                stats = future.result()

                stats["categories"] = (
                    categorize_video(stats)
                )

                results.append(stats)

                print(
                    f"[{completed}/{len(npz_files)}] "
                    f"{stats['video']} | "
                    f"{', '.join(stats['categories'])}"
                )

            except Exception as exc:

                failed.append({
                    "path": str(
                        path.relative_to(NPZ_ROOT)
                    ),
                    "error": str(exc)
                })

                print(
                    f"[{completed}/{len(npz_files)}] "
                    f"FAILED: {path.name}: {exc}"
                )

    # Preserve deterministic ordering
    results.sort(
        key=lambda x: x["path"]
    )

    # --------------------------------------------------------
    # DETAILED VIDEO STATISTICS CSV
    # --------------------------------------------------------

    video_fields = [
        "video", "class", "path",
        "num_frames", "height", "width",
        "gap", "fps",

        "mean", "median",
        "p5", "p50", "p95", "p99",
        "maximum",

        "zero_ratio", "active_ratio",

        "frame_mean_std",
        "frame_mean_cv",
        "frame_p95_std",

        "inactive_frames",
        "active_frame_ratio",

        "frame_mean_min",
        "frame_mean_max",
        "frame_mean_p95",

        "categories",
    ]

    csv_rows = []

    for item in results:

        row = item.copy()

        row["categories"] = ";".join(
            item["categories"]
        )

        csv_rows.append(row)

    write_csv(
        OUTPUT_ROOT / "detailed_video_statistics.csv",
        csv_rows,
        video_fields
    )

    # --------------------------------------------------------
    # CATEGORY TXT FILES
    # --------------------------------------------------------

    grouped = write_category_reports(
        results
    )

    # --------------------------------------------------------
    # CLASS-LEVEL STATISTICS CSV
    # --------------------------------------------------------

    class_rows = calculate_class_statistics(
        results
    )

    class_fields = list(
        class_rows[0].keys()
    ) if class_rows else []

    if class_rows:

        write_csv(
            OUTPUT_ROOT / "class_level_statistics.csv",
            class_rows,
            class_fields
        )

    # --------------------------------------------------------
    # FAILED FILE REPORT
    # --------------------------------------------------------

    with open(
        OUTPUT_ROOT / "failed_files.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Failed files: {len(failed)}\n\n"
        )

        for item in failed:

            f.write(
                f"{item['path']}\n"
                f"Error: {item['error']}\n\n"
            )

    # --------------------------------------------------------
    # SUMMARY REPORT
    # --------------------------------------------------------

    with open(
        OUTPUT_ROOT / "analysis_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 75 + "\n")
        f.write("RAFT MOTION QUALITY ANALYSIS SUMMARY\n")
        f.write("=" * 75 + "\n\n")

        f.write(
            f"Generated: {datetime.now()}\n"
        )

        f.write(
            f"Total files: {len(npz_files)}\n"
        )

        f.write(
            f"Successfully analyzed: {len(results)}\n"
        )

        f.write(
            f"Failed: {len(failed)}\n"
        )

        f.write(
            f"Workers: {NUM_WORKERS}\n"
        )

        f.write(
            f"Noise floor: {NOISE_FLOOR}\n\n"
        )

        f.write("CATEGORY COUNTS\n")
        f.write("-" * 75 + "\n")

        for category in CATEGORIES:

            videos = grouped[category]

            f.write(
                f"{category}: {len(videos)} "
                f"({len(videos) / max(len(results), 1) * 100:.2f}%)\n"
            )

        f.write("\nSCREENING THRESHOLDS\n")
        f.write("-" * 75 + "\n")

        f.write(
            f"Low P95: {LOW_P95_THRESHOLD}\n"
        )

        f.write(
            f"Low active ratio: {LOW_ACTIVE_RATIO}%\n"
        )

        f.write(
            f"Sparse active ratio: {SPARSE_ACTIVE_RATIO}%\n"
        )

        f.write(
            f"High P99: {HIGH_P99_THRESHOLD}\n"
        )

        f.write(
            f"Frame variation CV: {FRAME_VARIATION_THRESHOLD}\n"
        )

        f.write(
            "\nNote: Categories are statistical screening flags, "
            "not confirmed optical-flow noise labels.\n"
        )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE")
    print("=" * 65)

    print(f"Successfully analyzed: {len(results)}")
    print(f"Failed: {len(failed)}")

    print("\nCategory counts:")

    for category in CATEGORIES:

        print(
            f"{category}: {len(grouped[category])}"
        )

    print(
        f"\nReports saved in:\n{OUTPUT_ROOT.resolve()}"
    )


if __name__ == "__main__":

    main()
