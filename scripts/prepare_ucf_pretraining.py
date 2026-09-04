from pathlib import Path


SPLIT_DIR = Path("datasets/ucfTrainTestlist")
VIDEO_ROOT = Path("datasets/UCF-101")
OUTPUT_FILE = SPLIT_DIR / "experiment1_pretrain.txt"

SPLIT_FILES = [
    "trainlist01_train.txt",
    "trainlist01_val.txt",
    "trainlist02_train.txt",
    "trainlist02_val.txt",
]


def read_split_file(path):
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def create_pretraining_list():
    videos = []

    for filename in SPLIT_FILES:
        path = SPLIT_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")

        videos.extend(read_split_file(path))

    # Remove duplicates while preserving order.
    videos = list(dict.fromkeys(videos))

    with OUTPUT_FILE.open("w") as f:
        for line in videos:
            video_path, label = line.split()

            # Convert the original relative UCF101 path into
            # a path that can be resolved from the project root.
            video_path = VIDEO_ROOT / video_path

            f.write(f"{video_path} {label}\n")

    print(f"Created: {OUTPUT_FILE}")
    print(f"Videos: {len(videos)}")


if __name__ == "__main__":
    create_pretraining_list()