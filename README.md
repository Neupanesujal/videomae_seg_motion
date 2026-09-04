# Semantic and Motion Aware Masking for Self-Supervised Video Representation Learning

Master's thesis implementation based on **VideoMAE V1**.

## 1. Requirements

Recommended environment:

- Linux
- Python 3.10+
- NVIDIA GPU with a compatible driver
- CUDA-compatible PyTorch
- Git
- UCF101 dataset already downloaded

## 2. Setup on a New Machine

### Step 1 — Clone the repository

```bash
git clone <YOUR_GIT_REPOSITORY_URL>
cd videomae_seg_motion
```

### Step 2 — Create and activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python3 -m pip install --upgrade pip
```

### Step 3 — Install dependencies

If `requirements.txt` exists:

```bash
pip3 install -r requirements.txt
```

For PyTorch, use a build compatible with the NVIDIA driver/GPU on the new machine.

Verify:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### Step 4 — Prepare UCF101

The expected dataset location is:

```text
datasets/UCF-101/
```

Expected structure:

```text
datasets/
└── UCF-101/
    ├── ApplyEyeMakeup/
    ├── Archery/
    ├── ...
    └── YoYo/
```

Also ensure the split/list directory exists:

```text
datasets/ucfTrainTestlist/
```

### Step 5 — Prepare the Experiment 1 pretraining list

Experiment 1 uses UCF101 Split 1 + Split 2 for pretraining.

Expected file:

```text
datasets/ucfTrainTestlist/experiment1_pretrain.txt
```

Verify:

```bash
wc -l datasets/ucfTrainTestlist/experiment1_pretrain.txt
```

The current list contains **13,320 videos**.

### Step 6 — Prepare the VideoMAE pretrained checkpoint

Experiment 1 starts from a **VideoMAE V1 ViT-Base checkpoint pretrained on Kinetics-400**.

Expected path:

```text
checkpoints/pretrained/videomae_v1_base_kinetics_e1600.pth
```

Verify:

```bash
ls -lh checkpoints/pretrained/
```

Large checkpoints are intentionally kept outside Git.

## 3. Verify the Installation

Verify core imports:

```bash
python3 -c "import torch; import torchvision; import yaml; import cv2; import numpy; print('Dependencies OK')"
```

Verify CUDA:

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## 4. Run the Smoke Test

Before a long experiment, run:

```bash
python3 -m scripts.smoke_test_pretraining
```

The smoke test verifies:

- GPU access
- dataset loading
- VideoMAE model creation
- forward/backward pass
- checkpoint saving
- checkpoint resuming
- timing and logging

The smoke test is separate from the YAML experiment runner, so it does not necessarily create the same `config/` directory.

## 5. Run Experiment 1

Run:

```bash
python3 -m scripts.pretrain exp01_random
```

The configuration is composed from:

```text
configs/base/pretraining.yaml
configs/experiments/exp01_random.yaml
```

The merged configuration is automatically saved to:

```text
outputs/experiment1/exp01_random/config/
```

with:

```text
config.yaml
config.json
```

### Experiment 1 settings

- VideoMAE V1 ViT-Base
- Kinetics-400 pretrained initialization
- UCF101 Split 1 + Split 2 for pretraining
- Random tubelet masking
- Mask ratio: 0.75
- 16 frames
- Sampling rate: 4
- Input size: 224
- Tubelet size: 2
- Patch size: 16
- Decoder depth: 4
- AdamW
- Learning rate: 3e-4
- Weight decay: 0.05
- Warmup: 10 epochs
- Batch size: 8
- 100 epochs

## 6. Expected Experiment Output

```text
outputs/
└── experiment1/
    └── exp01_random/
        ├── checkpoints/
        ├── logs/
        │   ├── log.txt
        │   ├── training_log.json
        │   ├── checkpoint_timing.json
        │   ├── experiment_state.json
        │   └── summary.json
        ├── config/
        │   ├── config.yaml
        │   └── config.json
        ├── reconstructions/
        └── evaluation/
```

Training outputs should normally not be committed to Git.

## 7. Reconstruct Videos from Checkpoints

For a normal experiment:

```bash
python3 -m scripts.reconstruct_checkpoints     --experiment exp01_random
```

The reconstruction script selects deterministic UCF101 videos from different classes and saves:

- original videos
- masked videos
- reconstructed videos

under:

```text
outputs/experiment1/exp01_random/reconstructions/
```

### Existing smoke-test checkpoints

The current smoke test stores checkpoints directly under:

```text
outputs/experiment1/smoke_test/
```

Therefore use:

```bash
python3 -m scripts.reconstruct_checkpoints     --experiment smoke_test     --checkpoint_dir outputs/experiment1/smoke_test     --output_dir outputs/experiment1/smoke_test/reconstructions
```

## 8. Git and Large Files

Do not normally commit:

```text
datasets/
outputs/
*.pth
*.pt
*.ckpt
*.avi
*.mp4
```

The repository should contain source code, configurations, experiment definitions, documentation, and other lightweight files.

## 9. Reproduction Checklist

On a new machine:

```bash
git clone <YOUR_GIT_REPOSITORY_URL>
cd videomae_seg_motion

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

Then provide:

```text
datasets/UCF-101/
datasets/ucfTrainTestlist/
checkpoints/pretrained/videomae_v1_base_kinetics_e1600.pth
```

Verify:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

Run the smoke test:

```bash
python3 -m scripts.smoke_test_pretraining
```

Then run Experiment 1:

```bash
python3 -m scripts.pretrain exp01_random
```

## 10. Reproducibility

Do not manually change experiment hyperparameters from the command line.

Experiment definitions are stored under:

```text
configs/experiments/
```

The generated merged configuration is stored inside each experiment's output directory:

```text
outputs/experiment1/<experiment_name>/config/
```

This provides a record of the configuration used for the run.

## 11. Current Experiment Plan

```text
Experiment 1 — UCF101
    ├── Random masking
    ├── Semantic masking
    ├── Motion masking
    └── Semantic + Motion masking

Experiment 2 — UCF101 + Something-Something V2 subset
    ├── Random masking
    ├── Semantic masking
    ├── Motion masking
    └── Semantic + Motion masking
```