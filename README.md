# Blood Cell Segmentation: U-Net++ vs YOLOv8-Seg

This repository contains an end-to-end, notebook-based pipeline to train and compare:

- **U-Net++ (semantic segmentation)**
- **YOLOv8-Seg (instance segmentation)** 

The workflow is designed for **blood cell analysis** on the **BCCD Dataset with masks** and includes training across multiple random seeds, quantitative evaluation, and visualizations.

## What’s inside

- **`model_comparison.ipynb`**
  - Dataset loading + augmentations (Albumentations)
  - U-Net++ training/validation with checkpointing (best + last)
  - Test evaluation and qualitative visualizations
  - Instance-mask derivation from binary masks (watershed)
  - COCO conversion + YOLO-format dataset generation
  - YOLOv8-Seg training + evaluation
  - Comparative plots (metrics, training curves, failure cases, counting, speed vs accuracy)

## Setup

### 1) Create and activate an environment

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

Note: If you have an NVIDIA GPU, you may want to install the CUDA-enabled PyTorch build first using the official PyTorch instructions, then install the remaining requirements.

## Dataset

This project expects the **BCCD Dataset with masks** (commonly distributed via Kaggle).

1. Download the dataset locally.
2. Set `DATASET_ROOT` in `model_comparison.ipynb` to point to the dataset directory.

The notebook expects a structure similar to:

```
DATASET_ROOT/
  train/
    images/   (RGB images)
    masks/    (binary masks aligned with images)
  test/
    images/
    masks/
```

If your dataset uses different folder names, update the pairing logic in the notebook accordingly.

## Running the experiments

Open and run the notebook:

```bash
jupyter lab
# then open model_comparison.ipynb
```

Key configuration variables are defined near the top of each section (examples):

- `DATASET_ROOT`, `OUTPUT_DIR`
- `SEEDS` (multi-run evaluation)
- U-Net++: encoder, image size, epochs, learning rate, batch size
- YOLOv8-Seg: `YOLO_EPOCHS`, `YOLO_IMGSZ`, `YOLO_BATCH`

## Outputs

The notebook writes artifacts to the configured output folders, typically including:

- Model checkpoints:
  - `best_unet_plusplus_seed{seed}.pt`
  - `last_unet_plusplus_seed{seed}.pt`
  - YOLO weights under `.../yolov8_seg_seed{seed}/weights/`
- Metrics and figures:
  - performance comparison plots (mean ± std across seeds)
  - training curves
  - qualitative prediction grids
  - error analysis, cell-counting comparisons, speed vs accuracy trade-off

## Notes and limitations

- The instance segmentation pipeline derives instances from binary masks via watershed; if masks are noisy or touching objects are common, instance quality may degrade.
- Some steps (e.g., high-resolution training) are GPU-intensive. Reduce `imgsz`, `img_size`, or `batch` if you run out of memory.

## Citation and attribution

If you use this code in academic work, cite the dataset source (e.g., Kaggle listing) and the main libraries used (PyTorch, segmentation-models-pytorch, Ultralytics YOLO).