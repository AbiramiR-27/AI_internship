# ISNet Segmentation Training & Inference

This directory contains the Python codebase to train and evaluate a lightweight segmentation model (Substituted/Placeholder for ISNet training experiments). The framework has been transitioned fully to standard Python scripts and includes configurable options, input validation, and interactive Weights & Biases (wandb) image logging.

---

## 1. Environment Setup

It is recommended to run this codebase inside a virtual environment to prevent package conflicts.

### Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
Install the required packages using the local `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 2. Input Directory Structure & Formats

The custom dataset loader expects a local directory to be passed via the training configuration. You must organize your data as follows:

```
dataset_root/
├── images/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── masks/
    ├── image1.png
    ├── image2.png
    └── ...
```

### Required File Formats
1. **Raw Images (`images/`):** Matches common image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`.
2. **Ground Truth Masks (`masks/`):** Expected to be single-channel index/grayscale binary maps (matching name and mapped to `.png` or matching original filename/extension) representing foreground (white pixels) and background (black pixels).
3. **Alignment:** Ensure the names of images and corresponding masks are identical (e.g. `img_001.jpg` in `images/` maps to `img_001.png` or `img_001.jpg` in `masks/`).

---

## 3. Starting Training

You must run the training script using command-line arguments. Before starting, make sure you are logged in to Weights & Biases:
```bash
wandb login
```

### Running Training
Use the `--dataset_path` parameter to specify where your dataset is located:
```bash
python train.py --dataset_path /path/to/dataset_root
```

### Advanced Training Parameters
The script accepts several optional arguments:
- `--dataset_path` (Required): Path to your local dataset root directory.
- `--batch_size` (Default `4`): Mini-batch size.
- `--epochs` (Default `2`): Number of training epochs.
- `--lr` (Default `0.001`): Learning rate for the Adam optimizer.
- `--wandb_project` (Default `isnet-training`): Weights & Biases project folder.
- `--wandb_run_name` (Default `isnet-segmentation-run`): Name of the training run in W&B.

Example command:
```bash
python train.py --dataset_path ./data/custom_dataset --batch_size 8 --epochs 10 --lr 0.0005
```

### Experiment Logs in W&B
Logs are uploaded automatically to your W&B account. The logged items include:
- `loss`: Average training binary cross-entropy loss per epoch.
- `visualizations`: Side-by-side comparison images logged per epoch, showing the **Original Image**, **Ground Truth Mask**, and **Segmented Mask predicted by the model**.

---

## 4. Running Inference

To generate a segmentation mask for a test image using the trained model weights:

```bash
python inference.py --image_path test.jpg --model_path isnet_model.pth --output_path prediction.png
```

### Inference Parameters
- `--image_path` (Default `test.jpg`): Path to input test image.
- `--model_path` (Default `isnet_model.pth`): Path to trained model weight parameters.
- `--output_path` (Default `prediction.png`): Path where prediction mask will be saved.
- `--no_show` (Optional flag): Disables the interactive window pop-up of the mask (`plt.show()`), useful when running on remote headless servers or scripts.
