# Training Observations

## Objective

The objective of this experiment was to understand the workflow of training a segmentation model using an open dataset, tracking experiments with Weights & Biases (wandb), and performing inference on unseen images.

## Dataset

- Dataset: Oxford-IIIT Pet Dataset
- Task: Image Segmentation
- Input: RGB Images
- Output: Segmentation Masks

## Training Configuration

- Model: Lightweight Segmentation Model
- Optimizer: Adam
- Loss Function: Binary Cross Entropy (BCE)
- Batch Size: 4
- Epochs: 2

## Training Results

| Epoch | Loss |
|---------|---------|
| 1 | 0.0577 |
| 2 | 0.0424 |

## Observations

- The training loss decreased from 0.0577 to 0.0424 over two epochs.
- The model successfully learned basic foreground-background segmentation patterns.
- Weights & Biases (wandb) was used to monitor and log training metrics.
- The model was successfully saved after training.

## Inference Results

- A separate inference pipeline was implemented using `inference.py`.
- The trained model generated segmentation masks for unseen images.
- The output mask highlighted foreground regions with varying confidence levels.
- The grayscale mask provided a visual representation of the model's segmentation predictions.

## Conclusion

This experiment demonstrated the complete segmentation workflow, including dataset preparation, model training, experiment tracking, model saving, and inference generation.

---

## Post-Training Analysis & Review Notes

### 1. Model Architecture Discrepancy
* **Current Implementation**: A custom lightweight `SimpleSegModel` (2 downsampling layers, 2 upsampling layers) was used as a prototype.
* **Official IS-Net (DIS-Net)**: A much deeper architecture (6 encoder/decoder stages) featuring **Intermediate Supervision (IS)**.
* **Why it was used**: SimpleSegModel acts as a fast end-to-end integration check to test dataset loader parsing, optimizer steps, W&B logging, and file generation without requiring massive GPU resources. In production, this placeholder must be replaced by the official IS-Net model.

### 2. Low Loss (~0.04) vs. Incorrect Segmentation Mask
* **The Bug**: Oxford-IIIT Pet Dataset mask pixel values are integers: `1` (Foreground), `2` (Background), and `3` (Outline).
* **The Cause**: `transforms.ToTensor()` divides the grayscale image by 255 (yielding values ~0.0039, ~0.0078, ~0.0118). The subsequent thresholding function `(x > 0.5).float()` maps all these values to `0.0`.
* **The Result**: The ground truth masks are completely black (all zeros). The model learns to predict all zeros, yielding a misleadingly low loss but an empty (incorrect) mask.
* **The Fix**: Map the mask integers before dividing by 255 (e.g. comparing the PIL array values directly: `np.array(img) == 1`).

### 3. Git Repository Best Practices
* **Action**: Created `.gitignore` to ignore weights (`*.pth`), compiled files (`__pycache__/`), and W&B files (`wandb/`).
* **Why**: Storing large binary weights in Git bloats the repository size, slows down git actions, and is a poor practice since Git cannot perform binary diffs. Weights should be stored on external object storage or registered in W&B artifacts.