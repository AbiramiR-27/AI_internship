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