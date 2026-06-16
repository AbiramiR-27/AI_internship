"""
Training orchestration script for the IS-Net segmentation experiment.

Configures hyperparameters, parses command-line arguments, validates dataset paths,
instantiates model/optimizer, and manages the training epoch loops.
Logs loss metrics and side-by-side image visualizations to Weights & Biases (wandb).
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import matplotlib.pyplot as plt

from dataset import get_dataloader
from model import ISNetDIS

def log_images_to_wandb(images, masks, outputs, epoch, num_samples=4):
    """
    Renders original images, ground truth masks, and predicted masks 
    side-by-side into a single visualization image, and logs it to wandb.
    
    Parameters:
        images (Tensor): Batch of input images of shape (B, 3, H, W).
        masks (Tensor): Batch of ground truth masks of shape (B, 1, H, W).
        outputs (Tensor): Batch of predicted masks of shape (B, 1, H, W).
        epoch (int): Current epoch number.
        num_samples (int): Max number of samples to log.
    """
    image_logs = []
    limit = min(images.size(0), num_samples)
    
    for i in range(limit):
        # Transpose image to (H, W, C) for plotting
        orig_img = images[i].cpu().permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0.0, 1.0)
        
        # Squeeze mask channel dimension for 2D plotting
        gt_mask = masks[i].cpu().squeeze().numpy()
        pred_mask = outputs[i].detach().cpu().squeeze().numpy()
        
        # Draw side-by-side figures using Matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(orig_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        
        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title("Segmented Mask")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        # Convert figure to a numpy RGB array
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        
        try:
            rgb_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        except AttributeError:
            # Fallback for newer Matplotlib versions
            rgb_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
            
        plt.close(fig)
        
        image_logs.append(wandb.Image(rgb_img, caption=f"Epoch {epoch} - Sample {i}"))
        
    return image_logs

def main():
    parser = argparse.ArgumentParser(description="Train SimpleSegModel on local image segmentation dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to local dataset directory containing 'images' and 'masks'")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")
    parser.add_argument("--wandb_project", type=str, default="isnet-training", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default="isnet-segmentation-run", help="Weights & Biases run name")
    args = parser.parse_args()

    # Validate dataset directory path
    if not os.path.exists(args.dataset_path):
        print(f"\n[ERROR] The specified dataset path '{args.dataset_path}' does not exist.", file=sys.stderr)
        print("Please structure your folder as follows:", file=sys.stderr)
        print("dataset_root/", file=sys.stderr)
        print("  ├── images/  <- (JPG/PNG raw images)", file=sys.stderr)
        print("  └── masks/   <- (PNG ground truth binary masks)", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Initialize weights & biases (wandb) logger
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dataset_path": args.dataset_path,
            "device": str(device)
        }
    )

    # Initialize dataloader
    try:
        loader = get_dataloader(dataset_path=args.dataset_path, batch_size=args.batch_size)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    model = ISNetDIS().to(device)
    
    # Track model parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params:,} trainable parameters.")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training for {args.epochs} epochs...")
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_loss0 = 0.0

        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            preds, dfs = model(images)
            loss0, loss = model.compute_loss(preds, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss0 += loss0.item()

        avg_loss = running_loss / len(loader)
        avg_loss0 = running_loss0 / len(loader)
        print(f"Epoch {epoch + 1}/{args.epochs} | Total Loss: {avg_loss:.4f} | Loss0 (d1): {avg_loss0:.4f}")

        # Create visualizations of original, ground truth, and prediction masks using primary prediction preds[0]
        visualizations = log_images_to_wandb(images, masks, preds[0], epoch=epoch+1)

        # Log metrics and visualizations to wandb
        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "loss0": avg_loss0,
            "visualizations": visualizations,
            "input_image": wandb.Image(images[0].cpu(), caption=f"Epoch {epoch+1} - Input"),
            "ground_truth": wandb.Image(masks[0].cpu(), caption=f"Epoch {epoch+1} - Ground Truth"),
            "prediction": wandb.Image(preds[0][0].cpu(), caption=f"Epoch {epoch+1} - Prediction")
        })

    # Save model weights
    save_path = "isnet_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to '{save_path}'")
    wandb.finish()

if __name__ == "__main__":
    main()
