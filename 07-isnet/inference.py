"""
Inference script for running the trained IS-Net segmentation model.

Loads the saved model weights (.pth), processes input raw images, runs forward passes,
and outputs predicted grayscale segmentation probability masks to local disk.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import ISNetDIS

def main():
    parser = argparse.ArgumentParser(description="Run inference using the trained SimpleSegModel")
    parser.add_argument("--image_path", type=str, default="test.jpg", help="Path to input image for segmentation")
    parser.add_argument("--model_path", type=str, default="isnet_model.pth", help="Path to saved model weights (.pth)")
    parser.add_argument("--output_path", type=str, default="prediction.png", help="Path to save the generated prediction mask")
    parser.add_argument("--no_show", action="store_true", help="Do not display the final prediction mask window using plt.show()")
    args = parser.parse_args()

    # Validate file existences
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"[ERROR] Input image '{args.image_path}' not found.")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"[ERROR] Trained model weights '{args.model_path}' not found. Please train the model first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")
    # Load model
    model = ISNetDIS().to(device)
    
    # Robust loading of model weights (supports state_dict directly or dictionary wrappers)
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Setup image transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load and preprocess input image
    image = Image.open(args.image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Run forward pass (ISNetDIS outputs (preds, features))
    with torch.no_grad():
        preds, dfs = model(input_tensor)
        output = preds[0]  # Extract primary high-resolution mask (d1)

    # Convert predicted output to grayscale segmentation mask
    mask = output.squeeze().cpu().numpy()
    
    # Plot and save prediction mask
    plt.figure()
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Segmentation Mask")
    plt.axis("off")
    plt.savefig(args.output_path, bbox_inches="tight")
    plt.close()
    print(f"Binary mask successfully saved as: '{args.output_path}'")

    # Generate the transparent segmented image (foreground isolated, background removed)
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)
    segmented_image = image.copy()
    segmented_image.putalpha(mask_pil)
    
    # Save segmented foreground image (forced to PNG to support alpha channel transparency)
    base, ext = os.path.splitext(args.output_path)
    segmented_path = f"{base}_segmented.png"
    segmented_image.save(segmented_path)
    print(f"Segmented foreground image successfully saved as: '{segmented_path}'")

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
