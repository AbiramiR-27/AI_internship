import argparse

import torch

import numpy as np

from PIL import Image

from torchvision import transforms

from model import ISNetDIS

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", required=True, help="Path to input image")
parser.add_argument("--model_path", default="isnet_model.pth", help="Path to model weights")
parser.add_argument("--output_path", default="predicted_mask.png", help="Path to save output mask")
args = parser.parse_args()

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Initialize and load model
model = ISNetDIS().to(device)

model.load_state_dict(
    torch.load(
        args.model_path,
        map_location=device,
        weights_only=True  
    )
)

model.eval()

# Load image and keep track of original resolution

image = Image.open(args.image_path).convert("RGB")
original_size = image.size

# Resize to the trained model resolution (512x512) for optimal segmentation quality
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    preds, features = model(input_tensor)

# Extract first prediction mask and squeeze singleton batch/channel dimensions
mask = preds[0].squeeze().cpu().numpy()

# Post-processing: Normalize values between 0 and 255
mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
mask = (mask * 255).astype(np.uint8)

# Convert to PIL Image and resize back to original image size
mask_image = Image.fromarray(mask)
mask_image = mask_image.resize(original_size, Image.Resampling.LANCZOS)

# Save the clean grayscale mask
mask_image.save(args.output_path)

print(f"Saved {args.output_path}")