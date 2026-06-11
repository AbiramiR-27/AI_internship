import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import SimpleSegModel

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

    # Load model and weights
    model = SimpleSegModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Setup image transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load and preprocess input image
    image = Image.open(args.image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Run feedforward forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # Convert predicted output to grayscale segmentation mask
    mask = output.squeeze().cpu().numpy()

    # Plot and save prediction mask
    plt.figure()
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Segmentation Mask")
    plt.axis("off")

    plt.savefig(args.output_path, bbox_inches="tight")
    print(f"Prediction successfully saved as: '{args.output_path}'")

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
