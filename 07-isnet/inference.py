
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import SimpleSegModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleSegModel().to(device)

model.load_state_dict(
    torch.load("isnet_model.pth", map_location=device)
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

image = Image.open("test.jpg").convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)

mask = output.squeeze().cpu().numpy()

plt.imshow(mask, cmap="gray")
plt.title("Predicted Segmentation Mask")
plt.axis("off")

plt.savefig("prediction.png")
print("Prediction saved as prediction.png")

plt.show()
