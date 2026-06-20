
import torch

from PIL import Image

from torchvision import transforms

from model import ISNetDIS


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = ISNetDIS().to(device)

model.load_state_dict(
    torch.load(
        "isnet_model.pth",
        map_location=device
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.ToTensor()
])

image = Image.open(
    "test(1).jpg"
).convert("RGB")

input_tensor = transform(
    image
).unsqueeze(0).to(device)

with torch.no_grad():

    preds, features = model(
        input_tensor
    )

mask = preds[0]

mask = mask.squeeze().cpu().numpy()

import matplotlib.pyplot as plt

plt.imshow(
    mask,
    cmap="gray"
)

plt.axis("off")

plt.savefig(
    "predicted_mask.png"
)

print(
    "Saved predicted_mask.png"
)
