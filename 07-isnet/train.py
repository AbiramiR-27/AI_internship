
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from dataset import get_dataloader
from model import SimpleSegModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(
    project="isnet-training",
    name="isnet-segmentation-run"
)

loader = get_dataloader(batch_size=4)

model = SimpleSegModel().to(device)

criterion = nn.BCELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 2

for epoch in range(epochs):

    running_loss = 0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    print(
        f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}"
    )

    wandb.log({
        "epoch": epoch + 1,
        "loss": avg_loss
    })

torch.save(
    model.state_dict(),
    "isnet_model.pth"
)

print("Model Saved")

wandb.finish()
