
import argparse

import torch
import wandb

from dataset import get_dataloader
from model import ISNetDIS


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        required=True
    )

    parser.add_argument(
        "--epochs",
        default=5,
        type=int
    )

    parser.add_argument(
        "--batch_size",
        default=2,
        type=int
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    wandb.init(
        project="isnet-training"
    )

    loader = get_dataloader(
        args.dataset_path,
        args.batch_size
    )

    model = ISNetDIS().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    for epoch in range(args.epochs):

        model.train()

        running_loss = 0

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            preds, features = model(images)

            loss0, loss = model.compute_loss(
                preds,
                masks
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"Loss: {avg_loss:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss
        })

    torch.save(
        model.state_dict(),
        "isnet_model.pth"
    )

    wandb.finish()


if __name__ == "__main__":
    main()
