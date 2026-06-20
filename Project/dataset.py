import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None


class DISDataset(Dataset):

    def __init__(self, dataset_root, image_size=512):

        self.image_dir = os.path.join(
            dataset_root,
            "im"
        )

        self.mask_dir = os.path.join(
            dataset_root,
            "gt"
        )

        self.images = sorted(os.listdir(self.image_dir))

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        mask_name = os.path.splitext(image_name)[0] + ".png"

        mask_path = os.path.join(
            self.mask_dir,
            mask_name
        )

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return image, mask


def get_dataloader(dataset_root, batch_size=2):

    dataset = DISDataset(
        dataset_root=dataset_root,
        image_size=512
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    return loader