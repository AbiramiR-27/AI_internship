
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader

image_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

def get_dataloader(batch_size=4):

    dataset = OxfordIIITPet(
        root="data",
        download=True,
        target_types="segmentation",
        transform=image_transform,
        target_transform=mask_transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
