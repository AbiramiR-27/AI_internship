import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LocalSegmentationDataset(Dataset):
    """
    Custom Dataset class for loading local image segmentation data.
    
    Expected Directory Structure:
        dataset_path/
        ├── images/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        └── masks/
            ├── img1.png
            ├── img2.png
            └── ...
    """
    def __init__(self, dataset_path, image_transform=None, mask_transform=None):
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, "images")
        self.mask_dir = os.path.join(dataset_path, "masks")
        
        # Verify required directories exist
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(
                f"\n[ERROR] Dataset structure invalid at: '{dataset_path}'\n"
                "Expected directory structure:\n"
                "  ├── images/\n"
                "  └── masks/\n"
                "Please configure a valid dataset path and structure."
            )
            
        # Read sorted lists of file names
        self.image_filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])
        
        if not self.image_filenames:
            raise ValueError(f"[ERROR] No image files found in directory: '{self.image_dir}'")
            
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load the image
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Match target mask using the base name of the image
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Fallback to verify with original filename extension if png not found
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, img_name)
            
        if not os.path.exists(mask_path):
            found_mask = False
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                temp_path = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(temp_path):
                    mask_path = temp_path
                    found_mask = True
                    break
            if not found_mask:
                raise FileNotFoundError(
                    f"[ERROR] Matching mask file not found in '{self.mask_dir}' for image: '{img_name}'"
                )

        mask = Image.open(mask_path).convert("L")
        
        # Apply transforms if provided
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        return image, mask

def get_dataloader(dataset_path, batch_size=4, shuffle=True):
    """
    Initializes the custom segmentation dataset and returns a PyTorch DataLoader.
    
    Parameters:
        dataset_path (str): Absolute or relative path to dataset root.
        batch_size (int): Size of mini-batches.
        shuffle (bool): Shuffling enabled for training.
    """
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())  # Map mask to binary values 0.0 or 1.0
    ])

    dataset = LocalSegmentationDataset(
        dataset_path=dataset_path,
        image_transform=image_transform,
        mask_transform=mask_transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
