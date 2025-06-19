import os
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)

class SimCLRPairDataset(Dataset):
    def __init__(self, root_dir, processor_name="facebook/dinov2-base"):
        """
        Args:
            root_dir (str): Root directory containing 'original/' and 'augmented/' subfolders.
            processor_name (str): The name of the DINOv2 processor to use.
        """
        self.original_dir = os.path.join(root_dir, "original")
        self.augmented_dir = os.path.join(root_dir, "augmented")
        # self.processor = AutoImageProcessor.from_pretrained(processor_name, use_fast=True)


        # Get list of common filenames
        self.filenames = sorted(os.listdir(self.original_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        original_path = os.path.join(self.original_dir, filename)
        augmented_path = os.path.join(self.augmented_dir, filename)

        # Load images
        img1 = Image.open(original_path).convert("RGB")
        img2 = Image.open(augmented_path).convert("RGB")

        # print("[DATASET] Loaded images")
        # Preprocess using DINOv2 processor
        img1_tensor = processor(images=img1, return_tensors="pt")["pixel_values"].squeeze(0)
        img2_tensor = processor(images=img2, return_tensors="pt")["pixel_values"].squeeze(0)

        return img1_tensor, img2_tensor
