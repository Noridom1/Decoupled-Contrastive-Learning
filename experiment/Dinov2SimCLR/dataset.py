import os
import torch
from torch.utils.data import Dataset

class SimCLRPairDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Root directory that contains the `processed/original/` and `processed/augmented/` subfolders.
        """
        self.original_dir = os.path.join(root_dir, "processed", "original")
        self.augmented_dir = os.path.join(root_dir, "processed", "augmented")

        # Only keep .pt files
        self.filenames = sorted([
            fname for fname in os.listdir(self.original_dir) 
            if fname.endswith(".pt")
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # Full path to each .pt file
        original_tensor_path = os.path.join(self.original_dir, filename)
        augmented_tensor_path = os.path.join(self.augmented_dir, filename)

        # Load preprocessed tensors
        img1_tensor = torch.load(original_tensor_path)
        img2_tensor = torch.load(augmented_tensor_path)

        return img1_tensor, img2_tensor
