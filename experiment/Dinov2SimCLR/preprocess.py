import os
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor

def preprocess_images(data_dir, model_name="facebook/dinov2-base"):
    processor = AutoImageProcessor.from_pretrained(model_name)

    original_dir = os.path.join(data_dir, "original")
    augmented_dir = os.path.join(data_dir, "augmented")
    save_dir = os.path.join(data_dir, "processed")
    orig_save_dir = os.path.join(save_dir, "original")
    aug_save_dir = os.path.join(save_dir, "augmented")

    os.makedirs(orig_save_dir, exist_ok=True)
    os.makedirs(aug_save_dir, exist_ok=True)

    filenames = sorted(os.listdir(original_dir))

    for filename in tqdm(filenames, desc="Preprocessing images"):
        orig_path = os.path.join(original_dir, filename)
        aug_path = os.path.join(augmented_dir, filename)

        img1 = Image.open(orig_path).convert("RGB")
        img2 = Image.open(aug_path).convert("RGB")

        tensor1 = processor(images=img1, return_tensors="pt")["pixel_values"].squeeze(0)
        tensor2 = processor(images=img2, return_tensors="pt")["pixel_values"].squeeze(0)

        torch.save(tensor1, os.path.join(orig_save_dir, filename + ".pt"))
        torch.save(tensor2, os.path.join(aug_save_dir, filename + ".pt"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing original/ and augmented/")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base")
    args = parser.parse_args()

    preprocess_images(args.data_dir, args.model_name)
