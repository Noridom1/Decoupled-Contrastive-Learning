import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoImageProcessor
from model import DINOv2SimCLR  # Your model class

def load_model(checkpoint_path, model_name, feature_dim, device):
    # Load frozen backbone
    backbone = AutoModel.from_pretrained(model_name)
    for param in backbone.parameters():
        param.requires_grad = False

    # Load model
    model = DINOv2SimCLR(feature_dim=feature_dim, model_name=model_name).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_image_paths(folder):
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def load_image(path):
    image = Image.open(path).convert('RGB')
    return image

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and processor
    model = load_model(args.checkpoint, args.model_name, args.feature_dim, device)
    processor = AutoImageProcessor.from_pretrained(args.model_name)

    # Get image paths
    image_paths = get_image_paths(args.image_folder)
    image_paths.sort()

    features_dict = {}
    batch_images = []
    batch_names = []
    processed_count = 0

    for img_path in tqdm(image_paths, desc="Extracting Features"):
        img = load_image(img_path)
        batch_images.append(img)
        batch_names.append(os.path.basename(img_path))

        if len(batch_images) == args.batch_size:
            # Preprocess batch
            inputs = processor(images=batch_images, return_tensors='pt').to(device)

            with torch.no_grad():
                _, feats = model(inputs['pixel_values'])

            feats = feats.cpu().numpy()
            for name, feat in zip(batch_names, feats):
                features_dict[name] = feat

            # Reset
            batch_images, batch_names = [], []

            processed_count += args.batch_size
            if processed_count % args.save_interval == 0:
                with open(args.output_pickle, 'wb') as f:
                    pickle.dump(features_dict, f)
                print(f"✅ Periodic save at {processed_count} images")

    # Handle remaining images
    if batch_images:
        inputs = processor(images=batch_images, return_tensors='pt').to(device)
        with torch.no_grad():
            _, feats = model(inputs['pixel_values'])

        feats = feats.cpu().numpy()
        for name, feat in zip(batch_names, feats):
            features_dict[name] = feat

    # Final save
    with open(args.output_pickle, 'wb') as f:
        pickle.dump(features_dict, f)
    print(f"✅ Final save completed: {args.output_pickle}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder with images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_pickle", type=str, default="features.pkl", help="Path to output .pkl file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--save_interval", type=int, default=1000, help="Period to save interim results")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base", help="HuggingFace DINOv2 model name")
    parser.add_argument("--feature_dim", type=int, default=256, help="Projection head output dim")

    args = parser.parse_args()
    inference(args)
