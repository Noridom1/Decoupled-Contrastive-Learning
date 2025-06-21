import os
import json
from PIL import Image
from experiment.Dinov2SimCLR.utils.random_augmenter import RandomAugmenter
from tqdm import *

def generate_augmented_images(input_folder, output_folder='augmented', json_path='pairs.json'):
    os.makedirs(output_folder, exist_ok=True)
    augmenter = RandomAugmenter()
    pairs = []

    for fname in tqdm(os.listdir(input_folder)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        original_path = os.path.join(input_folder, fname)
        base_name = os.path.splitext(fname)[0]
        aug_name = f"{base_name}.jpg"
        aug_path = os.path.join(output_folder, aug_name)
        if os.path.exists(aug_path):
            continue

        img = Image.open(original_path).convert('RGB')
        aug_img = augmenter.apply(img)
        aug_img.save(aug_path)

        pairs.append((fname, aug_name))

    # Save pairs to JSON
    with open(json_path, 'w') as f:
        json.dump(pairs, f, indent=2)

original_folder = 'data_train/original'
augmented_folder = 'data_train/augmented'
pairs_path = 'data_train/pairs.json'

generate_augmented_images(original_folder, augmented_folder, pairs_path)
