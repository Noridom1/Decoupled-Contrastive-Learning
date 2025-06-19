from PIL import ImageEnhance, ImageFilter
import random
from transformation import *

class RandomAugmenter:
    def __init__(self):
        self.augmentations = [
            lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 2.5))),

            lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.4, 1.6)),

            lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.3, 1.9)),

            lambda img: img.crop((
                int(0.1 * img.width), int(0.1 * img.height),
                int(0.9 * img.width), int(0.9 * img.height)
            )).resize(img.size),

            # ✅ Custom transformations (unchanged)
            PuzzleShuffleTransformation(min_patches=6, max_patches=12, swap_ratio_range=(0.2, 0.4)).apply,
            EdgeSmearTransformation(border_ratio=0.1).apply,
            RotatedOverlayTransformation(angle_range=(-14, 14), opacity=0.6).apply,
            RandomPixelMasking().apply,
            RandomPixelMasking(block_size=5).apply
        ]

    def apply(self, img: Image.Image) -> Image.Image:
        transforms_to_apply = random.sample(self.augmentations, k=random.randint(1, 3))
        for transform in transforms_to_apply:
            img = transform(img)
        return img
