from PIL import Image, ImageOps, ImageFilter
import numpy as np
import random

# 1. Edge Smear Transformation
class EdgeSmearTransformation:
    def __init__(self, border_ratio=0.2):
        self.border_ratio = border_ratio

    def apply(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        border_size = int(min(width, height) * self.border_ratio)

        left_strip = image.crop((0, 0, 1, height)).resize((border_size, height))
        right_strip = image.crop((width-1, 0, width, height)).resize((border_size, height))
        top_strip = image.crop((0, 0, width, 1)).resize((width, border_size))
        bottom_strip = image.crop((0, height-1, width, height)).resize((width, border_size))

        new_image = Image.new("RGB", (width + 2 * border_size, height + 2 * border_size))
        new_image.paste(left_strip, (0, border_size))
        new_image.paste(right_strip, (width + border_size, border_size))
        new_image.paste(top_strip, (border_size, 0))
        new_image.paste(bottom_strip, (border_size, height + border_size))
        new_image.paste(image, (border_size, border_size))

        return new_image

# 2. Rotated Overlay Transformation
class RotatedOverlayTransformation:
    def __init__(self, angle_range=(-10, 10), opacity=0.4, background_color=(0, 255, 0)):
        self.angle_range = angle_range
        self.opacity = opacity
        self.background_color = background_color

    def apply(self, image: Image.Image) -> Image.Image:
        angle = random.uniform(*self.angle_range)
        rotated = image.rotate(angle, expand=True)

        bg_size = (int(image.width * 1.5), int(image.height * 1.5))
        background = Image.new("RGB", bg_size, self.background_color)

        offset = ((bg_size[0] - rotated.width) // 2, (bg_size[1] - rotated.height) // 2)
        background.paste(rotated, offset)

        # Create overlay with original image pasted at center
        overlay = background.copy()
        original_offset = ((bg_size[0] - image.width) // 2, (bg_size[1] - image.height) // 2)
        overlay.paste(image, original_offset)

        return Image.blend(background, overlay, self.opacity)


# 3. Puzzle Shuffle Transformation
class PuzzleShuffleTransformation:
    def __init__(self, min_patches=6, max_patches=10, swap_ratio_range=(0.2, 0.4)):
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.swap_ratio_range = swap_ratio_range  # (min%, max%)

    def apply(self, image: Image.Image) -> Image.Image:
        img_array = np.array(image)
        h, w, c = img_array.shape

        grid_n = random.randint(self.min_patches, self.max_patches)
        ph, pw = h // grid_n, w // grid_n

        patches = []
        for i in range(grid_n):
            for j in range(grid_n):
                top = i * ph
                left = j * pw
                bottom = h if i == grid_n - 1 else top + ph
                right = w if j == grid_n - 1 else left + pw
                patch = img_array[top:bottom, left:right, :]
                patches.append(patch)

        total_patches = len(patches)
        num_pairs = random.randint(
            int(total_patches * self.swap_ratio_range[0] / 2),
            int(total_patches * self.swap_ratio_range[1] / 2)
        )

        indices = list(range(total_patches))
        random.shuffle(indices)
        swap_pairs = [(indices[2 * i], indices[2 * i + 1]) for i in range(num_pairs)]

        for i1, i2 in swap_pairs:
            patches[i1], patches[i2] = patches[i2], patches[i1]

        new_img = np.zeros_like(img_array)
        idx = 0
        for i in range(grid_n):
            for j in range(grid_n):
                top = i * ph
                left = j * pw
                patch = patches[idx]
                h_patch, w_patch, _ = patch.shape
                new_img[top:top + h_patch, left:left + w_patch, :] = patch
                idx += 1

        return Image.fromarray(new_img)

class RandomPixelMasking:
    def __init__(self, drop_prob=0.08, block_size=10):
        """
        drop_prob: Probability of applying a noise block at each patch
        block_size: Size of the square noise blocks
        """
        self.drop_prob = drop_prob
        self.block_size = block_size

    def apply(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        h, w, c = img_np.shape

        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                if random.random() < self.drop_prob:
                    # Generate a random color block
                    color = np.random.randint(0, 256, size=(1, 1, c), dtype=np.uint8)
                    img_np[i:i+self.block_size, j:j+self.block_size] = color

        return Image.fromarray(img_np)
    
# img = Image.open("D:\\-EVENTA2025-Event-Enriched-Image-Captioning\\data\\test\\pub_images\\0ab5b08a0bbf835a.jpg")

# transform1 = EdgeSmearTransformation()
# transform2 = RotatedOverlayTransformation()
# transform3 = PuzzleShuffleTransformation()

# # img1 = transform1.apply(img)
# # img2 = transform2.apply(img)
# img3 = transform3.apply(img)

# # img1.show()
# # img2.show()
# img3.show()
