import os
import numpy as np
import PIL
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms

import random

imagenet_templates_small = [
    'a painting in the style of {}',
    'a rendering in the style of {}',
    'a cropped painting in the style of {}',
    'the painting in the style of {}',
    'a clean painting in the style of {}',
    'a dirty painting in the style of {}',
    'a dark painting in the style of {}',
    'a picture in the style of {}',
    'a cool painting in the style of {}',
    'a close-up painting in the style of {}',
    'a bright painting in the style of {}',
    'a cropped painting in the style of {}',
    'a good painting in the style of {}',
    'a close-up painting in the style of {}',
    'a rendition in the style of {}',
    'a nice painting in the style of {}',
    'a small painting in the style of {}',
    'a weird painting in the style of {}',
    'a large painting in the style of {}',
]

imagenet_dual_templates_small = [
    'a painting in the style of {} with {}',
    'a rendering in the style of {} with {}',
    'a cropped painting in the style of {} with {}',
    'the painting in the style of {} with {}',
    'a clean painting in the style of {} with {}',
    'a dirty painting in the style of {} with {}',
    'a dark painting in the style of {} with {}',
    'a cool painting in the style of {} with {}',
    'a close-up painting in the style of {} with {}',
    'a bright painting in the style of {} with {}',
    'a cropped painting in the style of {} with {}',
    'a good painting in the style of {} with {}',
    'a painting of one {} in the style of {}',
    'a nice painting in the style of {} with {}',
    'a small painting in the style of {} with {}',
    'a weird painting in the style of {} with {}',
    'a large painting in the style of {} with {}',
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(
        self,
        data_root,
        size,
        repeats,
        interpolation,
        flip_p,
        set,
        placeholder_token,
        per_image_tokens=False,
        center_crop=False
    ):
        super().__init__()
        self.set = set
        self.repeats = repeats
        self.size = size
        self.data_root = data_root
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.image_count = len(self.image_paths)
        self._len = self.image_count
        self.placeholder_token = placeholder_token
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.interpolation = {
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        
        if per_image_tokens:
            assert self.image_count < len(self.per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(self.per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if self.set == "train":
            self.image_count = self._len * self.repeats


    def __len__(self):
        return self._len

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i % self.image_count]
        image = Image.open(image_path, "r")

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.per_image_tokens and np.random.uniform() < 0.25:
            text = random.choice(imagenet_dual_templates_small).format(self.placeholder_token, per_img_token_list[i % self.image_count])
        else:
            text = random.choice(imagenet_templates_small).format(self.placeholder_token)
            
        example["caption"] = text
        
        if self.center_crop and not image.width == image.height:
            img = np.array(image).astype(np.uint8)
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                      (w - crop) // 2:(w + crop) // 2]
            image = Image.fromarray(img)
        
        if not (self.size, self.size) == image.size:
            image = image.resize(
                (self.size, self.size),
                resample=self.interpolation,
                reducing_gap=3
            )
            image = ImageEnhance.Sharpness(image).enhance(1.3)

        image = self.flip(image)
        
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
