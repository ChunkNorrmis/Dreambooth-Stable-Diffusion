import os
from typing import OrderedDict
import numpy as np
import PIL
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
from captionizer import caption_from_path, generic_captions_from_path, find_images

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(
        self,
        data_root,
        size,
        repeats,
        set,
        placeholder_token,
        coarse_class_text,
        interpolation,
        reg,
        flip_p,
        token_only=False,
        per_image_tokens=False,
        center_crop=False,
        mixing_prob=0.25
    ):
        super(PersonalizedBase).__init__()
        
        self.data_root = data_root
        self.image_paths = find_images(self.data_root)
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        self.placeholder_token = placeholder_token
        self.token_only = token_only
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob
        self.coarse_class_text = coarse_class_text
        self.size = size
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        if per_image_tokens:
            assert self.num_images < len(
                per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
            }[interpolation]

        self.reg = reg
        if self.reg and self.coarse_class_text:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i % self.num_images]
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["caption"] = ""
        if self.reg and self.coarse_class_text:
            example["caption"] = generic_captions_from_path(image_path, self.data_root, self.reg_tokens)
        else:
            example["caption"] = caption_from_path(image_path, self.data_root, self.coarse_class_text, self.placeholder_token)

        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                      (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        image = self.flip(image)

        if not (self.size, self.size) == image.size:
            image = image.resize((self.size, self.size), resample=self.interpolation, reducing_gap=3)
            image = ImageEnhance.Sharpness(image).enhance(1.2)

        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
