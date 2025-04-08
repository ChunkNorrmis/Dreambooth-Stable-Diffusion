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
        set,
        reg,
        data_root,
        placeholder_token,
        coarse_class_text,
        repeats,
        size,
        interpolation,
        center_crop,
        flip_p,
        token_only,
        per_image_tokens,
        mixing_prob
    ):
        super().__init__()
        self.set = set
        self.reg = reg
        self.data_root = data_root
        self.image_paths = find_images(self.data_root)
        self._image_count = len(self.image_paths)
        self.image_count = self._image_count
        self.placeholder_token = placeholder_token
        self.coarse_class_text = coarse_class_text
        self.repeats = repeats
        self.size = size
        self.center_crop = center_crop        
        self.token_only = token_only
        self.per_image_tokens = per_image_tokens
        self.mixing_prob = mixing_prob
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.interpolation = {
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS
        }[interpolation]
                
        if self.per_image_tokens:
            assert self._image_count < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if self.set == "train":
            self.image_count = self._image_count * self.repeats

        if self.reg and self.coarse_class_text:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])

    def __len__(self):
        return self.image_count

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i % self._image_count]
        image = Image.open(image_path, "r")

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["caption"] = ""
        if self.reg and self.coarse_class_text:
            example["caption"] = generic_captions_from_path(image_path, self.data_root, self.reg_tokens)
        else:
            example["caption"] = caption_from_path(image_path, self.data_root, self.coarse_class_text, self.placeholder_token)

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
            image = ImageEnhance.Sharpness(image).enhance(1.2)

        image = self.flip(image)
        
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
