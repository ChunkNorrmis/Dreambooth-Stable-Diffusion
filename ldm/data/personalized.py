import os
from typing import OrderedDict
import numpy as np
import PIL
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from captionizer import caption_from_path, generic_captions_from_path
from captionizer import find_images
from dreambooth_helpers.arguments import split_parse
import argparse

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

parser = split_parse()
arg, unknown = parser.parse_known_args()

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=arg.resize,
                 repeats=arg.repeats,
                 interpolation=arg.resampler,
                 flip_p=arg.flip_p,
                 set="train",
                 placeholder_token=arg.token,
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=arg.class_word,
                 token_only=arg.token_only,
                 reg=False
                 ):
        super(PersonalizedBase).__init__()

        self.data_root = data_root

        self.image_paths = find_images(self.data_root)

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.placeholder_token = placeholder_token
        self.token_only = token_only
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(
                per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
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

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                      (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        image = self.flip(image)
        
        if self.size is not None and (image.width, image.height) > (self.size, self.size):
            image = image.resize(
                (self.size, self.size),
                resample=self.interpolation,
                reducing_gap=3
            )
            image = np.array(image).astype(np.uint8)
            image = cv2.filter2D(image, -1, kernel=np.array([[0, -1, 0], [0, 2, 0], [0, 0, 0]]))

        example["image"] = (np.array(image) / 127.5 - 1.0).astype(np.float32)
        return example

