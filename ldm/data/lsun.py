import os
import numpy as np
import PIL
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms


class LSUNBase(Dataset):
    def __init__(
        self,
        txt_file:str,
        data_root:str,
        size:int,
        interpolation:str,
        flip_p:float
    ):
        super().__init__()
        with open(txt_file, "r") as f:
            self.image_paths = f.read().splitlines()
        self.data_root = data_root
        self.image_count = len(self.image_paths)
        self._len = self.image_count
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l) for l in self.image_paths],
        }
        self.size = size
        self.interpolation = {
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS
        }[interpolation]
        self.flip_p = flip_p
        self.flip = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        
        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.flip_p > 0.0:
            image = self.flip(image)
        
        if self.center_crop and not image.width == image.height:
            img = np.asarray(image, dtype=np.uint8)
            
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
            
            image = Image.fromarray(img)
        
        if not (self.size, self.size) >= image.size:
            image = image.resize(
                (self.size, self.size),
                resample=self.interpolation,
                reducing_gap=3
            )
            image = ImageEnhance.Sharpness(image).enhance(1.25)

        img = (np.asarray(image, dtype=np.uint8) / 127.5 - 1.0)
        example["image"] = np.array(img, dtype=np.float32)
        return example


class LSUNChurchesTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", data_root="data/lsun/churches", **kwargs)


class LSUNChurchesValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_val.txt", data_root="data/lsun/churches",
                         flip_p=flip_p, **kwargs)


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
                         flip_p=flip_p, **kwargs)


class LSUNCatsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)


class LSUNCatsValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
                         flip_p=flip_p, **kwargs)
