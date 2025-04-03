import json
import math
import os
import glob
import shutil
import sys
from datetime import datetime, timezone
import torch
from pytorch_lightning import seed_everything


class JoePennaDreamboothConfigSchemaV1:
    def __init__(self):
        super().__init__()

    def saturate(
        self,
        project_name:str,
        max_training_steps:int,
        save_every_x_steps:int,
        training_images_folder_path:str,
        regularization_images_folder_path:str,
        token:str,
        class_word:str,
        flip_percent:float,
        learning_rate:float,
        model_path:str,
        batch_size:int,
        num_workers:int,
        repeats:int,
        val_repeats:int,
        resolution:int,
        resampler:str,
        seed:int,
        debug:bool,
        gpu:int,
        token_only:bool,
        center_crop:bool,
        mix_probability:float,
        model_repo_id:str=None
    ):
        self.schema = 1
        self.project_name = project_name
        self.config_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d--%H-%M")
        self.project_config_filename = f"{self.config_date_time}-{self.project_name}-config.json"
        self.debug = debug
        self.gpu = gpu
        self.seed = seed
        seed_everything(self.seed)
        self.save_every_x_steps = save_every_x_steps

        if os.path.exists(training_images_folder_path):
            self.training_images_folder_path = os.path.relpath(training_images_folder_path)
        else:
            raise Exception(f"Training Images Path Not Found: '{self.training_images_folder_path}'.")

        _training_images_paths = [i for i in
            glob.glob(os.path.join(self.training_images_folder_path, '**', '*.jpg'), recursive=True) +
            glob.glob(os.path.join(self.training_images_folder_path, '**', '*.jpeg'), recursive=True) +
            glob.glob(os.path.join(self.training_images_folder_path, '**', '*.png'), recursive=True)
        ]

        _training_image_paths = [os.path.relpath(i, self.training_images_folder_path) for i in _training_images_paths]

        if len(_training_image_paths) > 0:
            self.training_images_count = len(_training_image_paths)
        else:
            raise Exception(f"No Training Images (*.png, *.jpg, *.jpeg) found in '{self.training_images_folder_path}'.")

        self.training_images = _training_images_paths
        self.token = token
        self.token_only = token_only
        if self.token_only is False:
            self.class_word = class_word
            if regularization_images_folder_path is not None and os.path.exists(regularization_images_folder_path):
                self.regularization_images_folder_path = os.path.relpath(regularization_images_folder_path)
            else:
                raise Exception(f"Regularization Images Path Not Found: '{self.regularization_images_folder_path}'.")

        self.model_repo_id = model_repo_id
        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            raise Exception(f"Model Path Not Found: '{model_path}'.")

        self.learning_rate = learning_rate
        if flip_percent > 0 and flip_percent < 1:
            self.flip_percent = flip_percent
        else:
            raise Exception("--flip_p: must be between 0 and 1")

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.repeats = repeats
        self.val_repeats = val_repeats

        if max_training_steps > 0:
            self.max_training_steps = max_training_steps
        else:
            self.max_training_steps = int(self.training_images_count * self.repeats)

        self.resolution = resolution
        self.resampler = resampler

        
        self.center_crop = center_crop
        self.mix_probability = mix_probability
        
        self.validate_gpu_vram()
        self._create_log_folders()

    def validate_gpu_vram(self):
        def convert_size(size_bytes):
            if size_bytes == 0:
                return "0B"
            size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return "%s %s" % (s, size_name[i])

        gpu_vram = int(torch.cuda.get_device_properties(self.gpu).total_memory)
        print(f"gpu_vram: {convert_size(gpu_vram)}")
        twenty_one_gigabytes = 22548578304
        if gpu_vram < twenty_one_gigabytes:
            raise Exception(f"VRAM: Currently unable to run on less than {convert_size(twenty_one_gigabytes)} of VRAM.")

    def saturate_from_file(self, config_file_path:str):
        if not os.path.exists(config_file_path):
            print(f"{config_file_path} not found.", file=sys.stderr)
            return None
        else:
            config_file = open(config_file_path)
            config_parsed = json.load(config_file)

            if config_parsed['schema'] == 1:
                self.saturate(
                    project_name=config_parsed['project_name'],
                    max_training_steps=config_parsed['max_training_steps'],
                    save_every_x_steps=config_parsed['save_every_x_steps'],
                    training_images_folder_path=config_parsed['training_images_folder_path'],
                    regularization_images_folder_path=config_parsed['regularization_images_folder_path'],
                    token=config_parsed['token'],
                    class_word=config_parsed['class_word'],
                    flip_percent=config_parsed['flip_percent'],
                    learning_rate=config_parsed['learning_rate'],
                    model_path=config_parsed['model_path'],
                    config_date_time=config_parsed['config_date_time'],
                    seed=config_parsed['seed'],
                    debug=config_parsed['debug'],
                    gpu=config_parsed['gpu'],
                    model_repo_id=config_parsed['model_repo_id'],
                    token_only=config_parsed['token_only'],
                    batch_size=config_parsed['batch_size'],
                    num_workers=config_parsed['num_workers'],
                    repeats=config_parsed['repeats'],
                    val_repeats=config_parsed['val_repeats'],
                    resolution=config_parsed['resolution'],
                    resampler=config_parsed['resampler'],
                    mix_probability=config_parsed['shuffle_rate'],
                    center_crop=config_parsed['center_crop']
                )
            else:
                print(f"Unrecognized schema: {config_parsed['schema']}", file=sys.stderr)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def create_checkpoint_file_name(self, steps: str):
        return f"{self.config_date_time}_{self.project_name}_{int(steps):05d}_steps.ckpt".replace(" ", "_")

    def save_config_to_file(self, save_path: str, create_active_config: bool = False):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        project_config_json = self.toJSON()
        config_save_path = os.path.join(save_path, self.project_config_filename)
        with open(config_save_path, "w") as config_file:
            config_file.write(project_config_json)

        if create_active_config:
            shutil.copy(config_save_path, os.path.join(save_path, "active-config.json"))
            print(project_config_json)
            print(f"✅ {self.project_config_filename} successfully generated.  Proceed to training.")

    def get_training_folder_name(self) -> str:
        return f"{self.config_date_time}_{self.project_name}"

    def log_directory(self) -> str:
        return os.path.join("logs", f"{self.config_date_time}_{self.project_name}")

    def log_checkpoint_directory(self) -> str:
        return os.path.join(self.log_directory(), "ckpts")

    def log_intermediate_checkpoints_directory(self) -> str:
        return os.path.join(self.log_checkpoint_directory(), "trainstep_ckpts")

    def log_config_directory(self) -> str:
        return os.path.join(self.log_directory(), "configs")

    def trained_models_directory(self) -> str:
        return "trained_models"

    def _create_log_folders(self):
        os.makedirs(self.log_directory(), exist_ok=True)
        os.makedirs(self.log_checkpoint_directory(), exist_ok=True)
        os.makedirs(self.log_config_directory(), exist_ok=True)
        os.makedirs(self.trained_models_directory(), exist_ok=True)
