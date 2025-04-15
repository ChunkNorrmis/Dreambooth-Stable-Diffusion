from ldm.util import instantiate_from_config
from ldm.modules.pruningckptio import PruningCheckpointIO
from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1


class callbacks():

    def __init__(self, config: JoePennaDreamboothConfigSchemaV1):
        self.config = config

    def metrics_over_trainsteps_checkpoint(self) -> dict:
        if self.config.save_every_x_steps > 0:
            return {
                "target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                "params": {
                    "dirpath": self.config.log_intermediate_checkpoints_directory(),
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    "save_top_k": -1,
                    "every_n_train_steps": self.config.save_every_x_steps,
                    "save_weights_only": True
                }
            }
        else:
            return {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "every_n_train_steps": self.config.save_every_x_steps,
                    "save_weights_only": True,
                }
            }

    def image_logger(self) -> dict:
        return {
            "target": "dreambooth_helpers.callback_helpers.ImageLogger",
            "params": {
                "batch_frequency": 500 if self.config.save_every_x_steps <= 0 else self.config.save_every_x_steps,
                "max_images": 8,
                "increase_log_steps": False,
            }
        }

    def model_checkpoint(self) -> dict:
        return {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": self.config.log_checkpoint_directory(),
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
                "every_n_train_steps": self.config.save_every_x_steps,
            }
        }

    def setup_callback(self, model_data_config, lightning_config) -> dict:
        return {
            "target": "dreambooth_helpers.callback_helpers.SetupCallback",
            "params": {
                "resume": "",
                "now": self.config.get_training_folder_name(),
                "logdir": self.config.log_directory(),
                "ckptdir": self.config.log_checkpoint_directory(),
                "cfgdir": self.config.log_config_directory(),
                "config": model_data_config,
                "lightning_config": lightning_config,
            }
        }

    def learning_rate_logger(self) -> dict:
        return {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            }
        }

    def cuda_callback(self) -> dict:
        return {
            "target": "dreambooth_helpers.callback_helpers.CUDACallback"
        }


def get_dreambooth_model_config(config: JoePennaDreamboothConfigSchemaV1) -> dict:
    return {
        "base_learning_rate": config.learning_rate,
        "target": "ldm.models.diffusion.ddpm.LatentDiffusion",
        "params": {
            "reg_weight": 1.0,
            "linear_start": 0.00085,
            "linear_end": 0.012,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "first_stage_key": "image",
            "cond_stage_key": "caption",
            "image_size": 64,
            "channels": 4,
            "cond_stage_trainable": True,
            "conditioning_key": "crossattn",
            "monitor": "val/loss_simple_ema",
            "scale_factor": 0.18215,
            "use_ema": False,
            "embedding_reg_weight": 0.0,
            "unfreeze_model": True,
            "model_lr": config.learning_rate,
            "personalization_config": {
                "target": "ldm.modules.embedding_manager.EmbeddingManager",
                "params": {
                    "placeholder_strings": ['*'],
                    "initializer_words": ["sculpture"],
                    "per_image_tokens": False,
                    "num_vectors_per_token": 1,
                    "progressive_words": False,
                }
            },
            "unet_config": {
                "target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                "params": {
                    "image_size": 32,
                    "in_channels": 4,
                    "out_channels": 4,
                    "model_channels": 320,
                    "attention_resolutions": [4, 2, 1],
                    "num_res_blocks": 2,
                    "channel_mult": [1, 2, 4, 4],
                    "num_heads": 8,
                    "use_spatial_transformer": True,
                    "transformer_depth": 1,
                    "context_dim": 768,
                    "use_checkpoint": True,
                    "legacy": False,
                },
            },
            "first_stage_config": {
                "target": "ldm.models.autoencoder.AutoencoderKL",
                "params": {
                    "embed_dim": 4,
                    "monitor": "val/rec_loss",
                    "ddconfig": {
                        "double_z": True,
                        "z_channels": 4,
                        "resolution": 512,
                        "in_channels": 3,
                        "out_ch": 3,
                        "ch": 128,
                        "ch_mult": [1, 2, 4, 4],
                        "num_res_blocks": 2,
                        "attn_resolutions": [],
                        "dropout": 0.0,
                    },
                    "lossconfig": {
                        "target": "torch.nn.Identity"
                    },
                },
            },
            "cond_stage_config": {
                "target": "ldm.modules.encoders.modules.FrozenCLIPEmbedder"
            },
            "ckpt_path": config.model_path
        }
    }


def get_dreambooth_data_config(config: JoePennaDreamboothConfigSchemaV1) -> dict:
    reg_block = {
        "target": "ldm.data.personalized.PersonalizedBase",
        "params": {
            "set": "train",
            "reg": True,
            "data_root": config.regularization_images_folder_path,
            "placeholder_token": config.token,
            "coarse_class_text": config.class_word,
            "repeats": config.val_repeats,
            "size": config.resolution,
            "interpolation": config.resampler,
            "center_crop": config.center_crop,
            "flip_p": config.flip_percent,
            "token_only": False,
            "per_image_tokens": False,
            "mixing_prob": config.mix_probability,
            "shuffle": True if config.mix_probability > 0.0 else False
        }
    }

    val_block = {
        "target": "ldm.data.personalized.PersonalizedBase",
        "params": {
            "set": "validation",
            "data_root": config.training_images_folder_path,
            "placeholder_token": config.token,
            "coarse_class_text": config.class_word,
            "repeats": config.val_repeats,
            "size": config.resolution,
            "interpolation": config.resampler,
            "center_crop": config.center_crop,
            "flip_p": config.flip_percent,
            "token_only": config.token_only,
            "per_image_tokens": False,
            "mixing_prob": config.mix_probability,
            "shuffle": True if config.mix_probability > 0.0 else False
        }
    }
    
    data_config = {
        "target": "main.DataModuleFromConfig",
        "params": {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "wrap": False,
            "train": {
                "target": "ldm.data.personalized.PersonalizedBase",
                "params": {
                    "set": "train",
                    "data_root": config.training_images_folder_path,
                    "placeholder_token": config.token,
                    "coarse_class_text": config.class_word,
                    "repeats": config.repeats,
                    "size": config.resolution,
                    "interpolation": config.resampler,
                    "center_crop": config.center_crop,
                    "flip_p": config.flip_percent,
                    "token_only": config.token_only,
                    "per_image_tokens": False,
                    "mixing_prob": config.mix_probability,
                    "shuffle": True if config.mix_probability > 0.0 else False
                }
            },
            "reg": reg_block if config.regularization_images_folder_path and not config.token_only else False,
            "validation": val_block if config.validate_training else False
            }
        }
    }

    return data_config


def get_dreambooth_model_data_config(model_config, data_config, lightning_config) -> dict:
    return {
        "model": model_config,
        "data": data_config,
        "lightning": lightning_config,
    }


def get_dreambooth_lightning_config(config: JoePennaDreamboothConfigSchemaV1) -> dict:
    cb = callbacks(config)
    lightning_config = {
        "modelcheckpoint": {
            "params": {
                "every_n_train_steps": 500 if config.save_every_x_steps <= 0 else config.save_every_x_steps,
            }
        },
        "callbacks": {
            "image_logger": cb.image_logger(),
        },
        "trainer": {
            "accelerator": "gpu",
            "devices": f"{config.gpu},",
            "benchmark": True,
            "accumulate_grad_batches": 1,
            "max_steps": config.max_training_steps,
        }
    }

    if config.save_every_x_steps > 0:
        lightning_config["callbacks"]["metrics_over_trainsteps_checkpoint"] = cb.metrics_over_trainsteps_checkpoint()

    return lightning_config


def get_dreambooth_trainer_config(config: JoePennaDreamboothConfigSchemaV1, model, lightning_config) -> dict:
    cb = callbacks(config)
    trainer_config = {
        "logger": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                "name": "CSVLogger",
                "save_dir": config.log_directory(),
            }
        },
        "checkpoint_callback": cb.model_checkpoint()
    }

    trainer_config.update(lightning_config["trainer"])

    if hasattr(model, "monitor"):
        trainer_config["checkpoint_callback"]["params"]["monitor"] = model.monitor
        trainer_config["checkpoint_callback"]["params"]["save_top_k"] = 1

        if config.debug:
            print(f"Monitoring {model.monitor} as checkpoint metric.")

    return trainer_config


def get_dreambooth_trainer_kwargs(config: JoePennaDreamboothConfigSchemaV1, trainer_config, callbacks_config) -> dict:
    trainer_kwargs = dict()
    trainer_kwargs["logger"] = instantiate_from_config(trainer_config["logger"])
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_config[k]) for k in callbacks_config]
    trainer_kwargs["max_steps"] = config.max_training_steps
    trainer_kwargs["plugins"] = PruningCheckpointIO()

    return trainer_kwargs


def get_dreambooth_callbacks_config(config: JoePennaDreamboothConfigSchemaV1, model_data_config,
                                    lightning_config) -> dict:
    cb = callbacks(config)
    callbacks_config = {
        "setup_callback": cb.setup_callback(
            model_data_config=model_data_config,
            lightning_config=lightning_config,
        ),
        "image_logger": cb.image_logger(),
        "learning_rate_logger": cb.learning_rate_logger(),
        "cuda_callback": cb.cuda_callback(),
        "checkpoint_callback": cb.model_checkpoint(),
    }

    if config.save_every_x_steps > 0:
        callbacks_config["metrics_over_trainsteps_checkpoint"] = cb.metrics_over_trainsteps_checkpoint()

    return callbacks_config
