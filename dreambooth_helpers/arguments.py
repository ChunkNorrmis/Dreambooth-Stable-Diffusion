import argparse

from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1


def parse_arguments() -> JoePennaDreamboothConfigSchemaV1:
    def _get_parser(**parser_kwargs):
        def str2bool(v):
            if isinstance(v, bool):
                return v
            if v.lower() in ("yes", "true", "t", "y", "1"):
                return True
            elif v.lower() in ("no", "false", "f", "n", "0"):
                return False
            else:
                raise argparse.ArgumentTypeError("Boolean value expected.")

        parser = argparse.ArgumentParser(**parser_kwargs)
        
        parser.add_argument(
            "--config_file_path",
            type=str,
            required=False,
            default=None,
            help="A config file containing all of your variables"
        )
        parser.add_argument(
            "--project_name",
            type=str,
            required=True,
            help="Name of the project"
        )
        parser.add_argument(
            "--debug",
            required=False,
            action="store_true",
            help="Enable debug logging",
        )
        parser.add_argument(
            "--seed",
            type=int,
            required=False,
            default=1337,
            help="seed for seed_everything",
        )
        parser.add_argument(
            "--max_training_steps",
            type=int,
            required=False,
            default=0,
            help="Number of training steps to run"
        )
        parser.add_argument(
            "--training_model",
            type=str,
            required=True,
            help="Path to model to train (model.ckpt)"
        )
        parser.add_argument(
            "--token_only",
            action="store_true",
            required=False,
            help="Train only using the token and no class."
        )
        parser.add_argument(
            "--token",
            type=str,
            required=True,
            help="Unique token you want to represent your trained model. Ex: firstNameLastName."
        )
        parser.add_argument(
            "--training_images",
            type=str,
            required=True,
            help="Path to training images directory"
        )
        parser.add_argument(
            "--class_word",
            type=str,
            required=False,
            default=None,
            help="Match class_word to the category of images you want to train. Example: 'man', 'woman', 'dog', or 'artstyle'."
        )
        parser.add_argument(
            "--regularization_images",
            type=str,
            required=False,
            default=None,
            help="Path to directory with regularization images"
        )
        parser.add_argument(
            "--mirror_rate",
            type=float,
            required=False,
            default=0.25,
            help="Image Mirroring probability"
                    "Example: if set to 0.5, will flip (mirror) your training images 50% of the time."
                    "This helps expand your dataset without needing to include more training images."
                    "This can lead to worse results for face training since most people's faces are not perfectly symmetrical."
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            required=False,
            default=5.0e-07,
            help="Set the learning rate. Defaults to 1.0e-06 (0.000001).  Accepts scientific notation."
        )
        parser.add_argument(
            "--save_every_x_steps",
            type=int,
            required=False,
            default=0,
            help="Saves a checkpoint every x steps"
        )
        parser.add_argument(
            "--gpu",
            type=int,
            required=False,
            default=0,
            help="Specify a GPU other than 0 to use for training.  Multi-GPU support is not currently implemented."
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            required=False,
            default=1,
            help="image batch size and number of epochs to perform for iterable datasets" 
        )
        parser.add_argument(
            "--workers",
            type=int,
            required=False,
            default=0,
            help="number of workers to deploy for data preprocessing"
        )
        parser.add_argument(
            "--repeats",
            type=int,
            required=False,
            default=100,
            help="number of repeats per image duirng training split"
        )
        parser.add_argument(
            "--val_repeats",
            type=int,
            required=False,
            default=10,
            help="number of repeats per image duirng validation split"
        )
        parser.add_argument(
            "--resolution",
            type=int,
            required=False,
            default=512,
            help="image resolution(N)^2 (N x N)"
        )
        parser.add_argument(
            "--resampler",
            type=str,
            required=False,
            choices=["bilinear", "bicubic", "lanczos"],
            default="lanczos"
        )
        parser.add_argument(
            "--center_crop",
            required=False,
            action="store_true",
            help="make ANY polygon your new favorite rhombus!!!11!1"
        )
        parser.add_argument(
            "--shuffle_rate",
            type=float,
            required=False,
            default=0.25,
            help='"random" image selection "probability"'
        )
        parser.add_argument(
            "--auxiliary",
            type=str,
            nargs="+",
            choices=["validation", "test", "predict"],
            required=False,
            default="validation",
            help="additional splits to perform in addition to training the training split"
        )
            
        return parser

    parser = _get_parser()
    opt, unknown = parser.parse_known_args()

    config = JoePennaDreamboothConfigSchemaV1()

    if opt.config_file_path is not None:
        config.saturate_from_file(config_file_path=opt.config_file_path)
    else:
        config.saturate(
            project_name=opt.project_name,
            seed=opt.seed,
            debug=opt.debug,
            gpu=opt.gpu,
            max_training_steps=opt.max_training_steps,
            save_every_x_steps=opt.save_every_x_steps,
            training_images_folder_path=opt.training_images,
            regularization_images_folder_path=opt.regularization_images,
            token=opt.token,
            token_only=opt.token_only,
            class_word=opt.class_word,
            flip_percent=opt.mirror_rate,
            learning_rate=opt.learning_rate,
            model_repo_id=None,
            model_path=opt.training_model,
            batch_size=opt.batch_size,
            num_workers=opt.workers,
            repeats=opt.repeats,
            val_repeats=opt.val_repeats,
            resolution=opt.resolution,
            resampler=opt.resampler,
            center_crop=opt.center_crop,
            mix_probability=opt.shuffle_rate,
            auxiliary=[opt.auxiliary]
        )

    return config
