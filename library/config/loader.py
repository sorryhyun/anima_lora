import argparse
from dataclasses import (
    asdict,
    dataclass,
)
import functools
import random
from textwrap import dedent, indent
import json
from pathlib import Path

# from toolz import curry
from typing import Dict, List, Optional, Sequence, Tuple, Union

import toml
import voluptuous
from voluptuous import (
    Any,
    ExactSequence,
    MultipleInvalid,
    Object,
    Required,
    Schema,
)

from library import train_util
from library.train_util import (
    DreamBoothSubset,
    DreamBoothDataset,
    DatasetGroup,
)
from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


def add_config_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dataset_config",
        type=Path,
        default=None,
        help="config file for detail settings",
    )


@dataclass
class BaseSubsetParams:
    image_dir: Optional[str] = None
    num_repeats: int = 1
    sample_ratio: float = 1.0
    shuffle_caption: bool = False
    caption_separator: str = (",",)
    keep_tokens: int = 0
    keep_tokens_separator: str = (None,)
    secondary_separator: Optional[str] = None
    enable_wildcard: bool = False
    color_aug: bool = False
    flip_aug: bool = False
    face_crop_aug_range: Optional[Tuple[float, float]] = None
    random_crop: bool = False
    caption_prefix: Optional[str] = None
    caption_suffix: Optional[str] = None
    caption_dropout_rate: float = 0.0
    caption_dropout_every_n_epochs: int = 0
    caption_tag_dropout_rate: float = 0.0
    token_warmup_min: int = 1
    token_warmup_step: float = 0
    custom_attributes: Optional[Dict[str, Any]] = None
    validation_seed: int = 0
    validation_split: float = 0.0
    resize_interpolation: Optional[str] = None


@dataclass
class DreamBoothSubsetParams(BaseSubsetParams):
    is_reg: bool = False
    class_tokens: Optional[str] = None
    caption_extension: str = ".caption"
    cache_info: bool = False
    alpha_mask: bool = False
    mask_dir: Optional[str] = None


@dataclass
class BaseDatasetParams:
    resolution: Optional[Tuple[int, int]] = None
    network_multiplier: float = 1.0
    debug_dataset: bool = False
    validation_seed: Optional[int] = None
    validation_split: float = 0.0
    resize_interpolation: Optional[str] = None


@dataclass
class DreamBoothDatasetParams(BaseDatasetParams):
    batch_size: int = 1
    enable_bucket: bool = False
    min_bucket_reso: int = 256
    max_bucket_reso: int = 1024
    bucket_reso_steps: int = 64
    bucket_no_upscale: bool = False
    prior_loss_weight: float = 1.0


@dataclass
class SubsetBlueprint:
    params: DreamBoothSubsetParams


@dataclass
class DatasetBlueprint:
    params: DreamBoothDatasetParams
    subsets: Sequence[SubsetBlueprint]


@dataclass
class DatasetGroupBlueprint:
    datasets: Sequence[DatasetBlueprint]


@dataclass
class Blueprint:
    dataset_group: DatasetGroupBlueprint


class ConfigSanitizer:
    # @curry
    @staticmethod
    def __validate_and_convert_twodim(klass, value: Sequence) -> Tuple:
        Schema(ExactSequence([klass, klass]))(value)
        return tuple(value)

    # @curry
    @staticmethod
    def __validate_and_convert_scalar_or_twodim(
        klass, value: Union[float, Sequence]
    ) -> Tuple:
        Schema(Any(klass, ExactSequence([klass, klass])))(value)
        try:
            Schema(klass)(value)
            return (value, value)
        except Exception:
            return ConfigSanitizer.__validate_and_convert_twodim(klass, value)

    # subset schema
    SUBSET_ASCENDABLE_SCHEMA = {
        "color_aug": bool,
        "face_crop_aug_range": functools.partial(
            __validate_and_convert_twodim.__func__, float
        ),
        "flip_aug": bool,
        "num_repeats": int,
        "sample_ratio": Any(float, int),
        "random_crop": bool,
        "shuffle_caption": bool,
        "keep_tokens": int,
        "keep_tokens_separator": str,
        "secondary_separator": str,
        "caption_separator": str,
        "enable_wildcard": bool,
        "token_warmup_min": int,
        "token_warmup_step": Any(float, int),
        "caption_prefix": str,
        "caption_suffix": str,
        "custom_attributes": dict,
        "resize_interpolation": str,
    }
    # DO means DropOut
    DO_SUBSET_ASCENDABLE_SCHEMA = {
        "caption_dropout_every_n_epochs": int,
        "caption_dropout_rate": Any(float, int),
        "caption_tag_dropout_rate": Any(float, int),
    }
    # DB means DreamBooth
    DB_SUBSET_ASCENDABLE_SCHEMA = {
        "caption_extension": str,
        "class_tokens": str,
        "cache_info": bool,
    }
    DB_SUBSET_DISTINCT_SCHEMA = {
        Required("image_dir"): str,
        "is_reg": bool,
        "alpha_mask": bool,
    }
    # datasets schema
    DATASET_ASCENDABLE_SCHEMA = {
        "batch_size": int,
        "bucket_no_upscale": bool,
        "bucket_reso_steps": int,
        "enable_bucket": bool,
        "max_bucket_reso": int,
        "min_bucket_reso": int,
        "validation_seed": int,
        "validation_split": float,
        "resolution": functools.partial(
            __validate_and_convert_scalar_or_twodim.__func__, int
        ),
        "network_multiplier": float,
        "resize_interpolation": str,
    }

    # options handled by argparse but not handled by user config
    ARGPARSE_SPECIFIC_SCHEMA = {
        "debug_dataset": bool,
        "max_token_length": Any(None, int),
        "prior_loss_weight": Any(float, int),
    }
    # for handling default None value of argparse
    ARGPARSE_NULLABLE_OPTNAMES = [
        "face_crop_aug_range",
        "resolution",
    ]
    # prepare map because option name may differ among argparse and user config
    ARGPARSE_OPTNAME_TO_CONFIG_OPTNAME = {
        "train_batch_size": "batch_size",
        "dataset_repeats": "num_repeats",
    }

    def __init__(self, support_dropout: bool) -> None:
        self.db_subset_schema = self.__merge_dict(
            self.SUBSET_ASCENDABLE_SCHEMA,
            self.DB_SUBSET_DISTINCT_SCHEMA,
            self.DB_SUBSET_ASCENDABLE_SCHEMA,
            self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
        )

        self.dataset_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.SUBSET_ASCENDABLE_SCHEMA,
            self.DB_SUBSET_ASCENDABLE_SCHEMA,
            self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
            {"subsets": [self.db_subset_schema]},
        )

        self.general_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.SUBSET_ASCENDABLE_SCHEMA,
            self.DB_SUBSET_ASCENDABLE_SCHEMA,
            self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
        )

        self.user_config_validator = Schema(
            {
                "general": self.general_schema,
                "datasets": [self.dataset_schema],
            }
        )

        self.argparse_schema = self.__merge_dict(
            self.general_schema,
            self.ARGPARSE_SPECIFIC_SCHEMA,
            {
                optname: Any(None, self.general_schema[optname])
                for optname in self.ARGPARSE_NULLABLE_OPTNAMES
            },
            {
                a_name: self.general_schema[c_name]
                for a_name, c_name in self.ARGPARSE_OPTNAME_TO_CONFIG_OPTNAME.items()
            },
        )

        self.argparse_config_validator = Schema(
            Object(self.argparse_schema), extra=voluptuous.ALLOW_EXTRA
        )

    def sanitize_user_config(self, user_config: dict) -> dict:
        try:
            return self.user_config_validator(user_config)
        except MultipleInvalid:
            logger.error("Invalid user config")
            raise

    # NOTE: In nature, argument parser result is not needed to be sanitize
    #   However this will help us to detect program bug
    def sanitize_argparse_namespace(
        self, argparse_namespace: argparse.Namespace
    ) -> argparse.Namespace:
        try:
            return self.argparse_config_validator(argparse_namespace)
        except MultipleInvalid:
            logger.error("Invalid cmdline parsed arguments.")
            raise

    # NOTE: value would be overwritten by latter dict if there is already the same key
    @staticmethod
    def __merge_dict(*dict_list: dict) -> dict:
        merged = {}
        for schema in dict_list:
            # merged |= schema
            for k, v in schema.items():
                merged[k] = v
        return merged


class BlueprintGenerator:
    BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME = {}

    def __init__(self, sanitizer: ConfigSanitizer):
        self.sanitizer = sanitizer

    # runtime_params is for parameters which is only configurable on runtime, such as tokenizer
    def generate(
        self,
        user_config: dict,
        argparse_namespace: argparse.Namespace,
        **runtime_params,
    ) -> Blueprint:
        sanitized_user_config = self.sanitizer.sanitize_user_config(user_config)
        sanitized_argparse_namespace = self.sanitizer.sanitize_argparse_namespace(
            argparse_namespace
        )

        # convert argparse namespace to dict like config
        # NOTE: it is ok to have extra entries in dict
        optname_map = self.sanitizer.ARGPARSE_OPTNAME_TO_CONFIG_OPTNAME
        argparse_config = {
            optname_map.get(optname, optname): value
            for optname, value in vars(sanitized_argparse_namespace).items()
        }

        general_config = sanitized_user_config.get("general", {})

        dataset_blueprints = []
        for dataset_config in sanitized_user_config.get("datasets", []):
            subsets = dataset_config.get("subsets", [])

            subset_blueprints = []
            for subset_config in subsets:
                params = self.generate_params_by_fallbacks(
                    DreamBoothSubsetParams,
                    [
                        subset_config,
                        dataset_config,
                        general_config,
                        argparse_config,
                        runtime_params,
                    ],
                )
                subset_blueprints.append(SubsetBlueprint(params))

            params = self.generate_params_by_fallbacks(
                DreamBoothDatasetParams,
                [dataset_config, general_config, argparse_config, runtime_params],
            )
            dataset_blueprints.append(DatasetBlueprint(params, subset_blueprints))

        dataset_group_blueprint = DatasetGroupBlueprint(dataset_blueprints)

        return Blueprint(dataset_group_blueprint)

    @staticmethod
    def generate_params_by_fallbacks(param_klass, fallbacks: Sequence[dict]):
        name_map = BlueprintGenerator.BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME
        search_value = BlueprintGenerator.search_value
        default_params = asdict(param_klass())
        param_names = default_params.keys()

        params = {
            name: search_value(
                name_map.get(name, name), fallbacks, default_params.get(name)
            )
            for name in param_names
        }

        return param_klass(**params)

    @staticmethod
    def search_value(key: str, fallbacks: Sequence[dict], default_value=None):
        for cand in fallbacks:
            value = cand.get(key)
            if value is not None:
                return value

        return default_value


def generate_dataset_group_by_blueprint(
    dataset_group_blueprint: DatasetGroupBlueprint,
    constant_token_buckets: bool = False,
) -> Tuple[DatasetGroup, Optional[DatasetGroup]]:
    datasets: List[DreamBoothDataset] = []

    for dataset_blueprint in dataset_group_blueprint.datasets:
        subsets = [
            DreamBoothSubset(**asdict(subset_blueprint.params))
            for subset_blueprint in dataset_blueprint.subsets
        ]
        dataset = DreamBoothDataset(
            subsets=subsets,
            **asdict(dataset_blueprint.params),
            is_training_dataset=True,
        )
        datasets.append(dataset)

    val_datasets: List[DreamBoothDataset] = []
    for dataset_blueprint in dataset_group_blueprint.datasets:
        if (
            dataset_blueprint.params.validation_split < 0.0
            or dataset_blueprint.params.validation_split > 1.0
        ):
            logging.warning(
                f"Dataset param `validation_split` ({dataset_blueprint.params.validation_split}) is not a valid number between 0.0 and 1.0, skipping validation split..."
            )
            continue

        if dataset_blueprint.params.validation_split == 0.0:
            continue

        subsets = [
            DreamBoothSubset(**asdict(subset_blueprint.params))
            for subset_blueprint in dataset_blueprint.subsets
        ]
        dataset = DreamBoothDataset(
            subsets=subsets,
            **asdict(dataset_blueprint.params),
            is_training_dataset=False,
        )
        val_datasets.append(dataset)

    def print_info(_datasets, dataset_type: str):
        info = ""
        for i, dataset in enumerate(_datasets):
            info += dedent(f"""\
                [{dataset_type} {i}]
                  batch_size: {dataset.batch_size}
                  resolution: {(dataset.width, dataset.height)}
                  resize_interpolation: {dataset.resize_interpolation}
                  enable_bucket: {dataset.enable_bucket}
            """)

            if dataset.enable_bucket:
                info += indent(
                    dedent(f"""\
                  min_bucket_reso: {dataset.min_bucket_reso}
                  max_bucket_reso: {dataset.max_bucket_reso}
                  bucket_reso_steps: {dataset.bucket_reso_steps}
                  bucket_no_upscale: {dataset.bucket_no_upscale}
                \n"""),
                    "  ",
                )
            else:
                info += "\n"

            for j, subset in enumerate(dataset.subsets):
                info += indent(
                    dedent(f"""\
                  [Subset {j} of {dataset_type} {i}]
                    image_dir: "{subset.image_dir}"
                    image_count: {subset.img_count}
                    num_repeats: {subset.num_repeats}
                    sample_ratio: {subset.sample_ratio}
                    shuffle_caption: {subset.shuffle_caption}
                    keep_tokens: {subset.keep_tokens}
                    caption_dropout_rate: {subset.caption_dropout_rate}
                    caption_dropout_every_n_epochs: {subset.caption_dropout_every_n_epochs}
                    caption_tag_dropout_rate: {subset.caption_tag_dropout_rate}
                    caption_prefix: {subset.caption_prefix}
                    caption_suffix: {subset.caption_suffix}
                    color_aug: {subset.color_aug}
                    flip_aug: {subset.flip_aug}
                    face_crop_aug_range: {subset.face_crop_aug_range}
                    random_crop: {subset.random_crop}
                    token_warmup_min: {subset.token_warmup_min},
                    token_warmup_step: {subset.token_warmup_step},
                    alpha_mask: {subset.alpha_mask}
                    resize_interpolation: {subset.resize_interpolation}
                    custom_attributes: {subset.custom_attributes}
                    is_reg: {subset.is_reg}
                    class_tokens: {subset.class_tokens}
                    caption_extension: {subset.caption_extension}
                """),
                    "  ",
                )

        logger.info(info)

    print_info(datasets, "Dataset")

    if len(val_datasets) > 0:
        print_info(val_datasets, "Validation Dataset")

    # make buckets first because it determines the length of dataset
    # and set the same seed for all datasets
    seed = random.randint(0, 2**31)  # actual seed is seed + epoch_no

    for i, dataset in enumerate(datasets):
        logger.info(f"[Prepare dataset {i}]")
        dataset.make_buckets(constant_token_buckets=constant_token_buckets)
        dataset.set_seed(seed)

    for i, dataset in enumerate(val_datasets):
        logger.info(f"[Prepare validation dataset {i}]")
        dataset.make_buckets(constant_token_buckets=constant_token_buckets)
        dataset.set_seed(seed)

    return (
        DatasetGroup(datasets),
        DatasetGroup(val_datasets) if val_datasets else None,
    )


def generate_dreambooth_subsets_config_by_subdirs(
    train_data_dir: Optional[str] = None, reg_data_dir: Optional[str] = None
):
    def extract_dreambooth_params(name: str) -> Tuple[int, str]:
        tokens = name.split("_")
        try:
            n_repeats = int(tokens[0])
        except ValueError:
            logger.warning("ignore directory without repeats")
            return 0, ""
        caption_by_folder = "_".join(tokens[1:])
        return n_repeats, caption_by_folder

    def generate(base_dir: Optional[str], is_reg: bool):
        if base_dir is None:
            return []

        base_dir: Path = Path(base_dir)
        if not base_dir.is_dir():
            return []

        subsets_config = []
        for subdir in base_dir.iterdir():
            if not subdir.is_dir():
                continue

            num_repeats, class_tokens = extract_dreambooth_params(subdir.name)
            if num_repeats < 1:
                continue

            subset_config = {
                "image_dir": str(subdir),
                "num_repeats": num_repeats,
                "is_reg": is_reg,
                "class_tokens": class_tokens,
            }
            subsets_config.append(subset_config)

        return subsets_config

    subsets_config = []
    subsets_config += generate(train_data_dir, False)
    subsets_config += generate(reg_data_dir, True)

    return subsets_config


def load_user_config(file: str) -> dict:
    file: Path = Path(file)
    if not file.is_file():
        raise ValueError("file not found")

    if file.name.lower().endswith(".json"):
        try:
            with open(file, "r") as f:
                config = json.load(f)
        except Exception:
            logger.error("Error on parsing JSON config file. Please check the format.")
            raise
    elif file.name.lower().endswith(".toml"):
        try:
            config = toml.load(file)
        except Exception:
            logger.error("Error on parsing TOML config file. Please check the format.")
            raise
    else:
        raise ValueError("not supported config file format")

    return config


# for config test
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--support_dropout", action="store_true")
    parser.add_argument("dataset_config")
    config_args, remain = parser.parse_known_args()

    parser = argparse.ArgumentParser()
    train_util.add_dataset_arguments(parser, True, False, config_args.support_dropout)
    train_util.add_training_arguments(parser, True)
    argparse_namespace = parser.parse_args(remain)
    train_util.prepare_dataset_args(argparse_namespace, False)

    logger.info("[argparse_namespace]")
    logger.info(f"{vars(argparse_namespace)}")

    user_config = load_user_config(config_args.dataset_config)

    logger.info("")
    logger.info("[user_config]")
    logger.info(f"{user_config}")

    sanitizer = ConfigSanitizer(config_args.support_dropout)
    sanitized_user_config = sanitizer.sanitize_user_config(user_config)

    logger.info("")
    logger.info("[sanitized_user_config]")
    logger.info(f"{sanitized_user_config}")

    blueprint = BlueprintGenerator(sanitizer).generate(user_config, argparse_namespace)

    logger.info("")
    logger.info("[blueprint]")
    logger.info(f"{blueprint}")
