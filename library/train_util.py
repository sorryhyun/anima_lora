# Local helpers that don't have a better home yet: SAI-spec metadata builder,
# argparse-namespace dataset arg normalization, prompt-file parsing, and
# accelerate tracker init. Everything else lives in library.datasets,
# library.training, library.runtime, or library.config — import from there.

import argparse
import json
import logging
import os
import re
import time
from typing import Dict, List

import safetensors
import toml
import torch
from accelerate import Accelerator

from library.models import sai_spec as sai_model_spec
from library.training import get_sanitized_config_or_none
from library.log import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

EPSILON = 1e-6


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """r
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata


def get_sai_model_spec_dataclass(
    args: argparse.Namespace,
    lora: bool,
    optional_metadata: dict[str, str] | None = None,
) -> sai_model_spec.ModelSpecMetadata:
    timestamp = time.time()

    title = args.metadata_title if args.metadata_title is not None else args.output_name

    if args.min_timestep is not None or args.max_timestep is not None:
        timesteps = (
            args.min_timestep if args.min_timestep is not None else 0,
            args.max_timestep if args.max_timestep is not None else 1000,
        )
    else:
        timesteps = None

    extracted_metadata = {}
    for attr_name in dir(args):
        if attr_name.startswith("metadata_") and not attr_name.startswith(
            "metadata___"
        ):
            value = getattr(args, attr_name, None)
            if value is not None:
                field_name = attr_name[9:]
                if field_name not in {"title", "author", "description", "license", "tags"}:
                    extracted_metadata[field_name] = value
    if optional_metadata:
        extracted_metadata.update(optional_metadata)

    return sai_model_spec.build_metadata_dataclass(
        lora,
        timestamp,
        title=title,
        reso=args.resolution,
        author=args.metadata_author,
        description=args.metadata_description,
        license=args.metadata_license,
        tags=args.metadata_tags,
        timesteps=timesteps,
        optional_metadata=extracted_metadata or None,
    )




def prepare_dataset_args(args: argparse.Namespace, support_metadata: bool):
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention
        args.caption_extention = None

    if args.resolution is not None:
        args.resolution = tuple([int(r) for r in args.resolution.split(",")])
        if len(args.resolution) == 1:
            args.resolution = (args.resolution[0], args.resolution[0])
        assert len(args.resolution) == 2, "resolution must be 'size' or 'width,height'"

    if args.face_crop_aug_range is not None:
        args.face_crop_aug_range = tuple(
            [float(r) for r in args.face_crop_aug_range.split(",")]
        )
        assert (
            len(args.face_crop_aug_range) == 2
            and args.face_crop_aug_range[0] <= args.face_crop_aug_range[1]
        ), "face_crop_aug_range must be two floats"
    else:
        args.face_crop_aug_range = None

    if support_metadata:
        if args.in_json is not None and (args.color_aug or args.random_crop):
            logger.warning(
                "latents in npz is ignored when color_aug or random_crop is True"
            )



def get_timesteps(
    min_timestep: int, max_timestep: int, b_size: int, device: torch.device
) -> torch.Tensor:
    if min_timestep < max_timestep:
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
    else:
        timesteps = torch.full((b_size,), max_timestep, device="cpu")
    timesteps = timesteps.long().to(device)
    return timesteps



def append_lr_to_logs(logs, lr_scheduler, optimizer_type, including_unet=True):
    names = []
    if including_unet:
        names.append("unet")
    names.append("text_encoder1")
    names.append("text_encoder2")
    names.append("text_encoder3")

    append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names):
    lrs = lr_scheduler.get_last_lr()

    for lr_index in range(len(lrs)):
        name = names[lr_index]
        logs["lr/" + name] = float(lrs[lr_index])

        if (
            optimizer_type.lower().startswith("DAdapt".lower())
            or optimizer_type.lower() == "Prodigy".lower()
        ):
            logs["lr/d*lr/" + name] = (
                lr_scheduler.optimizers[-1].param_groups[lr_index]["d"]
                * lr_scheduler.optimizers[-1].param_groups[lr_index]["lr"]
            )


def line_to_prompt_dict(line: str) -> dict:
    # subset of gen_img_diffusers
    prompt_args = line.split(" --")
    prompt_dict = {}
    prompt_dict["prompt"] = prompt_args[0]

    for parg in prompt_args:
        try:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["width"] = int(m.group(1))
                continue

            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["height"] = int(m.group(1))
                continue

            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["seed"] = int(m.group(1))
                continue

            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["sample_steps"] = max(1, min(1000, int(m.group(1))))
                continue

            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["scale"] = float(m.group(1))
                continue

            m = re.match(r"g ([\d\.]+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["guidance_scale"] = float(m.group(1))
                continue

            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["negative_prompt"] = m.group(1)
                continue

            m = re.match(r"ss (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["sample_sampler"] = m.group(1)
                continue

            m = re.match(r"cn (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["controlnet_image"] = m.group(1)
                continue

            m = re.match(r"ctr (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["cfg_trunc_ratio"] = float(m.group(1))
                continue

            m = re.match(r"rcfg (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["renorm_cfg"] = float(m.group(1))
                continue

            m = re.match(r"fs (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["flow_shift"] = m.group(1)
                continue

        except ValueError as ex:
            logger.error("Exception in parsing")
            logger.error(ex)

    return prompt_dict


def load_prompts(prompt_file: str) -> List[Dict]:
    # read prompts
    if prompt_file.endswith(".txt"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [
            line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"
        ]
    elif prompt_file.endswith(".toml"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [
            dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]
        ]
    elif prompt_file.endswith(".json"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    # preprocess prompts
    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)

        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    return prompts


def init_trackers(
    accelerator: Accelerator, args: argparse.Namespace, default_tracker_name: str
):
    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            default_tracker_name
            if args.log_tracker_name is None
            else args.log_tracker_name,
            config=get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

        if "wandb" in [tracker.name for tracker in accelerator.trackers]:
            wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)

            wandb_tracker.define_metric("epoch", hidden=True)
            wandb_tracker.define_metric("val_step", hidden=True)

            wandb_tracker.define_metric("global_step", hidden=True)


# endregion
