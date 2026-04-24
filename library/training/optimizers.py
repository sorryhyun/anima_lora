import argparse
import ast
import importlib
import logging
from typing import Callable, Tuple

import torch
import transformers
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def get_optimizer(args, trainable_params) -> tuple[str, str, object]:
    optimizer_type = args.optimizer_type
    if args.use_8bit_adam:
        assert not args.use_lion_optimizer, (
            "both option use_8bit_adam and use_lion_optimizer are specified"
        )
        assert optimizer_type is None or optimizer_type == "", (
            "both option use_8bit_adam and optimizer_type are specified"
        )
        optimizer_type = "AdamW8bit"

    elif args.use_lion_optimizer:
        assert optimizer_type is None or optimizer_type == "", (
            "both option use_lion_optimizer and optimizer_type are specified"
        )
        optimizer_type = "Lion"

    if optimizer_type is None or optimizer_type == "":
        optimizer_type = "AdamW"
    optimizer_type = optimizer_type.lower()

    if args.fused_backward_pass:
        assert optimizer_type == "Adafactor".lower(), (
            "fused_backward_pass currently only works with optimizer_type Adafactor"
        )
        assert args.gradient_accumulation_steps == 1, (
            "fused_backward_pass does not work with gradient_accumulation_steps > 1"
        )

    optimizer_kwargs = {}
    if args.optimizer_args is not None and len(args.optimizer_args) > 0:
        for arg in args.optimizer_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value

    lr = args.learning_rate
    optimizer = None
    optimizer_class = None

    if optimizer_type == "Lion".lower():
        try:
            import lion_pytorch
        except ImportError:
            raise ImportError("No lion_pytorch")
        logger.info(f"use Lion optimizer | {optimizer_kwargs}")
        optimizer_class = lion_pytorch.Lion
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.endswith("8bit".lower()):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes")

        if optimizer_type == "AdamW8bit".lower():
            logger.info(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
            optimizer_class = bnb.optim.AdamW8bit
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "SGDNesterov8bit".lower():
            logger.info(f"use 8-bit SGD with Nesterov optimizer | {optimizer_kwargs}")
            if "momentum" not in optimizer_kwargs:
                logger.warning(
                    "8-bit SGD with Nesterov must be with momentum, set momentum to 0.9"
                )
                optimizer_kwargs["momentum"] = 0.9

            optimizer_class = bnb.optim.SGD8bit
            optimizer = optimizer_class(
                trainable_params, lr=lr, nesterov=True, **optimizer_kwargs
            )

        elif optimizer_type == "Lion8bit".lower():
            logger.info(f"use 8-bit Lion optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.Lion8bit
            except AttributeError:
                raise AttributeError(
                    "No Lion8bit. Please install bitsandbytes 0.38.0 or later."
                )
        elif optimizer_type == "PagedAdamW8bit".lower():
            logger.info(f"use 8-bit PagedAdamW optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.PagedAdamW8bit
            except AttributeError:
                raise AttributeError(
                    "No PagedAdamW8bit. Please install bitsandbytes 0.39.0 or later."
                )
        elif optimizer_type == "PagedLion8bit".lower():
            logger.info(f"use 8-bit Paged Lion optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.PagedLion8bit
            except AttributeError:
                raise AttributeError(
                    "No PagedLion8bit. Please install bitsandbytes 0.39.0 or later."
                )

        if optimizer_class is not None:
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "PagedAdamW".lower():
        logger.info(f"use PagedAdamW optimizer | {optimizer_kwargs}")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes")
        try:
            optimizer_class = bnb.optim.PagedAdamW
        except AttributeError:
            raise AttributeError(
                "No PagedAdamW. Please install bitsandbytes 0.39.0 or later."
            )
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "PagedAdamW32bit".lower():
        logger.info(f"use 32-bit PagedAdamW optimizer | {optimizer_kwargs}")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes")
        try:
            optimizer_class = bnb.optim.PagedAdamW32bit
        except AttributeError:
            raise AttributeError(
                "No PagedAdamW32bit. Please install bitsandbytes 0.39.0 or later."
            )
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "SGDNesterov".lower():
        logger.info(f"use SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            logger.info("SGD with Nesterov must be with momentum, set momentum to 0.9")
            optimizer_kwargs["momentum"] = 0.9

        optimizer_class = torch.optim.SGD
        optimizer = optimizer_class(
            trainable_params, lr=lr, nesterov=True, **optimizer_kwargs
        )

    elif (
        optimizer_type.startswith("DAdapt".lower())
        or optimizer_type == "Prodigy".lower()
    ):
        actual_lr = lr
        lr_count = 1
        if isinstance(trainable_params, list) and isinstance(trainable_params[0], dict):
            lrs = set()
            actual_lr = trainable_params[0].get("lr", actual_lr)
            for group in trainable_params:
                lrs.add(group.get("lr", actual_lr))
            lr_count = len(lrs)

        if actual_lr <= 0.1:
            logger.warning(
                f"learning rate is too low. If using D-Adaptation or Prodigy, set learning rate around 1.0: lr={actual_lr}"
            )
            logger.warning("recommend option: lr=1.0")
        if lr_count > 1:
            logger.warning(
                f"when multiple learning rates are specified with dadaptation, only the first one will take effect: lr={actual_lr}"
            )

        if optimizer_type.startswith("DAdapt".lower()):
            try:
                import dadaptation
                import dadaptation.experimental as experimental
            except ImportError:
                raise ImportError("No dadaptation")

            if (
                optimizer_type == "DAdaptation".lower()
                or optimizer_type == "DAdaptAdamPreprint".lower()
            ):
                optimizer_class = experimental.DAdaptAdamPreprint
                logger.info(
                    f"use D-Adaptation AdamPreprint optimizer | {optimizer_kwargs}"
                )
            elif optimizer_type == "DAdaptAdaGrad".lower():
                optimizer_class = dadaptation.DAdaptAdaGrad
                logger.info(f"use D-Adaptation AdaGrad optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdam".lower():
                optimizer_class = dadaptation.DAdaptAdam
                logger.info(f"use D-Adaptation Adam optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdan".lower():
                optimizer_class = dadaptation.DAdaptAdan
                logger.info(f"use D-Adaptation Adan optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdanIP".lower():
                optimizer_class = experimental.DAdaptAdanIP
                logger.info(f"use D-Adaptation AdanIP optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptLion".lower():
                optimizer_class = dadaptation.DAdaptLion
                logger.info(f"use D-Adaptation Lion optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptSGD".lower():
                optimizer_class = dadaptation.DAdaptSGD
                logger.info(f"use D-Adaptation SGD optimizer | {optimizer_kwargs}")
            else:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}")

            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
        else:
            try:
                import prodigyopt
            except ImportError:
                raise ImportError("No Prodigy")

            logger.info(f"use Prodigy optimizer | {optimizer_kwargs}")
            optimizer_class = prodigyopt.Prodigy
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Adafactor".lower():
        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get(
            "warmup_init", False
        ):
            logger.info("set relative_step to True because warmup_init is True")
            optimizer_kwargs["relative_step"] = True
        logger.info(f"use Adafactor optimizer | {optimizer_kwargs}")

        if optimizer_kwargs["relative_step"]:
            logger.info("relative_step is true")
            if lr != 0.0:
                logger.warning("learning rate is used as initial_lr")
            args.learning_rate = None

            if isinstance(trainable_params, list) and isinstance(
                trainable_params[0], dict
            ):
                has_group_lr = False
                for group in trainable_params:
                    p = group.pop("lr", None)
                    has_group_lr = has_group_lr or (p is not None)

                if has_group_lr:
                    logger.warning("unet_lr and text_encoder_lr are ignored")
                    args.unet_lr = None
                    args.text_encoder_lr = None

            if args.lr_scheduler != "adafactor":
                logger.info("use adafactor_scheduler")
            args.lr_scheduler = f"adafactor:{lr}"

            lr = None
        else:
            if args.max_grad_norm != 0.0:
                logger.warning(
                    "because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0"
                )
            if args.lr_scheduler != "constant_with_warmup":
                logger.warning("constant_with_warmup will be good")
            if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                logger.warning("clip_threshold=1.0 will be good")

        optimizer_class = transformers.optimization.Adafactor
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "AdamW".lower():
        logger.info(f"use AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    # elif optimizer_type == "Rose".lower():
    #     # Rose (MatthewK78/Rose) — Range-Of-Slice Equilibration, stateless.
    #     # Disabled: kept as reference. Clone https://github.com/MatthewK78/Rose
    #     # into anima_lora/Rose/ to re-enable.
    #     import sys
    #     from pathlib import Path
    #
    #     rose_dir = Path(__file__).resolve().parents[2] / "Rose"
    #     if not (rose_dir / "rose.py").exists():
    #         raise ImportError(
    #             f"Rose optimizer: expected {rose_dir}/rose.py. "
    #             "Clone https://github.com/MatthewK78/Rose into anima_lora/Rose."
    #         )
    #     if str(rose_dir) not in sys.path:
    #         sys.path.insert(0, str(rose_dir))
    #     from rose import Rose
    #
    #     logger.info(f"use Rose optimizer | {optimizer_kwargs}")
    #     optimizer_class = Rose
    #     optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.endswith("schedulefree".lower()):
        try:
            import schedulefree as sf
        except ImportError:
            raise ImportError("No schedulefree")

        if optimizer_type == "RAdamScheduleFree".lower():
            optimizer_class = sf.RAdamScheduleFree
            logger.info(f"use RAdamScheduleFree optimizer | {optimizer_kwargs}")
        elif optimizer_type == "AdamWScheduleFree".lower():
            optimizer_class = sf.AdamWScheduleFree
            logger.info(f"use AdamWScheduleFree optimizer | {optimizer_kwargs}")
        elif optimizer_type == "SGDScheduleFree".lower():
            optimizer_class = sf.SGDScheduleFree
            logger.info(f"use SGDScheduleFree optimizer | {optimizer_kwargs}")
        else:
            optimizer_class = None

        if optimizer_class is not None:
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    if optimizer is None:
        case_sensitive_optimizer_type = args.optimizer_type
        logger.info(f"use {case_sensitive_optimizer_type} | {optimizer_kwargs}")

        if "." not in case_sensitive_optimizer_type:
            optimizer_module = torch.optim
        else:
            values = case_sensitive_optimizer_type.split(".")
            optimizer_module = importlib.import_module(".".join(values[:-1]))
            case_sensitive_optimizer_type = values[-1]

        optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    if hasattr(optimizer, "train") and callable(optimizer.train):
        optimizer.train()

    return optimizer_name, optimizer_args, optimizer


def get_optimizer_train_eval_fn(
    optimizer: Optimizer, args: argparse.Namespace
) -> Tuple[Callable, Callable]:
    if not is_schedulefree_optimizer(optimizer, args):
        return lambda: None, lambda: None

    train_fn = optimizer.train
    eval_fn = optimizer.eval

    return train_fn, eval_fn


def is_schedulefree_optimizer(optimizer: Optimizer, args: argparse.Namespace) -> bool:
    return args.optimizer_type.lower().endswith("schedulefree".lower())
