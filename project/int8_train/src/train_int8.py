"""Phase 2 entry point: the stock Anima trainer on an int8 base.

Strips ``--int8 {off,fwd,full}`` and runs ``library.anima.trainer.cli.main`` with the
rest. ``off`` runs the trainer untouched, so a bf16 arm goes through the same command.
Otherwise two patches go in first:

- ``LoRANetwork.apply_to`` converts the DiT's large linears (``int8_linear``) before
  the LoRA modules capture ``org_forward``. That is after ``create_network``, so
  ``weight_svd`` / ``grad_svd`` init still read bf16, and before ``compile_blocks``.
- ``AnimaTrainer._update_metadata`` stamps ``ss_int8_base=<mode>`` into the LoRA, and
  fails if no conversion happened (a network class that is not ``LoRANetwork``).

``--int8_file`` (a ``dit_gptq.py`` file made from the same checkpoint) supplies the
forward copies instead of round to nearest, and its ``int8_id`` is stamped as
``ss_int8_base_id``; without it the stamp is ``rtn``.

Refused: block swap (the offloader moves only ``.weight``), and every
``LoRANetwork`` method that reads or writes the base weight.

    make daemon-run ARGS="--stall-timeout 900 project/int8_train/src/train_int8.py \
        --int8 full --method lora --preset default"
"""

import argparse
import sys

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()  # train.py sets this before torch loads; match it


def install(mode: str, int8_file: str | None) -> None:
    """The two patches and the refusals above, for ``mode`` ``fwd`` or ``full``.
    ``profile_step.py --int8`` calls this too."""
    from safetensors import safe_open

    from int8_linear import convert_dit_  # sibling module
    from library.env import resolve_under_home

    int8_id = "rtn"
    if int8_file:
        int8_file = str(resolve_under_home(int8_file))
        with safe_open(int8_file, framework="pt") as f:
            int8_id = (f.metadata() or {})["int8_id"]
    from library.anima.trainer.trainer import AnimaTrainer
    from networks.lora_anima.network import LoRANetwork

    converted: list[str] = []
    orig_apply_to = LoRANetwork.apply_to
    orig_update_metadata = AnimaTrainer._update_metadata

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if apply_unet:
            if getattr(unet, "blocks_to_swap", None):
                raise SystemExit(
                    "int8 base: blocks_to_swap > 0 is refused (the offloader moves "
                    "only .weight, not wq / wq_t)"
                )
            converted.extend(convert_dit_(unet, mode, int8_file))
            print(
                f"int8 {mode}: converted {len(converted)} linears, forward copy "
                f"{int8_file or 'round to nearest'}",
                flush=True,
            )
        return orig_apply_to(self, text_encoders, unet, apply_text_encoder, apply_unet)

    def update_metadata(metadata, args):
        if not converted:
            raise SystemExit(
                "int8 base: no DiT linear was converted; the network is not a "
                "LoRANetwork, so this run would train on bf16"
            )
        orig_update_metadata(metadata, args)
        metadata["ss_int8_base"] = mode
        metadata["ss_int8_base_id"] = int8_id

    def refuse(name):
        def method(self, *a, **k):
            raise SystemExit(f"int8 base: LoRANetwork.{name} reads the base weight")

        return method

    LoRANetwork.apply_to = apply_to
    AnimaTrainer._update_metadata = staticmethod(update_metadata)
    for name in (
        "backup_weights",
        "restore_weights",
        "pre_calculation",
        "fuse_weights",
        "unfuse_weights",
    ):
        setattr(LoRANetwork, name, refuse(name))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--int8", choices=("off", "fwd", "full"), required=True)
    ap.add_argument("--int8_file", default=None)
    own, train_argv = ap.parse_known_args()
    if own.int8_file and own.int8 == "off":
        ap.error("--int8_file needs --int8 fwd or full")

    from library.anima.trainer import cli  # noqa: E402
    from library.log import setup_logging  # noqa: E402

    setup_logging()
    if own.int8 != "off":
        install(own.int8, own.int8_file)
    sys.exit(cli.main(train_argv))
