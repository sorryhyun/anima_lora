"""The MOSS-Image trainer on an int8 base (the MOSS counterpart of ``train_int8.py``).

Strips ``--int8 {off,fwd,full}`` and runs ``library.moss.train.run_train`` on the
``TrainRequest`` built from the rest. ``off`` runs the trainer untouched. Otherwise
two patches go in first:

- ``library.moss.lora.LoRANetwork.apply_to`` converts the six per-block linears
  (``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}``) before patching them: the
  patched forward captures ``module.__class__.forward`` at that point. That is after
  the network's ``__init__``, so ``down_init weight_svd*`` still reads bf16, and
  before ``compile_blocks``. Each converted linear keeps an empty bf16 ``weight``
  buffer, since the block forward reads ``proj["q"].weight.dtype``.
- ``LoRANetwork.metadata`` stamps ``int8_base=<mode>`` and ``int8_base_id=rtn``.

Refused: ``blocks_to_swap != 0`` (the offloader moves only ``.weight``) and
``algo loha`` (it builds ``W + ΔW``).

    make daemon-run ARGS="--stall-timeout 900 project/int8_train/src/train_int8_moss.py \\
        --int8 full -- --config configs/moss/lora.toml --blocks_to_swap 0 --grad_checkpointing"
"""

import argparse
import os
import re

MOSS_TARGET = re.compile(r"^layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)$")


def install(mode: str, req) -> None:
    """The two patches and the refusals above, for ``mode`` ``fwd`` or ``full``.
    ``profile_step_moss.py --int8`` calls this too."""
    if req.blocks_to_swap != 0:
        raise SystemExit(
            "int8 base: --blocks_to_swap 0 is required (the offloader moves only "
            ".weight, not wq / wq_t)"
        )
    if req.algo == "loha":
        raise SystemExit("int8 base: loha builds W + ΔW against the base weight")

    import torch

    from int8_linear import convert_  # sibling module
    from library.moss.lora import LoRANetwork

    converted: list[str] = []
    orig_apply_to = LoRANetwork.apply_to
    orig_metadata = LoRANetwork.metadata

    def apply_to(self):
        device = torch.device("cuda")
        for name, mod in self._targets:
            if not MOSS_TARGET.match(name):
                continue
            mod.to(device)
            convert_(mod, mode)
            if mode != "fwd":
                mod.register_buffer(
                    "weight",
                    torch.empty(0, dtype=torch.bfloat16, device=device),
                    persistent=False,
                )
            converted.append(name)
        torch.cuda.empty_cache()
        print(
            f"int8 {mode}: converted {len(converted)} linears, round to nearest",
            flush=True,
        )
        return orig_apply_to(self)

    def metadata(self, **extra):
        if not converted:
            raise SystemExit("int8 base: no DiT linear was converted")
        return orig_metadata(self, int8_base=mode, int8_base_id="rtn", **extra)

    LoRANetwork.apply_to = apply_to
    LoRANetwork.metadata = metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--int8", choices=("off", "fwd", "full"), required=True)
    own, train_argv = ap.parse_known_args()
    if train_argv and train_argv[0] == "--":
        train_argv = train_argv[1:]

    from library.moss.requests import TrainRequest

    req = TrainRequest.from_argv(train_argv)
    if req.expandable_segments:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if own.int8 != "off":
        install(own.int8, req)
    from library.moss.train import run_train

    run_train(req)
