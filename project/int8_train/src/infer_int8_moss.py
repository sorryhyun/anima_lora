"""MOSS-Image generation on the int8 base (the MOSS counterpart of ``infer_int8.py``).

Strips ``--int8 {off,on}`` and ``--int8_file`` and runs ``library.moss.generate`` on the
``GenerateRequest`` built from the rest. ``off`` runs it untouched, so a bf16 arm goes
through the same command. ``on`` converts the 252 block linears (``int8_linear`` mode
``infer``: forward copy only, bf16 weight dropped), so the DiT holds ~5 GB on the card
rather than 10:

- With a LoRA, in ``LoRANetwork.apply_to``: after the network's ``__init__`` (LoKr reads
  the bf16 weight's shape) and before the hooks capture ``module.__class__.forward``.
  The base then runs ``int8(x, W_q) + ΔW x``, what ``TrainRequest.int8`` trained.
- Without one, right after ``load_transformer``.

``--int8_file`` (a ``dit_gptq_moss.py`` file made from the loaded DiT) supplies the
forward copies instead of round to nearest. Images land in
``<out_dir>/int8_<base id>/``: a saved latent is reused on its job fingerprint, which
does not name the base, so the arms of an A/B can share ``--out_dir``. A LoRA's
``int8_base`` / ``int8_base_id`` stamps are logged against this base. Refused:
``blocks_to_swap`` (generation on this base has run no swap) and LoHa (it builds
``W + ΔW``).

    make daemon-run ARGS="project/int8_train/src/infer_int8_moss.py --int8 on \\
        --int8_file models/moss_image/moss_image_base_int8_gptq.safetensors \\
        -- --lora <path> --stream fp32 --out_dir output/tests/int8"
"""

import argparse
import dataclasses
from pathlib import Path

from library.env import resolve_under_home
from library.moss.requests import GenerateRequest

from library.moss.dit_int8.base import (
    check_base,
    convert_moss_,
    load_copies,
    moss_targets,
)


def install(req: GenerateRequest, int8_file: str) -> GenerateRequest:
    """The refusals, the conversion patches and the LoRA log; returns ``req`` with
    this base's ``out_dir``."""
    if req.blocks_to_swap:
        raise SystemExit(
            "int8 base: --blocks_to_swap 0 is required (generation on this base "
            "has run no swap)"
        )

    import torch
    from safetensors import safe_open

    from library.moss import generate
    from library.moss.lora import LoRANetwork

    copies, meta, base_id = {}, {}, "rtn"
    if int8_file:
        copies, meta = load_copies(int8_file)
        base_id = meta["int8_id"]
    if req.lora:
        with safe_open(str(resolve_under_home(req.lora)), framework="pt") as f:
            lora_meta = f.metadata() or {}
        if lora_meta.get("algo") == "loha":
            raise SystemExit("int8 base: loha builds W + ΔW against the base weight")
        lora_id = lora_meta.get("int8_base_id", "")
        print(
            f"int8 infer: {req.lora} trained on "
            f"{lora_meta.get('int8_base', 'a bf16 base')} {lora_id[:16] or '-'}, "
            f"{'matches' if lora_id == base_id else 'differs from'} this base",
            flush=True,
        )

    loaded = {}

    def convert(tr) -> None:
        targets = moss_targets(tr)
        if copies:
            check_base(meta, dict(targets))
        for name, mod in targets:
            convert_moss_(mod, "infer", copies.get(name))
        torch.cuda.empty_cache()
        print(
            f"int8 infer: converted {len(targets)} linears, base {base_id[:16]}",
            flush=True,
        )

    orig_load = generate.load_transformer
    orig_apply_to = LoRANetwork.apply_to

    def load_transformer(*args, **kwargs):
        loaded["tr"] = orig_load(*args, **kwargs)
        if not req.lora:
            convert(loaded["tr"])
        return loaded["tr"]

    def apply_to(self):
        convert(loaded["tr"])
        return orig_apply_to(self)

    generate.load_transformer = load_transformer
    LoRANetwork.apply_to = apply_to
    return dataclasses.replace(
        req, out_dir=str(Path(req.out_dir) / f"int8_{base_id[:8]}")
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--int8", choices=("off", "on"), required=True)
    ap.add_argument("--int8_file", default="")
    own, argv = ap.parse_known_args()
    if own.int8_file and own.int8 == "off":
        ap.error("--int8_file needs --int8 on")
    if argv and argv[0] == "--":
        argv = argv[1:]

    req = GenerateRequest.from_argv(argv)
    if own.int8 == "on":
        req = install(req, own.int8_file)
    from library.moss.generate import run_generate

    run_generate(req)
