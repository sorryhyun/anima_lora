"""Generation on the int8 base: ``inference.py`` with the DiT's large linears converted.

Strips ``--int8 {off,on}`` and runs ``inference.py`` with the rest. ``off`` runs it
untouched, so a bf16 arm goes through the same command. ``on`` makes two changes:

- LoRAs ride the P-GRAFT hook path (``--pgraft`` is appended, no cutoff, so they stay
  on for every step) instead of the static merge, which would fold the LoRA into the
  bf16 weight before quantization. The int8 base then runs ``int8(x, W_q) + B·A·x``,
  the function ``train_int8.py`` trained.
- ``attach_adapters`` converts the DiT (``int8_linear`` mode ``infer``: forward copy
  only, bf16 weight dropped) first, so the LoRA modules capture the int8 forward as
  ``org_forward``. That is after the model is on the GPU and before compile.

``--int8_file`` (a ``dit_gptq.py`` file made from the loaded checkpoint) supplies the
forward copies instead of round to nearest. Refused: block swap (the offloader moves
only ``.weight``). Each LoRA's ``ss_int8_base`` / ``ss_int8_base_id`` stamps are
logged against this base, so a LoRA trained on another base is visible.

    make daemon-run ARGS="project/int8_train/src/infer_int8.py --int8 on \\
        <inference.py args>"
"""

import argparse
import runpy
import sys

from library.env import resolve_under_home

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--int8", choices=("off", "on"), required=True)
ap.add_argument("--int8_file", default=None)
own, infer_argv = ap.parse_known_args()
if own.int8_file and own.int8 == "off":
    ap.error("--int8_file needs --int8 on")

if own.int8 == "on":
    from safetensors import safe_open

    from int8_linear import convert_dit_  # sibling module
    from library.inference import models as inference_models

    if "--lora_weight" in infer_argv and "--pgraft" not in infer_argv:
        infer_argv.append("--pgraft")
    base_id = "rtn"
    if own.int8_file:
        own.int8_file = str(resolve_under_home(own.int8_file))
        with safe_open(own.int8_file, framework="pt") as f:
            base_id = (f.metadata() or {})["int8_id"]
    orig_attach = inference_models.attach_adapters

    def attach_adapters(model, args, device, **kw):
        if getattr(model, "blocks_to_swap", None):
            raise SystemExit(
                "int8 base: blocks_to_swap > 0 is refused (the offloader moves only "
                ".weight, not wq)"
            )
        if getattr(args, "lora_cutoff_step", None) is not None:
            raise SystemExit("int8 base: --lora_cutoff_step is not supported")
        names = convert_dit_(model, "infer", own.int8_file)
        if not names:
            raise SystemExit("int8 base: no DiT linear was converted")
        print(
            f"int8 infer: converted {len(names)} linears, base {base_id[:16]}",
            flush=True,
        )
        for path in args.lora_weight or []:
            with safe_open(path, framework="pt") as f:
                meta = f.metadata() or {}
            mode = meta.get("ss_int8_base", "none (bf16 base)")
            lora_id = meta.get(
                "ss_int8_base_id", "rtn" if "ss_int8_base" in meta else ""
            )
            match = "matches" if lora_id == base_id else "differs from"
            print(
                f"int8 infer: {path} trained on int8 base {mode} "
                f"{lora_id[:16] or '-'}, {match} this base",
                flush=True,
            )
        return orig_attach(model, args, device, **kw)

    inference_models.attach_adapters = attach_adapters

sys.argv = ["inference.py", *infer_argv]
runpy.run_path(str(resolve_under_home("inference.py")), run_name="__main__")
