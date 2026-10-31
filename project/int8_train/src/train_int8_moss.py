"""The MOSS-Image trainer on an int8 base (the MOSS counterpart of ``train_int8.py``).

Strips ``--int8 {off,fwd,full,stream}`` and runs ``library.moss.train.run_train`` on the
``TrainRequest`` built from the rest. ``off`` runs the trainer untouched. Otherwise
two patches go in first:

- ``library.moss.lora.LoRANetwork.apply_to`` converts the six per-block linears
  (``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}``) before patching them: the
  patched forward captures ``module.__class__.forward`` at that point. That is after
  the network's ``__init__``, so ``down_init weight_svd*`` still reads bf16, and
  before ``compile_blocks``. Each converted linear keeps an empty bf16 ``weight``
  buffer, since the block forward reads ``proj["q"].weight.dtype``.
- ``LoRANetwork.metadata`` stamps ``int8_base=<mode>`` and ``int8_base_id`` (``rtn``,
  or the ``int8_id`` of ``--int8_file``).

``stream`` is ``full`` with ``wq_t`` on the host (``int8_linear.WqtStream``): the same
numbers, about 4.2 GiB less on the card, 4.4 GiB more pinned host memory.

Refused: ``blocks_to_swap != 0`` (the offloader moves only ``.weight``) and
``algo loha`` (it builds ``W + ΔW``).

    make daemon-run ARGS="--stall-timeout 900 project/int8_train/src/train_int8_moss.py \\
        --int8 full -- --config configs/moss/lora.toml --blocks_to_swap 0 --grad_checkpointing"
"""

import argparse
import atexit
import os
import re

MOSS_TARGET = re.compile(r"^layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)$")


def moss_targets(tr) -> list:
    return [(n, m) for n, m in tr.named_modules() if MOSS_TARGET.match(n)]


def moss_keys(name: str) -> list[str]:
    """A converted module's weight key in the Diffusers checkpoint."""
    return [f"{name}.weight"]


def load_copies(int8_file: str) -> tuple[dict, dict]:
    """``({module name: (wq, s)}, metadata)`` from a ``dit_gptq_moss.py`` file."""
    from int8_linear import load_int8_file  # sibling module
    from library.env import resolve_under_home

    return load_int8_file(str(resolve_under_home(int8_file)), moss_keys)


def check_base(meta: dict, by_name: dict) -> None:
    """Refuse a file made from another DiT; before any conversion. ``by_name``:
    module name → linear, holding at least the two ``base_fingerprint`` hashes."""
    from int8_linear import base_fingerprint  # sibling module

    ends = ("layers.0.proj.q", "layers.41.proj.ffn_down")
    if any(n not in by_name for n in ends):
        raise SystemExit("int8 base: the file's fingerprint linears are not targeted")
    if meta.get("base_fingerprint") != base_fingerprint(
        None, [(n, by_name[n]) for n in ends]
    ):
        raise SystemExit(f"int8 base: the file was made from {meta.get('source')}")


def convert_moss_(mod, mode: str, forward_copy=None) -> None:
    """``convert_`` on the card. Modes that drop the weight keep an empty bf16
    ``weight`` buffer: the block forward reads its dtype."""
    import torch

    from int8_linear import convert_  # sibling module

    device = torch.device("cuda")
    mod.to(device)
    convert_(mod, mode, forward_copy)
    if mode != "fwd":
        mod.register_buffer(
            "weight",
            torch.empty(0, dtype=torch.bfloat16, device=device),
            persistent=False,
        )


def stream_targets(by_name: dict) -> None:
    """``wq_t`` to the host, block by block, in the block's forward order."""
    import torch

    from int8_linear import stream_wqt_  # sibling module

    linears = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
    blocks = [[by_name[f"layers.{b}.proj.{p}"] for p in linears] for b in range(42)]
    stream = stream_wqt_(blocks)
    # what stays sat between the dropped weights and wq_t and would pin their pages
    device = blocks[0][0].wq.device
    kept = [(m, k) for mods in blocks for m in mods for k in ("wq", "s", "s_t")]
    for m, k in kept:
        m._buffers[k] = m._buffers[k].cpu()
    torch.cuda.empty_cache()
    for m, k in kept:
        m._buffers[k] = m._buffers[k].to(device)

    def report():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(
            f"wq_t stream: {stream.entered} blocks entered, {stream.cold} cold, "
            f"{stream.wait_s:.2f} s waited; peak {peak:.2f} GB allocated / "
            f"{reserved:.2f} GB reserved",
            flush=True,
        )

    atexit.register(report)


def install(mode: str, req, int8_file: str = "") -> None:
    """The two patches and the refusals above, for ``mode`` ``fwd``, ``full`` or ``stream``.
    ``int8_file`` (``dit_gptq_moss.py``) supplies the forward copies, checked against
    the loaded DiT; the backward copy stays round to nearest. ``profile_step_moss.py
    --int8`` calls this too."""
    if req.blocks_to_swap != 0:
        raise SystemExit(
            "int8 base: --blocks_to_swap 0 is required (the offloader moves only "
            ".weight, not wq / wq_t)"
        )
    if req.algo == "loha":
        raise SystemExit("int8 base: loha builds W + ΔW against the base weight")

    import torch

    from library.moss.lora import LoRANetwork

    converted: list[str] = []
    orig_apply_to = LoRANetwork.apply_to
    orig_metadata = LoRANetwork.metadata
    copies, file_meta, int8_id = {}, {}, "rtn"
    if int8_file:
        copies, file_meta = load_copies(int8_file)
        int8_id = file_meta["int8_id"]

    def apply_to(self):
        if copies:
            check_base(file_meta, dict(self._targets))
        for name, mod in self._targets:
            if not MOSS_TARGET.match(name):
                continue
            convert_moss_(mod, "full" if mode == "stream" else mode, copies.get(name))
            converted.append(name)
        if mode == "stream":
            stream_targets(dict(self._targets))
        torch.cuda.empty_cache()
        print(
            f"int8 {mode}: converted {len(converted)} linears, forward copies "
            f"{int8_id[:16] if copies else 'round to nearest'}",
            flush=True,
        )
        return orig_apply_to(self)

    def metadata(self, **extra):
        if not converted:
            raise SystemExit("int8 base: no DiT linear was converted")
        return orig_metadata(self, int8_base=mode, int8_base_id=int8_id, **extra)

    LoRANetwork.apply_to = apply_to
    LoRANetwork.metadata = metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--int8", choices=("off", "fwd", "full", "stream"), required=True)
    ap.add_argument("--int8_file", default="")
    own, train_argv = ap.parse_known_args()
    if train_argv and train_argv[0] == "--":
        train_argv = train_argv[1:]

    from library.moss.requests import TrainRequest

    req = TrainRequest.from_argv(train_argv)
    if req.expandable_segments:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if own.int8 != "off":
        install(own.int8, req, own.int8_file)
    from library.moss.train import run_train

    run_train(req)
