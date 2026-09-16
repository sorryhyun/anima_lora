"""Extract a plain LoRA from the weight delta between two full DiT checkpoints.

``ΔW = tuned − base`` per runtime-fused Linear (self_attn qkv / cross_attn kv
fused via ``ATTN_FUSE_SPECS`` — matching the training-runtime module set of a
plain-LoRA student), SVD-truncated to ``--rank``, written in the standard
defused checkpoint layout (``defuse_standard_qkv``) so the file loads through
every plain-LoRA consumer: ``--lora_weight`` inference, ``merge_to_dit``,
ComfyUI, and the turbo warm-start knob (``network.student_init_weights``).

Example — seed the DP-DMD student from the official anima_turboV10 release:

    python scripts/toolkits/extract_delta_lora.py \
        --tuned models/diffusion_models/anima_turboV10.net.safetensors \
        --rank 96 --out models/extracted/anima_turboV10_delta_r96.safetensors

``--include_adaln`` additionally extracts the adaln up-projections
(``adaln_up_{branch}``, checkpoint key ``adaln_modulation_{branch}.2``) — the
largest movers in the official turbo delta. Off by default: the student does
not target adaln yet, and stock loaders skip the keys (default exclude regex).
``--adaln_layout comfy`` names those keys after ComfyUI's model paths
(``lora_unet_blocks_{b}_adaln_modulation_{branch}_2``) so the file ships as a
ComfyUI-native LoRA — core's generic key map covers every model weight,
including adaln (``comfy/lora.py::model_lora_keys_unet``); the default
``runtime`` layout keeps the in-repo module names for the warm-start path.

Both checkpoints must be in the ``net.``-prefixed Anima layout (re-key a
ComfyUI ``model.diffusion_model.``-prefixed release first).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from networks.attn_fuse import ATTN_FUSE_SPECS
from networks.lora_modules.lora import defuse_standard_qkv

# Runtime-fused Linear set per block, mirroring the plain-LoRA student's
# targets (factory default excludes modulation/norm/embedder — see
# networks/lora_anima/config.py::_DEFAULT_EXCLUDE).
_BLOCK_RE = re.compile(r"net\.blocks\.(\d+)\.")
_ADALN_UP_BRANCHES = ("self_attn", "cross_attn", "mlp")


def _fused_module_specs(n_blocks: int) -> list[tuple[str, list[str]]]:
    """(lora_name, [base-checkpoint weight keys to concat along dim 0])."""
    by_attn = {s.attn_type: s for s in ATTN_FUSE_SPECS}
    specs: list[tuple[str, list[str]]] = []
    for b in range(n_blocks):
        p = f"net.blocks.{b}"
        qkv = by_attn["self_attn"]
        specs.append(
            (
                f"lora_unet_blocks_{b}_self_attn_{qkv.fused_letters}_proj",
                [f"{p}.self_attn.{c}_proj.weight" for c in qkv.component_letters],
            )
        )
        specs.append(
            (
                f"lora_unet_blocks_{b}_self_attn_output_proj",
                [f"{p}.self_attn.output_proj.weight"],
            )
        )
        kv = by_attn["cross_attn"]
        specs.append(
            (
                f"lora_unet_blocks_{b}_cross_attn_q_proj",
                [f"{p}.cross_attn.q_proj.weight"],
            )
        )
        specs.append(
            (
                f"lora_unet_blocks_{b}_cross_attn_{kv.fused_letters}_proj",
                [f"{p}.cross_attn.{c}_proj.weight" for c in kv.component_letters],
            )
        )
        specs.append(
            (
                f"lora_unet_blocks_{b}_cross_attn_output_proj",
                [f"{p}.cross_attn.output_proj.weight"],
            )
        )
        specs.append((f"lora_unet_blocks_{b}_mlp_layer1", [f"{p}.mlp.layer1.weight"]))
        specs.append((f"lora_unet_blocks_{b}_mlp_layer2", [f"{p}.mlp.layer2.weight"]))
    return specs


def _adaln_module_specs(n_blocks: int) -> list[tuple[str, list[str]]]:
    # Checkpoint layout: adaln_modulation_{branch}.2 == runtime adaln_up_{branch}
    # (see library/anima/weights.py::_dit_rename_hook). LoRA keys use the
    # runtime name so they attach directly once adaln_up is include_patterns'd.
    return [
        (
            f"lora_unet_blocks_{b}_adaln_up_{br}",
            [f"net.blocks.{b}.adaln_modulation_{br}.2.weight"],
        )
        for b in range(n_blocks)
        for br in _ADALN_UP_BRANCHES
    ]


def svd_truncate(
    delta: torch.Tensor, rank: int, act_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """ΔW → (lora_up [out,r], lora_down [r,in], captured energy fraction).

    With ``act_scale`` (per-input-channel activation RMS, ASVD-style) the
    truncation minimizes ``‖(ΔW − ΔW̃)·diag(s)‖`` — the expected output error
    on real activations — instead of plain Frobenius: SVD ``ΔW·diag(s)``,
    truncate, fold ``diag(1/s)`` into lora_down. The returned energy is then
    the activation-weighted (functional) capture.
    """
    if act_scale is not None:
        s = act_scale.to(delta.device, torch.float32)
        s = s.clamp(min=1e-3 * s.mean())
        delta = delta * s.unsqueeze(0)
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    r = min(rank, S.shape[0])
    e = S.pow(2)
    energy = (e[:r].sum() / e.sum().clamp(min=1e-30)).item()
    sq = S[:r].sqrt()
    up = (U[:, :r] * sq.unsqueeze(0)).contiguous()
    down = (sq.unsqueeze(1) * Vh[:r]).contiguous()
    if act_scale is not None:
        down = (down / s.unsqueeze(0)).contiguous()
    return up, down, energy


_LORA_NAME_RE = re.compile(r"^lora_unet_blocks_(\d+)_(self_attn|cross_attn|mlp)_(.+)$")
_ADALN_NAME_RE = re.compile(r"^lora_unet_blocks_(\d+)_(adaln_up_\w+)$")


def _module_path(lora_name: str) -> str:
    """lora_unet_blocks_0_self_attn_qkv_proj -> blocks.0.self_attn.qkv_proj."""
    if m := _ADALN_NAME_RE.match(lora_name):
        return f"blocks.{m.group(1)}.{m.group(2)}"
    m = _LORA_NAME_RE.match(lora_name)
    if m is None:
        raise ValueError(f"unexpected lora_name: {lora_name}")
    return f"blocks.{m.group(1)}.{m.group(2)}.{m.group(3)}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--base", default="models/diffusion_models/anima-base-v1.0.safetensors"
    )
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--rank", type=int, default=96)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--include_adaln",
        action="store_true",
        help="Also extract adaln_up_{branch} LoRAs (needs adaln targeting on load).",
    )
    ap.add_argument(
        "--adaln_rank",
        type=int,
        default=0,
        help="Truncate the adaln_up_{branch} deltas at their own rank "
        "(0 = use --rank). Their in-dim is 256, so they are near-lossless well "
        "below attn rank (r64 keeps ~99.5%% of the r96 energy). Per-module "
        "alpha == rank is written either way, so mixed-rank files load "
        "everywhere plain LoRA does.",
    )
    ap.add_argument(
        "--adaln_layout",
        choices=("runtime", "comfy"),
        default="runtime",
        help="Key naming for the adaln LoRAs: 'runtime' = in-repo module names "
        "(adaln_up_{branch}, for the turbo warm-start path), 'comfy' = ComfyUI "
        "state-dict paths (adaln_modulation_{branch}_2, loads natively in core).",
    )
    ap.add_argument(
        "--act_scales",
        default=None,
        help="Per-input-channel activation RMS file (safetensors keyed by "
        "runtime module path, e.g. blocks.0.self_attn.qkv_proj) — switches to "
        "activation-weighted (ASVD-style) truncation.",
    )
    ap.add_argument(
        "--act_alpha",
        type=float,
        default=1.0,
        help="Exponent on the activation RMS (1.0 = full whitening, 0.5 = "
        "softened, ASVD default).",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dtype", default="bfloat16")
    args = ap.parse_args()

    act_scales: dict[str, torch.Tensor] = {}
    if args.act_scales:
        from safetensors.torch import load_file

        act_scales = {
            k: v.float().pow(args.act_alpha)
            for k, v in load_file(args.act_scales).items()
        }

    fb = safe_open(args.base, framework="pt")
    ft = safe_open(args.tuned, framework="pt")
    base_keys = set(fb.keys())
    missing = base_keys - set(ft.keys())
    if missing:
        raise SystemExit(
            f"--tuned is missing {len(missing)} base keys (first: "
            f"{sorted(missing)[:3]}) — not the same architecture/layout?"
        )
    n_blocks = 1 + max(int(m.group(1)) for k in base_keys if (m := _BLOCK_RE.match(k)))

    specs = _fused_module_specs(n_blocks)
    if args.include_adaln:
        specs += _adaln_module_specs(n_blocks)

    save_dtype = getattr(torch, args.save_dtype)
    sd: dict[str, torch.Tensor] = {}
    energies: list[float] = []
    for lora_name, wkeys in specs:
        delta = torch.cat(
            [
                ft.get_tensor(k).to(args.device, torch.float32)
                - fb.get_tensor(k).to(args.device, torch.float32)
                for k in wkeys
            ],
            dim=0,
        )
        scale = act_scales.get(_module_path(lora_name)) if act_scales else None
        is_adaln = bool(_ADALN_NAME_RE.match(lora_name))
        if act_scales and scale is None and not is_adaln:
            print(f"warning: no act scale for {lora_name} — plain SVD for it")
        rank = args.adaln_rank if (is_adaln and args.adaln_rank > 0) else args.rank
        up, down, energy = svd_truncate(delta, rank, act_scale=scale)
        energies.append(energy)
        sd[f"{lora_name}.lora_up.weight"] = up.to(save_dtype).cpu()
        sd[f"{lora_name}.lora_down.weight"] = down.to(save_dtype).cpu()
        # alpha == rank -> scale 1.0: the file's up@down IS the delta.
        sd[f"{lora_name}.alpha"] = torch.tensor(float(down.shape[0]))

    defuse_standard_qkv(sd)

    if args.include_adaln and args.adaln_layout == "comfy":
        # adaln_up_{branch} (runtime name) -> adaln_modulation_{branch}_2
        # (ComfyUI state-dict path; see library/anima/weights.py::_dit_rename_hook).
        from networks.lora_utils import relayout_adaln_runtime_to_comfy

        sd = relayout_adaln_runtime_to_comfy(sd)

    metadata = {
        "ss_network_dim": str(args.rank),
        "ss_network_alpha": str(args.rank),
        "ss_delta_extract": json.dumps(
            {
                "base": Path(args.base).name,
                "tuned": Path(args.tuned).name,
                "rank": args.rank,
                "adaln_rank": args.adaln_rank if args.adaln_rank > 0 else None,
                "include_adaln": bool(args.include_adaln),
                "adaln_layout": args.adaln_layout if args.include_adaln else None,
                "act_scales": Path(args.act_scales).name if args.act_scales else None,
                "act_alpha": args.act_alpha if args.act_scales else None,
                "mean_energy": round(sum(energies) / len(energies), 4),
                "min_energy": round(min(energies), 4),
            }
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out), metadata=metadata)
    print(
        f"Wrote {out} — {len(specs)} runtime modules ({n_blocks} blocks), rank "
        f"{args.rank}, ΔW energy captured mean {sum(energies) / len(energies):.3f} "
        f"min {min(energies):.3f}"
    )


if __name__ == "__main__":
    main()
