"""Sync inference subsets of anima_lora into custom-node ``_vendor/`` trees.

Each ComfyUI node tries to import the live ``library.*`` first, falling back
to a bundled vendor copy when the host install isn't sitting inside the
anima_lora repo. This script keeps those vendor copies fresh.

Four targets (the tagger node no longer vendors anything — it depends on the
``anime_tools`` package and lives in that repo, ``anime_tools/comfyui/anima_tagger/``):

* ``custom_nodes/comfyui-anima-directedit/_vendor/`` — directedit primitives,
  trimmed sampling helper, the trimmed ``CONSTANT_TOKEN_BUCKETS`` constant,
  and a tiny ``library.anima.models`` stub so the lazy ``Anima`` annotation
  resolves. DirectEdit no longer pulls in AnimaTagger / vision / edit
  dispatcher — its node consumes ``source_tag`` / ``target_tag`` STRINGs
  directly, with image-driven captioning handled externally by
  ``AnimaTaggerCaption``.
* ``ComfyUI-Anima_lora-Adapter/_vendor/`` — the pure-compute router kernels
  imported by ``adapter.py`` + ``fera.py`` (FEI 2-band / n-band, σ sinusoidal
  features, σ-band partition mask). This node was **extracted to a standalone
  published repo** (a sibling checkout, default ``../../ComfyUI-Anima_lora-Adapter``;
  override with ``ANIMA_ADAPTER_NODE_REPO``); sync_vendor writes the vendor tree
  *into that repo*, which is the authoritative copy the node imports at runtime.
  ``library/inference/router_compute.py`` is the single import surface; it pulls
  ``library/runtime/fei.py`` and ``networks/lora_modules/router_state.py``
  transitively, so we vendor all three verbatim. Trained router weights are
  bit-sensitive to these kernels, so any drift between the live tree and
  vendored copy produces silently corrupted gates at inference. Also vendors
  ``library/anima/ext_vocab.py`` — the CJK vocab-pack runtime (tokenizer
  segmentation + ``HybridT5Encoder``) the ``AnimaVocabPackLoader`` node uses;
  the trained ext rows are keyed to its exact segmentation, so drift here
  silently mis-routes prompts to wrong rows. Skipped (with a
  warning) when the standalone repo isn't checked out beside anima_lora.
* ``custom_nodes/comfyui-anima-trainer/_vendor/`` — the stdlib daemon *client*
  the trainer node submits jobs through. Lets the node be installed outside
  the anima_lora repo and still talk to a running daemon over localhost HTTP.
  ``anima_daemon/config.py`` + ``client.py`` are copied verbatim; ``proc.py``
  is trimmed to ``read_pidfile`` only so the vendored client stays pure-stdlib
  (the live ``proc.py`` imports psutil for spawn/kill, which the node never
  needs — it errors if the daemon isn't already up rather than auto-starting).
* ``ComfyUI-Spectrum-KSampler/_vendor/`` — the pure-compute ``*_core`` kernels
  (FSG / SMC / CNS / SPD numerics) shared verbatim between the library's
  sampler-boundary plugins and the node's ComfyUI seam wrappers. Like the
  hydralora target this is a standalone published repo (default a sibling of
  anima_lora's parent; override ``ANIMA_SPECTRUM_NODE_REPO``); sync_vendor writes
  the tree *into that repo*. Each core is torch/numpy only — no ``comfy`` and no
  anima-model imports — so drift between the live tree and the vendored copy is
  the bug class this target eliminates. Skipped (with a warning) when the repo
  isn't checked out beside anima_lora.

Run before bumping a node version / publishing:

    python scripts/release/sync_vendor.py

The vendor tree mirrors the live namespace (``library.*`` / ``networks.*``)
so the copied files' internal imports keep working unchanged.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTEDIT_VENDOR = ROOT / "custom_nodes" / "comfyui-anima-directedit" / "_vendor"
TRAINER_VENDOR = ROOT / "custom_nodes" / "comfyui-anima-trainer" / "_vendor"

# The hydralora node (Anima Adapter Loader) is a standalone published repo;
# sync_vendor writes its router-kernel vendor tree *into that repo*. Default is
# a sibling checkout; override with ``ANIMA_ADAPTER_NODE_REPO``.
ADAPTER_NODE_REPO = Path(
    os.environ.get(
        "ANIMA_ADAPTER_NODE_REPO", ROOT.parents[1] / "ComfyUI-Anima_lora-Adapter"
    )
)
HYDRALORA_VENDOR = ADAPTER_NODE_REPO / "_vendor"

# The Spectrum KSampler is also a standalone published repo (default a sibling of
# anima_lora's parent; override with ``ANIMA_SPECTRUM_NODE_REPO``). sync_vendor
# writes the pure-compute *_core kernels (FSG / SMC / CNS / SPD numerics)
# into its ``_vendor/`` tree. The node imports the live ``library.*`` / ``networks.*``
# first and falls back to this tree when installed outside the repo.
SPECTRUM_NODE_REPO = Path(
    os.environ.get(
        "ANIMA_SPECTRUM_NODE_REPO", ROOT.parents[1] / "ComfyUI-Spectrum-KSampler"
    )
)
SPECTRUM_VENDOR = SPECTRUM_NODE_REPO / "_vendor"

DIRECTEDIT_VERBATIM: list[tuple[str, str]] = [
    (
        "library/inference/editing/directedit.py",
        "library/inference/editing/directedit.py",
    ),
    (
        "library/inference/editing/directedit_splice.py",
        "library/inference/editing/directedit_splice.py",
    ),
    # directedit.py hard-imports SMCCFGState; vendor the leaf so the standalone
    # tree is self-contained (torch-only, no further library deps).
    (
        "library/inference/corrections/smc_cfg.py",
        "library/inference/corrections/smc_cfg.py",
    ),
]

DIRECTEDIT_PACKAGE_DIRS: list[str] = [
    "library",
    "library/inference",
    "library/inference/editing",
    "library/inference/corrections",
    "library/anima",
    "library/datasets",
]

TRIMMED_SAMPLING = '''"""Trimmed extract of library/inference/sampling.py for the vendored
DirectEdit path. Contains only ``get_timesteps_sigmas`` — the only helper
the DirectEdit ComfyUI node calls. Drops the diffusers-based samplers the
full module exposes (not needed at inference here).

DO NOT EDIT — regenerated by scripts/release/sync_vendor.py.
"""

from __future__ import annotations

from typing import Tuple

import torch


def get_timesteps_sigmas(
    sampling_steps: int, shift: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate flow-matching timesteps + sigmas for ``sampling_steps`` Euler steps.

    ``timesteps`` is the DiT time arg on the σ∈[0,1] scale (== ``sigmas[:-1]``);
    the model rescales nothing, so callers feed it directly (no /1000).
    """
    sigmas = torch.linspace(1, 0, sampling_steps + 1)
    sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    sigmas = sigmas.to(torch.float32)
    timesteps = sigmas[:-1].to(dtype=torch.float32, device=device)
    return timesteps, sigmas
'''

TRIMMED_BUCKETS = '''"""Trimmed extract of library/datasets/buckets.py for the vendored
DirectEdit path. Contains only ``EDGE_TOKEN_BANDS`` — the per-tier free-fit token
bands (the discrete constant-token bucket pool was removed; free-fit is the only
resize mode).

DO NOT EDIT — regenerated by scripts/release/sync_vendor.py.
"""

from __future__ import annotations

# Per-tier (lo, hi) token bands — kept in lockstep with the live value, which
# ``anime_tools.buckets`` owns (library/datasets/buckets.py re-exports it). If
# the package's table changes, re-run scripts/release/sync_vendor.py.
EDGE_TOKEN_BANDS = {{EDGE_TOKEN_BANDS_LITERAL}}
'''

STUB_ANIMA_MODELS = '''"""Stub of library.anima.models for the vendored DirectEdit path.

The live module is ~2.4k LOC and defines the full Anima DiT. The vendored
``library/inference/directedit.py`` imports it solely for a type annotation
that is already lazy (``from __future__ import annotations``), so a tiny
placeholder class is enough to keep the import succeeding. The actual model
passed at runtime is whatever the ComfyUI MODEL socket carries.

DO NOT EDIT — regenerated by scripts/release/sync_vendor.py.
"""

from __future__ import annotations


class Anima:
    """Placeholder type. The live class lives in the parent anima_lora repo."""
'''


def _read_edge_token_bands_literal() -> str:
    """Render the live ``EDGE_TOKEN_BANDS`` (owned by ``anime_tools.buckets``,
    re-exported by ``library/datasets/buckets.py``) as a dict literal so the
    trimmed file mirrors the canonical per-tier token bands exactly. Avoids
    hand-syncing two copies."""
    from library.datasets.buckets import EDGE_TOKEN_BANDS

    lines = ["{"]
    for edge in sorted(EDGE_TOKEN_BANDS):
        lo, hi = EDGE_TOKEN_BANDS[edge]
        lines.append(f"    {edge}: ({lo}, {hi}),")
    lines.append("}")
    return "\n".join(lines) + "\n"


DIRECTEDIT_TRIMMED_TEMPLATES: list[tuple[str, str]] = [
    ("library/inference/sampling.py", TRIMMED_SAMPLING),
    ("library/datasets/buckets.py", TRIMMED_BUCKETS),
    ("library/anima/models.py", STUB_ANIMA_MODELS),
]


def _write_pkg_markers(vendor_root: Path, package_dirs: list[str]) -> None:
    for d in package_dirs:
        pkg = vendor_root / d
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text("")


def _copy_verbatim(vendor_root: Path, files: list[tuple[str, str]]) -> None:
    for src_rel, dst_rel in files:
        src = ROOT / src_rel
        dst = vendor_root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  copied  {src_rel}")


def _write_trimmed(vendor_root: Path, files: list[tuple[str, str]]) -> None:
    for dst_rel, content in files:
        dst = vendor_root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content)
        print(f"  trimmed {dst_rel}")


def _resolve_directedit_trimmed() -> list[tuple[str, str]]:
    """Substitute live constants into trimmed templates."""
    out: list[tuple[str, str]] = []
    bands_literal = _read_edge_token_bands_literal()
    for dst_rel, template in DIRECTEDIT_TRIMMED_TEMPLATES:
        content = template.replace("{{EDGE_TOKEN_BANDS_LITERAL}}", bands_literal)
        out.append((dst_rel, content))
    return out


def build_directedit_vendor() -> None:
    print(f"\n[directedit] -> {DIRECTEDIT_VENDOR.relative_to(ROOT)}")
    if DIRECTEDIT_VENDOR.exists():
        shutil.rmtree(DIRECTEDIT_VENDOR)
    DIRECTEDIT_VENDOR.mkdir(parents=True)
    (DIRECTEDIT_VENDOR / "__init__.py").write_text(
        '"""Bundled inference subset of anima_lora.\n\n'
        "Synced by scripts/release/sync_vendor.py — do not edit by hand.\n"
        '"""\n'
    )
    _write_pkg_markers(DIRECTEDIT_VENDOR, DIRECTEDIT_PACKAGE_DIRS)
    _copy_verbatim(DIRECTEDIT_VENDOR, DIRECTEDIT_VERBATIM)
    _write_trimmed(DIRECTEDIT_VENDOR, _resolve_directedit_trimmed())


# Hydralora vendor tree — pure-compute kernels for adapter.py + fera.py in the
# standalone ComfyUI-Anima_lora-Adapter repo. router_compute.py is the single
# import surface; it pulls fei.py + router_state.py transitively, so all three
# are vendored verbatim (router weights are bit-sensitive to these kernels).
HYDRALORA_VERBATIM: list[tuple[str, str]] = [
    ("library/inference/router_compute.py", "library/inference/router_compute.py"),
    ("library/runtime/fei.py", "library/runtime/fei.py"),
    ("networks/lora_modules/router_state.py", "networks/lora_modules/router_state.py"),
    # CJK vocab-pack runtime (segment_runs / HybridT5Encoder / load_ext_assets)
    # for the AnimaVocabPackLoader node — pure CPU, torch + safetensors only.
    ("library/anima/ext_vocab.py", "library/anima/ext_vocab.py"),
]

HYDRALORA_PACKAGE_DIRS: list[str] = [
    "library",
    "library/anima",
    "library/inference",
    "library/runtime",
    "networks",
    "networks/lora_modules",
]


def build_hydralora_vendor() -> None:
    if not ADAPTER_NODE_REPO.is_dir():
        print(
            f"\n[hydralora] SKIPPED — standalone node repo not found at "
            f"{ADAPTER_NODE_REPO}\n"
            f"            clone it beside anima_lora, or set "
            f"ANIMA_ADAPTER_NODE_REPO to its path."
        )
        return
    print(f"\n[hydralora] -> {HYDRALORA_VENDOR}")
    if HYDRALORA_VENDOR.exists():
        shutil.rmtree(HYDRALORA_VENDOR)
    HYDRALORA_VENDOR.mkdir(parents=True)
    (HYDRALORA_VENDOR / "__init__.py").write_text(
        '"""Bundled inference subset of anima_lora.\n\n'
        "Synced by scripts/release/sync_vendor.py — do not edit by hand.\n"
        '"""\n'
    )
    _write_pkg_markers(HYDRALORA_VENDOR, HYDRALORA_PACKAGE_DIRS)
    _copy_verbatim(HYDRALORA_VENDOR, HYDRALORA_VERBATIM)


# Trainer vendor tree — the stdlib daemon *client* the trainer node submits
# jobs through. config.py + client.py copied verbatim (pure stdlib); proc.py
# trimmed to read_pidfile only — dropping its psutil import keeps the vendored
# client pure-stdlib (the node never auto-starts the daemon, so spawn/kill is
# never exercised).
TRAINER_VERBATIM: list[tuple[str, str]] = [
    ("anima_daemon/config.py", "anima_daemon/config.py"),
    ("anima_daemon/client.py", "anima_daemon/client.py"),
]

TRAINER_PACKAGE_DIRS: list[str] = [
    "anima_daemon",
]

TRIMMED_DAEMON_PROC = '''"""Trimmed extract of anima_daemon/proc.py for the vendored daemon client.

Contains only ``read_pidfile`` — the single symbol ``client.py`` touches at
runtime (via ``_resolve_port``). The full live module routes spawn / kill /
liveness through psutil; none of that is needed by the trainer node, which
never auto-starts the daemon (it errors if the daemon isn't already up).
Dropping the psutil import keeps the vendored client pure-stdlib.

DO NOT EDIT — regenerated by scripts/release/sync_vendor.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def read_pidfile(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
'''

TRAINER_TRIMMED: list[tuple[str, str]] = [
    ("anima_daemon/proc.py", TRIMMED_DAEMON_PROC),
]


def build_trainer_vendor() -> None:
    print(f"\n[trainer] -> {TRAINER_VENDOR.relative_to(ROOT)}")
    if TRAINER_VENDOR.exists():
        shutil.rmtree(TRAINER_VENDOR)
    TRAINER_VENDOR.mkdir(parents=True)
    (TRAINER_VENDOR / "__init__.py").write_text(
        '"""Bundled stdlib daemon-client subset of anima_lora.\n\n'
        "Synced by scripts/release/sync_vendor.py — do not edit by hand.\n"
        '"""\n'
    )
    _write_pkg_markers(TRAINER_VENDOR, TRAINER_PACKAGE_DIRS)
    _copy_verbatim(TRAINER_VENDOR, TRAINER_VERBATIM)
    _write_trimmed(TRAINER_VENDOR, TRAINER_TRIMMED)


# Spectrum vendor tree — the pure-compute ``*_core`` kernels shared verbatim
# between the library's sampler-boundary plugins and the node's ComfyUI seam
# wrappers. Each core is torch/numpy only (no comfy / no anima-model imports),
# so the copied files' internal imports keep working unchanged. The node files
# import the live ``library.*`` / ``networks.*`` first and fall back to this tree.
SPECTRUM_VERBATIM: list[tuple[str, str]] = [
    (
        "library/inference/corrections/fsg_core.py",
        "library/inference/corrections/fsg_core.py",
    ),
    # SMC-CFG is already pure compute (torch-only); the library module IS the
    # shared core — no separate ``_core`` split. The node imports SMCCFGState
    # from here and keeps only its denoised↔v-space seam wrapper.
    (
        "library/inference/corrections/smc_cfg.py",
        "library/inference/corrections/smc_cfg.py",
    ),
    # Mod-guidance projection (σ-flat / σ-FiLM pooled-text head) + per-block
    # schedule. The node imports project_pooled / build_block_schedule from here,
    # replacing its hand-mirrored _project / _project_film / _build_schedule.
    (
        "library/inference/corrections/mod_guidance_core.py",
        "library/inference/corrections/mod_guidance_core.py",
    ),
    # Spectrum Chebyshev forecasters (ChebyshevForecaster + SpectrumPredictor) +
    # the SEA cache-decision metric / auto-δ calibration. Both are pure torch and
    # were hand-mirrored node-side (forecaster.py + the verbatim-ported SEA math);
    # the node now imports them and keeps only its ComfyUI seam (disk δ-cache,
    # model_function_wrapper state machine).
    ("networks/spectrum_forecast.py", "networks/spectrum_forecast.py"),
    ("networks/spectrum_sea.py", "networks/spectrum_sea.py"),
    # SPD spectral primitives (DCT helpers + spectral_expand geometry). The node
    # imports these so its SPEED sampler matches the CLI SPD path bit-for-bit.
    ("networks/spd_core.py", "networks/spd_core.py"),
    # CNS recolorer numerics (radial binning + γ-driven recolor). Path resolution
    # stays node-side; the node imports CNSRecolorer + radial_bins from here.
    (
        "library/inference/corrections/cns_core.py",
        "library/inference/corrections/cns_core.py",
    ),
    # FEI features (frequency-energy indicator) — shared by the Hydra/FeRA
    # FEI-routing path the node vendors.
    ("library/runtime/fei.py", "library/runtime/fei.py"),
]

SPECTRUM_PACKAGE_DIRS: list[str] = [
    "library",
    "library/inference",
    "library/inference/corrections",
    "library/runtime",
    "networks",
]


def build_spectrum_vendor() -> None:
    if not SPECTRUM_NODE_REPO.is_dir():
        print(
            f"\n[spectrum] SKIPPED — standalone node repo not found at "
            f"{SPECTRUM_NODE_REPO}\n"
            f"           clone it beside anima_lora, or set "
            f"ANIMA_SPECTRUM_NODE_REPO to its path."
        )
        return
    print(f"\n[spectrum] -> {SPECTRUM_VENDOR}")
    if SPECTRUM_VENDOR.exists():
        shutil.rmtree(SPECTRUM_VENDOR)
    SPECTRUM_VENDOR.mkdir(parents=True)
    (SPECTRUM_VENDOR / "__init__.py").write_text(
        '"""Bundled pure-compute kernel subset of anima_lora.\n\n'
        "Synced by scripts/release/sync_vendor.py — do not edit by hand.\n"
        '"""\n'
    )
    _write_pkg_markers(SPECTRUM_VENDOR, SPECTRUM_PACKAGE_DIRS)
    _copy_verbatim(SPECTRUM_VENDOR, SPECTRUM_VERBATIM)


def main() -> None:
    build_directedit_vendor()
    build_hydralora_vendor()
    build_trainer_vendor()
    build_spectrum_vendor()
    print("\nvendor trees fresh.")


if __name__ == "__main__":
    main()
