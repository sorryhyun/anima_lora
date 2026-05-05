"""DCW v4 fusion-head data loading: bench runs + text features.

Pulls per-(stem, seed) gap trajectories from `gaps_per_sample.npz` written
by `scripts/dcw/measure_bias.py`, and per-stem text features from the
cached `{stem}_anima_te.safetensors` sidecars under `post_image_dataset/lora/`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open

ASPECT_TABLE = {
    (832, 1248): 0,  # HD portrait — most common in cache
    (896, 1152): 1,  # 3:4 portrait
    (768, 1344): 2,  # tall portrait
    (1152, 896): 3,  # 3:4 landscape
    (1248, 832): 4,  # HD landscape
}
ASPECT_NAMES = ["832x1248", "896x1152", "768x1344", "1152x896", "1248x832"]
N_ASPECTS = 5


@dataclass
class Row:
    run_id: str
    aspect_id: int
    stem: str
    seed_idx: int
    gap_LL: np.ndarray  # (n_steps,) — used for target (residual on tail)
    v_rev_LL: np.ndarray  # (n_steps,) — used for input g_obs
    v_rev_source: str  # "native" | "synthetic" | "fallback"
    sigma_i: np.ndarray  # (n_steps,) — σ schedule for the run; per-row for LSQ targets


def load_bench_runs(
    results_roots: Path | list[Path],
    *,
    require_cfg: float = 4.0,
    require_mod_w: float = 3.0,
    skip_with_lora: bool = True,
) -> list[Row]:
    if isinstance(results_roots, (str, Path)):
        results_roots = [Path(results_roots)]
    rows: list[Row] = []
    seen_run_names: set[str] = set()  # de-dup if same name appears in multiple roots
    candidate_dirs: list[Path] = []
    for root in results_roots:
        if not root.exists():
            continue
        candidate_dirs.extend(p for p in root.iterdir() if p.is_dir())
    for run_dir in sorted(candidate_dirs):
        if run_dir.name in seen_run_names:
            continue
        seen_run_names.add(run_dir.name)
        npz_path = run_dir / "gaps_per_sample.npz"
        rj_path = run_dir / "result.json"
        if not (npz_path.exists() and rj_path.exists()):
            continue
        rj = json.loads(rj_path.read_text())
        a = rj.get("args", {})
        H, W = a.get("image_h"), a.get("image_w")
        if (H, W) not in ASPECT_TABLE:
            print(f"skip {run_dir.name}: aspect {H}x{W} not in table")
            continue
        if a.get("guidance_scale") != require_cfg:
            print(
                f"skip {run_dir.name}: cfg={a.get('guidance_scale')} != {require_cfg}"
            )
            continue
        if a.get("mod_w") != require_mod_w:
            print(f"skip {run_dir.name}: mod_w={a.get('mod_w')} != {require_mod_w}")
            continue
        if skip_with_lora and a.get("lora_weight"):
            print(f"skip {run_dir.name}: has LoRA {a['lora_weight']}")
            continue
        n_seeds = int(a.get("n_seeds", 1))
        z = np.load(npz_path, allow_pickle=True)
        stems = z["stems"]
        gap_LL = z["gap_LL"]  # (N, n_steps)
        if "v_rev_LL" in z.files:
            v_rev_LL = z["v_rev_LL"]
            source = "native"
        else:
            v_fwd_pop = _load_v_fwd_pop_mean(run_dir, band="LL")
            if v_fwd_pop is not None:
                v_rev_LL = (
                    gap_LL + v_fwd_pop[None, :]
                )  # broadcast (n_steps,) → (N, n_steps)
                source = "synthetic"
            else:
                v_rev_LL = gap_LL
                source = "fallback"
        sigma_i = _load_sigma_schedule(run_dir, n_steps=gap_LL.shape[1])
        aspect_id = ASPECT_TABLE[(H, W)]
        for r in range(len(stems)):
            img_idx = r // n_seeds
            seed_idx = r % n_seeds
            rows.append(
                Row(
                    run_id=run_dir.name,
                    aspect_id=aspect_id,
                    stem=str(stems[r]),
                    seed_idx=int(
                        img_idx * 1000 + seed_idx
                    ),  # globally unique within run
                    gap_LL=np.asarray(gap_LL[r], dtype=np.float64),
                    v_rev_LL=np.asarray(v_rev_LL[r], dtype=np.float64),
                    v_rev_source=source,
                    sigma_i=sigma_i,
                )
            )
    return rows


def _load_v_fwd_pop_mean(run_dir: Path, *, band: str = "LL") -> np.ndarray | None:
    """Read baseline_v_fwd_<band> column from per_step_bands.csv as a per-step mean."""
    csv_path = run_dir / "per_step_bands.csv"
    if not csv_path.exists():
        return None
    col = f"baseline_v_fwd_{band}"
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        try:
            return np.array([float(r[col]) for r in reader], dtype=np.float64)
        except KeyError:
            return None


def _load_sigma_schedule(run_dir: Path, *, n_steps: int) -> np.ndarray:
    """Read the sigma_i column from per_step_bands.csv (or per_step.csv as fallback).

    The σ schedule is run-level (same across rows within a run), so we read
    once per run_dir. Falls back to a linear ramp from 1.0 → 0.0 over n_steps
    if neither csv carries the column — old runs without the schedule will
    silently degrade to that approximation.
    """
    for name in ("per_step_bands.csv", "per_step.csv"):
        csv_path = run_dir / name
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            try:
                arr = np.array([float(r["sigma_i"]) for r in reader], dtype=np.float64)
            except KeyError:
                continue
            if len(arr) >= n_steps:
                return arr[:n_steps]
    return np.linspace(1.0, 0.0, n_steps, dtype=np.float64)


def load_text_features(
    stems: list[str], dataset_dir: Path, variant: int = 0
) -> dict[str, dict]:
    """Per-stem c_pool + caption_length + token_l2_std from te cache."""
    out: dict[str, dict] = {}
    for stem in stems:
        if stem in out:
            continue
        te_path = dataset_dir / f"{stem}_anima_te.safetensors"
        if not te_path.exists():
            print(f"warn: missing te cache for {stem}")
            continue
        with safe_open(str(te_path), framework="pt") as f:
            emb = f.get_tensor(f"crossattn_emb_v{variant}").float()  # (512, 1024)
            mask = f.get_tensor(f"attn_mask_v{variant}").bool()  # (512,)
        valid = emb[mask]  # (L, 1024)
        if valid.numel() == 0:
            continue
        c_pool = valid.mean(dim=0)  # (1024,)
        token_l2 = valid.norm(dim=-1)  # (L,)
        out[stem] = {
            "c_pool": c_pool.numpy().astype(np.float32),
            "caption_length": int(mask.sum().item()),
            "token_l2_std": float(token_l2.std().item()),
        }
    return out


def build_population_mu_g(rows: list[Row], n_steps: int) -> np.ndarray:
    """Single population-mean LL gap trajectory across all rows."""
    if not rows:
        return np.zeros(n_steps, dtype=np.float64)
    return np.stack([r.gap_LL for r in rows]).mean(axis=0)
