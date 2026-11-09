"""Self-distillation from the image-conditioned TE (``docs/moss/proposal/self_distill_lora.md``).

The off-policy arm of D-OPSD (arXiv 2605.05204): the student (the LoRA on the cached
caption embeds) regresses on the same DiT's velocity at the same noisy latent, with the
target image in the TE ahead of the caption (Qwen3.5-9B's vision tower,
``library.moss.vision_te``). The teacher is the base model (``distill base``) or an
EMA shadow of the LoRA (``distill ema``).

Teacher cache: ``{stem}.moss_vis.safetensors`` beside each ``{stem}.moss_te.safetensors``,
one ``embeds_<kind>`` per caption the student can draw (``full``, ``plain``, ``v{k}``,
``empty`` = the image alone), the page encoded once and every caption continued from it
(``vision_te.encode_variants``). Image at the latent's size, so ~1 vision row per 32² px.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

VIS_SUFFIX = ".moss_vis.safetensors"
TE_SUFFIX = ".moss_te.safetensors"


def teacher_path(te_path: Path) -> Path:
    return te_path.with_name(te_path.name[: -len(TE_SUFFIX)] + VIS_SUFFIX)


def _stamp(captions: dict[str, str], source_sha: str, encoder: str, tail: str) -> str:
    blob = json.dumps([captions, source_sha, encoder, tail], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def cache_teachers(req) -> dict:
    """Write the teacher file of every sample in ``req.cache`` whose captions, source,
    encoder or tail changed (``req.overwrite``: all of them)."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    from library.env import resolve_under_home
    from library.moss import vision_te
    from library.moss.caching.images import fit, load_rgb
    from library.moss.cache import encoder_id
    from library.moss.loader import empty_cache, load_tokenizer
    from library.moss.prompt_tail import TAILS
    from library.moss.training.data import sample_captions, samples_from_cache

    cache_dir = resolve_under_home(req.cache)
    samples = samples_from_cache(cache_dir)
    if not samples:
        raise SystemExit(f"no cached pairs under {cache_dir}")
    encoder = encoder_id(req.text_encoder) + " + vision tower bf16"
    todo = []
    for s in samples:
        with safe_open(str(s.te_path), "pt") as f:
            meta = f.metadata() or {}
        captions = sample_captions(s.te_path)
        stamp = _stamp(captions, meta.get("source_sha256", ""), encoder, req.tail)
        path = teacher_path(s.te_path)
        if path.is_file() and not req.overwrite:
            with safe_open(str(path), "pt") as f:
                if (f.metadata() or {}).get("stamp") == stamp:
                    continue
        todo.append((s, meta, captions, stamp, path))
    print(f"teachers: {len(todo)}/{len(samples)} to encode", flush=True)
    if not todo:
        return {"encoded": 0}

    device = torch.device("cuda")
    t0 = time.time()
    tok = load_tokenizer(req.model_dir)
    te = vision_te.load_te(
        tok,
        device,
        req.model_dir,
        precision=req.text_encoder,
        fp8_path=req.te_file or None,
    )
    t_load = time.time() - t0
    t0 = time.time()
    for s, meta, captions, stamp, path in todo:
        h, w = (n * 16 for n in s.latent_hw)
        page = fit(load_rgb(Path(meta["source"])), w, h)
        kinds = list(captions)
        out = vision_te.encode_variants(
            tok, te, page, [captions[k] for k in kinds], w * h, device, TAILS[req.tail]
        )
        tensors = {
            f"embeds_{k}": hidden.to("cpu", torch.bfloat16).contiguous()
            for k, (hidden, _n, _p) in zip(kinds, out)
        }
        n_img = out[0][1]
        save_file(
            tensors,
            str(path),
            metadata={
                "stamp": stamp,
                "encoder": encoder,
                "prompt_tail": req.tail,
                "image_rows": str(n_img),
                "size": f"{w}x{h}",
                "rows": json.dumps({k: int(v.shape[0]) for k, v in tensors.items()}),
            },
        )
        print(
            f"[vis] {s.stem} {w}x{h}: {n_img} image rows, "
            + ", ".join(f"{k} {v.shape[0]}" for k, v in tensors.items()),
            flush=True,
        )
    t_enc = time.time() - t0
    del te
    empty_cache()
    print(
        f"teachers: {len(todo)} in load {t_load:.1f}s + encode {t_enc:.1f}s", flush=True
    )
    return {"encoded": len(todo), "load_s": t_load, "encode_s": t_enc}


class Teachers:
    """The teacher embeds of each sample by ``choose_embeds`` kind, read from disk per
    step (a few MB each)."""

    def __init__(self, samples):
        from safetensors import safe_open

        self.paths, self.rows = {}, {}
        missing = []
        for s in samples:
            path = teacher_path(s.te_path) if s.te_path is not None else None
            if path is None or not path.is_file():
                missing.append(s.stem)
                continue
            with safe_open(str(path), "pt") as f:
                rows = json.loads((f.metadata() or {})["rows"])
            self.paths[s.stem] = path
            self.rows[s.stem] = {k[len("embeds_") :]: n for k, n in rows.items()}
        if missing:
            raise SystemExit(
                f"distill: {len(missing)} samples have no {VIS_SUFFIX} teacher file "
                f"(e.g. {missing[0]}); run scripts/moss/distill_cache.py on the cache"
            )

    def longest(self, stem: str) -> int:
        return max(self.rows[stem].values())

    def shortest(self, stem: str) -> int:
        return min(self.rows[stem].values())

    def get(self, stem: str, kind: str) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(str(self.paths[stem]), "pt") as f:
            key = f"embeds_{kind}"
            if key not in f.keys():
                raise SystemExit(
                    f"distill: {self.paths[stem].name} has no {key}; rebuild the "
                    "teacher cache after a caption change"
                )
            return f.get_tensor(key)


class TeacherWeights:
    """The adapter weights the teacher forward runs under. ``base``: multiplier 0.
    ``ema``: a shadow of the adapter params at ``decay`` per update, swapped in for
    the forward through one scratch copy."""

    def __init__(self, mode: str, network, params, decay: float):
        self.mode, self.network, self.params, self.decay = mode, network, params, decay
        self.shadow = self.scratch = None
        if mode == "ema":
            with torch.no_grad():
                self.shadow = [p.detach().clone() for p in params]
                self.scratch = [torch.empty_like(p) for p in params]

    def update(self) -> None:
        if self.shadow is not None:
            with torch.no_grad():
                torch._foreach_lerp_(self.shadow, self.params, 1.0 - self.decay)

    def __enter__(self):
        if self.mode == "base":
            self.network.set_multiplier(0.0)
        else:
            with torch.no_grad():
                torch._foreach_copy_(self.scratch, self.params)
                torch._foreach_copy_(self.params, self.shadow)
        return self

    def __exit__(self, *exc):
        if self.mode == "base":
            self.network.set_multiplier(1.0)
        else:
            with torch.no_grad():
                torch._foreach_copy_(self.params, self.scratch)
        return False
