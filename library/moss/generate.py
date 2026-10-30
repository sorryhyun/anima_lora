"""MOSS-Image text-to-image: the reference sampler, run in three phases.

Encode every distinct prompt with the layer-offloaded Qwen3.5-9B and drop it,
denoise every job with the DiT resident, then take the DiT off the card before
the fp32 VAE decode (at ~2 MP the decode needs more than is free beside the DiT).

``denoise`` is the sampler loop on its own (Euler or ER-SDE on the snapped
shift-5 grid, CFG with the uncond branch skipping block 22, all combining in
fp32); with ``sampler="euler"``, ``spectrum=None`` it is
``MOSSImagePipeline.__call__`` op for op.

``ref_image`` makes each job i2i: the reference is VAE-encoded at 1/``ref_scale``
of the output's sides and rides every forward (``denoise(ref=…)``); with
``ref_vision`` it also enters the TE through the vision tower ahead of the prompt
(``library.moss.vision_te``), as a colorize cache encodes the page.

    make daemon-run ARGS="scripts/moss/generate.py --prompt '…' --steps 30"
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import time
from pathlib import Path

import torch

from library.env import resolve_under_home
from library.moss.loader import (
    MAX_SEQUENCE_LENGTH,
    compile_blocks,
    empty_cache,
    encode_prompts,
    load_transformer,
    load_vae,
    pad_embeds,
    stream_fp32_,
)
from library.moss.model import decode, snapped_sigmas
from library.moss.requests import (
    DEFAULT_CFG,
    DEFAULT_UNCOND_SKIP,
    GenerateRequest,
    resolve_model_dir,
)


def _parse_skip(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v) for v in str(value).split(",") if v.strip())


def _features(transformer, latents, t, embeds, mask, skip, batched, ref=None):
    """Image-token features of the cond rows then the uncond rows, ``(R, H·W, D)``.
    ``embeds`` / ``mask`` / ``skip`` are per-branch lists (one entry = no CFG);
    ``ref`` (clean reference latents) goes to every branch."""
    kw = {} if ref is None else {"ref": ref}
    if len(embeds) == 1:
        return [
            transformer(
                latents,
                t,
                embeds[0],
                mask[0],
                skip_layers=skip[0],
                return_features=True,
                **kw,
            )
        ]
    if not batched:
        return [
            transformer(latents, t, e, m, skip_layers=s, return_features=True, **kw)
            for e, m, s in zip(embeds, mask, skip)
        ]
    rows = []
    skips = []
    for e, m, s in zip(embeds, mask, skip):
        for b in range(e.shape[0]):
            rows.append(e[b, m[b]])
            skips.append(s)
    joint, joint_mask = pad_embeds(rows, device=latents.device)
    feats = transformer.forward_batched(
        latents.repeat(len(embeds), 1, 1, 1),
        t,
        joint,
        joint_mask,
        skip_layers=skips,
        return_features=True,
        **({} if ref is None else {"ref": ref.repeat(len(embeds), 1, 1, 1)}),
    )
    return list(feats.chunk(len(embeds)))


@torch.no_grad()
def denoise(
    transformer,
    latents,
    cond,
    cond_mask,
    uncond,
    uncond_mask,
    sigmas,
    *,
    cfg=DEFAULT_CFG,
    uncond_skip=DEFAULT_UNCOND_SKIP,
    batched=False,
    spectrum=None,
    sampler="euler",
    seed=None,
    cns=None,
    dave=None,
    on_step=None,
    progress=None,
    ref=None,
):
    """Integrate ``latents (B,128,h,w)`` over ``sigmas`` (``snapped_sigmas``:
    descending, no terminal 0) with Euler or ER-SDE (``sampler``). Returns fp32
    latents.

    ``cond`` / ``uncond`` are ``(B,S,4096)`` embeds with bool masks; ``uncond`` is
    unused when ``cfg == 1``. ``spectrum`` (optional) is called as ``begin(sigmas)``,
    then per step ``is_full_step(i)`` → ``observe(i, features)`` after a real
    forward or ``forecast(i)`` instead of one; ``features`` are the image tokens
    after the last block, ``(R, H·W, D)``, cond rows then uncond rows.

    ``sampler="er_sde"`` swaps the Euler update for Anima's ER-SDE-Solver-3
    (``library.inference.sampling.ERSDESampler``) on the same σ grid, fed
    ``x̂₀ = x_t − σ·v``; ``seed`` seeds its injected noise and ``cns``
    (``library.moss.cns.MossCNS``) recolors it. ``on_step(i, x̂₀)`` sees each
    step's guided x̂₀ (CNS calibration).

    ``ref (B,128,h,w)``: clean reference latents in every forward (i2i probe).
    Under ``ref_isolate`` at a fixed ``ref_timestep`` the reference stream runs
    once for the loop (``RefCache``) and every forward reads its keys and values.

    ``dave`` (``library.moss.dave.MossDAVE``) is installed as the transformer's
    ``layer_hook`` on its active steps (both branches) and removed after.
    """
    device = latents.device
    batch, _, height, width = latents.shape
    latents = latents.to(dtype=torch.float32)
    # FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=…): fp32, ×1000, append 0.
    sigma_t = sigmas.to(dtype=torch.float32, device=device)
    timesteps = sigma_t * 1000
    sigma_t = torch.cat([sigma_t, torch.zeros(1, device=device)])
    do_cfg = cfg != 1.0
    skip = tuple(uncond_skip or ())
    embeds, masks, skips = [cond], [cond_mask], [()]
    if do_cfg:
        embeds.append(uncond)
        masks.append(uncond_mask)
        skips.append(skip)
    er_sde = None
    if sampler == "er_sde":
        from library.inference.sampling import ERSDESampler

        er_sde = ERSDESampler(sigma_t, seed=seed, device=device, cns=cns)
    elif sampler != "euler":
        raise ValueError(f"unknown sampler {sampler!r}")
    if spectrum is not None:
        spectrum.begin(sigmas)
    if dave is not None:
        dave.begin(len(timesteps))
    if ref is not None:
        from library.moss.model import RefCache

        transformer.ref_cache = RefCache()
    try:
        for i, t in enumerate(timesteps):
            if dave is not None:
                transformer.layer_hook = dave if dave.is_active(i) else None
            if spectrum is None or spectrum.is_full_step(i):
                branches = _features(
                    transformer, latents, t, embeds, masks, skips, batched, ref
                )
                if spectrum is not None:
                    spectrum.observe(i, torch.cat(branches))
            else:
                branches = list(spectrum.forecast(i).chunk(len(embeds)))
            # The head per branch, as each reference forward applies it.
            _mod, head_mod = transformer.time_conditioning(t, batch)
            velocity = [
                transformer.head(f, head_mod, height, width).float() for f in branches
            ]
            prediction = velocity[0]
            if do_cfg:
                prediction = velocity[1] + cfg * (velocity[0] - velocity[1])
            if on_step is not None:
                on_step(i, latents - sigma_t[i] * prediction)
            if er_sde is None:
                latents = latents + (sigma_t[i + 1] - sigma_t[i]) * prediction
            else:
                latents = er_sde.step(latents, latents - sigma_t[i] * prediction, i)
            if progress is not None:
                progress(i, len(timesteps))
    finally:
        if dave is not None:
            transformer.layer_hook = None
        transformer.ref_cache = None
    return latents


def initial_noise(
    seed: int, height: int, width: int, channels: int = 128
) -> torch.Tensor:
    """fp32 noise from a CPU generator, ``(1, 128, H/16, W/16)``: what the Space
    and ``randn_tensor(generator=torch.Generator('cpu'))`` draw."""
    gen = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(
        (1, channels, height // 16, width // 16), generator=gen, dtype=torch.float32
    )


def _jobs(req: GenerateRequest) -> list[dict]:
    base = dict(
        negative=req.negative_prompt,
        width=req.width,
        height=req.height,
        cfg=req.cfg,
        skip=_parse_skip(req.uncond_skip),
    )
    if req.example:
        jobs = []
        for n, path in enumerate(
            filter(None, (p.strip() for p in req.example.split(",")))
        ):
            path = resolve_under_home(path)
            p = json.loads(path.read_text(encoding="utf-8"))
            p = p.get("parameters", p)
            jobs.append(
                dict(
                    # Gallery entries are <id>/parameters.json: key by the folder.
                    key=f"{n:03d}_space{path.parent.name if path.stem == 'parameters' else path.stem}",
                    prompt=p["prompt"],
                    negative=p.get("negative_prompt", ""),
                    width=int(p["width"]),
                    height=int(p["height"]),
                    cfg=float(p.get("cfg", p.get("guidance_scale", req.cfg))),
                    seed=int(p["seed"]),
                    skip=_parse_skip(p.get("uncond_skip_layers", req.uncond_skip)),
                    steps=int(p.get("steps", p.get("num_inference_steps", req.steps))),
                    shift=float(p.get("shift", req.shift)),
                )
            )
        return jobs
    if req.prompts_file:
        source = resolve_under_home(req.prompts_file)
        prompts = [
            ln.strip()
            for ln in source.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    else:
        source, prompts = "prompt", [req.prompt.strip()] if req.prompt.strip() else []
    if not prompts:
        if not req.ref_image:
            raise SystemExit(f"no prompts in {source}")
        prompts = [""]  # the reference alone
    if req.prompt_format != "text":
        from library.moss.cache import caption_converter

        convert = caption_converter(req.prompt_format)
        prompts = [convert(p) if p else "" for p in prompts]
    jobs = [
        dict(
            base,
            key=f"{i:03d}",
            prompt=p,
            seed=req.seed + i,
            steps=req.steps,
            shift=req.shift,
        )
        for i, p in enumerate(prompts)
    ]
    return _with_refs(req, jobs) if req.ref_image else jobs


def ref_images(spec: str) -> list[Path]:
    """``ref_image``: one image, or every image directly in a folder."""
    from library.moss.scan import IMAGE_EXTS

    path = resolve_under_home(spec)
    if path.is_dir():
        found = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not found:
            raise SystemExit(f"no images in {path}")
        return found
    if not path.is_file():
        raise SystemExit(f"no reference image at {path}")
    return [path]


def _with_refs(req: GenerateRequest, jobs: list[dict]) -> list[dict]:
    """Each job once per reference image, sized to it: its aspect, area capped at
    ``width × height``, multiples of 16."""
    from PIL import Image

    from library.moss.cache import target_size

    if req.example:
        raise SystemExit("ref_image does not combine with example")
    resolution = round(math.sqrt(req.width * req.height))
    out = []
    for ref in ref_images(req.ref_image):
        with Image.open(ref) as im:
            width, height = target_size(*im.size, resolution)
        tag = re.sub(r"[^\w.-]+", "_", ref.stem)[:48]
        for j in jobs:
            out.append(
                dict(
                    j,
                    key=f"{j['key']}_{tag}",
                    width=width,
                    height=height,
                    ref=str(ref),
                    ref_mtime=ref.stat().st_mtime_ns,
                    ref_scale=req.ref_scale,
                    ref_vision=req.ref_vision,
                )
            )
    return out


def cached_embeds(
    model_dir, texts, cache_dir: Path, precision="bf16"
) -> dict[str, torch.Tensor]:
    """TE outputs keyed by text (and TE precision), read from / written to
    ``cache_dir`` so a rerun of the same prompts skips the text encoder."""
    from safetensors.torch import load_file, save_file

    def path(text):
        key = f"{MAX_SEQUENCE_LENGTH}\0{text}"
        if precision != "bf16":
            key = f"{precision}\0{key}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return cache_dir / f"{digest}.safetensors"

    out, missing = {}, []
    for text in dict.fromkeys(texts):
        if path(text).is_file():
            out[text] = load_file(path(text))["embeds"]
        else:
            missing.append(text)
    if missing:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for text, emb in encode_prompts(
            model_dir, missing, precision=precision
        ).items():
            save_file({"embeds": emb.contiguous()}, path(text))
            out[text] = emb
    print(f"[te] {len(out) - len(missing)} cached, {len(missing)} encoded", flush=True)
    return out


def vision_embeds(
    model_dir, jobs: list[dict], cache_dir: Path, precision: str
) -> dict[str, torch.Tensor]:
    """Per ``ref_vision`` job key, the TE output of its reference page (fitted to
    the job's size) followed by its prompt, through the vision tower; read from /
    written to ``cache_dir`` like ``cached_embeds``."""
    from safetensors.torch import load_file, save_file

    from library.moss.cache import fit, load_rgb

    def path(j):
        key = "\0".join(
            map(
                str,
                (
                    "vision",
                    precision,
                    j["ref"],
                    j["ref_mtime"],
                    j["width"],
                    j["height"],
                    j["prompt"],
                ),
            )
        )
        return (
            cache_dir / f"v_{hashlib.sha1(key.encode()).hexdigest()[:16]}.safetensors"
        )

    out, missing = {}, []
    for j in jobs:
        if path(j).is_file():
            out[j["key"]] = load_file(path(j))["embeds"]
        else:
            missing.append(j)
    if missing:
        from library.moss.loader import load_tokenizer
        from library.moss.vision_te import encode, load_te

        device = torch.device("cuda")
        tok = load_tokenizer(model_dir)
        te = load_te(tok, device, model_dir, precision=precision)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for j in missing:
            w, h = j["width"], j["height"]
            page = fit(load_rgb(Path(j["ref"])), w, h)
            emb, n_img = encode(tok, te, page, j["prompt"], w * h, device)
            emb = emb.to("cpu", torch.bfloat16).contiguous()
            save_file({"embeds": emb}, path(j))
            out[j["key"]] = emb
            print(
                f"[te] {j['key']} page {w}x{h}: {emb.shape[0]} rows ({n_img} image)",
                flush=True,
            )
        del te, tok
        empty_cache()
    print(
        f"[te] vision: {len(out) - len(missing)} cached, {len(missing)} encoded",
        flush=True,
    )
    return out


def ref_latents(model_dir, jobs: list[dict], device) -> dict[str, torch.Tensor]:
    """Per job key, its reference fitted to the job's size and then to
    ``ref_size`` (as ``cache.cache_refs`` builds ``*.moss_ref``), ``(1,128,h,w)``
    fp32 on the CPU."""
    from library.moss.cache import fit, load_rgb, ref_size, to_pixels
    from library.moss.model import encode

    vae = load_vae(model_dir, device=device)
    out = {}
    for j in jobs:
        w, h = j["width"], j["height"]
        rw, rh = ref_size(w, h, j["ref_scale"])
        image = fit(fit(load_rgb(Path(j["ref"])), w, h), rw, rh)
        out[j["key"]] = encode(to_pixels(image), vae).float().cpu()
        print(f"[ref] {j['key']}: {w}x{h} -> {rw}x{rh}", flush=True)
    del vae
    empty_cache()
    return out


def build_spectrum(req: GenerateRequest):
    """A fresh Spectrum forecaster at ``req``'s settings (one per denoise), or None."""
    if not req.spectrum:
        return None
    from library.moss.spectrum import MossSpectrum

    return MossSpectrum(
        window_size=req.spectrum_window_size,
        flex_window=req.spectrum_flex_window,
        warmup=req.spectrum_warmup,
        stop_caching_step=req.spectrum_stop_caching_step,
        w=req.spectrum_w,
        m=req.spectrum_m,
        lam=req.spectrum_lam,
    )


def _build_cns(req: GenerateRequest, cfg: float):
    """``MossCNS`` for one job's cfg, or None."""
    if not req.cns:
        return None
    from library.moss.cns import MossCNS

    return MossCNS.from_spec(req.cns, req.model, cfg, req.cns_strength)


def _build_dave(req: GenerateRequest, num_blocks: int):
    """DAVE hook for every job, or None."""
    if not req.dave:
        return None
    from library.moss.dave import MossDAVE

    dave = MossDAVE.from_spec(req.dave, num_blocks, req.dave_strength, req.dave_tau)
    if req.spectrum and req.dave_tau * req.steps > req.spectrum_warmup:
        # A forecast step runs no blocks, so DAVE skips it.
        print(
            f"[dave] tau {req.dave_tau} of {req.steps} steps reaches past Spectrum's "
            f"{req.spectrum_warmup} warmup steps; forecast steps are not attenuated",
            flush=True,
        )
    return dave


def _fingerprint(req: GenerateRequest, job: dict) -> dict:
    """What a saved latent was made from; a resume reuses it only on a match."""
    spectrum = {
        k: v for k, v in dataclasses.asdict(req).items() if k.startswith("spectrum")
    }
    # Stamped only off bf16, so bf16 latents without the field still match.
    te = {} if req.text_encoder == "bf16" else {"text_encoder": req.text_encoder}
    return dict(
        {k: v for k, v in job.items() if k != "skip"},
        **te,
        uncond_skip=list(job["skip"]),
        model=req.model,
        lora=str(resolve_under_home(req.lora)) if req.lora else None,
        lora_multiplier=req.lora_multiplier if req.lora else None,
        batched=req.batched,
        sampler=req.sampler,
        spectrum=spectrum if req.spectrum else None,
        dave=(
            dict(spec=req.dave, strength=req.dave_strength, tau=req.dave_tau)
            if req.dave
            else None
        ),
        **({"cns": dict(spec=req.cns, strength=req.cns_strength)} if req.cns else {}),
        **({"stream": req.stream} if req.stream != "bf16" else {}),
    )


def _latent_path(out_dir: Path, rec: dict) -> Path:
    return (
        out_dir / "latents" / f"{rec['key']}_{rec['model']}_s{rec['seed']}.safetensors"
    )


def _image_path(out_dir: Path, rec: dict) -> Path:
    return out_dir / f"{rec['key']}_{rec['model']}_s{rec['seed']}.png"


def save_latent(path: Path, latent: torch.Tensor, record: dict) -> None:
    """Final latent + its record in the metadata, written atomically."""
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file(
        {"latent": latent.contiguous()},
        tmp,
        metadata={"record": json.dumps(record, ensure_ascii=False)},
    )
    tmp.replace(path)


def read_latent_record(path: Path) -> dict | None:
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt") as f:
            return json.loads((f.metadata() or {})["record"])
    except Exception:
        return None


def load_latent(path: Path) -> torch.Tensor:
    from safetensors.torch import load_file

    return load_file(path)["latent"]


def _write_manifest(out_dir: Path, req: GenerateRequest, records: list[dict]) -> Path:
    manifest = out_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"request": req.to_argv(), "images": records}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    return manifest


def _denoise_pending(req, model_dir, out_dir, pending, device) -> None:
    """TE for the pending jobs only, then the DiT; each latent hits disk as soon
    as its denoise returns, so a killed job loses at most the one in flight."""
    jobs = [j for j, _ in pending]
    texts = [j["prompt"] for j in jobs if not j.get("ref_vision")]
    texts += [j["negative"] for j in jobs if j["cfg"] != 1.0]
    embeds = cached_embeds(model_dir, texts, out_dir / "_te_cache", req.text_encoder)
    seen = [j for j in jobs if j.get("ref_vision")]
    seen = vision_embeds(model_dir, seen, out_dir / "_te_cache", req.text_encoder)
    refs = [j for j in jobs if j.get("ref")]
    refs = ref_latents(model_dir, refs, device) if refs else {}

    t0 = time.time()
    transformer = load_transformer(model_dir, req.model, device=device)
    print(
        f"[dit] {req.model} loaded in {time.time() - t0:.1f}s, "
        f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB",
        flush=True,
    )
    if req.lora:
        from library.moss.lora import load_for_inference

        load_for_inference(
            transformer, resolve_under_home(req.lora), req.lora_multiplier
        )
        print(f"[lora] {req.lora} x{req.lora_multiplier}", flush=True)
    if req.stream == "fp32":
        stream_fp32_(transformer)
    if req.compile:
        compile_blocks(transformer.layers)
    dave = _build_dave(req, len(transformer.layers))
    for j, rec in pending:
        cond = seen[j["key"]] if j.get("ref_vision") else embeds[j["prompt"]]
        cond, cond_mask = pad_embeds([cond], device=device)
        uncond = uncond_mask = None
        if j["cfg"] != 1.0:
            uncond, uncond_mask = pad_embeds([embeds[j["negative"]]], device=device)
        sigmas = snapped_sigmas(j["steps"], j["shift"])
        noise = initial_noise(j["seed"], j["height"], j["width"]).to(device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        latent = denoise(
            transformer,
            noise,
            cond,
            cond_mask,
            uncond,
            uncond_mask,
            sigmas,
            cfg=j["cfg"],
            uncond_skip=j["skip"],
            batched=req.batched,
            spectrum=build_spectrum(req),
            sampler=req.sampler,
            seed=j["seed"],
            cns=_build_cns(req, j["cfg"]),
            dave=dave,
            ref=refs[j["key"]].to(device) if j["key"] in refs else None,
        ).cpu()
        torch.cuda.synchronize()
        seconds = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        rec.update(
            steps_run=len(sigmas),
            text_tokens=int(cond.shape[1]),
            denoise_s=round(seconds, 2),
            denoise_peak_gib=round(peak, 2),
            compiled=req.compile,
        )
        save_latent(_latent_path(out_dir, rec), latent, rec)
        print(
            f"[{req.model}] {j['key']} {j['width']}x{j['height']} cfg {j['cfg']} "
            f"{len(sigmas)} steps{f' ({dave.describe()})' if dave else ''}: "
            f"{seconds:.1f}s, peak {peak:.1f} GiB -> "
            f"{_latent_path(out_dir, rec).name}",
            flush=True,
        )
    # fp32 decode with the DiT off the card.
    transformer.to("cpu")
    del transformer
    empty_cache()


def _decode_all(model_dir, out_dir, records, device) -> None:
    """Decode each record's latent from disk; an existing PNG is kept."""
    todo = [r for r in records if not _image_path(out_dir, r).is_file()]
    for r in records:
        r["image"] = str(_image_path(out_dir, r))
    if not todo:
        return
    from diffusers.image_processor import VaeImageProcessor

    vae = load_vae(model_dir, device=device)
    processor = VaeImageProcessor(vae_scale_factor=16)
    for rec in todo:
        torch.cuda.reset_peak_memory_stats()
        image = decode(load_latent(_latent_path(out_dir, rec)).to(device), vae)
        processor.postprocess(image, output_type="pil")[0].save(rec["image"])
        rec["decode_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        print(
            f"[vae] {rec['key']} -> {rec['image']} (peak {rec['decode_peak_gib']} GiB)",
            flush=True,
        )
    del vae
    empty_cache()


def run_generate(req: GenerateRequest) -> Path:
    """Render every job; returns the output folder.

    Latents land in ``<out_dir>/latents/`` (record in the metadata) before any
    decode, and a rerun skips every job whose saved latent matches it, then
    decodes only the PNGs that are missing. ``decode_only`` decodes whatever is
    in ``latents/`` and loads nothing else."""
    if req.cns and req.sampler != "er_sde":
        raise SystemExit("cns recolors ER-SDE noise: needs --sampler er_sde")
    model_dir = resolve_model_dir(req.model_dir)
    out_dir = resolve_under_home(req.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    device = torch.device("cuda")

    if req.decode_only:
        records = [
            r
            for r in map(
                read_latent_record, sorted((out_dir / "latents").glob("*.safetensors"))
            )
            if r
        ]
        if not records:
            raise SystemExit(f"no latents in {out_dir / 'latents'}")
    else:
        jobs = _jobs(req)
        for j in jobs:
            if j["width"] % 16 or j["height"] % 16:
                raise SystemExit(
                    f"{j['key']}: {j['width']}x{j['height']} is not a multiple of 16"
                )
        records, pending = [], []
        for j in jobs:
            rec = _fingerprint(req, j)
            saved = _latent_path(out_dir, rec)
            old = read_latent_record(saved) if saved.is_file() else None
            if old is not None and all(old.get(k) == v for k, v in rec.items()):
                print(f"[resume] {j['key']}: latent on disk, skipped", flush=True)
                records.append(old)
                continue
            if old is not None:
                print(
                    f"[resume] {j['key']}: saved latent was made differently, redoing",
                    flush=True,
                )
                _image_path(out_dir, rec).unlink(missing_ok=True)
            records.append(rec)
            pending.append((j, rec))
        if pending:
            _denoise_pending(req, model_dir, out_dir, pending, device)

    _decode_all(model_dir, out_dir, records, device)
    manifest = _write_manifest(out_dir, req, records)
    print(f"manifest -> {manifest}", flush=True)
    return out_dir
