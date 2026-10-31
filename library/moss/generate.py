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

import time
from pathlib import Path

import torch

from library.env import resolve_under_home
from library.moss.generation.conditioning import (  # noqa: F401  (this module is their import path)
    cached_embeds,
    ref_latents,
    vision_embeds,
)
from library.moss.generation.jobs import (  # noqa: F401
    build_jobs,
    fingerprint,
    parse_skip,
    ref_images,
)
from library.moss.generation.records import (  # noqa: F401
    image_path,
    latent_path,
    load_latent,
    read_latent_record,
    save_latent,
    write_manifest,
)
from library.moss.loader import (
    compile_blocks,
    empty_cache,
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
    swap = req.blocks_to_swap
    transformer = load_transformer(
        model_dir, req.model, device=None if swap else device
    )
    print(
        f"[dit] {req.model} loaded in {time.time() - t0:.1f}s, "
        f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB",
        flush=True,
    )
    if req.lora:
        from library.moss.lora import load_for_inference

        network = load_for_inference(
            transformer, resolve_under_home(req.lora), req.lora_multiplier
        )
        # Held outside the blocks: the swapper never moves it.
        network.to(device)
        print(f"[lora] {req.lora} x{req.lora_multiplier}", flush=True)
    if req.stream == "fp32":
        stream_fp32_(transformer)
    attached = None
    if swap:
        from library.runtime.blockswap import place

        attached = place(
            transformer, "layers", device, blocks_to_swap=swap, label="[dit]"
        )
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
        save_latent(latent_path(out_dir, rec), latent, rec)
        print(
            f"[{req.model}] {j['key']} {j['width']}x{j['height']} cfg {j['cfg']} "
            f"{len(sigmas)} steps{f' ({dave.describe()})' if dave else ''}: "
            f"{seconds:.1f}s, peak {peak:.1f} GiB -> "
            f"{latent_path(out_dir, rec).name}",
            flush=True,
        )
    # fp32 decode with the DiT off the card.
    if attached is not None:
        attached.detach()
    transformer.to("cpu")
    del transformer
    empty_cache()


def _decode_all(model_dir, out_dir, records, device) -> None:
    """Decode each record's latent from disk; an existing PNG is kept."""
    todo = [r for r in records if not image_path(out_dir, r).is_file()]
    for r in records:
        r["image"] = str(image_path(out_dir, r))
    if not todo:
        return
    from diffusers.image_processor import VaeImageProcessor

    vae = load_vae(model_dir, device=device)
    processor = VaeImageProcessor(vae_scale_factor=16)
    for rec in todo:
        torch.cuda.reset_peak_memory_stats()
        image = decode(load_latent(latent_path(out_dir, rec)).to(device), vae)
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
        jobs = build_jobs(req)
        for j in jobs:
            if j["width"] % 16 or j["height"] % 16:
                raise SystemExit(
                    f"{j['key']}: {j['width']}x{j['height']} is not a multiple of 16"
                )
        records, pending = [], []
        for j in jobs:
            rec = fingerprint(req, j)
            saved = latent_path(out_dir, rec)
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
                image_path(out_dir, rec).unlink(missing_ok=True)
            records.append(rec)
            pending.append((j, rec))
        if pending:
            _denoise_pending(req, model_dir, out_dir, pending, device)

    _decode_all(model_dir, out_dir, records, device)
    manifest = write_manifest(out_dir, req, records)
    print(f"manifest -> {manifest}", flush=True)
    return out_dir
