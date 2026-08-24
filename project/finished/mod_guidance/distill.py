"""Modulation guidance distillation.

Trains ``pooled_text_proj`` to inject pooled text embedding into the AdaLN
modulation path. The entire DiT backbone is frozen; only the small projection
MLP (~8M params) receives gradients.

Distillation setup (Starodubcev et al., ICLR 2026, Section 5):
  - Teacher: normal forward with real crossattn_emb, pooled_text_proj disabled.
  - Student: forward with T5("") crossattn_emb (unconditional T5, matches Anima's
    own CFG-uncond path at inference time — ``library/inference/text.py:99-127``),
    but pooled_text_proj receives the real pooled text vector.
  - Loss: MSE(student_pred, teacher_pred).

The unconditional sidecar is normally produced by ``make preprocess-te`` (and
also re-stageable via ``prep.py`` Phase 1) at
``post_image_dataset/_anima_uncond_te.safetensors``.

This forces pooled_text_proj to encode text information through modulation,
complementing the cross-attention path.

GAD (geometry-aware distillation; ``--gad_weight > 0``, off by default) adds a
first-order term that also matches the teacher's *response to a text change* —
see ``project/finished/mod_guidance/plan.md`` and
``docs/findings/mod_guidance_quality_tag_axis.md`` §3 (text-derivative orthogonal).

Config lives in ``project/finished/mod_guidance/config.py`` (argparser + resolved
``ModConfig`` dataclass), mirroring the ``distill_turbo/`` precedent.

Usage:
    python -m project.finished.mod_guidance.distill [--iterations 4000] [--lr 1e-4] [--batch_size 1]
"""

from __future__ import annotations

import logging
import math
import os
import random

from library.runtime.allocator import default_expandable_segments

# Mirror of train.py's allocator default (project_daemon_wiring_pattern) —
# must run before the first CUDA allocation; freefit's varying seq_len
# fragments the reserved pool over a long run without it.
default_expandable_segments()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tqdm import tqdm  # noqa: E402

from library.anima import weights as anima_utils  # noqa: E402
from library.anima.models import Anima  # noqa: E402
from library.config.resolved import dataclass_tb_text  # noqa: E402
from library.datasets.cache import make_cached_collate  # noqa: E402
from library.datasets.cache import CachedDataset  # noqa: E402
from library.io.cache import (  # noqa: E402
    LATENT_CACHE_SUFFIX,
    discover_cached_pairs,
    get_latent_resolution,
)
from library.runtime.harness import (  # noqa: E402
    compile_dit_blocks_for_pool,
    enable_training_grad_ckpt,
    place_dit_for_training,
)
from library.training.distill_runtime import (  # noqa: E402
    create_tb_writer,
    ensure_dynamic_seq_for_freefit,
    resolve_device_dtype,
)
from library.training.forward import (  # noqa: E402
    PadCache,
    renoise,
    run_mini_train_forward,
    sample_sigma,
    to_dit_5d,
)
from library.training.schedulers import make_warmup_cosine_scheduler  # noqa: E402
from library.anima.uncond import (  # noqa: E402
    default_uncond_path,
    load_uncond_crossattn,
    uncond_for_batch,
)
from .config import build_argparser, resolve_config  # noqa: E402
from .teacher_cache import (  # noqa: E402
    TeacherCache,
    ValTeacherCache,
    prefill_teacher_cache,
)
from .validation import run_validation  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _pool_token_counts(cfg, patch: int) -> set[int]:
    """Distinct DiT token counts in the cached training pool.

    Self-describing compile budget (mirrors ``train.py::_derive_token_budget`` /
    ``distill_turbo``): the on-disk caches are the source of truth for which tiers
    the pool spans. Unlike SPD, the mod head trains at each latent's *native*
    resolution (no per-stage downsampling), so the token count is just
    ``(h_lat // patch) * (w_lat // patch)``.

    When ``--synth_data_dir`` is set the real latents are swapped for the synth
    pool (``CachedDataset`` remap) and samples with no synth latent are DROPPED,
    so the forward only ever sees synth shapes. Scan the synth dir alone in that
    case — unioning ``data_dir`` injects phantom token counts (real-only tiers
    that are never fed) that widen the symbolic seq axis to a range the traced
    graphs don't satisfy → ``ConstraintViolationError``.
    """
    import glob

    if cfg.synth_data_dir is not None:
        res_strs: set[str] = {
            get_latent_resolution(npz)
            for npz in glob.glob(
                os.path.join(cfg.synth_data_dir, "**", f"*{LATENT_CACHE_SUFFIX}"),
                recursive=True,
            )
        }
    else:
        res_strs = {
            get_latent_resolution(img.npz_path)
            for img in discover_cached_pairs(cfg.data_dir)
        }
    counts: set[int] = set()
    for res in res_strs:
        a, b = (int(v) for v in res.split("x"))  # latent dims (order-agnostic)
        counts.add((a // patch) * (b // patch))
    return counts


def _draw_gad_pair(
    idx_list, crossattn_emb, pooled_text, source, rng, dataset, device, dtype
):
    """Return ``(crossattn_B, pooled_B)`` — another sample's text, used as the
    GAD perturbation direction.

    ``batch`` rolls within the batch (needs batch_size>1). ``dataset`` draws a
    random *other* sample per batch element via random-access ``dataset[j]``
    (reproducible through ``rng``); crossattn caches are max-padded to a uniform
    seq length, so a cross-bucket pairing stacks cleanly.
    """
    if source == "batch":
        return (
            torch.roll(crossattn_emb, shifts=1, dims=0),
            torch.roll(pooled_text, shifts=1, dims=0),
        )
    n = len(dataset)
    cross_list, pool_list = [], []
    for b in range(crossattn_emb.shape[0]):
        cur = idx_list[b]
        j = rng.randrange(n)
        while n > 1 and j == cur:
            j = rng.randrange(n)
        s_j = dataset[j]
        cross_list.append(s_j["crossattn_emb"])
        pool_list.append(s_j["pooled_text"])
    cross_b = torch.stack(cross_list).to(device, dtype=dtype, non_blocking=True)
    pool_b = torch.stack(pool_list).to(device, dtype=dtype, non_blocking=True)
    return cross_b, pool_b


def main():
    args = build_argparser().parse_args()
    cfg = resolve_config(args)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # Dry run: test DataLoader collation without loading the model.
    if cfg.dry_run:
        dataset = CachedDataset(
            cfg.data_dir,
            batch_size=cfg.batch_size,
            sample_ratio=cfg.sample_ratio,
            synth_data_dir=cfg.synth_data_dir,
        )

        dl = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            collate_fn=make_cached_collate(),
        )
        total = len(dl)
        for i, (_idxs, lat, te, pooled) in enumerate(tqdm(dl, desc="dry-run")):
            if (i + 1) % 200 == 0:
                logger.info(
                    f"  batch {i + 1}/{total}  latents={lat.shape}  te={te.shape}  pooled={pooled.shape}"
                )
        logger.info(f"Dry run OK: {total} batches, no collation errors.")
        return

    device, dtype = resolve_device_dtype()

    # Load the T5("") uncond sidecar (staged by `make preprocess-te` / `prep.py`).
    uncond_te_path = cfg.uncond_te_path or str(default_uncond_path())
    uncond_te_1 = load_uncond_crossattn(uncond_te_path, device, dtype)
    logger.info(
        f"Loaded uncond crossattn from {uncond_te_path} "
        f"(shape={tuple(uncond_te_1.shape)})"
    )

    logger.info("Loading DiT model...")
    model: Anima = anima_utils.load_anima_model(
        device,
        cfg.dit_path,
        attn_mode=cfg.attn_mode,
        loading_device="cpu" if cfg.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )

    # pooled_text_proj isn't in the pretrained checkpoint, so its params are
    # meta tensors after load_state_dict — materialize on CPU before any .to().
    model.pooled_text_proj.to_empty(device="cpu")
    nn.init.kaiming_uniform_(model.pooled_text_proj[0].weight, a=math.sqrt(5))
    nn.init.zeros_(model.pooled_text_proj[0].bias)
    nn.init.zeros_(model.pooled_text_proj[-1].weight)
    nn.init.zeros_(model.pooled_text_proj[-1].bias)
    if cfg.mod_sigma_film:
        # Same meta→CPU materialization for the σ-FiLM sibling; zero-init ⇒
        # identity FiLM, so the head starts equal to the plain σ-flat head.
        model.pooled_text_sigma_film.to_empty(device="cpu")
        nn.init.zeros_(model.pooled_text_sigma_film.weight)
        nn.init.zeros_(model.pooled_text_sigma_film.bias)

    if cfg.resume:
        logger.info(f"Resuming from {cfg.resume}")
        from safetensors.torch import load_file

        state = load_file(cfg.resume)
        film_state = {
            k[len("sigma_film.") :]: v
            for k, v in state.items()
            if k.startswith("sigma_film.")
        }
        model.pooled_text_proj.load_state_dict(
            {k: v for k, v in state.items() if not k.startswith("sigma_film.")}
        )
        if film_state:
            model.pooled_text_sigma_film.load_state_dict(film_state)

    # Block swap (two forwards/step), then compile each block._forward
    # (native-shape flatten → no flash pad-leak into the target).
    place_dit_for_training(model, device, blocks_to_swap=cfg.blocks_to_swap)
    if cfg.torch_compile:
        # Self-describing compile budget: distinct token counts from the cached
        # pool, not a fixed table. The derivation is coupled to mod's synth-pool
        # logic so it stays here; the rest is the shared library helper.
        token_counts = _pool_token_counts(cfg, model.patch_spatial)
        # Free-fit fail-safe (mirrors train.py's auto-enable, which never reaches
        # this bespoke loop — project_daemon_wiring_pattern). cfg is frozen → local
        # override.
        dynamic_seq = ensure_dynamic_seq_for_freefit(
            token_counts, cfg.compile_dynamic_seq, logger=logger
        )
        pc = compile_dit_blocks_for_pool(
            model,
            token_counts,
            enabled=True,
            dynamic_seq=dynamic_seq,
            mode=cfg.compile_inductor_mode,
            activation_memory_budget=cfg.activation_memory_budget,
            grad_ckpt=cfg.grad_ckpt,
            logger=logger,
        )
        logger.info(
            "torch_compile: %d block._forward compiled (mode=%s, dynamic_seq=%s); "
            "%d distinct token counts in pool, %s.",
            len(model.blocks),
            cfg.compile_inductor_mode,
            dynamic_seq,
            pc.n_shapes,
            (
                f"seq_range={pc.seq_range} (one symbolic graph)"
                if dynamic_seq
                else "static per-shape graphs"
            ),
        )

    # Grad-ckpt recomputes block activations in backward (only the student pass
    # holds them — teacher is no_grad). Keep train() mode: Block.forward gates
    # checkpointing on self.training.
    enable_training_grad_ckpt(model, enabled=cfg.grad_ckpt)
    model.train()

    # The trainable mod head: pooled_text_proj, plus the σ-FiLM generator when
    # enabled. Used for unfreeze / fp32 cast / optimizer / grad-norm / save.
    mod_modules = [model.pooled_text_proj]
    if cfg.mod_sigma_film:
        model.enable_pooled_text_sigma_film = True
        mod_modules.append(model.pooled_text_sigma_film)
        logger.info(
            "σ-FiLM ON: mod head hidden is FiLM-modulated by the time embedding "
            "(text push scales/re-aims per σ)"
        )

    def mod_parameters():
        for m in mod_modules:
            yield from m.parameters()

    # Freeze everything, then unfreeze the mod head.
    for param in model.parameters():
        param.requires_grad_(False)
    for param in mod_parameters():
        param.requires_grad_(True)

    # Arm the student's pooled_text_proj path; without it the zero-init gate
    # skips the proj and starves its gradient (teacher stays skip_pooled_text_proj).
    model.enable_pooled_text_modulation = True

    for m in mod_modules:
        m.to(dtype=torch.float32)

    trainable_params = sum(p.numel() for p in mod_parameters())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable: {trainable_params:,} / {total_params:,} params "
        f"({trainable_params / total_params * 100:.4f}%)"
    )

    optimizer = torch.optim.AdamW(
        list(mod_parameters()),
        lr=cfg.lr,
        fused=torch.cuda.is_available(),
    )

    warmup_steps = (
        int(cfg.warmup) if cfg.warmup >= 1 else int(cfg.warmup * cfg.iterations)
    )
    scheduler = make_warmup_cosine_scheduler(
        optimizer, cfg.iterations, cfg.lr, warmup_steps=warmup_steps
    )

    dataset = CachedDataset(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        split="train",
        validation_split=cfg.validation_split,
        validation_seed=cfg.validation_seed,
        sample_ratio=cfg.sample_ratio,
        synth_data_dir=cfg.synth_data_dir,
    )

    val_dataset = None
    val_dataloader = None
    if cfg.validation_split > 0.0:
        val_dataset = CachedDataset(
            cfg.data_dir,
            batch_size=cfg.batch_size,
            split="val",
            validation_split=cfg.validation_split,
            validation_seed=cfg.validation_seed,
            sample_ratio=cfg.sample_ratio,
            synth_data_dir=cfg.synth_data_dir,
        )

    # Bucket-grouped batch sampler: every batch is one resolution (so the
    # stacking collate works at batch_size>1); largest-token bucket pinned first.
    collate_fn = make_cached_collate()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=dataset.make_batch_sampler(shuffle=cfg.shuffle, seed=cfg.seed),
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if val_dataset is not None and len(val_dataset) > 0:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
    elif cfg.validation_split > 0.0:
        logger.warning(
            "validation_split>0 but val set is empty after bucket-remainder drop; "
            "skipping validation. Lower batch_size or raise validation_split."
        )

    # Teacher prediction cache: keyed by (sample_idx, sigma_idx) so revisits
    # skip the teacher forward; sigmas pre-sampled from the original sampler's
    # sigmoid(scale·N(0,1)) grid, noise deterministic per pair.
    teacher_cache = None
    if not cfg.no_teacher_cache:
        teacher_cache = TeacherCache(
            K=cfg.teacher_cache_K,
            sigmoid_scale=cfg.sigmoid_scale,
            base_seed=cfg.teacher_cache_seed,
        )
        _peek = dataset[0]["latents"]
        bytes_per_entry = _peek.numel() * 2
        approx_gb = len(dataset) * cfg.teacher_cache_K * bytes_per_entry / 1e9
        logger.info(
            f"Teacher cache enabled: K={cfg.teacher_cache_K} sigmas, "
            f"{len(dataset)} samples → up to {len(dataset) * cfg.teacher_cache_K} entries, "
            f"~{approx_gb:.2f} GB RAM at full fill (bf16)."
        )
        if cfg.prefill_teacher_cache:
            prefill_teacher_cache(teacher_cache, dataset, model, device, dtype)

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)

    writer, _run_log = create_tb_writer(
        cfg.log_dir, dataclass_tb_text(cfg), enabled=not cfg.no_log, logger=logger
    )

    # GAD setup. gad_weight=0 → fully off (no extra forwards, bit-for-bit the
    # MSE-only path). Resolve the perturbation-pair source against batch_size.
    gad_enabled = cfg.gad_weight > 0.0
    gad_source = None
    gad_rng = None
    if gad_enabled:
        if cfg.gad_pair_source == "batch":
            gad_source = "batch"
        elif cfg.gad_pair_source == "dataset":
            gad_source = "dataset"
        else:  # auto
            gad_source = "batch" if cfg.batch_size > 1 else "dataset"
        if gad_source == "batch" and cfg.batch_size < 2:
            logger.warning(
                "gad_pair_source=batch needs batch_size>1 (a roll of a size-1 "
                "batch pairs a sample with itself → ΔT=0). Falling back to "
                "dataset-random pairing."
            )
            gad_source = "dataset"
        gad_rng = random.Random(cfg.seed)
        logger.info(
            f"GAD ON: weight={cfg.gad_weight}, h={cfg.gad_h}, loss={cfg.gad_loss}, "
            f"pair_source={gad_source} (+1 grad student forward, +1 no_grad teacher "
            "forward per accum step; two-phase backward keeps peak VRAM at one "
            "student graph — fits without --grad_ckpt)."
        )

    grad_accum = cfg.grad_accum
    logger.info(
        f"Starting distillation: {cfg.iterations} iterations, "
        f"grad_accum={grad_accum} (effective batch={cfg.batch_size * grad_accum})"
    )

    data_iter = iter(dataloader)
    running_loss = 0.0
    log_interval = 50

    val_enabled = val_dataloader is not None and cfg.validate_every_n_steps > 0
    best_val_loss = float("inf")
    val_teacher_cache = (
        ValTeacherCache() if val_enabled and not cfg.no_val_teacher_cache else None
    )

    pad_cache = PadCache(dtype)
    progress = tqdm(range(cfg.iterations), desc="distill")
    accum_loss_t = torch.zeros((), device=device)
    accum_mse_t = torch.zeros((), device=device)
    accum_gad_t = torch.zeros((), device=device)
    accum_gad_cos_t = torch.zeros((), device=device)
    for step in progress:
        accum_loss_t.zero_()
        accum_mse_t.zero_()
        accum_gad_t.zero_()
        accum_gad_cos_t.zero_()

        for accum_step in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            idx_list = batch["idx"]
            latents = batch["latents"]
            crossattn_emb = batch["crossattn_emb"]
            pooled_text = batch["pooled_text"]

            latents = latents.to(device, dtype=dtype, non_blocking=True)
            crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
            pooled_text = pooled_text.to(device, dtype=dtype, non_blocking=True)

            B = latents.shape[0]

            # Sigma + noise: with the cache, draw from the K-grid + deterministic
            # per-(sample, sigma) noise so hits and misses give identical
            # (latents, noise, sigma) to the student; without it, the original
            # continuous-sigmoid sampler + fresh noise.
            if teacher_cache is not None:
                sigma_idx_list = teacher_cache.sample_sigma_idx(B)
                sigmas = torch.tensor(
                    [teacher_cache.get_sigma(si) for si in sigma_idx_list],
                    device=device,
                    dtype=latents.dtype,
                )
                noise_parts = [
                    teacher_cache.make_noise(
                        idx_list[i],
                        sigma_idx_list[i],
                        (1,) + tuple(latents.shape[1:]),
                        device,
                        latents.dtype,
                    )
                    for i in range(B)
                ]
                noise = torch.cat(noise_parts, dim=0)
            else:
                sigma_idx_list = None
                noise = torch.randn_like(latents)
                sigmas = sample_sigma(
                    B,
                    sigmoid_scale=cfg.sigmoid_scale,
                    device=device,
                    dtype=torch.float32,
                )

            timesteps = sigmas  # [0, 1] range (model expects this)

            # Noisy input: (1-σ)·latents + σ·noise, then 5D (B,16,1,H,W) for the DiT.
            noisy_input = to_dit_5d(renoise(latents, sigmas, noise))

            # Padding mask (all zeros = no padding); recycled per spatial shape.
            padding_mask = pad_cache.get(latents)

            # Teacher forward: real crossattn, pooled_text_proj skipped (skipped
            # entirely on a full-batch cache hit).
            cached_list = None
            if teacher_cache is not None:
                cached_list = [
                    teacher_cache.get(idx_list[i], sigma_idx_list[i]) for i in range(B)
                ]
                all_hit = all(c is not None for c in cached_list)
            else:
                all_hit = False

            if all_hit:
                teacher_pred = torch.cat(
                    [c.to(device, dtype=dtype) for c in cached_list], dim=0
                )
            else:
                # clone=True: teacher_pred must outlive the student call (and any
                # third compiled-fn invocation) before it's read into the loss.
                teacher_pred = run_mini_train_forward(
                    model,
                    noisy_input,
                    timesteps,
                    crossattn_emb,
                    padding_mask=padding_mask,
                    dtype=dtype,
                    no_grad=True,
                    clone=True,
                    skip_pooled_text_proj=True,
                )
                if teacher_cache is not None:
                    for i in range(B):
                        if cached_list[i] is None:
                            teacher_cache.put(
                                idx_list[i], sigma_idx_list[i], teacher_pred[i : i + 1]
                            )

            # Student forward: T5("") crossattn, real pooled text through proj.
            # requires_grad_ needed for gradient checkpointing.
            noisy_input = noisy_input.requires_grad_()
            uncond_crossattn = uncond_for_batch(uncond_te_1, crossattn_emb)
            student_pred = run_mini_train_forward(
                model,
                noisy_input,
                timesteps,
                uncond_crossattn,
                padding_mask=padding_mask,
                dtype=dtype,
                pooled_text_override=pooled_text,
            )
            # MSE loss (base pointwise term).
            mse_loss = nn.functional.mse_loss(
                student_pred.float(), teacher_pred.float()
            )

            if not gad_enabled:
                (mse_loss / grad_accum).backward()
                accum_loss_t += (mse_loss / grad_accum).detach()
                accum_mse_t += mse_loss.detach()
                continue

            # GAD term: match the teacher's finite-difference text response. ΔT
            # rides cross-attn (teacher, no_grad), ΔS rides the modulation MLP
            # (student). Two-phase backward holds peak VRAM at ONE student graph:
            # back-prop the base term first (frees student_pred's activations
            # before the perturbed graph), then treat student(A) as detached.
            # Target student(B) → student(A) + ΔT — since L_mse drives the
            # A-residual to ~0, GAD pulls the B-residual the same way
            # (synergistic). Grad flows through the perturbed (B) student only.
            (mse_loss / grad_accum).backward()
            # clone lifts the A-baseline VALUES off student_pred's (possibly
            # reused) static output buffer before the perturbed forwards overwrite it.
            student_a = student_pred.detach().clone()

            cross_b, pool_b = _draw_gad_pair(
                idx_list,
                crossattn_emb,
                pooled_text,
                gad_source,
                gad_rng,
                dataset,
                device,
                dtype,
            )
            if cfg.gad_h == 1.0:
                cross_pert, pool_pert = cross_b, pool_b
            else:
                cross_pert = crossattn_emb + cfg.gad_h * (cross_b - crossattn_emb)
                pool_pert = pooled_text + cfg.gad_h * (pool_b - pooled_text)

            # Perturbed teacher (frozen, no_grad): ΔT = v(cross_pert) − v(cross_A).
            teacher_pert = run_mini_train_forward(
                model,
                noisy_input,
                timesteps,
                cross_pert,
                padding_mask=padding_mask,
                dtype=dtype,
                no_grad=True,
                clone=True,
                skip_pooled_text_proj=True,
            )
            dT = (teacher_pert - teacher_pred).float()

            # Perturbed student (grad → pooled_text_proj): ΔS = v(pool_pert) − v(pool_A).
            student_pert = run_mini_train_forward(
                model,
                noisy_input,
                timesteps,
                uncond_crossattn,
                padding_mask=padding_mask,
                dtype=dtype,
                pooled_text_override=pool_pert,
            )
            dS = student_pert.float() - student_a.float()

            if cfg.gad_loss == "cosine":
                gad_loss = 1.0 - nn.functional.cosine_similarity(
                    dS.flatten(), dT.flatten(), dim=0
                )
            else:  # l2 (paper-faithful; also penalizes the magnitude gap)
                gad_loss = nn.functional.mse_loss(dS, dT)

            with torch.no_grad():
                gad_cos = nn.functional.cosine_similarity(
                    dS.flatten(), dT.flatten(), dim=0
                )
            (cfg.gad_weight * gad_loss / grad_accum).backward()

            accum_loss_t += (
                mse_loss.detach() + cfg.gad_weight * gad_loss.detach()
            ) / grad_accum
            accum_mse_t += mse_loss.detach()
            accum_gad_t += gad_loss.detach()
            accum_gad_cos_t += gad_cos

        grad_norm = None
        if writer is not None and (step + 1) % cfg.log_interval == 0:
            sq = 0.0
            for p in mod_parameters():
                if p.grad is not None:
                    sq += p.grad.detach().float().pow(2).sum().item()
            grad_norm = sq**0.5

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        accum_loss = accum_loss_t.item()
        running_loss += accum_loss
        lr = scheduler.get_last_lr()[0]

        if writer is not None and (step + 1) % cfg.log_interval == 0:
            writer.add_scalar("train/loss", accum_loss, step + 1)
            writer.add_scalar(
                "train/loss_mse", accum_mse_t.item() / grad_accum, step + 1
            )
            writer.add_scalar("train/lr", lr, step + 1)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm, step + 1)
            if gad_enabled:
                writer.add_scalar(
                    "train/loss_gad", accum_gad_t.item() / grad_accum, step + 1
                )
                writer.add_scalar(
                    "train/gad_cos", accum_gad_cos_t.item() / grad_accum, step + 1
                )
            if teacher_cache is not None:
                tc_total = teacher_cache.hits + teacher_cache.misses
                hit_rate = teacher_cache.hits / tc_total if tc_total else 0.0
                writer.add_scalar("teacher_cache/hit_rate", hit_rate, step + 1)
                writer.add_scalar("teacher_cache/size", len(teacher_cache), step + 1)

        if (step + 1) % log_interval == 0:
            avg = running_loss / log_interval
            progress.set_postfix(loss=f"{avg:.6f}", lr=f"{lr:.2e}")
            if writer is not None:
                writer.add_scalar("train/loss_avg50", avg, step + 1)
            running_loss = 0.0
        else:
            progress.set_postfix(loss=f"{accum_loss:.6f}", lr=f"{lr:.2e}")

        do_validate = (
            val_dataloader is not None
            and cfg.validate_every_n_steps > 0
            and (
                (step + 1) % cfg.validate_every_n_steps == 0
                or (step + 1) == cfg.iterations
            )
        )
        improved = False
        overall_mean = None
        if do_validate:
            per_sigma_mean, overall_mean = run_validation(
                model,
                val_dataloader,
                device=device,
                dtype=dtype,
                sigmas=cfg.validation_sigmas,
                max_steps=cfg.max_validation_steps,
                seed=cfg.validation_seed,
                uncond_te_1=uncond_te_1,
                teacher_cache=val_teacher_cache,
            )
            sigma_str = ", ".join(
                f"σ={s:.2f}:{v:.4e}" for s, v in per_sigma_mean.items()
            )
            logger.info(f"[val @ step {step + 1}] mean={overall_mean:.6f}  {sigma_str}")
            if writer is not None:
                writer.add_scalar("val/loss", overall_mean, step + 1)
                for s, v in per_sigma_mean.items():
                    writer.add_scalar(f"val/loss_sigma_{s:.2f}", v, step + 1)
                if val_teacher_cache is not None:
                    vc_total = val_teacher_cache.hits + val_teacher_cache.misses
                    vc_hit_rate = val_teacher_cache.hits / vc_total if vc_total else 0.0
                    writer.add_scalar(
                        "val_teacher_cache/hit_rate", vc_hit_rate, step + 1
                    )
                    writer.add_scalar(
                        "val_teacher_cache/size", len(val_teacher_cache), step + 1
                    )
            if overall_mean < best_val_loss:
                best_val_loss = overall_mean
                improved = True

        # Save checkpoint: when validation is enabled, only overwrite on
        # val-loss improvement. Otherwise fall back to step-cadence saves.
        if val_enabled:
            should_save = improved
        else:
            should_save = (step + 1) % cfg.save_every == 0 or (
                step + 1
            ) == cfg.iterations
        if should_save:
            save_path = cfg.output_path
            state = {
                k: v.to(torch.bfloat16)
                for k, v in model.pooled_text_proj.state_dict().items()
            }
            if cfg.mod_sigma_film:
                # σ-FiLM weights ride under a 'sigma_film.' prefix so load can
                # route them back (and auto-arm enable_pooled_text_sigma_film).
                for k, v in model.pooled_text_sigma_film.state_dict().items():
                    state[f"sigma_film.{k}"] = v.to(torch.bfloat16)
            save_file(state, save_path)
            if val_enabled:
                logger.info(
                    f"Saved checkpoint at step {step + 1} "
                    f"(val={overall_mean:.6f}, new best) -> {save_path}"
                )
            else:
                logger.info(f"Saved checkpoint at step {step + 1} -> {save_path}")
        elif do_validate:
            logger.info(
                f"Skipped save at step {step + 1}: "
                f"val={overall_mean:.6f} >= best={best_val_loss:.6f}"
            )

    if teacher_cache is not None:
        tc_total = teacher_cache.hits + teacher_cache.misses
        hit_rate = (teacher_cache.hits / tc_total * 100) if tc_total else 0.0
        logger.info(
            f"Teacher cache final: {len(teacher_cache)} entries, "
            f"{teacher_cache.hits} hits / {teacher_cache.misses} misses "
            f"({hit_rate:.1f}% hit rate)"
        )

    if val_teacher_cache is not None:
        vc_total = val_teacher_cache.hits + val_teacher_cache.misses
        vc_hit_rate = (val_teacher_cache.hits / vc_total * 100) if vc_total else 0.0
        logger.info(
            f"Val teacher cache final: {len(val_teacher_cache)} entries, "
            f"{val_teacher_cache.hits} hits / {val_teacher_cache.misses} misses "
            f"({vc_hit_rate:.1f}% hit rate)"
        )

    if writer is not None:
        writer.close()
    logger.info("Distillation complete.")


if __name__ == "__main__":
    main()
