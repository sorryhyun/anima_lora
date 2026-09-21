#!/usr/bin/env python3
"""DP-DMD Phase 0 — does first-step anchoring recover diversity on Anima?

Background. We are evaluating DP-DMD (arXiv 2602.03139) as a replacement for the
CA-decoupled DMD2 turbo path that collapses sample diversity. DP-DMD's whole bet
is that the diversity loss lives in the *first* denoising step: supervise step 1
against a teacher's multi-step anchor (and detach after it) and the remaining DMD
steps refine quality without re-collapsing. Before any training rewrite we test
the premise *as an inference intervention* — zero gradient steps.

Three rollouts per (prompt × R seeds), all deterministic Euler so diversity comes
purely from the initial noise ε:

  * teacher   — base DiT, ``teacher_steps`` + CFG  → diversity CEILING
  * student   — turbo LoRA, ``student_steps``, cfg=1 → collapsed FLOOR (known low)
  * anchor@b  — run the teacher down to the student's step-b boundary σ (a diverse
                landing point), then hand that latent to the student to finish
                steps b..N. b=1 is the DP-DMD first-step bet; b=2,3 sweep *where*
                diversity is determined (reproduces their Table 3 / Fig A on Anima).

The teacher is integrated over the UNION of the teacher + student σ grids, so it
lands *exactly* on each student boundary σ (no grid-mismatch bias) and its σ=0
terminal doubles as the teacher-full image — one rollout yields the ceiling and
every anchor.

Decision. gap = div(teacher) − div(student); recovery_b = (div(anchor_b) − div(
student)) / gap. If recovery@1 is large (≳0.5) and the b-sweep shows diversity is
set early, first-step supervision is a live lever on Anima → GO. If anchor@1
re-collapses toward the student floor, the later DMD steps wash it out → NO-GO.

Diversity is Eq. 9 (1 − mean pairwise cosine) over PE-Core pooled features — the
in-repo encoder behind CMMD; a relative-diversity signal across arms, not DINOv3.

    python bench/turbo/probe_first_step_anchor.py \
        --turbo output/ckpt/anima_turbo_H_4k.safetensors --n_seeds 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from anima_lora import GenerationRequest, load_vae  # noqa: E402
from bench._anima import add_model_args  # noqa: E402
from library.inference.models import load_shared_models  # noqa: E402
from library.runtime.harness import build_anima  # noqa: E402
from library.inference.text import prepare_text_inputs  # noqa: E402
from library.inference.sampling import get_timesteps_sigmas  # noqa: E402
from library.inference.adapters import (  # noqa: E402
    compute_and_set_hydra_fei,
    set_hydra_crossattn,
    set_hydra_sigma,
)
from library.runtime.device import clean_memory_on_device  # noqa: E402
from library.vision.encoders import get_encoder_info  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402

# Curated prompts where prompt-conditioned diversity is meaningful (open-ended
# subjects + composition latitude). One line each.
DEFAULT_PROMPTS = [
    "a smiling woman with red hair, green eyes, and dimples",
    "a knight standing in a ruined cathedral, dramatic lighting",
    "a cat and a dog sitting on a sofa",
    "a futuristic city skyline at sunset, highly detailed",
    "a portrait of an old fisherman, weathered face",
    "a fox wandering through a snowy forest, golden hour",
    "a bowl of ramen on a wooden table, steam rising",
    "a dragon perched on a castle tower, stormy sky",
]


def _build_args(opts) -> argparse.Namespace:
    """A fully-populated inference Namespace (base DiT, no static LoRA merge)."""
    req = GenerationRequest(
        dit=opts.dit,
        vae=opts.vae,
        text_encoder=opts.text_encoder,
        prompt=DEFAULT_PROMPTS[0],
        save_path="output/tests/_dpdmd_probe.png",
        infer_steps=opts.student_steps,
        guidance_scale=1.0,
        image_size=(opts.size[0], opts.size[1]),  # (H, W)
        seed=opts.seed,
    )
    return req.to_args()


def _union_sigmas(teacher_sigmas, student_sigmas) -> torch.Tensor:
    """Descending unique union of both σ grids (spans 1 → 0)."""
    vals = torch.cat([teacher_sigmas.flatten(), student_sigmas.flatten()])
    uniq = torch.unique(vals)  # ascending
    return torch.flip(uniq, dims=[0])  # descending


@torch.no_grad()
def _integrate(
    anima,
    turbo_net,
    latents,
    sigma_arr,
    start,
    stop,
    *,
    enabled,
    cfg,
    embed,
    neg_embed,
    pad,
    device,
    snapshot_sigmas=None,
    snapshots=None,
):
    """Deterministic Euler over sigma_arr[start:stop+1]. Optional σ snapshots.

    Teacher vs student toggles via ``set_multiplier`` (not ``set_enabled``):
    multiplier 0.0 zeroes the LoRA delta without changing control flow, so a
    compiled DiT keeps one graph family per token count instead of recompiling
    on every teacher↔student flip.
    """
    turbo_net.set_multiplier(1.0 if enabled else 0.0)
    for i in range(start, stop):
        s_i = sigma_arr[i]
        s_next = sigma_arr[i + 1]
        t_b = s_i.to(device=device, dtype=torch.bfloat16).expand(latents.shape[0])

        set_hydra_sigma(anima, t_b)
        compute_and_set_hydra_fei(anima, latents)
        set_hydra_crossattn(anima, embed)
        v = anima(latents, t_b, embed, padding_mask=pad)

        if cfg != 1.0:
            set_hydra_crossattn(anima, neg_embed)
            v_u = anima(latents, t_b, neg_embed, padding_mask=pad)
            v = v_u.float() + cfg * (v.float() - v_u.float())

        dt = (s_i - s_next).to(device=device, dtype=torch.float32)
        latents = (latents.float() - dt * v.float()).to(latents.dtype)

        if snapshot_sigmas is not None:
            for b_sig in snapshot_sigmas:
                if abs(float(s_next) - float(b_sig)) < 1e-9:
                    snapshots[float(b_sig)] = latents.clone()
    return latents


def _diversity(feats: torch.Tensor) -> float:
    """Eq. 9: 1 − mean pairwise cosine over R rows."""
    f = torch.nn.functional.normalize(feats.float(), dim=-1)
    cos = f @ f.t()
    r = f.shape[0]
    off = (cos.sum() - r) / (r * (r - 1))
    return float(1.0 - off)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--turbo", default="output/ckpt/anima_turbo_H_4k.safetensors")
    add_model_args(p)
    p.add_argument("--n_seeds", type=int, default=6, help="R: samples per prompt")
    p.add_argument("--student_steps", type=int, default=4)
    p.add_argument("--teacher_steps", type=int, default=28)
    p.add_argument("--teacher_cfg", type=float, default=4.0)
    p.add_argument("--student_cfg", type=float, default=1.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument(
        "--handoffs", default="1,2,3", help="student boundary indices to anchor at"
    )
    p.add_argument(
        "--negative_prompt",
        default="lowres, bad anatomy, jpeg artifacts, worst quality",
    )
    p.add_argument("--max_prompts", type=int, default=len(DEFAULT_PROMPTS))
    p.add_argument("--prompts_file", default=None, help="one prompt per line")
    p.add_argument("--seed", type=int, default=1234, help="base seed; +j per sample")
    p.add_argument("--decode_chunk", type=int, default=2)
    p.add_argument("--vae_chunk", type=int, default=256, help="VAE spatial tile size")
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile DiT blocks for the rollout (teacher/student toggle via "
        "set_multiplier keeps one graph family). Inductor's on-disk cache makes "
        "repeat runs skip the warmup; --no-compile to disable.",
    )
    p.add_argument("--label", default=None)
    opts = p.parse_args()

    torch.set_grad_enabled(False)  # whole probe is inference — no graphs anywhere
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    handoffs = [int(x) for x in opts.handoffs.split(",") if x.strip()]
    assert all(1 <= b < opts.student_steps for b in handoffs), "handoff in [1, N)"

    if opts.prompts_file:
        prompts = [
            ln.strip()
            for ln in Path(opts.prompts_file).read_text().splitlines()
            if ln.strip()
        ]
    else:
        prompts = DEFAULT_PROMPTS
    prompts = prompts[: opts.max_prompts]
    R = opts.n_seeds
    H, W = opts.size

    args = _build_args(opts)
    args.device = device
    # Fields read by build_anima (the harness encoding compile-after-apply).
    args.dtype = "bf16"
    args.attn_mode = getattr(args, "attn_mode", "flash") or "flash"
    args.gradient_checkpointing = False
    args.compile = opts.compile
    args.compile_mode = None

    # σ grids ------------------------------------------------------------------
    teacher_sigmas = get_timesteps_sigmas(opts.teacher_steps, opts.flow_shift, "cpu")[1]
    student_sigmas = get_timesteps_sigmas(opts.student_steps, opts.flow_shift, "cpu")[1]
    union = _union_sigmas(teacher_sigmas, student_sigmas)
    # Anchor boundaries: incoming σ for student step b == student_sigmas[b].
    anchor_targets = [float(student_sigmas[b]) for b in handoffs]

    # 1. Build base DiT + toggleable turbo adapter -----------------------------
    # DiT loads BEFORE encoding: prepare_text_inputs runs the DiT's llm-adapter
    # (anima._preprocess_text_embeds) to produce crossattn_emb, so it needs the
    # real model. TE (0.6B) coexists briefly during encode, then is freed.
    # build_anima installs the turbo monkey-patches THEN compiles (the
    # compile-after-apply invariant); toggle teacher/student via set_multiplier.
    print(f"[probe] loading DiT + turbo (compile={opts.compile}) …")
    bundle = build_anima(args, dit_path=opts.dit, adapter=opts.turbo, train_mode=False)
    anima, turbo_net = bundle.anima, bundle.network

    # 2. Encode every prompt once (TE up, encode through llm-adapter, free) ----
    print(f"[probe] encoding {len(prompts)} prompts …")
    shared = load_shared_models(args)
    embeds = []  # (embed_pos, embed_neg) on CPU bf16, shape (1, L, D)
    for pr in prompts:
        ctx, ctx_null = prepare_text_inputs(
            args,
            device=device,
            anima=anima,
            shared_models=shared,
            prompt=pr,
            negative_prompt=opts.negative_prompt,
            text_encoder_path=args.text_encoder,
        )
        embeds.append((ctx["embed"][0].cpu(), ctx_null["embed"][0].cpu()))
    del shared
    clean_memory_on_device(device)

    pad = torch.zeros(R, 1, H // 8, W // 8, dtype=torch.bfloat16, device=device)

    # 3. Generate latents for every (prompt × arm) -----------------------------
    latents_store = {}  # (pi, arm) -> (R,16,1,h,w) cpu
    for pi, pr in enumerate(prompts):
        e_pos = embeds[pi][0].to(device, torch.bfloat16).expand(R, -1, -1)
        e_neg = embeds[pi][1].to(device, torch.bfloat16).expand(R, -1, -1)

        gens = [
            torch.Generator(device="cpu").manual_seed(opts.seed + j) for j in range(R)
        ]
        z0 = torch.stack(
            [
                torch.randn(16, 1, H // 8, W // 8, generator=g, dtype=torch.float32)
                for g in gens
            ]
        ).to(device, torch.bfloat16)

        # teacher rollout over the union grid → ceiling + all anchors
        snaps: dict[float, torch.Tensor] = {}
        teacher_final = _integrate(
            anima,
            turbo_net,
            z0.clone(),
            union,
            0,
            len(union) - 1,
            enabled=False,
            cfg=opts.teacher_cfg,
            embed=e_pos,
            neg_embed=e_neg,
            pad=pad,
            device=device,
            snapshot_sigmas=anchor_targets,
            snapshots=snaps,
        )
        latents_store[(pi, "teacher")] = teacher_final.cpu()

        # student full rollout → floor
        student_final = _integrate(
            anima,
            turbo_net,
            z0.clone(),
            student_sigmas,
            0,
            opts.student_steps,
            enabled=True,
            cfg=opts.student_cfg,
            embed=e_pos,
            neg_embed=e_neg,
            pad=pad,
            device=device,
        )
        latents_store[(pi, "student")] = student_final.cpu()

        # anchor@b → student resume from teacher snapshot
        for b in handoffs:
            anchor = snaps[float(student_sigmas[b])]
            out_b = _integrate(
                anima,
                turbo_net,
                anchor.clone(),
                student_sigmas,
                b,
                opts.student_steps,
                enabled=True,
                cfg=opts.student_cfg,
                embed=e_pos,
                neg_embed=e_neg,
                pad=pad,
                device=device,
            )
            latents_store[(pi, f"anchor{b}")] = out_b.cpu()
        print(f"[probe]   prompt {pi + 1}/{len(prompts)} rolled out")

    # free DiT (and its compiled graphs) before bringing up the VAE ------------
    if device.type == "cuda":
        print(f"[mem] before free: {torch.cuda.memory_allocated() / 1e9:.2f} GB alloc")
    del anima, turbo_net, bundle, pad
    import torch._dynamo as _dynamo

    _dynamo.reset()  # drop compiled-graph tensors so the VAE has the whole GPU
    clean_memory_on_device(device)
    torch.cuda.empty_cache()
    if device.type == "cuda":
        print(f"[mem] after free:  {torch.cuda.memory_allocated() / 1e9:.2f} GB alloc")

    # 4. Decode + PE-Core features + diversity ---------------------------------
    print("[probe] decoding + PE-Core diversity …")
    from PIL import Image
    from torchvision.utils import make_grid, save_image

    # The DiT (incl. its compiled graphs) frees to ~0.04 GB before this point,
    # so decode runs on GPU whether or not the rollout was compiled.
    dec_dev = device
    # spatial_chunk_size tiles the heavy 3D-causal-conv decode so a full-res
    # image fits without co-residence headroom.
    vae = load_vae(
        args.vae,
        device=dec_dev,
        disable_mmap=True,
        spatial_chunk_size=opts.vae_chunk,
        dtype=torch.bfloat16,
        eval=True,
    )
    info = get_encoder_info("pe")
    pe = info.loader(dec_dev, info.default_model_id())
    proc = info.processor_factory(336)

    arms = ["teacher", "student"] + [f"anchor{b}" for b in handoffs]
    run_dir = make_run_dir("dpdmd", opts.label)
    grid_dir = run_dir / "grids"
    grid_dir.mkdir(exist_ok=True)

    div = {a: [] for a in arms}  # per-prompt diversities
    for pi, pr in enumerate(prompts):
        for a in arms:
            lat = latents_store[(pi, a)].to(dec_dev, torch.bfloat16)
            pixels = []  # [-1,1]
            for c in range(0, R, opts.decode_chunk):
                pixels.append(vae.decode_to_pixels(lat[c : c + opts.decode_chunk]))
            px = torch.cat(pixels, dim=0).float().clamp(-1, 1)
            if px.dim() == 5:  # (R,3,1,H,W) — drop the temporal axis
                px = px.squeeze(2)
            px01 = (px + 1.0) / 2.0
            pil = [
                Image.fromarray(
                    (px01[i].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                )
                for i in range(R)
            ]
            feats = pe(
                proc(pil)["pixel_values"].to(dec_dev, torch.bfloat16)
            ).pooler_output
            div[a].append(_diversity(feats))
            save_image(
                make_grid(px01, nrow=R, padding=2),
                str(grid_dir / f"p{pi:02d}_{a}.png"),
            )

    # 5. Report ----------------------------------------------------------------
    div_mean = {a: float(sum(v) / len(v)) for a, v in div.items()}
    tea, stu = div_mean["teacher"], div_mean["student"]
    gap = tea - stu
    recovery = {
        f"anchor{b}": (div_mean[f"anchor{b}"] - stu) / gap
        if gap > 1e-6
        else float("nan")
        for b in handoffs
    }
    verdict = (
        "GO"
        if (gap > 1e-3 and recovery.get("anchor1", 0) >= 0.5)
        else "NO-GO/INCONCLUSIVE"
    )

    print("\n==== DP-DMD Phase 0 — first-step anchor diversity ====")
    print(f"  teacher (ceiling) : {tea:.4f}")
    print(f"  student (floor)   : {stu:.4f}    gap = {gap:+.4f}")
    for b in handoffs:
        print(
            f"  anchor@{b}         : {div_mean[f'anchor{b}']:.4f}    "
            f"recovery = {recovery[f'anchor{b}']:+.2%}"
        )
    print(f"  VERDICT: {verdict}")
    print(f"  grids → {grid_dir}")

    write_result(
        run_dir,
        script=__file__,
        args=opts,
        device=device,
        metrics={
            "diversity_mean": div_mean,
            "diversity_per_prompt": div,
            "gap_teacher_minus_student": gap,
            "recovery": recovery,
            "verdict": verdict,
            "prompts": prompts,
            "anchor_sigmas": {str(b): float(student_sigmas[b]) for b in handoffs},
        },
        extra={
            "R": R,
            "student_steps": opts.student_steps,
            "teacher_steps": opts.teacher_steps,
        },
    )
    print(f"  result.json → {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
