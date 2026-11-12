#!/usr/bin/env python3
"""Handoff probe: which student step loses the teacher's layout?

The 4-step student is role-separated: step 0 is a regression on the teacher's
anchor, steps 1..3 take the DM / GAN / CDM signal. Arm 0
(``project/anima_turbo/reports/arm0.md``) reads one gap that both seeds share:
the student draws frontal standing figures where the teacher draws the jump from
below and the tumbling group. This probe hands the trajectory over at a student
step boundary, in both directions, and renders the result. Forward only.

Arms per (prompt, seed), one noise per column:

  * ``T``      teacher, ``teacher_steps`` Euler steps, cfg ``teacher_cfg``
  * ``S``      student, ``student_steps`` Euler steps, cfg 1
  * ``T>S@b``  teacher down to the student's boundary b (σ of student step b),
               then the student finishes steps b..N-1
  * ``S>T@b``  the student runs steps 0..b-1, then the teacher finishes from that σ
  * ``A>S@1``  step 0 taken with the training's anchor target itself (the CFG'd
               teacher's mean velocity ε → z_tk over ``anchor_k`` of
               ``anchor_steps`` Euler steps, ``steps.teacher_anchor``), then the
               student finishes: what the step-0 regression gives at zero loss
  * ``P>S@1``  (``--step0_lora``) that LoRA draws step 0, the student finishes

``teacher_steps`` must be a multiple of ``student_steps``, so the student grid is
a subset of the teacher grid and ``T`` is a plain teacher render.

Read on the renders: at which boundary the teacher's layout is lost. A paired
number goes beside them: the cosine between the low-passed final latent of an arm
and of ``T`` for the same noise (per-channel mean removed), and the held-out
anchor regression loss (MSE of the step-0 velocity against the anchor target,
the training's ``div_loss``) of every LoRA. Runs on the int8 base and the fp32
stream, like the judge renders.

    make daemon-run ARGS="--queue --stall-timeout 600 bench/turbo/probe_handoff.py \
        --students output/ckpt/1112_0834/1112_0834_250_ema.safetensors \
        [--step0_lora <anchor-only LoRA>]"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageDraw

from anima_lora import GenerationRequest
from bench._anima import add_model_args
from bench._common import make_run_dir, start_heartbeat, write_result
from library.anima.stream import latent_dtype
from library.env import resolve_under_home
from library.inference.models import load_dit_model, load_shared_models, swap_adapters
from library.inference.output import decode_to_pil
from library.inference.sampling import get_timesteps_sigmas
from library.inference.text import prepare_text_inputs
from library.runtime.device import clean_memory_on_device

_FLAG = re.compile(r"\s--(\w+)\s+(\S+)")


def read_prompts(path: Path) -> tuple[list[str], list[int], tuple[int, int]]:
    """A ``--from_file`` prompt list → unique prompts, seeds, (H, W)."""
    prompts: list[str] = []
    seeds: list[int] = []
    sizes = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        flags = dict(_FLAG.findall(line))
        text = line.split(" --", 1)[0].strip()
        if text not in prompts:
            prompts.append(text)
        if "d" in flags and int(flags["d"]) not in seeds:
            seeds.append(int(flags["d"]))
        sizes.add((int(flags.get("h", 1024)), int(flags.get("w", 1024))))
    if len(sizes) != 1:
        raise SystemExit(f"one image size per prompt file, got {sorted(sizes)}")
    return prompts, seeds or [1], sizes.pop()


@torch.no_grad()
def euler(anima, x, sigmas, start, stop, *, student, cfg, embed, neg, pad, keep=()):
    """Euler over ``sigmas[start:stop + 1]``. ``keep``: grid indices whose latent
    (the state at ``sigmas[k]``) is returned as a CPU snapshot."""
    anima._pgraft_network.set_enabled(student)
    snaps = {}
    for i in range(start, stop):
        t = sigmas[i].to(x.dtype).expand(x.shape[0])
        v = anima(x, t, embed, padding_mask=pad)
        if cfg != 1.0:
            v_u = anima(x, t, neg, padding_mask=pad)
            v = v_u.float() + cfg * (v.float() - v_u.float())
        x = (x.float() - (sigmas[i] - sigmas[i + 1]) * v.float()).to(x.dtype)
        if i + 1 in keep:
            snaps[i + 1] = x.cpu()
    return x, snaps


def lowpass_cos(a: torch.Tensor, b: torch.Tensor, pool: int) -> float:
    """Cosine between two final latents after a ``pool``× average pool, with the
    per-channel mean removed."""

    def lp(x):
        x = x.float().squeeze(2)
        x = torch.nn.functional.avg_pool2d(x, pool)
        return (x - x.mean(dim=(-2, -1), keepdim=True)).flatten()

    return float(torch.nn.functional.cosine_similarity(lp(a), lp(b), dim=0))


def sheet(cells: list[list[Image.Image]], cols: list[str], rows: list[str], w: int):
    """Rows × columns of images scaled to width ``w``, with a header and row tags."""
    h = round(cells[0][0].height * w / cells[0][0].width)
    head, pad = 28, 4
    out = Image.new(
        "RGB", (len(cols) * (w + pad) + pad, head + len(rows) * (h + pad)), "white"
    )
    draw = ImageDraw.Draw(out)
    for c, name in enumerate(cols):
        draw.text((pad + c * (w + pad) + 6, 8), name, fill="black")
    for r, row in enumerate(cells):
        for c, im in enumerate(row):
            xy = (pad + c * (w + pad), head + r * (h + pad))
            out.paste(im.resize((w, h), Image.LANCZOS), xy)
        draw.text((pad + 6, head + r * (h + pad) + 6), rows[r], fill="yellow")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.set_defaults(dit="models/diffusion_models/anima_merged14.safetensors")
    p.add_argument("--students", nargs="+", required=True, help="student LoRA files")
    p.add_argument("--prompts_file", default="bench/turbo/judge_hard.txt")
    p.add_argument("--student_steps", type=int, default=4)
    p.add_argument("--teacher_steps", type=int, default=28)
    p.add_argument("--teacher_cfg", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--ts_handoffs", default="1,2,3", help="T>S boundaries")
    p.add_argument("--st_handoffs", default="1,2", help="S>T boundaries")
    p.add_argument("--lowpass", type=int, default=8, help="latent average-pool factor")
    p.add_argument(
        "--anchor_k", type=int, default=6, help="dpdmd.k_anchor of the A arm"
    )
    p.add_argument(
        "--anchor_steps",
        type=int,
        default=12,
        help="dpdmd.teacher_anchor_steps of the A arm (0 turns the arm off)",
    )
    p.add_argument(
        "--step0_lora",
        default="",
        help="a LoRA that draws step 0 for every student (the P>S@1 arm)",
    )
    p.add_argument("--cell_width", type=int, default=320)
    p.add_argument("--vae_chunk_size", type=int, default=64)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label", default="handoff")
    opts = p.parse_args()

    start_heartbeat()
    device = torch.device("cuda")
    N, T = opts.student_steps, opts.teacher_steps
    if T % N:
        raise SystemExit("teacher_steps must be a multiple of student_steps")
    ts_b = [int(b) for b in opts.ts_handoffs.split(",") if b.strip()]
    st_b = [int(b) for b in opts.st_handoffs.split(",") if b.strip()]
    assert all(1 <= b < N for b in ts_b + st_b), "handoff in [1, N)"

    prompts, seeds, (H, W) = read_prompts(resolve_under_home(opts.prompts_file))
    students = [str(resolve_under_home(s)) for s in opts.students]
    names = [Path(s).stem for s in students]
    step0 = str(resolve_under_home(opts.step0_lora)) if opts.step0_lora else ""
    anchor_on = opts.anchor_steps > 0

    extra = ["--int8", "--vae_disable_cache"]
    if opts.compile:
        extra.append("--compile_blocks")
    args = GenerationRequest(
        dit=opts.dit,
        vae=opts.vae,
        text_encoder=opts.text_encoder,
        prompt=prompts[0],
        save_path="output/tests/_handoff",
        infer_steps=T,
        guidance_scale=opts.teacher_cfg,
        flow_shift=opts.flow_shift,
        image_size=(H, W),
        seed=seeds[0],
        attn_mode="flash",
        lora_weight=[step0 or students[0]],
        lora_multiplier=1.0,
        vae_chunk_size=opts.vae_chunk_size,
        extra_argv=tuple(extra),
    ).to_args()
    args.device = device

    t_sig = get_timesteps_sigmas(T, opts.flow_shift, device)[1].to(device)
    s_sig = get_timesteps_sigmas(N, opts.flow_shift, device)[1].to(device)
    t_idx = {b: b * (T // N) for b in range(N + 1)}  # student boundary → teacher index
    for b, k in t_idx.items():
        assert abs(float(t_sig[k]) - float(s_sig[b])) < 1e-6, (b, k)
    if anchor_on:
        if not 1 <= opts.anchor_k < opts.anchor_steps:
            raise SystemExit("anchor_k in [1, anchor_steps)")
        a_sig = get_timesteps_sigmas(opts.anchor_steps, opts.flow_shift, device)[1]
        a_sig = a_sig.to(device)
        t_k = float(a_sig[opts.anchor_k])

    anima = load_dit_model(args, device, torch.bfloat16)
    lat_dtype = latent_dtype(anima)

    shared = load_shared_models(args)
    embeds = []
    for pr in prompts:
        ctx, ctx_null = prepare_text_inputs(
            args, device=device, anima=anima, shared_models=shared, prompt=pr
        )
        embeds.append((ctx["embed"][0].cpu(), ctx_null["embed"][0].cpu()))
    del shared
    clean_memory_on_device(device)

    pad = torch.zeros(1, 1, H // 8, W // 8, dtype=torch.bfloat16, device=device)
    finals: dict[tuple, torch.Tensor] = {}  # (student | None, arm, pi, seed) → latent
    t_snaps: dict[tuple, dict] = {}
    a_states: dict[tuple, torch.Tensor] = {}  # A arm: the state at σ_1
    v_targets: dict[tuple, torch.Tensor] = {}
    p_states: dict[tuple, torch.Tensor] = {}  # P arm: the step-0 LoRA's state at σ_1
    anchor_mse: dict[str, list[float]] = {}

    def kwargs(pi):
        return dict(
            embed=embeds[pi][0].to(device, torch.bfloat16),
            neg=embeds[pi][1].to(device, torch.bfloat16),
            pad=pad,
        )

    def noise(seed):
        # CPU generator, as generate_body draws. inference.py's --from_file batch
        # path draws on CUDA, so these are not the judge renders' noise.
        g = torch.Generator(device="cpu").manual_seed(seed)
        return randn_tensor(
            (1, 16, 1, H // 8, W // 8), generator=g, device=device, dtype=lat_dtype
        )

    @torch.no_grad()
    def first_velocity(model, eps, embed, pad, **_):
        model._pgraft_network.set_enabled(True)
        t = s_sig[0].to(eps.dtype).expand(1)
        return model(eps, t, embed, padding_mask=pad).float()

    def mse_to_anchor(who, v, pi, seed):
        if anchor_on:
            vt = v_targets[(pi, seed)].to(device)
            anchor_mse.setdefault(who, []).append(
                float(torch.nn.functional.mse_loss(v, vt))
            )

    # Teacher pass (adapter off), plus the anchor and the step-0 LoRA's step 0.
    for pi in range(len(prompts)):
        kw = kwargs(pi)
        for seed in seeds:
            eps = noise(seed)
            x, snaps = euler(
                anima, eps, t_sig, 0, T, student=False,
                cfg=opts.teacher_cfg, keep=[t_idx[b] for b in ts_b], **kw,
            )  # fmt: skip
            finals[(None, "T", pi, seed)] = x.cpu()
            t_snaps[(pi, seed)] = snaps
            if anchor_on:
                # steps.teacher_anchor: K CFG'd Euler steps, mean velocity ε → z_tk,
                # applied over the student's step 0 (σ 1 → σ_1).
                z, _ = euler(
                    anima, eps, a_sig, 0, opts.anchor_k, student=False,
                    cfg=opts.teacher_cfg, **kw,
                )  # fmt: skip
                vt = (eps.float() - z.float()) / (1.0 - t_k)
                v_targets[(pi, seed)] = vt.cpu()
                # Reference rows: the teacher's own instantaneous velocity at σ 1
                # (cond only, and CFG'd) against the anchor's mean velocity.
                with torch.no_grad():
                    anima._pgraft_network.set_enabled(False)
                    t1 = s_sig[0].to(eps.dtype).expand(1)
                    v_c = anima(eps, t1, kw["embed"], padding_mask=pad).float()
                    v_u = anima(eps, t1, kw["neg"], padding_mask=pad).float()
                mse_to_anchor("teacher_cond", v_c, pi, seed)
                mse_to_anchor(
                    "teacher_cfg", v_u + opts.teacher_cfg * (v_c - v_u), pi, seed
                )
                a_states[(pi, seed)] = (
                    (eps.float() - (s_sig[0] - s_sig[1]) * vt).to(eps.dtype).cpu()
                )
            if step0:
                mse_to_anchor("step0_lora", first_velocity(anima, eps, **kw), pi, seed)
                x, _ = euler(anima, eps, s_sig, 0, 1, student=True, cfg=1.0, **kw)
                p_states[(pi, seed)] = x.cpu()
        print(f"[handoff] teacher: prompt {pi + 1}/{len(prompts)} done", flush=True)

    for si, name in enumerate(names):
        if si or step0:
            swap_adapters(anima, args, device, [students[si]])
        for pi in range(len(prompts)):
            kw = kwargs(pi)
            for seed in seeds:
                eps = noise(seed)
                mse_to_anchor(name, first_velocity(anima, eps, **kw), pi, seed)
                x, s_snaps = euler(
                    anima, eps, s_sig, 0, N, student=True, cfg=1.0, keep=st_b, **kw
                )
                finals[(name, "S", pi, seed)] = x.cpu()
                if anchor_on:
                    x0 = a_states[(pi, seed)].to(device)
                    x, _ = euler(anima, x0, s_sig, 1, N, student=True, cfg=1.0, **kw)
                    finals[(name, "A>S@1", pi, seed)] = x.cpu()
                if step0:
                    x0 = p_states[(pi, seed)].to(device)
                    x, _ = euler(anima, x0, s_sig, 1, N, student=True, cfg=1.0, **kw)
                    finals[(name, "P>S@1", pi, seed)] = x.cpu()
                for b in ts_b:
                    x0 = t_snaps[(pi, seed)][t_idx[b]].to(device)
                    x, _ = euler(anima, x0, s_sig, b, N, student=True, cfg=1.0, **kw)
                    finals[(name, f"T>S@{b}", pi, seed)] = x.cpu()
                for b in st_b:
                    x0 = s_snaps[b].to(device)
                    x, _ = euler(
                        anima, x0, t_sig, t_idx[b], T, student=False,
                        cfg=opts.teacher_cfg, **kw,
                    )  # fmt: skip
                    finals[(name, f"S>T@{b}", pi, seed)] = x.cpu()
            print(f"[handoff] {name}: prompt {pi + 1}/{len(prompts)} done", flush=True)

    del anima
    import torch._dynamo as _dynamo

    _dynamo.reset()
    clean_memory_on_device(device)
    torch.cuda.empty_cache()

    from library.models import qwen_vae

    vae = qwen_vae.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=opts.vae_chunk_size,
        disable_cache=True,
        vae_2d=True,
    )
    vae.to(torch.bfloat16)
    vae.eval()
    vae.to(device)

    run_dir = make_run_dir("turbo", opts.label)
    png_dir = run_dir / "png"
    png_dir.mkdir(exist_ok=True)
    sig = {b: round(float(s_sig[b]), 3) for b in range(N)}
    arms = ["T"] + [f"T>S@{b}" for b in sorted(ts_b, reverse=True)]
    cols = ["T (teacher)"]
    cols += [f"T>S@{b} (T to σ {sig[b]})" for b in sorted(ts_b, reverse=True)]
    if anchor_on:
        arms.append("A>S@1")
        cols.append(f"A>S@1 (anchor to σ {sig[1]})")
    if step0:
        arms.append("P>S@1")
        cols.append(f"P>S@1 (step0 LoRA to σ {sig[1]})")
    arms += ["S"] + [f"S>T@{b}" for b in st_b]
    cols += ["S (student)"] + [f"S>T@{b} (S to σ {sig[b]})" for b in st_b]

    pil: dict[tuple, Image.Image] = {}
    for key, lat in finals.items():
        pil[key] = decode_to_pil(vae, lat, device)
        who, arm, pi, seed = key
        tag = arm.replace(">", "to").replace("@", "_b")
        pil[key].save(png_dir / f"{who or 'teacher'}_p{pi}_s{seed}_{tag}.png")

    cos: dict[str, dict[str, list[float]]] = {}
    grids = []
    for name in names:
        cos[name] = {a: [] for a in arms if a != "T"}
        for pi in range(len(prompts)):
            cells = []
            for seed in seeds:
                row = []
                for a in arms:
                    key = (None if a == "T" else name, a, pi, seed)
                    row.append(pil[key])
                    if a != "T":
                        cos[name][a].append(
                            lowpass_cos(
                                finals[key], finals[(None, "T", pi, seed)], opts.lowpass
                            )
                        )
                cells.append(row)
            out = run_dir / f"{name}_p{pi}.jpg"
            sheet(cells, cols, [f"seed {s}" for s in seeds], opts.cell_width).save(
                out, quality=92
            )
            grids.append(str(out))

    n_s = len(seeds)
    metrics = {
        "lowpass_cos_to_teacher": {
            name: {
                a: {
                    "mean": sum(v) / len(v),
                    "per_prompt": [
                        sum(v[i * n_s : (i + 1) * n_s]) / n_s
                        for i in range(len(prompts))
                    ],
                }
                for a, v in per.items()
            }
            for name, per in cos.items()
        },
        "anchor_mse": {
            who: {
                "mean": sum(v) / len(v),
                "per_prompt": [
                    sum(v[i * n_s : (i + 1) * n_s]) / n_s for i in range(len(prompts))
                ],
            }
            for who, v in anchor_mse.items()
        },
        "anchor": (
            {"k": opts.anchor_k, "steps": opts.anchor_steps, "t_k": t_k}
            if anchor_on
            else None
        ),
        "step0_lora": step0 or None,
        "boundary_sigma": sig,
        "prompts": prompts,
        "seeds": seeds,
    }
    print("\n==== handoff: low-passed latent cosine to the teacher, same noise ====")
    for name, per in metrics["lowpass_cos_to_teacher"].items():
        print(f"  {name}")
        for a, m in per.items():
            pp = " ".join(f"{x:.3f}" for x in m["per_prompt"])
            print(f"    {a:8s} mean {m['mean']:.3f}   per prompt {pp}")
    if metrics["anchor_mse"]:
        print("==== held-out anchor MSE (training div_loss) ====")
        for who, m in metrics["anchor_mse"].items():
            pp = " ".join(f"{x:.4f}" for x in m["per_prompt"])
            print(f"  {who:30s} mean {m['mean']:.4f}   per prompt {pp}")
    write_result(
        run_dir,
        script=__file__,
        args=opts,
        device=device,
        metrics=metrics,
        artifacts=grids,
    )
    print(f"  grids → {run_dir}")


if __name__ == "__main__":
    main()
