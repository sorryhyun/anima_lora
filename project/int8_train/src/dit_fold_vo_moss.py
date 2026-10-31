"""SmoothQuant on MOSS-Image's ``o`` input, folded into ``v``: a new int8 file from a
``dit_gptq_moss.py`` one.

``o`` reads the attention output, whose channel ``j`` is an average of ``v``'s output
row ``j`` (RoPE touches only ``q`` and ``k``). So ``o`` columns times ``S`` and ``v``
rows over ``S`` is the same function: ``v`` keeps its ``wq`` and divides its row scales,
and ``o`` alone is rotated and GPTQ'd again. ``S`` is ``te_gptq.py --smooth``'s,
``amax|X_j|^α / amax|W_·j|^(1−α)`` with geometric mean 1, ``amax|X_j|`` from every third
sample. ``q``, ``k``, ``v``'s ``wq`` and the FFN linears keep the file's copies, without
recalibrating on the slightly different inputs a new ``o`` gives them.

Calibration is ``dit_gptq_moss.py``'s picks at its seed. Every block's linears but ``o``
run the file's copies, so ``o`` is scored and fitted on W8A8 inputs. ``--alphas`` gives
one arm per α next to ``none`` (the fold off).

``--probe N`` scores the arms on every Nth block, keeps the stream on the file's copies
and writes nothing. Otherwise ``--alphas`` is one α, every block folds, and the output
is a full file plus ``<stem>.fold.safetensors`` (``layers.N`` → ``S``). Forward only:
``wq_t`` and LoRA adapters need ``S``, which nothing reads yet, so ``grad_error_moss.py``
gives ``pred_err`` on this file and no gradient.

    make daemon-run ARGS="--stall-timeout 1800 project/int8_train/src/dit_fold_vo_moss.py \
        --probe 7 --per_set 8 --alphas 0.25,0.5,1"
"""

from __future__ import annotations

import argparse
import json
import random
import time

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from library.env import resolve_under_home  # noqa: E402
from library.moss import cache as moss_cache  # noqa: E402
from library.moss import int8  # noqa: E402
from library.moss.dit_int8.base import (  # noqa: E402
    check_base,
    convert_moss_,
    load_copies,
    moss_keys,
    moss_targets,
)
from library.moss.loader import load_transformer, variant_dir  # noqa: E402
from library.moss.model import snapped_sigmas  # noqa: E402
from library.moss.requests import DEFAULT_SHIFT, DEFAULT_STEPS  # noqa: E402
from library.moss.train import EMPTY_PROMPT_PATH  # noqa: E402

# sibling modules
from dit_gptq import Acc, _Stop, gptq, write_file  # noqa: E402
from dit_gptq_moss import (  # noqa: E402
    SETS,
    SUFFIXES,
    linear_row,
    pick_items,
    run_block,
    samples_of,
)

BASE_FILE = "models/moss_image/moss_image_base_int8_gptq.safetensors"
OUT_FILE = "models/moss_image/moss_image_base_int8_gptq_vo.safetensors"


def fold_scale(x_amax, w, alpha) -> torch.Tensor:
    wa = w.abs().amax(0)
    s = x_amax.clamp(min=1e-5).pow(alpha) / wa.clamp(min=1e-5).pow(1 - alpha)
    return s / s.log().mean().exp()


def until_o(tr, layer, index, o, samples, device, on_input, current) -> None:
    """``on_input`` on each sample's ``o`` input; the block stops there."""

    def hook(_mod, inputs):
        on_input(inputs[0])
        raise _Stop

    handle = o.register_forward_pre_hook(hook)
    for s in samples:
        current["domain"] = s["domain"]
        try:
            run_block(tr, layer, index, s, device)
        except _Stop:
            pass
    handle.remove()


class Arm:
    """One ``S``: the statistics of ``o``'s input divided by it, per domain."""

    def __init__(self, s, k, domains, device):
        self.s = s
        self.accs = {d: Acc(k, device) for d in domains}
        self.amax2, self.energy, self.tokens = 0.0, 0.0, 0

    def add(self, domain, x):
        x = x.reshape(-1, x.shape[-1]).float() / self.s
        self.accs[domain].hook(None, (x,))
        xr = int8.rotate(x)
        self.amax2 += xr.abs().amax(-1).pow(2).sum().item()
        self.energy += xr.pow(2).mean(-1).sum().item()
        self.tokens += xr.shape[0]

    def crest(self) -> float:
        """rms per-token amax of the rotated input over its rms."""
        return (self.amax2 / self.energy) ** 0.5


def fit(arm, W, block, damp) -> tuple:
    """``(wq, row scale (N, 1), report row)`` of ``o`` with ``arm``'s ``S`` folded."""
    Wr = int8.rotate(W * arm.s)
    scale = int8.row_scale(Wr)
    Hr = sum(a.Hr for a in arm.accs.values())

    def grid(c, s=scale):
        return int8.qdq(c, s)

    Q = gptq(Wr, Hr, grid, block, damp, True)
    row = linear_row(arm.accs, Wr, int8.qdq(Wr, scale), Q)
    row["crest"] = arm.crest()
    wq = (Q / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
    return wq, scale[:, None], row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--model", default="base")
    ap.add_argument("--base_file", default=BASE_FILE)
    ap.add_argument("--cache_root", default="post_image_dataset/moss")
    ap.add_argument("--sets", default=SETS)
    ap.add_argument("--per_set", type=int, default=30)
    ap.add_argument("--min_tokens", type=int, default=3000)
    ap.add_argument("--max_tokens", type=int, default=4800)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--shift", type=float, default=DEFAULT_SHIFT)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", default="0.5")
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--out", default=OUT_FILE)
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    if not args.probe and len(alphas) != 1:
        raise SystemExit("--alphas: one α unless --probe")
    torch.set_grad_enabled(False)
    t0 = time.perf_counter()
    device = torch.device("cuda")
    base_path = resolve_under_home(args.base_file)
    out = resolve_under_home(args.out)

    tr = load_transformer(args.model_dir, args.model, device=device)
    tr.ref_timestep = 0.0
    targets = dict(moss_targets(tr))
    copies, base_meta = load_copies(args.base_file)
    check_base(base_meta, targets)
    empty = moss_cache.read_te(resolve_under_home(EMPTY_PROMPT_PATH))

    rng = random.Random(args.seed)
    picked = pick_items(
        resolve_under_home(args.cache_root),
        args.sets.split(","),
        args.per_set,
        args.min_tokens,
        args.max_tokens,
        rng,
    )
    base_report = json.loads(base_path.with_suffix(".report.json").read_text())
    stems = {k: [it.stem for it in v] for k, v in picked.items()}
    if args.per_set == 30 and stems != base_report["stems"]:
        print("[fold] calibration picks differ from the base report's", flush=True)
    sigmas = snapped_sigmas(args.steps, args.shift).tolist()
    samples, n = [], 0
    for items in picked.values():
        for it in items:
            gen = torch.Generator(device=device).manual_seed(args.seed * 100003 + n)
            samples += samples_of(tr, it, empty, rng.choice(sigmas), gen, device)
            n += 1
    domains = sorted({s["domain"] for s in samples}, reverse=True)
    print(
        f"[calib] {n} images, {len(samples)} samples, {time.perf_counter() - t0:.0f}s",
        flush=True,
    )

    current = {"domain": domains[0]}
    report, folds = {}, {}
    for i, layer in enumerate(tr.layers):
        lins = {suf: targets[f"layers.{i}.proj.{suf}"] for suf in SUFFIXES}
        for suf, m in lins.items():
            if suf != "o":
                convert_moss_(m, "infer", copies[f"layers.{i}.proj.{suf}"])
        o, v = lins["o"], lins["v"]
        o_name, v_name = f"layers.{i}.proj.o", f"layers.{i}.proj.v"

        chosen = None
        if not args.probe or i % args.probe == 0:
            W = o.weight.detach().float()
            found = []

            def on_amax(x):
                found.append(x.reshape(-1, x.shape[-1]).float().abs().amax(0))

            until_o(tr, layer, i, o, samples[::3], device, on_amax, current)
            x_amax = torch.stack(found).amax(0)
            del found
            arms = {"none": Arm(torch.ones_like(x_amax), W.shape[1], domains, device)}
            for a in alphas:
                arms[f"a{a:g}"] = Arm(
                    fold_scale(x_amax, W, a), W.shape[1], domains, device
                )

            def on_input(x):
                for arm in arms.values():
                    arm.add(current["domain"], x)

            until_o(tr, layer, i, o, samples, device, on_input, current)
            fitted = {k: fit(a, W, args.block, args.damp) for k, a in arms.items()}
            report[o_name] = {k: f[2] for k, f in fitted.items()}
            for k, (_, _, row) in fitted.items():
                s = arms[k].s
                print(
                    f"[fold] block {i:2d} {k:6s} w8a8 rtn {row['w8a8_rtn']:.3e} gptq "
                    f"{row['w8a8_gptq']:.3e} t2i {row['t2i_w8a8_gptq']:.3e} i2i "
                    f"{row.get('i2i_w8a8_gptq', 0.0):.3e} w8a16 "
                    f"{row['w8a16_gptq']:.3e} crest {row['crest']:.2f} S "
                    f"{s.amin().item():.2f}..{s.amax().item():.2f} | "
                    f"{time.perf_counter() - t0:.0f}s",
                    flush=True,
                )
            if not args.probe:
                chosen = f"a{alphas[0]:g}"
                folds[f"layers.{i}"] = arms[chosen].s.cpu().contiguous()
                wq, scale, _ = fitted[chosen]
                wq_v, s_v = copies[v_name]
                s_v = s_v / arms[chosen].s.cpu()[:, None]
                copies[o_name] = (wq.cpu(), scale.cpu())
                copies[v_name] = (wq_v, s_v)
                v.s = s_v.to(device)
            del arms, fitted

        convert_moss_(o, "infer", copies[o_name])
        for s in samples:
            y = run_block(tr, layer, i, s, device)
            if y is not None:
                s["x"] = y.cpu()
        torch.cuda.empty_cache()
        print(
            f"[fold] block {i:2d} advanced, peak "
            f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB | "
            f"{time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    keys = next(iter(report.values()))
    total = {
        arm: {
            k: sum(r[arm][k] for r in report.values()) / len(report) for k in keys[arm]
        }
        for arm in keys
    }
    print("[fold] mean over o " + json.dumps(total), flush=True)
    if args.probe:
        print("[fold] --probe: nothing written", flush=True)
        return

    del tr, samples
    torch.cuda.empty_cache()
    src_dir = variant_dir(args.model_dir, args.model) / "transformer"
    srcs = sorted(src_dir.glob("*.safetensors"))
    meta = {
        k: base_meta[k]
        for k in ("source", "source_sha256", "base_fingerprint", "calibration", "gptq")
    }
    meta["method"] = "gptq_vo"
    meta["fold"] = json.dumps(
        {"alpha": alphas[0], "base_int8_id": base_meta["int8_id"]}
    )
    int8_id = write_file(srcs, out, copies, meta, moss_keys)
    save_file(folds, str(out.with_suffix(".fold.safetensors")))
    out.with_suffix(".report.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "int8_id": int8_id,
                "stems": stems,
                "mean": total,
                "linears": report,
            },
            indent=1,
        )
    )
    print(
        f"[fold] wrote {out} ({out.stat().st_size / 2**30:.2f} GiB) int8_id "
        f"{int8_id[:16]} in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
