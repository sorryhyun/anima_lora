#!/usr/bin/env python
"""delta_scale — the seed's row Δ shrunk at encode, no training (2026-10-01)

``garble_replace``'s strings repeat (`かかな応さぃなしいしいい`). The CPU
``probe`` leg asks whether glyph order survives the adapter: a 4-glyph word
in all 24 slot orders, each word token's adapter output (and the DiT
cross-attn keys ``k_norm(k_proj(·))`` of blocks ``DIT_BLOCKS``) split into an
identity / slot / residual share (balanced two-way). At Δ scale 1 the slot
share falls with the reads (seed 0.18 → warm 0.10 → cold 0.009 on かなしい);
on the seed rows the absolute slot variance peaks at Δ 0.5–0.75 and halves at
1.0, while cold's falls monotonically. ``rows`` + ``read`` render that: the
seed rows with every row's Δ × ``--scales`` on the ``sent`` ruler, paired
against the seed's routed floor (Δ 1).

Legs:
- ``probe`` (CPU): the decomposition per arm in ``PROBE_ARMS`` × ``--scales``;
  ``--abs`` prints absolute variances instead of shares;
- ``rows`` (CPU) → ``OUT/experiments/delta_scale_seed_s<NNN>/trained.pt``: the
  seed's ``trained.pt`` with ``raw`` × s (``seed_merged`` = the seed);
- ``read`` (GPU): ``garble_replace``'s read leg on each scaled arm.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="--label delta_scale \\
      project/cjk_anima_scale/experiments/delta_scale/run_exp.py \\
      --label s075 --legs rows read --scales 0.75 0.5"
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import types
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # the seed trained routed, reads routed
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap, load_experiment  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

EXP = OUT / "experiments"
DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
QWEN = "models/text_encoders/qwen_3_06b_base.safetensors"
PROBE_ARMS = {
    "seed": SEED_ROWS,
    "warm": EXP / "garble_replace_warm" / "trained.pt",
    "warm_short50": EXP / "garble_replace_warm_short50" / "trained.pt",
    "cold": EXP / "garble_replace_cold" / "trained.pt",
    "cold_grid50": EXP / "garble_replace_cold_grid50" / "trained.pt",
}
PROBE_PROMPT = (
    "1girl, solo, blonde hair, school uniform, classroom, sitting at desk, "
    "smile, looking at viewer"
)
PROBE_WORDS = ("かなしい", "パソコン", "山田太郎")  # 4 distinct singles of the 57
DIT_BLOCKS = (0, 7, 14, 21, 27)
DIT_HEAD_DIM = 128


def arm_name(s: float) -> str:
    return f"delta_scale_seed_s{round(s * 100):03d}"


def rows(s: float) -> Path:
    import torch

    sd = torch.load(SEED_ROWS, map_location="cpu", weights_only=False)
    sd["delta"] = dict(sd["delta"], raw=sd["delta"]["raw"] * s)
    sd["seed_merged"] = str(SEED_ROWS)
    sd["delta_scale"] = s
    out = EXP / arm_name(s) / "trained.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out)
    print(f"rows Δ × {s} → {out}", flush=True)
    return out


def probe(scales: list, absolute: bool) -> dict:
    """Identity / slot / residual of each word token, per arm × Δ scale."""
    import torch

    from common.hooks import ExtDelta
    from library.anima import text_strategies as TS
    from library.anima.vocab_pack import default_vocab_pack, load_vocab_pack
    from library.anima.weights import load_llm_adapter, load_qwen3_text_encoder
    from library.inference.text import ensure_text_strategies
    from safetensors import safe_open

    torch.set_grad_enabled(False)
    pack = load_vocab_pack(default_vocab_pack())
    ensure_text_strategies(QWEN, vocab_pack=pack)
    tok = TS.TokenizeStrategy.get_strategy()
    enc = TS.TextEncodingStrategy.get_strategy()
    qwen = load_qwen3_text_encoder(QWEN, dtype=torch.float32, device="cpu")
    qwen = qwen[0] if isinstance(qwen, tuple) else qwen
    adapter = load_llm_adapter(DIT, dtype=torch.float32, device="cpu", vocab_pack=pack)
    holder = types.SimpleNamespace(llm_adapter=adapter)
    f = safe_open(DIT, "pt")
    dk = {
        b: (
            f.get_tensor(f"net.blocks.{b}.cross_attn.k_proj.weight").float(),
            f.get_tensor(f"net.blocks.{b}.cross_attn.k_norm.weight").float(),
        )
        for b in DIT_BLOCKS
    }
    qcache: dict = {}

    def encode(word):
        text = f'{PROBE_PROMPT}, japanese text. Japanese text reads as "{word}".'
        tokens = tok.tokenize(text)
        if text not in qcache:
            qcache[text] = enc.encode_tokens(tok, [qwen], [t.clone() for t in tokens])
        e = qcache[text]
        out = adapter(
            e[0].float(),
            e[2],
            target_attention_mask=e[3].bool(),
            source_attention_mask=e[1].bool(),
        )[0]
        return out, e[2][0]

    def keys(ctx, b):
        W, nw = dk[b]
        k = (ctx @ W.T).view(ctx.shape[0], -1, DIT_HEAD_DIM)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6) * nw
        return k.reshape(ctx.shape[0], -1)

    def split(V, gid, sid):
        V = V - V.mean(0)
        G = torch.stack([V[gid == g].mean(0) for g in gid.unique()])[gid]
        S = torch.stack([V[sid == s].mean(0) for s in sid.unique()])[sid]
        parts = [(G**2).sum(), (S**2).sum(), ((V - G - S) ** 2).sum()]
        d = V.shape[0] if absolute else (V**2).sum()
        return [round(float(p / d), 4) for p in parts]

    out: dict = {}
    for arm, path in PROBE_ARMS.items():
        sd = torch.load(path, map_location="cpu", weights_only=False)["delta"]
        delta = ExtDelta.from_state(holder, sd, "cpu")
        delta.raw.data = delta.raw.data.float()
        for sc in scales:
            delta.scale = sc
            for word in PROBE_WORDS:
                g = list(word)
                pos = (encode(word)[1] != encode("".join(g[1:] + g[:1]))[1]).nonzero()
                pos = pos.flatten()
                assert len(pos) == 4, (word, pos)
                V, K, gid, sid = [], [], [], []
                for perm in itertools.permutations(range(4)):
                    o = encode("".join(g[i] for i in perm))[0][pos]
                    V.append(o)
                    K.append(torch.stack([keys(o, b) for b in DIT_BLOCKS]))
                    gid += list(perm)
                    sid += range(4)
                gid, sid = torch.tensor(gid), torch.tensor(sid)
                K = torch.cat(K, 1)
                ks = [split(K[j], gid, sid) for j in range(len(DIT_BLOCKS))]
                rec = {
                    "adapter": split(torch.cat(V), gid, sid),
                    "dit_keys": [
                        round(sum(x[i] for x in ks) / len(ks), 4) for i in range(3)
                    ],
                }
                out[f"{arm}|{sc:g}|{word}"] = rec
                print(
                    f"{arm:13s} Δ {sc:<5g} {word}  adapter id/slot/resid "
                    f"{rec['adapter']}  DiT keys {rec['dit_keys']}",
                    flush=True,
                )
        for h in delta.handles:
            h.remove()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs", nargs="+", default=["rows"], choices=["probe", "rows", "read"]
    )
    p.add_argument("--scales", nargs="+", type=float, default=[0.75])
    p.add_argument("--abs", action="store_true", help="probe: absolute variances")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    print(f"legs {args.legs} · scales {args.scales}", flush=True)
    for s in args.scales:
        print(f"  arm {EXP / arm_name(s)}", flush=True)
    if args.dry_run:
        return
    metrics: dict = {}
    if "probe" in args.legs:
        metrics["probe"] = probe(args.scales, args.abs)
    if "rows" in args.legs:
        for s in args.scales:
            rows(s)
    if "read" in args.legs:
        G = load_experiment("garble_replace")
        for s in args.scales:
            metrics[f"read_s{s:g}"] = G.read(f"s{s:g}", arm_name(s))
    run_dir = make_run_dir(
        "delta_scale",
        label=args.label,
        root=LINE / "experiments" / "delta_scale" / "results",
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(EXP / arm_name(s)) for s in args.scales],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
