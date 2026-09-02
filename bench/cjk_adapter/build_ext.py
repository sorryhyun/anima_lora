#!/usr/bin/env python3
"""Build the CJK extended-vocab sidecar for the LLM Adapter (Phase 1, no training).

Reads two tensors (no model instantiation on the default path):
  - ``net.llm_adapter.embed.weight``  [32128, 1024] from the DiT checkpoint
  - ``model.embed_tokens.weight``     [151936, 1024] from the Qwen3 encoder

Fits the anchor map W on surface-identical tokens, then writes
``<out>.safetensors`` (rows, fp32) + ``<out>.json`` (id mapping + build
stats). The id mapping is tokenizer-deterministic — every recipe below
yields the identical mapping, only the row *values* differ, so distill
caches (keyed on ids) and trained packs (which store materialized rows)
are unaffected by a rebuild.

Recipes (plan_zh.md Z0, probes 2026-09-02):

  v1 (shipped through synthja_v5 / synthjako3):
      --map ridge --char-init fragment-mean
  v2 (the default now):
      --map procrustes-mix --char-init contextual
    ridge + 0.6·Procrustes keeps the ext keys spread (PR 236 → 373,
    collisions 16 % → 1 %, held-out cos 0.75 → 0.70); the char-fallback
    rows are initialised from the *contextual* Qwen hidden state of the
    character (read in a fixed tag-context prefix) instead of the mean of
    shared byte-fragment rows (which put 99.9 % of JA tag-kanji row pairs
    above cos 0.5 — ``probes/z0_probe.py``). The contextual map is fitted
    on the 11.6k CJK chars that are clean single Qwen tokens — same script
    and register as the rows it serves — from their per-dim standardized
    contextual state to their own token-row key (ridge 0.1 + 0.6·Procrustes,
    held-out cos 0.43), then applied to the byte-fragment chars and the char
    layer is row-normalized to the T5 mean norm (Qwen's massive-activation
    dims otherwise leave a few rows 5× the rest). EN anchors are the wrong
    fit set here (held-out 0.46, PR 6 — the states share Qwen's anisotropic
    common direction; measured 2026-09-02). Result: top-200 JA tag kanji
    char rows at cos 0.16 / 1.9 % > 0.5 vs the token-row bar 0.19 / 0 %;
    the rare tail (ext-A, unseen ideographs, most hangul syllables) stays
    clustered — Qwen reads those as one "unknown" state, and no map splits
    identical inputs. The contextual path loads the Qwen3 encoder — run it
    on the GPU through the daemon:

    make daemon-run ARGS="bench/cjk_adapter/build_ext.py"

  CPU-only v1 rebuild:  uv run python bench/cjk_adapter/build_ext.py \\
      --map ridge --char-init fragment-mean --out bench/cjk_adapter/assets/ext_embed_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from anima_lora import default_checkpoints  # noqa: E402
from bench.cjk_adapter import ext_vocab  # noqa: E402
from library.anima.weights import (  # noqa: E402
    load_qwen3_text_encoder,
    load_qwen3_tokenizer,
    load_t5_tokenizer,
)
from library.env import resolve_under_home  # noqa: E402

ASSET_PREFIX = Path(__file__).resolve().parent / "assets" / "ext_embed"

# Tag-register context the contextual init reads each surface in. Position 0
# of a Qwen sequence is the attention-sink slot (huge-norm, content-free), so
# a bare char would pool its own lead byte at the sink; a short tag prefix
# puts every pooled position past it and matches the register the rows are
# trained in. Anchor surfaces carry their own leading space (``▁cat`` → " cat"),
# chars do not — hence the two prefix forms.
CTX_PREFIX_SPACED = "1girl, "
CTX_PREFIX_BARE = "1girl,"


def read_tensor(path: str, key: str) -> torch.Tensor:
    with safe_open(str(resolve_under_home(path)), framework="pt") as f:
        return f.get_tensor(key)


@torch.no_grad()
def contextual_vectors(
    text_encoder,
    qwen_tok,
    surfaces: list[str],
    *,
    device: torch.device,
    batch: int = 256,
    log_every: int = 40,
) -> torch.Tensor:
    """Mean last-layer Qwen hidden state over each surface's own tokens.

    Each surface is encoded as ``prefix + surface``; the pooled positions are
    the tokens whose char offsets start inside the surface (the prefix and
    any space that fused into a prefix token are excluded).
    """
    out = torch.empty(len(surfaces), text_encoder.config.hidden_size)
    for b0 in range(0, len(surfaces), batch):
        chunk = surfaces[b0 : b0 + batch]
        texts, starts = [], []
        for s in chunk:
            prefix = CTX_PREFIX_BARE if s.startswith(" ") else CTX_PREFIX_SPACED
            texts.append(prefix + s)
            starts.append(len(prefix))
        enc = qwen_tok(
            texts,
            add_special_tokens=False,
            return_offsets_mapping=True,
            padding=True,
            padding_side="right",
            return_tensors="pt",
        )
        offs = enc.pop("offset_mapping")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        hidden = text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        hidden = hidden.float().cpu()
        for j, s0 in enumerate(starts):
            sel = [
                t
                for t in range(offs.shape[1])
                if int(enc["attention_mask"][j, t]) and int(offs[j, t, 0]) >= s0
            ]
            if not sel:  # surface fused entirely into a prefix token — take the last
                sel = [int(enc["attention_mask"][j].sum()) - 1]
            out[b0 + j] = hidden[j, sel].mean(0)
        if log_every and (b0 // batch) % log_every == 0:
            print(f"  contextual {b0 + len(chunk)}/{len(surfaces)}", flush=True)
    return out


def contextual_char_init(
    S_ctx: torch.Tensor,
    single_keys: torch.Tensor,
    C_ctx: torch.Tensor,
    chars: list[str],
    *,
    ridge: float,
    mix: float,
    holdout: int = 1500,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Map the byte-fragment chars' contextual states into key space.

    Fit set = the clean single-token CJK chars: ``S_ctx`` (their contextual
    states) → ``single_keys`` (their token-row keys under the embedding map).
    States are standardized per dim on the union of both char sets first —
    Qwen hidden states carry a few massive dims and a large shared direction;
    mapped raw, every char row inherits it and the layer collapses (PR 10,
    57 % random-pair collisions on the first v2 attempt). Rows are returned
    unit-norm; the caller rescales the layer to the T5 mean norm.
    """
    allc = torch.cat([S_ctx, C_ctx])
    mu, sd = allc.mean(0), allc.std(0).clamp(min=1e-6)
    Xs, Xc = (S_ctx - mu) / sd, (C_ctx - mu) / sd
    W_ctx, ctx_holdout = ext_vocab.fit_anchor_map(
        Xs,
        single_keys,
        list(range(len(Xs))),
        list(range(len(Xs))),
        ridge=ridge,
        holdout=holdout,
        method="procrustes-mix",
        mix=mix,
    )
    mapped = torch.nn.functional.normalize(Xc @ W_ctx, dim=-1)
    char_init = {ch: mapped[i] for i, ch in enumerate(chars)}
    stats = {
        "ctx_holdout_cos": ctx_holdout,
        "ctx_prefix": [CTX_PREFIX_SPACED, CTX_PREFIX_BARE],
        "ctx_fit_set": "single-token CJK chars",
        "ctx_fit_n": int(len(Xs)),
        "ctx_ridge": ridge,
        "ctx_mix": mix,
        "n_char_contextual": len(chars),
    }
    return char_init, stats


def main() -> None:
    ckpt = default_checkpoints()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dit", default=ckpt.dit)
    parser.add_argument("--text_encoder", default=ckpt.text_encoder)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument(
        "--map",
        default="procrustes-mix",
        choices=ext_vocab.MAP_METHODS,
        help="anchor map fit; v1 = ridge, v2 = procrustes-mix",
    )
    parser.add_argument("--mix", type=float, default=0.6, help="Procrustes weight")
    parser.add_argument(
        "--char-init",
        default="contextual",
        choices=("fragment-mean", "contextual"),
        help="char-fallback row init; v1 = fragment-mean, v2 = contextual (loads Qwen)",
    )
    parser.add_argument("--ctx-batch", type=int, default=256)
    parser.add_argument(
        "--ctx-cache",
        type=Path,
        default=None,
        help="save/load the raw contextual states (anchors + chars) so the map "
        "can be iterated on CPU without re-encoding",
    )
    parser.add_argument(
        "--ctx-ridge", type=float, default=0.1, help="contextual map ridge"
    )
    parser.add_argument(
        "--ctx-mix", type=float, default=0.6, help="contextual map Procrustes weight"
    )
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--out", type=Path, default=ASSET_PREFIX, help="path prefix")
    opts = parser.parse_args()

    t5_embed = read_tensor(opts.dit, "net.llm_adapter.embed.weight")
    qwen_embed = read_tensor(opts.text_encoder, "model.embed_tokens.weight")
    print(f"t5 table {tuple(t5_embed.shape)}, qwen table {tuple(qwen_embed.shape)}")

    qwen_tok = load_qwen3_tokenizer(str(resolve_under_home(opts.text_encoder)))
    t5_tok = load_t5_tokenizer()

    print("collecting clean CJK tokens + anchors …")
    clean = ext_vocab.collect_clean_qwen_tokens(qwen_tok)
    t5_ids, qwen_ids = ext_vocab.build_anchor_pairs(t5_tok, qwen_tok)
    print(f"clean CJK tokens: {len(clean)}, anchors: {len(t5_ids)}")

    W, holdout_cos = ext_vocab.fit_anchor_map(
        qwen_embed,
        t5_embed,
        t5_ids,
        qwen_ids,
        ridge=opts.ridge,
        method=opts.map,
        mix=opts.mix,
    )
    print(f"anchor map ({opts.map}): held-out cosine {holdout_cos:.4f}")

    char_init: dict[str, torch.Tensor] | None = None
    ctx_stats: dict = {}
    if opts.char_init == "contextual":
        device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if opts.device == "auto"
            else opts.device
        )
        chars = list(ext_vocab.char_row_surfaces(qwen_tok, clean))
        single = [
            (qid, surf)
            for qid, surf in sorted(clean.items())
            if len(surf) == 1 and ext_vocab.is_cjk_char(surf)
        ]
        single_qids = [q for q, _ in single]
        cached = None
        if opts.ctx_cache and opts.ctx_cache.exists():
            cached = torch.load(opts.ctx_cache)
            if cached.get("chars") != chars or cached.get("single_qids") != single_qids:
                print(
                    f"  ctx cache {opts.ctx_cache} is for a different char set — re-encoding"
                )
                cached = None
        if cached is None:
            print(f"contextual char init on {device}: loading Qwen3 encoder …")
            te, _ = load_qwen3_text_encoder(
                str(resolve_under_home(opts.text_encoder)),
                dtype=torch.float32 if device.type == "cpu" else torch.bfloat16,
                device="cpu",
            )
            te.to(device).eval()
            print(f"  encoding {len(single)} single-token CJK chars (fit set) …")
            S_ctx = contextual_vectors(
                te,
                qwen_tok,
                [surf for _, surf in single],
                device=device,
                batch=opts.ctx_batch,
            )
            print(f"  encoding {len(chars)} byte-fragment chars …")
            C_ctx = contextual_vectors(
                te, qwen_tok, chars, device=device, batch=opts.ctx_batch
            )
            del te
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if opts.ctx_cache:
                opts.ctx_cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "chars": chars,
                        "single_qids": single_qids,
                        "S_ctx": S_ctx,
                        "C_ctx": C_ctx,
                    },
                    opts.ctx_cache,
                )
                print(f"  saved contextual states to {opts.ctx_cache}")
        else:
            S_ctx, C_ctx = cached["S_ctx"], cached["C_ctx"]
            print(f"  loaded contextual states from {opts.ctx_cache}")
        single_keys = qwen_embed[single_qids].float() @ W
        char_init, ctx_stats = contextual_char_init(
            S_ctx, single_keys, C_ctx, chars, ridge=opts.ctx_ridge, mix=opts.ctx_mix
        )
        print(
            f"  contextual map (single-char fit): held-out cos→key {ctx_stats['ctx_holdout_cos']:.4f}"
        )

    print("building extended table (incl. per-char fallback rows) …")
    table, mapping = ext_vocab.build_ext_table(
        qwen_tok, qwen_embed, W, clean, char_init=char_init
    )
    n_qwen, n_char = len(mapping["qwen"]), len(mapping["char"])
    print(f"ext rows: {table.shape[0]} ({n_qwen} qwen tokens + {n_char} chars)")

    # Least-squares maps shrink toward the mean — rescale so new rows live at
    # the T5 table's typical row norm (the adapter downstream is scale-aware).
    # Per layer: the qwen-token rows and the char rows come through different
    # maps under the contextual recipe, so each layer is brought to the T5
    # mean norm on its own (identical to the old global scale when both layers
    # share one map up to their own mean).
    t5_norm = float(t5_embed.float().norm(dim=-1).mean())
    scales = {}
    for name, sl in (("qwen", slice(0, n_qwen)), ("char", slice(n_qwen, None))):
        layer_norm = float(table[sl].norm(dim=-1).mean())
        scales[name] = t5_norm / layer_norm
        table[sl] = table[sl] * scales[name]
        print(
            f"row-norm {name}: pre-scale {layer_norm:.3f} → t5 {t5_norm:.3f} (×{scales[name]:.3f})"
        )

    opts.out.parent.mkdir(parents=True, exist_ok=True)
    save_file({"ext_embed": table.contiguous()}, str(opts.out) + ".safetensors")
    mapping["stats"] = {
        "anchors": len(t5_ids),
        "holdout_cos": holdout_cos,
        "ridge": opts.ridge,
        "map": opts.map,
        "mix": opts.mix if opts.map == "procrustes-mix" else None,
        "char_init": opts.char_init,
        "t5_row_norm": t5_norm,
        "norm_scale": scales,
        **ctx_stats,
    }
    Path(str(opts.out) + ".json").write_text(
        json.dumps(mapping, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {opts.out}.safetensors / .json")


if __name__ == "__main__":
    main()
