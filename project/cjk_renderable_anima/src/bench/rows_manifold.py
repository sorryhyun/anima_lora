#!/usr/bin/env python
"""rows_manifold — does a trained ext-row table have structure, and does
it share a representation with the pretrained quoted-text frames?

Two stages, both read-only on a finished ``rows`` arm (plan_synth branch C):

``--stage rows`` (CPU, seconds): the table in **row space** (row-norm units).
  spectrum / participation ratio vs a gaussian control, the shared mean
  component, within- vs between-family cos (hiragana / katakana / voiced /
  small / kanji / word), voiced-pair and hira↔kata-pair cos vs random pairs,
  a Mantel test of Δ-cos against glyph pixel-cos, cos to the row's own pack
  row, energy inside the stock T5 table's top PCs, and per-row hit (from
  ``eval_reads.json``) against norm / exposure / crowding.

``--stage adapter`` (Qwen TE + llm_adapter forward; ``--device cuda`` on the
  daemon or CPU): the table's **image at the adapter output** under the
  training frames (``reads_as`` / ``bubble_reads`` / ``saying`` / ``sign`` /
  ``bare`` / ``plain``) — cos of each row's image ``d = out(on) − out(off)`` to
  the EN quote direction Q of the same frame, the shared component of the
  images and where it points, the frame shift a trained row receives vs the
  shift an EN word receives, energy in the EN-quoted-code subspace, and
  frame-to-frame consistency of ``d`` (context-free or not).

    .venv/bin/python project/cjk_renderable_anima/src/bench/rows_manifold.py \
        --arm_dir output/wake_probe/rows_synth_full_fm10k_full_s53k_qoff --stage rows
    make daemon-run ARGS="--stall-timeout 0 project/cjk_renderable_anima/src/bench/rows_manifold.py \
        --arm_dir output/wake_probe/rows_synth_full_fm10k_full_s53k_qoff --stage adapter --device cuda"

Writes ``<arm_dir>/manifold/{rows,adapter}.json`` and prints the tables.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

F = torch.nn.functional

EN = [
    "HELLO", "STOP", "YES", "WAIT", "SORRY", "WHAT", "RUN", "HELP", "GO", "HEY", "NO", "OK",
    "LOVE", "FIRE", "COLD", "HOME", "NIGHT", "DREAM", "MOON", "STAR", "BOOK", "TEA", "CAT", "DOG",
]  # fmt: skip

# The training frames (scenes/stage.py FRAMES), JA-swapped as the data stage
# writes them; ``plain`` is the unframed control. ``{lang}`` = japanese / english.
FRAMES = {
    "reads_as": 'manga, speech bubble, {lang} text. {Lang} text reads as "{t}".',
    "bubble_reads": 'manga, speech bubble, {lang} text. There is a speech bubble that reads "{t}".',
    "saying": 'manga, 1girl, speech bubble, {lang} text. She is saying "{t}".',
    "sign": 'manga, 1girl, holding sign, sign, {lang} text. She is holding a sign that reads "{t}".',
    "bare": 'manga, speech bubble, {lang} text. "{t}".',
    "plain": "manga, speech bubble, {lang} text, {t}.",
}

# ---------------------------------------------------------------- shared


def load_arm(arm_dir: Path):
    sd = torch.load(arm_dir / "trained.pt", map_location="cpu", weights_only=False)
    raw = sd["delta"]["raw"].float()
    ext_ids = [int(i) for i in sd["delta"]["ext_ids"]]
    return sd, raw, ext_ids


def load_text_side():
    from library.anima.vocab_pack import load_vocab_pack
    from library.env import default_checkpoints
    from library.inference.text import ensure_text_strategies

    ck = default_checkpoints()
    pack = load_vocab_pack(ck.vocab_pack)
    tok, enc = ensure_text_strategies(ck.text_encoder, vocab_pack=ck.vocab_pack)
    return ck, pack, tok, enc


def classify(text: str) -> str:
    from common.text import (
        HIRA,
        KANA_EXT_HIRA,
        KANA_EXT_KATA,
        KANA_SMALL,
        KANJI_RE,
        KATA,
    )

    if len(text) != 1:
        return "word"
    if text in HIRA:
        return "hira"
    if text in KATA:
        return "kata"
    if text in KANA_SMALL:
        return "small"
    if text in KANA_EXT_HIRA:
        return "hira_v"
    if text in KANA_EXT_KATA:
        return "kata_v"
    if KANJI_RE.match(text):
        return "kanji"
    return "other"


def per_row_hits(arm_dir: Path) -> dict[str, list[bool]]:
    """text → exact flags over seeds, singles groups only (largest box read)."""
    p = arm_dir / "eval_reads.json"
    if not p.exists():
        return {}
    hits: dict[str, list[bool]] = defaultdict(list)
    for r in json.load(open(p)):
        if r["cond"] == "trained" and r["group"] in (
            "single",
            "single_ext",
            "single_kanji",
        ):
            hits[r["text"]].append(bool(r["exact"]))
    return dict(hits)


def exposure(arm_dir: Path, tok, ext_ids: set[int]) -> tuple[Counter, Counter]:
    """rows → training items that tokenize to the row; also the frame mix."""
    from library.anima.ext_vocab import T5_TABLE_SIZE

    data_dir = (
        arm_dir.parent / f"data_{arm_dir.name.split('_', 1)[1].rsplit('_', 2)[0]}"
    )
    # arm dir = rows_<data_tag>_<arm_tag>; arm_tag here has two '_' parts (full_s53k_qoff → 3)
    cands = [
        d
        for d in arm_dir.parent.glob("data_*")
        if arm_dir.name[5:].startswith(d.name[5:] + "_")
    ]
    if cands:
        data_dir = max(cands, key=lambda d: len(d.name))
    items = data_dir / "train.jsonl"
    rows_n: Counter = Counter()
    frames: Counter = Counter()
    if not items.exists():
        print(f"(no {items}; exposure skipped)")
        return rows_n, frames
    texts = Counter()
    for line in open(items):
        it = json.loads(line)
        texts[it["text"]] += 1
        frames[it["caption"].split(". ")[-1].split('"')[0].strip() or "bare"] += 1
    for t, n in texts.items():
        ids = tok.tokenize(
            FRAMES["reads_as"].format(lang="japanese", Lang="Japanese", t=t)
        )[2]
        for v in set(ids.flatten().tolist()):
            if v >= T5_TABLE_SIZE and (v - T5_TABLE_SIZE) in ext_ids:
                rows_n[v - T5_TABLE_SIZE] += n
    return rows_n, frames


def offdiag(M: torch.Tensor) -> torch.Tensor:
    n = M.shape[0]
    return M[~torch.eye(n, dtype=torch.bool)]


def cosmat(X: torch.Tensor) -> torch.Tensor:
    Xn = F.normalize(X, dim=1)
    return Xn @ Xn.T


def spectrum(X: torch.Tensor) -> dict:
    Xc = X - X.mean(0)
    s = torch.linalg.svdvals(Xc)
    e = s**2 / (s**2).sum()
    pr = float((s**2).sum() ** 2 / (s**4).sum())
    return {"pr": pr, **{f"top{k}": float(e[:k].sum()) for k in (1, 2, 4, 16, 64)}}


def fmt(x):
    return f"{x:+.3f}" if isinstance(x, float) else str(x)


# ---------------------------------------------------------------- rows stage


def stage_rows(a):
    from train.encoder import row_texts

    sd, raw, ext_ids = load_arm(a.arm_dir)
    ck, pack, tok, enc = load_text_side()
    texts = row_texts(tok, pack, ext_ids)
    idx = {e: i for i, e in enumerate(ext_ids)}
    rows = [e for e in ext_ids if e in texts]
    cls = {e: classify(texts[e]) for e in rows}
    fam = defaultdict(list)
    for e in rows:
        fam[cls[e]].append(e)
    D = raw[[idx[e] for e in rows]]
    n, dim = D.shape
    out: dict = {
        "n_rows": n,
        "dim": dim,
        "families": {k: len(v) for k, v in fam.items()},
    }
    print(f"table {n} named rows × {dim}; families {out['families']}")

    # --- norms, mean component
    norms = D.norm(dim=1)
    mean = D.mean(0)
    out["norm"] = {
        "mean": float(norms.mean()),
        "median": float(norms.median()),
        "max": float(norms.max()),
    }
    out["mean_component"] = {
        "norm": float(mean.norm()),
        "energy_share": float(mean.norm() ** 2 / (norms**2).mean()),
        "cos_rows_to_mean": float(F.cosine_similarity(D, mean[None]).mean()),
    }
    print(
        f"row norm mean {norms.mean():.3f} (row-norm units); shared mean: |m| {mean.norm():.3f}, "
        f"energy share {out['mean_component']['energy_share']:.3f}, cos(row, m) {out['mean_component']['cos_rows_to_mean']:.3f}"
    )

    # --- spectrum vs gaussian
    g = torch.randn_like(D) * norms.mean() / math.sqrt(dim)
    out["spectrum"] = {"table": spectrum(D), "gaussian": spectrum(g)}
    Dm = D - mean
    out["spectrum"]["table_mean_removed_pairwise_cos"] = float(
        offdiag(cosmat(Dm)).mean()
    )
    out["spectrum"]["raw_pairwise_cos"] = float(offdiag(cosmat(D)).mean())
    print(
        "spectrum (centred):",
        {k: round(v, 3) for k, v in out["spectrum"]["table"].items()},
    )
    print(
        "gaussian control  :",
        {k: round(v, 3) for k, v in out["spectrum"]["gaussian"].items()},
    )
    print(
        f"pairwise cos raw {out['spectrum']['raw_pairwise_cos']:.3f}, mean-removed {out['spectrum']['table_mean_removed_pairwise_cos']:.3f}"
    )

    # Shared direction m̂ and the residual R = Δ − (Δ·m̂) m̂. (Plain mean
    # subtraction turns every weak row into −m, so untrained rows look like
    # one tight cluster; projecting the direction out keeps them small.)
    mhat = mean / mean.norm()
    coef = D @ mhat
    R = D - coef[:, None] * mhat[None]
    U1, S1, V1 = torch.linalg.svd(Dm, full_matrices=False)
    pcs = Dm @ V1[:4].T
    out["shared_direction"] = {
        "cos_mhat_pc1": abs(float(F.cosine_similarity(mhat, V1[0], dim=0)))
    }
    # is m̂ the c_flat / trigger vector earlier arms learned as a separate parameter?
    for other in (
        "rows_synth_s0_s24k_S0",
        "rows_synth_s0b_s24k_S0b",
        "rows_synth_micro6_m6_s2k_cap075",
    ):
        p_ = a.arm_dir.parent / other / "trained.pt"
        if p_.exists():
            osd = torch.load(p_, map_location="cpu", weights_only=False)
            if "c_flat" in osd:
                c = osd["c_flat"].float()
                out["shared_direction"][f"cos_mhat_c_flat[{other}]"] = float(
                    F.cosine_similarity(mhat, c, dim=0)
                )
                om = osd["delta"]["raw"].float().mean(0)
                out["shared_direction"][f"cos_mhat_rowmean[{other}]"] = float(
                    F.cosine_similarity(mhat, om, dim=0)
                )
    print(
        "shared direction m̂:",
        {k: round(v, 3) for k, v in out["shared_direction"].items()},
    )

    # --- within / between family (residual R, shared direction projected out)
    C = cosmat(R)
    fams = sorted(fam, key=lambda k: -len(fam[k]))
    pos = {e: i for i, e in enumerate(rows)}
    tab = {}
    print(
        "\n| family | n | norm | along m̂ | resid norm | resid within cos | vs others | PC1 / PC2 / PC3 score | hit rate (n) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    hits = per_row_hits(a.arm_dir)
    for f_ in fams:
        ii = torch.tensor([pos[e] for e in fam[f_]])
        oo = torch.tensor([pos[e] for e in rows if cls[e] != f_])
        w = offdiag(C[ii][:, ii]).mean().item() if len(ii) > 1 else float("nan")
        b = C[ii][:, oo].mean().item() if len(oo) else float("nan")
        h = [x for e in fam[f_] for x in hits.get(texts[e], [])]
        tab[f_] = {
            "n": len(ii),
            "norm": float(norms[ii].mean()),
            "along_mhat": float(coef[ii].mean()),
            "resid_norm": float(R[ii].norm(dim=1).mean()),
            "within_cos": w,
            "between_cos": b,
            "pc_scores": [float(pcs[ii, k].mean()) for k in range(3)],
            "hit_rate": (sum(h) / len(h)) if h else None,
            "hit_n": len(h),
        }
        hr = f"{tab[f_]['hit_rate']:.2f} ({len(h)})" if h else "—"
        pc = " / ".join(f"{v:+.2f}" for v in tab[f_]["pc_scores"])
        print(
            f"| {f_} | {len(ii)} | {norms[ii].mean():.3f} | {coef[ii].mean():+.3f} | {R[ii].norm(dim=1).mean():.3f} | {w:+.3f} | {b:+.3f} | {pc} | {hr} |"
        )
    out["family"] = tab
    Dm = R  # every structure test below runs on the residual
    # family separability: nearest-centroid leave-one-out accuracy
    cents = {f_: Dm[[pos[e] for e in fam[f_]]] for f_ in fams}
    correct = 0
    for e in rows:
        f0 = cls[e]
        best, bf = -2, None
        for f_ in fams:
            M = cents[f_]
            if f_ == f0:
                if len(M) < 2:
                    continue
                c = (M.sum(0) - Dm[pos[e]]) / (len(M) - 1)
            else:
                c = M.mean(0)
            s = float(F.cosine_similarity(Dm[pos[e]], c, dim=0))
            if s > best:
                best, bf = s, f_
        correct += bf == f0
    out["family_loo_centroid_acc"] = correct / n
    out["family_chance"] = max(len(v) for v in fam.values()) / n
    print(
        f"leave-one-out nearest-family-centroid accuracy {correct / n:.3f} (majority chance {out['family_chance']:.3f})"
    )

    # --- pair tests
    text_row = {texts[e]: e for e in rows}
    from common.text import HIRA, KATA

    def pair_test(name, pairs, pool_a, pool_b):
        got = [(x, y) for x, y in pairs if x in text_row and y in text_row]
        if len(got) < 3:
            return None
        cs = [
            float(
                F.cosine_similarity(Dm[pos[text_row[x]]], Dm[pos[text_row[y]]], dim=0)
            )
            for x, y in got
        ]
        A = [text_row[t] for t in pool_a if t in text_row]
        B = [text_row[t] for t in pool_b if t in text_row]
        rng = random.Random(0)
        rnd = []
        for _ in range(4000):
            x, y = rng.choice(A), rng.choice(B)
            if x != y:
                rnd.append(float(F.cosine_similarity(Dm[pos[x]], Dm[pos[y]], dim=0)))
        m = sum(cs) / len(cs)
        pct = sum(r < m for r in rnd) / len(rnd) * 100
        # rank of the true partner among all pool_b rows
        ranks = []
        for x, y in got:
            sims = C[pos[text_row[x]]][[pos[b] for b in B]]
            ranks.append(
                int(
                    (
                        sims
                        > float(
                            F.cosine_similarity(
                                Dm[pos[text_row[x]]], Dm[pos[text_row[y]]], dim=0
                            )
                        )
                    ).sum()
                )
                + 1
            )
        r = {
            "n": len(got),
            "mean_cos": m,
            "random_mean": sum(rnd) / len(rnd),
            "pct": pct,
            "partner_rank_median": float(np.median(ranks)),
            "pool": len(B),
            "top1": sum(k == 1 for k in ranks),
        }
        print(
            f"  {name:22s} n {len(got):3d}  cos {m:+.3f} (random {r['random_mean']:+.3f}, pct {pct:5.1f})  partner rank median {r['partner_rank_median']:.0f}/{len(B)}, top-1 {r['top1']}/{len(got)}"
        )
        return r

    print(
        "\npair tests (residual table, m̂ projected out; pct = percentile of the mean pair cos among random pairs; ≥95 = structure):"
    )
    voiced = [("か", "が"), ("き", "ぎ"), ("く", "ぐ"), ("け", "げ"), ("こ", "ご"), ("さ", "ざ"), ("し", "じ"), ("す", "ず"), ("せ", "ぜ"), ("そ", "ぞ"),
              ("た", "だ"), ("ち", "ぢ"), ("つ", "づ"), ("て", "で"), ("と", "ど"), ("は", "ば"), ("ひ", "び"), ("ふ", "ぶ"), ("へ", "べ"), ("ほ", "ぼ"),
              ("は", "ぱ"), ("ひ", "ぴ"), ("ふ", "ぷ"), ("へ", "ぺ"), ("ほ", "ぽ")]  # fmt: skip
    voiced_k = [(chr(ord(x) + 0x60), chr(ord(y) + 0x60)) for x, y in voiced]
    hk = list(zip(HIRA, KATA))
    small = [("あ", "ぁ"), ("い", "ぃ"), ("う", "ぅ"), ("え", "ぇ"), ("お", "ぉ"), ("つ", "っ"), ("や", "ゃ"), ("ゆ", "ゅ"), ("よ", "ょ"),
             ("ア", "ァ"), ("イ", "ィ"), ("ウ", "ゥ"), ("エ", "ェ"), ("オ", "ォ"), ("ツ", "ッ"), ("ヤ", "ャ"), ("ユ", "ュ"), ("ヨ", "ョ")]  # fmt: skip
    dak_pairs = [
        ("ば", "ぱ"),
        ("び", "ぴ"),
        ("ぶ", "ぷ"),
        ("べ", "ぺ"),
        ("ぼ", "ぽ"),
        ("バ", "パ"),
        ("ビ", "ピ"),
        ("ブ", "プ"),
        ("ベ", "ペ"),
        ("ボ", "ポ"),
    ]
    allv = [texts[e] for e in rows if cls[e] in ("hira_v", "kata_v")]
    alls = [texts[e] for e in rows if cls[e] == "small"]
    allk = [texts[e] for e in rows if cls[e] == "kanji"]
    allh = [texts[e] for e in rows if cls[e] == "hira"]
    allkt = [texts[e] for e in rows if cls[e] == "kata"]
    out["pairs"] = {
        "voiced_hira (か↔が, same shape + mark)": pair_test(
            "voiced hira", voiced, allh, allv
        ),
        "voiced_kata (カ↔ガ)": pair_test("voiced kata", voiced_k, allkt, allv),
        "dakuten↔handakuten (ば↔ぱ)": pair_test("ば↔ぱ", dak_pairs, allv, allv),
        "hira↔kata (あ↔ア, same sound, other shape)": pair_test(
            "hira↔kata", hk, allh, allkt
        ),
        "large↔small (つ↔っ, same shape, size)": pair_test(
            "large↔small", small, allh + allkt, alls
        ),
    }
    ids = [("明", ["日", "月"]), ("林", ["木", "木"]), ("森", ["木"]), ("休", ["人", "木"]), ("好", ["女", "子"]), ("男", ["田", "力"]), ("岩", ["山", "石"]),
           ("加", ["力", "口"]), ("相", ["木", "目"]), ("困", ["口", "木"]), ("体", ["人", "本"]), ("何", ["人", "可"]), ("時", ["日", "寺"]), ("間", ["門", "日"]),
           ("思", ["田", "心"]), ("愛", ["心"]), ("私", ["禾", "厶"]), ("出", ["山", "山"]), ["姉", ["女", "市"]], ["妹", ["女", "未"]], ["娘", ["女", "良"]],
           ["嫌", ["女", "兼"]], ["始", ["女", "台"]], ["待", ["寺"]], ["持", ["手", "寺"]], ["指", ["手", "旨"]], ["言", ["口"]], ["話", ["言", "舌"]],
           ["語", ["言", "吾"]], ["読", ["言", "売"]], ["晩", ["日", "免"]], ["昨", ["日", "乍"]], ["早", ["日", "十"]], ["暑", ["日", "者"]]]  # fmt: skip
    ids_pairs = [(c, at) for c, ats in ids for at in ats]
    out["pairs"]["kanji_component (明↔日)"] = pair_test(
        "kanji ↔ component", ids_pairs, allk, allk
    )

    # --- Mantel: Δ-cos vs glyph pixel cos (single-char rows, one font)
    try:
        from common.render.flat import find_fonts
        from PIL import Image, ImageDraw, ImageFont

        fonts = find_fonts()
        singles = [e for e in rows if cls[e] not in ("word", "other")]
        if fonts and len(singles) > 10:
            font = ImageFont.truetype(fonts[0], 48, index=0)
            pix = []
            for e in singles:
                im = Image.new("L", (64, 64), 255)
                d = ImageDraw.Draw(im)
                l_, t_, r_, b_ = d.textbbox((0, 0), texts[e], font=font)
                d.text(
                    ((64 - (r_ - l_)) / 2 - l_, (64 - (b_ - t_)) / 2 - t_),
                    texts[e],
                    fill=0,
                    font=font,
                )
                pix.append(
                    1
                    - torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).flatten()
                )
            P = torch.stack(pix)
            P = P - P.mean(0)
            pc = offdiag(cosmat(P)).numpy()
            ii = [pos[e] for e in singles]
            dc = offdiag(C[ii][:, ii]).numpy()
            from scipy.stats import spearmanr

            rho = spearmanr(pc, dc).correlation
            rng = np.random.default_rng(0)
            m = len(singles)
            Cs = C[ii][:, ii].numpy()
            null = []
            for _ in range(300):
                p_ = rng.permutation(m)
                null.append(
                    spearmanr(pc, Cs[p_][:, p_][~np.eye(m, dtype=bool)]).correlation
                )
            pval = (np.sum(np.abs(null) >= abs(rho)) + 1) / (len(null) + 1)
            # by family
            byfam = {}
            for f_ in ("hira", "kata", "kanji"):
                ss = [e for e in singles if cls[e] == f_]
                if len(ss) > 8:
                    jj = [singles.index(e) for e in ss]
                    Pf = P[jj]
                    byfam[f_] = float(
                        spearmanr(
                            offdiag(cosmat(Pf)).numpy(),
                            offdiag(
                                C[[pos[e] for e in ss]][:, [pos[e] for e in ss]]
                            ).numpy(),
                        ).correlation
                    )
            out["mantel_pixel_vs_delta"] = {
                "n": m,
                "spearman": float(rho),
                "perm_p": float(pval),
                "by_family": byfam,
                "font": Path(fonts[0]).name,
            }
            print(
                f"\nMantel Δ-cos vs glyph pixel-cos ({m} single-char rows, {Path(fonts[0]).name}): spearman {rho:+.3f}, perm p {pval:.3f}; by family {byfam}"
            )
    except Exception as ex:  # noqa: BLE001
        print("Mantel skipped:", ex)

    # --- relation to the pack rows and the stock table
    from library.anima.weights import load_llm_adapter

    adapter = load_llm_adapter(
        ck.dit, dtype=torch.float32, device="cpu", vocab_pack=pack
    ).eval()
    W = adapter.embed.weight.detach().float()
    from library.anima.ext_vocab import T5_TABLE_SIZE

    own = pack.table.float()[
        rows
    ]  # the pack's ext rows live in pack.table, not in embed.weight
    stock = W[:T5_TABLE_SIZE]
    ext_all = pack.table.float()
    rs = float(sd["delta"]["row_scale"])
    out["pack_relation"] = {
        "cos_delta_to_own_row": float(F.cosine_similarity(D, own).mean()),
        "delta_norm_over_own_norm": float((norms * rs / own.norm(dim=1)).mean()),
        "cos_mean_delta_to_stock_mean": float(
            F.cosine_similarity(mean, stock.mean(0), dim=0)
        ),
        "cos_mean_delta_to_ext_mean": float(
            F.cosine_similarity(mean, ext_all.mean(0), dim=0)
        ),
    }
    U, S, Vh = torch.linalg.svd(stock - stock.mean(0), full_matrices=False)
    top = Vh[:256]

    def energy(X):
        return float(((X @ top.T) ** 2).sum(1).div((X**2).sum(1)).mean())

    out["pack_relation"]["energy_in_stock_top256"] = {
        "delta": energy(Dm),
        "delta_raw": energy(D),
        "gaussian": energy(torch.randn(256, dim)),
        "stock_rows": energy(
            stock[torch.randperm(T5_TABLE_SIZE)[:512]] - stock.mean(0)
        ),
        "pack_ext_rows": energy(own - stock.mean(0)),
    }
    print(
        "\npack relation:",
        {
            k: (
                round(v, 3)
                if isinstance(v, float)
                else {kk: round(vv, 3) for kk, vv in v.items()}
            )
            for k, v in out["pack_relation"].items()
        },
    )

    # --- hit vs geometry
    rows_n, frames = exposure(a.arm_dir, tok, set(rows))
    out["frames_in_training"] = dict(frames)
    crowd = (C - torch.eye(n) * 2).max(1).values  # nearest other row cos
    recs = []
    for e in rows:
        h = hits.get(texts[e])
        recs.append(
            {
                "text": texts[e],
                "family": cls[e],
                "norm": float(norms[pos[e]]),
                "cos_mean": float(F.cosine_similarity(D[pos[e]], mean, dim=0)),
                "crowd": float(crowd[pos[e]]),
                "exposure": int(rows_n.get(e, 0)),
                "hits": (sum(h) if h else None),
                "hit_n": (len(h) if h else 0),
            }
        )
    out["rows"] = recs
    ev = [r for r in recs if r["hit_n"]]
    if ev:
        from scipy.stats import spearmanr

        y = np.array([r["hits"] / r["hit_n"] for r in ev])
        corr = {}
        for k in ("norm", "cos_mean", "crowd", "exposure"):
            x = np.array([r[k] for r in ev], dtype=float)
            corr[k] = float(spearmanr(x, y).correlation)
        out["hit_correlates_spearman"] = corr
        print(f"\nhit (singles exact, {len(ev)} rows) vs row stats, spearman: {corr}")
        for f_ in fams:
            sub = [r for r in ev if r["family"] == f_]
            if len(sub) >= 6:
                yh = [r for r in sub if r["hits"]]
                ym = [r for r in sub if not r["hits"]]
                if yh and ym:
                    print(
                        f"  {f_:7s} hit rows n {len(yh)} norm {np.mean([r['norm'] for r in yh]):.3f} crowd {np.mean([r['crowd'] for r in yh]):.3f} exp {np.mean([r['exposure'] for r in yh]):.0f} | miss rows n {len(ym)} norm {np.mean([r['norm'] for r in ym]):.3f} crowd {np.mean([r['crowd'] for r in ym]):.3f} exp {np.mean([r['exposure'] for r in ym]):.0f}"
                    )
    (a.arm_dir / "manifold").mkdir(exist_ok=True)
    json.dump(
        out,
        open(a.arm_dir / "manifold" / "rows.json", "w"),
        ensure_ascii=False,
        indent=1,
    )
    print("wrote", a.arm_dir / "manifold" / "rows.json")


# ---------------------------------------------------------------- adapter stage


@torch.no_grad()
def stage_adapter(a):
    from common.hooks import ExtDelta
    from train.encoder import row_texts

    from library.anima.ext_vocab import T5_TABLE_SIZE
    from library.anima.weights import load_llm_adapter
    from library.inference.models import load_text_encoder

    dev = torch.device(a.device)
    dt = torch.float32 if dev.type == "cpu" else torch.bfloat16
    sd, raw, ext_ids = load_arm(a.arm_dir)
    ck, pack, tok, enc = load_text_side()
    texts = row_texts(tok, pack, ext_ids)
    te = load_text_encoder(text_encoder=ck.text_encoder, dtype=dt, device=dev).eval()
    adapter = load_llm_adapter(ck.dit, dtype=dt, device=dev, vocab_pack=pack).eval()
    holder = types.SimpleNamespace(llm_adapter=adapter)
    delta = ExtDelta.from_state(holder, sd["delta"], dev)
    rows = [e for e in ext_ids if e in texts]
    cls = {e: classify(texts[e]) for e in rows}
    if a.limit:
        rng = random.Random(0)
        by = defaultdict(list)
        for e in rows:
            by[cls[e]].append(e)
        rows = [e for f_ in by for e in rng.sample(by[f_], min(a.limit, len(by[f_])))]
    hits = per_row_hits(a.arm_dir)
    print(f"{len(rows)} rows, families {Counter(cls[e] for e in rows)}, device {dev}")

    def encode(caps):
        tokens = tok.tokenize(caps)
        pe, am, t5, t5m = enc.encode_tokens(tok, [te], tokens)
        return pe.to(dev), am.to(dev), t5.long().to(dev), t5m.to(dev)

    def adapter_out(pe, am, t5, t5m):
        return adapter(
            pe, t5, target_attention_mask=t5m, source_attention_mask=am
        ).float()

    def codes_for(caps, want_pos):
        """want_pos(i, t5_row) → positions; returns (on, off) mean codes per caption (nan when none)."""
        on, off = [], []
        for s in range(0, len(caps), a.batch):
            pe, am, t5, t5m = encode(caps[s : s + a.batch])
            delta.scale = 1.0
            o1 = adapter_out(pe, am, t5, t5m)
            delta.scale = 0.0
            o0 = adapter_out(pe, am, t5, t5m)
            for i in range(len(t5)):
                p = want_pos(s + i, t5[i])
                if p:
                    on.append(o1[i, p].mean(0).cpu())
                    off.append(o0[i, p].mean(0).cpu())
                else:
                    on.append(torch.full((o1.shape[-1],), float("nan")))
                    off.append(on[-1].clone())
        return torch.stack(on), torch.stack(off)

    t5tok = tok.t5_tokenizer
    # --- EN codes per frame
    en_codes = {}
    for fname, tpl in FRAMES.items():
        caps = [tpl.format(lang="english", Lang="English", t=w) for w in EN]
        ids = [set(t5tok(w, add_special_tokens=False)["input_ids"]) for w in EN]
        c_on, c_off = codes_for(
            caps, lambda i, row: [k for k, v in enumerate(row.tolist()) if v in ids[i]]
        )
        assert torch.allclose(c_on, c_off, equal_nan=True), (
            "delta touched an EN caption"
        )
        en_codes[fname] = c_off
    ok = torch.ones(len(EN), dtype=torch.bool)
    for v in en_codes.values():
        ok &= ~torch.isnan(v[:, 0])
    en_codes = {k: v[ok] for k, v in en_codes.items()}
    P = en_codes["plain"]
    Q = {k: (v - P).mean(0) for k, v in en_codes.items() if k != "plain"}
    Qn = {k: v / v.norm() for k, v in Q.items()}
    Qavg = torch.stack(list(Q.values())).mean(0)
    Qavg_n = Qavg / Qavg.norm()
    en_shift_cos = {
        k: float(F.cosine_similarity(en_codes[k] - P, Qn[k][None]).mean()) for k in Q
    }
    print(
        f"EN words located {int(ok.sum())}/{len(EN)}; per-word cos(shift, Q_frame): { {k: round(v, 2) for k, v in en_shift_cos.items()} }"
    )
    allq = torch.cat([en_codes[k] for k in Q])
    Uq, Sq, Vq = torch.linalg.svd(allq - allq.mean(0), full_matrices=False)
    en_sub = Vq[:16]
    Qmean = {k: v.mean(0) for k, v in en_codes.items()}

    # --- trained rows per frame
    row_on, row_off = {}, {}
    for fname, tpl in FRAMES.items():
        caps = [tpl.format(lang="japanese", Lang="Japanese", t=texts[e]) for e in rows]
        want = [T5_TABLE_SIZE + e for e in rows]
        c_on, c_off = codes_for(
            caps, lambda i, row: [k for k, v in enumerate(row.tolist()) if v == want[i]]
        )
        row_on[fname], row_off[fname] = c_on, c_off
        print(
            f"  frame {fname}: rows located {int((~torch.isnan(c_on[:, 0])).sum())}/{len(rows)}"
        )
    okr = torch.ones(len(rows), dtype=torch.bool)
    for v in row_on.values():
        okr &= ~torch.isnan(v[:, 0])
    rows = [e for e, k in zip(rows, okr.tolist()) if k]
    row_on = {k: v[okr] for k, v in row_on.items()}
    row_off = {k: v[okr] for k, v in row_off.items()}
    fams = sorted(
        set(cls[e] for e in rows), key=lambda f_: -sum(cls[e] == f_ for e in rows)
    )
    famidx = {
        f_: torch.tensor([i for i, e in enumerate(rows) if cls[e] == f_]) for f_ in fams
    }

    out: dict = {"n_rows": len(rows), "en_shift_cos": en_shift_cos, "frames": {}}
    rnd = torch.randn(512, P.shape[1])
    out["random_cos_scale"] = float(F.cosine_similarity(rnd, Qavg_n[None]).abs().mean())
    hdr = "| frame | cos(d, Q_frame) | cos(d, Q_avg) | |d|/|off| | d pairwise cos | cos(mean d, Q_avg) | d energy in EN-quoted top16 (off / gauss) | cos(on, mean EN quoted) / (off, ·) | row frame-shift cos Q: on / off (EN word) |"
    print("\n" + hdr)
    print("|" + "---|" * 9)
    for fname in FRAMES:
        on, off = row_on[fname], row_off[fname]
        d = on - off
        dn = F.normalize(d, dim=1)
        rec = {
            "cos_d_Qframe": float(F.cosine_similarity(d, Qn[fname][None]).mean())
            if fname in Qn
            else None,
            "cos_d_Qavg": float(F.cosine_similarity(d, Qavg_n[None]).mean()),
            "d_rel_norm": float((d.norm(dim=1) / off.norm(dim=1)).mean()),
            "d_pairwise_cos": float(offdiag(dn @ dn.T).mean()),
            "d_spectrum": spectrum(d),
            "cos_meand_Qavg": float(F.cosine_similarity(d.mean(0), Qavg, dim=0)),
            "cos_meand_Qframe": float(F.cosine_similarity(d.mean(0), Q[fname], dim=0))
            if fname in Q
            else None,
            "cos_meand_meanENquoted": float(
                F.cosine_similarity(d.mean(0), Qmean[fname], dim=0)
            ),
            "d_energy_en_sub": float(
                ((d @ en_sub.T) ** 2).sum(1).div((d**2).sum(1)).mean()
            ),
            "off_energy_en_sub": float(
                ((off @ en_sub.T) ** 2).sum(1).div((off**2).sum(1)).mean()
            ),
            "gauss_energy_en_sub": float(
                ((rnd @ en_sub.T) ** 2).sum(1).div((rnd**2).sum(1)).mean()
            ),
            "cos_on_meanEN": float(F.cosine_similarity(on, Qmean[fname][None]).mean()),
            "cos_off_meanEN": float(
                F.cosine_similarity(off, Qmean[fname][None]).mean()
            ),
            "nearest_en_cos_on": float(
                (F.normalize(on, dim=1) @ F.normalize(en_codes[fname], dim=1).T)
                .max(1)
                .values.mean()
            ),
            "nearest_en_cos_off": float(
                (F.normalize(off, dim=1) @ F.normalize(en_codes[fname], dim=1).T)
                .max(1)
                .values.mean()
            ),
        }
        if fname != "plain":
            s_on = on - row_on["plain"]
            s_off = off - row_off["plain"]
            rec["rowshift_cos_Q_on"] = float(
                F.cosine_similarity(s_on, Qn[fname][None]).mean()
            )
            rec["rowshift_cos_Q_off"] = float(
                F.cosine_similarity(s_off, Qn[fname][None]).mean()
            )
            rec["rowshift_rel_on"] = float((s_on.norm(dim=1) / on.norm(dim=1)).mean())
            rec["rowshift_rel_off"] = float(
                (s_off.norm(dim=1) / off.norm(dim=1)).mean()
            )
        rec["by_family"] = {
            f_: {
                "cos_d_Qavg": float(
                    F.cosine_similarity(d[famidx[f_]], Qavg_n[None]).mean()
                ),
                "cos_d_meand": float(
                    F.cosine_similarity(d[famidx[f_]], d.mean(0)[None]).mean()
                ),
                "d_rel_norm": float(
                    (d[famidx[f_]].norm(dim=1) / off[famidx[f_]].norm(dim=1)).mean()
                ),
            }
            for f_ in fams
        }
        out["frames"][fname] = rec
        qf = fmt(rec["cos_d_Qframe"]) if rec["cos_d_Qframe"] is not None else "—"
        sh = (
            f"{rec['rowshift_cos_Q_on']:+.2f} / {rec['rowshift_cos_Q_off']:+.2f} ({en_shift_cos[fname]:+.2f})"
            if fname != "plain"
            else "—"
        )
        print(
            f"| {fname} | {qf} | {rec['cos_d_Qavg']:+.3f} | {rec['d_rel_norm']:.2f} | {rec['d_pairwise_cos']:+.3f} | {rec['cos_meand_Qavg']:+.3f} | {rec['d_energy_en_sub']:.3f} ({rec['off_energy_en_sub']:.3f} / {rec['gauss_energy_en_sub']:.3f}) | {rec['cos_on_meanEN']:+.2f} / {rec['cos_off_meanEN']:+.2f} | {sh} |"
        )

    # --- context: is d the same vector across frames?
    names = [k for k in FRAMES]
    ctx = {}
    print("\nframe-to-frame cos of a row's image d (1 = context-free at the adapter):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            di = row_on[names[i]] - row_off[names[i]]
            dj = row_on[names[j]] - row_off[names[j]]
            ctx[f"{names[i]}~{names[j]}"] = float(F.cosine_similarity(di, dj).mean())
    print("  " + ", ".join(f"{k} {v:+.2f}" for k, v in ctx.items()))
    out["d_cross_frame_cos"] = ctx
    # residual after removing the reads_as image: how much of d(saying) is new?
    d0 = row_on["reads_as"] - row_off["reads_as"]
    for fname in ("saying", "sign", "bubble_reads", "bare", "plain"):
        d1 = row_on[fname] - row_off[fname]
        proj = (d1 * d0).sum(1) / (d0**2).sum(1)
        resid = d1 - proj[:, None] * d0
        out["frames"][fname]["resid_vs_reads_as"] = float(
            (resid.norm(dim=1) / d1.norm(dim=1)).mean()
        )
        out["frames"][fname]["resid_cos_Qframe"] = float(
            F.cosine_similarity(resid, Qn.get(fname, Qavg_n)[None]).mean()
        )
    print(
        "  residual of d after projecting out d(reads_as): "
        + ", ".join(
            f"{k} {out['frames'][k]['resid_vs_reads_as']:.2f} (cos Q {out['frames'][k]['resid_cos_Qframe']:+.2f})"
            for k in ("saying", "sign", "bubble_reads", "bare", "plain")
        )
    )

    # --- family table for the two product frames + hit correlation
    print(
        "\n| family | n | cos(d, Q_avg) reads_as | saying | cos(d, mean d) reads_as | |d|/|off| | hit rate |"
    )
    print("|---|---|---|---|---|---|---|")
    for f_ in fams:
        r0 = out["frames"]["reads_as"]["by_family"][f_]
        r1 = out["frames"]["saying"]["by_family"][f_]
        h = [x for i in famidx[f_].tolist() for x in hits.get(texts[rows[i]], [])]
        hr = f"{sum(h) / len(h):.2f} ({len(h)})" if h else "—"
        print(
            f"| {f_} | {len(famidx[f_])} | {r0['cos_d_Qavg']:+.3f} | {r1['cos_d_Qavg']:+.3f} | {r0['cos_d_meand']:+.3f} | {r0['d_rel_norm']:.2f} | {hr} |"
        )
    d = row_on["reads_as"] - row_off["reads_as"]
    per = []
    for i, e in enumerate(rows):
        h = hits.get(texts[e])
        per.append(
            {
                "text": texts[e],
                "family": cls[e],
                "cos_d_Qavg": float(F.cosine_similarity(d[i], Qavg_n, dim=0)),
                "cos_d_meand": float(F.cosine_similarity(d[i], d.mean(0), dim=0)),
                "d_rel": float(d[i].norm() / row_off["reads_as"][i].norm()),
                "hits": sum(h) if h else None,
                "hit_n": len(h) if h else 0,
            }
        )
    out["rows"] = per
    ev = [r for r in per if r["hit_n"]]
    if ev:
        from scipy.stats import spearmanr

        y = np.array([r["hits"] / r["hit_n"] for r in ev])
        out["hit_correlates_spearman"] = {
            k: float(spearmanr(np.array([r[k] for r in ev]), y).correlation)
            for k in ("cos_d_Qavg", "cos_d_meand", "d_rel")
        }
        print(
            f"\nhit (singles exact, {len(ev)} rows) vs adapter-image stats, spearman: {out['hit_correlates_spearman']}"
        )
    (a.arm_dir / "manifold").mkdir(exist_ok=True)
    json.dump(
        out,
        open(a.arm_dir / "manifold" / "adapter.json", "w"),
        ensure_ascii=False,
        indent=1,
    )
    print("wrote", a.arm_dir / "manifold" / "adapter.json")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arm_dir", type=Path, required=True)
    p.add_argument("--stage", choices=["rows", "adapter"], default="rows")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--limit", type=int, default=0, help="adapter: rows per family (0 = all)"
    )
    a = p.parse_args()
    torch.set_grad_enabled(False)
    (stage_rows if a.stage == "rows" else stage_adapter)(a)


if __name__ == "__main__":
    main()
