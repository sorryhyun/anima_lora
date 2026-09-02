"""Z0 gate (plan_zh.md): key-space separability of the char-fallback rows.

CPU, no model. For each ext asset prefix given on the command line:
  * pairwise cos among the char-layer rows of the top-N JA *tag* kanji
    (counted over the real registers of the JA corpus) vs the same statistic
    over the top-N qwen-token kanji rows — the gate is char ≤ qwen-token;
  * random-pair collision rate (>0.5) and effective dim (PR) of the ext keys,
    per layer and overall (the map_probe numbers, on the shipped asset).

    .venv/bin/python project/cjk_aware_anima/probes/z0_probe.py \
        bench/cjk_adapter/assets/ext_embed_v1 bench/cjk_adapter/assets/ext_embed
"""

import collections
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from library.anima import ext_vocab  # noqa: E402
from library.anima.weights import load_qwen3_tokenizer, load_t5_tokenizer  # noqa: E402

PAIRS = REPO / "post_image_dataset/cjk_distill/pairs.jsonl"
REGISTERS = {"tags", "tags_alt", "names"}
TOP = 200
N_PAIRS = 20000
torch.manual_seed(0)


def stats(M: torch.Tensor) -> str:
    Mn = F.normalize(M.float(), dim=-1)
    G = Mn @ Mn.T
    n = len(M)
    off = G[~torch.eye(n, dtype=bool)]
    return f"cos {off.mean():.3f} sd {off.std():.3f} | >0.5 {float((off > 0.5).float().mean() * 100):5.1f}%"


def pr(M: torch.Tensor) -> float:
    C = torch.cov((M.float() - M.float().mean(0)).T)
    ev = torch.linalg.eigvalsh(C).clamp(min=0)
    return float(ev.sum() ** 2 / (ev**2).sum())


def main(prefixes: list[str]) -> None:
    qtok = load_qwen3_tokenizer(
        str(REPO / "models/text_encoders/qwen_3_06b_base.safetensors")
    )
    t5 = load_t5_tokenizer(None)
    mapping0 = json.loads(Path(prefixes[0] + ".json").read_text(encoding="utf-8"))
    enc = ext_vocab.HybridT5Encoder.from_mapping(t5, qtok, mapping0)
    N0, NQ = ext_vocab.T5_TABLE_SIZE, len(mapping0["qwen"])
    inv_char = {v: k for k, v in mapping0["char"].items()}
    inv_qwen = {v: qtok.decode([int(k)]).strip() for k, v in mapping0["qwen"].items()}

    texts = []
    with PAIRS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("register") in REGISTERS:
                texts.append(r["ja"])
            if len(texts) >= N_PAIRS:
                break
    c_char: collections.Counter = collections.Counter()
    c_qwen: collections.Counter = collections.Counter()
    for s in texts:
        ids, mask = enc.encode(s, 512)
        for i, m in zip(ids, mask):
            if not m or i < N0:
                continue
            r = i - N0
            if r >= NQ:
                c_char[r] += 1
            elif len(inv_qwen[r]) == 1 and "一" <= inv_qwen[r] <= "鿿":
                c_qwen[r] += 1
    top_char = [r for r, _ in c_char.most_common(TOP)]
    top_qwen = [r for r, _ in c_qwen.most_common(TOP)]
    print(
        f"JA tag corpus ({len(texts)} captions): char-layer kanji types {len(c_char)}, "
        f"top-{TOP} = {''.join(inv_char[r] for r in top_char[:40])}…"
    )
    print(
        f"qwen-token kanji types {len(c_qwen)}, top-{TOP} = {''.join(inv_qwen[r] for r in top_qwen[:40])}…"
    )

    g = torch.Generator().manual_seed(0)
    rand_all = torch.randint(0, mapping0["rows"], (2000,), generator=g)
    rand_char = NQ + torch.randint(0, mapping0["rows"] - NQ, (2000,), generator=g)
    rand_qwen = torch.randint(0, NQ, (2000,), generator=g)
    for prefix in prefixes:
        table, mapping = ext_vocab.load_ext_assets(Path(prefix))
        assert (
            mapping["qwen"] == mapping0["qwen"] and mapping["char"] == mapping0["char"]
        )
        st = mapping.get("stats", {})
        print(
            f"\n== {prefix}  (map {st.get('map', 'ridge')}, char_init {st.get('char_init', 'fragment-mean')}, "
            f"holdout {st.get('holdout_cos', float('nan')):.3f})"
        )
        print(f"  top-{TOP} JA tag kanji, char layer : {stats(table[top_char])}")
        print(
            f"  top-{TOP} JA tag kanji, qwen tokens: {stats(table[top_qwen])}   <- gate bar"
        )
        print(
            f"  random 2000 ext rows (all)        : {stats(table[rand_all])} | PR {pr(table[rand_all]):6.1f}"
        )
        print(
            f"  random 2000 char rows             : {stats(table[rand_char])} | PR {pr(table[rand_char]):6.1f}"
        )
        print(
            f"  random 2000 qwen rows             : {stats(table[rand_qwen])} | PR {pr(table[rand_qwen]):6.1f}"
        )
        # cross-layer: does a char row sit near the qwen row of the same char?
        # (chars that are clean single tokens have no char row, so compare
        # char rows against the qwen rows of kanji sharing ≥1 byte fragment —
        # not meaningful; instead report char-vs-qwen top sets' cross cos)
        A = F.normalize(table[top_char].float(), dim=-1)
        B = F.normalize(table[top_qwen].float(), dim=-1)
        print(f"  cross char×qwen top sets mean cos : {float((A @ B.T).mean()):.3f}")


if __name__ == "__main__":
    main(
        sys.argv[1:]
        or [
            "bench/cjk_adapter/assets/ext_embed_v1",
            "bench/cjk_adapter/assets/ext_embed",
        ]
    )
