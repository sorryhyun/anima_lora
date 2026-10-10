# ocr_reader — PaddleOCR-VL reader v4 (JA + KO + ZH)

Opened 2026-10-10 on branch `ocr-reader`, out of
[`finished/cjk_aware_anima_dit`](../finished/cjk_aware_anima_dit/README.md)
(the reader half: `findings.md` § "Settled — the SFX reader", `eval.md`,
`ocr/`).

## Why

The shipped reader (`anime_tools.ocr.sfx`, Hub
`sorryhyun/paddleocr-vl-1.6-manga-lora` v3 = `vl16_b2_norm4`) is a
Japanese-only fine-tune. It trains the NaViT tower + projector in full and
puts an r16 LoRA on the ERNIE LM, using Manga109-s COO plus speech data. Two
things push for a v4:

1. **Korean / Chinese got worse.** B′ misreads Korean and Chinese as Japanese
   (`findings.md` § Do not re-propose). This has never been measured, because
   the KO/ZH gate (plan2 K3) was never built. The only KO/ZH-trained arm
   (`vl16_pl_kozh`) used NC pseudo rows, so it cannot ship.
2. **More data.** The plan is to collect licence-clean data beyond Manga109,
   so the reader no longer leans only on manga-ocr's corpus.

We also want to reopen the **parametrization** question: the tower is
trained in full and the LM gets a LoRA. Nothing has measured the opposite
split (tower LoRA + LM full FT). The first probe is in
[`reports/tower_delta_svd.md`](reports/tower_delta_svd.md).

## State

| phase | status |
|---|---|
| P0 — gates (K3 KO/ZH set) | not started |
| P1 — data intake | user collecting |
| P2a — rank-truncated tower eval | spectrum probe done; eval skipped |
| P2b — tower LoRA + LM full FT arm | run 1 beats v3 on sincos (+31 strict, paired z 3.1), ties COO, 7.0 vs 12.1 GB — [`reports/p2b_tower_lora_lm_full.md`](reports/p2b_tower_lora_lm_full.md); seed 2 + attribution arm owed |
| P3 — mixed-language v4 run | blocked on P0 + P1 |

Plan and gates: [`roadmap.md`](roadmap.md). Open questions:
[`questions.md`](questions.md).

## The gate — sincos 597

This line scores sincos with
**`--labels project/ocr_reader/assets/sfx_labels_sincos_597.tsv`** (933 rows,
597 scored SFX). It is the finished line's 617-row file minus the 8 pages that
were deliberately removed from the dataset (2026-10-10). Every arm, v3
included, is scored on it. Never mix it with the `/ 617` figures in
`finished/…/eval.md`. v3 on this basis: strict 289, ♡-blind 336.

```
ANIMA_MANGA109S_ROOT=~/manga109s/Manga109s_released_2026_05_21 make daemon-run ARGS="--stall-timeout 900 \
  project/finished/cjk_aware_anima_dit/ocr/eval_sfx.py --reader vl16 --ckpt output/ocr/<run>/ep1 \
  --name <run>_597 --labels project/ocr_reader/assets/sfx_labels_sincos_597.tsv"
```

## Inherited — read before any arm

- Baseline ledger (rescored key): `finished/cjk_aware_anima_dit/findings.md`
  § "The reader ledger". sincos 617 SFX rows; v3/B′ sit in a ~10-line band at
  ~50 %. Binomial SE ≈ 12 lines, so a gap smaller than that is not a finding.
- The "Do not re-propose" list in that same `findings.md` still applies. Most
  relevant here:
  - Never screen or size a KO/ZH pool with B′.
  - Never label from `animetext-ocr` (those reads come from hayai, and they
    carry no 띄어쓰기).
  - Never judge a KO/ZH arm on the JA gates alone.
  - Never strip whitespace from training targets.
  - Check the tokenizer before choosing any target spelling.
- Licence: AnimeText is CC-BY-NC-SA. A shipped reader must be built without
  it, and without any pseudo rows derived from it.
- Training / eval code stays in `finished/cjk_aware_anima_dit/ocr/`
  (`finetune_vl16_lora.py`, `eval_sfx.py`, `rescore_eval.py`,
  `pseudo_label.py`). Code search skips `project/finished/`, so open those
  files by path. Every GPU step goes through the daemon.
- Hardware: RTX 5070 Ti 16 GB. No bitsandbytes on cu132.
