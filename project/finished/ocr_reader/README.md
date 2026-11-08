# ocr_reader — PaddleOCR-VL reader v5 (JA + KO + ZH, line breaks)

Opened 2026-10-10 on branch `ocr-reader`, out of
[`finished/cjk_aware_anima_dit`](../cjk_aware_anima_dit/README.md)
(the reader half: `findings.md` § "Settled — the SFX reader", `eval.md`,
`ocr/`).

## Verdict (closed 2026-10-15)

Shipped: **v4 = `vl16_p2b_kozh_ep2`** — tower r64 LoRA + LM decoder full FT
(P2b) on Manga109-s COO + speech + gelnote KO/ZH pseudo rows, 2 epochs. It is
Hub `sorryhyun/paddleocr-vl-1.6-manga-lora` `v4/`, pinned by `anime_tools`
v0.7.12 (`anime_tools.ocr.sfx`). Against v3 it wins sincos SFX (289 → 346 /
597 strict), COO SFX, and brings KO/ZH back to stock level (v3: KO 1 / 48).

Ceiling: **neither more labels nor more data moved it.** The hitocrawl ja↔ko
twins (Opus relabel of the low tail, 1 224 rows, plus 15 830 confident v4
self-labels) trained as `vl16_p2b_kozh_hito` do not beat v4 (sincos strict
−17, z −2.1; COO flat; K3 KO 26 → 25) — [`reports/p3_kozh_hito.md`](reports/p3_kozh_hito.md).
OvisOCR2 and OvisOCR3 are not candidates ([`reports/ovisocr2.md`](reports/ovisocr2.md), [`reports/ovisocr3.md`](reports/ovisocr3.md)).

Left open (not run): seed 2 of P2b / P3, the P2b attribution arm, a KO tier
check on hito crops, speech calibration on COO test, a licence-clean KO/ZH
hand-lettered SFX source to clear stock. The hito crops and Opus labels stay
at `/media/sorryhyun/new/dataset/hito_ocr/` (licence unclear — never ship
from them). Code runs by path from here (`ocr/`, `gate/`, `hito/`); the state
table below is the record as of close.

**v5 (2026-11-09) = `vl16_p2b_kozh_nl`**, replacing v4 as Hub `v5/` (`fb08a10b`), pinned by
`anime_tools` v0.9.5 — v4's command with two changes
([`docs/proposal/ocr_line_breaks.md`](../../../docs/proposal/ocr_line_breaks.md)): the targets keep
the labels' line breaks as `\n` (`TARGET_NORM` 5), and the Manga109 crops are rebuilt so the 12 %
pad stops at the page edge (`pilot.deskew_crop(trim=True)`; the old cut replicated border
lines into ~13 % of the COO SFX crops; old tree kept as `derived/crops_replicate/`). Paired
against v4 rescored on the rebuilt crops: sincos SFX strict 337 → 350, ♡-blind 386 → 414 (z +3.0);
sincos speech, COO SFX / speech flat; K3 KO 26 → 22 (z −1.4, misreads), ZH 30 → 30. KO
breaks all 5 annotated multi-line K3 crops at the right line count, ZH 1 / 5. Reports:
`reports/ocr_eval_*_nl_ep2*.md` (v5), `reports/ocr_eval_*_ep2_clip*.md` (v4 on the rebuilt crops).

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
| P0 — gates (K3 KO/ZH set) | gelnote holdout cut — 50 KO + 50 ZH crops, one per page, whole pages held out (`assets/k3_gelnote.tsv`, `gate/k3_gelnote.py`); Opus labels (96 scored), user-reviewed 2026-10-11; scorer `ocr/eval_k3.py` (`--compare` re-scores stored preds against the current labels) |
| P1 — data intake | gelnote KO/ZH pseudo rows: stock VL-1.6 teacher × hayai v2.5 Nova voter, `pseudo_label.py --pool gelnote` → KO 1 706 / ZH 2 132 kept of 11 326 (`derived/manifest_pseudo_gelnote_{ko,zh}.parquet`); P2b smoke with both appended ok |
| P1 — relabel choice (K3) | `gate/k3_agreement.py` → [`reports/k3_agreement.md`](reports/k3_agreement.md): KO teacher v4 × voter hayai25 keeps 26 / 48 at 24 / 26 (stock × hayai25: 20 at 19 / 20; spaced 21 vs 13); ZH v4 × stock keeps 29 at 26 / 29 (stock × hayai25 drops 19 right rows — hayai25 reads ZH 16 / 48). v4 mean token logprob ranks well: top quarter 12 / 12 KO, 11 / 12 ZH |
| P1b — hitocrawl twins | `hito/build_crops.py`: ja↔ko edition twins (`~/hitocrawl`, licence unclear), phash-aligned, ORB-registered, diffed → 19 794 regions on 3 016 pages, both sides cut, at `/media/sorryhyun/new/dataset/hito_ocr/ko/`. `hito/prelabel.py`: v4 reads + logprob (`prelabel_v4lp.jsonl`); usable after guards / script check JA 17 186, KO 14 172. JA calibration `gate/sfx_calibration.py` → [`reports/sfx_calibration_v4.md`](reports/sfx_calibration_v4.md): on sincos 597 SFX, logprob ≥ −0.012 is 85 % strict / 93 % ♡-blind exact, < −0.146 is 14 %; hito JA sits far more confident (median −0.006 vs −0.030, mostly speech). Opus relabel of the low tail (1 000 hito, `n_tokens ≥ 4`, + 200 dropped gelnote rows; second-pass checked) → **469 usable**, 289 of them KO hand-lettered SFX that v4 reads as katakana (0 / 345 match). Rounds 2–3 (`hito/opus_batch.py` + `opus_prompts/`; ranks 1 001–2 500 + round 1's 457 excluded crops relabelled): labellers flag instead of refuse, nothing excluded at merge → **+755 usable** (the redo set gave only 61: mostly diff noise); manifest `pseudo_opus_relabel` = 1 224 rows (`hito/opus_manifest.py`, `--drop_flagged` → 988). Confident half of the v4 self-labels → `pseudo_hito_v4` (15 830 rows, `hito/v4_manifest.py`, logprob ≥ side median, unchecked on hito). [`reports/opus_relabel.md`](reports/opus_relabel.md). Galleries flagged for audit: `hito_ocr/FLAGGED.md` (outside the repo). Training arm `vl16_p2b_kozh_hito` (P3 ×2 + both hito manifests) does not beat v4: sincos SFX strict 346 → 329 (z −2.1, mostly lost hearts; ♡-blind flat), COO flat, K3 KO 26 → 25 (kana leak 3 → 0) — [`reports/p3_kozh_hito.md`](reports/p3_kozh_hito.md). Owed: KO tier check on hito crops, speech calibration (COO test) |
| P2a — rank-truncated tower eval | spectrum probe done; eval skipped |
| P2b — tower LoRA + LM full FT arm | run 1 beats v3 on sincos (+31 strict, paired z 3.1), ties COO, 7.0 vs 12.1 GB — [`reports/p2b_tower_lora_lm_full.md`](reports/p2b_tower_lora_lm_full.md); seed 2 + attribution arm owed |
| P3 — mixed-language v4 run | `vl16_p2b_kozh` = P2b + gelnote KO/ZH pseudo: KO/ZH back to stock level (v3 lost them: KO 1/48), JA gates flat or up — [`reports/p3_kozh.md`](reports/p3_kozh.md). ×2 epochs (`vl16_p2b_kozh_ep2`): sincos strict 316 → 346 (z +3.2, mostly hearts), COO SFX +49 (z +3.6), KO/ZH unchanged — [`reports/p3_kozh_ep2.md`](reports/p3_kozh_ep2.md); chosen to replace ep1 as Hub `v4/`. Owed: seed 2, more KO/ZH data to clear stock |
| outside readers | OvisOCR2 (page parser, Qwen3.5-0.8B): ZH at stock, KO −9, sincos SFX 2 / 597 (crops) and 3 / 597 recall on whole pages — not a candidate — [`reports/ovisocr2.md`](reports/ovisocr2.md); OvisOCR3 (2026-10-21, `plain_ocr` prompt): KO 22 / 48 (stock level), ZH 27 / 48 (rewrites trad → simp), sincos SFX 9 / 597 — not a candidate. A 0.2-epoch LoRA reads 166 / 597; VL P3 at the same 16 k crops reads 233 (z +5.5) and wins COO too, so Ovis does not learn faster; VL's split on Ovis (LM decoder full FT) is no better (♡-blind 188 vs LoRA 198) — [`reports/ovisocr3.md`](reports/ovisocr3.md) |

Plan and gates: [`roadmap.md`](roadmap.md). Open questions:
[`questions.md`](questions.md).

## The gate — sincos 597

This line scores sincos with
**`--labels project/finished/ocr_reader/assets/sfx_labels_sincos_597.tsv`** (933 rows,
597 scored SFX). It is the finished line's 617-row file minus the 8 pages that
were deliberately removed from the dataset (2026-10-10). Every arm, v3
included, is scored on it. Never mix it with the `/ 617` figures in
`finished/…/eval.md`. v3 on this basis: strict 289, ♡-blind 336.

```
ANIMA_MANGA109S_ROOT=/media/sorryhyun/new/dataset/manga109s/Manga109s_released_2026_05_21 make daemon-run ARGS="--stall-timeout 900 \
  project/finished/ocr_reader/ocr/eval_sfx.py --reader vl16 --ckpt output/ocr/<run>/ep1 \
  --name <run>_597 --labels project/finished/ocr_reader/assets/sfx_labels_sincos_597.tsv"
```

## Gate images — `eval_data/` (gitignored)

Labels stay in the tracked `assets/` TSVs. The images every gate reads are
frozen under `project/finished/ocr_reader/eval_data/` by `gate/build_eval_data.py`:

- `sincos/crops/<row>.png` + `index.tsv`: the 933 label rows cut at pad 0.12,
  the same cut `eval_sfx.py` makes. `sincos/pages/` holds the 155 pages behind
  them. `eval_sfx.py` reads the frozen crops first (at pad 0.12) and then the
  frozen pages, so the gate no longer depends on `post_image_dataset/`.
  Checked pixel-identical to the on-the-fly cut (926 / 926 scored rows).
- `k3_gelnote/crops/` + `pages/`: the 100 K3 rows of `assets/k3_gelnote.tsv`.

Manga109-s (corpus + `derived/` manifests and crops) lives at
`/media/sorryhyun/new/dataset/manga109s/`. `~/manga109s` is a symlink to it,
so older commands still resolve.

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
- Training / eval code lives in [`ocr/`](ocr/) (`finetune_vl16_lora.py`,
  `eval_sfx.py`, `rescore_eval.py`, `pseudo_label.py` and their imports),
  copied 2026-10-10 from `finished/cjk_aware_anima_dit/ocr/`; `ocr/pilot.py`
  holds the helpers it used to load from `finished/cjk_aware_anima/datasets/`.
  Edit here, not the finished copy. `eval_sfx.py` defaults to the 597 labels and
  writes its reports to this line's `reports/`. Every GPU step goes through the
  daemon.
- Hardware: RTX 5070 Ti 16 GB. No bitsandbytes on cu132.
