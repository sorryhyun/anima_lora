# plan2 — Korean / Chinese pseudo-labels (2026-09-09)

Replaces the previous plan2 (P1 JA self-training · P2 label pick · P3 SSL
ladder). P1 shipped as `vl16_pl_20k`; its row is in [`eval.md`](eval.md).
Old text: `git show d0eb669f:project/cjk_aware_anima_dit/plan2.md`.

Apply P1's method to Korean and Chinese. Same recipe, one change: the teacher
is routed by script.

**stock + hayai** for Korean, because manga-ocr is JA-only and B′ emits
kana/kanji inside Korean lines, so neither can teach or vote on it. Chinese does
not inherit that pair — K0.

## K0 — is Chinese like Korean? — **no** (measured 2026-09-09)

Four readers × three arms, `probes/ocr_voter_hayai_yield.py`,
`output/ocr/pseudo/voter_matrix.json`. Pairwise filter yield:

| arm | stock+hayai | stock+B′ | B′+hayai | anything+manga-ocr |
|---|---|---|---|---|
| `ko` (n=130) | **60.0 %** | 36.9 % | 35.4 % | 0 |
| `zh` (n=271) | 27.7 % | 49.8 % | 26.2 % | ≤ 0.7 % |

Korean is settled: **stock teaches, hayai votes**. Chinese is not like it —
hayai emits kana on 3.7 % of the `zh` arm and simplified on only 75.6 %, so it
is a weak ZH voter, and the arm was selected on **B′'s** markers, which makes
B′'s 49.8 % circular. ZH stays open pending a screen that B′ did not pick.

Both script arms were B′-selected, so neither sizes its pool; the `rand` arm
(n=1000) puts KO at 0.5 % by stock and by hayai, against B′'s 0.13 % over 100k.

## K1 — screen with hayai, read the hits with stock

Not a second full sweep. hayai reads the draw; the rows it calls
hangul-dominant are the KO arm; stock re-reads only those (≈ 0.5 % of the
pool), so the expensive reader costs half a percent of a pass.

```
… sweep  --reader hayai
… sweep  --reader stock --rows_from hayai --rows_script ko
```

**Pilot, the `pl100k` draw (2026-09-09).** hayai calls 467 rows hangul-dominant
— **0.467 %**, against B′'s 0.130 %, so B′ undercounts Korean 3.6×. stock read
those 467 in a few minutes.

**Run at scale, `kscreen` (2026-09-10).** `ocr/screen_shards.py` streams the
shard parquets — crops in memory, hayai reads every one, only the `--script`
hits are cut to disk — so the pool is screened without a 2 M-crop draw. Train
shards 1–5, **2 428 682 crops at 158 crops/s (4.3 h), 26 282 hits (1.08 %)**:
ko 15 202, zh 11 080. stock then read all 26 282 in **21 min at 20.6 crops/s**
(`sweep --reader stock`, no `--rows_from` — the screen's draw *is* the hit set,
so one pass covers both arms; `--rows_from hayai --rows_script ko` would cut it
back to KO).

Screen and teacher agree closely: of hayai's 15 202 ko, stock calls 12 333 ko
and only 39 zh; of its 11 080 zh, stock calls 9 725 zh and 0 ko. Train shard 0
and val/test are **not** screened.

Do **not** buy the screen from `JustANormalTinkerer/animetext-ocr` (3.39 M rows
of the same pool, `transcription` + `language`). Its reads are hayai's — 39/108
byte-exact against our own hayai output on shared crops, the rest same-model
decode drift — so it is the voter, not a teacher, and its Korean carries no
띄어쓰기 (28.6 % of ko rows hold a space vs stock's 58 %). Its `language`
column is also useless for ZH: every one of shard 0's 427 `zh` rows is Japanese
(`紅魔館`, `咲夜!?` — the langid calls any kana-free kanji line zh), and none
pass `simplified_zh`. For KO the column matches `hangul_dominant` 25/25.

## K2 — teacher stock, voter hayai, label keeps its spacing

`pseudo_label.py filter --teacher stock --voter hayai --script ko`. Guards
unchanged (empty / truncated / too_long / char_run / ngram); route on the
teacher's output; the label takes the teacher's string.

Spacing is the reason the teacher must be stock and not hayai: stock writes
`알고 있었어.`, hayai writes `알고있었어.`. `exact_key` is whitespace-blind, so
the two still agree and hayai can vote — but only stock's string carries the
띄어쓰기, and 띄어쓰기 is lexical in Korean.

That spacing survives to the target only because the target normalizer was
fixed the same day — it used to delete whitespace outright. Pre-fix checkpoints
are marked dirty; see [`whitespace_fixed.md`](whitespace_fixed.md).

**Pilot.** 378 rows read by both, 94.7 % guard-ok, **182 kept (48.2 %)** — in
line with P1's Japanese 47.9 %. 55.5 % of the kept labels carry a space.
Horizontal crops keep at 53.2 %, vertical at 35.4 %. The voter is not merely
space-blind but space-*wrong* (`왜 그게 / 제 탓입니까!` vs hayai's
`왜그게제 탓입니까!`), which is why the label takes the teacher's string.

**Run at scale on `kscreen` (2026-09-10).** Route on the teacher, one pass per
arm (`filter --teacher stock --voter hayai --script {ko,zh} --out_suffix …`):

| arm | read by both | guard ok | **kept** | agree | horizontal / vertical |
|---|---|---|---|---|---|
| `ko` | 12 333 | 96.29 % | **5 930 (48.1 %)** | 49.9 % | 51.7 % / 41.1 % |
| `zh` |  9 764 | 96.54 % | **3 300 (33.8 %)** | 35.0 % | 32.9 % / 37.4 % |

KO reproduces the pilot at 32× the N — 48.1 % vs 48.2 %, spacing 58.7 % vs
55.5 %, the same horizontal > vertical ordering — so the pilot was not a small-N
fluke.

ZH now has a **non-circular** number: K0's 27.7 % (n=271) was measured on a
B′-selected arm, this 35.0 % (n=9 426) on a hayai-selected one. So hayai is a
weaker ZH voter than KO voter, but less weak than K0 read — and ZH inverts KO's
orientation ordering (vertical keeps better than horizontal).

## K3 — the gate

sincos and COO are both Japanese, so a KO/ZH arm currently has no pass or fail.
Hand-label a KO/ZH set from K1's mined crops into the sincos schema, `text_rec`
= stock's read, `status = draft`. Split gate / calibration; the calibration half
gives precision of the kept rows per voter, which no run on this line has
measured.

## K4 — the arm

B′ recipe + `--extra_manifest` on top of the P1 pseudo rows. Append, not swap
(col100: 1.6 % append +2, 22.3 % swap −34).

Gate: K3's held-out set rises, and the Japanese gates do not drop below
`vl16_pl_20k` — sincos SFX ♡-blind 402 / 617, COO SFX 2189, COO speech 2260.
Report strict beside ♡-blind.

### Run 1 — `vl16_pl_kozh` (2026-09-10): Japanese half of the gate not met

B′ recipe unchanged, `--extra_manifest pseudo_pl20k pseudo_kscreen_ko
pseudo_kscreen_zh --speech_ratio 0.568955` (speech pinned at B′'s 38 582, so
the pseudo rows are the only difference from `vl16_pl_20k`); train 106 394 =
grey 38 582 sfx + 38 582 speech + pseudo 29 230. One arm, ko and zh together —
without K3 a split run costs 2× and answers nothing extra. Both rows rescored
through `eval_table.py`, so the basis matches.

| | B′ | `vl16_pl_20k` | `vl16_pl_kozh` | Δ |
|---|---|---|---|---|
| sincos SFX ♡-blind | 375 | **402** | **391** / 617 | **−11** |
| sincos strict | 312 | 334 | 331 | −3 |
| COO SFX ♡-blind | 2127 | **2189** | **2167** / 2558 | **−22** |
| COO speech ♡-blind | 2259 | 2260 | **2273** / 2559 | **+13** |
| in-domain val SFX | 86.2 % | 86.1 % | 86.9 % | +0.8 pp |

−11 / −22 are past the ±1-line batching jitter and past the ±3 spread the three
SSL rows show; no repeat-seed run exists on this line, so that is a bound, not a
variance.

**What it does and does not say.** It does *not* say the arm is worthless: the
gate is a guardrail on the Japanese line, and KO/ZH capability — the thing the
arm is for — is **still unmeasured**, because K3 does not exist. It says the
KO/ZH rows hand back part of P1's SFX win while staying well above B′.

Direction is consistent: SFX down, speech up. `cmd_manifest` stamps every
pseudo row `kind = "sfx"` (`load_split` draws only sfx/speech), but AnimeText
crops are dialogue/UI lines, not manga onomatopoeia — so the append dilutes the
SFX pool. P1's 20 k rows have that same shape yet *raised* SFX, the difference
being that they are Japanese and so still carry signal for a Japanese SFX read;
KO/ZH rows carry none and only dilute.

So do not read run 1 as "KO/ZH hurts". Read it as "the Japanese SFX gate costs
~1 pp to buy KO/ZH rows, and nobody has yet measured what was bought."

## Constraints

The AnimeText pool is CC-BY-NC-SA, so any pseudo-label student is research-only
(`vl16_pl_20k` included) and the shipped reader stays B′.

## Anti-re-proposal

- Do not size the KO/ZH pool with B′ — it misreads both as Japanese, and a
  B′-selected arm cannot score B′ (K0's `zh` row).
- Do not re-run K0's `zh` arm off a B′ screen; that is the confound, not the
  answer.
- Do not label KO/ZH from `JustANormalTinkerer/animetext-ocr` — it is hayai's
  own output (K1), so it is the voter, and its Korean has no 띄어쓰기.
- Do not re-run K4 run 1's ko+zh arm expecting a different Japanese number, and
  do not judge a KO/ZH arm on the Japanese gates alone — build K3 first.
- No prompt hints / language tags (findings § Context — closed).
- No SSL tower arms — closed at "tied with B′" (`eval.md`).
- Never put a `/71` or pre-`acd41d72` number next to one from `eval.md`.
