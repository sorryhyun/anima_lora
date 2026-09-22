# cf_sense — Gate 0 of `idea.md` (2026-09-22)

> Can the caption move the frozen DiT's x0-estimate off a rendered glyph B
> toward A, and at which σ? Six `cf_sense` runs, no training. **Single
> glyphs: identity leverage is zero below σ 0.7 for native text and rows
> alike** (EN 0.000 / 0.012 / 0.051 at 0.35 / 0.5 / 0.6; rows 0.000 / 0.002 /
> 0.005), ≈ 0.2 at the σ 0.8 peak (EN 0.197, rows 0.155); rows have no order
> leverage at any σ. The written kill fires for singles. **Multi-glyph
> strings (addendum): the band moves with the box** — native two-word text
> holds 0.19 / 0.25 / 0.20 at 0.5 / 0.6 / 0.7 and three-word order 0.26 at
> 0.5, dead at 0.8–0.9 — **and the trained multi-glyph rows have none of it
> in either table** (≤ 0.06), because no multi-glyph row has been trained
> where that leverage lives. Gate 0 killed for singles, passed for strings;
> the next step is Gates 1–2 on a multi-glyph micro block at 0.5–0.7, and
> "`--t_max 0.6` is backwards" is re-scoped to singles.

Index: [`README.md`](README.md). Proposal: [`../idea.md`](../idea.md).
Stage: `src/eval/cf_sense.py`. Raw pack (sha `7b9fce0b…`) in every launch.

## What ran

| job | arm dir | lang | pairs | wall |
|---|---|---|---|---|
| `20260922-123640-334eee` | `rows_step1_0921_s30k/cf_sense_en/` | en (base only) | 24 id + 24 order | 1.5 m |
| `20260922-123857-de77fc` | `rows_step1_0921_s30k/cf_sense_ja/` | ja (trained + floor) | 24 id + 24 order | 1.5 m |
| `20260922-123936-b15274` | `rows_step1_0921m_merge/cf_sense_ja/` | ja (trained + floor) | 24 id + 24 order | 1.5 m |

```
… src/wake_probe.py --stage cf_sense --arm rows --data_tag step1_0921 --arm_tag s30k --cf_lang en --cf_pairs 24
… --cf_lang ja                                       # s30k
… --data_tag step1_0921m --arm_tag merge --cf_lang ja  # merged table
```

σ grid 0.35 / 0.5 / 0.6 / 0.7 / 0.8 / 0.9, one layout per pair, seed 0, 512².
`move = pos(c_A) − pos(c_B)` with `pos(c) = ⟨x̂0(c) − x0_B, x0_A − x0_B⟩ /
‖x0_A − x0_B‖²` in the glyph box (0 = the estimate sits on B, 1 = on A).
`leak` = out-of-box / in-box per-cell energy of `x̂0(c_A) − x̂0(c_B)`.

The first ja launch (`20260922-123640-96313c`) crashed on `sd["row_text"]` —
arms since the pack rewrite save `delta.ext_ids` only. `cf_sense` now decodes
ids through the pack (`probe.merge_tables.row_text_map`) when the key is
absent; `classify_str --cls_lang ja` still reads `row_text` and will hit the
same `KeyError` on these arms.

The merged table keeps s30k's 374 rows verbatim (`merge.json`
`on_overlap: keep-base`, max abs diff 0.0 over the shared rows), so its `id`
read is bit-identical to s30k's; only `order` differs in the third decimal
(a 2-kana caption that tokenises to one of the merge-added pieces). One read
below.

## Identity (`id` pairs: two units of one length, 24)

| σ | EN base, move | EN move > 0.25 | rows trained, move | rows > 0.25 | rows floor | EN leak | rows leak |
|---|---|---|---|---|---|---|---|
| 0.35 | +0.000 ± 0.001 | 0.00 | +0.000 ± 0.001 | 0.00 | +0.000 | 0.331 | 0.427 |
| 0.50 | +0.012 ± 0.026 | 0.00 | +0.002 ± 0.006 | 0.00 | +0.000 | 0.140 | 0.228 |
| 0.60 | +0.051 ± 0.077 | 0.04 | +0.005 ± 0.011 | 0.00 | +0.000 | 0.055 | 0.139 |
| 0.70 | +0.166 ± 0.104 | 0.25 | +0.049 ± 0.074 | 0.04 | +0.004 | 0.010 | 0.124 |
| **0.80** | **+0.197 ± 0.089** | 0.29 | **+0.155 ± 0.112** | 0.29 | +0.001 | 0.003 | 0.019 |
| 0.90 | +0.067 ± 0.081 | 0.04 | +0.120 ± 0.113 | 0.12 | −0.001 | 0.043 | 0.045 |

`pos(c_A)` / `pos(c_B)` at 0.9: EN 0.50 / 0.43, rows 0.45 / 0.33 — at σ 0.9
the estimate sits halfway between A and B under either caption (the blank /
mean glyph), and text's leverage is the difference of two blurs.

## Order (`order` pairs: the same two units, swapped, 24)

| σ | EN base, move | EN move > 0.25 | rows trained, move | rows floor | EN leak | rows leak |
|---|---|---|---|---|---|---|
| 0.35 | +0.007 ± 0.023 | 0.00 | −0.000 ± 0.000 | +0.000 | 0.118 | 0.762 |
| 0.50 | +0.090 ± 0.073 | 0.04 | −0.000 ± 0.000 | −0.000 | 0.011 | 0.570 |
| **0.60** | **+0.227 ± 0.068** | 0.21 | +0.000 ± 0.002 | +0.000 | 0.001 | 0.457 |
| 0.70 | +0.162 ± 0.090 | 0.08 | −0.003 ± 0.019 | +0.002 | 0.003 | 0.413 |
| 0.80 | +0.100 ± 0.074 | 0.08 | −0.001 ± 0.029 | +0.002 | 0.011 | 0.101 |
| 0.90 | +0.022 ± 0.028 | 0.00 | +0.014 ± 0.019 | −0.000 | 0.289 | 0.165 |

## Per glyph (rows, trained, `id`, move at σ 0.7 / 0.8 / 0.9)

Live at 0.8 (> 0.25): シ/つ 0.34 / 0.33 / 0.03 · た/キ 0.06 / 0.33 / 0.01 ·
へ/う 0.06 / 0.31 / 0.07 · く/ヨ 0.06 / 0.31 / 0.18 · う/ろ 0.10 / 0.26 / −0.04 ·
イ/ア 0.05 / 0.26 / 0.02 · み/ち 0.12 / 0.26 / 0.17.
Live only at 0.9: せ/ろ 0.01 / 0.04 / 0.39 · す/へ 0.01 / 0.03 / 0.28 ·
ス/り −0.00 / 0.05 / 0.28 · は/え −0.00 / −0.00 / 0.23 · サ/こ 0.10 / 0.04 / 0.21 ·
こ/な −0.00 / 0.01 / 0.20.
Dead throughout: チ/イ 0.01 / 0.02 / 0.05 · キ/く 0.05 / 0.03 / −0.01 ·
オ/や −0.02 / 0.14 / −0.02.
Order pairs: every one within ± 0.05 at every σ except ンカ/カン (−0.09 /
−0.10 at 0.7 / 0.8 — the caption pushes the estimate the wrong way).

EN by word length (`id`, move at 0.6 / 0.7 / 0.8): 4-letter pairs (RENO/PANU,
VAVE/ZOLO, GINE/KULU, MOMO/TEZU, DUMU/RAMI) 0.00 / 0.00–0.12 / 0.06–0.32;
7-letter pairs (LIGESUL, VADEBOF, NAVIROZ, RUKOSAS, REBUFAR) 0.11–0.28 /
0.27–0.34 / 0.15–0.29. The σ at which text takes over rises as the box
shrinks; a single kana is the small end.

## Reads

1. **Identity sensitivity is zero below σ 0.7, for everyone.** Native Latin
   text — the ceiling any row could reach — moves the estimate 0 / 1 / 5 % of
   the way off B at 0.35 / 0.5 / 0.6. The kill in `idea.md` Gate 0 (EN move
   < ≈ 0.1 below 0.7) fires. `idea.md` § 2 point 1 — the CF residual
   `(x0_A − x0_B)/σ` "does not vanish at low σ, it grows" — is true of the
   residual and irrelevant to the gradient: `∂v_θ/∂e_A` along `d` is what
   the row multiplies, and it is ≈ 0 there. A CF item at 0.5–0.7 is the
   § 2 caveat exactly: a large residual the row cannot reduce, i.e. Adam
   norm inflation with no identity bought.
2. **Where text is live it holds ≈ 20 % of the estimate, the input 80 %.**
   EN 0.197 at 0.8, the trained rows 0.155 (78 % of the ceiling) and 0.120
   at 0.9 (above EN's 0.067 — the rows were trained at 0.7–0.9 and are live
   exactly there; native text was not). The untouched pack rows (`floor`)
   have zero leverage at every σ: what leverage the rows have is trained.
   So a CF item in 0.7–0.9 leaves ≈ 80 % of `(x0_A − x0_B)/σ` unreducible
   by the row at every step — the row-norm pressure Gate 1 was written to
   catch is the expected regime, not a failure mode to rule out.
3. **Order is the one thing native text moves where identity is locked**
   (0.23 at 0.6, 0.16 at 0.7, 0.09 at 0.5, leak ≤ 0.01 — the swap happens
   inside the box). This is the `classify_str --cls_lang en` band (order
   live 0.5–0.8) seen from a rendered input. **The kana rows have none of
   it**: two trained single rows concatenated read 0.000 ± 0.03 at every σ
   against the reversed render, `trained` and `floor` alike. Rows carry
   identity at 0.8; they do not carry an order the input contradicts.
4. **Leak is not a concern in the live band** (0.003–0.02); it climbs toward
   1 only where `move` → 0 and both energies are noise.

## Verdict on the gates (single-glyph read — superseded by the addendum's revised verdict below)

- **Gate 0: killed as written for identity.** The mechanism that made CF
  "more effective per draw" — a new gradient in the 0.5–0.7 band where plain
  FM has none — is not there; the frozen DiT gives text nothing to do below
  0.7 whether the text is native or a trained row.
- **Gates 1–2** survive only as the 0.7–0.9 variance argument (§ 2 points 2–3:
  trigger projected out, the identity part deterministic) on plain FM's own
  band. The `CF 0.5–0.9` arm and its `plain 0.5–0.9` confound are dropped.
  If run: one 25-min micro arm `--cf_input 0.5 --t_min 0.7 --t_max 0.9` with
  its `--cf_input 0` twin, reading `in_box_cf` / `row_norm` (Gate 1) and
  singles + native (Gate 2) from the same pair; Gate 1's kill is now the
  expected outcome, so a pass is the surprise.
- **Gate 3 (order): closed for the row form under test.** Permutation
  siblings can only teach an order the row can influence, and concatenated
  single rows influence none at any σ. A string row (a multi-glyph piece of
  the merged table) was not measured — `_ja_pairs` concatenates singles — so
  the door for *string* rows is unread, not shut.
- Gate 4 has nothing to read yet.

Recommendation: shelve CF unless the 0.7–0.9 twin pair above is worth
50 min on its own; nothing in Gate 0 says it will beat plain at equal draws,
and the argument left is the same class as ΔFM's, which did not.

## Addendum, same day: multi-glyph strings (`--cf_rows piece`)

The read above is the single-glyph case. The EN per-length table already said
the band moves with the box, so the probe gained `--cf_rows piece`: EN `id` =
two two-word strings of matched lengths, `order` = a three-word string vs its
first two words swapped; JA `id` = two kana-only multi-glyph rows of one glyph
count, `order` = two rows concatenated vs swapped (the caption tokenises the
concatenation as a sentence would). Three runs, 24 + 24 pairs each:

| job | arm dir | lang |
|---|---|---|
| `20260922-125300-e70ed3` | `rows_step1_0921m_merge/cf_sense_en_piece/` | en (base) |
| `20260922-125300-a09c11` | `rows_step1_0921m_merge/cf_sense_ja_piece/` | ja — merged table (multi-glyph rows from `z_s152k`, trained 0.7–0.9) |
| `20260922-125300-a9e419` | `rows_step2_0921m_plain_bs05c25_30k/cf_sense_ja_piece/` | ja — the sentence-run table (0.5–0.9, μ 0.3; cos 0.9985 to its seed over 2 271 rows) |

| σ | EN id (2-word) | EN order (3-word) | rows id (merge) | rows id (sentence) | rows order (either) | floor |
|---|---|---|---|---|---|---|
| 0.35 | +0.037 ± 0.047 | **+0.139 ± 0.113** | +0.000 | +0.000 | −0.000 | 0.000 |
| 0.50 | **+0.193 ± 0.061** | **+0.264 ± 0.103** | +0.001 | +0.001 | +0.000 | 0.000 |
| 0.60 | **+0.248 ± 0.068** | **+0.218 ± 0.142** | +0.006 | +0.006 | +0.011 | 0.000 |
| 0.70 | **+0.203 ± 0.103** | +0.161 ± 0.122 | +0.032 | +0.033 | +0.028 / +0.030 | −0.001 |
| 0.80 | +0.101 ± 0.072 | +0.066 ± 0.056 | +0.059 ± 0.072 | +0.059 ± 0.072 | +0.024 / +0.025 | +0.001 |
| 0.90 | +0.018 ± 0.020 | +0.009 ± 0.013 | +0.023 | +0.023 | +0.004 / +0.005 | −0.001 |

EN leak in the live band 0.001–0.008 (box-local). `move > 0.25` for EN: id
0.17 / 0.46 / 0.33 of pairs at 0.5 / 0.6 / 0.7, order 0.58 at 0.5; per pair
the strongest are RENO PANUBU (0.33 / 0.35 at 0.6 / 0.7), RUKOSAS BEVUGE
(0.39 / 0.37), VAZOG REDOB (0.30 / 0.35); order RENO PANUBU MUTI 0.47 / 0.48 at
0.5 / 0.6, NARAV VUGUFE SOMAK 0.49 at 0.6. Trained rows: the best `id` pair is
やはり/さんに 0.17 at 0.8, してくれる/いたします 0.12 at 0.7; everything else
< 0.1 at every σ.

### Reads

5. **For native multi-token text the live band is σ 0.5–0.7** — identity
   0.19 / 0.25 / 0.20, order 0.26 / 0.22 / 0.16 — and it is *dead* at 0.8–0.9
   where the single-word band peaks. The single-glyph kill was the
   small-box case, not the DiT's rule: with a two-word box the caption holds
   a quarter of the estimate at 0.6, 5× the single-word number. The
   sensitivity CF needs at 0.5–0.7 exists for strings.
6. **The trained multi-glyph rows have none of it, at any σ, in either
   table** (≤ 0.06 at the 0.8 mean; ≈ 0 in 0.5–0.7). The merged table's
   pieces were trained at 0.7–0.9 (`z_s152k`) and the sentence run moved
   them to cos 0.9985 of that seed, so no multi-glyph row has been trained
   where its leverage would live. Contrast the singles: trained at 0.7–0.9,
   live at 0.8–0.9 at 78 % of the native ceiling. Leverage sits where the
   row was trained; the multi-glyph rows were trained where multi-token
   text has none.
7. This also re-scopes **"`--t_max 0.6` is backwards"** (`findings.md`, what
   does not move it): that arm ran on singles, whose native ceiling is
   0.05 at 0.6. For strings the native model decides identity and order
   at 0.5–0.7, and no arm has trained multi-glyph rows there with a
   band that excludes 0.8–0.9.

### Verdict, revised

- **Gate 0 is killed for single-glyph rows and passed for multi-glyph
  strings** (EN ceiling 0.25 at 0.6 against the 0.1 kill). The kill
  above stands for singles; the CF proposal moves to the sentence line.
- **Gates 1–2 on pieces are the next step**, and the band confound
  `idea.md` Gate 2 named is now first-class for the sentence line
  regardless of CF: a micro block of kana-only multi-glyph rows, 25-min
  arms — plain 0.7–0.9 · plain 0.5–0.7 · CF 0.5 at 0.5–0.7 — read with
  `cf_sense --cf_rows piece` on each arm (does the row's leverage appear
  where it trained?) beside the sentence rulers (sub-exact, native `en` /
  `swap`, joint hit). Gate 1's kill (`in_box_cf` high while `row_norm`
  climbs) is a live question again, not the expected outcome.
- Gate 3 (order): the EN order ceiling is 0.26 at 0.5 for three-word
  strings; the rows' 0.03 says the same as read 6. Open behind Gates 1–2.

### Low-σ grid (same day, `20260922-131213-2da903`, `cf_sense_en_piece_low/`)

EN two-word / three-word strings on `--cf_t 0.2,0.3,0.4,0.5,0.6`, same pairs:

| σ | id move | order move | id leak |
|---|---|---|---|
| 0.20 | +0.000 ± 0.001 | +0.005 ± 0.010 | 0.172 |
| 0.30 | +0.011 ± 0.013 | +0.085 ± 0.096 | 0.048 |
| 0.40 | +0.085 ± 0.086 | +0.180 ± 0.111 | 0.005 |
| 0.50 | +0.186 ± 0.063 | +0.238 ± 0.111 | 0.001 |
| 0.60 | +0.241 ± 0.069 | +0.209 ± 0.107 | 0.001 |

8. **The string band has a floor at σ ≈ 0.4**: below it the input pins the
   glyphs for the caption's identity (≤ 1 % at 0.2–0.3) and nearly so for
   order (≤ 9 %). Order wakes one step earlier than identity (0.18 vs 0.085
   at 0.4). A training band for strings is **0.4–0.7** (0.5–0.7 if identity
   alone is the target); 0.2–0.7 would spend 40 % of uniform draws where
   the rows have nothing to multiply.
