# plan_zh2 U5 — symbol register `tags_sym` (2026-09-04)

*Follows `reports/0903_coverage_arms.md`. Same `synthjakozh1_r256` recipe
(`param=global`, rank 256, span loss, `--trust provenance`, 12k steps, bs 32,
lr 1e-3, `--holdout_rows 0.05`) on the four restaged caches **plus**
`cache_sym`; the base arm is `u4base_r256` (identical recipe, no symbol
register). Code: `datasets/build_pairs_sym.py`,
`datasets/assets/tag_glossary_sym.json`, `assets/sym_eval_prompts.json`,
`probes/sym_grid_judge.py`.*

## Premise check — the corpus has (almost) no symbol tags

plan_zh2 U5 assumed "~40 tags with > 20 occurrences in `image_dataset`". A
census over the EN side of the three training corpora (11,869 images) with the
symbol-routed encoder finds **ten** segments that touch a symbol row at n ≥ 3,
and none above 20:

| tag | n | note |
|---|---|---|
| `^^^` | 13 | real Danbooru tag |
| `^ ^` | 4 | gelbooru surface of `^_^` |
| `kaguya-sama wa kokurasetai ~…~` | 6 | copyright name, `~` is the only symbol |
| `:<=` | 3 | corrupted `:<` |
| others | 3–4 | artist commentary (`W600mm×H350mm`, `✨`, `<<<`) |

(`image_dataset` alone: 5 tags, `^^^` 34 / `^ ^` 14.) The `<unk>` mass
findings §11 counted in the zh cache was the name separator `·` and title
`~`, which the plan already excludes. So the register cannot be "the tag's
wiki definition over its corpus occurrences" — it is **minted end to end**:
every glossary entry gets 300 template captions regardless of corpus count.
Nothing in the recipe depends on the premise; the gate does not either.

## Teacher list — verified against the wiki, six drafts corrected

A general-purpose agent checked every candidate against the
`dartags/danbooru-wiki-2609` wiki dump (2026-09-02; live Danbooru refused the
host) and Sept-2025 tag counts. Symbol-route characters are only `^ < \ ~ \`
{ }` plus non-ASCII (`>`, `|`, `_`, `:` are T5-spellable), so the reachable
Danbooru tag set is short — 12 canonical tags ≥ 800 posts, all in the
glossary except the Kaguya-sama copyright. Corrections that matter:

- `^^^` is a **crown-shaped realization mark beside the head** (58k), not
  "surprise lines above the head" — that is `notice lines` (`\|/`, 38k),
  which the plan's wording would have taught.
- `\||/` (6.3k) and `\m/` (8.6k) are **finger gestures** (middle+ring together;
  the ILY sign), not raised arms / rock horns — rock horns are `\n/` (1.7k).
- `\(^o^)/` tags the **kaomoji appearing as text** in the image (121), not a
  pose — dropped. `^o^` and `>o<` have 0 posts — dropped. `^ ^` is not a
  Danbooru tag (`^^` aliases to `^_^`) but is what gelbooru captions carry, so
  it stays as a second surface of `^_^` (different ext rows: `^ ^` → `^`,` ^`;
  `^_^` → `^`,`^`).

Final: 21 surfaces / 15 Danbooru tags — 15 expression/gesture entries
(`^^^ ^_^ ^ ^ ^^ ^q^ :< ;< >_< <o>_<o> <|>_<|> \||/ \o/ \m/ double \m/ \n/`)
and 6 decorative glyphs (`☆ ★ → star (symbol)`, `♪ → musical note`, `♡ ♥ →
heart`, `× → x (symbol)`).

Row sharing worth knowing: `:<`, `;<`, `>_<`, `<o>_<o>`, `<|>_<|>` all touch
**only the `<` row** (58968) on the ext side — the `:` / `;` / `>` / `_` / `o`
around it are T5 tokens. Likewise `\||/ \o/ \m/ \n/` share the `\` row and
`^_^ ^q^` share `^`,`^`. Their meanings are separated by the Qwen-side
context the adapter attends over, not by rows; the ext row's job is to stop
the whole surface collapsing into one `<unk>` query.

## Data

`pairs_sym.jsonl`: 6,300 pairs (21 × 300), template = a random
`image_dataset` caption with ≥ 6 segments, one random non-leading general
slot replaced — teacher side the clause, student side the symbol
(`via: wiki_verified`). Context: 3,113 EN-pinned (both sides EN except the
symbol — the `names`-register exactness argument) / 1,092 ja / 1,048 ko /
1,047 zh composed through the glossaries. `cache_sym`: 5,800 train / 500
holdout, 708k span tokens. Ids `SYM/<template image>/<tag>/<k>` so the
image-grouped holdout spreads each symbol over both splits.

## Arms

| arm | pack | train registers |
|---|---|---|
| base | `u4base_r256` | the 14 of 0903 |
| U5 | `u5sym_r256` | + `tags_sym` |

Distill readouts: `cos_student_vs_en_by_register[tags_sym]` on the 500-pair
sym holdout, `eval.row_holdout.gap_cos` (must stay comparable with base), the
standard holdout unchanged.

## Gate — 20-prompt symbol grid

`assets/sym_eval_prompts.json`: 4 blocks × 5 prompts (`^^^`, `:<`, `^ ^`/`^_^`,
`☆`), `en` = teacher clause in the symbol's slot, `ja`/`zh` = symbol verbatim.
`run_bench.py --arms en,ja_ext,zh_ext`, seeds 42 / 7 / 1234, both packs →
60 cells per pack per seed. Judge: the Anima Tagger's dbv4 backbone
(`caformer_b36.dbv4-full`) probability of the block's tag (`^^^` 55k, `:<`
43k, `^_^` 119k posts in its card); `☆` has no card tag (`star (symbol)` is not
in dbv4-full) and is read by eye. Pass = `ja_ext`/`zh_ext` hit rate under the
U5 pack ≥ the `en` arm's.

## Results (interim — 3 base seeds, 1 U5 seed; 2026-09-04 12:45)

### Distill metrics: unchanged where it should be, sym readout missing

`u5sym_r256` vs `u4base_r256` (`bench/cjk_distill/results/20260904-1040-u5-sym-r256`):
every standard number moves ≤ 0.01 (`cos_student_vs_en_attn` 0.756 → 0.747,
`discrimination_near_attn` 0.839 → 0.851, held-out span loss 0.109 → 0.109);
`rows_visited` 9,863 → 9,864 (the symbol rows are ~1 row-equivalent).
Row holdout: `held.cos` 0.587 → 0.572, `held.top1` 0.458 → 0.408,
`gap_cos` 0.289 → 0.307 — a −0.016 / −0.05 move on a single seed with no
noise floor measured for this metric yet; noted, not read.
`cos_student_vs_en_by_register` has **no `tags_sym` row**: `--eval_limit 1200`
scores a prefix of the pooled holdout and `cache_sym` is the last cache. The
two queued `--steps 0 --init_pack` runs (`u5-symeval-u4base` /
`u5-symeval-u5sym`, jobs `20260904-114017-0199f1` / `-252916`) score the 500
sym holdout pairs under each pack — read `metrics.eval.cos_student_vs_en_by_register.tags_sym`
and `attn_by_register.tags_sym` from their `result.json` when they land.

### Gate grid — dbv4 judge (mean prob / hit rate @ card threshold, 5 cells each)

Base pack, three seeds (`u5-grid-u4base-s{42,7,1234}`):

| block | arm | s42 | s7 | s1234 |
|---|---|---|---|---|
| `^^^` | en (clause) | 0.13 / 0.40 | 0.01 / 0.00 | 0.18 / 0.40 |
| `^^^` | ja_ext | 0.46 / 0.80 | 0.41 / 0.60 | 0.49 / 1.00 |
| `^^^` | zh_ext | 0.62 / 1.00 | 0.55 / 1.00 | 0.43 / 1.00 |
| `:<` | en (clause) | 0.11 / 0.20 | 0.07 / 0.20 | 0.33 / 0.40 |
| `:<` | ja_ext | 0.29 / 0.40 | 0.06 / 0.00 | 0.30 / 0.40 |
| `:<` | zh_ext | 0.38 / 0.60 | 0.35 / 0.60 | 0.30 / 0.40 |
| `^_^` | en (clause) | 0.35 / 0.60 | 0.48 / 1.00 | 0.26 / 0.60 |
| `^_^` | ja_ext | 0.11 / 0.20 | 0.01 / 0.00 | 0.01 / 0.00 |
| `^_^` | zh_ext | 0.01 / 0.00 | 0.00 / 0.00 | 0.01 / 0.00 |

U5 pack vs base, seed 42 (`u5-grid-u5sym-s42`; the `en` arm is pack-independent):

| block | arm | base s42 | U5 s42 |
|---|---|---|---|
| `^^^` | ja_ext | 0.46 / 0.80 | 0.59 / 0.80 |
| `^^^` | zh_ext | 0.62 / 1.00 | **0.34 / 0.60** |
| `:<` | ja_ext | 0.29 / 0.40 | **0.01 / 0.00** |
| `:<` | zh_ext | 0.38 / 0.60 | **0.05 / 0.00** |
| `^_^` | ja_ext | 0.11 / 0.20 | 0.06 / 0.00 |
| `^_^` | zh_ext | 0.01 / 0.00 | 0.04 / 0.00 |

`☆` (by eye, seed 42): the EN `star (symbol)` arm draws star glyphs in 4–5/5
cells; base `ja_ext`/`zh_ext` in 1–2/5; U5 in 1–2/5 (sparkles, no glyphs) —
no change.

### Reading

1. **The gate's premise is inverted for the symbol that mattered most.** Under
   the *base* pack, the JA/ZH prompt carrying `^^^` renders the mark at
   0.6–1.0 hit rate while the EN prompt carrying the wiki clause manages
   0.0–0.4. The description is a *weaker* conditioning than Qwen's own reading
   of `^^^` in a tag context — the student's content comes from the Qwen
   hidden states, and Qwen knows what `^^^` means; the ext row's only job was
   to stop the T5-side query collapsing into `<unk>`, which the symbol block
   already does. So "≥ the EN-description rate" was met before U5 ran, and
   distilling toward the clause can only pull the symbol *down* toward it.
2. **That is what U5 did.** Seed 42: `:<` goes from 0.4/0.6 to 0/0 in both
   languages, `^^^` zh from 1.0 to 0.6. The cells still frown (montage
   `cmp_s42_s2.png`) — as a generic frown, not the beak-shaped `:<` mouth the
   dbv4 tag denotes. The clause "small v-shaped closed-mouth frown" is
   semantically right and iconically wrong; the symbol-as-token carried the
   exact shape and the paraphrase loses it.
3. **Where the symbol genuinely fails under base (`^_^` ~0, `☆` 1–2/5), U5 did
   not fix it.** Two candidate reasons, not separated: the 300 minted
   pairs/symbol are ~560 span visits at 12k steps, near the 300 render floor
   §5a measured for tags; and for `☆` the clause `star (symbol)` *does* render
   in the EN arm, so the row simply did not move far enough toward it.

**Verdict (interim, 1 U5 seed):** the wiki-clause teacher is the wrong
teacher for expression symbols — it is weaker than the native reading it
replaces. U5 as specified does not pass its gate and hurts the two symbols
that already worked. Seeds 7 / 1234 decide whether the `:<` collapse is
seed-stable (a 0.4→0.0 hit-rate swing on 5 cells is one or two images); the
sym-holdout evals say whether the pack learned the clause at all (if
`tags_sym` cos is high on U5 and low on base, the loss did its job and the
teacher is the problem; if both are low, the register is under-visited and
point 3 is capacity).

**What would be worth trying instead** (not run): a teacher that *keeps* the
native reading — the EN caption with the symbol's **Danbooru tag name as
Qwen reads it** is not available on the T5 teacher side (`<unk>`), so the
candidate is self-distillation: teacher = the *student's own* adapter output
on the EN-context pair *without* the symbol row masked, i.e. anchor the row
to what Qwen already says, only for `^_^`/`☆`-class symbols where base fails.
Or accept the finding: the symbol block alone (findings §11) is the
deliverable for `^^^`/`:<`, and `^_^`/`☆` want more *data* (real captions
carrying them), not a paraphrase teacher.

### To finish the table

```
# after jobs 8da0c2 / 9bb421 (grids) and 0199f1 / 252916 (evals) are done:
R=bench/cjk_adapter/results
.venv/bin/python project/cjk_aware_anima/probes/sym_grid_judge.py --device cpu \
  --prompts project/cjk_aware_anima/assets/sym_eval_prompts.json \
  --runs $R/*u5-grid-u4base-s42 $R/*u5-grid-u4base-s7 $R/*u5-grid-u4base-s1234 \
         $R/*u5-grid-u5sym-s42 $R/*u5-grid-u5sym-s7 $R/*u5-grid-u5sym-s1234 \
  --labels b42 b7 b1234 u42 u7 u1234 --out project/cjk_aware_anima/reports/u5_sym_grid_judge.json
# sym-holdout cos per pack: bench/cjk_distill/results/*u5-symeval-*/result.json
#   → metrics.eval.cos_student_vs_en_by_register.tags_sym, attn_by_register.tags_sym
```
