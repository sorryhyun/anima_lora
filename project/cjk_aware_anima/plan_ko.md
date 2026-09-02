# CJK-aware Anima — Korean extension plan

> **Continued in [`plan_ko2.md`](plan_ko2.md)** (2026-09-01): the
> corpus-enhanced iteration — r5 KB re-arbitration, the `desc_ko` prose
> register, and the re-scoped G5. This file stays the record of K0–K3.

*Appends `ko` to the shipped v1 line ([`plan.md`](plan.md) Phase 3). Drafted
2026-08-30 from the on-disk state; nothing below is measured yet unless it
cites [`findings.md`](findings.md). Read that file and `datasets/README.md`
first — every JA lever that was closed is closed for KO too (MT-prompt
tuning, mT5, adapter LoRA, more synthetic pairs, JESC/STAIR-style prose).*

Premise: **the encoder does not change.** Korean is a *corpus* job plus one
joint retrain of the same pack. What carries over for free, checked on disk:

| Piece | KO status |
|---|---|
| `ext_vocab._CJK_RANGES` | hangul syllables / jamo / compat jamo already routed to the ext table |
| `bench/cjk_adapter/assets/ext_embed.*` | already holds **all 3,473 pure-hangul Qwen pieces + 8,933 hangul char rows** (anchor-init, never trained — risk 3 in `plan.md`: they are the "zero-shot ko rows") |
| `HybridT5Encoder` / `scripts/distill_cjk/` | language-agnostic; the `ja` field is just "student-side text" |
| `mt.py` (`MTEngine`) | `LANG_NAMES` has `ko`; `target_lang="ko"` is a parameter everywhere |
| `wikidata_lexicon.py` | `--langs ja ko zh` supported; the shipped asset was built with `langs=['ja']` only → **rebuild** |
| Danbooru wiki dump | **3,760** entries carry hangul `other_names` (vs 73,452 with kana) — small but it is exactly the community register (`twintails`→트윈테일, `maid`→메이드, `touhou`→동방) |
| `p1atdev/danbooru-ja-tag-pair` | JA-only; **no KO analog known** (verify on HF before assuming) → MT fills a larger share, so the KO review is bigger than JA's |
| Native prose source (D7 LoveHina analog) | **none** — the register-drift instrument has no KO counterpart; see risks |

Tokenisation reality (Qwen byte-BPE on KO): common syllables are whole
pieces (`이 다 하 는 …`), most content words split 1 piece/syllable
(`트윈테일` → 4, `세라복` → 3), and some syllables fall to UTF-8 fragments
(`머` in `검은 머리`) → the char-row fallback. So KO rows are *more* char-like
than JA, i.e. the same "rare names fall to char pieces" regime findings §1
already accepted; not a reason to touch the vocab.

## Phase K0 — size the gap (CPU + one daemon job, no data work)

**DONE 2026-08-31** → [`reports/0831_ko_phase0.md`](reports/0831_ko_phase0.md):
`ko_ext` cos 0.072 (inert, < 0.3 → full K1 corpus), disc healthy (0.146 —
alignment failure, not collapse); teacher 0.786 in the JA band; coverage
100 % v=0; risk-2 spacing probe cleared.

Same shape as the 2026-08-15 JA probe, on the *current* `synthja` pack:

1. Author `assets/ko_eval_prompts.json` — the 21 JA ids (`t1_tags_school` …
   `n*`, `c*`) with the KO wording a Korean user would type (Arca Live
   register, not textbook Korean: `1girl` → `소녀 1명`/`1girl` — decide per
   row and record it; `looking at viewer` → `카메라 응시`; `twintails` →
   `트윈테일`). `en` stays the faithful teacher-side translation.
2. `run_bench.py --prompts ko_eval_prompts.json` arms `ko_unk` (stock T5),
   `ko_ext` (synthja pack, zero-shot KO rows), `ko_t5en` (teacher).
   Expected from findings §1: `ko_ext` ≈ 0.05 cos / disc ≈ 0.9 — i.e. the
   hangul rows are inert until trained. If it is *already* ≥ 0.3 the
   shared `param=global` correction transfers across scripts and K2 can
   start from a smaller corpus.
3. `gates/coverage.py --prompts ko_eval_prompts.json --pairs pairs_synth.jsonl`
   — should report every KO content row at 0 visits; this is the baseline the
   K1 corpus must move.

Output: `reports/09xx_ko_phase0.md` (one table). No go/no-go here — it only
fixes the yardstick.

## Phase K1 — corpus (CPU except the two `--mt` passes)

**DONE 2026-08-31** → [`reports/0831_ko_phase_k1.md`](reports/0831_ko_phase_k1.md):
62,494 pairs (tags/tags_alt/names 50,190 + names_synth_ko 12,304), glossary
99.66% occ coverage, gate green on every t*/q*/n*/c* prompt except t5's 똑 —
resolved in K1.5 round 2 (커플룩 on both sides). Plan deviation that helped:
the KR KB (`models/danbooru_tags_classified.csv`) is the KO tag-pair analog —
83.6% of general types — wired as `src: "kb"`.

Mirror the JA builders; do not fork the pipeline, parameterise it. Every
step writes `assets/*_ko.*` next to the JA asset.

1. **Lexicon**: `wikidata_lexicon.py --langs ja ko` → `wikidata_lexicon.json`
   gains `ko` labels (KO transliterations of JA names: 하쿠레이 레이무). Keep
   the `≥2-token + Q95074` guard. JA output must be byte-identical for the
   `ja` keys (diff the asset).
2. **Glossary** `tag_glossary.py --lang ko` → `assets/tag_glossary_ko.json` +
   `tag_glossary_review_ko.md`. Language-specific pieces to add, in the
   existing priority order (`overrides → artist passthrough → rating →
   lexicon → wiki other_names → MT`):
   - `is_korean(s)`: any hangul syllable. No Shift-JIS / kanji-inventory
     trick is needed — hangul is unambiguous. Han-only `other_names` are
     **not** Korean (hanja is not what users type).
   - Back-translation scoring unchanged (KO→EN via Hy-MT2-7B, token F1,
     `--accept-f1 0.75`, ties → shorter).
   - Hand-checked KO exemplar list for the MT prompt in `mt.py` (the JA rule
     "never draw exemplars from the unverified wiki head" applies verbatim).
   - New `datasets/tag_overrides_ko.json` (committed) for the review fixes.
   - **`--mt` is mandatory on the rebuild** (same trap as JA: the CPU path
     silently drops the MT tier). Runs through the daemon with
     `--gpu-budget 13GiB --max-new-tokens 32`.
   Expect `mt_unverified` well above JA's 36.6% (no tag-pair source). That is
   what the review is for; G4b says noisy spans still beat none, so ship
   with the review done on the top-N by occurrence, not on everything.
3. **Review sign-off** (human) — same two axes as JA (polysemy `bow`; and the
   KO-specific one: **loanword vs native** — `armor` → 갑옷 vs 아머,
   `cape` → 망토, `sailor uniform` → 세라복 vs 세일러복). Fixes →
   `tag_overrides_ko.json` → `--reselect`. Same rule as JA: re-cache only
   rows whose wording changed.
4. **Pairs** `build_pairs.py --lang ko --glossary tag_glossary_ko.json`:
   registers `tags_ko`, `tags_alt_ko`, `names_ko`; joiner `", "` only
   (`pick_joiner` already records the joiner per pair since 2026-08-30 —
   for KO pass an rng-free/`ALT_JOINER_FRAC=0` path; nobody types `、` in
   Korean). The axis fallback (`resolve_axis`: index → wiki category →
   artist-OC) applies unchanged — KO names must never reach MT either.
   Field name stays `ja` (it is "student text"; renaming touches every
   consumer for no gain) but every KO record carries `"lang": "ko"`.
   D2 commentary and D6 quotes: **skip** for KO (no native source; the JA
   quote registers are eval-only anyway).
5. **Synthetic names** `synth_names.py --lang ko --context ko` →
   `names_synth_ko`, same rarity-weighted allocation to the visit floor, but
   **capped** (see the cache budget below). Names pinned from the `ko` lexicon
   labels; captions with no resolvable KO name are skipped, as for JA.
6. `gates/coverage.py --prompts ko_eval_prompts.json` on the merged pairs:
   no user-facing KO tag token under floor. If a t* prompt still has a 0-visit
   row, it goes into a KO instance of plan.md §5a (targeted tag widening)
   before training, not after.

## Phase K1.5 — user inspection of the corpus (human; added 2026-08-31)

**DONE 2026-08-31** — three rounds. Round 1: exemplars, rating band, 14
overrides. Round 2 (user audit over the review file + spotcheck): 16 semantic
overrides + all 40 `* thighhighs` variants onto the 니삭스 exemplar —
`tag_overrides_ko.json` now 71 entries / 9.6% of occurrences; the t5
똑같은 옷 decision landed as **커플룩** on both the glossary and the eval
prompt. `--reselect` + pairs/synth rebuild (CPU); gate fully green on every
t*/q*/n*/c* prompt (s*/q3 prose rows expected-open as in JA). Details in
[`reports/0831_ko_phase_k2.md`](reports/0831_ko_phase_k2.md) §K1.5.

**Round 3 (2026-08-31)** →
[`reports/0831_ko_glossary_audit_r3.md`](reports/0831_ko_glossary_audit_r3.md).
The disagreement table is per-tag and stops at n=454, so two surfaces survived
it: **wording collisions** (two EN tags handed one KO wording — 336 raw groups
/ 15% of occurrences) and the **`mt_unverified` tail below the floor** (4,416
tags / 82,502 occ, errors starting immediately: `paizuri`→포즈,
`cuffs`→손목찌개, `yuri`→유리). Both are now generated by
`datasets/audit_glossary.py`. Overrides **92 → 345** (16.1% of occurrences,
including a round-4 pass of user vetoes and the MT-eaten emoticon tags —
`:t`/`:s`/`:i` all read `:소녀 1명`), collision groups pending **93 → 3**,
gate re-run unchanged and green. Merges
the user keeps (bow/ribbon) live in `collisions_accepted_ko.json` so they are
not re-raised each round. ~~**Owed:** 65.8% of pairs changed student text and the
stager is positionally keyed, so `cache_ko` must be re-staged in full and
`synthjako` retrained before K3 speaks about the shipped wording.~~
**Cleared same evening** → [`reports/0901_ko_phase_k3.md`](reports/0901_ko_phase_k3.md):
pairs rebuilt, `cache_ko` fully re-staged, `2c-synthjako-v2` retrained
(21:04–21:58); the on-disk `cjk_vocab_pack_synthjako` is the v2 and matches
the committed wording.

After K1 completes, before any GPU is spent on K2 staging/training, the user
inspects the corpus and signs off:

1. **Glossary review round 2** — `tag_glossary_review_ko.md` (the MT-arbitrated
   top-200 disagreements + whatever round 1 — 2026-08-31: exemplars, rating
   band, 14 overrides — did not cover).
2. **Corpus spot-check** — `spotcheck_ko.md` (~200 sampled pairs, EN vs KO,
   including `names_synth_ko` samples).
3. **Coverage gate output** — the K1 step-6 table: every eval-prompt token
   over floor, or the §5a widening list that will fix it.

Fixes loop back cheaply while nothing is trained yet: `tag_overrides_ko.json`
→ `--reselect` (CPU) → rebuild pairs. **K2 runs only after this sign-off.**
(The post-train render-grid eyeball stays where it always was — K3's grids.)

## Phase K2 — cache + joint retrain (GPU)

**DONE 2026-08-31** → [`reports/0831_ko_phase_k2.md`](reports/0831_ko_phase_k2.md):
`cache_ko` staged separately (61,994+500 pairs, 44 GB) exactly per the
preferred shape below — `distill --cache_dir` grew the comma-list reader
(`CachedPairs` concatenates staged dirs; JA never re-encoded). Joint retrain
`2c-synthjako-v1` → `output/ckpt/cjk_vocab_pack_synthjako`: v4 recipe +
`--init_pack cjk_vocab_pack_synthja_v4`, KO registers sampled 0.55 → ≈28% of
a batch, `--eval_limit 1000` (holdout concat is JA-then-KO; the first-N eval
slice would otherwise score JA only), 45 GPU-min. Holdout: tags_ko attn
0.46–0.47 (0.061 zero-shot at K0), names_ko 0.955; JA registers near the v4
band (names recovery 0.879→0.780 is the watch item, partly sample
composition).

**v2 retrain 2026-08-31 (post-r3/r4 wording)** →
[`reports/0901_ko_phase_k3.md`](reports/0901_ko_phase_k3.md): same recipe on
the rebuilt corpus (`2c-synthjako-v2`, 289,209 pairs). JA holdout unchanged
(names 0.894 stable — the recovery-dip watch did not worsen); tags_ko attn
0.463→0.370 under the reworded holdout is the new **watch item** (grid shows
no regression; re-read after G2/G5).

**Cache budget.** `cache_synth2` is ~171 GB for 262,852 pairs (~0.66 MB/pair).
After the 2026-08-30 clean-up (ConceptEdit tar, `easycontrol/{phash_edit,
subject_edit}`, `ltxmodels` removed) the volume has ~150 GB free with the
JA cache in place, so a KO corpus at the full JA recipe (~150 GB) *just*
fits but leaves no headroom. Preferred shape regardless:

- Stage KO into a **separate** cache dir (`cache_ko`) — the stager is
  pair-keyed and `distill.py` takes one `--pairs`/`--cache_dir`; add a
  `--cache_dir` list (or a merged manifest) rather than re-staging JA. JA
  rows are untouched, so nothing already cached is re-encoded.
- **Start KO at ≈ 60k pairs** (~40 GB): `tags_ko` 17k + `tags_alt_ko` 17k +
  `names_ko` ≈ 16k + `names_synth_ko` ≈ 13k; grow `names_synth_ko` only if
  the K3 name gate asks for it. The JA `names_synth_ja` register is 176k of
  the 263k and buys visits, not vocabulary (findings: "more captions multiply
  the same glossary"), so if disk binds, **trim `names_synth_ja` first**.
- If more room is genuinely needed, the lever is a `--max_pairs`-style
  per-register cap at stage time, not a lower-precision cache.

Train **one pack, jointly**: the settled recipe (`--loss span --steps 12000
--batch_size 32 --param global --trust provenance`, register sampling passed
explicitly) with the KO registers added to `--train_registers` and sampled so
KO ≈ 25–30 % of a batch (`--register_sampling`), warm-started from the
signed-off `synthja` weights. Label `synthjako`. ~20–30 GPU-min.

Why joint and not a second pack: `param=global` is a shared low-rank + diag +
gain over *all* ext rows — a KO-only pack trained from scratch would move the
JA rows too, and two packs cannot be loaded at once. That same sharing is the
risk in the other direction (KO training drifting JA), which is the first
gate below.

## Phase K3 — gates

Ordered; the first two are kill criteria for the *joint* pack.

1. **G1** EN bit-exact — unit test, unchanged. **GREEN 2026-08-31.**
2. **JA non-regression** — the 2c grid `ja_ext` vs the `synthja` render at
   the same seed: every t*/q*/m* prompt visually unchanged; per-register
   `cos_student_vs_en_attn` within the `synthja` band on the JA holdout.
   *Fail → ship `synthja` for JA and fall back to a KO-weighted sampling
   sweep (0.1 / 0.2); if JA still moves, KO ships as a separate pack behind a
   language switch and the plan records that `global` does not share across
   scripts.*
   **Metric level GREEN 2026-08-31 for v1** — both same-seed grids
   (`20260831-1827/-1843`) within ±0.013 of the v4 twins, discrimination
   unchanged; `global` does share across scripts. **Shipped v2 pack: metric
   level GREEN 2026-09-01** (the 22:01 grid jobs died with the daemon; re-run
   as `20260901-0809/-0825`): mixed grid fully in band, main grid in band on
   every t*/n*/m* prompt with two marginal q* flat-cos deltas (q1 −0.015 /
   q2 −0.016), discrimination unchanged. **Render level GREEN 2026-09-01** —
   eyeball + 3-seed probe on the flagged prompts (`n2q2-probe-*`): the
   seed-42 n2/q2 anomalies are chaos flips present in v4 at other seeds, not
   v2 drift; core tags stable v1→v2. See `reports/0901_ko_phase_k3.md`.
3. **KO recovery** — `ko_ext ≈ ko_t5en` on the rendered grid for t*/c*
   prompts; per-register readout at the JA `tags` band (teacher ceiling
   0.823 is the same teacher). `n*` full-KO names are **expected fails**
   (same as JA v1, plan.md Phase 5b territory).
   **Metric level 2026-08-31**: holdout attn tags_ko 0.46–0.47 (v1) / 0.370
   (v2, reworded holdout — watch item), under the JA tags band (0.52–0.56);
   v2 grid (`20260831-2201-ko-k3-recovery-grid-v2`) `ko_ext` ≈ v1 on every
   prompt, off the K0 floor on every t* (flat cos is not the recovery
   instrument — the shipped JA pack sits at the same flat level).
   **GREEN 2026-09-01 — render eyeball signed off (user) on the v2 grid.**
   Independent render read (Claude, same day, in `reports/0901_ko_phase_k3.md`):
   recovery real but uneven at the mid-frequency tier — KO-specific binding
   misses 흑백/그레이스케일 (JA binds monochrome fine) and 쌍둥이 (weak);
   t3's armor miss is shared with JA (ext ceiling, not KO). These go to the
   §5a widening / `mt_unverified`-tail list, alongside the G5 result.
4. **Coverage** — no KO tag token under floor; far-disc ≤ 0.2.
   **GREEN 2026-08-31** (K1.5 gate table; disc_far 0.087 train / 0.144 grid).
5. **Register drift** — **OWED before ship.** No D7 analog exists for KO.
   Substitute: hold out 300
   *hand-typed* Arca-Live-style KO prompts (not composed from the glossary)
   and report their readout vs the composed holdout; a gap > the JA D7 gap is
   the signal. Owed before ship, cheap (CPU + one eval job).

## Phase K4 — ship

Rides `plan.md` Phase 3 steps 3–5 unchanged: same sidecar, same auto-discovery,
the same `cjk_vocab_pack.md` under `docs/methods/` (add a KO section + the KO TE-cache
regeneration note), same ComfyUI node — the pack file just grows its trained
rows. Guidebook line goes through the `translator` agent for `가이드북.md`.

## Not planned

- A KO-specific vocab/tokenizer, jamo decomposition, or any change to
  `ext_vocab.py` — the rows exist.
- KO quoted-text (D6) or in-image hangul rendering — that is the glyph line
  (`plan.md` Phase 4), which stays deferred for JA first.
- zh — same recipe again once KO has shipped; the wiki has far more zh
  `other_names` than ko, so it is the easier third language, not the second.
- Any adapter-LoRA / rank / attn arm for KO — plan3 closed these with no
  ranking instrument, language does not reopen them.

## Risks specific to KO

1. **Loanword-vs-native register** is the KO polysemy axis and MT chooses
   inconsistently; only the review fixes it. Budget the review time, it is
   larger than JA's.
2. **Spacing**: `segment_runs` folds whitespace into CJK runs and pure-hangul
   pieces with a leading space exist (`' 이'`), but K0 must confirm a
   space-separated KO tag (`검은 머리`) encodes to the same rows whether or
   not it follows a comma — otherwise `tags_ko` trains rows the user never
   hits.
3. **No native KO prose** for drift detection (gate 5's substitute is weaker
   than D7).
4. **Cache disk** — 50 GB free; the K2 cap is not optional.
