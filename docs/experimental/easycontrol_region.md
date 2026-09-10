# EasyControl · Region (paint-to-character)

Task. Cond = the scene with a solid gray paint region; the model regenerates
what is under the paint coherently with everything around it. The caption
steers identity (optionally with an `On the left, <name>` clause); the cond owns
the scene. Frozen DiT, per-block cond LoRA + `b_cond` gate — the standard
EasyControl stack ([`easycontrol.md`](easycontrol.md)).

Pipeline: `easycontrol_adapters/region/prep.py` (select → SAM3 masks → gate +
augment + paint → captions → VAE cond cache → TE cache), descriptor
`configs/easycontrol/region.toml`, bench `bench/region/run_bench.py`, mask
QA `easycontrol_adapters/region/contact_sheet.py`.

```
make easycontrol-staging    EASYADAPTER=region   # select · SAM · cond · captions  (GPU: SAM3)
python easycontrol_adapters/region/contact_sheet.py   # → {base}/contact_sheets/index.html
make easycontrol-preprocess EASYADAPTER=region   # VAE cond latents + TE variants
make easycontrol            EASYADAPTER=region --queue
make daemon-run ARGS="--label region-bench bench/region/run_bench.py --label v5"
```

## History (bench `bench/region/results/`)

| ver | data | mean_iou | iou_lift | girl_in_paint | bg_psnr | verdict |
|---|---|---|---|---|---|---|
| v1 `20260823-1156` | white canvas, black paint, b_cond −4 | 0.155 | 0.00 | 0.29 | — | inert (sparse cond + −4 gate never opened) |
| v2ep6 `20260823-2136` | white canvas, b_cond −2 | 0.327 | 0.17 | 0.45 | — | follows weakly |
| v3 `20260824-0712` | real image background, gray 128 | 0.553 | 0.50 | 0.99 | 31.7 | follows; scene survives |
| v4 `20260825-0732` (bg2) | v3 + cond_noise 0.01, caption_dropout 0.1 | 0.771 | 0.73 | 1.00 | 31.0 | follows well — but fills the paint |
| v5 `20260825-1338` | pairs + slack (one-sided) + head + positioned captions | 0.425 | — | 1.00 | 32.2 | harmonizes: area_ratio 0.46 tight / 0.38 slack (v4: 0.8 both), background continued inside the paint on every layout; centre corr 0.92/0.89, found 0.94 |

The EasyEdit zero-training hybrid was scored on the same plates (`20260824-2228`):
found 25% vs 88%, background washout — refuted for this task (small in-place
edits only).

## v4 → v5: why it filled the paint, and the three new slices

v4's failure was baked into the data, not the model. Every training paint was a
tight superset of the character mask (`_augment_mask` levels 0–3: guard
dilation + small blur / wobble), so the character filled ~90 %+ of every blob
and touched its boundary — the model learned *paint = silhouette to fill*, and
the bench agreed with it: `area_ratio ≈ iou` per layout (0.77 / 0.87 / 0.81),
while `bg_psnr` was measured only *outside* the paint. There was no metric for
"did the background continue inside the paint around the character".

v5 keeps one paint semantics — paint = region to regenerate; everything
outside is context — and adds three things:

1. Slack level (harmonize lever, `aug_weights[4]`). The paint is a loose
   region that contains the character off-centre with the real background
   as the target under the rest. Small characters get an area-targeted
   box/ellipse (`slack_grow` × bbox, growth distributed toward the sides with
   room, capped at `max_slack_coverage` of the image); a full-frame figure —
   the norm in this dataset, bboxes cover >60 % — gets a thin halo plus the
   silhouette shifted toward one random side, bounded to ≤ 1.6× her area
   (girl fills ~65–75 % of the paint; an all-round fat halo took 0.3-coverage
   girls to 0.7 paints — rejected 2026-08-25). No room at all → a tight level.
   Still a strict superset of the mask (leak invariant unchanged).
2. Pair slice (`pairs = true`): 1girl1boy images with only the girl
   painted (SAM3 `girl` minus `boy`), the boy left visible as real context —
   the strongest harmonization teacher (match a visible partner's lighting,
   scale, eye-line). Gates: the hole gate is loosened (`pair_max_hole_frac`,
   girl-minus-boy carves real holes) and ≥ `min_partner_visible` of the boy
   must stay outside the paint (`partner_occluded` drop otherwise).
3. Face level (concentrate lever, `aug_weights[5]`): paint only the girl's
   head — SAM3 `head` (face + hair; the bare `face` concept is inconsistent
   about the fringe) ∩ her dilated mask, largest blob; ≥ `min_face_frac` of
   the image, ≤ 70 % of her mask — body visible, the inpaint-subset case. No
   usable head → slack.

Captions. Each staged image gets `{stem}.variants.txt` with the resized
tree's flat rows verbatim plus a positioned copy of each (`… . On the
<pos>, <girl character name>.`) when the caption has no clauses yet and the
girl has exactly one character name (unnamed → flat only; a clause needs a
subject tag). `<pos>` = the girl's reading-order word — pairs through the
grammar's `assign_positions` (left/right), solo by bbox-centre side
(left / middle / right; the corpus never says "center"). Encoded into
`{base}/text` (`text_cache_dir`), so the shared LoRA TE cache is untouched.
The loader samples v0 20 % / the rest uniformly, so flat and positioned rows
are seen at roughly equal rates.

Bench additions. `slack_left` / `slack_wide` layouts (+ `large_center`) form
`SLACK_LAYOUTS`; new per-sample `inpaint_bg_psnr` = PSNR vs the plate over
`paint − dilate(girl)` (scored only when ≥ 2 % of the paint is not girl);
headline split `mean_area_ratio_slack` vs `mean_area_ratio_tight` and
`mean_inpaint_bg_psnr_slack` / `inpaint_bg_scored_frac_slack`. The v5 goal on
slack layouts is `area_ratio ≪ 1` with `inpaint_bg_psnr ≈ bg_psnr`; tight
layouts should keep v4's fill behaviour. Old metrics are unchanged, so v4 →
v5 stays comparable on the original six layouts.

### v5 read (2026-08-25)

`bench/region/results/20260825-1338-v5`: the character no longer fills the
paint — `area_ratio` fell from ~0.8 to 0.46 (tight) / 0.38 (slack) with
`girl_in_paint` still 1.00 and position tracking intact (centre corr
0.92 / 0.89, area corr 0.72), and the contact sheet shows desks / bushes /
street continued *inside* the ellipse around the girl on every layout. IoU
dropped 0.77 → 0.43 as a direct consequence (the girl is smaller than the
paint) — read IoU as "size follows paint", not as quality, from v5 on.
`inpaint_bg_psnr` came out ~13 dB on every layout even where the sheet shows
a clean continuation: the regenerated band is plausible but not pixel-aligned
with the plate, so PSNR measures the wrong thing there — a perceptual/edge
metric (LPIPS or a Canny-IoU against the plate) is the owed replacement.
Failure modes seen: 2/32 no girl found (top_left street, faint garden girl),
and on tight layouts the girl is now sometimes smaller than a user would
want (a tighter paint restores size; no inference knob yet).

## Real-image use (`bench/region/run_real.py`, `20260825-1407-v5-real`)

The plate bench scores placement on synthetic scenes. `run_real.py` runs the
user's actual flow: 8 safe staged images (1 pair + 7 named solo), the
training-style paints (tight = smooth SAM silhouette, slack, face/head) over
the real image, × prompt cases (`own` caption / `positioned` clause /
identity `swap` / `minimal` `safe, 1girl, solo` / `action` with pose tags
swapped) + an unpainted `control`; scored by SAM3 (paint metrics + PSNR vs the
real source) and the Anima Tagger (rating, `persona_hit`, `char_hit`,
`char_leak`). Sheets: `results/<run>/contact/index.html`.

| paint | girl_in_paint | area_ratio | iou_src | bg_psnr | persona_hit (swap) | char_leak |
|---|---|---|---|---|---|---|
| tight | 0.94 | 0.99 | 0.86 | 38.5 | 0.34 | 0.00 |
| slack | 0.95 | 0.75 | 0.69 | 37.5 | 0.47 | 0.00 |
| face | 0.29 (body visible — expected) | — | 0.97 | 36.0 | 0.25 | 0.14 |

Read: tight + slack work as a user would expect — the girl lands in the
paint (found 0.99), silhouette kept on tight (iou_src 0.86), the scene outside
survives (bg_psnr 37–40 dB against a real reference), and the swap changes
hair/eyes with the original character never re-identified by the tagger
(`char_leak` 0). On the pair image the boy is untouched and the swapped girl
matches his lighting/scale. `positioned` ≈ `own` (the clause is inert on a
single subject, as designed). Weak spots: (1) `minimal` on slack loses the
scene inside the paint (found 0.88, bg_psnr 29.7; the pair image hallucinated
a different figure) — with no identity in the prompt the paint region
free-runs, so the caption should carry at least outfit/hair; (2) face
paint leaks outside the paint: the swap case recoloured the skirt
(outside the head paint) on 2/8 images — `bg_psnr` cannot see it because it
excludes the girl region, so read the sheet; a girl-region-outside-paint PSNR
is owed; (3) persona/tag recall is low in absolute terms (0.34–0.47 / ~0.44)
but the `control` arm is 0.64 with the same tagger, so ~half of that is
tagger threshold, not the adapter. Safe-rated inputs stayed safe-rated
60 % / control 75 % — the swap case (0.25) drags it, mostly "sensitive"
flips on skirt/legs renders, not nsfw content.

## Gotchas

- **`make easycontrol-preprocess EASYADAPTER=region ARGS="--cond_overwrite"` after
  any repaint** — the cond cache is keyed `{stem}_{WxH}` and silently keeps a
  same-stem latent from an older paint recipe (bit us 2026-08-25: v4 solo
  latents survived the v5 repaint).
- `report.json` is now the per-image ledger (`records`: slice, level,
  position, boxes, coverages, `partner_visible`) — the captions stage and the
  contact sheets read it; an `--overwrite` cond run regenerates it.
- The SAM stage runs four passes scoped by the `select/{solo,pair}` symlink
  trees: pair-girl (girl − boy) first into `masks/`, then pair-boy,
  solo-girl, face. Existing masks are skipped, so the order is what keeps a
  pair's girl-minus-boy mask from being overwritten by a plain girl mask;
  `--sam_force` regenerates all.
- Staging via the daemon: `make daemon-run ARGS="--queue tasks.py
  easycontrol-staging" EASYADAPTER=region` (the env var reaches the job).
- `paint_color` in `bench/region/run_bench.py` and `contact_sheet.py` must
  match the descriptor (128 gray).
