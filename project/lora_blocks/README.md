# lora_blocks — precise MOSS i2i edits from reusable parts

Opened 2026-11-07 out of `project/moss_te_sae` (the copy-head and reference-edge probes).
An i2i edit splits into **copy** (a trained protocol, reused across tasks), **gate** (where
the copy holds: a bias on the main-query → reference-key edges, set from a mask, no
parameters) and **draw** (a trained block per kind of content). Tasks are assembled from
these and fine-tuned briefly, instead of one curated pair set and one long run per task.

- [proposal.md](proposal.md) — the blocks, composition, phases P1–P5 and their gates, the code
  it needs.
- Evidence: [`../moss_te_sae/reports/copy_protocol.md`](../moss_te_sae/reports/copy_protocol.md)
  (the copy is a protocol, not a head; the edge gate), [`../moss_te_sae/reports/i2i_copy_head.md`](../moss_te_sae/reports/i2i_copy_head.md).
- [plan.md](plan.md) — the order of work: Korean strokes first (per-glyph read, colorize copy in the text boxes, C, L).
- [opinion.md](opinion.md) — a reading of the idea against the literature; open design points.
- [findings/](findings/) — conclusions and conjectures above the reports:
  [copy.md](findings/copy.md) (a copy block saves pairs and the 200–300-step phase change, not steps of a long run),
  [gate.md](findings/gate.md) (G fixes "where" on shipped LoRAs; `diag` inside for removal, `boost` outside for lettering, `hard` invents).
- [reports/copy_vs_step.md](reports/copy_vs_step.md) — the copy head forms at steps ~175–225 of any pair set, after the pixels already copy; training past it buys breadth.
- [reports/p0_gate.md](reports/p0_gate.md) — P0: the gate fixes both shipped LoRAs' "where" failures with no training; `diag1` inside AnimeText boxes removes the watermark sanitize v3 keeps, `boost` outside the layout boxes stops swap_text rewriting unasked regions; `hard` invents content.
- [reports/colorize_inboost.md](reports/colorize_inboost.md) — a copy boost inside colorize v5's text boxes makes strokes heavier and leaves every deformed glyph as it was; the gate cannot fix lettering.
- [reports/hangul_s500.md](reports/hangul_s500.md) — step 3: a Hangul block from C on 600 synthetic font-lettered pages (500 steps) letters thinner than ksplit e2 in the synthetic layout and garbles uncaptioned Japanese; the exact concat on e2 doubles / blurs strokes.
- [reports/hangul_v2_s500.md](reports/hangul_v2_s500.md) — step 3, v2 data (balloons cleaned on JA raw / KO edition, the edition's line layout, bold faces, partial captions): the block alone letters both Korean eval pages right in a bold clean gothic and keeps uncaptioned Japanese; e2 is already crisp there.
- [reports/hangul_ko10.md](reports/hangul_ko10.md) — ten held-out Korean pages (`inputs_ko10/`): v2 alone misspells far more than ksplit e2 (short vowel strokes lost, lines duplicated); the exact concat v2 + e2 fails on every page; e2 breaks on long dense narration; byte-split syllables seen once are drawn right, the unseen `끜` comes out `끔`.
- [reports/hangul_v3_ft.md](reports/hangul_v3_ft.md) — v3 data (flat narration boxes, Regular faces, tail syllables) mixed with e2's pairs, 600 steps from e2: removes e2's pseudo-glyphs on one long narration page, adds garbles and vowel errors elsewhere; not a win.
- [reports/p1_c_fill.md](reports/p1_c_fill.md) — P1: C (400 steps, clean illustrations, text-layout holes under `ref_gate cut`) removes the sanitize eval pages' stamps and watermark under `cut` with no sanitize pair; the copy head forms by step 200.
- Probes: `project/moss_te_sae/src/{copy_head_ext,copy_transplant,ref_edge_mask,ref_gate,copy_vs_step}.py`; data `src/build_c_data.py`.

- [reports/p2_sanitize_ft.md](reports/p2_sanitize_ft.md) — P2: 300 steps of sanitize from C remove text on the hard manga pages as v3 does, from kaiming they leave blocks; neither holds the tone (EMA off: ΔL swings 2–4 between saves in both arms); `cut` on a fine-tune invents on large masks.
- [reports/c_fill_v2.md](reports/c_fill_v2.md) — C v2 (tone, EMA, B&W colorize pairs, 800 steps): copies as C and fixes its darkening on one page, but letters the large masks under `cut` and brightens three pages ~1 L; the tone term runs at 2–3× the FM gradient and swings with σ.
- [reports/swap_dense_copy_init.md](reports/swap_dense_copy_init.md) — swap_text on 1,509 dense re-crawled Korean pages, 500 steps from C v2 against kaiming: copy init does no harm (slightly better outside the text, slightly worse lettering); both fix e2's long narration and lose short-dialogue syllables.
- [reports/swap_dense2516_main.md](reports/swap_dense2516_main.md) — the main run, shipped as swap_text v3: the 2,516 dense Korean pages + en400 from C v2, 2 epochs; ko10 ~11 wrong syllables against ksplit e2's ~15 and no dropped line, the short dialogue back; English exact on 11 of 12 renders.
- [reports/c_fill_mod16_bw10.md](reports/c_fill_mod16_bw10.md) — C with `mod_rank` 16 and no tone term: paired loss 5–14 % below C v2, the tone term was C v2's late gradient swing, and the adapter learns the reference-side modulation the edit tasks share; B&W cut to 10 % (bw10) gives the best copy outside the mask and Δb near 0, but the page stays +0.5–1.1 L bright and `cut` still letters the large masks and now lets the original text back.
- [reports/t2i_rows_ablate.md](reports/t2i_rows_ablate.md) — a t2i style LoRA cut at inference: the delta on the image rows alone renders as the full LoRA, on the text rows alone as no LoRA; blocks 26–41 alone keep the style and the content changes with it; cutting the text queries off the image keys costs Base attributes and does not change what the LoRA does.
- [reports/offset_kv_copy.md](reports/offset_kv_copy.md) — the offset + kv copy block: Base's early positional own-cell heads vanish under `ref_pos offset`, so the copy is the late lookup alone (same b39 h19 / b40 h20 circuit plus b41 h6), forms ~10× later and half as broad; its LoRA norm is 10× the aligned C's, so P2's sanitize ft (lr 5e-5, 300 steps) moves it 10 % and keeps its failures (stamp and lettering return, reframes).
- [request.md](request.md) — request: isolate the text stream as the reference is (cross / intra read), soft tokens for what it loses. Only the render half of its first step is done, on t2i ([reports/t2i_rows_ablate.md](reports/t2i_rows_ablate.md)).

State: P0, P1 and P2's first pass done; C (`output/ckpt_moss_i2i/c_fill_s400`) + `cut` is a zero-shot sanitize, and C as init gives v3-like removal at 300 steps but not v3's tone. The Hangul block: v1 font data negative; v2 data (edition layout, balloons) gives a working Korean letterer from C in 500 steps. On the held-out ko10 pages v2 alone misspells far more than e2 and exact concat fails, so the block is not yet a swap_text part; e2's weak spot is long dense narration. The Hangul block is shelved (its data deleted 2026-11-08) until swap_text's Korean loss is fixed. **The current copy block is `output/ckpt_moss_i2i/c_fill_bw10_mod16_s800`** (2026-11-09, [reports/c_fill_mod16_bw10.md](reports/c_fill_mod16_bw10.md)): `mod_rank` 16, no tone term, B&W pairs at 10 %. Fine-tunes from it run with `mod_rank 16` (`init_lora` refuses its modulation keys otherwise). As a sanitize init it removes as C does with less residue; under `cut` alone it still letters large masks and lets masked text back.
