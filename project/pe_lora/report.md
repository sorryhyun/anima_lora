# pe_lora — Qwen-Image-2.1-PE-T2I as a LoRA on the MOSS TE

Qwen/Qwen-Image-2.1-PE-T2I (a prompt rewriter) is a full fine-tune of Qwen3.5-9B instruct,
the model MOSS-Image ships as its text encoder. This line extracts the difference as a
LoRA on the shipped INT8 TE and asks what the MOSS DiT does with it.

## Claims

1. **The PE LoRA runs on the shipped INT8 TE.** Extracted at r128 (down_proj r256) and
   applied as unmerged branches (`src/pe_mode.py`), it gives answer KL 0.187 against the
   PE teacher, where the bare 9B gives 0.85, and 16 / 16 valid JSON outputs. Switching back
   to TE mode reproduces the TE's encodes bit for bit (max |Δh| 0.0, re-checked
   2026-11-07). (2026-10-27)
2. **PE mode moves the TE output a lot, mostly in directions the DiT reads weakly.** The
   DiT reads Δ = h_pe − h_te through `text_embedder.proj` less strongly than it reads h or
   random directions, and less of Δ falls in P's top subspace than random would put there.
   PE did not learn "what an image model reads". (2026-11-07)
3. **PE's think text, placed before `</think>` and encoded in TE mode, steers content
   about as strongly as replacing the caption with PE's answer.** The caption rows are
   unchanged; specific details PE planned in its thinking render. Unblinded, 5 prompts ×
   2 seeds, and the bare-9B control is confounded (§2), so the share that is PE's own is
   not separated. (2026-11-07)

## 1. What the DiT reads of the PE delta

`src/dit_read.py`, `output/tests/pe_dit_read/20261107-1609/`. 230 prompts (`moss_te_sae`'s
`probe.py` set: 150 layout-JSON captions from five MOSS caches, 80 NL prompts), `sure` tail,
INT8 TE in TE mode and in PE mode. P = `text_embedder.proj.weight` (its bias cancels in Δ,
and the two text blocks after it are not modelled).

| | layout body | NL body | suffix (incl. tail) | `</think>` row |
|---|---|---|---|---|
| ‖Δ‖ / ‖h‖ | 0.52 | 0.34 | 0.64 / 0.50 | 0.65 / 0.45 |
| ‖PΔ‖ / ‖Ph‖ | 0.47 | 0.32 | 0.63 / 0.49 | 0.66 / 0.45 |
| read ‖PΔ‖/‖Δ‖ (h: ‖Ph‖/‖h‖) | 1.77 (1.92) | 1.80 (1.84) | 1.86 (1.93) | 1.86 (1.81) |
| share in P's top-256 right singular vectors, Δ (h) | 3.5 % (4.9 %) | 3.8 % (4.9 %) | 3.7 % (4.6 %) | 3.5 % (3.9 %) |

Suffix and `</think>` cells read layout / NL. Random directions read 2.12 (median), and
their top-256 share is 6.25 %. The prefix rows read 3.0 for h, 2.0 for Δ.

Over the pooled body rows:
- the prompt-constant share ‖mean Δ‖² / mean ‖Δ‖² is 0.14;
- mean Δ points against mean h (cos −0.63), so part of the delta shrinks the shared
  component;
- the top 16 principal components of Δ hold 30 % of its energy and read 1.17–1.89.

The rows that move most as P sees them are JSON delimiters (`":"`, `","`), the structural
rows P suppresses (`moss_te_sae/reports/sae_fit.md`).

Implication: encoding in PE mode would hand the DiT an input changed by about half, as P
sees it. The direct "PE-mode encode" arm was not run.

## 2. PE's thinking as context

`src/think_inject.py`, `output/tests/pe_think_inject/` (`gens.json`, `grid_p{0..4}.png`).

**Setup.**
- One INT8 TE does both jobs. It generates under PE's system prompt (Hub
  `system_prompt.txt`, card sampling, max_new 2048) in PE mode and in TE mode (the bare
  9B), then encodes in TE mode.
- Render: MOSS-Image Base, no LoRA, ER-SDE 50, CFG 3, Spectrum, fp32 stream, 1024²,
  seeds 1 / 2, `sure` tail on cond and uncond.
- Prompts: the three `bench/moss_ab/lora_ab.py` PROMPTS (layout JSON, `@channel
  (caststation)`) and two short NL briefs (cat on a rainy windowsill, girl in a red
  raincoat at a night bus stop).

| arm | cond | text rows |
|---|---|---|
| `sure` | caption + `\n</think>\n\nSure,` | 31–80 |
| `inst_think` | caption + the bare 9B's think + tail | 1896–2129 |
| `pe_think` | caption + PE's think + tail | 576–947 |
| `pe_answer` | PE's answer paragraph as the caption, `sure` tail | 473–736 |

**Generations.** PE finished every prompt in 1028–1528 tokens, all five as valid JSON. The
bare 9B hit max_new on all five (0 / 5 finished). Its think is markdown meta-reasoning
("Thinking Process: 1. Analyze the Request …").

**Observed (unblinded).**
- **Planned details render under `pe_think` as under `pe_answer`:**
  - p0: the chat panel PE laid out, titled "Channel", with "[18:17] Hello!", "Miku!",
    "Peace!";
  - p3: the mug at lower right, the plant at lower left and the framed landscape;
  - p4: the route number "5" on a dark sign, and a photographic night street in place of
    `sure`'s painted one;
  - p2: a school building behind the bench.
- **The image reads the think rows themselves, not only `</think>`.** The
  `assistant_tail` path carries scene content, not just a sharpening key.
- **Caption fidelity is not shown either way.** Under p0 s1, `pe_think` keeps Miku's tie
  and headset while `pe_answer` drops her outfit (white turtleneck) on both seeds.
  `pe_think` s2 also drops it.
- **PE reads the artist tag as a channel name.** It plans and draws UI overlays and
  "castchannel" lettering (p0, p1). On tag captions with artist tags, its thinking adds
  wrong content.
- **`inst_think` degrades on every prompt:** flat shading, an off-model p1, and an
  "@Channel" watermark on p0 / p1. Unfinished meta-reasoning at 2–3.6× PE's length
  confounds it with length, so it does not isolate PE's contribution.

## Open

- **A clean control** for §2:
  - the bare 9B run to completion, or length-matched;
  - another prompt's PE think, which separates the think's content from its presence.
- **The rows the image reads in the think block:** `tail_attn.py`'s per-segment mass and
  contribution, extended to the think rows.
- **Qwen/Qwen-Image-2.1-PE-I2I** (Qwen3.5-VL 9B edit rewriter, `Qwen3_5ForConditionalGeneration`):
  - its thinking splits "keep" from "change" given the input image, the axis of
    `project/lora_blocks/`;
  - first check whether its `visual.*` differs from the MOSS TE's tower;
  - then inject its think on one i2i task.
- From 2026-10-27, still open:
  - static-KV decode (18 ms/tok against 66) matched only 95 / 512 greedy tokens and its
    quality is unchecked;
  - the comfy_kitchen `int8_linear` custom op exists only as a monkeypatch in
    `src/bench_speed.py`.

## Artifacts

Off-repo, `/media/sorryhyun/new/qwen35_pe/`:
- `lora/pe_white_r512.safetensors` — truncatable factors;
- `lora/pe_mode_int8_r128d256.safetensors` — PE-mode tensors for the INT8 TE;
- `pe/system_prompt.txt` — re-fetched from the Hub 2026-11-07.

The calibration Grams were pruned, so re-extracting needs them recomputed (`src/calib.py`).
