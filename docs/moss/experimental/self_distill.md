# Self-distilled LoRA (balanced hybrid95)

Status: experimental. Default since 2026-11-14 (it replaces add90 ema97, the default from
2026-11-09). Tested on two artist sets: caststation (30 images, 480 steps) and sincos (354
images, 2,832 steps). Config: `configs/moss/selfdistill.toml`. Earlier arms and their verdict:
[`reports/self_distill_arms.md`](../reports/self_distill_arms.md); why the EMA teacher was
dropped: [`proposal/self_distill_lora.md`](../proposal/self_distill_lora.md).

A t2i LoRA trained with the plain flow loss, except on steps at the highest noise, where the
target is a teacher's velocity instead of `ε − x₀`. The teacher is Base, given the training image
through Qwen3.5-9B's vision tower ahead of the caption. With the image in context, MOSS reproduces
the composition, outfits and layout of the target zero-shot
([vision_tower_reference.md](../findings/vision_tower_reference.md)). At σ ≥ 0.95 `x_t` holds
almost nothing of the image, so the flow target there asks the LoRA to memorise the training
composition. The teacher's velocity is how the model itself would draw that image.

## The recipe

```mermaid
flowchart LR
    x0["x₀ (cached latent)"] --> xt
    eps["ε, σ"] --> xt["x_t = (1−σ)·x₀ + σ·ε"]

    cap["TE(caption)<br/>cached .moss_te"] --> S
    xt --> S["student v_θ<br/>LoRA, grad"]
    S --> us(["u_s"])

    us -->|"σ < 0.95"| FM["flow MSE<br/>‖u_s − (ε − x₀)‖²"]
    xt -.->|"σ ≥ 0.95 only"| T
    img["TE(image + caption)<br/>cached .moss_vis"] -.-> T["teacher v<br/>Base (LoRA × 0), no grad"]
    T -.-> ut(["u_t"])
    us -.-> D["teacher MSE<br/>‖u_s − u_t‖²"]
    ut -.-> D

    FM --> L(("loss"))
    D -->|"× √(flow MSE / teacher MSE)"| L
    L --> step["Muon step on θ"]
```

Solid edges run on every step below σ 0.95; dotted ones on the ~9 % of steps at σ ≥ 0.95,
which have no flow term. The scale gives a teacher step the gradient norm on the prediction of
the flow step it replaces. Unscaled, the teacher MSE is ~0.024 against a flow MSE of ~0.51 there,
so the teacher step pushes about 0.2× as hard, and Muon's momentum (0.95) leaves its direction
to the flow steps around it.

| Key | Value | What it does |
|---|---|---|
| `distill` | `base` | teacher = Base (the LoRA at multiplier 0), the image + caption in the TE |
| `distill_sigma_min` | `0.95` | the teacher forward only on steps with σ ≥ 0.95, ~9 % of steps |
| `distill_add` | `0` | replace the flow target on those steps |
| `distill_balance` | `true` | scale the teacher MSE by `√(flow MSE / teacher MSE)` each step |
| `epochs` | `16` | 480 steps on 30 images |
| `lr` | `1e-4` | Muon, as the caststation runs |
| `ema_window` | `0.139` | the saved adapter EMA |

Everything else is `configs/moss/lora.toml`'s default: int8 Base, fp32 stream, rank 32, Muon.

Why these and not the other arms:

- **σ ≥ 0.95, not 0.9.** Against a trained LoRA the Base teacher adds nothing at σ 0.9 (the
  share of its shift that lowers the flow loss is ≤ 0); what it adds there is a pull toward
  Base's palette. At 0.95 it is still signal.
- **Base, not an EMA copy of the LoRA.** The EMA teacher (add90 ema97) moves with the student:
  on sincos at 2,832 steps that LoRA ended above Base's flow loss on its own training images at
  σ ≥ 0.95, with an extra hand on tag prompts. A fixed teacher has no such loop.
- **Balanced.** Unbalanced replace (`distill_balance false`) changed the fit only at σ 0.98 on
  caststation; balanced lowers it at every σ (table below).

**When it hurts.** On caststation's improved captions (`mask_dilate 16` + OCR boxes) the
flow-matching student passed the Base teacher's flow loss at σ 0.95 within 480 steps. Balanced
hybrid95 then pulled it back toward Base and softened the renders: high-frequency energy ×0.6 on
22 of 24 paired renders ([`reports/caststation_casbox.md`](../reports/caststation_casbox.md)).
Before using it on a set, run `bench/moss_self_distill/teacher_probe.py` with `--lora <flow-only
LoRA>`. If `lora:cap → base:img` has a negative eff at σ 0.95, the teacher has nothing left to give.

## Running it

```mermaid
flowchart LR
    subgraph P["Preprocess (cache.py)"]
        te["TE pass<br/>.moss_te"] --> vae["VAE pass<br/>.moss_latent"]
        vae -.->|"distill_teacher ☑"| vis["teacher pass<br/>.moss_vis"]
    end
    subgraph T["Train"]
        chk{".moss_vis missing<br/>or older than .moss_te?"} -->|yes| dc["distill_cache.py"]
        dc --> tr["train.py<br/>selfdistill.toml"]
        chk -->|no| tr
    end
    P --> T --> lora[("LoRA<br/>output/ckpt_moss/")]
```

The teacher embeds, `{stem}.moss_vis.safetensors` beside each TE file, are the training image at
the latent's size (~1 vision row per 32² px) followed by every caption the student can draw
(full, plain, variants), through Qwen3.5-9B with its vision tower. They are stamped with the
captions, source, encoder and tail, and only changed files re-encode.

**GUI** (`make gui-moss`, or `make gui` in MOSS mode):

1. **Preprocess**: tick **distill_teacher** to write the teacher embeds after the TE and VAE
   passes (same `text_encoder` as the TE pass). Optional, see step 3.
2. **Train**: set **Recipe** to *Self-distill (selfdistill.toml)*. That writes the keys above into
   the form and resets the keys the other recipes set to their defaults; `cache`, `output`,
   `model` and the int8 / bf16 plan stay. The Recipe item follows the form: `distill` set to
   anything but `off` under Advanced shows Self-distill too.
3. **Train** checks the cache first: when a sample's teacher file is missing or older than its TE
   file (a re-caption rewrites the TE file), it queues `distill_cache.py` ahead of training, with
   the TE weights stamped on the cache. "Preprocess, then train" queues it unless
   distill_teacher is ticked.

**CLI**:

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/moss/cache.py \
  --src <folder> --out post_image_dataset/moss/<set> --distill_teacher"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py \
  --config configs/moss/selfdistill.toml \
  --cache post_image_dataset/moss/<set> --output output/ckpt_moss/<name>.safetensors"
```

On a cache that already exists, `scripts/moss/distill_cache.py --cache post_image_dataset/moss/<set>`
writes the teacher embeds alone (`--text_encoder fp8|bf16` when the TE pass did not use int8).
Training stops with a message when a sample has no teacher file. It does not check the stamp, so
re-run the teacher pass after a re-caption (the GUI's Train does this).

The output is an ordinary LoRA: `generate.py`, the GUI's Test and the ComfyUI MOSS nodes load it
as any other. Its metadata stamps the `distill*` keys.

## Cost

int8 Base, fp32 stream, 16 GB card:

| run | s/step | steps | time |
|---|---|---|---|
| caststation, plain LoRA | 1.02 | 480 | 8.2 min |
| caststation, balanced hybrid95 | 1.05 | 480 (45 teacher) | 8.4 min |
| sincos, plain LoRA (swap 28) | 0.98 | 2,832 | 46.1 min |
| sincos, balanced hybrid95 (swap 28) | 1.04 | 2,832 (270 teacher) | 49.1 min |

The teacher forward runs only on teacher steps. Teacher cache: about a minute for 30 images
after the TE loads, ~10 min for 354. The compile range spans the teacher's longer token axis
(vision rows ahead of the caption; caststation 4,192–5,613 tokens).

## Results

Fit to the training images, flow loss of the LoRA minus Base's on the same `x_t`
(`bench/moss_self_distill/teacher_probe.py`, 2 noise draws per image; bootstrap 95 % over
images). Below 0 = closer to the training images than Base.

caststation, 30 images, 480 steps, rating-free layout-JSON cache
(`results/20261114-21{32,38,44}-nr_*`):

| LoRA | σ 0.9 | σ 0.95 | σ 0.98 |
|---|---|---|---|
| plain | −0.0231 | −0.0236 [−0.030, −0.017] | −0.0289 |
| hybrid95, unbalanced | −0.0238 | −0.0250 [−0.030, −0.021] | −0.0138 |
| **hybrid95, balanced** | −0.0188 | −0.0146 [−0.018, −0.011] | −0.0113 |

sincos, 24 of 354 images, 2,832 steps, tag captions (`results/20261114-2248-sincos_bal95`;
the other rows from 20261114-16xx, captions with the rating):

| LoRA | σ 0.9 | σ 0.95 | σ 0.98 |
|---|---|---|---|
| plain | −0.0124 | −0.0108 [−0.014, −0.007] | −0.0095 [−0.013, −0.006] |
| add90 ema97 (previous default) | −0.0056 | **+0.0049** [+0.002, +0.008] | **+0.0054** [+0.001, +0.010] |
| **hybrid95, balanced** | −0.0110 | −0.0063 [−0.009, −0.004] | −0.0028 [−0.005, −0.001] |

- Balanced hybrid95 fits the training images ~40 % less than plain at σ 0.95 and stays below
  Base at the length where add90 ema97 passed it.
- Renders, read by the user: on sincos (Base, plain, add90 ema97, balanced; 6 tag + 6 layout-JSON
  prompts and Space 004, `output/tests/sincos_distill_ab/bal95_sheets/`) balanced reads a little
  better than the others. On caststation (`output/tests/caststation_distill_ab/norating/sheet4_*`)
  the three LoRAs differ cell by cell with no consistent direction and no broken cell.
- The training loss does not rank these: below σ 0.95 it matches plain (sincos 0.2934 against
  0.2930; add90 ema97 0.3041).

## When to use it

- The LoRA should hold up on prompts unlike its training captions, or on scenes with several
  characters: the teacher replaces the target that memorises the training compositions.
- When the strongest style transfer is the goal, plain LoRA still has the edge.

## Limits

- **The teacher still pulls at the end.** On the sincos LoRA the Base teacher's shift is still
  signal at σ 0.95 (0.59 of it lowers the flow loss) after 2,832 steps, so it has no fixed
  point there; much longer runs are untested.
- Two artist sets. Character sets, other styles and i2i are untested.
- Refused with `delta_rows` other than `all`, `trigger_registers` and `ref_adapters all`: their
  row masks are built for the student's sequence, and the teacher's has the image rows ahead of
  the caption. So it does not combine with the trigger-registers recipe.
- `distill_extra_sigma` with `repa_weight` is refused (both use the capture hook).
