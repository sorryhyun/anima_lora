# moss_text_stream — what the text rows hold as they pass the blocks

Teacher-forced probes on colorize v5 (isolated reference, `RefCache`), generation's DiT by
default (int8 block linears, fp32 stream; `--base` / `--stream`). Arms, metrics and the reads
fixed before each run: the module docstrings. Write-up:
`docs/moss/findings/text_stream_page_state.md`.

| Script | Caption | What it measures |
|---|---|---|
| `probe.py` | empty (the ten template rows, identical on every page at block 0), illustration pairs | per block: drift, attention budget, share of the rows' energy that differs between pages, ridge probes to the page's palette and the reference's layout; velocity change when the rows fed to the blocks are swapped or frozen |
| `caption_rows.py` | a manga page's layout JSON (character colour boxes, text boxes) | velocity change when the rows come from the same caption run on another page: all rows, caption or template rows alone, per block range; `--base_t2i` runs Base without the LoRA and the reference |

    make daemon-run ARGS="--queue --stall-timeout 600 bench/moss_text_stream/probe.py"
    make daemon-run ARGS="--queue --stall-timeout 600 bench/moss_text_stream/caption_rows.py"
    make daemon-run ARGS="--queue --stall-timeout 600 bench/moss_text_stream/caption_rows.py --base_t2i --pages 12"

The caches are draws of the v5 recipe (`configs/moss/i2i/colorize/cache.toml`), built with
`scripts/moss/colorize_cache.py --holdout 0 --manga_holdout 0` and

| `--out` | flags |
|---|---|
| `post_image_dataset/moss/i2i_colorize_probe160` | `--n 160 --manga 0` |
| `post_image_dataset/moss/i2i_colorize_probe_manga` | `--n 0 --manga 120 --tag_dropout 0.5` |

## Runs (2026-11-14)

`dv`: mean squared difference of the velocity from the live forward's, σ 0.95 / 0.6 / 0.3.

| Run | Script, base | Pages | Time |
|---|---|---|---|
| `20261114-1456-colorize` | `probe.py`, bf16 | 160, 32 in the swaps | 24 min |
| `20261114-1503-jitter` | `probe.py`, bf16, the jitter arms | 8 | 5 min |
| `20261114-1550-colorize_fp32` | `probe.py`, int8 + fp32 stream | 160, 32 in the swaps | 22 min |
| `20261114-1528-caption_rows` | `caption_rows.py`, int8 + fp32 stream | 24 | 17 min |
| `20261114-1555-caption_rows_base_t2i` | `caption_rows.py --base_t2i`, int8 + fp32 stream | 12 (the first 12 of the 24) | 6 min |

`probe.py`, another page's template rows at every block against the controls:

| rows fed | bf16 | int8 + fp32 stream |
|---|---|---|
| own × (1 + 1e-3·ε) | 0.00010 / 0.00021 / 0.00053 (8 pages) | 0.00018 / 0.00039 / 0.00103 |
| same page, other noise draw | 0.00014 / 0.00024 / 0.00054 | 0.00019 / 0.00040 / 0.00104 |
| another page | 0.00023 / 0.00044 / 0.00136 | 0.00026 / 0.00059 / 0.00179 |
| own + isotropic draw of that norm | 0.00028 / 0.00037 / 0.00087 (8 pages) | 0.00029 / 0.00050 / 0.00129 |
| the block-0 input at every block | 0.261 / 0.640 / 0.873 | 0.266 / 0.648 / 0.873 |

- The two bases agree on the drift, the attention budget, the page share (≤ 0.3 % through
  block 40) and the ridge probes to the second decimal.
- `20261114-1550-colorize_fp32` stored the rows in bf16, so its `own` arm reads the floor
  (0.00017 / 0.00039 / 0.00102), not 0. The script stores fp32 now; `caption_rows.py` always
  did, and its `own` is 0.

`caption_rows.py`: the caption run on another page moves the velocity by 0.00137 / 0.00234 /
0.00388 on v5 (24 pages) and 0.00551 / 0.00473 / 0.00555 on Base t2i (12 pages), above an
isotropic draw of its norm on 23 / 23 / 21 of 24 and 12 / 12 / 10 of 12 pages, all of it in
blocks 26–41. Full tables: the write-up.
