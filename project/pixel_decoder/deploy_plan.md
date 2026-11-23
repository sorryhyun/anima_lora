# Hub deploy: `sorryhyun/anima-pixel-decoder` split into `x1/`, `x2/`, `multi/`

The user (2026-11-23): split the Hub repo by path into `x1/`, `x2/`, `multi/`; the current ×1
files move to `x1/` renamed `old`, the sharp file with them; the new ×1 ships as
`anima_pixel_decoder_base`. Done 2026-11-23 (Hub commits `1e0352bb`, `0860b825`).

## Target layout

| path | head | from |
|---|---|---|
| `x1/anima_pixel_decoder_base.safetensors` | `x1dll_1123_0821` (×1 base on the DINO term, 12k) | new |
| `x1/anima_pixel_head_old.safetensors` | `realr_1119_2200` | root `anima_pixel_head.safetensors` |
| `x1/anima_pixel_head_sharp_old.safetensors` | `x1fgan_1122_1247` | root `anima_pixel_head_sharp.safetensors` |
| `x2/…` | `x2bgan2_1121_1340` and the rest of the ×2 files | root `anima_pixel_head_x2*.safetensors` |
| `multi/` | — | empty for now |

Root after the move: `README.md`, `LICENSE.md`, `NOTICE.md`, `.gitattributes`, `samples/`.

## Gate before upload

- `x1dll_1123_0821` finished, and the queued reads behind it done: `x1llread_1123_0821` (the
  reference read of `x1ll_1122_1000` in the new code) and `x1dllb28_1123_0821` (`p3_generated.py`,
  `x1ll` vs `x1dll`, 28 steps, cfg 4).
- The user's word on the read.

## Steps

1. Upload `output/pixel_head/x1dll_1123_0821/head.safetensors` as
   `x1/anima_pixel_decoder_base.safetensors`.
2. Move the root files to their new paths (one Hub commit; `hf` copy + delete).
3. Code that names the old paths, in the same commit here:
   - `library/downloads.py`: `ANIMA_PIXEL_HEAD_FILE` / `ANIMA_PIXEL_HEAD_X2_FILE` to the
     subfolder paths; the `anima_pixel_head` row points at the new base. The local file name under
     `models/vae/` and `default_pixel_head_path()` decide whether an existing install re-fetches.
   - The rest of the references: `library/anima/pixel_head.py`, `library/inference/args.py`,
     `examples/11_pixel_decoder.py`, `bench/pixel_readout/comfy_e2e.py`, `tests/test_pixel_head.py`,
     `gui/dialogs/system.py`, `gui/i18n/*.py`, `docs/anima/inference/pixel_head.md`,
     `docs/anima/experimental/pixel_decoder.md`.
   - ComfyUI node, if it fetches by file name (`custom-nodes` skill).
4. Model card: the layout table, which file is the default, the old files kept for comparison.
5. `plan.md` § State, the ×1 files table: release → `x1/anima_pixel_decoder_base`.

## Decided (the user, 2026-11-23)

- Hub `old/` deleted (its earlier ×1 and `x2_preview` files are gone from the Hub).
- `x2/` keeps the file names (`anima_pixel_head_x2`, `anima_pixel_head_x2_v2`).
- The root head files are removed after the copy; old root links no longer resolve.
- `multi/` does not exist until a file goes in (the Hub keeps no empty folders).

## Left

- ComfyUI-Anima-PixelDecoder 0.3.3 (lists `*pixel_decoder*` files in the VAE folders; README on
  the new paths): committed locally, not pushed or published.
- The GAN finish on the new base (`plan.md` To do 2) ships later in `x1/` as its own file.
