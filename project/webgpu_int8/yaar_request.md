# yaar change request: user-apps/anima, int8 text encoder

New files on HF (`sorryhyun/anima-turbo-4step`). The graph's external-data `location` is the bare
sidecar file name:

| Path under `onnx/` | Bytes |
|---|---|
| `int8/text_encoder_int8.onnx` | 3,350,223 |
| `int8/text_encoder_int8.onnx.data` | 630,688,768 |

Same inputs and outputs as `text_encoder_fp16`: `input_ids` / `attention_mask` int64 `(1,512)` →
fp16 `qwen_prompt_embeds` `(1,512,1024)`. The tokenizer, padding and conditioner do not change.

Changes, all in `user-apps/anima/src/`:

1. `text/encoder.ts`: `ENCODER = 'int8/text_encoder_int8'`. The phase label becomes
   `Loading text encoder (631 MB)…`, and the header comment's "1.19 GB" figure needs updating.
   The prompt-embeds LRU comment says "1.46 GB text path"; it is now about 0.9 GB.
2. `ml/assets.ts`: in `EXPECTED_BYTES`, add `'int8/text_encoder_int8.onnx.data': 630_688_768`.
3. `ml/downloadWeights.ts`: in `MANIFEST`, replace the two `text_encoder_fp16` rows with the two
   rows above.
4. `app/weights.ts`: in `checkItem('text-encoder', …)`, use the int8 paths.

5. `ml/sessions.ts`, in `loadModel`: one `dataPath` is used for both the fetch
   (`resolveExternalData(dataPath)`) and ORT's `externalData` `path` key. Under `int8/` they
   differ. The fetch needs `int8/text_encoder_int8.onnx.data`, but the key has to match the
   graph's `location`, which is `text_encoder_int8.onnx.data`. Key on the basename:

   ```ts
   sessionOptions: {
     externalData: [{ path: dataPath.split('/').pop()!, data: url }],
     ...extraSessionOptions,
   },
   ```

   The non-square buckets are unaffected: their `dataName` is already a bare file name.

Ops it needs on the WebGPU EP: `com.microsoft` `MatMulNBits` (bits 8, block 32,
`accuracy_level` 4; DP4A on non-Apple adapters with subgroups, fp16 dequant otherwise) and
`GatherBlockQuantized` (uint8, bits 8). onnxruntime-web 1.30.0 registers both. Both need
`shader-f16`, as the fp16 graph already does.

To roll back, point `ENCODER` back at `text_encoder_fp16`. The fp16 files stay on HF.

# yaar change request 2: one resolution-free int8 DiT and one VAE for every ratio

New files on HF (`sorryhyun/anima-turbo-4step`, under `onnx/`):

| Path under `onnx/` | Bytes | What |
|---|---|---|
| `int8/dit_int8_dyn.segments.json` | 3,091 | manifest, same schema as today's |
| `int8/dit_int8_dyn.seg0.onnx` … `seg6.onnx` | ~1.3–1.5 MB each (seg6 0.36 MB) | resolution-free segment graphs |
| `int8/dit_int8.onnx.data` | 2,372,323,328 | shared sidecar (was 3,913,271,296) |
| `int8/vae_decoder_dyn_fp16.onnx` | 562,856 | resolution-free VAE decoder (fp16; under `int8/` with the rest of the set); weights stay in the root `vae_decoder_512_fp16.onnx.data` |

The DiT is the v4_preview turbo (merged14 + turbo EMA 1112_1512, baked), int8 ConvRot GPTQ on
the 224 block linears, everything else fp16, residual rescaled ×1/16 as in the `_r16` export.

What changed in the graphs:
- The latent grid is free: `latent` is `(1, 16, 1, H, W)` with any H, W (even, i.e. image sides
  multiple of 16); `noise_pred` comes back the same shape. Token count, patch grid and RoPE
  tables are computed in-graph from `Shape(latent)`.
- **Every segment now lists `latent` among its `inputs`** (it reads only the shape).
  `runSegmented` already keeps the feeds in `env`, so nothing changes there.
- The VAE takes `latent (1, 16, H, W)` and returns `image (1, 3, 8H, 8W)` for any H, W.

Changes, all in `user-apps/anima/src/`:

1. `generation/buckets.ts`: every bucket's `dit` becomes `'int8/dit_int8_dyn'` and `vae`
   `'int8/vae_decoder_dyn_fp16'`. `ditDataFor` returns `'int8/dit_int8.onnx.data'` and `vaeDataFor`
   returns `SHARED_VAE_DATA` for **every** bucket (the square-bucket special case goes: the VAE
   graph's `location` is `vae_decoder_512_fp16.onnx.data`, at the root of `onnx/`; with request 1's
   basename keying in `loadModel`, the fetch reads the root file and the key matches the graph). `SHARED_DIT_DATA` becomes
   `'int8/dit_int8.onnx.data'`. A ratio is now just `W`/`H`; any multiple of 16 works, so
   new ratios or larger sizes need no new files.
2. `generation/session.ts`:
   - `ditSessionOptions`: the int8 graph carries the r16 rescale, so graph optimization must
     stay on. Today's check is `model.includes('_r16')`; the new name has no `_r16` and would
     get `graphOptimizationLevel: 'disabled'`. Use e.g.
     `model.includes('_r16') || model.includes('dit_int8')`.
   - `releaseOtherDits`: the matcher is `n.startsWith('dit_')`, and `'int8/dit_int8_dyn'` does
     not start with it, so the text path's `releaseAllDits()` would never free the DiT. Match
     on the basename: `n.split('/').pop()!.startsWith('dit_')`.
3. `ml/assets.ts`: in `EXPECTED_BYTES`, add `'int8/dit_int8.onnx.data': 2_372_323_328`.
4. `ml/downloadWeights.ts`: `BUCKET_GRAPHS` becomes one set — `int8/dit_int8_dyn.segments.json`,
   the seven `int8/dit_int8_dyn.segN.onnx`, `int8/vae_decoder_dyn_fp16.onnx` — instead of one per
   bucket; the DiT sidecar row points at `int8/dit_int8.onnx.data`.
5. `app/weights.ts`: the per-bucket `checkItem`s collapse to one DiT and one VAE item.
6. `ui/App.ts`: `releaseOtherDits(bucketById(v).dit)` on a ratio change now keeps the one
   resident DiT (same name), so switching ratio no longer reloads weights. The first run at a
   new size compiles its shaders, later runs at that size reuse them.

Optional, now that the DiT is 2.37 GB: with the int8 text encoder (0.63 GB) and the conditioner
(0.27 GB) the whole set is about 3.3 GB, so on desktop the text path does not need to evict the
DiT (`releaseAllDits` in `text/encoder.ts`). Keeping it resident removes the DiT reload on every
new prompt.

Needs `shader-f16` (as before) and, for the DP4A path, `subgroups` on a non-Apple adapter. On
Linux + NVIDIA, Chrome must run with
`--enable-features=Vulkan --enable-dawn-features=vulkan_enable_f16_on_nvidia`.

To roll back, point `dit` / `vae` back at the per-bucket names; the fp16 files stay on HF.

# yaar change request 3: int8 DiT with fp32 MatMulNBits (fixes the DP4A path)

The request-2 DiT (`int8/dit_int8_dyn`) gives broken images on every adapter that takes
onnxruntime-web's DP4A path (`subgroups`, non-Apple); the fp16-dequant path (Apple, no
subgroups) is fine. The DP4A shader multiplies the activation scale by the weight scale in the
op's float type first. In fp16 that product is 1e-5 to 1e-10 here, below fp16's smallest
normal (6.1e-5), and for the r16-folded linears (both `output_proj`s, `mlp.layer2`) it rounds
to zero: cross-attention's output is lost in all 28 blocks. The new set runs each
`MatMulNBits` in fp32 between two `Cast`s, with fp32 scales. The int8 weights are the same.

New files on HF (`sorryhyun/anima-turbo-4step`, under `onnx/`). They have new names because the
graphs and the manifest are cached in IndexedDB by url: old cached graphs over a replaced
sidecar would read wrong byte ranges.

| Path under `onnx/` | Bytes |
|---|---|
| `int8/dit_int8_mm32_dyn.segments.json` | 3,171 |
| `int8/dit_int8_mm32_dyn.seg0.onnx` … `seg6.onnx` | ~1.3–1.5 MB each (seg6 0.36 MB) |
| `int8/dit_int8_mm32.onnx.data` | 2,475,083,776 |

Same inputs, outputs and segment schema as request 2. The VAE and the text encoder do not change.

Changes, all in `user-apps/anima/src/`:

1. `generation/buckets.ts`: `DIT = 'int8/dit_int8_mm32_dyn'` and
   `SHARED_DIT_DATA = 'int8/dit_int8_mm32.onnx.data'`.
2. `ml/assets.ts`: in `EXPECTED_BYTES`, replace the `'int8/dit_int8.onnx.data'` row with
   `'int8/dit_int8_mm32.onnx.data': 2_475_083_776`.
3. `ml/downloadWeights.ts`: in `BUCKET_GRAPHS`, the `segments.json` row is 3,171 bytes. The other
   rows follow `DEFAULT_BUCKET.dit` and `SHARED_DIT_DATA`.
4. Comments and labels that say "2.37 GB" (`generation/buckets.ts`, `generation/session.ts`,
   `generation/diagnostics.ts`, `text/encoder.ts`, `ml/downloadWeights.ts`, `ml/sessions.ts`):
   the sidecar is now 2.48 GB.

Nothing else changes: `ditSessionOptions` matches on `dit_int8` and `releaseOtherDits` on the
basename's `dit_` prefix, and both still match the new name.

Left behind on a machine that ran request 2: nine files under
`storage/apps/anima/weights/int8/` (`dit_int8_dyn.*`, `dit_int8.onnx.data`, 2.37 GB) and the
old graphs in the `anima-weights` IndexedDB. Nothing reads them after this change.

Not run in a browser yet. Checked on CPU: the new graphs match the old ones on float activations
(`noise_pred` 0.0068 against torch fp32, old 0.0069), and a replay of the DP4A shader arithmetic
on all 224 linears gives 0.3–0.6% per linear in fp32 (old, fp16: 7% / 17% / 99% median on
`self_attn.output_proj` / `mlp.layer2` / `cross_attn.output_proj`).

To roll back, point `DIT` and `SHARED_DIT_DATA` back at the request-2 names; those files stay on HF.

# yaar change request 4: the v4 DiT, two segments

The turbo moves to v4: the merged14 soup with the Hub v3 turbo LoRA baked on it, as on HF
`anima_turbo_4step_v4.safetensors`. The int8 DiT is that checkpoint, rebuilt the request-3 way
(ConvRot GPTQ on the same grid, fp32 `MatMulNBits` between `Cast`s, the same 224 linears). The set
is cut into **2 segments instead of 7**: source segments 0–2 and 3–6 concatenated, sidecar bytes
unchanged. The request-2 files (`int8/dit_int8_dyn.*`, `int8/dit_int8.onnx.data`) were deleted
from HF on 2026-11-13.

New files on HF (`sorryhyun/anima-turbo-4step`, under `onnx/`):

| Path under `onnx/` | Bytes |
|---|---|
| `int8/dit_int8_v4.segments.json` | 920 |
| `int8/dit_int8_v4.seg0.onnx` | 4,216,096 |
| `int8/dit_int8_v4.seg1.onnx` | 4,786,732 |
| `int8/dit_int8_v4.onnx.data` | 2,475,083,776 |

Segment 0 reads sidecar bytes `[0, 1169852416)` (1.09 GiB) and segment 1 reads
`[1169852416, 2475083776)` (1.22 GiB). Inputs, outputs and manifest schema are as before; the
segments pass `linear_2`, `silu_1`, `unsqueeze_13`, `unsqueeze_15`, `add_234`, `split_39_split_2`,
`add_237` between them.

Changes, all in `user-apps/anima/src/`:

1. `generation/buckets.ts`: `DIT = 'int8/dit_int8_v4'` and
   `SHARED_DIT_DATA = 'int8/dit_int8_v4.onnx.data'`.
2. `ml/assets.ts`: in `EXPECTED_BYTES`, replace the `'int8/dit_int8_mm32.onnx.data'` row with
   `'int8/dit_int8_v4.onnx.data': 2_475_083_776` (same size).
3. `ml/downloadWeights.ts`: `DIT_SEGMENTS = 2`, the `segments.json` row 920 bytes, and the
   per-segment progress size about 4_500_000.
4. `ml/sessions.ts`: the "Segmented models" comment says the DiT ships as ~340 MB segments. They
   are now 1.09 / 1.22 GiB. `InferenceSession.create` holds a segment's bytes while it uploads
   them, so the load peak per session rises from about 0.37 GiB to 1.22 GiB on top of the GPU
   copy. That does not matter on desktop. On a phone, keep the request-3 set (`dit_int8_mm32_dyn`,
   7 segments, still on HF) if the 2-segment load fails there.

`ditSessionOptions` (matches `dit_int8`) and `releaseOtherDits` (basename prefix `dit_`) still
match the new name.

To roll back, point `DIT` / `SHARED_DIT_DATA` at the request-3 names and set `DIT_SEGMENTS = 7`.

Also deleted from HF on 2026-11-13: the per-ratio fp16 DiT sets (`dit_<bucket>_fp16_r16.*` and
`dit_512_fp16_r16.onnx.data`), the per-ratio VAE graphs (`vae_decoder_<bucket>_fp16.onnx`) and
`text_encoder_fp16.onnx` / `.data`. The rollbacks of requests 1–3 that point back at those files no
longer work. Under `onnx/`, only `text_conditioner_fp16.onnx` / `.data` and
`vae_decoder_512_fp16.onnx.data` stay; `int8/vae_decoder_dyn_fp16.onnx` reads the latter.
