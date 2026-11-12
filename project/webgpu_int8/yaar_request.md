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
