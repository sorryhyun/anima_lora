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
