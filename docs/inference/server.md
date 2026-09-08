# Resident inference server

`scripts/inference_server.py` — load the DiT / VAE / text-encoder once and
serve many generations over a localhost HTTP port, instead of paying the
model-load cost per image. It's the inference counterpart of the training
`anima_daemon/`: same discoverability skin (a localhost port + a pidfile at a
fixed per-user location), wrapped around the resident-model lifetime the Anima
Tagger's `anime_tools.tagger.cli.autotag_server` worker established.

The generation engine is unchanged — the server is a thin HTTP shell over the
existing programmatic API (`get_generation_settings` → `generate(…,
shared_models=…)` → `decode_to_pil`, the same calls `examples/01_generate.py`
makes). `generate()` already reuses a warm DiT out of `shared_models["model"]`
and never frees it; the server just keeps that dict alive across requests.

## Why a separate process (and not a daemon job kind)

The training daemon's whole contract is to free the GPU between serial,
mutually-exclusive jobs — `manager._gpu_guard` actively reaps VRAM before each
launch. A resident inference model is the opposite workload, so it can't live
*inside* the daemon. Instead it runs as its own "polite tenant" process that
yields the card when training needs it (see *Coexistence* below).

| | Resident inference server | Training daemon |
|---|---|---|
| Model lifetime | resident (warm across requests) | per-job (loaded, run, freed) |
| Purpose | many small requests, warm GPU | serialize heavy exclusive jobs |
| Transport | localhost HTTP + pidfile | localhost HTTP + pidfile |

## Run it

```bash
# start the server (lazy: no VRAM until the first /generate)
python scripts/inference_server.py serve            # default port 8766
python scripts/inference_server.py serve --warm     # preload TE+VAE at startup

# from anywhere — the client discovers the port via the pidfile
python scripts/inference_server.py generate --prompt "a red fox in snow" \
    --width 1024 --height 1024 --steps 30 --cfg 3.5
python scripts/inference_server.py generate --prompt "<subject> portrait" \
    --lora_weight output/ckpt/my_lora.safetensors --multiplier 0.8

python scripts/inference_server.py status   # GET /health
python scripts/inference_server.py unload   # free VRAM, keep alive
python scripts/inference_server.py stop     # graceful shutdown
```

## HTTP surface (127.0.0.1 only — non-goal: remote / auth)

| Method | Path | Body / result |
|--------|------|---------------|
| `GET`  | `/`        | plain-text README |
| `GET`  | `/tools`   | JSON-Schema manifest of the endpoints |
| `GET`  | `/health`  | `{ok, loaded, idle_seconds, port, device, loaded_sig}` |
| `POST` | `/generate`| `{prompt, width, height, steps, cfg, seed, sampler, flow_shift, lora_weight, lora_multiplier, save_path, extra_argv}` → `{ok, path, seed, width, height, reloaded, elapsed_s}` |
| `POST` | `/unload`  | free VRAM, stay alive → `{ok, unloaded}` |
| `POST` | `/stop`    | graceful shutdown → `{ok, stopping}` |

Only `prompt` is required; everything else defaults (`get_generation_settings`
fills every knob via `GenerationRequest.to_args()`). Long-tail method flags
(Spectrum / etc.) ride `extra_argv`, exactly as in
`examples/03_generate_with_correction.py`. The decoded PNG is written to disk
(`output/inference/samples/<ts>_<seed>.png` by default, or your `save_path`) and
the path is returned.

## Discovery

Mirrors `anima_daemon/config.py`. The server writes its `{pid, create_time,
port}` to two places so any client finds it without hardcoding:

1. in-repo — `output/inference/server.json`
2. per-user mirror — `~/.anima/inference.json` (override `$ANIMA_INFERENCE_PIDFILE`)

`port` falls back to an ephemeral one on collision (just like the daemon), so the
pidfile is the source of truth, never a hardcoded `8766`. `discover_base_url()`
in the same module returns `http://127.0.0.1:<port>` or `None` if no server is up.

## Coexistence with the training daemon (two services, one card)

Two mechanisms, both load-bearing because an always-on warm DiT would otherwise
starve a training launch of VRAM:

1. **Cooperative eviction.** Before launching any job, the daemon's GPU guard
   pings a discovered inference server's `/unload`
   (`manager._evict_resident_inference`) — it frees VRAM but stays alive and
   reloads lazily on its next `/generate`. Best-effort: if no server is running
   it's a couple of cheap `stat()`s; all failures are swallowed.
2. Idle TTL. A background reaper auto-unloads after
   `$ANIMA_INFERENCE_IDLE_TTL` seconds idle (default 600; `0` disables, also
   `--idle-ttl`), so a forgotten server doesn't camp on the card. The reaper
   skips an in-flight generation (non-blocking lock acquire).

The net effect: the daemon stays the GPU's traffic cop, and the inference server
is warm-but-yielding.

## Adapter switching

The warm DiT bakes in the adapter from `lora_weight`. When a request's adapter
set differs from the loaded one, the server drops just the DiT (TE/VAE are
adapter-independent) so `generate()` reloads with the new adapter; the
"signature" is `(dit, lora_weight, multipliers, text_encoder)`, reported as
`loaded_sig` in `/health` and `reloaded` in the `/generate` result. Switching
adapters every request defeats the warm-reuse — group requests by adapter.

## Environment knobs

| Var | Default | Effect |
|-----|---------|--------|
| `ANIMA_INFERENCE_PORT` | `8766` | requested port (ephemeral fallback on collision) |
| `ANIMA_INFERENCE_IDLE_TTL` | `600` | idle seconds before auto-unload (`0` = never) |
| `ANIMA_INFERENCE_PIDFILE` | `~/.anima/inference.json` | per-user pidfile mirror location |

Model paths default to `default_checkpoints()` (`ANIMA_DIT` / `ANIMA_VAE` /
`ANIMA_TEXT_ENCODER` → `configs/base.toml` → built-ins); a request may override
`dit` / `vae` / `text_encoder` per call.
