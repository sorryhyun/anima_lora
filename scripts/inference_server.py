#!/usr/bin/env python3
"""Resident Anima inference server — load the DiT/VAE/TE once, generate many.

Discovery mirrors ``anima_daemon/`` (a localhost HTTP port + a pidfile at a
fixed per-user location). ``generate(args, settings, shared_models=…)`` reuses
a warm DiT out of ``shared_models["model"]``; this file is the HTTP shell,
discovery, and the coexistence handshake. It is a separate process because the
daemon frees the GPU between jobs (``manager._gpu_guard``), so a resident model
instead *yields* the card on request.

GPU coexistence with the training daemon (two services, one card):

  1. **Cooperative eviction.** ``POST /unload`` drops the models to free VRAM but
     keeps the process alive + discoverable; the next ``/generate`` lazily
     reloads. The daemon pings this before launching a job (see
     ``manager._evict_resident_inference``).
  2. **Idle TTL.** A background reaper unloads after ``ANIMA_INFERENCE_IDLE_TTL``
     seconds idle (default 600), so a forgotten server doesn't camp on the card.

HTTP surface (all 127.0.0.1 only — non-goal: remote / auth):

  GET  /            human-readable README (this docstring's gist)
  GET  /tools       JSON-Schema manifest of the callable endpoints
  GET  /health      {ok, loaded, idle_seconds, port, device, loaded_sig}
  POST /generate    body = a generation request (see TOOLS); → {ok, path, seed, …}
  POST /unload      free VRAM, stay alive → {ok, unloaded}
  POST /stop        graceful shutdown → {ok, stopping}

CLI (the same file is the client — no extra module to import):

  python scripts/inference_server.py serve [--port N] [--warm] [--idle-ttl S]
  python scripts/inference_server.py generate --prompt "…" [--steps 30 --cfg 3.5 …]
  python scripts/inference_server.py status
  python scripts/inference_server.py unload
  python scripts/inference_server.py stop

``serve`` is lazy by default (no VRAM until the first ``/generate``); pass
``--warm`` to preload at startup so the first request is fast.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# scripts/ is not an installed package — put the repo root on sys.path so
# `import anima_lora` / `library` resolve when run as `python scripts/...`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("anima.inference_server")

# --------------------------------------------------------------------------- #
# Discovery — mirrors anima_daemon/config.py so any client (CLI / agent /
# MCP / a vendored node) finds {port} without hardcoding it. Kept inline so this
# file stays self-contained and importable without dragging in torch.
# --------------------------------------------------------------------------- #
HOST = "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("ANIMA_INFERENCE_PORT", "8766"))
IDLE_TTL = float(os.environ.get("ANIMA_INFERENCE_IDLE_TTL", "600"))

STATE_DIR = ROOT / "output" / "inference"
PIDFILE = STATE_DIR / "server.json"


def global_pidfile() -> Path:
    """Per-user mirror at a fixed, repo-independent path (see daemon/config.py).

    Override with ``$ANIMA_INFERENCE_PIDFILE``.
    """
    override = os.environ.get("ANIMA_INFERENCE_PIDFILE")
    if override:
        return Path(override)
    return Path.home() / ".anima" / "inference.json"


def discover_pidfile() -> Path | None:
    """First existing of: in-repo pidfile, then the per-user mirror. None if down."""
    if PIDFILE.exists():
        return PIDFILE
    mirror = global_pidfile()
    if mirror.exists():
        return mirror
    return None


def discover_base_url() -> str | None:
    """Return ``http://127.0.0.1:<port>`` of a running server, or None."""
    pf = discover_pidfile()
    if pf is None:
        return None
    try:
        info = json.loads(pf.read_text())
    except (OSError, ValueError):
        return None
    port = info.get("port")
    return f"http://{HOST}:{port}" if port else None


def _write_pidfiles(port: int) -> None:
    rec = json.dumps({"pid": os.getpid(), "create_time": time.time(), "port": port})
    for pf in (PIDFILE, global_pidfile()):
        try:
            pf.parent.mkdir(parents=True, exist_ok=True)
            pf.write_text(rec)
        except OSError:
            logger.warning("could not write pidfile %s", pf, exc_info=True)


def _clear_pidfiles() -> None:
    for pf in (PIDFILE, global_pidfile()):
        try:
            pf.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Engine — the resident model holder. All torch/anima imports are deferred into
# `_ensure_loaded` so importing this module (and `discover_*`) stays cheap.
# --------------------------------------------------------------------------- #
class InferenceEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()  # single GPU → serialize generations
        self._shared: dict = {}  # {"model": DiT, "text_encoder": TE} — warm
        self._vae = None
        self._device = None
        self._loaded_sig: tuple | None = None  # adapter signature of the warm DiT
        self._last_activity = time.time()

    # -- introspection (cheap, lock-free) ----------------------------------- #
    @property
    def loaded(self) -> bool:
        return "model" in self._shared or self._vae is not None

    def status(self) -> dict:
        return {
            "loaded": self.loaded,
            "idle_seconds": round(time.time() - self._last_activity, 1),
            "device": str(self._device) if self._device else None,
            "loaded_sig": list(self._loaded_sig) if self._loaded_sig else None,
        }

    # -- model lifetime ----------------------------------------------------- #
    def _ensure_device(self):
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _ensure_loaded(self, args) -> None:
        """Warm the adapter-independent models (TE + VAE). The DiT is loaded by
        ``generate()`` into ``self._shared['model']`` because it bakes in the
        adapter from ``args.lora_weight``."""
        import torch

        from anima_lora import load_vae
        from library.inference.models import load_text_encoder

        device = self._ensure_device()
        if "text_encoder" not in self._shared:
            te = load_text_encoder(
                text_encoder=args.text_encoder, dtype=torch.bfloat16, device=device
            )
            te.eval()
            self._shared["text_encoder"] = te
        if self._vae is None:
            self._vae = load_vae(
                args.vae,
                device=device,
                disable_mmap=True,
                dtype=torch.bfloat16,
                eval=True,
                vae_2d=getattr(args, "vae_2d", True),
            )

    def unload(self) -> dict:
        """Drop every resident model and free VRAM; stay alive + discoverable."""
        with self._lock:
            was = self.loaded
            self._shared.pop("model", None)
            self._shared.pop("text_encoder", None)
            self._vae = None
            self._loaded_sig = None
            if self._device is not None:
                from library.runtime.device import clean_memory_on_device

                clean_memory_on_device(self._device)
            self._last_activity = time.time()
        if was:
            logger.info("unloaded resident models (freed VRAM)")
        return {"ok": True, "unloaded": was}

    def maybe_idle_unload(self, ttl: float) -> None:
        """Reaper hook: unload if idle past TTL and not mid-generation."""
        if not self.loaded:
            return
        if time.time() - self._last_activity < ttl:
            return
        # Don't interrupt an in-flight generation; skip if the lock is held.
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._shared.pop("model", None)
            self._shared.pop("text_encoder", None)
            self._vae = None
            self._loaded_sig = None
            if self._device is not None:
                from library.runtime.device import clean_memory_on_device

                clean_memory_on_device(self._device)
            self._last_activity = time.time()
            logger.info("idle-unloaded resident models after %.0fs", ttl)
        finally:
            self._lock.release()

    # -- generation --------------------------------------------------------- #
    @staticmethod
    def _adapter_sig(args) -> tuple:
        mult = getattr(args, "lora_multiplier", None) or []
        if not isinstance(mult, (list, tuple)):
            mult = [mult]
        return (
            getattr(args, "dit", None),
            tuple(getattr(args, "lora_weight", None) or ()),
            tuple(str(m) for m in mult),
            getattr(args, "text_encoder", None),
        )

    def generate(self, req: dict) -> dict:
        import random

        from anima_lora import (
            GenerationRequest,
            decode_to_pil,
            default_checkpoints,
            generate,
            get_generation_settings,
        )

        ckpt = default_checkpoints()
        width = int(req.get("width", 1024))
        height = int(req.get("height", 1024))
        seed = req.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        lora_weight = req.get("lora_weight") or []
        lora_mult = req.get("lora_multiplier", 1.0)
        if lora_weight and not isinstance(lora_mult, list):
            lora_mult = [lora_mult] * len(lora_weight)

        request = GenerationRequest(
            dit=req.get("dit") or ckpt.dit,
            vae=req.get("vae") or ckpt.vae,
            text_encoder=req.get("text_encoder") or ckpt.text_encoder,
            prompt=req["prompt"],
            negative_prompt=req.get("negative_prompt", ""),
            image_size=(height, width),
            infer_steps=int(req.get("steps", req.get("infer_steps", 30))),
            guidance_scale=float(req.get("cfg", req.get("guidance_scale", 3.5))),
            flow_shift=float(req.get("flow_shift", 3.0)),
            sampler=req.get("sampler", "euler"),
            seed=int(seed),
            lora_weight=lora_weight,
            lora_multiplier=lora_mult,
            extra_argv=tuple(req.get("extra_argv", ())),
        )
        args = request.to_args()
        args.device = self._ensure_device()

        with self._lock:
            self._ensure_loaded(args)
            # If the requested adapter set differs from the warm DiT, drop it so
            # generate() reloads with the new adapter (TE/VAE are adapter-free).
            sig = self._adapter_sig(args)
            if self._loaded_sig is not None and sig != self._loaded_sig:
                self._shared.pop("model", None)
                if self._device is not None:
                    from library.runtime.device import clean_memory_on_device

                    clean_memory_on_device(self._device)
            reloaded = "model" not in self._shared

            t0 = time.time()
            gen_settings = get_generation_settings(args)
            latent = generate(args, gen_settings, shared_models=self._shared)
            image = decode_to_pil(self._vae, latent, args.device)
            self._loaded_sig = sig

            save_path = req.get("save_path")
            if not save_path:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                save_dir = STATE_DIR / "samples"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(save_dir / f"{stamp}_{seed}.png")
            else:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path)
            elapsed = time.time() - t0
            self._last_activity = time.time()

        return {
            "ok": True,
            "path": save_path,
            "seed": int(seed),
            "width": width,
            "height": height,
            "steps": request.infer_steps,
            "cfg": request.guidance_scale,
            "reloaded": reloaded,
            "elapsed_s": round(elapsed, 2),
        }


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
TOOLS = {
    "generate": {
        "method": "POST",
        "path": "/generate",
        "description": "Text-to-image generation with the warm DiT. Lazily loads "
        "models on first call; reloads only if the adapter set changes.",
        "input_schema": {
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {"type": "string"},
                "negative_prompt": {"type": "string", "default": ""},
                "width": {"type": "integer", "default": 1024},
                "height": {"type": "integer", "default": 1024},
                "steps": {"type": "integer", "default": 30},
                "cfg": {"type": "number", "default": 3.5},
                "seed": {"type": ["integer", "null"], "default": None},
                "sampler": {"type": "string", "default": "euler"},
                "flow_shift": {"type": "number", "default": 3.0},
                "lora_weight": {"type": "array", "items": {"type": "string"}},
                "lora_multiplier": {"type": ["number", "array"]},
                "save_path": {"type": ["string", "null"]},
                "extra_argv": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    "unload": {
        "method": "POST",
        "path": "/unload",
        "description": "Free VRAM; stay alive.",
    },
    "health": {
        "method": "GET",
        "path": "/health",
        "description": "Liveness + load state.",
    },
    "stop": {"method": "POST", "path": "/stop", "description": "Graceful shutdown."},
}

_README = (
    "Anima resident inference server.\n\n"
    'POST /generate {"prompt": "…", "width": 1024, "height": 1024, '
    '"steps": 30, "cfg": 3.5}\n'
    "POST /unload   free VRAM (stays alive, reloads on next /generate)\n"
    "GET  /health   {ok, loaded, idle_seconds, …}\n"
    "GET  /tools    JSON-Schema manifest\n"
    "POST /stop     graceful shutdown\n"
)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # quiet — route our own logging through the module logger instead of stderr
    def log_message(self, *args) -> None:  # noqa: D401
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"))

    @property
    def engine(self) -> InferenceEngine:
        return self.server.engine  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "":
            self._send(200, _README.encode("utf-8"), "text/plain; charset=utf-8")
        elif self.path == "/tools":
            self._json(200, TOOLS)
        elif self.path == "/health":
            st = self.engine.status()
            st.update(ok=True, port=self.server.server_address[1])
            self._json(200, st)
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        if self.path == "/unload":
            self._json(200, self.engine.unload())
        elif self.path == "/stop":
            self._json(200, {"ok": True, "stopping": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        elif self.path == "/generate":
            try:
                req = json.loads(raw or b"{}")
            except ValueError:
                self._json(400, {"ok": False, "error": "invalid JSON body"})
                return
            if not req.get("prompt"):
                self._json(400, {"ok": False, "error": "'prompt' is required"})
                return
            try:
                self._json(200, self.engine.generate(req))
            except Exception as e:  # noqa: BLE001 — report, keep serving
                logger.exception("generation failed")
                self._json(500, {"ok": False, "error": str(e)})
        else:
            self._json(404, {"ok": False, "error": "not found"})


def _bind(port: int) -> ThreadingHTTPServer:
    """Bind the requested port; fall back to an ephemeral one on collision."""
    try:
        return ThreadingHTTPServer((HOST, port), _Handler)
    except OSError:
        logger.warning("port %s busy — falling back to an ephemeral port", port)
        return ThreadingHTTPServer((HOST, 0), _Handler)


def serve(
    port: int = DEFAULT_PORT, warm: bool = False, idle_ttl: float = IDLE_TTL
) -> None:
    from library.log import setup_logging

    setup_logging()
    engine = InferenceEngine()
    httpd = _bind(port)
    httpd.engine = engine  # type: ignore[attr-defined]
    actual_port = httpd.server_address[1]
    _write_pidfiles(actual_port)

    if warm:
        from anima_lora import GenerationRequest, default_checkpoints

        ckpt = default_checkpoints()
        args = GenerationRequest(
            dit=ckpt.dit, vae=ckpt.vae, text_encoder=ckpt.text_encoder, prompt=""
        ).to_args()
        args.device = engine._ensure_device()
        engine._ensure_loaded(args)
        logger.info("warmed TE + VAE at startup")

    # Idle reaper: unload after `idle_ttl`s of no activity (0 disables).
    stop_reaper = threading.Event()

    def _reaper() -> None:
        while not stop_reaper.wait(30.0):
            if idle_ttl > 0:
                engine.maybe_idle_unload(idle_ttl)

    threading.Thread(target=_reaper, daemon=True).start()

    logger.info(
        "inference server ready on http://%s:%d (idle_ttl=%.0fs, warm=%s)",
        HOST,
        actual_port,
        idle_ttl,
        warm,
    )
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_reaper.set()
        _clear_pidfiles()
        engine.unload()
        logger.info("inference server stopped")


# --------------------------------------------------------------------------- #
# CLI client (same file — discover the server and talk to it)
# --------------------------------------------------------------------------- #
def _client_call(method: str, path: str, payload: dict | None = None) -> dict:
    base = discover_base_url()
    if base is None:
        return {"ok": False, "error": "no running inference server (pidfile not found)"}
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:  # noqa: BLE001
            return {"ok": False, "error": f"HTTP {e.code}"}
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"unreachable: {e.reason}"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Resident Anima inference server + client")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="run the resident server (this process)")
    p_serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_serve.add_argument("--warm", action="store_true", help="preload TE+VAE at start")
    p_serve.add_argument("--idle-ttl", type=float, default=IDLE_TTL)

    p_gen = sub.add_parser("generate", help="send a generation request")
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--negative_prompt", default="")
    p_gen.add_argument("--width", type=int, default=1024)
    p_gen.add_argument("--height", type=int, default=1024)
    p_gen.add_argument("--steps", type=int, default=30)
    p_gen.add_argument("--cfg", type=float, default=3.5)
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.add_argument("--lora_weight", nargs="*", default=[])
    p_gen.add_argument("--multiplier", type=float, nargs="*", default=[1.0])
    p_gen.add_argument("--save_path", default=None)

    sub.add_parser("status", help="GET /health of a running server")
    sub.add_parser("unload", help="POST /unload — free VRAM, keep alive")
    sub.add_parser("stop", help="POST /stop — graceful shutdown")

    args = p.parse_args(argv)

    if args.cmd == "serve":
        serve(port=args.port, warm=args.warm, idle_ttl=args.idle_ttl)
        return 0
    if args.cmd == "status":
        out = _client_call("GET", "/health")
    elif args.cmd == "unload":
        out = _client_call("POST", "/unload")
    elif args.cmd == "stop":
        out = _client_call("POST", "/stop")
    elif args.cmd == "generate":
        mult = args.multiplier
        if args.lora_weight and len(mult) == 1:
            mult = mult * len(args.lora_weight)
        out = _client_call(
            "POST",
            "/generate",
            {
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "cfg": args.cfg,
                "seed": args.seed,
                "lora_weight": args.lora_weight,
                "lora_multiplier": mult,
                "save_path": args.save_path,
            },
        )
    else:  # unreachable (subparser required)
        return 2

    print(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
