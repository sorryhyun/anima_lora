"""Anima LoRA Trainer ComfyUI node.

A single, deliberately small node: feed it one IMAGE + a caption, pick the base
Anima checkpoint, a rank, an epoch count and a GPU tier, and on launch it trains
a T-LoRA + OrthoLoRA against that one image/prompt pair.

Design notes:

- **No MODEL input.** Holding the base DiT resident in ComfyUI *and* spawning a
  training subprocess on the same GPU is an easy OOM. So this node does not take
  a MODEL socket; instead it loads the chosen Anima checkpoint itself **after**
  training finishes and returns a patched MODEL — a drop-in for
  ``UNETLoader → Anima Adapter Loader``.
- **Daemon required.** Training always runs in the local training daemon's own
  detached subprocess (out of the ComfyUI process). If the daemon isn't up this
  node errors out rather than auto-starting it.
"""

from __future__ import annotations

import datetime as _dt
import os

import folder_paths  # ComfyUI builtin

# Training deps are imported lazily inside `train()` so that merely loading this
# module in ComfyUI doesn't force `library.*` imports (which pull torch
# extensions and slow startup).


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _anima_lora_root() -> str:
    from .training import find_anima_lora_root

    return find_anima_lora_root(os.path.dirname(__file__))


def _load_node_defaults() -> dict:
    """Read the sibling ``node_defaults.toml`` of trainer-node-only overrides.

    These tune the few-image training regime this node runs in (DataLoader
    worker policy, log cadence) without editing the shared ``configs/base.toml``.
    They're layered *below* the UI fields in ``train()`` so user choices win.
    Re-read on every run; a missing or unparseable file is treated as empty so
    the node still works if it's deleted.
    """
    import tomllib

    path = os.path.join(os.path.dirname(__file__), "node_defaults.toml")
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(
            f"[Anima Trainer] could not read node_defaults.toml ({e}); "
            f"using base/preset defaults.",
            flush=True,
        )
        return {}


def _input_subdirs() -> list[str]:
    """List immediate subdirectories of ComfyUI's input dir, as relative names.

    Populates the folder-mode trainer's dataset dropdown. Re-evaluated whenever
    ComfyUI rebuilds INPUT_TYPES (graph load / refresh), so newly-added dataset
    folders show up after a node refresh. Returns ``[""]`` (a single blank entry)
    when the input dir has no subfolders, so the node still loads.
    """
    try:
        root = folder_paths.get_input_directory()
        subdirs = sorted(
            entry
            for entry in os.listdir(root)
            if os.path.isdir(os.path.join(root, entry))
        )
    except OSError:
        subdirs = []
    return subdirs or [""]


_MASK_NONE = "(none)"


def _input_subdirs_optional() -> list[str]:
    """Input-dir subfolders prefixed with a ``(none)`` sentinel for the mask picker."""
    subdirs = [d for d in _input_subdirs() if d]
    return [_MASK_NONE, *subdirs]


def _comfy_loras_dir() -> str:
    """Return the directory to save trained LoRAs into, under ComfyUI's loras.

    Prefers ComfyUI's *native* ``models/loras`` over any path an
    ``extra_model_paths.yaml`` entry registers (which, with ``is_default: true``,
    would otherwise sort first — e.g. anima_lora's ``output/``). Falls back to
    the first registered loras path when the native dir isn't registered.
    """
    paths = folder_paths.get_folder_paths("loras") or []
    if not paths:
        raise RuntimeError("ComfyUI has no 'loras' folder registered.")
    models_dir = getattr(folder_paths, "models_dir", None)
    if models_dir:
        native = os.path.abspath(os.path.join(models_dir, "loras"))
        for p in paths:
            if os.path.abspath(p) == native:
                return p
    return paths[0]


def _comfy_model_path(folder: str, preferred: str) -> str | None:
    """Resolve a model file through ComfyUI's ``folder_paths``, or ``None``.

    Lets the trainer source the VAE / text-encoder used for caching from
    whatever ComfyUI registers (its ``models/`` dirs, or any ``base_path`` from
    ``extra_model_paths.yaml``) — the same way the base DiT is already resolved —
    instead of assuming a copy under ``anima_lora/models/``. Returns ``None``
    when nothing usable is found, so the caller falls back to the
    preprocess-config defaults rather than passing a bad path.

    Tries the canonical Anima filename first; if absent, accepts a sole file in
    the folder (the common case — one VAE, one TE); otherwise gives up rather
    than guess among several.
    """
    path = folder_paths.get_full_path(folder, preferred)
    if path:
        return path
    files = folder_paths.get_filename_list(folder) or []
    if len(files) == 1:
        return folder_paths.get_full_path(folder, files[0])
    return None


def _resolve_daemon_client():
    """Return the bundled stdlib ``DaemonClient`` class.

    The trainer node is self-contained: it talks to a running Anima daemon over
    localhost HTTP through the pure-stdlib client vendored under ``_vendor/``
    (kept verbatim-in-sync with the live ``anima_daemon/client.py`` by
    ``scripts/release/sync_vendor.py``). We import it via a *relative* import so we
    never bind a generic top-level name — an old live-first probe that did
    ``import anima_daemon.client`` against a ``../..`` it assumed was the repo
    (which is ComfyUI's root in a standalone install) could poison
    ``sys.modules`` with an unrelated on-path package and break the vendor
    fallback (``ModuleNotFoundError``).

    The client never *starts* a daemon — it only talks to one. Its
    ``config.discover_pidfile`` finds a running daemon's pidfile (the per-user
    ``~/.anima/daemon.json`` mirror, ``$ANIMA_DAEMON_PIDFILE``,
    ``$ANIMA_LORA_ROOT``, or an in-repo path) to follow even an ephemeral
    fallback port; failing all that it uses ``127.0.0.1:8765`` (override with
    ``$ANIMA_DAEMON_PORT``).
    """
    from ._vendor.anima_daemon.client import DaemonClient

    return DaemonClient


def _trainer_tmp_root() -> str:
    """Where single-image-mode datasets are staged before submission.

    Prefers the repo's ``output/tmp_trainer`` (so it's covered by the repo
    gitignore and easy to prune), but falls back to ComfyUI's temp dir when the
    node is installed outside the anima_lora tree — the daemon reads it over a
    plain filesystem path either way (same machine, localhost-only).
    """
    try:
        return os.path.join(_anima_lora_root(), "output", "tmp_trainer")
    except Exception:
        return os.path.join(folder_paths.get_temp_directory(), "anima_trainer")


# ---------------------------------------------------------------------------
# Train via daemon → block until done → return saved safetensors path
# ---------------------------------------------------------------------------


def _train_and_save(
    *,
    method: str,
    preset: str,
    overrides: dict,
    image=None,
    prompt: str = "",
    dataset_dir: str = "",
    mask=None,
    mask_dir: str = "",
) -> str:
    """Submit training to the local daemon and block until done.

    Either an ``image`` + ``prompt`` (single-image mode) or a ``dataset_dir``
    (directory mode) supplies the data; ``prepare_dataset_dir`` picks the mode.
    An optional ``mask`` (MASK tensor, single-image) or ``mask_dir`` (directory)
    turns on masked loss.

    The daemon runs ``accelerate launch … train.py`` in its **own** detached
    subprocess, so a CUDA OOM / segfault kills only the job — not ComfyUI. This
    call polls ``GET /jobs/{id}`` and drives a ``ProgressBar`` until the job
    reaches a terminal state, then returns the absolute path of the saved
    safetensors.

    Raises ``RuntimeError`` if the daemon is not already running — this node
    does not auto-start it.
    """
    import comfy.model_management

    from .dataset_prep import prepare_dataset_dir

    # Daemon must already be up. We do NOT auto-start it here.
    DaemonClient = _resolve_daemon_client()
    client = DaemonClient()
    if client.health() is None:
        raise RuntimeError(
            "Anima training daemon is not running. Start it with `make daemon` "
            "(or `python tasks.py daemon`) from the anima_lora/ directory, then "
            "re-run this node."
        )

    # image set → single-image mode (writes the IMAGE batch + caption sidecars);
    # dataset_dir set → directory mode (user's dir of images + .txt sidecars).
    # src_dir = originals (read-only input to preprocess); image_dir/cache_dir =
    # where resized images + caches land; dataset_cfg names image_dir + cache_dir.
    (
        src_dir,
        _image_dir,
        _cache_dir,
        dataset_cfg,
        n_images,
        resolved_mask_dir,
    ) = prepare_dataset_dir(
        image,
        prompt,
        dataset_dir,
        tmp_root=_trainer_tmp_root(),
        mask=mask,
        mask_dir=mask_dir,
    )

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"anima_trainer_{ts}"
    output_dir = _comfy_loras_dir()

    overrides = dict(overrides)
    overrides.setdefault("dataset_config", dataset_cfg)
    overrides.setdefault("output_dir", output_dir)
    overrides.setdefault("output_name", output_name)
    # Masks present → force masked loss on (the gui-method TOML leaves it off).
    # The dataset config already carries the resolved mask_dir.
    if resolved_mask_dir:
        overrides["masked_loss"] = True

    print(
        f"[Anima Trainer] submitting method={method} preset={preset} "
        f"images={n_images}{' +masks' if resolved_mask_dir else ''} "
        f"→ {output_name}.safetensors",
        flush=True,
    )

    # Free ComfyUI-held VRAM so the (separate) training process has room for its
    # own DiT + optimizer state. The daemon spawns the job; we just watch.
    comfy.model_management.unload_all_models()
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # train.py refuses to run on an incomplete latent/TE cache, and the temp
    # dir starts empty. So we submit a `preprocess-config` *command* job that
    # bucket-resizes + caches the dataset, carrying a `chain_train` spec: the
    # daemon enqueues the follow-on training job itself the moment preprocess
    # succeeds (and persists the link as `chained_job_id`), so the chain
    # completes even if ComfyUI stops polling. Both run on the daemon's single
    # serial GPU queue, so they can't fight over VRAM.
    # Cache against the models ComfyUI registers (the DiT the user selected, plus
    # the VAE + text-encoder resolved through folder_paths) so preprocess never
    # assumes a copy under anima_lora/models/. Unresolved ones are simply omitted
    # → preprocess-config falls back to its config-default models/ paths.
    pp_argv = [
        "tasks.py",
        "preprocess-config",
        "--dataset_config",
        dataset_cfg,
        "--src",
        src_dir,
    ]
    dit_path = overrides.get("pretrained_model_name_or_path")
    if dit_path:
        pp_argv += ["--dit", dit_path]
    vae_path = _comfy_model_path("vae", "qwen_image_vae.safetensors")
    if vae_path:
        pp_argv += ["--vae", vae_path]
    qwen3_path = _comfy_model_path("text_encoders", "qwen_3_06b_base.safetensors")
    if qwen3_path:
        pp_argv += ["--qwen3", qwen3_path]

    pp_job_id = client.submit_command(
        label="preprocess",
        argv=pp_argv,
        chain_train={
            "method": method,
            "preset": preset,
            "methods_subdir": "gui-methods",
            "overrides": overrides,
        },
    )["job_id"]
    print(f"[Anima Trainer] preprocess queued as job {pp_job_id}", flush=True)

    # Phase 1: wait for preprocess. No progress bar (it emits no step total).
    pp_job = _poll_job(client, pp_job_id, label="preprocess")
    pp_state = pp_job.get("state")
    if pp_state != "done":
        raise RuntimeError(
            f"Preprocess job {pp_job_id} ended as '{pp_state}': "
            f"{pp_job.get('error') or '(no detail)'}. "
            f"See the job stdout log: {pp_job.get('stdout_path')}"
        )

    # The daemon stamps the chained training job id on the (now done) command
    # job. If it's missing the chain didn't fire — fail loudly rather than hang.
    job_id = pp_job.get("chained_job_id")
    if not job_id:
        raise RuntimeError(
            f"Preprocess job {pp_job_id} finished but did not chain a training "
            f"job. See the job stdout log: {pp_job.get('stdout_path')}"
        )
    print(f"[Anima Trainer] training queued as job {job_id}", flush=True)

    # Phase 2: wait for training, driving a step ProgressBar.
    expected = os.path.join(output_dir, f"{output_name}.safetensors")
    job = _poll_job(client, job_id, label="train", with_progress=True)
    state = job.get("state")

    if state == "done":
        path = expected if os.path.exists(expected) else job.get("ckpt_path")
        if not path or not os.path.exists(path):
            raise RuntimeError(
                f"Training finished but no checkpoint found (expected {expected}). "
                f"See the job stdout log: {job.get('stdout_path')}"
            )
        print(f"[Anima Trainer] saved {path}", flush=True)
        return path

    raise RuntimeError(
        f"Training job {job_id} ended as '{state}': {job.get('error') or '(no detail)'}. "
        f"See the job stdout log: {job.get('stdout_path')}"
    )


def _poll_job(client, job_id: str, *, label: str, with_progress: bool = False) -> dict:
    """Poll ``GET /jobs/{id}`` until terminal; return the final job dict.

    Prints state transitions tagged with ``label``. When ``with_progress`` is
    set, lazily sizes a ComfyUI ``ProgressBar`` from the job's ``total_steps``
    (read from its progress.jsonl) and advances it by ``global_step``.

    If the user hits ComfyUI's interrupt (Cancel) button while we're blocked
    here, the daemon job is detached and keeps burning the GPU regardless. So we
    poll ``processing_interrupted()`` each tick and, when set, abort the job via
    ``client.stop(job_id)`` — the same call ``make daemon-kill JOB=<id>`` makes —
    before re-raising ``InterruptProcessingException`` so ComfyUI marks the node
    cancelled.
    """
    import time

    import comfy.model_management
    import comfy.utils

    pbar = None
    total = None
    last_state = None
    while True:
        if comfy.model_management.processing_interrupted():
            print(
                f"[Anima Trainer] interrupted — aborting {label} job {job_id} "
                f"(daemon stays up)",
                flush=True,
            )
            try:
                client.stop(job_id)
            except Exception as e:  # best-effort; still surface the interrupt
                print(
                    f"[Anima Trainer] failed to stop job {job_id}: {e}",
                    flush=True,
                )
            raise comfy.model_management.InterruptProcessingException()

        try:
            job = client.get(job_id)
        except Exception as e:  # daemon hiccup — keep polling briefly
            print(f"[Anima Trainer] poll error ({label}): {e}", flush=True)
            time.sleep(2.0)
            continue

        state = job.get("state")
        if state != last_state:
            print(f"[Anima Trainer] {label} job {job_id}: {state}", flush=True)
            last_state = state

        if with_progress:
            if total is None:
                total = _read_total_steps(job.get("progress_path"))
                if total:
                    pbar = comfy.utils.ProgressBar(total)
            step = (job.get("latest") or {}).get("global_step")
            if pbar is not None and isinstance(step, int):
                pbar.update_absolute(min(step, total), total)

        if state in ("done", "error", "stopped"):
            return job
        time.sleep(1.5)


def _read_total_steps(progress_path) -> int | None:
    """Read ``total_steps`` from the run_start line of a job's progress.jsonl.

    The daemon and ComfyUI share a machine, so reading the local file is the
    cheapest way to size the progress bar without bloating the HTTP API.
    """
    if not progress_path or not os.path.isfile(progress_path):
        return None
    try:
        import json

        with open(progress_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if '"run_start"' not in line:
                    continue
                rec = json.loads(line)
                if rec.get("ev") == "run_start":
                    val = rec.get("total_steps")
                    return int(val) if val else None
    except (OSError, ValueError):
        return None
    return None


def _load_anima_model(anima_model: str):
    """Load a base Anima DiT from ComfyUI's diffusion_models folder as a MODEL.

    Replicates ComfyUI's ``UNETLoader.load_unet`` (default weight dtype) so the
    returned object is a normal ``ModelPatcher`` — identical to what the user
    would get by chaining a UNETLoader.
    """
    import comfy.sd

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", anima_model)
    return comfy.sd.load_diffusion_model(unet_path, model_options={})


def _apply_lora_to_model(model, file_path: str, strength: float) -> None:
    """Patch the trained LoRA onto ``model`` in place via ComfyUI's native path.

    The trainer only ever emits a T-LoRA, which serialises as plain LoRA keys:
    the T-LoRA rank mask is training-only (inference is full-rank). So ComfyUI's
    stock machinery — ``model_lora_keys_unet`` + ``convert_lora`` + ``load_lora``
    — maps and applies it directly, exactly as the built-in LoraLoader node would
    on a natively-loaded Anima DiT. No Anima adapter loader (HydraLoRA live
    routing) is involved, so we don't pull in the comfyui-hydralora node here.
    """
    import comfy.lora
    import comfy.lora_convert
    import comfy.utils

    lora_sd = comfy.utils.load_torch_file(file_path, safe_load=True)
    lora_sd = _fold_inv_scale(lora_sd)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    lora_sd = comfy.lora_convert.convert_lora(lora_sd)
    loaded = comfy.lora.load_lora(lora_sd, key_map)
    model.add_patches(loaded, strength)


def _fold_inv_scale(lora_sd: dict) -> dict:
    """Fold ``per_channel_scaling`` ``inv_scale`` into ``lora_down`` and drop it.

    Inert for the trainer's default T-LoRA (no ``per_channel_scaling`` →
    no ``.inv_scale`` keys). Kept as a guard so the native patcher never silently
    drops an ``.inv_scale`` suffix it doesn't recognise and applies a delta
    that's off by ``s_norm`` per input column. Mirrors ``LoRAModule.merge_to``:
    ``down *= inv_scale`` then strip the key. Returns a new dict.
    """
    inv_keys = [k for k in lora_sd if k.endswith(".inv_scale")]
    if not inv_keys:
        return lora_sd
    import torch

    out = dict(lora_sd)
    for inv_key in inv_keys:
        down_key = f"{inv_key[: -len('.inv_scale')]}.lora_down.weight"
        inv_scale = out.pop(inv_key)
        down = out.get(down_key)
        if down is None or down.dim() != 2:
            continue
        out[down_key] = down.to(torch.float) * inv_scale.to(torch.float).unsqueeze(0)
    return out


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class AnimaLoRATrainer:
    """Train an Anima LoRA (T-LoRA + OrthoLoRA) from one image + caption.

    Loads the chosen base checkpoint itself after training and returns a patched
    MODEL — use it exactly like the output of an Anima Adapter Loader.
    """

    @classmethod
    def INPUT_TYPES(cls):
        unets = folder_paths.get_filename_list("diffusion_models") or [""]
        return {
            "required": {
                "anima_model": (
                    unets,
                    {
                        "tooltip": (
                            "Base Anima DiT checkpoint (ComfyUI diffusion_models "
                            "folder). Used for training and reloaded afterwards to "
                            "produce the output MODEL."
                        )
                    },
                ),
                "image": ("IMAGE", {"tooltip": "The image(s) to train on."}),
                "text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Caption for the training image (STRING input).",
                    },
                ),
                "save_as": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Output LoRA filename (without extension), saved into "
                            "ComfyUI's loras folder. Leave blank for an "
                            "auto-timestamped name."
                        ),
                    },
                ),
                "rank": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 256,
                        "tooltip": "LoRA rank (network_dim); alpha is tied to it.",
                    },
                ),
                "epochs": (
                    "INT",
                    {"default": 25, "min": 1, "max": 10000},
                ),
                "lr": (
                    "FLOAT",
                    {
                        "default": 5e-5,
                        "min": 1e-7,
                        "max": 1e-2,
                        "step": 1e-6,
                        "round": False,
                        "tooltip": "Learning rate (network learning_rate).",
                    },
                ),
                "gpu": (["8GB", "16GB", "high"], {"default": "16GB"}),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Optional loss mask(s). White = train on this region, "
                            "black = ignore. One mask per image, or a single mask "
                            "broadcast to all. Connecting it turns on masked loss."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "train"
    CATEGORY = "anima/training"
    DESCRIPTION = (
        "Train an Anima T-LoRA + OrthoLoRA from a single image + caption, then "
        "load the chosen base checkpoint and return it with the trained LoRA "
        "applied — a drop-in for the Anima Adapter Loader's MODEL output. "
        "Optionally accepts a MASK to train with masked loss. Training runs in "
        "the local Anima daemon (errors if the daemon isn't running); the UI "
        "stays responsive and a progress bar tracks the run."
    )

    def train(
        self,
        anima_model: str,
        image,
        text: str,
        rank: int,
        epochs: int,
        lr: float,
        save_as: str,
        gpu: str,
        mask=None,
    ):
        from .training import GPU_TIER_PRESET

        anima_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", anima_model
        )

        overrides: dict = {
            "network_dim": int(rank),
            "network_alpha": float(rank),
            "max_train_epochs": int(epochs),
            "learning_rate": float(lr),
            # Train against the user-selected base checkpoint.
            "pretrained_model_name_or_path": anima_path,
        }
        # Layer the node-only training defaults *below* the explicit UI fields
        # already in `overrides` (rank/epochs/lr/model) — setdefault means
        # anything the user typed in the node still wins. The gpu dropdown only
        # selects the hardware preset (see GPU_TIER_PRESET); blocks_to_swap and
        # the checkpointing flags come from that preset + node_defaults.toml.
        for key, value in _load_node_defaults().items():
            overrides.setdefault(key, value)

        # Optional user-supplied output name; blank → auto-timestamped default in
        # `_train_and_save`. Strip any path parts / extension so it can't escape
        # the loras folder or end up double-suffixed (`x.safetensors.safetensors`).
        name = os.path.splitext(os.path.basename(save_as.strip()))[0]
        if name:
            overrides["output_name"] = name

        saved_path = _train_and_save(
            method="tlora",
            preset=GPU_TIER_PRESET[gpu],
            overrides=overrides,
            image=image,
            prompt=text,
            mask=mask,
        )

        # Load the base DiT ourselves (avoids holding it resident during the run)
        # and return it patched — a drop-in for the Anima Adapter Loader.
        model = _load_anima_model(anima_model)
        _apply_lora_to_model(model, saved_path, 1.0)
        return (model,)


class AnimaLoRATrainerFolder:
    """Train an Anima LoRA (T-LoRA + OrthoLoRA) from a folder of images + captions.

    Like ``AnimaLoRATrainer`` but reads its dataset from a directory of images
    each paired with a same-stem ``.txt`` caption sidecar, instead of a single
    connected IMAGE. Loads the chosen base checkpoint after training and returns
    a patched MODEL — a drop-in for the Anima Adapter Loader's MODEL output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        unets = folder_paths.get_filename_list("diffusion_models") or [""]
        return {
            "required": {
                "anima_model": (
                    unets,
                    {
                        "tooltip": (
                            "Base Anima DiT checkpoint (ComfyUI diffusion_models "
                            "folder). Used for training and reloaded afterwards to "
                            "produce the output MODEL."
                        )
                    },
                ),
                "dataset_dir": (
                    _input_subdirs(),
                    {
                        "tooltip": (
                            "Subfolder of ComfyUI's input/ directory holding the "
                            "training images, each with a same-stem .txt caption "
                            "sidecar next to it. Read-only (images are "
                            "bucket-resized into a temp dir). Refresh the node to "
                            "pick up newly-added folders."
                        ),
                    },
                ),
                "save_as": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Output LoRA filename (without extension), saved into "
                            "ComfyUI's loras folder. Leave blank for an "
                            "auto-timestamped name."
                        ),
                    },
                ),
                "rank": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 256,
                        "tooltip": "LoRA rank (network_dim); alpha is tied to it.",
                    },
                ),
                "epochs": (
                    "INT",
                    {"default": 25, "min": 1, "max": 10000},
                ),
                "lr": (
                    "FLOAT",
                    {
                        "default": 5e-5,
                        "min": 1e-7,
                        "max": 1e-2,
                        "step": 1e-6,
                        "round": False,
                        "tooltip": "Learning rate (network learning_rate).",
                    },
                ),
                "gpu": (["8GB", "16GB", "high"], {"default": "16GB"}),
            },
            "optional": {
                "mask_dir": (
                    _input_subdirs_optional(),
                    {
                        "tooltip": (
                            "Optional subfolder of ComfyUI's input/ directory "
                            "holding `{stem}_mask.png` loss masks (white = keep), "
                            "one per training image by matching stem. Pick "
                            "'(none)' to train without masked loss."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "train"
    CATEGORY = "anima/training"
    DESCRIPTION = (
        "Train an Anima T-LoRA + OrthoLoRA from a folder of images + caption "
        "sidecars, then load the chosen base checkpoint and return it with the "
        "trained LoRA applied — a drop-in for the Anima Adapter Loader's MODEL "
        "output. Optionally point it at a folder of `{stem}_mask.png` masks to "
        "train with masked loss. Training runs in the local Anima daemon (errors "
        "if the daemon isn't running); the UI stays responsive and a progress "
        "bar tracks the run."
    )

    def train(
        self,
        anima_model: str,
        dataset_dir: str,
        rank: int,
        epochs: int,
        lr: float,
        save_as: str,
        gpu: str,
        mask_dir: str = _MASK_NONE,
    ):
        from .training import GPU_TIER_PRESET

        anima_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", anima_model
        )

        # Resolve the chosen subfolder name to an absolute path under ComfyUI's
        # input dir. `get_input_directory` honours the same root the dropdown was
        # built from in `_input_subdirs`.
        if not dataset_dir:
            raise ValueError(
                "No dataset folder selected. Drop a folder of images + .txt "
                "captions into ComfyUI's input/ directory and refresh the node."
            )
        dataset_path = os.path.join(folder_paths.get_input_directory(), dataset_dir)

        # Optional mask folder, resolved the same way; "(none)" → no masked loss.
        mask_path = ""
        if mask_dir and mask_dir != _MASK_NONE:
            mask_path = os.path.join(folder_paths.get_input_directory(), mask_dir)

        overrides: dict = {
            "network_dim": int(rank),
            "network_alpha": float(rank),
            "max_train_epochs": int(epochs),
            "learning_rate": float(lr),
            # Train against the user-selected base checkpoint.
            "pretrained_model_name_or_path": anima_path,
        }
        # Layer the node-only training defaults *below* the explicit UI fields
        # already in `overrides` (rank/epochs/lr/model) — setdefault means
        # anything the user typed in the node still wins. The gpu dropdown only
        # selects the hardware preset (see GPU_TIER_PRESET); blocks_to_swap and
        # the checkpointing flags come from that preset + node_defaults.toml.
        for key, value in _load_node_defaults().items():
            overrides.setdefault(key, value)

        # Optional user-supplied output name; blank → auto-timestamped default in
        # `_train_and_save`. Strip any path parts / extension so it can't escape
        # the loras folder or end up double-suffixed (`x.safetensors.safetensors`).
        name = os.path.splitext(os.path.basename(save_as.strip()))[0]
        if name:
            overrides["output_name"] = name

        saved_path = _train_and_save(
            method="tlora",
            preset=GPU_TIER_PRESET[gpu],
            overrides=overrides,
            dataset_dir=dataset_path,
            mask_dir=mask_path,
        )

        # Load the base DiT ourselves (avoids holding it resident during the run)
        # and return it patched — a drop-in for the Anima Adapter Loader.
        model = _load_anima_model(anima_model)
        _apply_lora_to_model(model, saved_path, 1.0)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "AnimaLoRATrainer": AnimaLoRATrainer,
    "AnimaLoRATrainerFolder": AnimaLoRATrainerFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLoRATrainer": "Anima LoRA Trainer",
    "AnimaLoRATrainerFolder": "Anima LoRA Trainer (Folder)",
}
