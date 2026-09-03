"""Training entry-points for shipped methods (lora family + lora-gui + EasyControl).

Each ``cmd_*`` is a thin shim that translates env vars + extra argv into the
right ``train.py`` (via ``accelerate launch``) call. Experimental methods
(postfix, ip-adapter) live in ``scripts/experimental_tasks/training.py`` and are
wired up under ``make exp-*`` in ``tasks.py``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tomllib
from collections import Counter
from pathlib import Path

import toml

from ._common import (
    PY,
    ROOT,
    _preset,
    _resolve_run_mode,
    bespoke_preset_flags,
    queue_command,
    run,
    run_command,
    train,
)

# EasyControl control-task projects are descriptor-driven: a self-contained
# ``configs/easycontrol/<EASYADAPTER>.toml`` (``name`` slug + [staging]/[preprocess]/
# [training] knob tables + a [[datasets]] blueprint tail) selected via the
# EASYADAPTER env var. All train the base ``easycontrol`` method with the
# descriptor's [training] table folded in as CLI overrides. Per-adapter
# staging/preprocess bodies are registered in ``_EASY_ADAPTERS`` below.


def _easyadapter() -> str:
    """Resolve the EASYADAPTER env var (validated). "" → default easycontrol."""
    adapter = (os.environ.get("EASYADAPTER") or "").strip()
    if adapter and adapter not in _EASY_ADAPTERS:
        raise SystemExit(
            f"Unknown EASYADAPTER={adapter!r}. Known: {sorted(_EASY_ADAPTERS)}."
        )
    return adapter


def cmd_lora(extra):
    train("lora", extra)


def cmd_register(extra):
    """Register-token adapter on a FROZEN Anima DiT (_archive/proposals/headroom_register_tokens.md).

    DSR-style register tokens inserted at ``insert_block`` (default 8) plus a
    trained self-attn QKV surface (``networks/methods/register.py`` /
    ``configs/methods/register.toml``). Compile is supported (train.py widens
    the dynamic-seq bound by K); block swap stays forced off. Override knobs
    via ``--network_args`` or the config::

        make register                                     # K36 @ block 8, unfrozen QKV, arm B
        make register ARGS="--network_args num_registers=16 qkv_mode=lora"
        make register ARGS="--network_args num_registers=0"   # LoRA-only drift control (arm L)
        make register ARGS="--network_args insert_block=0"    # entry insertion (RQ3 geometry)

    Inference is the ComfyUI node ``custom_nodes/comfyui-anima-register`` (kept
    live — register tokens can't merge into DiT weights)."""
    train("register", extra)


def cmd_turbo(extra):
    """Turbo Anima — DP-DMD distillation (docs: docs/methods/turbo.md).

    Bypasses train.py / accelerate (single-GPU bespoke loop).
    Reads ``configs/methods/turbo.toml``; trailing args are forwarded so user
    CLI flags override TOML values, e.g.::

        make turbo                                  # defaults: rank=64, 4-step
        make turbo ARGS="--student_rank 64 --iterations 5000"
        make turbo ARGS="--single_prompt_idx 0"     # Phase 0 single-prompt overfit
        make turbo --queue                          # enqueue on the daemon

    The output is a normal LoRA — a distilled student ships at
    https://huggingface.co/sorryhyun/anima-turbo-4step (infer with
    ``make test-turbo`` / ``--infer_steps 4 --cfg 1.0``).

    Honors ``PRESET`` (default ``default``) — translates ``blocks_to_swap`` and
    ``gradient_checkpointing`` into CLI flags (``PRESET=low_vram`` enables
    grad ckpt + unsloth offload; ``PRESET=half/quarter/tenth`` shrinks the
    dataset via ``--sample_ratio``). ``extra`` is appended last, so CLI wins.

    ``--queue`` anywhere in ``extra`` enqueues the distillation as a daemon
    command-job and returns immediately instead of running inline. Job is
    labeled ``turbo`` so the GUI's Turbo tab can re-attach to it.
    """
    extra = list(extra or [])
    preset_flags = bespoke_preset_flags(_preset())
    argv = ["-m", "scripts.distill_turbo.distill", *preset_flags, *extra]
    if "--queue" in argv:
        argv.remove("--queue")
        queue_command("turbo", argv)
        return
    run([PY, *argv])


def cmd_soup(extra):
    """Uncond-init soup training (docs: bench/memorization/report.md).

    One pipeline (``scripts/soup/pipeline.py``): a short uncond inter-train on a
    diluted pool (reused if the checkpoint already exists) → 3 seeded captioned
    fine-tunes from that init → an exact ΔW soup SVD-truncated back to the
    method's ``network_dim``. Output:
    ``output/ckpt/anima_soup_<slug>.safetensors`` (+ ``.snapshot.toml``)::

        make soup                                     # uses soup.toml path_pattern
        make soup PATH_PATTERN="sincos/*"             # attach-by-default
        make soup TARGET=sincos                       # shorthand for "sincos/*"
        make soup PATH_PATTERN="a/*|b/*" NAME=ab --queue
        make soup ARTISTS_SHARD=1_6                   # round-robin artist shard
        make soup CUSTOM=soup                         # gui-methods/custom/soup.toml
        make soup TARGET=sincos ARGS="--network_dim 32 --max_train_epochs 8"

    Selection is a fnmatch **path_pattern** (``|`` = alternatives, matched
    against each image's path relative to its subset image_dir) rather than a
    single artist dir. ``ARTISTS_SHARD=k_N`` is an alternative selector — one
    round-robin shard of the artist subdirs (slug ``shard1of6``); mutually
    exclusive with ``PATH_PATTERN``/``TARGET``. With none set it falls back to
    the top-level ``path_pattern`` / ``artists_shard`` in the soup config.
    ``TARGET`` is shorthand: ``TARGET=x`` ⇒ ``PATH_PATTERN="x/*" NAME=x``.
    ``ARGS`` reaches the fine-tune runs. Env knobs (all optional): ``NAME``
    (output slug), ``POOL_PATH_PATTERN`` (Phase-1 uncond pool glob, default
    "*" = whole dataset), ``UNCOND_RATIO`` (0.5), ``UNCOND_EPOCHS`` (2),
    ``NUM_SOUP`` (3 — fine-tunes to soup, seeds 1001..1000+N), ``LR_POOL``
    (per-ingredient LRs, e.g. ``"1e-5,2e-5,4e-5"``, cycled) or
    ``LR_INTERVAL`` (``"1e-5:4e-5"``, geometric over NUM_SOUP points),
    ``RANK`` (default network_dim), ``PRESET``.
    ``ARGS`` normally reaches the fine-tunes only; ``--sigma_lowres*`` is the
    exception — it is a whole-pipeline data-routing knob, so it is replayed onto
    the uncond run too and folded into that checkpoint's name (a sigma soup
    can't reuse an init trained without it).
    Submitted as ONE daemon command job — the pipeline runs train.py
    subprocesses directly (nested daemon submission would deadlock the serial
    queue).
    """
    pattern = os.environ.get("PATH_PATTERN")
    shard = os.environ.get("ARTISTS_SHARD")
    name = os.environ.get("NAME")
    target = os.environ.get("TARGET")
    if not pattern and target:
        pattern = f"{target}/*"
        name = name or target
    argv = [
        "-m",
        "scripts.soup.pipeline",
        "--preset",
        _preset(),
    ]
    # CUSTOM=<file> runs an ad-hoc soup config from configs/gui-methods/custom/
    # (shared scratch dir, kept out of the tracked configs/soup/soup.toml).
    # The file seeds BOTH the [soup] pipeline knobs and the fine-tune method config.
    custom = os.environ.get("CUSTOM")
    if custom:
        stem = Path(custom).stem
        expected = ROOT / "configs" / "gui-methods" / "custom" / f"{stem}.toml"
        if not expected.exists():
            available = (
                sorted(p.stem for p in expected.parent.glob("*.toml"))
                if expected.parent.is_dir()
                else []
            )
            print(
                f"Unknown custom soup config: {custom!r} (looked for {expected})\n"
                f"Available in configs/gui-methods/custom/: "
                f"{', '.join(available) or '(none)'}",
                file=sys.stderr,
            )
            sys.exit(1)
        argv += ["--config", str(expected)]
    # No PATH_PATTERN/TARGET → the pipeline falls back to the top-level
    # path_pattern in configs/soup/soup.toml (its argparse default).
    if pattern:
        argv += ["--path_pattern", pattern]
    if shard:
        argv += ["--artists_shard", shard]
    if name:
        argv += ["--name", name]
    if os.environ.get("POOL_PATH_PATTERN"):
        argv += ["--pool_path_pattern", os.environ["POOL_PATH_PATTERN"]]
    for env, flag in (
        ("UNCOND_RATIO", "--uncond_ratio"),
        ("UNCOND_EPOCHS", "--uncond_epochs"),
        ("UNCOND_INIT", "--uncond_init"),
        ("NUM_SOUP", "--num_soup"),
        ("LR_POOL", "--lr_pool"),
        ("LR_INTERVAL", "--lr_interval"),
        ("RANK", "--rank"),
    ):
        if os.environ.get(env):
            argv += [flag, os.environ[env]]

    mode, extra = _resolve_run_mode(list(extra or []))
    run_command("soup", [*argv, *extra], mode=mode)


def cmd_lora_gui(extra):
    """Train from configs/gui-methods/<variant>.toml.

    Variant is taken from GUI_PRESETS env var, falling back to the first
    positional extra arg (``python tasks.py lora-gui tlora ...``), then to
    ``lora`` (plain). Extra args after the variant are forwarded as usual.

    ``CUSTOM=<file>`` runs an ad-hoc config from ``configs/gui-methods/custom/``
    instead — a scratch tree kept out of the tracked per-variant files, for
    hand-edited one-offs::

        make lora-gui CUSTOM=lora.toml     # → configs/gui-methods/custom/lora.toml

    The file stem is the method name and the subdir is ``gui-methods/custom``
    (so ``--method`` auto-discovers it). ``.toml`` is optional. CUSTOM wins over
    GUI_PRESETS when both are set.
    """
    custom = os.environ.get("CUSTOM")
    if custom:
        stem = Path(custom).stem
        subdir = ROOT / "configs" / "gui-methods" / "custom"
        expected = subdir / f"{stem}.toml"
        if not expected.exists():
            available = (
                sorted(p.stem for p in subdir.glob("*.toml")) if subdir.is_dir() else []
            )
            print(
                f"Unknown custom config: {custom!r} (looked for {expected})\n"
                f"Available in configs/gui-methods/custom/: "
                f"{', '.join(available) or '(none)'}",
                file=sys.stderr,
            )
            sys.exit(1)
        train(stem, extra, methods_subdir="gui-methods/custom")
        return

    variant = os.environ.get("GUI_PRESETS")
    if not variant and extra and not extra[0].startswith("-"):
        variant = extra[0]
        extra = extra[1:]
    variant = variant or "lora"

    expected = ROOT / "configs" / "gui-methods" / f"{variant}.toml"
    if not expected.exists():
        available = sorted(
            p.stem for p in (ROOT / "configs" / "gui-methods").glob("*.toml")
        )
        print(
            f"Unknown gui-methods variant: {variant!r}\n"
            f"Available: {', '.join(available)}",
            file=sys.stderr,
        )
        sys.exit(1)

    train(variant, extra, methods_subdir="gui-methods")


def _toml_table_to_argv(table: dict) -> list[str]:
    """Flatten a flat TOML table into ``--key value`` train.py argv.

    Bools become bare ``--flag`` when true (omitted when false); lists spread
    into ``--key v1 v2``; scalars become ``--key str(value)``.
    """
    argv: list[str] = []
    for key, val in table.items():
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        elif isinstance(val, (list, tuple)):
            argv.append(flag)
            argv.extend(str(v) for v in val)
        else:
            argv.append(flag)
            argv.append(str(val))
    return argv


def _easy_cfg_path(adapter: str) -> Path:
    """The descriptor file for an EasyControl adapter project."""
    return ROOT / "configs" / "easycontrol" / f"{adapter}.toml"


def _easy_load(adapter: str) -> tuple[dict, str, str]:
    """Load ``configs/easycontrol/<adapter>.toml`` → ``(cfg, name, base)``.

    The top-level ``name`` key (default = the file stem) is the single source
    of truth for the run: it picks the ``post_image_dataset/easycontrol/<name>/``
    base tree and the ``anima_easycontrol_<name>`` output_name default.
    Explicit ``[preprocess]`` path keys / ``[training].output_name`` still win
    if present (back-compat).
    """
    path = _easy_cfg_path(adapter)
    if not path.is_file():
        raise SystemExit(
            f"{path} not found — run `make easycontrol-staging "
            f"EASYADAPTER={adapter}` first to materialize the staging tree."
        )
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    name = str(cfg.get("name") or path.stem).strip()
    base = f"post_image_dataset/easycontrol/{name}"
    return cfg, name, base


def _resolve_blueprint_path(path: str, name: str) -> str:
    """Resolve a blueprint subset path against the current ``name`` slug.

    Interpolates the ``{name}`` placeholder the miner writes into the
    blueprint tail; falls back to swapping the ``<slug>`` component of an
    older ``post_image_dataset/easycontrol/<slug>/...`` path. Non-matching
    (custom) paths are left untouched.
    """
    path = path.replace("{name}", name)
    parts = Path(path).parts
    if len(parts) >= 3 and parts[:2] == ("post_image_dataset", "easycontrol"):
        return str(Path(*parts[:2], name, *parts[3:]))
    return path


def _easy_train_extra(adapter: str, extra) -> list[str]:
    """Build train.py extra-argv for an ``EASYADAPTER=<adapter>`` descriptor run.

    ``configs/easycontrol/<adapter>.toml`` is a multi-purpose file, but
    train.py's dataset-config validator rejects any top-level key outside the
    blueprint. So we extract just the blueprint sections into a clean
    generated sidecar (``--dataset_config``), then fold the optional
    ``[training]`` table (``output_name`` defaulting to
    ``anima_easycontrol_<name>``) into CLI overrides. User ``extra`` argv is
    appended last so it still wins.
    """
    cfg, name, base = _easy_load(adapter)
    blueprint = {k: cfg[k] for k in ("general", "datasets") if k in cfg}
    if not blueprint.get("datasets"):
        raise SystemExit(
            f"{_easy_cfg_path(adapter)} has no [[datasets]] blueprint — run "
            f"`make easycontrol-staging EASYADAPTER={adapter}` first."
        )

    # Resolve subset paths against the current `name` slug so a `name` change
    # reroutes training.
    for ds in blueprint.get("datasets", []):
        for s in ds.get("subsets", []):
            for key in (
                "image_dir",
                "cache_dir",
                "cond_cache_dir",
                "text_cache_dir",
                "latent_cache_dir",
            ):
                if key in s:
                    s[key] = _resolve_blueprint_path(s[key], name)

    # Write the blueprint-only dataset config under the slug base dir. Stable
    # path so --queue can re-read it later; regenerated each invocation.
    base_dir = ROOT / base
    base_dir.mkdir(parents=True, exist_ok=True)
    ds_path = base_dir / "dataset_config.toml"
    ds_path.write_text(
        f"# AUTO-GENERATED from configs/easycontrol/{adapter}.toml — do not edit.\n"
        "# Blueprint-only copy (train.py's dataset-config validator rejects the\n"
        "# name + [staging]/[preprocess]/[training] knobs in the source file).\n\n"
        + toml.dumps(blueprint),
        encoding="utf-8",
    )

    # output_name defaults to the name-derived slug; explicit [training].output_name wins.
    training = dict(cfg.get("training") or {})
    training.setdefault("output_name", f"anima_easycontrol_{name}")

    return [
        "--dataset_config",
        str(ds_path),
        *_toml_table_to_argv(training),
        *list(extra or []),
    ]


def cmd_easycontrol(extra):
    """EasyControl. ``EASYADAPTER=<name>`` selects a control-task project
    described by ``configs/easycontrol/<name>.toml`` (e.g. ``colorize`` or
    ``near_twins``); unset → the default ref==target easycontrol.toml.

    A descriptor run always trains the base ``easycontrol`` method, folding the
    descriptor's blueprint tail in via ``--dataset_config`` and its optional
    ``[training]`` table in as CLI overrides (see ``_easy_train_extra``)."""
    adapter = _easyadapter()
    if adapter in _EASY_ADAPTERS:
        train("easycontrol", _easy_train_extra(adapter, extra))
        return
    train("easycontrol", extra)


def _resize_tree(
    src: str,
    dst: str,
    *,
    min_pixels: int,
    target_res: tuple[int, ...],
    recursive: bool = True,
    freefit_max_ratio=None,
) -> None:
    """Resize a staged tree into buckets — the ``anime_tools`` resize stage as
    a ``ResizeRequest`` (the same one ``make preprocess-resize`` runs), for the
    EasyControl pair trees whose knobs come from a descriptor TOML rather than
    the config chain. Captions are never copied: TE reads the source tree."""
    from anime_tools.stages.requests import ResizeRequest

    from ._common import execute_stage, stage_by_id

    fields: dict = {
        "src": src,
        "dst": dst,
        "min_pixels": int(min_pixels),
        "recursive": bool(recursive),
        "copy_captions": False,
    }
    if target_res and tuple(target_res) != (1024,):
        fields["target_res"] = tuple(target_res)
    if freefit_max_ratio is not None:
        fields["freefit_max_ratio"] = float(freefit_max_ratio)
    execute_stage(stage_by_id("resize"), ResizeRequest(**fields))


def _near_twins_preprocess(adapter: str, cfg: dict, base: str, extra) -> None:
    """Resize + VAE/TE caching for the mined near-twin pair tree.

    Every knob is read from the ``[preprocess]`` table of
    ``configs/easycontrol/near_twins.toml``. The staging/resized/cache/cond
    dirs default to ``post_image_dataset/easycontrol/<name>/{staging,resized,
    cache,cond}``; an explicit path key still overrides.

    The mined ``staging/`` tree holds native-resolution images (symlinks to
    the corpus), so this first resizes them into buckets under ``resized/``
    (the training ``image_dir``), then VAE/TE-encodes into ``cache/``.
    """
    pp = cfg.get("preprocess") or {}
    staging = pp.get("image_dir", f"{base}/staging")
    resized = pp.get("resized_dir", f"{base}/resized")
    cache = pp.get("cache_dir", f"{base}/cache")
    recursive = ["--recursive"] if pp.get("recursive", True) else []
    # Bucket tiers: descriptor's [preprocess].target_res wins, else base.toml's
    # target_res; final fallback [1024]. CAVEAT (free-fit): a pair whose
    # members free-fit to different shapes is cross-shape-paired (or dropped if
    # truly unpaired) — see _near_twins_build_cond.
    target_res = pp.get("target_res")
    if target_res is None:
        from ._common import _path_overrides

        target_res = _path_overrides().get("target_res", [1024])
    if not isinstance(target_res, (list, tuple)):
        target_res = [target_res]

    # Resize staging tree into buckets. min_pixels defaults to 0 (not 0.5MP) so a
    # small member can't be dropped and orphan its pair partner.
    _resize_tree(
        staging,
        resized,
        min_pixels=int(pp.get("min_pixels", 0)),
        target_res=tuple(int(e) for e in target_res),
        recursive=bool(recursive),
        freefit_max_ratio=pp.get("freefit_max_ratio"),
    )
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
            "--dir",
            resized,
            "--cache_dir",
            cache,
            "--vae",
            pp.get("vae", "models/vae/qwen_image_vae.safetensors"),
            "--batch_size",
            str(pp.get("batch_size", 4)),
            "--chunk_size",
            str(pp.get("chunk_size", 64)),
            *recursive,
        ]
    )
    # Text-encoder outputs (captions copied during resize).
    run(
        [
            PY,
            "scripts/preprocess/cache_text_embeddings.py",
            "--dir",
            resized,
            "--cache_dir",
            cache,
            "--qwen3",
            pp.get("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors"),
            "--dit",
            pp.get("dit", "models/diffusion_models/anima-base-v1.0.safetensors"),
            "--caption_shuffle_variants",
            str(pp.get("caption_shuffle_variants", 4)),
            "--caption_tag_dropout_rate",
            str(pp.get("caption_tag_dropout_rate", 0.1)),
            *recursive,
        ]
    )
    # Optional REPA vision-encoder sidecars, gated on [preprocess] pe_encoder so
    # plain runs skip the encoder pass. Idempotent (pre-skips cached).
    pe_encoder = pp.get("pe_encoder")
    if pe_encoder:
        run(
            [
                PY,
                "scripts/preprocess/cache_pe_encoder.py",
                "--dir",
                resized,
                "--cache_dir",
                cache,
                "--encoder",
                str(pe_encoder),
                *recursive,
            ]
        )
    # Pair the cond/ tree (the _tags reference latent for each _no_tags target).
    _near_twins_build_cond(pp, base)


def _near_twins_build_cond(pp: dict, base: str) -> None:
    """Materialize the ``cond/`` latent tree for the near-twins *removal* task.

    Pairing convention: the denoising target is the clean ``_no_tags`` member;
    its ``_tags`` twin is the EasyControl condition reference. The loader
    resolves the cond latent by the *target* stem under ``cond_cache_dir``, so
    for each ``{id}_no_tags_{WxH}_anima.npz`` target we symlink the sibling
    ``{id}_tags_{W'xH'}_anima.npz`` into
    ``cond/<artist>/{id}_no_tags_{W'xH'}_anima.npz`` (filed at the *twin's*
    bucket). Under free-fit a twin may land at a different shape (W'xH' ≠ WxH)
    and is still paired — cond≠target shapes are supported (cond_diff_loss
    self-skips on a mismatch). A truly unpaired target is skipped with a
    warning.

    Pure symlinks over the existing cache; the tree is rebuilt from scratch
    each run so a dropped pair can't leave a stale link behind.
    """
    cache_dir = ROOT / pp.get("cache_dir", f"{base}/cache")
    cond_dir = ROOT / pp.get("cond_dir", f"{base}/cond")
    if not cache_dir.is_dir():
        raise SystemExit(
            f"{cache_dir} not found — run the VAE/TE caching pass first "
            "(`make easycontrol-preprocess EASYADAPTER=near_twin`)."
        )
    if cond_dir.exists():
        shutil.rmtree(cond_dir)

    pat = re.compile(r"^(?P<id>.+)_no_tags_(?P<bucket>\d{4}x\d{4})_anima\.npz$")
    twin_pat = re.compile(r"_tags_(?P<bucket>\d{4}x\d{4})_anima\.npz$")
    linked = skipped = crossfit = 0
    for npz in sorted(cache_dir.rglob("*_no_tags_*_anima.npz")):
        m = pat.match(npz.name)
        if not m:
            continue
        # Prefer the same-bucket twin (zero-cost, the constant-bucket common case).
        twin = npz.with_name(f"{m['id']}_tags_{m['bucket']}_anima.npz")
        twin_bucket = m["bucket"]
        if not twin.is_file():
            # Free-fit may land the _tags twin at a different shape — fine,
            # cond≠target is supported. Pair at whatever bucket it has.
            cands = sorted(npz.parent.glob(f"{m['id']}_tags_*_anima.npz"))
            if not cands:
                print(
                    f"  [near_twin cond] no _tags twin for "
                    f"{npz.relative_to(cache_dir)} — skipping (truly unpaired).",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            twin = cands[0]
            twin_bucket = twin_pat.search(twin.name)["bucket"]
            crossfit += 1
        # File under the target stem at the TWIN's actual bucket so the loader's
        # stem-glob fallback resolves it. Same-bucket case: name is unchanged.
        link = (
            cond_dir
            / npz.relative_to(cache_dir).parent
            / f"{m['id']}_no_tags_{twin_bucket}_anima.npz"
        )
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(twin.resolve())
        linked += 1
    print(
        f"[near_twins cond] linked {linked} cond latents into {cond_dir}"
        + (f" ({crossfit} cross-shape under free-fit)" if crossfit else "")
        + (f" ({skipped} skipped)" if skipped else "")
    )


def _near_twins_stage(adapter: str, cfg: dict, base: str, extra) -> None:
    """Mine the in-artist near-twin pair tree (the near_twins/sanitize staging step).

    The miner self-reads its ``[staging]`` table + ``name`` slug from the
    descriptor and rewrites the blueprint tail back into the *same* file, so
    both ``--config``/``--config-out`` point at
    ``configs/easycontrol/<adapter>.toml`` (else a ``sanitize`` run would mine
    into the miner's near_twins.toml default). ``cfg``/``base`` are unused —
    the signature just matches the registry contract.
    """
    cfg_path = str(_easy_cfg_path(adapter))
    run(
        [
            PY,
            "-m",
            "easycontrol_adapters.tools.near_twins",
            "--config",
            cfg_path,
            "--config-out",
            cfg_path,
            *extra,
        ]
    )


def _colorize_prep_paths(base: str) -> list[str]:
    """Slug-derived prep.py path flags so ``name`` reroutes colorize's trees.

    ``--src`` stays the shared color corpus (``post_image_dataset/resized``);
    only the synthetic staging tree + cond/text/target caches ride the slug.
    Injected before the descriptor knob tables so a ``[staging]``/
    ``[preprocess]`` key (or user ``extra``) still wins via last-flag
    precedence."""
    return [
        "--staging",
        f"{base}/staging",
        "--cond_cache_dir",
        f"{base}/cond",
        "--text_cache_dir",
        f"{base}/text",
        "--target_cache_dir",
        f"{base}/target",
    ]


def _colorize_stage(adapter: str, cfg: dict, base: str, extra) -> None:
    """Colorize staging: synthesize the synthetic B&W manga condition tree.

    Runs only prep.py's mangafy stage over the shared color corpus into
    ``{base}/staging``. Knobs come from the descriptor's ``[staging]`` table.
    The cond-latent + color-only-text caching is the separate preprocess pass."""
    knobs = _toml_table_to_argv(cfg.get("staging") or {})
    run(
        [
            PY,
            "easycontrol_adapters/colorization/prep.py",
            "--skip_encode",
            "--skip_target",
            "--skip_text",
            *_colorize_prep_paths(base),
            *knobs,
            *list(extra or []),
        ]
    )


def _colorize_preprocess(adapter: str, cfg: dict, base: str, extra) -> None:
    """Colorize preprocess: cache cond latents + color-only text over the staged tree.

    Runs prep.py's encode + color-text stages. The color *target* latents + TE
    are reused from the shared LoRA cache. Knobs come from the descriptor's
    ``[preprocess]`` table; pass ``ARGS="--no-skip_mangafy"`` to re-stage inline."""
    knobs = _toml_table_to_argv(cfg.get("preprocess") or {})
    run(
        [
            PY,
            "easycontrol_adapters/colorization/prep.py",
            "--skip_mangafy",
            *_colorize_prep_paths(base),
            *knobs,
            *list(extra or []),
        ]
    )


def _inpaint_prep_paths(base: str) -> list[str]:
    """Slug-derived prep.py path flags so ``name`` reroutes inpaint's trees.

    ``--src`` stays the shared corpus (``post_image_dataset/resized``); only
    the masked staging tree + cond/text caches ride the slug. Injected before
    the descriptor knob tables so a ``[staging]``/``[preprocess]`` key (or
    user ``extra``) still wins via last-flag precedence."""
    return [
        "--staging",
        f"{base}/staging",
        "--cond_cache_dir",
        f"{base}/cond",
        "--text_cache_dir",
        f"{base}/text",
    ]


def _inpaint_stage(adapter: str, cfg: dict, base: str, extra) -> None:
    """Inpaint staging: synthesize the gray-masked condition tree.

    Runs only prep.py's mask stage over the shared corpus into
    ``{base}/staging``. Knobs come from the descriptor's ``[staging]`` table.
    The cond-latent + caption-text caching is the separate preprocess pass."""
    knobs = _toml_table_to_argv(cfg.get("staging") or {})
    run(
        [
            PY,
            "easycontrol_adapters/inpainting/prep.py",
            "--skip_encode",
            "--skip_text",
            *_inpaint_prep_paths(base),
            *knobs,
            *list(extra or []),
        ]
    )


def _inpaint_preprocess(adapter: str, cfg: dict, base: str, extra) -> None:
    """Inpaint preprocess: cache cond latents over the masked staging tree.

    Runs prep.py's encode + caption-text stages. Target latents are reused
    from the shared LoRA cache (no re-encode); text stage re-caches full
    captions into ``{base}/text``. Pass ``ARGS="--no-skip_mask"`` to re-stage
    inline, or ``--skip_text`` to reuse the shared LoRA TE cache instead."""
    knobs = _toml_table_to_argv(cfg.get("preprocess") or {})
    run(
        [
            PY,
            "easycontrol_adapters/inpainting/prep.py",
            "--skip_mask",
            *_inpaint_prep_paths(base),
            *knobs,
            *list(extra or []),
        ]
    )


def _region_stage(adapter: str, cfg: dict, base: str, extra) -> None:
    """Region staging: solo-1girl + 1girl1boy targets → SAM3 masks → paint cond tree.

    Runs prep.py's select + sam + cond + captions stages (GPU: SAM3) into ``{base}``.
    Knobs come from the descriptor's ``[staging]`` table. The cond-latent
    caching is the separate preprocess pass."""
    knobs = _toml_table_to_argv(cfg.get("staging") or {})
    run(
        [
            PY,
            "easycontrol_adapters/region/prep.py",
            "--skip_encode",
            "--skip_text",
            "--base",
            base,
            *knobs,
            *list(extra or []),
        ]
    )


def _region_preprocess(adapter: str, cfg: dict, base: str, extra) -> None:
    """Region preprocess: VAE-cache the painted cond tree into ``{base}/cond`` and
    TE-cache the staged flat + positioned caption variants into ``{base}/text``.

    Target latents/PE are reused from the shared LoRA cache (no re-encode)."""
    knobs = _toml_table_to_argv(cfg.get("preprocess") or {})
    run(
        [
            PY,
            "easycontrol_adapters/region/prep.py",
            "--skip_select",
            "--skip_sam",
            "--skip_cond",
            "--skip_captions",
            "--base",
            base,
            *knobs,
            *list(extra or []),
        ]
    )


def _subject_stage(adapter: str, cfg: dict, base: str, extra) -> None:
    """Subject staging: mine cross-image same-character pairs (directedit_ec Phase 2).

    ``easycontrol_adapters/tools/subject_pairs.py`` reads its ``[staging]``
    table and rewrites the blueprint tail in place (near_twins contract).
    CPU-only: staging is symlinks + a ``pairs.json`` manifest + the ``cond/``
    latent tree — no encode pass (latents/TE reused from the shared cache)."""
    cfg_path = str(_easy_cfg_path(adapter))
    run(
        [
            PY,
            "-m",
            "easycontrol_adapters.tools.subject_pairs",
            "--config",
            cfg_path,
            "--config-out",
            cfg_path,
            *extra,
        ]
    )


def _subject_preprocess(adapter: str, cfg: dict, base: str, extra) -> None:
    """Subject preprocess: rebuild ``cond/`` from ``pairs.json`` (idempotent).

    Nothing to encode — re-run this after a shared-corpus re-preprocess so cond
    symlinks track any bucket moves."""
    run(
        [
            PY,
            "-m",
            "easycontrol_adapters.tools.subject_pairs",
            "--config",
            str(_easy_cfg_path(adapter)),
            "--cond-only",
            *extra,
        ]
    )


def _subject_edit_stage(adapter: str, cfg: dict, base: str, extra) -> None:
    """Subject-edit staging: mine delta-caption edit pairs (directedit_ec Phase 2.5).

    ``easycontrol_adapters/tools/subject_edit_pairs.py`` — subject_pairs
    contract, but the staged ``.txt`` files are REAL files holding the tag
    delta vs the cond partner, not caption symlinks. Still CPU-only; the TE
    encode over the delta captions is the preprocess step."""
    cfg_path = str(_easy_cfg_path(adapter))
    run(
        [
            PY,
            "-m",
            "easycontrol_adapters.tools.subject_edit_pairs",
            "--config",
            cfg_path,
            "--config-out",
            cfg_path,
            *extra,
        ]
    )


def _subject_edit_preprocess(adapter: str, cfg: dict, base: str, extra) -> None:
    """Subject-edit preprocess: rebuild ``cond/`` + TE-encode the delta captions.

    The delta captions are new text, so the shared LoRA TE cache does not
    apply — ``cache_text_embeddings.py`` runs over ``{base}/staging`` into
    ``{base}/text``. VAE latents + PE ride the shared cache untouched. Re-run
    after re-mining (pass ``--overwrite`` — delta texts change but stems
    don't, and the existence check can't see that)."""
    run(
        [
            PY,
            "-m",
            "easycontrol_adapters.tools.subject_edit_pairs",
            "--config",
            str(_easy_cfg_path(adapter)),
            "--cond-only",
        ]
    )
    from library.env import default_checkpoints

    ck = default_checkpoints()
    knobs = _toml_table_to_argv(cfg.get("preprocess") or {})
    run(
        [
            PY,
            "scripts/preprocess/cache_text_embeddings.py",
            "--dir",
            f"{base}/staging",
            "--recursive",
            "--cache_dir",
            f"{base}/text",
            "--qwen3",
            ck.text_encoder,
            "--dit",
            ck.dit,
            *knobs,
            *list(extra or []),
        ]
    )


def _phash_edit_preprocess(adapter: str, cfg: dict, base: str, extra) -> None:
    """Resize + VAE/TE caching for the phash-mined edit-pair pool.

    Differs from :func:`_near_twins_preprocess` in what gets *encoded*. There a
    pair tree is staged per pair, so a member that joins several pairs (and both
    directions of each) is resized and VAE-encoded once per view — 7,424 encodes
    over 2,722 distinct images at the shipped phash_edit knobs. Here the miner
    stages a deduplicated ``pool/`` instead, and the pair views are symlinks:

      1. purge the derived pair links (so the VAE pass only ever sees the pool)
      2. resize ``pool/`` → ``resized/``            [distinct images]
      3. VAE-cache ``resized/`` → ``cache/``        [distinct images]
      4. link the pair views: ``resized/{pair}_no_tags.<ext>`` + its delta
         caption, the target latent under the pair stem, the cond latent under
         the same stem in ``cond/``
      5. TE-encode ``--path_pattern '*_no_tags.*'`` [delta captions only]

    A latent depends only on the image, so step 3 is the whole saving; the only
    pair-specific artifact is the caption, which step 5 encodes exactly once per
    directed pair. Step 1 matters on re-runs: a stale link left in ``resized/``
    would otherwise be handed to the VAE pass as if it were a pool member.
    """
    pp = cfg.get("preprocess") or {}
    pool = pp.get("pool_dir", f"{base}/pool")
    resized = pp.get("resized_dir", f"{base}/resized")
    cache = pp.get("cache_dir", f"{base}/cache")
    cond = pp.get("cond_dir", f"{base}/cond")
    # Synthetic colorize arm: selected targets symlinked into mono_src/, mangafied
    # into mono/, VAE-encoded into mono_cache/ — the cond side of those records.
    mono_src = pp.get("mono_src_dir", f"{base}/mono_src")
    mono = pp.get("mono_dir", f"{base}/mono")
    mono_cache = pp.get("mono_cache_dir", f"{base}/mono_cache")
    manifest = ROOT / pp.get("manifest", f"{base}/pairs.json")
    recursive = ["--recursive"] if pp.get("recursive", True) else []

    if not manifest.is_file():
        raise SystemExit(
            f"{manifest} not found — run `make easycontrol-staging "
            f"EASYADAPTER={adapter}` first to mine the pair pool."
        )

    target_res = pp.get("target_res")
    if target_res is None:
        from ._common import _path_overrides

        target_res = _path_overrides().get("target_res", [1024])
    if not isinstance(target_res, (list, tuple)):
        target_res = [target_res]

    _phash_edit_purge_links(resized, cond, mono_src, mono, mono_cache)
    _resize_tree(
        pool,
        resized,
        min_pixels=int(pp.get("min_pixels", 0)),
        target_res=tuple(int(e) for e in target_res),
        recursive=bool(recursive),
    )
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
            "--dir",
            resized,
            "--cache_dir",
            cache,
            "--vae",
            pp.get("vae", "models/vae/qwen_image_vae.safetensors"),
            "--batch_size",
            str(pp.get("batch_size", 4)),
            "--chunk_size",
            str(pp.get("chunk_size", 64)),
            *recursive,
        ]
    )
    pe_encoder = pp.get("pe_encoder")
    if pe_encoder:
        run(
            [
                PY,
                "scripts/preprocess/cache_pe_encoder.py",
                "--dir",
                resized,
                "--cache_dir",
                cache,
                "--encoder",
                str(pe_encoder),
                *recursive,
            ]
        )
    _phash_edit_colorize_cond(manifest, resized, mono_src, mono, mono_cache, pp)
    _phash_edit_build_links(manifest, resized, cache, cond, mono_cache)
    run(
        [
            PY,
            "scripts/preprocess/cache_text_embeddings.py",
            "--dir",
            resized,
            "--cache_dir",
            cache,
            "--path_pattern",
            "*_no_tags.*",  # pool members are never targets and carry no caption
            "--qwen3",
            pp.get("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors"),
            "--dit",
            pp.get("dit", "models/diffusion_models/anima-base-v1.0.safetensors"),
            "--caption_shuffle_variants",
            str(pp.get("caption_shuffle_variants", 4)),
            "--caption_tag_dropout_rate",
            str(pp.get("caption_tag_dropout_rate", 0.0)),
            *recursive,
        ]
    )


def _phash_edit_purge_links(resized: str, cond: str, *derived: str) -> None:
    """Drop every derived pair view, leaving the resized pool itself intact.

    ``derived`` are whole trees rebuilt from scratch each run (the mangafied
    colorize cond staging + its latent cache): the colorize selection moves when
    the mine does, so a surviving tree would silently keep encoding images no
    record references any more.
    """
    resized_dir, cond_dir = ROOT / resized, ROOT / cond
    for d in (cond_dir, *(ROOT / x for x in derived)):
        if d.exists():
            shutil.rmtree(d)
    n = 0
    if resized_dir.is_dir():
        for p in resized_dir.rglob("*_no_tags.*"):
            p.unlink()
            n += 1
    if n:
        print(f"[phash_edit] purged {n} stale pair views from {resized}")


_PHASH_IMG_EXTS = (".png", ".webp", ".jpg", ".jpeg")


def _phash_edit_colorize_cond(
    manifest: Path, resized: str, mono_src: str, mono: str, mono_cache: str, pp: dict
) -> int:
    """Synthesize + encode the cond side of the ``kind="colorize"`` records.

    Those records have no mined partner: the condition is a **mangafied B&W**
    view (XDoG + screentone, ``easycontrol_adapters/colorization/prep.py``) of
    the target itself. Mangafying the already-bucketed ``resized/`` image keeps
    the cond at the target's exact shape for free.

    Only the selected targets are staged — a symlink tree in ``mono_src/`` — so
    the extra VAE work is one encode per colorize record, not one per pool
    member. Must run **before** :func:`_phash_edit_build_links`: after that the
    pool dir also holds the ``_no_tags`` pair views, which are not sources.
    """
    pairs = json.loads(manifest.read_text(encoding="utf-8"))["pairs"]
    keys = sorted({p["target"] for p in pairs if p.get("kind") == "colorize"})
    if not keys:
        return 0

    resized_dir, src_dir = ROOT / resized, ROOT / mono_src
    staged = 0
    for key in keys:
        d = resized_dir / Path(key).parent
        stem = Path(key).name
        img = next(
            (q for e in _PHASH_IMG_EXTS if (q := d / f"{stem}{e}").exists()), None
        )
        if img is None:
            print(
                f"  [phash_edit] colorize {key}: no resized image — skipping.",
                file=sys.stderr,
            )
            continue
        link = src_dir / Path(key).parent / img.name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(img.resolve())
        staged += 1
    if not staged:
        return 0
    print(f"[phash_edit] colorize: staged {staged} targets for mangafy → {mono}")

    run(
        [
            PY,
            "easycontrol_adapters/colorization/prep.py",
            "--src",
            mono_src,
            "--staging",
            mono,
            "--skip_encode",
            "--skip_target",
            "--skip_text",
            # No speech-bubble masks exist for the crawl pool, so there is
            # nothing to paste back; screen the whole page.
            "--skip_text_mask",
            "--engine",
            str(pp.get("mangafy_engine", "gpu")),
            "--recursive",
        ]
    )
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
            "--dir",
            mono,
            "--cache_dir",
            mono_cache,
            "--vae",
            pp.get("vae", "models/vae/qwen_image_vae.safetensors"),
            "--batch_size",
            str(pp.get("batch_size", 4)),
            "--chunk_size",
            str(pp.get("chunk_size", 64)),
            "--recursive",
        ]
    )
    return staged


def _phash_edit_build_links(
    manifest: Path, resized: str, cache: str, cond: str, mono_cache: str = ""
) -> None:
    """Materialize each directed pair as symlinks over the encoded pool.

    Per pair: the target's resized image and latent under the ``{pair}_no_tags``
    stem, its delta caption as a real ``.txt``, and the cond's latent filed under
    the *target* stem inside ``cond/`` — at the cond's own bucket, which under
    free-fit need not match the target's (cond≠target shapes are supported;
    ``cond_diff_loss`` self-skips on a mismatch).

    Three record kinds share this layout, differing only in where the cond latent
    comes from and what the caption says:

    ``edit``      the mined partner's latent; caption = the tag delta.
    ``identity``  the target's OWN latent (cond and target are one cached file);
                  caption empty — a no-op instruction.
    ``colorize``  the mangafied latent from ``mono_cache/``; caption drawn per
                  variant from the record's ``variants`` list.

    A record carrying ``variants`` also gets a ``{stem}.variants.txt`` sidecar,
    which the TE step treats as the source of truth instead of generating its own
    shuffles — that is how the colorize arm gets its own dropout regime inside a
    run whose global ``caption_tag_dropout_rate`` is 0.
    """
    from anime_tools.captions.variants import (
        variants_sidecar_path,
        write_variants_sidecar,
    )

    resized_dir, cache_dir, cond_dir = ROOT / resized, ROOT / cache, ROOT / cond
    mono_dir = ROOT / mono_cache if mono_cache else None
    pairs = json.loads(manifest.read_text(encoding="utf-8"))["pairs"]

    def _latents(key: str, root: Path | None = None) -> list[Path]:
        d = (root or cache_dir) / Path(key).parent
        stem = Path(key).name
        return sorted(d.glob(f"{stem}_*_anima.npz"))

    # Image extensions only: resized/ also holds the .txt captions this step
    # writes, and a bare "{stem}.*" glob would happily return one of those.
    _IMG_EXTS = (".png", ".webp", ".jpg", ".jpeg")

    def _image(key: str) -> Path | None:
        d = resized_dir / Path(key).parent
        if not d.is_dir():
            return None
        stem = Path(key).name
        return next((p for e in _IMG_EXTS if (p := d / f"{stem}{e}").exists()), None)

    linked = skipped = 0
    kinds: Counter = Counter()
    for p in pairs:
        kind = p.get("kind", "edit")
        # colorize's cond is the synthetic mangafied view, encoded into its own
        # cache under the same key; every other kind reads the shared pool cache.
        cond_root = mono_dir if kind == "colorize" else None
        tgt_img, tgt_npz, cond_npz = (
            _image(p["target"]),
            _latents(p["target"]),
            _latents(p["cond"], cond_root),
        )
        if tgt_img is None or not tgt_npz or not cond_npz:
            missing = (
                "target image"
                if tgt_img is None
                else "target latent"
                if not tgt_npz
                else f"cond latent ({kind})"
            )
            print(
                f"  [phash_edit] {p['pair_id']}: no {missing} — skipping pair.",
                file=sys.stderr,
            )
            skipped += 1
            continue
        stem = f"{p['pair_id']}_no_tags"
        img_link = resized_dir / p["artist"] / f"{stem}{tgt_img.suffix}"
        img_link.parent.mkdir(parents=True, exist_ok=True)
        img_link.symlink_to(tgt_img.resolve())
        img_link.with_suffix(".txt").write_text(p["delta_caption"], encoding="utf-8")
        variants = p.get("variants")
        if variants:
            write_variants_sidecar(
                variants_sidecar_path(img_link),
                [(f"v{i}", t) for i, t in enumerate(variants)],
            )

        bucket = re.search(r"_(\d+x\d+)_anima\.npz$", tgt_npz[0].name)[1]
        link = cache_dir / p["artist"] / f"{stem}_{bucket}_anima.npz"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(tgt_npz[0].resolve())

        cond_bucket = re.search(r"_(\d+x\d+)_anima\.npz$", cond_npz[0].name)[1]
        clink = cond_dir / p["artist"] / f"{stem}_{cond_bucket}_anima.npz"
        clink.parent.mkdir(parents=True, exist_ok=True)
        clink.symlink_to(cond_npz[0].resolve())
        linked += 1
        kinds[kind] += 1
    print(
        f"[phash_edit] linked {linked} pair views ({resized}/…_no_tags + {cond}) "
        f"{dict(kinds)}" + (f" ({skipped} skipped)" if skipped else "")
    )


def _phash_edit_stage(adapter: str, cfg: dict, base: str, extra) -> None:
    """Mine aligned instruction-edit pairs out of the raw crawl pool by phash.

    Pairs come from ``$CAPTION_CORPUS_DIR/retrieved`` via gelcrawl's cached
    256-bit ``imagehash.phash`` (Hamming threshold, CPU-only, seconds); each
    pair's caption is the tag delta between its members, i.e. an edit
    instruction. Emits the ``_tags``/``_no_tags`` tree so the near_twins
    preprocess pass applies verbatim. The tool self-reads its ``[staging]``
    table and rewrites the blueprint tail.
    """
    cfg_path = str(_easy_cfg_path(adapter))
    run(
        [
            PY,
            "-m",
            "easycontrol_adapters.tools.phash_edit_pairs",
            "--config",
            cfg_path,
            "--config-out",
            cfg_path,
            *extra,
        ]
    )


def _twin_edit_stage(adapter: str, cfg: dict, base: str, extra) -> None:
    """Stage the aligned-pair instruction-edit tree from the Phase-0 census
    manifest (direction-doubled delta-caption pairs + empty-instruction
    identity no-ops). CPU-only symlinks; the tool self-reads its ``[staging]``
    table and rewrites the blueprint tail (near_twins pattern)."""
    cfg_path = str(_easy_cfg_path(adapter))
    run(
        [
            PY,
            "-m",
            "easycontrol_adapters.tools.twin_edit_pairs",
            "--config",
            cfg_path,
            "--config-out",
            cfg_path,
            *extra,
        ]
    )


# Per-adapter materialization bodies (training is generic via _easy_train_extra);
# only `stage` (data gen) + `preprocess` (VAE/TE caching) differ per adapter.
# Both receive ``(adapter, cfg, base, extra)``.
_EASY_ADAPTERS = {
    "near_twins": {"stage": _near_twins_stage, "preprocess": _near_twins_preprocess},
    # sanitize reuses the near-twin miner wholesale (same pipeline, different
    # discriminator tags via its own descriptor file).
    "sanitize": {"stage": _near_twins_stage, "preprocess": _near_twins_preprocess},
    "colorize": {"stage": _colorize_stage, "preprocess": _colorize_preprocess},
    "inpaint": {"stage": _inpaint_stage, "preprocess": _inpaint_preprocess},
    # Paint-to-character: solo-1girl + 1girl1boy targets, SAM3 masks gated +
    # augmented (exact→slack, face) into a real-background paint cond tree.
    "region": {"stage": _region_stage, "preprocess": _region_preprocess},
    "subject": {"stage": _subject_stage, "preprocess": _subject_preprocess},
    "subject_edit": {
        "stage": _subject_edit_stage,
        "preprocess": _subject_edit_preprocess,
    },
    # Aligned-pair instruction editor: bespoke staging over the census manifest,
    # then the near_twins preprocess pass verbatim (same _tags/_no_tags shape).
    # NB the twin_edit tool + descriptor were removed with the directedit_ec
    # archive (2026-08-19); this entry is dead until they are restored.
    "twin_edit": {"stage": _twin_edit_stage, "preprocess": _near_twins_preprocess},
    # phash-mined aligned instruction editor: the twin_edit objective on pairs
    # found by perceptual hash over the RAW crawl pool instead of by tag delta
    # over the curated one. Bespoke preprocess: the miner stages a deduplicated
    # pool and the pair views are symlinks, so each image is encoded ONCE.
    "phash_edit": {"stage": _phash_edit_stage, "preprocess": _phash_edit_preprocess},
}


def cmd_easycontrol_preprocess(extra):
    """Full EasyControl preprocess: VAE latents + text-encoder outputs.

    Source: ``easycontrol-dataset/``  Caches: ``post_image_dataset/easycontrol/``.

    ``EASYADAPTER=<adapter>`` instead runs the adapter's descriptor-driven
    preprocess (every knob from the ``[preprocess]`` table of
    ``configs/easycontrol/<adapter>.toml``):
      • ``colorize`` caches the synthetic-manga *condition* latents, the
        white-balanced *target* latents, and color-only text over the
        already-staged tree; target TE/PE are reused from the LoRA cache.
      • ``near_twins`` resizes + VAE/TE-caches the mined pair tree and
        symlinks the ``cond/`` reference latents.
    """
    adapter = _easyadapter()
    if adapter in _EASY_ADAPTERS:
        cfg, _name, base = _easy_load(adapter)
        _EASY_ADAPTERS[adapter]["preprocess"](adapter, cfg, base, extra)
        return

    src = "easycontrol-dataset"
    dst = "post_image_dataset/easycontrol"
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
        ]
    )
    run(
        [
            PY,
            "scripts/preprocess/cache_text_embeddings.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            "--caption_shuffle_variants",
            "4",
            "--caption_tag_dropout_rate",
            "0.1",
        ]
    )


def cmd_easycontrol_staging(extra):
    """Generate an EasyControl adapter's *staging* dataset (no VAE/TE caching).

    The adapter-specific data-generation step that materializes the training/
    condition tree, kept separate from the later ``easycontrol-preprocess``
    VAE/TE caching pass. Knobs come from the ``[staging]`` table of
    ``configs/easycontrol/<adapter>.toml``; extra CLI args override them.

    ``EASYADAPTER=near_twins`` mines the in-artist near-twin pair tree into
    ``post_image_dataset/easycontrol/near_twins/staging/`` and (re)writes the
    descriptor's blueprint tail, e.g.::

        make easycontrol-staging EASYADAPTER=near_twins \\
            ARGS="--region --artists ama_mitsuki"

    ``EASYADAPTER=colorize`` runs only the mangafy stage (synthesize the
    synthetic B&W manga condition tree; idempotent, skips already-staged
    PNGs), e.g.::

        make easycontrol-staging EASYADAPTER=colorize ARGS="--engine cv2 --limit 8"
    """
    adapter = _easyadapter()
    spec = _EASY_ADAPTERS.get(adapter)
    if spec is None or "stage" not in spec:
        raise SystemExit(
            "easycontrol-staging needs a staging-capable EASYADAPTER. "
            f"Known: {sorted(_EASY_ADAPTERS)}.\n"
            "(The default EasyControl reads easycontrol-dataset/ directly.)"
        )
    cfg, _name, base = _easy_load(adapter)
    spec["stage"](adapter, cfg, base, extra)
