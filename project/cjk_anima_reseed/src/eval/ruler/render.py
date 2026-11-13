"""The render pass (GPU): each string at its source image's aspect
(``build.AREA`` pixels), seed 0, 28 steps, cfg 4, routed, one arm at a time
through one ``ExtDelta`` over the union of the arms' ext ids; a render on
disk is not drawn again, so the floor (``FLOOR``: the EN refs, retrain_kana,
seed_retrain_0930) is rendered once and every later arm reads against it."""

from __future__ import annotations

import json
from pathlib import Path

from eval.ruler import (
    CFG,
    EN,
    MODELS,
    RULER,
    SEED_RENDER,
    STEPS,
    VIEW,
    items,
    mode_dir,
    model_of,
    render_file,
)
from eval.ruler.arms import arm_dirs, tables


class Renderer:
    def __init__(self, model: str = ""):
        import torch

        from reseed import REPO

        from common.hooks import ExtDelta
        from common.models import load_generator, load_vae

        dyn = torch._dynamo.config  # 13 ruler shapes + the check's, one graph each
        for k in ("cache_size_limit", "recompile_limit"):
            if hasattr(dyn, k):
                setattr(dyn, k, 64)
        self.torch = torch
        steps, cfg, over = STEPS, CFG, {}
        if model:
            mv = MODELS[model]
            steps, cfg = mv["steps"], mv["cfg"]
            over["dit"] = str(REPO / mv["dit"])
            if mv.get("lora"):
                over["lora_weight"] = [str(REPO / mv["lora"])]
                over["lora_multiplier"] = 1.0
        self.args, self.gen, self.device, self.shared = load_generator(
            512, steps, cfg, RULER / "_tmp", **over
        )
        self.anima = self.shared["model"]
        self.anima.eval()
        self.vae = load_vae(self.device)
        self.ids, t = tables(["retrain_kana"])
        dim = t["retrain_kana"].shape[1]
        self.delta = ExtDelta(self.anima, self.ids, dim, self.device, row_scale=1.0)

    def set_arm(self, table) -> None:
        self.shared["conds_cache"].clear()
        if table is None:
            self.delta.scale = 0.0
            return
        self.delta.scale = 1.0
        self.delta.raw.data.copy_(table.to(self.delta.raw.device))

    def render(self, fn: Path, caption: str, seed: int, wh) -> None:
        import copy

        from common.models import decode_image
        from library.inference.generation import generate

        if fn.exists():
            return
        a2 = copy.deepcopy(self.args)
        a2.prompt, a2.seed = caption, seed
        a2.image_size = (wh[1], wh[0])
        with self.torch.no_grad():
            lat = generate(a2, self.gen, self.shared)
        fn.parent.mkdir(parents=True, exist_ok=True)
        decode_image(self.vae, lat, self.device).save(fn)


def check(r: Renderer, table) -> float:
    """retrain_kana's first cached plain render, re-rendered on this path."""
    import numpy as np
    from PIL import Image

    m = json.loads(
        (
            arm_dirs()["retrain_kana"] / "native_r4_plain" / "native_reads.json"
        ).read_text("utf-8")
    )[0]
    ref = Image.open(m["file"])
    fn = RULER / "_tmp" / "check_rk.png"
    fn.unlink(missing_ok=True)
    r.set_arm(table)
    r.render(fn, m["caption"], m["seed"], ref.size)
    d = float(
        np.abs(
            np.asarray(Image.open(fn), np.float32)
            - np.asarray(ref.convert("RGB"), np.float32)
        ).mean()
    )
    print(f"check retrain_kana {Path(m['file']).name}: mean |Δpx| {d:.3f}", flush=True)
    assert d < 12.0, "the ruler path does not reproduce retrain_kana's render"
    return d


def render(names: list) -> None:
    import time

    its = sorted(items(), key=lambda m: (m["shape"], m["i"]))  # one compile per shape
    is_en = {a: a.split("#")[0] == EN for a in names}
    _, tabs = tables([a for a in names if not is_en[a]] + ["retrain_kana"])
    info: dict = {}
    t0 = time.time()
    r = None
    # one generator per model, in the order the arms name them
    for model in dict.fromkeys(map(model_of, names)):
        arms = [a for a in names if model_of(a) == model]
        if all(render_file(a, m["i"]).exists() for a in arms for m in its):
            continue
        if r is not None:
            import gc

            import torch

            r = None
            gc.collect()
            torch.cuda.empty_cache()
        r = Renderer(model)
        if not model:
            info["check_rk"] = check(r, tabs["retrain_kana"])
        for a in arms:
            render_arm(r, a, its, None if is_en[a] else tabs[a], is_en[a], info, t0)
    mode_dir().mkdir(parents=True, exist_ok=True)
    (mode_dir() / "render_log.json").write_text(json.dumps(info, indent=1))
    print(f"render: {info} in {(time.time() - t0) / 60:.1f} min", flush=True)


def render_arm(r, a, its, table, en, info, t0) -> None:
    import time

    todo = [m for m in its if not render_file(a, m["i"]).exists()]
    r.set_arm(table)
    for n, m in enumerate(todo):
        pr = m["prompts"][VIEW.mode]
        cap = pr["en_caption"] if en else pr["caption"]
        r.render(render_file(a, m["i"]), cap, SEED_RENDER, m["shape"])
        print(
            f"  {a}: {n + 1} / {len(todo)} r{m['i']:02d} {m['shape']} "
            f"({(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )
    info[a] = {"rendered": len(todo)}
