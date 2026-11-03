"""MOSS-Image Base DiT conditioned on Qwen3.5-9B Base instead of the shipped instruct TE.

The instruct TE is the shipped INT8 file (the bf16 shards are not on disk; INT8 vs bf16
is 2.1 % per token). Both TEs read the MOSS tokenizer's ids. A prompt is 3 prefix rows
(``<|im_start|>user\\n``), the caption rows and 7 suffix rows
(``<|im_end|>\\n<|im_start|>assistant\\n<think>\\n``). Arms, cond / uncond:

- ``instruct``: shipped.
- ``base``: Base on the shipped template, uncond Base.
- ``splice``: instruct's prefix and suffix rows around Base's caption rows (same ids,
  row for row), uncond instruct (an empty prompt has only template rows).
- ``base_nosuffix``: Base on prefix + caption, no suffix rows; uncond Base on the prefix.
- ``base_describe``: Base on the template with ``Describe this image: `` before the
  caption; uncond Base on the empty prompt.
- ``instruct_ok``: instruct on the shipped prompt + ``\\n</think>\\n\\nOK`` (a closed,
  empty think block and the answer ``OK``, Qwen's non-thinking form); uncond the same
  on the empty prompt.
- ``instruct_think``: instruct on the shipped prompt + ``Okay,`` (the think block left
  open, the start of a thought); uncond the same on the empty prompt.
- ``instruct_sure``: instruct on the shipped prompt + ``\\n</think>\\n\\nSure,`` (a closed,
  empty think block and the start of an answer); uncond the same on the empty prompt.
- ``instruct_ok_cond``: ``instruct_ok``'s cond, shipped uncond.
- ``instruct_pad4``: the shipped prompt + four ``\\n`` tokens (``instruct_ok``'s length,
  no content); uncond the same on the empty prompt.

The lora_ab PROMPTS (layout JSON, caststation artist tag), seeds 1 / 2, ER-SDE 50,
CFG 3, Spectrum, fp32 stream, no LoRA. One grid per prompt (rows: seed; columns: arm).

    make daemon-run ARGS="--stall-timeout 900 project/moss_te_sae/src/base_te_render.py"
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModel

from bench.moss_ab._render import decoded, grid, render, save_latent
from bench.moss_ab.lora_ab import PROMPTS
from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss.cache import caption_converter
from library.moss.loader import (
    compile_blocks,
    empty_cache,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
    offload_by_layer,
    pad_embeds,
    prompt_ids,
    prompt_text,
    stream_fp32_,
    te_reserve_gb,
)
from library.moss.model import snapped_sigmas

BASE_DIR = "/media/sorryhyun/data/qwen35_base/Qwen3.5-9B-Base"
ARMS = (
    "instruct",
    "base",
    "splice",
    "base_nosuffix",
    "base_describe",
    "instruct_ok",
    "instruct_think",
    "instruct_sure",
    "instruct_ok_cond",
    "instruct_pad4",
)
N_PREFIX, N_SUFFIX = 3, 7
DESCRIBE = "Describe this image: "
OK_ANSWER = "\n</think>\n\nOK"
THINK_START = "Okay,"
SURE_ANSWER = "\n</think>\n\nSure,"


@torch.no_grad()
def encode_ids(te_dir, seqs: dict, device):
    """``seqs`` ``{key: ids (1, S)}`` → ``{key: (S, 4096) cpu}``. ``te_dir`` ``int8``:
    the shipped instruct TE."""
    longest = max(ids.shape[1] for ids in seqs.values())
    if te_dir == "int8":
        _, te = load_text_encoder(default_moss_dir(), device, longest, precision="int8")
    else:
        te = AutoModel.from_pretrained(te_dir, dtype=torch.bfloat16).eval()
        offload_by_layer(te, device, te_reserve_gb(longest))
    out = {}
    for k, ids in seqs.items():
        ids = ids.to(device)
        o = te(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        out[k] = o.last_hidden_state[0].cpu()
    del te
    empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", default="@channel (caststation)")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument(
        "--prompts_file",
        default="",
        help="one prompt per line ({artist} substituted) in place of PROMPTS",
    )
    ap.add_argument("--base_dir", default=BASE_DIR)
    ap.add_argument("--out", default="output/tests/base_te_render")
    args = ap.parse_args()
    w, h = map(int, args.size.split("x"))
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")
    out = resolve_under_home(args.out)
    (out / "latents").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    conv = caption_converter("layout_json")
    templates = PROMPTS
    if args.prompts_file:
        lines = resolve_under_home(args.prompts_file).read_text().splitlines()
        templates = [line for line in lines if line.strip()]
    prompts = [conv(p.format(artist=args.artist)) for p in templates]
    tok = load_tokenizer(default_moss_dir())

    def ids(text):
        return prompt_ids(tok, text).input_ids

    empty = ids("")
    assert empty.shape[1] == N_PREFIX + N_SUFFIX
    shipped = {("ship", i): ids(p) for i, p in enumerate(["", *prompts])}
    for v in shipped.values():
        assert torch.equal(v[0, :N_PREFIX], empty[0, :N_PREFIX])
        assert torch.equal(v[0, -N_SUFFIX:], empty[0, N_PREFIX:])
    base_seqs = dict(shipped)
    base_seqs.update({("nosuf", k[1]): v[:, :-N_SUFFIX] for k, v in shipped.items()})
    base_seqs.update(
        {("desc", i + 1): ids(DESCRIBE + p) for i, p in enumerate(prompts)}
    )
    inst_seqs = dict(shipped)
    inst_seqs.update(
        {
            ("ok", i): tok(
                prompt_text(tok, p) + OK_ANSWER, return_tensors="pt"
            ).input_ids
            for i, p in enumerate(["", *prompts])
        }
    )
    inst_seqs.update(
        {
            ("think", i): tok(
                prompt_text(tok, p) + THINK_START, return_tensors="pt"
            ).input_ids
            for i, p in enumerate(["", *prompts])
        }
    )
    inst_seqs.update(
        {
            ("sure", i): tok(
                prompt_text(tok, p) + SURE_ANSWER, return_tensors="pt"
            ).input_ids
            for i, p in enumerate(["", *prompts])
        }
    )
    nl = tok.convert_tokens_to_ids("Ċ")
    inst_seqs.update(
        {
            ("pad4", k[1]): torch.cat((v, torch.full((1, 4), nl, dtype=v.dtype)), 1)
            for k, v in shipped.items()
        }
    )
    inst = encode_ids("int8", inst_seqs, device)
    base = encode_ids(args.base_dir, base_seqs, device)

    def cond_uncond(arm, pi):
        k = pi + 1
        if arm == "instruct":
            return inst["ship", k], inst["ship", 0]
        if arm == "base":
            return base["ship", k], base["ship", 0]
        if arm == "splice":
            a, b = inst["ship", k], base["ship", k]
            c = torch.cat((a[:N_PREFIX], b[N_PREFIX:-N_SUFFIX], a[-N_SUFFIX:]))
            return c, inst["ship", 0]
        if arm == "base_nosuffix":
            return base["nosuf", k], base["nosuf", 0]
        if arm == "instruct_ok_cond":
            return inst["ok", k], inst["ship", 0]
        if arm == "instruct_pad4":
            return inst["pad4", k], inst["pad4", 0]
        if arm == "instruct_sure":
            return inst["sure", k], inst["sure", 0]
        if arm == "instruct_think":
            return inst["think", k], inst["think", 0]
        if arm == "instruct_ok":
            return inst["ok", k], inst["ok", 0]
        if arm == "base_describe":
            return base["desc", k], base["ship", 0]
        raise ValueError(arm)

    sigmas = snapped_sigmas(args.steps)
    jobs = [(a, pi, s) for a in arms for pi in range(len(prompts)) for s in seeds]

    def lat(job):
        a, pi, s = job
        return out / "latents" / f"{a}_p{pi}_s{s}.safetensors"

    todo = [j for j in jobs if not lat(j).exists()]
    if todo:
        tr = load_transformer("", "base", device=device)
        stream_fp32_(tr)
        compile_blocks(tr.layers)
        for job in todo:
            a, pi, s = job
            cond, unc = cond_uncond(a, pi)
            uncond, umask = pad_embeds([unc], device)
            z = render(
                tr, cond, uncond, umask, sigmas, seed=s, height=h, width=w, cfg=args.cfg
            )
            save_latent(z, lat(job))
            print(
                f"{a} p{pi} s{s} ({cond.shape[0]} rows) -> {lat(job).name}", flush=True
            )
        del tr
        empty_cache()

    vae = load_vae("", device=device)
    for pi in range(len(prompts)):
        rows = [
            (
                f"p{pi} s{s}",
                [(a, decoded(lat((a, pi, s)), vae, device)) for a in arms],
            )
            for s in seeds
        ]
        grid(rows, cell_h=640).save(out / f"grid_arms_p{pi}.png")
    (out / "index.json").write_text(
        json.dumps({"args": vars(args), "prompts": prompts}, indent=2)
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
