"""PE's thinking as context for MOSS-Image: the rewrite enters through the think block.

The TE is causal, so text after the generation prompt leaves the caption rows' states
unchanged, and the image reads ``</think>`` as a caption-dependent key
(``docs/moss/findings/assistant_tail.md``). One catalog INT8 TE serves both roles:

1. generate — PE mode (``pe_mode_int8_r128d256``) and TE mode (the bare 9B, the control)
   each rewrite every prompt under PE's system prompt, thinking on (card sampling);
2. encode — TE mode, i.e. the TE the DiT was trained on, every arm with the ``sure``
   tail on cond and uncond:

   ``sure``        shipped: caption + ``\\n</think>\\n\\nSure,``
   ``inst_think``  caption + the 9B's own think text + ``\\n</think>\\n\\nSure,``
   ``pe_think``    caption + PE's think text + ``\\n</think>\\n\\nSure,``
   ``pe_answer``   PE's answer paragraph as the caption, ``sure`` tail

MOSS-Image Base, no LoRA, ER-SDE 50, CFG 3, Spectrum, fp32 stream, compiled blocks.
Prompts: the lora_ab PROMPTS (layout JSON, caststation artist tag) and two short NL
briefs. Generations cache to ``gens.json``, latents to ``latents/``; a rerun skips both.

    make daemon-run ARGS="--stall-timeout 900 project/pe_lora/src/think_inject.py"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from bench.moss_ab._render import decoded, grid, render, save_latent
from bench.moss_ab.lora_ab import PROMPTS
from calib import generate
from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss.cache import caption_converter
from library.moss.loader import (
    MAX_SEQUENCE_LENGTH,
    compile_blocks,
    empty_cache,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
    pad_embeds,
    prompt_text,
    stream_fp32_,
)
from library.moss.model import snapped_sigmas
from pe_mode import PEMode

ARMS = ("sure", "inst_think", "pe_think", "pe_answer")
SURE = "\n</think>\n\nSure,"
BRIEFS = [
    "a cat sleeping on a windowsill on a rainy afternoon",
    "a girl in a red raincoat waiting at a bus stop at night",
]


def split_gen(tok, row):
    """``(think, answer paragraph)`` of one generation; the answer is PE's JSON
    ``prompt`` field when it parses, else the raw answer text."""
    text = tok.decode(row["gen_ids"], skip_special_tokens=False)
    for t in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(t, "")
    think, sep, answer = text.partition("</think>")
    if not sep:
        return think.strip(), ""
    answer = answer.strip()
    try:
        obj = json.loads(answer)
        if isinstance(obj, dict):
            strs = [v for v in obj.values() if isinstance(v, str)]
            answer = obj.get("prompt") or max(strs, key=len, default=answer)
    except Exception:
        pass
    return think.strip(), answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pe_tensors",
        type=Path,
        default=Path(
            "/media/sorryhyun/new/qwen35_pe/lora/pe_mode_int8_r128d256.safetensors"
        ),
    )
    ap.add_argument(
        "--system",
        type=Path,
        default=Path("/media/sorryhyun/new/qwen35_pe/pe/system_prompt.txt"),
    )
    ap.add_argument("--artist", default="@channel (caststation)")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--gen_seed", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="output/tests/pe_think_inject")
    args = ap.parse_args()
    w, h = map(int, args.size.split("x"))
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")
    out = resolve_under_home(args.out)
    (out / "latents").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    raw = [p.format(artist=args.artist) for p in PROMPTS] + BRIEFS
    conv = caption_converter("layout_json")
    captions = [conv(p) for p in raw]
    tok = load_tokenizer(default_moss_dir())

    gens_path = out / "gens.json"
    embeds_path = out / "embeds.safetensors"
    if not embeds_path.exists():
        _, te = load_text_encoder(
            None, device, MAX_SEQUENCE_LENGTH, precision="int8", tok=tok
        )
        if gens_path.exists():
            gens = json.loads(gens_path.read_text())
            pm = None
        else:
            pm = PEMode(te, load_file(str(args.pe_tensors)))
            system = args.system.read_text().strip()
            gens = {}
            for mode, name in (("pe", "pe"), ("te", "inst")):
                pm.set_mode(mode)
                rows = generate(
                    pm.causal,
                    tok,
                    system,
                    raw,
                    len(raw),
                    args.max_new,
                    args.gen_seed,
                    print,
                )
                gens[name] = []
                for r in rows:
                    think, answer = split_gen(tok, r)
                    gens[name].append(
                        {
                            "prompt": r["prompt"],
                            "n_gen": r["n_gen"],
                            "finished": r["finished"],
                            "think": think,
                            "answer": answer,
                        }
                    )
            pm.set_mode("te")
            gens_path.write_text(json.dumps(gens, indent=1, ensure_ascii=False))
            print(f"[gen] wrote {gens_path}", flush=True)
        if pm is not None:
            pm.detach()
            del pm
            empty_cache()

        def seq(text):
            return tok(text, return_tensors="pt").input_ids

        seqs = {"uncond": seq(prompt_text(tok, "") + SURE)}
        for i, c in enumerate(captions):
            base = prompt_text(tok, c)
            seqs[f"sure_{i}"] = seq(base + SURE)
            for name in ("inst", "pe"):
                think = gens[name][i]["think"]
                seqs[f"{name}_think_{i}"] = seq(base + think + SURE)
            answer = gens["pe"][i]["answer"] or c
            seqs[f"pe_answer_{i}"] = seq(prompt_text(tok, answer) + SURE)
        embeds = {}
        with torch.no_grad():
            for k, ids in seqs.items():
                ids = ids.to(device)
                o = te(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                )
                embeds[k] = o.last_hidden_state[0].cpu().contiguous()
                print(f"[te] {k}: {ids.shape[1]} rows", flush=True)
        from safetensors.torch import save_file

        save_file(embeds, str(embeds_path))
        del te
        empty_cache()
    embeds = load_file(str(embeds_path))

    sigmas = snapped_sigmas(args.steps)
    jobs = [(a, pi, s) for a in arms for pi in range(len(captions)) for s in seeds]

    def lat(job):
        a, pi, s = job
        return out / "latents" / f"{a}_p{pi}_s{s}.safetensors"

    todo = [j for j in jobs if not lat(j).exists()]
    if todo:
        tr = load_transformer("", "base", device=device)
        stream_fp32_(tr)
        compile_blocks(tr.layers)
        uncond, umask = pad_embeds([embeds["uncond"]], device)
        for job in todo:
            a, pi, s = job
            cond = embeds[f"{a}_{pi}"]
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
    for pi in range(len(captions)):
        rows = [
            (f"p{pi} s{s}", [(a, decoded(lat((a, pi, s)), vae, device)) for a in arms])
            for s in seeds
        ]
        grid(rows, cell_h=640).save(out / f"grid_p{pi}.png")
    (out / "index.json").write_text(
        json.dumps(
            {
                "args": {k: str(v) for k, v in vars(args).items()},
                "captions": captions,
                "rows": {k: int(v.shape[0]) for k, v in embeds.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
