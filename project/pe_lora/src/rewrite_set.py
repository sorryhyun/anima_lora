"""A folder of Anima tag captions → PE-rewritten prose captions, for a tag vs prose LoRA A/B.

PE mode (``pe_mode_int8_r128d256`` on the catalog INT8 TE), thinking on; with ``--vision on``
(default) the image goes in the user turn ahead of the brief, read by the TE's own vision
tower (the PE LoRA touches only the language model). The brief is the
tag caption minus rating and artist tags (PE reads ``@channel (caststation)`` as a channel
name and draws UI), plus the frame orientation from the image beside it. The written caption
is ``--prefix`` (the artist trigger) + PE's ``rewritten_prompt``.

``--system`` picks the rewriter's instructions: the Hub prompt (``system_prompt.txt``,
~450-word descriptions that invent whatever the brief leaves open) or
``project/pe_lora/system_caption.txt`` (only what the tags state, length follows the tags).

Writes ``<out>/<stem>.txt`` and ``<out>/gens.jsonl`` (brief, think, answer, token counts);
stems already written are skipped, so a rerun resumes.

    make daemon-run ARGS="--stall-timeout 900 project/pe_lora/src/rewrite_set.py \\
        --src 'post_image_dataset/resized/channel_(caststation)' --out <dir>"
"""

from __future__ import annotations

import argparse
import json
import os
import time

# the compiled decode keeps a CUDA-graph pool; unfragmented segments leave room for the next prefill
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from anime_tools.captions.position_clauses import compose_caption, parse_caption
import fast_decode
from calib import generate
from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import vision_te
from library.moss.loader import MAX_SEQUENCE_LENGTH, load_text_encoder, load_tokenizer
from pe_mode import PEMode
from think_inject import split_gen

RATINGS = {"general", "safe", "sensitive", "questionable", "explicit", "nsfw", "sfw"}
IMAGE_EXTS = (".png", ".webp", ".jpg", ".jpeg")


def brief_of(caption: str, image: Path | None) -> str:
    pc = parse_caption(caption)
    flat = [t for t in pc.flat_tags if t not in RATINGS and not t.startswith("@")]
    brief = compose_caption(flat, pc.clauses)
    if image is not None:
        w, h = Image.open(image).size
        frame = (
            "Vertical" if h > w * 1.05 else "Horizontal" if w > h * 1.05 else "Square"
        )
        brief = f"{brief.rstrip('.')}. {frame} image."
    return brief


def vision_causal(te, lm_head, device="cuda"):
    """``Qwen3_5ForConditionalGeneration`` over the TE (vision tower + the PE-mode language
    model) and PE's head, for ``generate`` with images in the user turn."""
    from accelerate import init_empty_weights
    from transformers import Qwen3_5ForConditionalGeneration

    class _VL(Qwen3_5ForConditionalGeneration):
        @property
        def device(self):
            return torch.device(device)

    with init_empty_weights():
        vl = _VL(te.config)
    vl.model = te
    vl.lm_head = lm_head
    return vl.eval()


def chat_text(tok, system, prompt, thinking):
    return tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


@torch.no_grad()
def generate_vl(
    model,
    proc,
    tok,
    system,
    prompts,
    images,
    max_new,
    seed,
    thinking,
    pad_len=0,
    pad_bs=0,
    **gen_kw,
):
    """``calib.generate``'s rows for one batch, each user turn = image + brief.
    ``pad_len`` / ``pad_bs`` fix the batch shape (left-padded to ``pad_len`` tokens, the last
    row repeated up to ``pad_bs`` rows) so a compiled decode builds one graph for every batch."""
    torch.manual_seed(seed)
    tok.padding_side = "left"
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    texts = [chat_text(tok, system, p, thinking) for p in prompts]
    images = list(images)
    while len(texts) < pad_bs:
        texts.append(texts[-1])
        images.append(images[-1])
    pad = (
        {"padding": "max_length", "max_length": pad_len}
        if pad_len
        else {"padding": True}
    )
    t0 = time.time()
    enc = proc(text=texts, images=images, return_tensors="pt", **pad).to("cuda")
    out = model.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        eos_token_id=eos,
        pad_token_id=tok.pad_token_id,
        **gen_kw,
    )
    plen = enc["input_ids"].shape[1]
    out = out.cpu()
    del enc
    torch.cuda.empty_cache()
    rows = []
    for i, p in enumerate(prompts):
        gen = out[i, plen:].tolist()
        finished = any(t in eos for t in gen)
        while gen and gen[-1] in eos + [tok.pad_token_id]:
            gen.pop()
        answer = (
            tok.decode(gen, skip_special_tokens=True).partition("</think>")[2].strip()
        )
        try:
            json.loads(answer)
            ok = True
        except Exception:
            ok = False
        rows.append(
            {"gen_ids": gen, "finished": finished, "json_ok": ok, "n_gen": len(gen)}
        )
    print(
        f"[gen] {len(prompts)} rows {time.time() - t0:.0f}s prefill={plen} "
        f"tokens={[r['n_gen'] for r in rows]}",
        flush=True,
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of <stem>.txt tag captions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stems", default="", help="comma list; empty = every caption")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prefix", default="An illustration by @channel (caststation). ")
    ap.add_argument(
        "--brief_suffix",
        default="Keep the description under 120 words.",
        help="job instruction appended to each brief (PE obeys these silently)",
    )
    ap.add_argument("--think", choices=("on", "off"), default="on")
    ap.add_argument(
        "--vision",
        choices=("on", "off"),
        default="on",
        help="on: the image goes in the user turn ahead of the tags (vision tower resident)",
    )
    ap.add_argument("--max_pixels", type=int, default=896 * 896)
    ap.add_argument(
        "--fast",
        choices=("on", "off"),
        default="on",
        help="static KV cache + compiled decode (fast_decode.py, 21 vs 68 ms/step at bs 4); "
        "vision path only",
    )
    ap.add_argument(
        "--bs",
        type=int,
        default=10,
        help="decode is launch-bound: a batch costs ~one row; 32 OOMs in prefill (text only)",
    )
    ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument("--gen_seed", type=int, default=0)
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
        default=None,
        help="default: system_caption_vl.txt with --vision on, else system_caption.txt",
    )
    args = ap.parse_args()
    src = resolve_under_home(args.src)
    out = resolve_under_home(args.out)
    out.mkdir(parents=True, exist_ok=True)

    caps = sorted(
        p
        for p in src.glob("*.txt")
        if p.name.count(".") == 1  # not .layout / .variants
    )
    if args.stems:
        keep = set(args.stems.split(","))
        caps = [p for p in caps if p.stem in keep]
    caps = [p for p in caps if not (out / p.name).exists()]
    if args.limit:
        caps = caps[: args.limit]
    if not caps:
        print("nothing to do", flush=True)
        return

    briefs, imgs = [], []
    for p in caps:
        img = next(
            (p.with_suffix(e) for e in IMAGE_EXTS if p.with_suffix(e).exists()), None
        )
        if args.vision == "on" and img is None:
            raise SystemExit(f"--vision on: no image beside {p}")
        imgs.append(img)
        briefs.append(
            f"{brief_of(p.read_text().strip(), img)} {args.brief_suffix}".strip()
        )

    vision = args.vision == "on"
    gen_kw = fast_decode.enable() if args.fast == "on" and vision else {}
    tok = load_tokenizer(default_moss_dir())
    if vision:
        te = vision_te.load_te(tok, torch.device("cuda"), precision="int8")
    else:
        _, te = load_text_encoder(
            None, torch.device("cuda"), MAX_SEQUENCE_LENGTH, precision="int8", tok=tok
        )
    pm = PEMode(te, load_file(str(args.pe_tensors)))
    pm.set_mode("pe")
    if vision:
        vl = vision_causal(te, pm.causal.lm_head)
        proc = vision_te.processor(tok, args.max_pixels)
    sys_path = args.system or Path(
        "project/pe_lora/system_caption_vl.txt"
        if vision
        else "project/pe_lora/system_caption.txt"
    )
    system = resolve_under_home(sys_path).read_text().strip()
    shape = {}
    if gen_kw:  # one batch shape → one compiled graph (each new shape costs ~40 s)
        tok.padding_side = "left"
        longest = max(
            proc(
                text=[chat_text(tok, system, br, args.think == "on")],
                images=[Image.open(i).convert("RGB")],
                return_tensors="pt",
            )["input_ids"].shape[1]
            for br, i in zip(briefs, imgs)
        )
        shape = {"pad_len": -(-longest // 64) * 64, "pad_bs": args.bs}
        print(f"[fast] batch shape {args.bs} x {shape['pad_len']} tokens", flush=True)

    log = open(out / "gens.jsonl", "a")
    for b in range(0, len(caps), args.bs):
        chunk, cbriefs = caps[b : b + args.bs], briefs[b : b + args.bs]
        if vision:
            pics = [Image.open(i).convert("RGB") for i in imgs[b : b + args.bs]]
            rows = generate_vl(
                vl,
                proc,
                tok,
                system,
                cbriefs,
                pics,
                args.max_new,
                args.gen_seed + b,
                args.think == "on",
                **shape,
                **gen_kw,
            )
        else:
            rows = generate(
                pm.causal,
                tok,
                system,
                cbriefs,
                args.bs,
                args.max_new,
                args.gen_seed + b,
                print,
                thinking=args.think == "on",
            )
        for p, brief, r in zip(chunk, cbriefs, rows):
            think, answer = split_gen(tok, r)
            answer = " ".join(answer.split())
            ok = r["finished"] and r["json_ok"] and bool(answer)
            log.write(
                json.dumps(
                    {
                        "stem": p.stem,
                        "brief": brief,
                        "n_gen": r["n_gen"],
                        "n_think": len(tok(think)["input_ids"]),
                        "finished": r["finished"],
                        "json_ok": r["json_ok"],
                        "words": len(answer.split()),
                        "think": think,
                        "answer": answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log.flush()
            if ok:
                (out / p.name).write_text(args.prefix + answer + "\n")
            print(
                f"{p.stem} ok={ok} words={len(answer.split())}\n  {answer[:400]}",
                flush=True,
            )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
