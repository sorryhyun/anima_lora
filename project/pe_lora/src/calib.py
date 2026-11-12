"""PE-T2I calibration: generate rewrites, then collect per-linear input covariances.

Loads Qwen-Image-2.1-PE-T2I as ``Qwen3_5ForCausalLM`` with every decoder linear in
INT8 ConvRot W8A8 (``library.moss.int8``) so it fits a 16 GB card, generates
thinking-mode rewrites for the calibration prompts (card sampling params), then
teacher-forces each full sequence (system + user + generated) one at a time and
accumulates the uncentered input Gram ``Σ xᵀx`` for every distinct linear input:

  attn_in   q/k/v_proj (full attention) or linear_attn.in_proj_* (GDN)
  attn_out  self_attn.o_proj or linear_attn.out_proj
  mlp_in    mlp.gate_proj / up_proj
  mlp_down  mlp.down_proj
  final     last hidden state after the final norm (lm_head input)

Hooks read the bf16 input before the int8 rotation, so the statistics see the
activations of a (quantized-weight) PE forward. Grams are fp32 on the CPU, one
safetensors per layer under ``<out>/cov/``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from library.moss import int8
from library.moss.fp8 import LINEARS

GROUP_OF = {
    "self_attn.q_proj": "attn_in",
    "linear_attn.in_proj_qkv": "attn_in",
    "self_attn.o_proj": "attn_out",
    "linear_attn.out_proj": "attn_out",
    "mlp.gate_proj": "mlp_in",
    "mlp.down_proj": "mlp_down",
}


def load_prompts(handwritten: Path, captions_root: Path, n_captions: int, seed: int):
    prompts = [
        line.strip() for line in handwritten.read_text().splitlines() if line.strip()
    ]
    files = sorted(captions_root.rglob("*.txt"))
    files = [f for f in files if not f.name.endswith(".variants.txt")]
    rng = random.Random(seed)
    rng.shuffle(files)
    for f in files:
        if n_captions <= 0:
            break
        tags = _caption_tags(f.read_text(errors="ignore").strip())
        if not tags:
            continue
        prompts.append(", ".join(tags[: rng.randint(6, 24)]))
        n_captions -= 1
    return prompts


def _caption_tags(text: str) -> list[str]:
    """Flat tag list of a resized-tree sidecar (tag caption or layout JSON)."""
    from anime_tools.captions.position_clauses import parse_caption

    if not text:
        return []
    if text.startswith("{"):
        try:
            s = json.loads(text)["summary"]
        except Exception:
            return []
        head = [*s.get("characters", []), *s.get("series", [])]
        return head + [t.strip() for t in s.get("caption", "").split(",") if t.strip()]
    return list(parse_caption(text).flat_tags)


def load_model(pe_dir: Path, log):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        pe_dir, dtype=torch.bfloat16, device_map="cpu"
    ).eval()
    n = 0
    for name, module in list(model.named_modules()):
        for cname, child in list(module.named_children()):
            full = f"{name}.{cname}"
            if not isinstance(child, torch.nn.Linear) or ".layers." not in full:
                continue
            if full.split(".layers.", 1)[1].split(".", 1)[1] not in LINEARS:
                continue
            q, scale = int8.quantize_weight(child.weight.data)
            setattr(module, cname, int8.Int8ConvRotLinear(q, scale, int8.GROUP, True))
            n += 1
    log(f"[load] {n} linears -> int8 W8A8")
    return model.to("cuda")


@torch.no_grad()
def generate(model, tok, system, prompts, bs, max_new, seed, log, thinking=True):
    torch.manual_seed(seed)
    tok.padding_side = "left"
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        for p in prompts
    ]
    rows = []
    for b in range(0, len(texts), bs):
        t0 = time.time()
        enc = tok(texts[b : b + bs], return_tensors="pt", padding=True).to("cuda")
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            eos_token_id=eos,
            pad_token_id=tok.pad_token_id,
        )
        plen = enc["input_ids"].shape[1]
        for i, p in enumerate(prompts[b : b + bs]):
            prompt_ids = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
            gen = out[i, plen:].tolist()
            finished = any(t in eos for t in gen)
            while gen and gen[-1] in eos + [tok.pad_token_id]:
                gen.pop()
            text = tok.decode(gen, skip_special_tokens=True)
            answer = text.partition("</think>")[2].strip()
            try:
                json.loads(answer)
                ok = True
            except Exception:
                ok = False
            rows.append(
                {
                    "prompt": p,
                    "prompt_ids": prompt_ids,
                    "gen_ids": gen,
                    "finished": finished,
                    "json_ok": ok,
                    "answer": answer,
                    "n_gen": len(gen),
                }
            )
        done = rows[-len(prompts[b : b + bs]) :]
        log(
            f"[gen] {b + len(done)}/{len(texts)} {time.time() - t0:.0f}s "
            f"tokens={[r['n_gen'] for r in done]} json_ok={sum(r['json_ok'] for r in done)}"
        )
    return rows


@torch.no_grad()
def collect(model, rows, out_dir: Path, log):
    layers = model.model.layers
    acc: dict[str, torch.Tensor] = {}
    hooks = []

    def add(key, x):
        x = x.reshape(-1, x.shape[-1]).float()
        g = (x.T @ x).cpu()
        if key in acc:
            acc[key] += g
        else:
            acc[key] = g

    for i, layer in enumerate(layers):
        for path, group in GROUP_OF.items():
            mod = layer.get_submodule(path) if _has(layer, path) else None
            if mod is None:
                continue
            key = f"{i}.{group}"
            hooks.append(
                mod.register_forward_pre_hook(lambda m, a, key=key: add(key, a[0]))
            )
    n_tok = 0
    t0 = time.time()
    for j, r in enumerate(rows):
        ids = torch.tensor([r["prompt_ids"] + r["gen_ids"]], device="cuda")
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state
        add("final", h)
        n_tok += ids.shape[1]
        if j % 8 == 0:
            log(f"[cov] {j + 1}/{len(rows)} tokens={n_tok} {time.time() - t0:.0f}s")
    for h in hooks:
        h.remove()
    cov_dir = out_dir / "cov"
    cov_dir.mkdir(parents=True, exist_ok=True)
    meta = {"tokens": str(n_tok), "sequences": str(len(rows))}
    for i in range(len(layers)):
        part = {k.split(".", 1)[1]: v for k, v in acc.items() if k.startswith(f"{i}.")}
        save_file(part, str(cov_dir / f"layer{i:02d}.safetensors"), metadata=meta)
    save_file(
        {"final": acc["final"]}, str(cov_dir / "final.safetensors"), metadata=meta
    )
    log(f"[cov] {len(acc)} grams over {n_tok} tokens -> {cov_dir}")


def _has(layer, path):
    try:
        layer.get_submodule(path)
        return True
    except AttributeError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "prompts_handwritten.txt",
    )
    ap.add_argument(
        "--captions_root", type=Path, default=Path("post_image_dataset/resized")
    )
    ap.add_argument("--n_captions", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from library.env import resolve_under_home

    log = lambda s: print(s, flush=True)  # noqa: E731
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(
        args.prompts, resolve_under_home(args.captions_root), args.n_captions, args.seed
    )
    if args.limit:
        prompts = prompts[: args.limit]
    log(f"[prompts] {len(prompts)}")
    tok = AutoTokenizer.from_pretrained(args.pe_dir)
    system = (args.pe_dir / "system_prompt.txt").read_text().strip()
    model = load_model(args.pe_dir, log)
    log(f"[load] cuda alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    gen_path = args.out_dir / "generations.jsonl"
    if gen_path.exists():
        rows = [json.loads(line) for line in gen_path.read_text().splitlines()]
        log(f"[gen] reusing {len(rows)} rows from {gen_path}")
    else:
        rows = generate(
            model, tok, system, prompts, args.bs, args.max_new, args.seed, log
        )
        gen_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        log(
            f"[gen] finished={sum(r['finished'] for r in rows)}/{len(rows)} "
            f"json_ok={sum(r['json_ok'] for r in rows)}"
        )
    collect(model, rows, args.out_dir, log)


if __name__ == "__main__":
    main()
