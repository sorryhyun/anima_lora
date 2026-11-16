"""PE mode on the shipped MOSS INT8 TE, scored like ``eval.py``.

Teacher targets come from ``eval.py``'s held-out PE generations, scored by the PE
skeleton (INT8 RTN, cached to ``teacher_targets.pt``). The student is the catalog
TE (``load_text_encoder(precision="int8")``, GPTQ + SmoothQuant) with
``pe_mode.convert`` tensors attached as unmerged LoRA branches. Also checks that
TE mode after a PE round trip encodes identically to the untouched TE.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from calib import generate, load_model
from eval import parse_variant, score, teacher_targets, think_split
from pe_mode import PEMode, convert

PROBE = "a girl with silver hair reading a book in a sunlit library"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe_dir", type=Path, required=True)
    ap.add_argument("--base_dir", type=Path, required=True)
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--eval_dir", type=Path, required=True)
    ap.add_argument("--variants", default="r128d256,r256")
    ap.add_argument("--free_gen", default="r128d256")
    ap.add_argument("--n_free", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument(
        "--save", type=Path, default=None, help="write the first variant's tensors"
    )
    args = ap.parse_args()
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from library.moss import loader
    from library.moss.fp8 import fp8_te_path

    log = lambda s: print(s, flush=True)  # noqa: E731
    rows = [
        json.loads(line)
        for line in (args.eval_dir / "teacher_generations.jsonl")
        .read_text()
        .splitlines()
    ]
    tok = AutoTokenizer.from_pretrained(args.pe_dir)
    split_at = [think_split(tok, r["gen_ids"]) for r in rows]
    tpath = args.eval_dir / "teacher_targets.pt"
    if tpath.exists():
        targets = torch.load(tpath)
    else:
        teacher = load_model(args.pe_dir, log)
        targets = teacher_targets(teacher, rows)
        torch.save(targets, tpath)
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
    log(f"[teacher] {len(targets)} target sets")

    tok_moss, te = loader.load_text_encoder(None, precision="int8", log=log)
    h0 = loader.encode_prompt(tok_moss, te, PROBE).float().cpu()
    te_path = fp8_te_path(precision="int8")
    system = (args.pe_dir / "system_prompt.txt").read_text().strip()
    prompts = [r["prompt"] for r in rows]
    results = {"variants": {}, "free_gen": {}}
    for i, v in enumerate(args.variants.split(",")):
        spec = parse_variant(v)
        tensors = convert(
            args.lora,
            args.base_dir,
            args.pe_dir,
            te_path,
            spec["rank"],
            spec["down"],
            log,
        )
        if i == 0 and args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            save_file({k: t.contiguous() for k, t in tensors.items()}, str(args.save))
            log(f"[save] {args.save} ({args.save.stat().st_size / 2**30:.2f} GiB)")
        pm = PEMode(te, tensors)
        del tensors
        log(
            f"[attach] {v}: {len(pm.branches)} branches, {len(pm.swaps)} swaps, "
            f"cuda {torch.cuda.memory_allocated() / 2**30:.1f} GiB"
        )
        pm.set_mode("pe")
        res = score(pm.causal, rows, targets, split_at)
        results["variants"][v] = res
        log(
            f"[score] {v:12s} "
            + " | ".join(
                f"{s}: kl={d['kl']:.4f} nll={d['nll']:.3f} top1={d['top1']:.3f}"
                for s, d in res.items()
            )
        )
        if v in args.free_gen.split(","):
            g = generate(
                pm.causal, tok, system, prompts[: args.n_free], 8, args.max_new, 2, log
            )
            results["free_gen"][v] = {
                "json_ok": sum(r["json_ok"] for r in g),
                "finished": sum(r["finished"] for r in g),
                "n": len(g),
                "mean_tokens": sum(r["n_gen"] for r in g) / len(g),
                "answers": [{"prompt": r["prompt"], "answer": r["answer"]} for r in g],
            }
            log(f"[free_gen] {v}: json_ok={results['free_gen'][v]['json_ok']}/{len(g)}")
        pm.set_mode("te")
        h1 = loader.encode_prompt(tok_moss, te, PROBE).float().cpu()
        results["variants"][v]["te_roundtrip_maxabs"] = float((h1 - h0).abs().max())
        log(
            f"[te] round trip max |Δh| = {results['variants'][v]['te_roundtrip_maxabs']}"
        )
        pm.detach()
        del pm
        gc.collect()
        torch.cuda.empty_cache()
        (args.eval_dir / "results_te.json").write_text(
            json.dumps(results, indent=1, ensure_ascii=False)
        )
    log("[done]")


if __name__ == "__main__":
    main()
