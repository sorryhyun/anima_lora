"""One Anima tag caption → three MOSS prompts, for a tag vs natural-language render A/B.

PE mode (``pe_mode_int8_r128d256`` on the catalog INT8 TE) rewrites the caption under
PE's system prompt, thinking on, ``--n`` samples. Written to ``--out``:

  ``prompts_tags.txt``       the tag caption (generate with ``prompt_format layout_json``)
  ``prompts_nl.txt``         PE's answer paragraph, one line per sample (``text``)
  ``prompts_layout_nl.txt``  the caption's layout JSON with ``summary.caption`` replaced
                             by PE's paragraph: meta / characters / series / artists stay
                             structured fields, the description is prose (``text``)
  ``gens.json``              think + answer + token counts per sample

    make daemon-run ARGS="--stall-timeout 900 project/pe_lora/src/nl_rewrite.py \\
        --caption '…' --out output/tests/pe_tag_vs_nl/<name>"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from calib import generate
from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss.cache import caption_converter
from library.moss.loader import MAX_SEQUENCE_LENGTH, load_text_encoder, load_tokenizer
from pe_mode import PEMode
from think_inject import split_gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption", required=True)
    ap.add_argument(
        "--brief_suffix",
        default="",
        help="appended to the caption in PE's user turn only (e.g. a frame hint)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--gen_seed", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=2048)
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
    args = ap.parse_args()
    out = resolve_under_home(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(default_moss_dir())
    _, te = load_text_encoder(
        None, torch.device("cuda"), MAX_SEQUENCE_LENGTH, precision="int8", tok=tok
    )
    pm = PEMode(te, load_file(str(args.pe_tensors)))
    pm.set_mode("pe")
    brief = (args.caption + " " + args.brief_suffix).strip()
    rows = generate(
        pm.causal,
        tok,
        args.system.read_text().strip(),
        [brief] * args.n,
        args.n,
        args.max_new,
        args.gen_seed,
        print,
    )

    layout = json.loads(caption_converter("layout_json")(args.caption))
    gens, nl, layout_nl = [], [], []
    for r in rows:
        think, answer = split_gen(tok, r)
        answer = " ".join(answer.split())
        gens.append(
            {
                "n_gen": r["n_gen"],
                "finished": r["finished"],
                "think": think,
                "answer": answer,
            }
        )
        if not answer:
            continue
        nl.append(answer)
        obj = json.loads(json.dumps(layout))
        obj["summary"]["caption"] = answer
        layout_nl.append(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))

    (out / "gens.json").write_text(
        json.dumps({"brief": brief, "gens": gens}, indent=1, ensure_ascii=False)
    )
    (out / "prompts_tags.txt").write_text(args.caption + "\n")
    (out / "prompts_nl.txt").write_text("\n".join(nl) + "\n")
    (out / "prompts_layout_nl.txt").write_text("\n".join(layout_nl) + "\n")
    for i, g in enumerate(gens):
        print(f"[{i}] n_gen={g['n_gen']} finished={g['finished']}\n{g['answer']}\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
