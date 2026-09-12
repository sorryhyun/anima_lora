"""Residual probe — does an adapter-space metric see the rare-name failure?

plan3 closed on the finding that every distill metric (span / attn cosine to
the EN teacher) saturates while the render still lacks the character: 0.96
attn cosine to a teacher that *does* render Reimu, and no Reimu. The whole-
vector cosine is dominated by the bulk the tags explain; the name's identity
is a low-energy residual on top of it.

This probe isolates that residual. For each character prompt it encodes the
full prompt and its name-stripped twin on both sides and takes the *name
contribution*::

    Δ_T = pool(teacher(EN full))  − pool(teacher(EN tags))
    Δ_S = pool(student(JA full))  − pool(student(JA tags))      (ja_ext arm)

reporting ``cos(Δ_S, Δ_T)``, ``‖Δ_S‖ / ‖Δ_T‖`` and the **margin** — cos to the
own character's teacher residual minus the best cos to any *other* character's
(is the DiT-facing vector pushed toward Reimu specifically, or just "a name").
The old whole-vector cosine is reported beside it as the control.

Several packs are scored in one job (the DiT is reloaded per pack so the ext
rows / LoRA hooks never leak between arms), and ``--labels`` rank-correlates
every candidate column against the eyeball verdicts in ``grid_labels.json`` —
a metric is only trusted if it orders the arms the way the eyes did.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from anima_lora import GenerationRequest, default_checkpoints  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.cjk_adapter import ext_vocab  # noqa: E402
from bench.cjk_adapter.run_bench import (  # noqa: E402
    DELIM,
    DELIM_EXT,
    SplitTokenizeStrategy,
)
from library.anima import strategy as strategy_anima  # noqa: E402
from library.anima import text_strategies  # noqa: E402
from library.inference import (  # noqa: E402
    check_inputs,
    load_dit_model,
    load_text_encoder,
    prepare_text_inputs,
)
from library.runtime.device import clean_memory_on_device  # noqa: E402

VARIANTS = ("full", "tags", "tags_nc")


def pooled(embed) -> torch.Tensor:
    """Masked mean of the adapter output over non-pad T5 positions."""
    x, mask = embed[0][0].float(), embed[3][0].bool()
    return x[mask].mean(0)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def encode_arm(args, device, anima, shared, prompt: str) -> torch.Tensor:
    a = copy.copy(args)
    a.prompt = prompt
    ctx, _ = prepare_text_inputs(a, device, anima, shared)
    return pooled(ctx["embed"]).cpu()


def prompt_variants(entry: dict) -> dict[str, dict[str, str]]:
    """{variant: {arm: composed prompt}} for the en / ja_t5en / ja_ext arms."""
    out = {}
    for v in VARIANTS:
        suf = "" if v == "full" else f"_{v}"
        en, ja = entry[f"en{suf}"], entry[f"ja{suf}"]
        out[v] = {
            "en": en,
            "ja_t5en": f"{ja}{DELIM}{en}",
            "ja_ext": f"{ja}{DELIM_EXT}{ja}",
        }
    return out


def score_pack(opts, args, device, prefix: Path, prompts: dict, lora: bool) -> dict:
    """Encode every (prompt, variant, arm) under one pack; return residual rows."""
    tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
    ext_table, mapping = ext_vocab.load_ext_assets(prefix)
    tokenize_strategy.ext_encoder = ext_vocab.HybridT5Encoder.from_mapping(
        tokenize_strategy.t5_tokenizer, tokenize_strategy.qwen3_tokenizer, mapping
    )
    anima = load_dit_model(args, device, torch.bfloat16)
    emb = anima.llm_adapter.embed
    new_w = torch.cat(
        [emb.weight.data, ext_table.to(emb.weight.dtype).to(emb.weight.device)]
    )
    anima.llm_adapter.embed = torch.nn.Embedding.from_pretrained(new_w)
    lora_info = None
    if lora:
        from scripts.distill_cjk.adapter_lora import AdapterLoRA

        sib = prefix.with_name(prefix.name + ".adapter_lora.safetensors")
        al = AdapterLoRA.load(anima.llm_adapter, sib)
        lora_info = {"rank": al.rank, "targets": list(al.targets)}
    text_encoder = load_text_encoder(args, dtype=torch.bfloat16, device=device)
    shared = {"text_encoder": text_encoder, "conds_cache": {}}

    vec: dict[tuple, torch.Tensor] = {}
    for pid, entry in prompts.items():
        if pid.startswith("_"):
            continue
        for v, arms in prompt_variants(entry).items():
            for arm, prompt in arms.items():
                vec[(pid, v, arm)] = encode_arm(args, device, anima, shared, prompt)
    del text_encoder, shared, anima
    clean_memory_on_device(device)

    chars = prompts["_character"]
    rows = {}
    for pid in chars:
        r = {}
        for v in ("tags", "tags_nc"):
            d_t = vec[(pid, "full", "en")] - vec[(pid, v, "en")]
            d_s = vec[(pid, "full", "ja_ext")] - vec[(pid, v, "ja_ext")]
            others = [
                cos(d_s, vec[(q, "full", "en")] - vec[(q, v, "en")])
                for q in chars
                if chars[q] != chars[pid]
            ]
            self_cos = cos(d_s, d_t)
            r[f"res_cos_{v}"] = self_cos
            r[f"res_norm_ratio_{v}"] = float(d_s.norm() / d_t.norm().clamp_min(1e-6))
            r[f"res_margin_{v}"] = self_cos - max(others)
            # teacher-side sanity: how separable are the characters' residuals
            r[f"teacher_self_minus_other_{v}"] = 1.0 - max(
                cos(d_t, vec[(q, "full", "en")] - vec[(q, v, "en")])
                for q in chars
                if chars[q] != chars[pid]
            )
        # controls: the whole-vector cosines the distill metrics already use
        r["full_cos_ext_vs_en"] = cos(
            vec[(pid, "full", "ja_ext")], vec[(pid, "full", "en")]
        )
        r["tags_cos_ext_vs_en"] = cos(
            vec[(pid, "tags", "ja_ext")], vec[(pid, "tags", "en")]
        )
        rows[pid] = r
    return {"rows": rows, "lora": lora_info, "n_ext_rows": int(ext_table.shape[0])}


def rank_against_labels(results: dict, labels: dict, prompts: dict) -> dict:
    """Spearman ρ + AUC of every metric column vs the eyeball identity label."""
    import numpy as np

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        # tie-aware: average ranks
        for arr, r in ((x, rx), (y, ry)):
            for val in np.unique(arr):
                m = arr == val
                r[m] = r[m].mean()
        return float(np.corrcoef(rx, ry)[0, 1])

    cols = sorted(
        next(iter(results.values()))["rows"][next(iter(prompts["_character"]))]
    )
    xs: dict[str, list] = {c: [] for c in cols}
    ys, keys = [], []
    for arm, res in results.items():
        if arm not in labels:
            continue
        for pid, r in res["rows"].items():
            lab = labels[arm].get(pid)
            if lab is None:
                continue
            ys.append(float(lab["identity"]))
            keys.append(f"{arm}/{pid}")
            for c in cols:
                xs[c].append(r[c])
    y = np.asarray(ys)
    out = {"n_points": len(ys), "points": keys, "by_metric": {}}
    pos, neg = y >= 1.0, y <= 0.0
    for c in cols:
        x = np.asarray(xs[c])
        rho = spearman(x, y) if len(set(y)) > 1 else float("nan")
        auc = float("nan")
        if pos.any() and neg.any():
            auc = float((x[pos][:, None] > x[neg][None, :]).mean())
        out["by_metric"][c] = {"spearman": float(rho), "auc_identity": auc}
    return out


def main() -> None:
    ckpt = default_checkpoints()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dit", default=ckpt.dit)
    p.add_argument("--vae", default=ckpt.vae)
    p.add_argument("--text_encoder", default=ckpt.text_encoder)
    p.add_argument("--label", default=None)
    p.add_argument(
        "--packs",
        nargs="+",
        required=True,
        help="pack prefixes (output/ckpt/cjk_vocab/cjk_vocab_pack_<arm>); a pack with an "
        ".adapter_lora.safetensors sibling is scored WITH the LoRA hooked.",
    )
    p.add_argument(
        "--prompts",
        type=Path,
        default=Path("project/cjk_aware_anima/assets/ja_eval_prompts_residual.json"),
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("project/cjk_aware_anima/assets/grid_labels.json"),
        help="eyeball verdicts to rank every metric column against ('' to skip)",
    )
    opts = p.parse_args()

    run_dir = make_run_dir("cjk_adapter", opts.label)
    print(f"run dir: {run_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = GenerationRequest(
        dit=opts.dit,
        vae=opts.vae,
        text_encoder=opts.text_encoder,
        prompt="placeholder",
        save_path=str(run_dir),
    ).to_args()
    args.device = device
    check_inputs(args)
    tokenize_strategy = SplitTokenizeStrategy(qwen3_path=opts.text_encoder)
    text_strategies.TokenizeStrategy.set_strategy(tokenize_strategy)
    text_strategies.TextEncodingStrategy.set_strategy(
        strategy_anima.AnimaTextEncodingStrategy()
    )
    prompts = json.loads(opts.prompts.read_text(encoding="utf-8"))

    results = {}
    for pack in opts.packs:
        prefix = Path(pack)
        arm = prefix.name.replace("cjk_vocab_pack_", "")
        lora = prefix.with_name(prefix.name + ".adapter_lora.safetensors").exists()
        print(f"== {arm} (lora={lora})")
        results[arm] = score_pack(opts, args, device, prefix, prompts, lora)
        for pid, r in results[arm]["rows"].items():
            print(
                f"  {pid}: res_cos={r['res_cos_tags']:+.3f} ratio={r['res_norm_ratio_tags']:.2f} "
                f"margin={r['res_margin_tags']:+.3f} | nc cos={r['res_cos_tags_nc']:+.3f} "
                f"margin={r['res_margin_tags_nc']:+.3f} | full_cos={r['full_cos_ext_vs_en']:.3f}"
            )

    metrics: dict = {"packs": results}
    if opts.labels and str(opts.labels):
        labels = json.loads(opts.labels.read_text(encoding="utf-8"))
        metrics["ranking"] = rank_against_labels(results, labels, prompts)
        print(
            f"\nranking vs eyeball identity ({metrics['ranking']['n_points']} points):"
        )
        for c, s in sorted(
            metrics["ranking"]["by_metric"].items(),
            key=lambda kv: -abs(kv[1]["spearman"]),
        ):
            print(
                f"  {c:32s} spearman={s['spearman']:+.3f} auc={s['auc_identity']:.3f}"
            )
    write_result(
        run_dir,
        script=__file__,
        args=opts,
        metrics=metrics,
        label=opts.label,
        device=device,
    )


if __name__ == "__main__":
    main()
