#!/usr/bin/env python3
"""CJK conditioning probe — where does the LLM Adapter's content come from?

The adapter (`library/anima/models.py::LLMAdapter`) cross-attends T5-token
queries against Qwen3 hidden states. The old t5-v1_1-xxl SentencePiece has no
CJK coverage, so a ja/zh prompt's target side collapses to `▁ <unk> </s>` —
3 informative positions out of 512 — regardless of what Qwen3 understood.

This bench answers, per language (ja/ko/zh), with everything else held fixed
(seed, noise, sampler):

  arm            Qwen3 side      T5 side          tests
  ─────────────  ──────────────  ───────────────  ────────────────────────────
  en (baseline)  English         English          reference output
  native         native CJK      native CJK       today's pipeline (unk-wall)
  t5en           native CJK      English transl.  context-dominance: does
                                                  content flow from Qwen3?
  t5rom          native CJK      romanization     is *any* non-unk anchor
                                                  stream enough?
  q_en           English         native CJK       query-dominance (reverse)

Readouts: (a) adapter-output cosine of each arm vs the en baseline plus a
P1-vs-P2 discrimination cosine per arm type (if two different ja prompts give
~identical conditioning, the pathway is dead), (b) rendered images per arm.

If t5en/t5rom ≈ en, CJK support is a tokenize-strategy shim (no weights
touched). If only en works, the T5 vocab needs a real CJK transplant +
adapter finetune.

Usage (GPU work goes through the daemon)::

    make daemon-run ARGS="bench/cjk_adapter/run_bench.py --label probe1"
    # embedding diagnostics only (no images):
    make daemon-run ARGS="bench/cjk_adapter/run_bench.py --skip_images"
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from anima_lora import GenerationRequest, default_checkpoints  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.cjk_adapter import ext_vocab  # noqa: E402
from library.anima import strategy as strategy_anima  # noqa: E402
from library.anima import text_strategies  # noqa: E402
from library.inference import (  # noqa: E402
    check_inputs,
    generate_body,
    load_dit_model,
    load_text_encoder,
    prepare_text_inputs,
    save_output,
)
from library.models import qwen_vae as qwen_image_autoencoder_kl  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

DELIM = "|||T5|||"
DELIM_EXT = "|||T5EXT|||"

T5_UNK_ID = 2
T5_PAD_ID = 0
T5_TABLE_SIZE = ext_vocab.T5_TABLE_SIZE


class SplitTokenizeStrategy(strategy_anima.AnimaTokenizeStrategy):
    """Tokenize strategy that lets a prompt carry a separate T5-side text.

    ``"<qwen text>|||T5|||<t5 text>"`` routes the left side to the Qwen3
    tokenizer and the right side to the T5 tokenizer;
    ``"<qwen text>|||T5EXT|||<t5 text>"`` routes the right side through the
    extended-vocab :class:`ext_vocab.HybridT5Encoder` (set ``ext_encoder``
    before use). Prompts without a delimiter behave exactly like the stock
    strategy. Probe-only — nothing in the shipped pipeline emits delimiters.
    """

    ext_encoder = None  # HybridT5Encoder, attached in main() when --ext

    def tokenize(self, text):
        texts = [text] if isinstance(text, str) else list(text)
        qwen_texts, t5_rows = [], []
        for t in texts:
            if DELIM_EXT in t:
                q, t5 = t.split(DELIM_EXT, 1)
                assert self.ext_encoder is not None, "--ext assets not loaded"
                ids, mask = self.ext_encoder.encode(t5, self.t5_max_length)
                t5_rows.append((torch.tensor(ids), torch.tensor(mask)))
            else:
                if DELIM in t:
                    q, t5 = t.split(DELIM, 1)
                else:
                    q = t5 = t
                enc = self.t5_tokenizer(
                    t5,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=self.t5_max_length,
                )
                t5_rows.append((enc["input_ids"][0], enc["attention_mask"][0]))
            qwen_texts.append(q)

        qwen3_encoding = self.qwen3_tokenizer(
            qwen_texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.qwen3_max_length,
        )
        return [
            qwen3_encoding["input_ids"],
            qwen3_encoding["attention_mask"],
            torch.stack([r[0] for r in t5_rows]),
            torch.stack([r[1] for r in t5_rows]),
        ]


# Two content prompts per language: P1 carries embedded text-to-render (the
# OCR-caption use case), P2 is a plain scene — the pair drives the
# discrimination readout. `rom` is a plain-ASCII romanization.
PROMPTS = {
    "p1_sign": {
        "en": (
            "1girl, black hair, school uniform, classroom, holding a wooden "
            'sign with "GOOD MORNING" written on it, anime style'
        ),
        "ja": "黒髪の女子学生が教室で「おはよう」と書かれた木の看板を持っている、アニメ風",
        "ja_rom": (
            'kurokami no joshigakusei ga kyoushitsu de "ohayou" to kakareta '
            "ki no kanban wo motteiru, anime-fuu"
        ),
        "ko": '검은 머리 여학생이 교실에서 "좋은 아침"이라고 적힌 나무 팻말을 들고 있다, 애니메이션 스타일',
        "ko_rom": (
            'geomeun meori yeohaksaengi gyosileseo "joeun achim" irago '
            "jeokhin namu paetmareul deulgo itda, aenimeisyeon seutail"
        ),
        "zh": "黑发女学生在教室里拿着写有「早上好」的木牌，动漫风格",
        "zh_rom": (
            'heifa nüxuesheng zai jiaoshi li nazhe xieyou "zaoshang hao" '
            "de mupai, dongman fengge"
        ),
    },
    "p2_beach": {
        "en": "a woman in a red dress walking a white dog on the beach at sunset",
        "ja": "赤いドレスの女性が夕暮れの浜辺で白い犬を散歩させている",
        "ja_rom": (
            "akai doresu no josei ga yuugure no hamabe de shiroi inu wo sanpo saseteiru"
        ),
        "ko": "빨간 드레스를 입은 여성이 해질녘 해변에서 하얀 강아지를 산책시키고 있다",
        "ko_rom": (
            "ppalgan deureseureul ibeun yeoseongi haejilnyeok haebyeoneseo "
            "hayan gangajireul sanchaeksikigo itda"
        ),
        "zh": "穿红色连衣裙的女人在日落时的海滩上遛一只白色的狗",
        "zh_rom": (
            "chuan hongse lianyiqun de nüren zai riluo shi de haitan shang "
            "liu yizhi baise de gou"
        ),
    },
}


def build_arms(languages, with_ext=False, prompts=None, keep=None):
    """Return {content: {arm_name: composed prompt string}}.

    ``keep`` restricts the arm set (the Phase-2c eyeball grid wants only
    ``en,ja_t5en,ja_ext`` over ~20 prompts, not 6 arms over 2). A prompt set
    that omits ``{lang}_rom`` simply drops the ``t5rom`` arm — romanization was
    closed in Phase 0, so an eval set has no reason to carry transliterations.
    """
    arms = {}
    for content, p in (prompts or PROMPTS).items():
        if content.startswith("_"):  # `_comment` and friends in a JSON set
            continue
        table = {"en": p["en"]}
        for lang in languages:
            if lang not in p:
                continue
            native, en = p[lang], p["en"]
            table[f"{lang}_native"] = native
            table[f"{lang}_t5en"] = f"{native}{DELIM}{en}"
            if f"{lang}_rom" in p:
                table[f"{lang}_t5rom"] = f"{native}{DELIM}{p[f'{lang}_rom']}"
            table[f"{lang}_q_en"] = f"{en}{DELIM}{native}"
            if with_ext:
                table[f"{lang}_ext"] = f"{native}{DELIM_EXT}{native}"
        if keep:
            table = {k: v for k, v in table.items() if k in keep}
        arms[content] = table
    return arms


def t5_side_stats(embed):
    """Non-pad / unk / extended-row counts for the T5 target ids of one prompt."""
    t5_ids, t5_mask = embed[2][0], embed[3][0]
    nonpad = int(t5_mask.sum().item())
    unks = int((t5_ids[:nonpad] == T5_UNK_ID).sum().item())
    ext = int((t5_ids[:nonpad] >= T5_TABLE_SIZE).sum().item())
    return {"t5_tokens_nonpad": nonpad, "t5_unk": unks, "t5_ext": ext}


def flat_cos(a, b):
    return float(
        F.cosine_similarity(
            a.flatten().float().unsqueeze(0), b.flatten().float().unsqueeze(0)
        ).item()
    )


def main() -> None:
    ckpt = default_checkpoints()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dit", default=ckpt.dit)
    parser.add_argument("--vae", default=ckpt.vae)
    parser.add_argument("--text_encoder", default=ckpt.text_encoder)
    parser.add_argument("--label", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["ja", "ko", "zh"],
        choices=["ja", "ko", "zh"],
    )
    parser.add_argument(
        "--skip_images",
        action="store_true",
        help="Adapter-output diagnostics only — no sampling, no VAE.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Block-compile the DiT (single 1024² shape → one graph, warmup "
        "paid once across all arms).",
    )
    parser.add_argument(
        "--lora",
        nargs="*",
        default=None,
        help="DiT LoRA checkpoint(s) to load (merge-or-live per checkpoint "
        "metadata, applied inside load_dit_model before any block compile). "
        "Use it to render an ext-trained LoRA through the ext encoder.",
    )
    parser.add_argument("--lora_multiplier", type=float, default=1.0)
    parser.add_argument(
        "--ext",
        action="store_true",
        help="Add {lang}_ext arms: native CJK on both sides, T5 side through "
        "the extended vocab (build assets first via build_ext.py).",
    )
    parser.add_argument(
        "--ext_prefix",
        type=Path,
        default=Path(__file__).resolve().parent / "assets" / "ext_embed",
        help="Path prefix of the {.safetensors,.json} vocab pack the ext arms "
        "read. Defaults to the Phase-1 zero-shot build; point it at a distilled "
        "pack (output/ckpt/cjk_vocab_pack*) to score Phase 2c.",
    )
    parser.add_argument(
        "--adapter_lora",
        default=None,
        help="ext-gated adapter LoRA sidecar (plan3) to hook onto llm_adapter "
        "before encoding; 'auto' = <ext_prefix>.adapter_lora.safetensors. "
        "EN / t5en arms carry no ext id so their gate is 0 — only *_ext arms "
        "see the delta.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="JSON {content: {en, ja, ...}} replacing the built-in 2-prompt "
        "set — the wider eyeball grid. Missing {lang}_rom drops the t5rom arm.",
    )
    parser.add_argument(
        "--arms",
        default="",
        help="comma-separated arm allow-list (e.g. 'en,ja_t5en,ja_ext'); "
        "default = every arm the languages imply.",
    )
    opts = parser.parse_args()

    run_dir = make_run_dir("cjk_adapter", opts.label)
    print(f"run dir: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = GenerationRequest(
        dit=opts.dit,
        vae=opts.vae,
        text_encoder=opts.text_encoder,
        prompt="placeholder",
        save_path=str(run_dir),
        infer_steps=opts.steps,
        guidance_scale=opts.cfg,
        image_size=(opts.size[0], opts.size[1]),
        seed=opts.seed,
        lora_weight=opts.lora or None,
        lora_multiplier=opts.lora_multiplier if opts.lora else None,
    ).to_args()
    args.device = device
    # Block compile (repo-preferred over whole-model torch.compile): applied
    # inside load_dit_model, after any adapter attach (compile-after-apply).
    args.compile_blocks = bool(opts.compile)
    check_inputs(args)

    # Install the split strategy BEFORE anything calls ensure_text_strategies
    # (which is a no-op once both singletons are set).
    tokenize_strategy = SplitTokenizeStrategy(qwen3_path=opts.text_encoder)
    text_strategies.TokenizeStrategy.set_strategy(tokenize_strategy)
    text_strategies.TextEncodingStrategy.set_strategy(
        strategy_anima.AnimaTextEncodingStrategy()
    )

    ext_table = None
    if opts.ext:
        ext_table, mapping = ext_vocab.load_ext_assets(opts.ext_prefix)
        tokenize_strategy.ext_encoder = ext_vocab.HybridT5Encoder.from_mapping(
            tokenize_strategy.t5_tokenizer, tokenize_strategy.qwen3_tokenizer, mapping
        )
        trained = (mapping.get("training") or {}).get("losses")
        print(
            f"ext vocab: {ext_table.shape[0]} rows loaded from {opts.ext_prefix} "
            f"({'distilled: ' + str(trained) if trained else 'zero-shot build'})"
        )

    prompts = None
    if opts.prompts:
        prompts = json.loads(opts.prompts.read_text(encoding="utf-8"))
    keep = {a.strip() for a in opts.arms.split(",") if a.strip()}
    arms = build_arms(opts.languages, with_ext=opts.ext, prompts=prompts, keep=keep)

    # ---- encode all arms (DiT needed: the adapter lives inside it) ----------
    anima = load_dit_model(args, device, torch.bfloat16)
    if ext_table is not None:
        # Append the mapped CJK rows to the adapter's frozen query table.
        emb = anima.llm_adapter.embed
        new_w = torch.cat(
            [emb.weight.data, ext_table.to(emb.weight.dtype).to(emb.weight.device)]
        )
        anima.llm_adapter.embed = torch.nn.Embedding.from_pretrained(new_w)
    adapter_lora = None
    if opts.adapter_lora:
        from scripts.distill_cjk.adapter_lora import AdapterLoRA

        lora_path = (
            opts.ext_prefix.with_name(
                opts.ext_prefix.name + ".adapter_lora.safetensors"
            )
            if opts.adapter_lora == "auto"
            else Path(opts.adapter_lora)
        )
        if ext_table is None:
            raise SystemExit(
                "--adapter_lora needs --ext (the gate never opens otherwise)"
            )
        adapter_lora = AdapterLoRA.load(anima.llm_adapter, lora_path)
        print(
            f"adapter LoRA: r={adapter_lora.rank} targets={','.join(adapter_lora.targets)} "
            f"({adapter_lora.n_params() / 1e6:.2f} M) hooked from {lora_path}"
        )
    elif opts.ext:
        sib = opts.ext_prefix.with_name(
            opts.ext_prefix.name + ".adapter_lora.safetensors"
        )
        if sib.exists():
            print(
                f"NOTE: {sib.name} exists but --adapter_lora not given — rows-only arm"
            )
    text_encoder = load_text_encoder(args, dtype=torch.bfloat16, device=device)
    shared = {"text_encoder": text_encoder, "conds_cache": {}}

    contexts = {}  # (content, arm) -> (context, context_null)
    for content, table in arms.items():
        for arm, prompt in table.items():
            arm_args = copy.copy(args)
            arm_args.prompt = prompt
            ctx, ctx_null = prepare_text_inputs(arm_args, device, anima, shared)
            contexts[(content, arm)] = (ctx, ctx_null)
    del text_encoder, shared
    clean_memory_on_device(device)

    # ---- adapter-output diagnostics ----------------------------------------
    diagnostics = {}
    for content, table in arms.items():
        # `--arms` may drop `en` (ext-only grids); cos_vs_en is then undefined.
        base_ctx = contexts.get((content, "en"))
        base = base_ctx[0]["embed"][0] if base_ctx is not None else None
        rows = {}
        for arm in table:
            emb = contexts[(content, arm)][0]["embed"]
            row = t5_side_stats(emb)
            row["cos_vs_en"] = flat_cos(emb[0], base) if base is not None else None
            rows[arm] = row
        diagnostics[content] = rows

    # P1-vs-P2 discrimination per arm type: cosine between the two contents'
    # adapter outputs. ~1.0 for an arm type means different prompts collapse
    # to the same conditioning under that routing.
    contents = list(arms)
    c1, c2 = contents[0], contents[1]
    discrimination = {}
    for arm in arms[c1]:
        discrimination[arm] = flat_cos(
            contexts[(c1, arm)][0]["embed"][0], contexts[(c2, arm)][0]["embed"][0]
        )
    # With a wider prompt set the first-two-prompts reading is an arbitrary
    # sample of one pair; average over every content pair as well. The
    # `_p1_vs_p2` key keeps its exact old meaning so historical runs stay
    # comparable.
    discrimination_mean = {}
    if len(contents) > 2:
        for arm in arms[c1]:
            vals = [
                flat_cos(
                    contexts[(a, arm)][0]["embed"][0], contexts[(b, arm)][0]["embed"][0]
                )
                for a, b in itertools.combinations(contents, 2)
                if arm in arms[a] and arm in arms[b]
            ]
            discrimination_mean[arm] = sum(vals) / len(vals)

    print("\n=== adapter-output diagnostics ===")
    for content, rows in diagnostics.items():
        print(f"[{content}]")
        for arm, row in rows.items():
            print(
                f"  {arm:14s} t5_nonpad={row['t5_tokens_nonpad']:3d} "
                f"unk={row['t5_unk']:2d} ext={row['t5_ext']:2d} "
                f"cos_vs_en={row['cos_vs_en']:+.4f}"
                if row["cos_vs_en"] is not None
                else "cos_vs_en=n/a"
            )
    print("[p1-vs-p2 discrimination]")
    for arm, cos in discrimination.items():
        print(f"  {arm:14s} cos={cos:+.4f}", end="")
        print(
            f"   mean-over-pairs={discrimination_mean[arm]:+.4f}"
            if discrimination_mean
            else ""
        )

    # ---- render every arm (same seed / noise, conditioning is the only     ----
    # ---- free variable), then decode after the DiT is freed                ----
    images = {}
    if not opts.skip_images:
        latents = {}
        with torch.no_grad():
            for content, table in arms.items():
                for arm in table:
                    ctx, ctx_null = contexts[(content, arm)]
                    arm_args = copy.copy(args)
                    arm_args.prompt = arms[content][arm]
                    print(f"sampling {content}/{arm} …")
                    latents[(content, arm)] = generate_body(
                        arm_args, anima, ctx, ctx_null, device, opts.seed
                    ).cpu()
        del anima
        clean_memory_on_device(device)

        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae, device="cpu", disable_mmap=True, vae_2d=args.vae_2d
        )
        vae.to(device, dtype=torch.bfloat16)
        vae.eval()
        for (content, arm), latent in latents.items():
            arm_args = copy.copy(args)
            arm_args.prompt = arms[content][arm]
            save_output(
                arm_args,
                vae,
                latent.to(device),
                device,
                original_base_name=f"{content}_{arm}",
            )
            images[f"{content}_{arm}"] = True
        del vae
        clean_memory_on_device(device)

    write_result(
        run_dir,
        script=__file__,
        args=opts,
        label=opts.label,
        metrics={
            "diagnostics": diagnostics,
            "discrimination_p1_vs_p2": discrimination,
            **(
                {"discrimination_mean_over_pairs": discrimination_mean}
                if discrimination_mean
                else {}
            ),
            "ext_prefix": str(opts.ext_prefix) if opts.ext else None,
            "lora": list(opts.lora or []),
            "lora_multiplier": opts.lora_multiplier if opts.lora else None,
            "adapter_lora": (
                {
                    "r": adapter_lora.rank,
                    "targets": list(adapter_lora.targets),
                    "n_params": adapter_lora.n_params(),
                }
                if adapter_lora is not None
                else None
            ),
            "arms": {c: list(t) for c, t in arms.items()},
            "images_rendered": sorted(images),
        },
    )
    print(f"\nresult.json → {run_dir}")


if __name__ == "__main__":
    main()
