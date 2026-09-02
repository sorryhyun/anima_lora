"""Local MT engine for the CJK distillation corpus — Hy-MT2 (Tencent, Apache-2.0).

Phase 2a needs *meaning-preserving* JA, not human-quality prose: the teacher is
computed at embedding level from the EN side, so MT output is first-class
training data (see ``../plan.md``). What it *does* need is instruction
following — the lexicon pre-substitution is a terminology constraint
("``acheron`` translates to ``黄泉``"), and tag-strings must come back with
their comma structure intact. Hy-MT2 is a translation-specialised model with
documented prompt templates for exactly those two things.

Model: ``tencent/Hy-MT2-1.8B`` (bf16 ≈ 3.6 GB — leaves the whole 16 GB card for
batch). 7B/30B-A3B are drop-in via ``--model`` if a spot-check says 1.8B is not
enough.

This module is a library first (``MTEngine``); ``--smoke`` runs a small
self-check that prints tag/caption translations for eyeballing. GPU work goes
through the daemon:

    make daemon-run ARGS="project/cjk_aware_anima/datasets/mt.py --smoke"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "tencent/Hy-MT2-1.8B"
LANG_NAMES = {"ja": "Japanese", "ko": "Korean", "zh": "Chinese", "en": "English"}

# Sampling params recommended by the model card for 1.8B / 7B.
CARD_SAMPLING = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.6,
    "top_k": 20,
    "repetition_penalty": 1.05,
}

_ONLY = (
    "Note that you must **ONLY output the translated result without any "
    "additional explanation**"
)


@dataclass
class Request:
    """One translation unit. Optional fields map to Hy-MT2 prompt templates."""

    text: str
    terms: list[tuple[str, str]] = field(default_factory=list)  # (source, target)
    background: str | None = None
    style: str | None = None
    keep_delimiters: bool = False
    meta: dict = field(default_factory=dict)  # passthrough, never sent to the model


def build_prompt(req: Request, target_lang: str = "ja") -> str:
    """Compose the card's single-feature templates into one instruction.

    Order follows the card: reference material first (terminology, background),
    then the task line carrying the style / delimiter constraints, then the
    source text last so it is adjacent to the generation point.
    """
    lang = LANG_NAMES.get(target_lang, target_lang)
    head: list[str] = []
    if req.terms:
        head.append("Reference the following translations:")
        head += [f"{src} translates to {tgt}" for src, tgt in req.terms]
        head.append("")
    if req.background:
        head.append("[Background Information]")
        head.append(req.background)
        head.append("")

    task = f"Translate the following text into {lang}."
    if req.style:
        task += f" The translation style must strictly conform to [**{req.style}**]."
    if req.keep_delimiters:
        task += (
            " You must **retain the exact same number of delimiters in the "
            "translation. Strictly do not omit, escape, or translate these "
            "symbols, and pay close attention to their placement**."
        )
    task += f" {_ONLY}:"

    return "\n".join([*head, task, "", req.text])


def cache_key(
    model_path: str,
    prompt: str,
    *,
    target_lang: str,
    max_new_tokens: int,
    greedy: bool,
    seed: int = 0,
) -> str:
    """Hash everything that changes the output — model, decode mode, prompt.

    Module-level so a caller can look a translation up **without loading the
    model**: a partial pass can be harvested from its cache on CPU.
    """
    mode = "greedy" if greedy else f"sample{seed}"
    blob = f"{model_path}|{target_lang}|{mode}|{max_new_tokens}|{prompt}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def load_cache(cache: "Path | None") -> dict[str, str]:
    """key -> translation, tolerating a torn last line from a killed job."""
    if cache is None or not cache.exists():
        return {}
    out: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[row["key"]] = row["out"]
    return out


class MTEngine:
    """Batched chat-template generation over a local Hy-MT2 checkpoint."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        greedy: bool = False,
        seed: int = 0,
        max_new_tokens: int = 512,
        gpu_budget: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_path = model_path
        self.greedy = greedy
        self.seed = seed
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # 7B bf16 is ~15.2 GB against ~15.5 GB usable on a 16 GB card — it OOMs
        # by a hair. `gpu_budget` caps what lands on the GPU and lets accelerate
        # keep the overflow blocks on CPU, streaming them per forward (the same
        # trick as the trainer's blocks_to_swap). A couple of offloaded blocks
        # cost far less than dropping to the 1.8B.
        kwargs: dict = {"dtype": torch.bfloat16}
        if gpu_budget:
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {0: gpu_budget, "cpu": "64GiB"}
        else:
            kwargs["device_map"] = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()
        devs = {str(d) for d in getattr(self.model, "hf_device_map", {}).values()}
        if devs - {"0", "cuda:0", "cuda"}:
            print(
                f"  [mt] device map spans {sorted(devs)} (CPU blocks stream per forward)"
            )

    def gen_kwargs(self) -> dict:
        if self.greedy:
            return {"do_sample": False}
        return dict(CARD_SAMPLING)

    def translate(
        self,
        requests: list[Request],
        *,
        target_lang: str = "ja",
        batch_size: int = 32,
        progress_every: int = 0,
        max_new_tokens: int | None = None,
        cache: "Path | None" = None,
    ) -> list[str]:
        """Batch-generate, appending each batch to ``cache`` as it lands.

        Keep ``max_new_tokens`` tight for short units: a batch runs until
        *every* sequence finishes, so one rambling output stretches the whole
        batch to the cap — at 512 that made tag translation ~14x slower than it
        needed to be.

        A full pass is hours of GPU, so nothing is held only in memory: every
        batch is appended to a JSONL cache keyed by the exact prompt, and a
        re-run skips whatever is already there. Killing or losing a job then
        costs one batch, not the whole pass.
        """
        torch = self.torch
        budget = max_new_tokens or self.max_new_tokens
        torch.manual_seed(self.seed)

        prompts = [build_prompt(r, target_lang) for r in requests]
        keys = [self._cache_key(p, target_lang, budget) for p in prompts]
        done: dict[str, str] = load_cache(cache)
        todo = [i for i, k in enumerate(keys) if k not in done]
        if cache is not None and len(todo) < len(requests):
            print(
                f"  [mt] resuming: {len(requests) - len(todo)}/{len(requests)} already "
                f"cached in {cache}",
                flush=True,
            )

        handle = None
        if cache is not None and todo:
            cache.parent.mkdir(parents=True, exist_ok=True)
            handle = cache.open("a", encoding="utf-8")
        try:
            for start in range(0, len(todo), batch_size):
                idx = todo[start : start + batch_size]
                enc = self.tokenizer(
                    [
                        self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompts[i]}],
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                        for i in idx
                    ],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(self.model.device)
                with torch.no_grad():
                    ids = self.model.generate(
                        **enc,
                        max_new_tokens=budget,
                        pad_token_id=self.tokenizer.pad_token_id,
                        **self.gen_kwargs(),
                    )
                new = ids[:, enc["input_ids"].shape[-1] :]
                for i, text in zip(
                    idx, self.tokenizer.batch_decode(new, skip_special_tokens=True)
                ):
                    done[keys[i]] = text.strip()
                    if handle:
                        handle.write(
                            json.dumps(
                                {"key": keys[i], "out": text.strip()},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                if handle:
                    handle.flush()
                if progress_every and (start // batch_size) % progress_every == 0:
                    print(
                        f"  [mt] {min(start + batch_size, len(todo))}/{len(todo)}",
                        flush=True,
                    )
        finally:
            if handle:
                handle.close()
        return [done[k] for k in keys]

    def _cache_key(self, prompt: str, target_lang: str, budget: int) -> str:
        return cache_key(
            self.model_path,
            prompt,
            target_lang=target_lang,
            max_new_tokens=budget,
            greedy=self.greedy,
            seed=self.seed,
        )


TAG_BACKGROUND = (
    "These are Danbooru-style annotation tags for an anime/illustration image "
    "dataset. Each tag names a visual attribute, pose, clothing item, body "
    "feature, camera framing, or scene element. Translate as a short noun "
    "phrase a Japanese user would type as an image-generation prompt keyword, "
    "not as a sentence. Use the wording actually used by Japanese illustration "
    "communities (pixiv / Japanese Danbooru tags), including established "
    "katakana loanwords, rather than a literal word-by-word rendering."
)

# Few-shot exemplars for the Terminology template: teach the *register* of
# Japanese illustration tags (loanwords, established idiom) rather than
# hand-curating every head tag. Deliberately disjoint from IDIOM_PROBE below,
# so the probe measures generalisation and not recall of what we handed over.
TAG_FEWSHOT: list[tuple[str, str]] = [
    ("1girl", "女の子1人"),
    ("solo", "一人"),
    ("blonde hair", "金髪"),
    ("short hair", "ショートヘア"),
    ("hair ornament", "髪飾り"),
    ("open mouth", "口を開ける"),
    ("bare shoulders", "肩出し"),
    ("thighhighs", "ニーハイ"),
    ("school uniform", "制服"),
    ("simple background", "シンプルな背景"),
    ("from behind", "後ろから"),
    ("full body", "全身"),
    ("sitting", "座る"),
    ("closed eyes", "目を閉じる"),
    ("large breasts", "巨乳"),
    ("censored", "修正あり"),
]

TAG_BACKGROUND_KO = (
    "These are Danbooru-style annotation tags for an anime/illustration image "
    "dataset. Each tag names a visual attribute, pose, clothing item, body "
    "feature, camera framing, or scene element. Translate as a short noun "
    "phrase a Korean user would type as an image-generation prompt keyword, "
    "not as a sentence. Use the wording actually used by Korean illustration "
    "communities (Arca Live / Korean-speaking Danbooru users), including "
    "established loanwords written in hangul, rather than a literal "
    "word-by-word rendering. Never use hanja."
)

# KO few-shot exemplars (plan_ko.md K1: hand-checked list, never drawn from the
# unverified wiki head). Drafted 2026-08-31; part of the K1 review sign-off.
# Kept disjoint from IDIOM_PROBE so a future KO probe port stays held-out.
TAG_FEWSHOT_KO: list[tuple[str, str]] = [
    ("1girl", "소녀 1명"),
    ("solo", "솔로"),
    ("blonde hair", "금발"),
    ("short hair", "단발"),
    ("hair ornament", "머리 장식"),
    ("open mouth", "입 벌림"),
    ("bare shoulders", "어깨 노출"),
    ("thighhighs", "니삭스"),
    ("school uniform", "교복"),
    ("simple background", "단순한 배경"),
    ("from behind", "뒤에서"),
    ("full body", "전신"),
    ("sitting", "앉아 있음"),
    ("closed eyes", "눈 감음"),
    ("large breasts", "거유"),
    ("censored", "검열"),
]

TAG_BACKGROUND_ZH = (
    "These are Danbooru-style annotation tags for an anime/illustration image "
    "dataset. Each tag names a visual attribute, pose, clothing item, body "
    "feature, camera framing, or scene element. Translate into Simplified "
    "Chinese as a short noun phrase a Chinese user would type as an "
    "image-generation prompt keyword, not as a sentence. Use the wording "
    "actually used by Chinese illustration communities (NGA / Bilibili / "
    "Chinese-speaking Danbooru users, e.g. 双马尾, 呆毛, 兽耳), rather than a "
    "literal word-by-word rendering. Simplified characters only."
)

# zh few-shot exemplars (plan_zh.md Z1): hand-checked against the NGA
# community translation, never drawn from the unverified wiki head. Kept
# disjoint from IDIOM_PROBE so a future zh probe port stays held-out.
TAG_FEWSHOT_ZH: list[tuple[str, str]] = [
    ("1girl", "1个女性"),
    ("solo", "单人"),
    ("blonde hair", "金发"),
    ("short hair", "短发"),
    ("hair ornament", "发饰"),
    ("open mouth", "张嘴"),
    ("bare shoulders", "露肩"),
    ("thighhighs", "过膝袜"),
    ("school uniform", "校服"),
    ("simple background", "简单背景"),
    ("from behind", "背影"),
    ("full body", "全身"),
    ("sitting", "坐着"),
    ("closed eyes", "闭眼"),
    ("large breasts", "巨乳"),
    ("censored", "打码"),
]

# Held-out idiom probe: tags whose Japanese form is an established community
# idiom a literal translation gets wrong. `expect` is the accepted rendering
# (substring match — inflection/particles around it are fine); `literal_trap`
# records what a word-by-word rendering produces, for the report.
IDIOM_PROBE: list[tuple[str, str, str]] = [
    ("twintails", "ツインテール", "二本の髪"),
    ("looking at viewer", "カメラ目線", "視点を見る"),
    ("blush", "赤面", "顔赤み"),
    ("ponytail", "ポニーテール", "馬の尻尾"),
    ("ahoge", "アホ毛", ""),
    ("braid", "三つ編み", ""),
    ("cat ears", "猫耳", "猫の耳"),
    ("animal ears", "獣耳", "動物の耳"),
    ("small breasts", "貧乳", "小さな胸"),
    ("navel", "へそ", ""),
    ("collarbone", "鎖骨", ""),
    ("armpits", "脇", "腋の下"),
    ("swimsuit", "水着", ""),
    ("barefoot", "裸足", ""),
    ("bar censor", "バー修正", "バー検閲"),
    ("mosaic censoring", "モザイク", "モザイク検閲"),
    ("serafuku", "セーラー服", "セラフク"),
    ("wariza", "割座", "ワリザ"),
    ("ahegao", "アヘ顔", ""),
    ("cowboy shot", "カウボーイショット", ""),
    ("speech bubble", "吹き出し", "スピーチバブル"),
    ("hair between eyes", "前髪", "目尻の髪"),
    ("sidelocks", "もみあげ", "横の髪"),
    ("one eye closed", "ウィンク", "片目を閉じる"),
    ("tongue out", "舌を出す", ""),
    ("sweatdrop", "汗", ""),
    ("pointy ears", "とがった耳", ""),
    ("multicolored hair", "多色", "多色の髪"),
]


def _probe(args: argparse.Namespace) -> None:
    """Score few-shot exemplars vs a bare prompt on held-out idiom tags.

    The question this answers: can the Terminology template *teach the
    register* from ~16 examples, or does the head of the tag vocabulary need
    hand curation? Arms differ only in whether TAG_FEWSHOT is in the prompt.
    """
    engine = MTEngine(args.model, greedy=args.greedy, gpu_budget=args.gpu_budget)
    tags = [t for t, _, _ in IDIOM_PROBE]
    arms = {
        "shot0": [Request(text=t, background=TAG_BACKGROUND) for t in tags],
        "fewshot": [
            Request(text=t, background=TAG_BACKGROUND, terms=TAG_FEWSHOT) for t in tags
        ],
    }
    got = {
        name: engine.translate(reqs, batch_size=args.batch_size)
        for name, reqs in arms.items()
    }

    print(f"\n=== idiom probe — {args.model} (greedy={args.greedy}) ===")
    print(f"{'tag':22s} {'expected':12s} {'shot0':22s} {'fewshot':22s}")
    score = dict.fromkeys(arms, 0)
    for i, (tag, expect, _trap) in enumerate(IDIOM_PROBE):
        cells = []
        for name in arms:
            out = got[name][i]
            hit = expect in out
            score[name] += hit
            cells.append(("✓ " if hit else "✗ ") + out)
        print(f"{tag:22s} {expect:12s} {cells[0]:22s} {cells[1]:22s}")
    n = len(IDIOM_PROBE)
    print("\nidiom hit rate:")
    for name in arms:
        print(f"  {name:8s} {score[name]:2d}/{n} = {100 * score[name] / n:.0f}%")

    if args.out:
        payload = {
            "model": args.model,
            "greedy": args.greedy,
            "n": n,
            "score": score,
            "rows": [
                {
                    "tag": t,
                    "expect": e,
                    **{name: got[name][i] for name in arms},
                }
                for i, (t, e, _) in enumerate(IDIOM_PROBE)
            ],
        }
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")


def _smoke(args: argparse.Namespace) -> None:
    """Translate a handful of real tags + captions and print them for eyeballing."""
    root = Path(__file__).resolve().parent
    lex = json.loads((root / "assets" / "wikidata_lexicon.json").read_text())
    chars = lex["characters"]

    tags = [
        "1girl",
        "long hair",
        "looking at viewer",
        "cat ears",
        "thigh strap",
        "open mouth",
        "cowboy shot",
        "bar censor",
        "hair between eyes",
        "sitting on lap",
        "white dress",
        "blush",
        "twintails",
        "navel",
        "wariza",
        "off-shoulder shirt",
        "holding umbrella",
        "night sky",
    ]
    reqs = [Request(text=t, background=TAG_BACKGROUND) for t in tags]

    caption_terms = [
        (k.split(" (")[0], v["ja"])
        for k, v in list(chars.items())[:0]  # filled per-caption below
    ]
    captions = [
        (
            "sensitive, 1girl, acheron (honkai: star rail), long hair, purple "
            "hair, sword, looking at viewer, night, city lights",
            [("acheron", chars["acheron (honkai: star rail)"]["ja"])],
        ),
        (
            "1girl, wakamo (blue archive), blue archive, halo, smile, white "
            "dress, sitting, garden",
            [("wakamo", chars.get("wakamo (blue archive)", {}).get("ja", "若藻"))],
        ),
    ]
    del caption_terms

    engine = MTEngine(args.model, greedy=args.greedy, gpu_budget=args.gpu_budget)

    print("\n=== tags (background-conditioned) ===", flush=True)
    for tag, ja in zip(tags, engine.translate(reqs, batch_size=args.batch_size)):
        print(f"  {tag:28s} → {ja}")

    print(
        "\n=== captions: tag-string register (delimiter-locked + terms) ===", flush=True
    )
    creqs = [
        Request(text=c, terms=t, background=TAG_BACKGROUND, keep_delimiters=True)
        for c, t in captions
    ]
    for (c, terms), ja in zip(
        captions, engine.translate(creqs, batch_size=args.batch_size)
    ):
        print(f"  EN: {c}\n  terms: {terms}\n  JA: {ja}\n")

    print("=== captions: natural-sentence register ===", flush=True)
    sreqs = [
        Request(
            text=c,
            terms=t,
            background=TAG_BACKGROUND,
            style="a natural single-paragraph Japanese description of the image",
        )
        for c, t in captions
    ]
    for (c, _), ja in zip(
        captions, engine.translate(sreqs, batch_size=args.batch_size)
    ):
        print(f"  EN: {c}\n  JA: {ja}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true", help="print sample translations")
    ap.add_argument(
        "--probe", action="store_true", help="score few-shot vs bare on idiom tags"
    )
    ap.add_argument("--greedy", action="store_true", help="deterministic decode")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None, help="probe: write JSON here")
    ap.add_argument(
        "--gpu-budget",
        default=None,
        help="cap GPU weight memory (e.g. 13GiB) and offload the rest to CPU — "
        "how the 7B fits a 16 GB card",
    )
    args = ap.parse_args()
    if args.probe:
        _probe(args)
    elif args.smoke:
        _smoke(args)
    else:
        ap.error("nothing to do — pass --probe or --smoke (this module is a library)")


if __name__ == "__main__":
    main()
