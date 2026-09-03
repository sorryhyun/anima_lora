"""CJK distillation config: argparser + resolved frozen dataclass.

Mirrors ``project/finished/mod_guidance/config.py`` — CLI-first, no TOML layer, since
every documented invocation drives the scripts purely through flags.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]

DEFAULT_PAIRS = REPO / "post_image_dataset" / "cjk_distill" / "pairs.jsonl"
DEFAULT_CACHE = REPO / "post_image_dataset" / "cjk_distill" / "cache"
DEFAULT_EXT = REPO / "bench" / "cjk_adapter" / "assets" / "ext_embed"
DEFAULT_OUT = REPO / "output" / "ckpt" / "cjk_vocab_pack"

# Per-`via` trust used by the span loss. The composed caption is the *student's*
# input, so a mistranslated tag trains that tag's ext rows toward the wrong
# English meaning (`colored inner hair` → 色付きの陰毛 = "colored pubic hair").
# `mt_unverified` is 34% of tag occurrences, so dropping it outright would cost
# a third of the corpus — down-weighting keeps the pair and demotes the span.
TRUST_POLICIES = {
    "all": {},  # every span at 1.0
    "provenance": {
        "override": 1.0,
        "wikidata": 1.0,
        "wiki_verified": 1.0,
        # A tag-pair candidate that won the back-translation arbitration
        # (D1-pairs item 2) — same verification layer as `wiki_verified`, same
        # community field one snapshot older.
        "tagpair_verified": 1.0,
        "wiki_han": 1.0,
        "rating": 1.0,
        "passthrough": 1.0,
        # `names`-register segments deliberately kept EN: teacher and student
        # agree on their ids, so the span isolates the swapped name's
        # contextual influence — exact signal, weight is the mixing knob.
        "en_pinned": 1.0,
        "mt_verified": 0.8,
        "wiki": 0.7,
        # `tagpair` fills tags the glossary left unresolved (p1atdev, CC0): the
        # community's own other_names, but an older snapshot, partly LLM-filled,
        # and with no back-translation behind the choice. Above `mt_unverified`,
        # below `wiki` — and note these rows have *no* supervision otherwise, so
        # the comparison is against 0, not against a better wording.
        "tagpair": 0.6,
        # KO r5: sub-floor KB keyword that lost (or never faced) back-translation
        # arbitration — community field, unverified, same trust class as
        # `tagpair`. Replaces `mt_unverified` wording for 97% of the KO tail
        # (reports/0901_ko_phase_k3.md).
        "kb_unverified": 0.6,
        # zh (plan_zh.md Z1): a curated community-pack wording (NGA / byzod)
        # that back-translated to the tag — same class as `wiki_verified`.
        "kb_verified": 1.0,
        # zh: curated pack wording chosen without back-translation support
        # (`kb` primary tier, above the floor, no candidate cleared F1) —
        # human community translation, unverified: `tagpair` class.
        "kb": 0.6,
        # desc_ko full-width spans (EN wiki sentence ↔ KB KO description):
        # human community translation, but loosely aligned (the KO side is a
        # summary, sometimes a sentence longer) — mt_verified class.
        "kb_desc": 0.8,
        "mt_unverified": 0.3,
        "unresolved": 0.0,
        "unmapped": 0.0,
    },
    # Ablation arm for G4: does the noise actually hurt, or is it averaged out?
    "verified_only": {
        "override": 1.0,
        "wikidata": 1.0,
        "wiki_verified": 1.0,
        "wiki_han": 1.0,
        "rating": 1.0,
        "passthrough": 1.0,
        "en_pinned": 1.0,
        "mt_verified": 1.0,
        "wiki": 1.0,
        "tagpair_verified": 1.0,  # back-translation-verified, unlike the fill
        "tagpair": 0.0,  # nothing back-translated it — this arm drops it
        "mt_unverified": 0.0,
        "unresolved": 0.0,
        "unmapped": 0.0,
    },
}

LOSS_NAMES = ("flat", "span", "attn", "pool")
PARAM_MODES = ("global", "row", "global_row")
MODES = ("train", "capacity", "oracle")


def parse_register_sampling(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in str(spec or "").split(","):
        if item.strip():
            k, v = item.split(":")
            out[k.strip()] = float(v)
    return out


def parse_register_span_scale(spec: str) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for item in str(spec or "").split(","):
        if item.strip():
            key, v = item.split("=")
            reg, via = key.split(":")
            out[(reg.strip(), via.strip())] = float(v)
    return out


def parse_losses(spec: str) -> dict[str, float]:
    """``"attn:1.0,span:0.5"`` → ``{"attn": 1.0, "span": 0.5}``."""
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, weight = part.partition(":")
        name = name.strip()
        if name not in LOSS_NAMES:
            raise ValueError(f"unknown loss {name!r} (known: {', '.join(LOSS_NAMES)})")
        out[name] = float(weight) if weight else 1.0
    if not out:
        raise ValueError("--loss selected nothing")
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # ---- inputs -----------------------------------------------------------
    p.add_argument("--dit", default=None, help="DiT safetensors (adapter source)")
    p.add_argument("--text_encoder", default=None, help="Qwen3 text encoder")
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--ext_prefix", type=Path, default=DEFAULT_EXT)
    p.add_argument(
        "--cache_dir",
        default=str(DEFAULT_CACHE),
        help="staged cache dir; distill also accepts a comma-separated list "
        "for joint training over separately staged corpora (plan_ko K2: "
        "cache_synth3,cache_ko — JA rows are never re-encoded)",
    )
    p.add_argument(
        "--registers",
        default="",
        help="comma-separated register allow-list for the *cache* (default: all)",
    )
    p.add_argument(
        "--train_registers",
        default="tags,tags_alt,names",
        help="registers the loss actually trains on. D6's quote registers are "
        "excluded by default because their teacher is degraded by construction "
        "— `quote_preserved` puts the raw JA string into the teacher's stock-"
        "spiece side (→ <unk>) and `quote_translated` replaces it with [TEXT], "
        "so both teach the ext rows to be vacuous exactly where verbatim glyph "
        "identity matters. They stay cached and are reported at eval; glyph "
        "identity is Phase 4's job. Pass '' to train on everything.",
    )
    p.add_argument("--max_pairs", type=int, default=0, help="0 = all")
    p.add_argument("--holdout", type=int, default=500, help="pairs held out of train")

    # ---- what is trainable ------------------------------------------------
    p.add_argument(
        "--param",
        default="global",
        choices=PARAM_MODES,
        help="global = 2-i-a (low-rank+diag correction shared by ALL ext rows, "
        "generalizes to the 95%% never visited); row = per-row residuals only; "
        "global_row = 2-i-b (both).",
    )
    p.add_argument("--rank", type=int, default=64, help="rank of the global map")
    p.add_argument(
        "--freeze_diag",
        action="store_true",
        help="global/global_row: keep the per-dim diagonal at identity (not "
        "trained) — low-rank + scalar gain only. Preserves the init key "
        "geometry the learned diagonal otherwise collapses.",
    )
    p.add_argument(
        "--min_visits",
        type=int,
        default=5,
        help="rows below this visit count get no per-row residual (they ride "
        "the global map alone) — 933 of 3002 visited rows are seen 1-4×.",
    )
    p.add_argument(
        "--tunable_rows_from",
        type=int,
        default=0,
        help="freeze every ext row below this index at its init (row/global_row "
        "modes) — the word-minting smoke trains only appended rows on top of a "
        "trained pack passed as --ext_prefix.",
    )
    p.add_argument(
        "--span_focus_from",
        type=int,
        default=0,
        help="zero the span-loss weight of every span whose student tokens are "
        "all below this ext-row index — concentrates the whole gradient on "
        "minted-row spans instead of diluting it across the frozen 90%%.",
    )
    p.add_argument(
        "--span_focus_bg",
        type=float,
        default=0.0,
        help="with --span_focus_from: weight kept by non-minted spans instead "
        "of 0 (plan_ko3 M2a mixed focus — a small background weight keeps the "
        "surrounding scene in the loss so minted rows stay on-manifold; the "
        "smoke's pure focus drifted renders into sketch/robot styles).",
    )
    p.add_argument(
        "--row_anchor",
        type=float,
        default=0.0,
        help="plan_ko3 M2b init-anchor: add λ·mean(‖residual‖²/‖init‖²) over "
        "the tunable rows so a minted row cannot buy span-cos with a large "
        "off-manifold excursion (m1 drift was ‖Δ‖/‖init‖ 0.13–0.29 and "
        "renders left the manifold while span loss improved).",
    )

    # ---- coverage (plan_zh2 U4 / U1) -----------------------------------------
    p.add_argument(
        "--holdout_rows",
        type=float,
        default=0.0,
        help="U4 row-disjoint holdout: hold out this fraction of the *visited* "
        "rows (seeded, stratified by script) — every span touching one leaves "
        "the training pool (spans, not pairs) and is scored at eval as "
        "`eval.row_holdout.*`, the first direct read of how the map does on a "
        "row it never trained. 0 = off.",
    )
    p.add_argument(
        "--holdout_rows_min_visits",
        type=int,
        default=5,
        help="rows eligible for --holdout_rows need at least this many visits "
        "(a row seen once has one occurrence to score).",
    )
    p.add_argument(
        "--holdout_rows_max_visits",
        type=int,
        default=500,
        help="rows at or above this visit count are never held out — the "
        "500+ band (~140 rows) carries a third of all visits, so holding it "
        "out strips ~5%% of the pool's span tokens for a question it does "
        "not answer. 0 = no cap.",
    )
    p.add_argument(
        "--holdout_rows_eval",
        type=int,
        default=2048,
        help="held-out spans scored per eval (the same count of trained spans "
        "is scored alongside as the in-distribution control).",
    )
    p.add_argument(
        "--span_min_visits",
        type=int,
        default=0,
        help="U1 visit floor: a span whose student tokens contain any ext row "
        "visited fewer than this many times (over the training pool) gets "
        "weight 0 — a row seen once is not a teacher; it rides the map like "
        "an unvisited one and is tagged `mapped-unseen` in the pack. 2 drops "
        "singletons, 5 matches --min_visits. 0 = off.",
    )
    p.add_argument(
        "--span_min_visits_bg",
        type=float,
        default=0.0,
        help="with --span_min_visits: weight kept by below-floor spans instead "
        "of 0 (mirrors --span_focus_bg).",
    )

    # ---- adapter capacity (plan3: ext-gated LoRA on the LLM Adapter) --------
    p.add_argument(
        "--adapter_lora",
        type=int,
        default=0,
        help="rank of an ext-gated LoRA on the adapter's per-block Linears "
        "(0 = off, rows only). Trained jointly with the ext table; the delta "
        "is gated to sequences carrying an ext id so pure-EN prompts stay "
        "bit-exact by construction. Written as <out>.adapter_lora.safetensors.",
    )
    p.add_argument(
        "--adapter_lora_targets",
        default="self_qkvo,cross_q",
        help="comma list of self_q|self_k|self_v|self_o|self_qkvo|cross_q|"
        "cross_k|cross_v|cross_o|cross_kv|mlp (per block, all 6 blocks)",
    )
    p.add_argument(
        "--adapter_lora_lr",
        type=float,
        default=1e-4,
        help="LoRA param-group LR (the table keeps --lr)",
    )
    p.add_argument(
        "--init_pack",
        type=Path,
        default=None,
        help="warm-start the ext rows from a trained pack prefix (its "
        "materialized ext_embed becomes the table's init) instead of the "
        "zero-shot --ext_prefix build",
    )

    # ---- objective --------------------------------------------------------
    p.add_argument("--loss", default="attn:1.0", help='e.g. "attn:1.0,span:0.5"')
    p.add_argument(
        "--register_sampling",
        default="",
        help='per-register batch sampling weight, e.g. "tags:1,names_synth:0.25" '
        "(unlisted registers weight 1.0; pairs are drawn proportional to weight)",
    )
    p.add_argument(
        "--register_span_scale",
        default="",
        help="multiply span weights of one via inside one register, e.g. "
        '"names_synth:en_pinned=0.3" — a distill-time knob, no cache rebuild',
    )
    p.add_argument(
        "--trust",
        default="provenance",
        choices=sorted(TRUST_POLICIES),
        help="per-span weight policy for the span loss",
    )
    p.add_argument(
        "--attn_blocks",
        default="0,13,27",
        help="DiT blocks sampled for the probe bank (early/mid/late of 28)",
    )
    p.add_argument("--attn_queries", type=int, default=64, help="probe queries/block")
    p.add_argument(
        "--query_bank",
        type=Path,
        default=None,
        help="real cross-attn queries (default: bench/cjk_distill/assets/"
        "query_bank.safetensors, built by scripts/distill_cjk/build_query_bank.py)",
    )
    p.add_argument(
        "--allow_random_queries",
        action="store_true",
        help="probe with random directions instead — reproduces the withdrawn "
        "G2 run only; the readout space is near-blind to wording (report_0816_phase2.md)",
    )

    # ---- loop -------------------------------------------------------------
    p.add_argument("--mode", default="train", choices=MODES)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--wd", type=float, default=0.0, help="decay of row residuals → init"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument(
        "--eval_limit",
        type=int,
        default=256,
        help="held-out pairs scored per eval (0 = the whole split). 256 keeps a "
        "G2 arm's eval cheap; a register-decomposed readout wants the whole "
        "split, since a 256-record prefix leaves some registers thin.",
    )
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--capacity_pairs", type=int, default=32, help="G0 overfit set size")

    # ---- outputs ----------------------------------------------------------
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--label", default=None, help="bench run-dir label")
    p.add_argument("--no_save", action="store_true")
    return p


@dataclass(frozen=True)
class CJKDistillConfig:
    dit: str
    text_encoder: str
    pairs: Path
    ext_prefix: Path
    cache_dir: Path  # primary (staging writes here); distill reads cache_dirs
    cache_dirs: tuple[Path, ...]
    registers: tuple[str, ...]
    train_registers: tuple[str, ...]
    max_pairs: int
    holdout: int

    param: str
    rank: int
    min_visits: int
    freeze_diag: bool = False
    tunable_rows_from: int = 0
    span_focus_from: int = 0
    span_focus_bg: float = 0.0
    row_anchor: float = 0.0
    holdout_rows: float = 0.0
    holdout_rows_min_visits: int = 5
    holdout_rows_max_visits: int = 500
    holdout_rows_eval: int = 2048
    span_min_visits: int = 0
    span_min_visits_bg: float = 0.0

    adapter_lora: int = 0
    adapter_lora_targets: str = "self_qkvo,cross_q"
    adapter_lora_lr: float = 1e-4
    init_pack: Path | None = None

    losses: dict[str, float] = field(default_factory=dict)
    trust: str = "provenance"
    register_sampling: dict[str, float] = field(default_factory=dict)
    register_span_scale: dict[tuple[str, str], float] = field(default_factory=dict)
    attn_blocks: tuple[int, ...] = ()
    attn_queries: int = 64
    query_bank: Path | None = None
    allow_random_queries: bool = False

    mode: str = "train"
    steps: int = 2000
    batch_size: int = 8
    lr: float = 1e-3
    wd: float = 0.0
    seed: int = 0
    eval_every: int = 250
    eval_limit: int = 256
    log_every: int = 25
    capacity_pairs: int = 32

    out: Path = DEFAULT_OUT
    label: str | None = None
    no_save: bool = False

    @property
    def trust_weights(self) -> dict[str, float]:
        return TRUST_POLICIES[self.trust]


def resolve_config(args: argparse.Namespace) -> CJKDistillConfig:
    """CLI namespace → frozen dataclass, plus the setup-time sanity checks."""
    from anima_lora import default_checkpoints

    ckpt = default_checkpoints()
    losses = parse_losses(args.loss)

    if args.mode == "oracle":
        # G0b feeds the student the teacher's own ids: the loss must be
        # identically zero. Only a position-wise loss can witness that.
        if losses != {"flat": 1.0}:
            logger.info("mode=oracle → forcing --loss flat:1.0")
        losses = {"flat": 1.0}

    if "span" in losses and args.trust == "all":
        logger.warning(
            "span loss with --trust all: mt_unverified spans (34%% of tag "
            "occurrences) supervise at full weight"
        )
    if args.param == "row" and args.min_visits <= 1:
        logger.warning(
            "--param row --min_visits<=1 gives a free 1024-dim vector to rows "
            "seen once; expect memorization, not generalization"
        )

    if args.adapter_lora:
        from scripts.distill_cjk.adapter_lora import parse_targets

        parse_targets(args.adapter_lora_targets)  # fail at parse time, not at attach
        if "attn" in losses:
            logger.info(
                "--adapter_lora with the attn loss: plan3 Phase 2 regulariser. "
                "(§9's 'attn hurts renders' was rows-only; with LoRA capacity, "
                "span-only smears — the attn term charges that. Health-metric "
                "interaction, not a wiring hazard.)"
            )
    if (
        args.init_pack is not None
        and not Path(args.init_pack).with_suffix(".safetensors").exists()
    ):
        raise FileNotFoundError(f"--init_pack {args.init_pack}.safetensors not found")

    if not 0.0 <= args.holdout_rows < 1.0:
        raise ValueError("--holdout_rows must be in [0, 1)")
    if args.holdout_rows and "span" not in losses:
        logger.warning(
            "--holdout_rows without the span loss: rows are held out of a loss "
            "that is not being trained; the row-holdout metric is still reported"
        )

    blocks = tuple(int(b) for b in str(args.attn_blocks).split(",") if b.strip() != "")
    if "attn" in losses and not blocks:
        raise ValueError("--loss attn needs at least one --attn_blocks entry")

    return CJKDistillConfig(
        dit=args.dit or ckpt.dit,
        text_encoder=args.text_encoder or ckpt.text_encoder,
        pairs=Path(args.pairs),
        ext_prefix=Path(args.ext_prefix),
        cache_dir=Path(str(args.cache_dir).split(",")[0]),
        cache_dirs=tuple(Path(p) for p in str(args.cache_dir).split(",") if p.strip()),
        registers=tuple(r for r in str(args.registers).split(",") if r.strip()),
        train_registers=tuple(
            r.strip() for r in str(args.train_registers).split(",") if r.strip()
        ),
        max_pairs=int(args.max_pairs),
        holdout=int(args.holdout),
        param=args.param,
        rank=int(args.rank),
        min_visits=int(args.min_visits),
        freeze_diag=bool(args.freeze_diag),
        tunable_rows_from=int(args.tunable_rows_from),
        span_focus_from=int(args.span_focus_from),
        span_focus_bg=float(args.span_focus_bg),
        row_anchor=float(args.row_anchor),
        holdout_rows=float(args.holdout_rows),
        holdout_rows_min_visits=int(args.holdout_rows_min_visits),
        holdout_rows_max_visits=int(args.holdout_rows_max_visits),
        holdout_rows_eval=int(args.holdout_rows_eval),
        span_min_visits=int(args.span_min_visits),
        span_min_visits_bg=float(args.span_min_visits_bg),
        adapter_lora=int(args.adapter_lora),
        adapter_lora_targets=str(args.adapter_lora_targets),
        adapter_lora_lr=float(args.adapter_lora_lr),
        init_pack=Path(args.init_pack) if args.init_pack else None,
        losses=losses,
        trust=args.trust,
        register_sampling=parse_register_sampling(args.register_sampling),
        register_span_scale=parse_register_span_scale(args.register_span_scale),
        attn_blocks=blocks,
        attn_queries=int(args.attn_queries),
        query_bank=Path(args.query_bank) if args.query_bank else None,
        allow_random_queries=bool(args.allow_random_queries),
        mode=args.mode,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        wd=float(args.wd),
        seed=int(args.seed),
        eval_every=int(args.eval_every),
        eval_limit=int(args.eval_limit),
        log_every=int(args.log_every),
        capacity_pairs=int(args.capacity_pairs),
        out=Path(args.out),
        label=args.label,
        no_save=bool(args.no_save),
    )
