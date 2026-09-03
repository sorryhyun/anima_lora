"""CJK vocab-pack distillation loop (Phase 2b).

Three modes, in the order you should run them:

``oracle``   G0b — feed the student the teacher's own ids. The loss must be
             identically zero. Validates the loop wiring *and* the trimming
             invariant (a non-pad output must not depend on how many pads
             follow it, which is what lets the cache store trimmed rows).
``capacity`` G0 — overfit a handful of pairs. This is the cheapest possible
             answer to the question the whole phase hangs on: can the ext rows
             *express* the teacher at all? If a 32-pair set cannot be driven to
             ~0, escalate to 2-ii (adapter LoRA) before spending corpus work.
``train``    the real run.

The teacher never runs here — its outputs are cached (see ``cache.py``), and
teacher and student are the same frozen adapter differing only in which
embedding rows their ids reach.

Usage (GPU work goes through the daemon)::

    make daemon-run ARGS="-m scripts.distill_cjk.distill --mode oracle"
    make daemon-run ARGS="-m scripts.distill_cjk.distill --mode capacity"
    make daemon-run ARGS="-m scripts.distill_cjk.distill --loss attn:1.0,span:0.5"
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import weights as anima_weights  # noqa: E402
from scripts.distill_cjk import (  # noqa: E402
    adapter_lora as adapter_lora_mod,
    attn_bank,
    config as cfg_mod,
    ext_table,
)
from scripts.distill_cjk import losses as loss_mod  # noqa: E402
from scripts.distill_cjk.data import CachedPairs, collate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEQ_TOTAL = 512  # what the DiT actually sees (MAX_CROSSATTN_TOKENS)


def _pad_to(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[1] >= length:
        return x[:, :length]
    return F.pad(x, (0, 0, 0, length - x.shape[1]))


def flat_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine over the zero-padded sequences (the probe's metric)."""
    L = max(a.shape[1], b.shape[1])
    a = _pad_to(a, L).reshape(a.shape[0], -1).float()
    b = _pad_to(b, L).reshape(b.shape[0], -1).float()
    return F.cosine_similarity(a, b, dim=-1)


def student_forward(adapter, batch, ids_key="s_ids", mask_key="s_mask"):
    out = adapter(
        source_hidden_states=batch["qwen"],
        target_input_ids=batch[ids_key],
        target_attention_mask=batch[mask_key],
        source_attention_mask=batch["qwen_mask"],
    )
    return loss_mod.zero_pads(out, batch[mask_key])


ALL_LOSSES = {"flat": 1.0, "span": 1.0, "attn": 1.0, "pool": 1.0}


def readout_vec(x, mask, probes) -> torch.Tensor:
    """Concatenated cross-attention readout over the probe bank — ``[B, ...]``.

    The space the DiT actually consumes: permutation-invariant over text slots,
    with the ``512 - N`` zero pads counted as sink mass.
    """
    return torch.cat(
        [
            attn_bank.readout(x, mask, p, seq_total=SEQ_TOTAL).reshape(x.shape[0], -1)
            for p in probes
        ],
        dim=-1,
    ).unsqueeze(1)  # fake seq axis so flat_cos's padding path is a no-op


def discrimination(
    outs: list[torch.Tensor], records: list[dict], seed: int = 0
) -> dict:
    """Collapse guard, **stratified by how far apart the two prompts are**.

    A single cosine threshold over arbitrary prompt pairs is not interpretable.
    The Phase-0 probe's 0.2 was measured between `p1_sign` (a classroom sign)
    and `p2_beach` (a dog on a beach) — maximally distant prompts, where 0.91
    meant the pathway was dead. Applied to two captions that differ only in
    wording, the same threshold punishes correct behavior: they *should* land
    on nearly the same conditioning.

    So two numbers, read in opposite directions:

    ``far``   different images ⇒ the pathology gate. High = prompts collapse.
    ``near``  same image, `tags` vs `tags_alt` (alternate wording of the same
              content) ⇒ high is *expected*. The informative reading is the
              distance from 1.0: that residue is the only evidence that
              wording — and, downstream, the individual glyphs an OCR caption
              carries — reaches conditioning at all. Exactly 1.0 would mean
              per-character identity is invisible, which is the failure mode
              the whole ext-vocab line exists to avoid.

    D6's two registers share their JA text verbatim, so they are excluded from
    `near` — they would be a tautological 1.0, not a measurement.
    """
    if not outs:
        return {}
    base = [r["id"].rsplit("/", 1)[0] for r in records]
    reg = [r["register"] for r in records]
    by_base: dict[str, list[int]] = {}
    for i, b in enumerate(base):
        by_base.setdefault(b, []).append(i)

    near = [
        (i, j)
        for idxs in by_base.values()
        for a, i in enumerate(idxs)
        for j in idxs[a + 1 :]
        if {reg[i], reg[j]} == {"tags", "tags_alt"}
    ]
    rng = random.Random(seed)
    far = []
    for _ in range(4 * len(outs)):
        i, j = rng.randrange(len(outs)), rng.randrange(len(outs))
        if base[i] != base[j]:
            far.append((i, j))
        if len(far) >= 256:
            break

    def cos_over(pairs):
        if not pairs:
            return float("nan")
        vals = []
        for i, j in pairs:
            a, b_ = outs[i].unsqueeze(0), outs[j].unsqueeze(0)
            vals.append(float(flat_cos(a, b_)[0]))
        return sum(vals) / len(vals)

    return {
        "discrimination_far": cos_over(far),
        "discrimination_near": cos_over(near),
        "n_pairs_far": len(far),
        "n_pairs_near": len(near),
    }


def _register_triangle(d: dict[str, list[float]]) -> dict:
    """One register's readout-space triangle: student / teacher / native + R."""
    mean = {k: sum(v) / len(v) for k, v in d.items()}
    span = mean["t"] - mean["n"]
    return {
        "n": len(d["s"]),
        "student": mean["s"],
        "teacher": mean["t"],
        "native": mean["n"],
        "recovery": (mean["s"] - mean["n"]) / span if abs(span) > 1e-6 else 0.0,
    }


@torch.no_grad()
def evaluate(
    adapter, cached, cfg, device, dtype, limit=None, batch_size=32, probes=None
) -> dict:
    """Recovery fraction + the Phase-2c gate quantities, on held-out pairs.

    A bare student-vs-teacher cosine flatters itself: teacher and student share
    the Qwen context, so an *untrained* student already scores well above zero.
    What matters is how much of the addressable gap closed —

        R = (cos(student, en_ref) - cos(native, en_ref))
            / (cos(teacher, en_ref) - cos(native, en_ref))

    where ``native`` is today's unk-wall and ``en_ref`` is the all-EN arm the
    Phase-0 probe measured against.

    **Two recoveries, and the flat one is not the honest one.** ``recovery`` is
    built on the flat position-wise cosine the Phase-0 probe used, and there the
    teacher gets a free ride: its T5 side *is* the reference's token sequence
    (only the Qwen side differs), while the student's tokenization differs by
    construction. So the denominator is inflated and the student is charged for
    a difference the DiT cannot even observe — the same objection that demoted
    L_flat from an objective to a control, applied to the metric.
    ``recovery_attn`` is the same triangle measured in the cross-attention
    readout space: permutation-invariant, sink-aware, and blind to how the two
    arms happened to segment their text. Rank arms on that one.

    Every candidate loss is scored here, not just the one being trained — that
    cross-tab is the whole point of G2: an arm that wins only on its own
    objective has told you nothing.
    """
    if limit is None:
        limit = getattr(cfg, "eval_limit", 256)
    n = len(cached) if limit <= 0 else min(limit, len(cached))
    if n == 0:
        return {}
    acc = {
        k: []
        for k in ("s_ref", "t_ref", "n_ref", "s_t", "s_ref_a", "t_ref_a", "n_ref_a")
    }
    held: dict[str, list[float]] = {}
    per_register: dict[str, list[float]] = {}
    # …and the same decomposition in the readout space. `recovery_attn` is a
    # *mix* statistic (G3), so the aggregate is only comparable between runs
    # that hold out the same register proportions — which a corpus change
    # breaks. The per-register triangle is what survives a re-mix.
    per_register_a: dict[str, dict[str, list[float]]] = {}
    outs: list[torch.Tensor] = []
    outs_a: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        rng_idx = list(range(start, min(start + batch_size, n)))
        if len(rng_idx) < 2:
            break
        b = cached.batch(rng_idx, device, dtype)
        s = student_forward(adapter, b)
        for j, ln in enumerate(b["s_mask"].sum(1).tolist()):
            outs.append(s[j, : int(ln)])
        _, parts = loss_mod.compute(
            {k: v for k, v in ALL_LOSSES.items() if k != "attn" or probes},
            student=s,
            s_mask=b["s_mask"],
            teacher=loss_mod.zero_pads(b["teacher"], b["teacher_mask"]),
            t_mask=b["teacher_mask"],
            spans=b["span_pack"],
            probes=probes,
            seq_total=SEQ_TOTAL,
        )
        for k, v in parts.items():
            held.setdefault(k, []).append(v)
        t = loss_mod.zero_pads(b["teacher"], b["teacher_mask"])
        ref = loss_mod.zero_pads(b["ref"], b["ref_mask"])
        nat = loss_mod.zero_pads(b["native"], b["native_mask"])
        s_ref = flat_cos(s, ref)
        acc["s_ref"] += s_ref.tolist()
        acc["t_ref"] += flat_cos(t, ref).tolist()
        acc["n_ref"] += flat_cos(nat, ref).tolist()
        acc["s_t"] += flat_cos(s, t).tolist()
        for reg, v in zip(b["registers"], s_ref.tolist()):
            per_register.setdefault(reg, []).append(v)

        # …and the same triangle in the readout space (see `recovery_attn`).
        if probes:
            rs, rt, rr, rn = (
                readout_vec(x, m, probes)
                for x, m in (
                    (s, b["s_mask"]),
                    (t, b["teacher_mask"]),
                    (ref, b["ref_mask"]),
                    (nat, b["native_mask"]),
                )
            )
            outs_a.extend(rs[j] for j in range(rs.shape[0]))
            s_ref_a = flat_cos(rs, rr)
            t_ref_a = flat_cos(rt, rr)
            n_ref_a = flat_cos(rn, rr)
            acc["s_ref_a"] += s_ref_a.tolist()
            acc["t_ref_a"] += t_ref_a.tolist()
            acc["n_ref_a"] += n_ref_a.tolist()
            for reg, sv, tv, nv in zip(
                b["registers"], s_ref_a.tolist(), t_ref_a.tolist(), n_ref_a.tolist()
            ):
                d = per_register_a.setdefault(reg, {"s": [], "t": [], "n": []})
                d["s"].append(sv)
                d["t"].append(tv)
                d["n"].append(nv)

    mean = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}
    span = mean["t_ref"] - mean["n_ref"]
    span_a = mean["t_ref_a"] - mean["n_ref_a"]
    disc = discrimination(outs, cached.records[: len(outs)], seed=cfg.seed)
    # The same collapse control *in the readout space* — without it, a high
    # `cos_student_vs_en_attn` is unreadable: if two unrelated prompts also sit
    # at 0.9 there, the readout is simply too lossy to distinguish anything.
    disc_a = discrimination(outs_a, cached.records[: len(outs_a)], seed=cfg.seed)
    disc.update(
        discrimination_far_attn=disc_a.get("discrimination_far", float("nan")),
        discrimination_near_attn=disc_a.get("discrimination_near", float("nan")),
    )
    return {
        "n_eval": n,
        "cos_student_vs_en": mean["s_ref"],
        "cos_teacher_vs_en": mean["t_ref"],
        "cos_native_vs_en": mean["n_ref"],
        "cos_student_vs_teacher": mean["s_t"],
        **disc,
        "recovery": (mean["s_ref"] - mean["n_ref"]) / span if abs(span) > 1e-6 else 0.0,
        "recovery_attn": (
            (mean["s_ref_a"] - mean["n_ref_a"]) / span_a if abs(span_a) > 1e-6 else 0.0
        ),
        "cos_student_vs_en_attn": mean["s_ref_a"],
        "cos_teacher_vs_en_attn": mean["t_ref_a"],
        "cos_native_vs_en_attn": mean["n_ref_a"],
        "cos_student_vs_en_by_register": {
            k: sum(v) / len(v) for k, v in sorted(per_register.items())
        },
        "attn_by_register": {
            k: _register_triangle(v) for k, v in sorted(per_register_a.items())
        },
        "held_out_loss": {k: sum(v) / len(v) for k, v in sorted(held.items())},
    }


def compute_visits(cached_train, pool, n_rows: int) -> torch.Tensor:
    """Ext-row visit histogram over the *training* pool only.

    A row that appears solely in an excluded register gets no supervision, so
    it must not get a free per-row residual either. Pool-dependent but
    arm-independent, so G2 computes it once.
    """
    return ext_table.visit_counts(n_rows, (cached_train.get(i).s_ids for i in pool))


def build_table(cfg, cached_train, pool, ext_init, device, visits=None) -> tuple:
    if visits is None:
        visits = compute_visits(cached_train, pool, ext_init.shape[0])
    tunable = (visits >= cfg.min_visits).nonzero(as_tuple=True)[0]
    if cfg.tunable_rows_from:
        tunable = tunable[tunable >= cfg.tunable_rows_from]
    logger.info(
        "ext rows: %d total, %d visited, %d tunable (>=%d visits, from row %d)",
        ext_init.shape[0],
        int((visits > 0).sum()),
        len(tunable),
        cfg.min_visits,
        cfg.tunable_rows_from,
    )
    table = ext_table.ExtTable(
        ext_init,
        mode=cfg.param,
        rank=cfg.rank,
        tunable_rows=tunable,
        freeze_diag=cfg.freeze_diag,
    ).to(device)
    n_params = sum(p.numel() for p in table.parameters())
    logger.info("trainable parameters: %.2f M (%s)", n_params / 1e6, cfg.param)
    return table, visits


def run_oracle(adapter, cached, cfg, device, dtype) -> dict:
    """G0b: student ids := teacher ids ⇒ the loss must be identically zero."""
    worst = 0.0
    for start in range(0, min(64, len(cached)), cfg.batch_size):
        items = [
            cached.get(i)
            for i in range(start, min(start + cfg.batch_size, len(cached)))
        ]
        if not items:
            break
        b = collate(items, device, dtype)
        out = student_forward(adapter, b, ids_key="t_ids", mask_key="teacher_mask")
        t = loss_mod.zero_pads(b["teacher"], b["teacher_mask"])
        worst = max(worst, float((1.0 - flat_cos(out, t)).max()))
    logger.info("oracle: worst 1-cos = %.3e", worst)
    return {"oracle_worst_1_minus_cos": worst, "passed": worst < 1e-3}


def make_pool(cfg, train_cache) -> list[int]:
    """Training indices after the register filter (and the G0 capacity cut)."""
    pool = list(range(len(train_cache)))
    if cfg.train_registers:
        keep = set(cfg.train_registers)
        pool = [i for i in pool if train_cache.records[i]["register"] in keep]
        logger.info(
            "training on registers %s — %d pairs, %d cached-but-excluded "
            "(still reported at eval)",
            ",".join(cfg.train_registers),
            len(pool),
            len(train_cache) - len(pool),
        )
        if not pool:
            raise SystemExit(
                f"--train_registers {cfg.train_registers} matched nothing in the cache"
            )
    if cfg.mode == "capacity":
        pool = pool[: cfg.capacity_pairs]
        logger.info("G0 capacity probe: overfitting %d pairs", len(pool))
    return pool


def pool_weights(cfg, train_cache, pool: list[int]) -> list[float] | None:
    """Per-index draw weights from --register_sampling (None = uniform).

    A synthetic register can outnumber the real ones 4:1 (names_synth, 2026-08-27);
    drawn uniformly it starves `tags` of gradient visits and the full-JA renders
    regress. Weights rebalance the *draw*, not the pairs — the cache is untouched.
    """
    if not cfg.register_sampling:
        return None
    w = [
        float(cfg.register_sampling.get(train_cache.records[i]["register"], 1.0))
        for i in pool
    ]
    by_reg: dict[str, float] = {}
    for i, wi in zip(pool, w):
        by_reg[train_cache.records[i]["register"]] = (
            by_reg.get(train_cache.records[i]["register"], 0.0) + wi
        )
    tot = sum(by_reg.values())
    logger.info(
        "register sampling share: %s",
        ", ".join(f"{k}={v / tot:.2f}" for k, v in sorted(by_reg.items())),
    )
    return w


def scale_register_spans(cfg, train_cache) -> None:
    """Apply --register_span_scale in place: (register, via) → weight multiplier."""
    if not cfg.register_span_scale:
        return
    n = 0
    for rec in train_cache.records:
        reg = rec["register"]
        for s in rec.get("spans") or []:
            k = float(cfg.register_span_scale.get((reg, str(s[3])), 1.0))
            if k != 1.0:
                s[2] = float(s[2]) * k
                n += 1
    train_cache._span_cache.clear()
    logger.info("register span scale applied to %d spans", n)


def load_context(cfg, device, dtype) -> dict:
    """Everything an arm needs that is identical across arms — loaded once.

    G2 runs a dozen arms; the 4.3 GB cache and the adapter are the expensive
    part, so they are hoisted here and the arms only rebuild the ext table.
    """
    from bench.cjk_adapter import ext_vocab

    train_cache = CachedPairs(cfg.cache_dirs, "train")
    try:
        hold_cache = CachedPairs(cfg.cache_dirs, "holdout")
    except FileNotFoundError:
        logger.warning("no holdout split cached — eval metrics disabled")
        hold_cache = None

    ext_init, mapping = ext_vocab.load_ext_assets(cfg.ext_prefix)
    if cfg.init_pack is not None:
        # Warm start (plan3): the trained rows become the init, so the global
        # map starts at identity *on top of* the best rows-only pack and the
        # arm measures capacity added, not capacity-plus-relearning.
        warm, warm_map = ext_vocab.load_ext_assets(cfg.init_pack)
        if warm.shape != ext_init.shape:
            raise ValueError(
                f"--init_pack rows {tuple(warm.shape)} != ext assets {tuple(ext_init.shape)}"
            )
        logger.info(
            "warm start: ext rows from %s (trained: %s)",
            cfg.init_pack,
            (warm_map.get("training") or {}),
        )
        ext_init = warm
    adapter = anima_weights.load_llm_adapter(cfg.dit, dtype=dtype, device="cpu")
    adapter.to(device).eval()
    # The probe bank is built unconditionally: even an arm that does not train
    # on `attn` is *scored* on it at eval time.
    probes = attn_bank.build_bank(
        cfg.dit,
        cfg.attn_blocks,
        n_query=cfg.attn_queries,
        seed=cfg.seed,
        device=device,
        dtype=torch.float32,
        query_bank=cfg.query_bank,
        allow_random_queries=cfg.allow_random_queries,
    )
    if probes:
        # The readout's common offset, fitted once on the frozen teacher outputs:
        # arm-independent (so the G2 cross-tab stays comparable) and batch-
        # independent. Without it every readout cosine saturates at ~1 — see
        # attn_bank.fit_centers.
        n_center = min(512, len(train_cache))
        with torch.no_grad():
            attn_bank.fit_centers(
                probes,
                (
                    (
                        loss_mod.zero_pads(b["teacher"], b["teacher_mask"]),
                        b["teacher_mask"],
                    )
                    for b in (
                        train_cache.batch(
                            list(range(s, min(s + 64, n_center))),
                            device,
                            torch.float32,
                        )
                        for s in range(0, n_center, 64)
                    )
                ),
                seq_total=SEQ_TOTAL,
            )
    pool = make_pool(cfg, train_cache)
    weights = pool_weights(cfg, train_cache, pool)
    scale_register_spans(cfg, train_cache)
    return {
        "train_cache": train_cache,
        "hold_cache": hold_cache,
        "ext_init": ext_init,
        "mapping": mapping,
        "adapter": adapter,
        "orig_embed": adapter.embed,
        "probes": probes,
        "pool": pool,
        "pool_weights": weights,
        "visits": compute_visits(train_cache, pool, ext_init.shape[0]),
    }


def train_arm(cfg, ctx, device, dtype) -> tuple[dict, object, torch.Tensor]:
    """Train one (parameterization × loss) arm. Returns (metrics, table, visits)."""
    train_cache = ctx["train_cache"]
    hold_cache = ctx["hold_cache"]
    ext_init = ctx["ext_init"]
    adapter = ctx["adapter"]
    probes = ctx["probes"]
    pool = ctx["pool"]
    weights = ctx.get("pool_weights")

    adapter.embed = ctx["orig_embed"]  # restore before re-attaching a fresh table
    table, visits = build_table(
        cfg, train_cache, pool, ext_init, device, visits=ctx["visits"]
    )
    ext_table.attach(adapter, table)

    lora = ctx.get("adapter_lora")
    if lora is not None:
        lora.detach()
    lora = None
    param_groups = [{"params": list(table.parameters()), "lr": cfg.lr}]
    if cfg.adapter_lora > 0:
        lora = adapter_lora_mod.AdapterLoRA(
            adapter, rank=cfg.adapter_lora, targets=cfg.adapter_lora_targets
        ).to(device)
        lora.attach(adapter)
        ctx["adapter_lora"] = lora
        param_groups.append(
            {"params": list(lora.parameters()), "lr": cfg.adapter_lora_lr}
        )
        logger.info(
            "adapter LoRA: r=%d targets=%s  %.2f M params, lr %g (table lr %g)",
            cfg.adapter_lora,
            ",".join(lora.targets),
            lora.n_params() / 1e6,
            cfg.adapter_lora_lr,
            cfg.lr,
        )
    opt = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.wd)
    rng = random.Random(cfg.seed)
    history: list[dict] = []
    first_loss = None
    t0 = time.time()

    # The models here are tiny (a 6-block adapter over ~120 tokens), so the
    # step is dominated by host work — shard reads, padding, H2D copies. Build
    # batch k+1 on a worker thread while the GPU runs batch k.
    def next_batch():
        k = min(cfg.batch_size, len(pool))
        idx = (
            rng.choices(pool, weights=weights, k=k)
            if weights
            else [rng.choice(pool) for _ in range(k)]
        )
        return train_cache.batch(idx, device, dtype)

    loader = ThreadPoolExecutor(max_workers=1)
    prefetch = loader.submit(next_batch)

    # Step-0 baseline. Teacher and student share the Qwen context, so an
    # *untrained* student already scores well above zero — every later number
    # is only meaningful as a delta over this row.
    baseline = (
        evaluate(adapter, hold_cache, cfg, device, dtype, probes=probes)
        if hold_cache
        else {}
    )
    if baseline:
        baseline["step"] = 0
        history.append(baseline)
        logger.info(
            "baseline @0: recovery %.3f  cos(s,en) %.3f  cos(s,t) %.3f  disc_far %.3f",
            baseline["recovery"],
            baseline["cos_student_vs_en"],
            baseline["cos_student_vs_teacher"],
            baseline["discrimination_far"],
        )

    def focus_spans(b: dict) -> dict:
        """--span_focus_from: keep only spans that touch a minted ext row."""
        pk = b.get("span_pack")
        if not cfg.span_focus_from or not pk:
            return b
        floor = ext_table.T5_TABLE_SIZE + cfg.span_focus_from
        minted = (b["s_ids"].reshape(-1)[pk["s_flat"]] >= floor).float()
        hits = torch.zeros(pk["n_spans"], device=minted.device)
        hits.index_add_(0, pk["s_seg"], minted)
        keep = torch.where(
            hits > 0,
            torch.ones_like(hits),
            torch.full_like(hits, cfg.span_focus_bg),
        )
        b["span_pack"] = {**pk, "w": pk["w"] * keep}
        return b

    for step in range(1, cfg.steps + 1):
        b = focus_spans(prefetch.result())
        prefetch = loader.submit(next_batch)
        student = student_forward(adapter, b)
        total, parts = loss_mod.compute(
            cfg.losses,
            student=student,
            s_mask=b["s_mask"],
            teacher=loss_mod.zero_pads(b["teacher"], b["teacher_mask"]),
            t_mask=b["teacher_mask"],
            spans=b["span_pack"],
            probes=probes,
            seq_total=SEQ_TOTAL,
        )
        if cfg.row_anchor and table.has_rows and table.n_tunable_rows():
            ref = table.init[table.tunable_rows].pow(2).sum(-1).clamp_min(1e-8)
            anchor = (table.residual.pow(2).sum(-1) / ref).mean()
            total = total + cfg.row_anchor * anchor
            parts = {**parts, "anchor": float(anchor.detach())}
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if first_loss is None:
            first_loss = float(total.detach())

        if step % cfg.log_every == 0 or step == 1:
            logger.info(
                "step %d/%d loss %.4f  %s",
                step,
                cfg.steps,
                float(total.detach()),
                " ".join(f"{k}={v:.4f}" for k, v in parts.items()),
            )
        if hold_cache and (step % cfg.eval_every == 0 or step == cfg.steps):
            ev = evaluate(adapter, hold_cache, cfg, device, dtype, probes=probes)
            ev.update(
                step=step,
                loss=float(total.detach()),
                **{f"loss_{k}": v for k, v in parts.items()},
            )
            history.append(ev)
            logger.info(
                "  eval @%d: recovery %.3f  cos(s,en) %.3f  cos(s,t) %.3f  disc_far %.3f",
                step,
                ev["recovery"],
                ev["cos_student_vs_en"],
                ev["cos_student_vs_teacher"],
                ev["discrimination_far"],
            )

    metrics = {
        "mode": cfg.mode,
        "param": cfg.param,
        "losses": cfg.losses,
        "trust": cfg.trust,
        "first_loss": first_loss,
        "final_loss": float(total.detach()),
        "final_loss_parts": parts,
        "wall_s": round(time.time() - t0, 1),
        "n_train_pairs": len(pool),
        "n_tunable_rows": table.n_tunable_rows(),
        "rows_visited": int((visits > 0).sum()),
        "rows_total": int(ext_init.shape[0]),
        "history": history,
        "eval": history[-1] if history else {},
    }
    if cfg.mode == "capacity":
        # G0's verdict: can the ext rows express the teacher at all? A residual
        # floor here is a capacity statement, not an optimization one.
        metrics["capacity_floor"] = float(total.detach())
        metrics["capacity_drop"] = (first_loss or 0.0) - float(total.detach())
    if lora is not None:
        # Verify the arm off its weights, not its argv: an untouched LoRA
        # (up still zero) means the second param group never moved.
        up_rms = (
            torch.stack(
                [
                    m.lora_up.weight.detach().float().pow(2).mean()
                    for m in lora.loras.values()
                ]
            )
            .mean()
            .sqrt()
        )
        metrics["adapter_lora"] = {
            "r": lora.rank,
            "alpha": lora.alpha,
            "targets": list(lora.targets),
            "lr": cfg.adapter_lora_lr,
            "n_params": lora.n_params(),
            "up_rms": float(up_rms),
        }
    if cfg.init_pack is not None:
        metrics["init_pack"] = str(cfg.init_pack)
    if table.has_global:
        metrics["global_gain"] = float(torch.exp(table.log_gain))
        metrics["freeze_diag"] = bool(table.freeze_diag)
        metrics["global_diag_rms"] = float(
            torch.exp(table.log_diag).pow(2).mean().sqrt()
        )

    loader.shutdown(wait=False, cancel_futures=True)
    return metrics, table, visits


def save_vocab_pack(cfg, ctx, table, visits, metrics, device) -> None:
    from safetensors.torch import save_file

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"ext_embed": table.materialize().cpu()},
        str(cfg.out.with_suffix(".safetensors")),
    )
    out_map = dict(ctx["mapping"])
    out_map["provenance"] = table.provenance(visits.to(device))
    out_map["training"] = {
        k: metrics[k] for k in ("mode", "param", "losses", "trust", "final_loss")
    }
    if "init_pack" in metrics:
        out_map["training"]["init_pack"] = metrics["init_pack"]
    if metrics.get("freeze_diag"):
        out_map["training"]["freeze_diag"] = True
    lora = ctx.get("adapter_lora")
    if lora is not None:
        sidecar = cfg.out.with_name(cfg.out.name + ".adapter_lora.safetensors")
        lora.save(sidecar)
        out_map["training"]["adapter_lora"] = {
            k: metrics["adapter_lora"][k] for k in ("r", "alpha", "targets", "lr")
        }
        out_map["training"]["adapter_lora"]["file"] = sidecar.name
        metrics["adapter_lora"]["file"] = str(sidecar)
        logger.info("adapter LoRA sidecar → %s", sidecar)
    cfg.out.with_suffix(".json").write_text(
        json.dumps(out_map, ensure_ascii=False), encoding="utf-8"
    )
    metrics["vocab_pack"] = str(cfg.out)
    logger.info("vocab pack → %s.{safetensors,json}", cfg.out)


def main() -> None:
    args = cfg_mod.build_argparser().parse_args()
    cfg = cfg_mod.resolve_config(args)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    run_dir = make_run_dir("cjk_distill", cfg.label or cfg.mode)
    logger.info("run dir: %s", run_dir)

    ctx = load_context(cfg, device, dtype)

    if cfg.mode == "oracle":
        table, _ = build_table(
            cfg,
            ctx["train_cache"],
            ctx["pool"],
            ctx["ext_init"],
            device,
            visits=ctx["visits"],
        )
        ext_table.attach(ctx["adapter"], table)
        metrics = run_oracle(
            ctx["adapter"], ctx["hold_cache"] or ctx["train_cache"], cfg, device, dtype
        )
        write_result(
            run_dir, script=__file__, args=args, label=cfg.label, metrics=metrics
        )
        if not metrics["passed"]:
            raise SystemExit(
                f"G0b FAILED: worst 1-cos {metrics['oracle_worst_1_minus_cos']:.3e} "
                "— the loop or the trimming invariant is broken"
            )
        logger.info("G0b PASSED")
        return

    metrics, table, visits = train_arm(cfg, ctx, device, dtype)
    if not cfg.no_save and cfg.mode == "train":
        save_vocab_pack(cfg, ctx, table, visits, metrics, device)

    write_result(run_dir, script=__file__, args=args, label=cfg.label, metrics=metrics)
    logger.info("result.json → %s", run_dir)


if __name__ == "__main__":
    main()
