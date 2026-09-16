"""Stage the CJK distillation cache: Qwen hidden states + teacher adapter output.

Both arms share the Qwen side and the teacher is fully frozen, so for a fixed
corpus every ``(qwen_hidden(ja), teacher_out)`` pair is invariant across
training runs *and* across parameterizations — compute it once here and the
training loop never loads the text encoder at all.

The cache is also independent of the ext table's *values* (the student's ids
come from the mapping, the teacher's from stock spiece), so re-running a
different ``--param``/``--loss`` arm reuses it. Rebuild only when the corpus,
the tokenizer, or the ext *mapping* changes.

Usage (GPU work goes through the daemon)::

    make daemon-run ARGS="-m scripts.distill_cjk.cache --holdout 500"
"""

from __future__ import annotations

import json
import os
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from library.anima import weights as anima_weights  # noqa: E402
from scripts.distill_cjk import config as cfg_mod  # noqa: E402
from scripts.distill_cjk.data import PairEncoder, load_pairs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SHARD_SIZE = 512


# ---- process-pool staging ---------------------------------------------------
# The per-pair work (padded fast-tokenizer calls + the pure-Python hybrid ext
# encoder + span alignment) is GIL-bound Python, so a worker *thread* never
# overlapped with the main loop in practice (~26 pairs/s, GPU 0↔100%). Each
# worker process builds its own PairEncoder once.
_WORKER_ENCODER: PairEncoder | None = None


def _worker_init(text_encoder_path: str, ext_prefix: str) -> None:
    global _WORKER_ENCODER
    torch.set_num_threads(1)
    _WORKER_ENCODER = PairEncoder(text_encoder_path, Path(ext_prefix))


def _worker_encode(chunk: list[dict], trust: dict[str, float], eval_arms: bool):
    assert _WORKER_ENCODER is not None
    return [_WORKER_ENCODER.encode(p, trust, eval_arms=eval_arms) for p in chunk]


def _encode_split(
    rows: list[dict],
    *,
    encoder: PairEncoder,
    text_encoder,
    adapter,
    trust: dict[str, float],
    out_dir: Path,
    device,
    batch_size: int,
    eval_arms: bool = False,
    workers: int = 1,
) -> dict:
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    shard: dict[str, torch.Tensor] = {}
    shard_idx = 0
    n_span_tokens = 0

    def flush():
        nonlocal shard, shard_idx
        if not shard:
            return
        save_file(shard, str(out_dir / f"shard_{shard_idx:04d}.safetensors"))
        shard = {}
        shard_idx += 1

    # Staging is tokenizer-bound, not GPU-bound: the models are a 0.6B encoder
    # and a 6-block adapter (milliseconds per batch) while each pair pays Qwen +
    # T5 + the pure-Python hybrid ext encoder. Encode on a process pool with a
    # bounded prefetch so chunk k+W is tokenized while the GPU runs chunk k.
    chunks = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    prefetch = max(2, 2 * workers)

    def _run_chunks():
        if workers <= 1:
            for c in chunks:
                yield c, [encoder.encode(p, trust, eval_arms=eval_arms) for p in c]
            return
        init_args = (encoder.text_encoder_path, str(encoder.ext_prefix))
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init, initargs=init_args
        ) as pool:
            futs = []
            it = iter(chunks)
            for c in it:
                futs.append((c, pool.submit(_worker_encode, c, trust, eval_arms)))
                if len(futs) >= prefetch:
                    break
            while futs:
                c, f = futs.pop(0)
                yield c, f.result()
                nxt = next(it, None)
                if nxt is not None:
                    futs.append(
                        (nxt, pool.submit(_worker_encode, nxt, trust, eval_arms))
                    )

    for chunk_idx, (chunk, enc) in enumerate(_run_chunks()):
        start = chunk_idx * batch_size
        qwen_ids = torch.stack([e.qwen_ids for e in enc]).to(device)
        qwen_mask = torch.stack([e.qwen_mask for e in enc]).to(device)
        t_ids = torch.stack([e.t_ids for e in enc]).to(device)
        t_mask = torch.stack([e.t_mask for e in enc]).to(device)

        def run(hidden, src_mask, ids, mask):
            out = adapter(
                source_hidden_states=hidden,
                target_input_ids=ids,
                target_attention_mask=mask,
                source_attention_mask=src_mask,
            )
            # Mirror the inference boundary exactly: pads are zeroed *after*
            # the adapter (library/inference/text.py), so a pad position
            # carries no Qwen signal — it is a plain attention sink downstream.
            out[~mask.bool()] = 0
            return out

        with torch.no_grad():
            hidden = text_encoder(
                input_ids=qwen_ids, attention_mask=qwen_mask
            ).last_hidden_state
            hidden[~qwen_mask.bool()] = 0
            teacher = run(hidden, qwen_mask, t_ids, t_mask)

            ref = native = None
            if eval_arms:
                n_ids = torch.stack([e.n_ids for e in enc]).to(device)
                n_mask = torch.stack([e.n_mask for e in enc]).to(device)
                native = run(hidden, qwen_mask, n_ids, n_mask)
                r_ids = torch.stack([e.ref_qwen_ids for e in enc]).to(device)
                r_mask = torch.stack([e.ref_qwen_mask for e in enc]).to(device)
                r_hidden = text_encoder(
                    input_ids=r_ids, attention_mask=r_mask
                ).last_hidden_state
                r_hidden[~r_mask.bool()] = 0
                ref = run(r_hidden, r_mask, t_ids, t_mask)

        for j, e in enumerate(enc):
            nq = int(e.qwen_mask.sum())
            nt = int(e.t_mask.sum())
            ns = int(e.s_mask.sum())
            key = f"p{len(records):06d}"
            shard[f"{key}.qwen"] = hidden[j, :nq].to(torch.bfloat16).cpu()
            shard[f"{key}.teacher"] = teacher[j, :nt].to(torch.bfloat16).cpu()
            shard[f"{key}.sids"] = e.s_ids[:ns].to(torch.int32)
            shard[f"{key}.tids"] = e.t_ids[:nt].to(torch.int32)
            if eval_arms:
                nn_ = int(e.n_mask.sum())
                shard[f"{key}.ref"] = ref[j, :nt].to(torch.bfloat16).cpu()
                shard[f"{key}.native"] = native[j, :nn_].to(torch.bfloat16).cpu()
            spans = [
                [s.teacher, s.student, s.weight, s.via, s.f1]
                for s in e.spans
                if max(s.teacher) < nt and max(s.student) < ns
            ]
            n_span_tokens += sum(len(s[1]) for s in spans)
            records.append(
                {
                    "id": e.pid,
                    "register": e.register,
                    "shard": f"shard_{shard_idx:04d}.safetensors",
                    "key": key,
                    "n_qwen": nq,
                    "n_teacher": nt,
                    "n_student": ns,
                    "spans": spans,
                }
            )

        if len(shard) >= SHARD_SIZE * (6 if eval_arms else 4):
            flush()
        if start % (batch_size * 20) == 0:
            logger.info("  encoded %d/%d", start + len(chunk), len(rows))
    flush()

    meta = {
        "pairs": records,
        "n_pairs": len(records),
        "n_span_tokens": n_span_tokens,
        "n_with_spans": sum(1 for r in records if r["spans"]),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def main() -> None:
    parser = cfg_mod.build_argparser()
    parser.add_argument(
        "--encode_batch", type=int, default=32, help="pairs per Qwen forward"
    )
    parser.add_argument(
        "--encode_workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 2)),
        help="tokenizer worker processes (1 = serial in-process)",
    )
    args = parser.parse_args()
    cfg = cfg_mod.resolve_config(args)
    if len(cfg.cache_dirs) > 1:
        raise SystemExit(
            "staging writes one dir — pass a single --cache_dir "
            "(the comma list is a distill-time reader feature)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, held = load_pairs(
        cfg.pairs,
        registers=cfg.registers,
        max_pairs=cfg.max_pairs,
        holdout=cfg.holdout,
        seed=cfg.seed,
    )

    encoder = PairEncoder(cfg.text_encoder, cfg.ext_prefix)
    text_encoder, _ = anima_weights.load_qwen3_text_encoder(
        cfg.text_encoder, dtype=torch.bfloat16, device="cpu"
    )
    text_encoder.to(device).eval()
    adapter = anima_weights.load_llm_adapter(
        cfg.dit, dtype=torch.bfloat16, device="cpu"
    )
    adapter.to(device).eval()
    adapter.requires_grad_(False)

    summary = {}
    for split, rows in (("train", train), ("holdout", held)):
        if not rows:
            continue
        logger.info("encoding %s (%d pairs)", split, len(rows))
        meta = _encode_split(
            rows,
            encoder=encoder,
            text_encoder=text_encoder,
            adapter=adapter,
            trust=cfg.trust_weights,
            out_dir=cfg.cache_dir / split,
            device=device,
            batch_size=args.encode_batch,
            eval_arms=(split == "holdout"),
            workers=args.encode_workers,
        )
        summary[split] = {k: v for k, v in meta.items() if k != "pairs"}
        logger.info("  → %s", summary[split])

    (cfg.cache_dir / "cache_info.json").write_text(
        json.dumps(
            {
                "pairs": str(cfg.pairs),
                "ext_prefix": str(cfg.ext_prefix),
                "dit": cfg.dit,
                "text_encoder": cfg.text_encoder,
                "trust": cfg.trust,
                "seed": cfg.seed,
                "splits": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("cache → %s", cfg.cache_dir)


if __name__ == "__main__":
    main()
