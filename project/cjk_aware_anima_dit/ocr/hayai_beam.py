"""Vectorised beam search for the ``hayai-ocr`` VLM (``HayaiModel.generate``).

Upstream's ``num_beams > 1`` path (``modeling_hayai.py`` @ v2.1.5, the static-KV
revision the reader pins; older snapshots without ``cache_seqlens`` fall back to
the model's own ``generate``) runs a
Python loop over ``batch × 2·beams`` candidates per decode step with an
``.item()`` on each — that is the launch/sync wall behind the 25 % GPU
utilisation in ``findings.md`` § hayai throughput. This module keeps the
upstream prefill and decoder step verbatim and replaces only the candidate
bookkeeping with batched tensor ops, so one step costs a fixed handful of
kernel launches regardless of batch size.

Semantics are upstream's, including its two quirks, so reads stay comparable
with the ``pl100k`` tables and the K0 voter matrix:

* candidates are walked in score order and an EOS candidate counts as a
  finished hypothesis only while fewer than ``num_beams`` live continuations
  have been taken (upstream breaks out of the walk at that point);
* the running-sequence copy is **sequential in-place** upstream
  (``running[b, slot] = running[b, parent]`` for slot 0, 1, …), so a slot whose
  parent row was already overwritten this step inherits the *new* content
  of that row. ``faithful=True`` (default) reproduces that; ``faithful=False``
  copies from the pre-step state, which is the intended beam search.

The KV cache is reordered by the true parents either way, as upstream does.
Precision is a separate knob: ``amp=False`` runs fp32 end to end (2026-09-09
decision for the K1 screen — bs 128 fp32), ``amp=True`` is upstream's fp16
autocast.
"""

from __future__ import annotations

import inspect
import sys

import torch
import torch.nn.functional as F


def generate(
    model,
    pixel_values: torch.Tensor,
    pixel_attention_mask: torch.Tensor,
    spatial_shapes: torch.Tensor,
    tokenizer,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    faithful: bool = True,
    amp: str | bool = "bf16",
) -> list[str]:
    if num_beams == 1:
        # Upstream's greedy path is already vectorised.
        return model.generate(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            repetition_penalty=repetition_penalty,
        )
    mod = sys.modules[type(model).__module__]
    compute_batch_2d_mrope_freqs = mod.compute_batch_2d_mrope_freqs
    if (
        "cache_seqlens"
        not in inspect.signature(model.decoder.layers[0].forward).parameters
    ):
        return model.generate(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
        )

    device = pixel_values.device
    b = pixel_values.size(0)
    B = num_beams
    L = max_new_tokens
    dec = model.decoder
    V = dec.vocab_size

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    # ``amp``: "bf16" (default), "fp16" (upstream's autocast), or falsy for
    # fp32. Under fp16 ~1.6 % of strings flip with batch composition.
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp or "", None)
    NEG = -1e9

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype or torch.float32,
        enabled=(amp_dtype is not None and device.type == "cuda"),
    ):
        # ---- prefill: verbatim upstream --------------------------------------
        vision_outputs = model.vision_encoder(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        visual_features = vision_outputs.last_hidden_state
        m_vision = visual_features.size(1)

        bos_tokens = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
        x = torch.cat(
            [dec.projector(visual_features), dec.token_embeddings(bos_tokens)], dim=1
        )

        d_axis = dec.layers[0].attn.d_head // 2
        freqs = 1.0 / (
            10000.0 ** (torch.arange(0, d_axis, 2, device=device).float() / d_axis)
        )
        t_text_all = torch.arange(L + 1, device=device, dtype=torch.float32)
        text_freqs_1d_all = torch.outer(t_text_all, freqs)
        text_freqs_all = torch.cat([text_freqs_1d_all, text_freqs_1d_all], dim=-1)
        cos_text_all = torch.cos(text_freqs_all)
        sin_text_all = torch.sin(text_freqs_all)

        mask = dec.generate_block_causal_mask(m_vision, 1, device)
        cos_batch, sin_batch = compute_batch_2d_mrope_freqs(
            spatial_shapes, m_vision, 1, d_head=64, device=device
        )
        # Static KV cache, preallocated as upstream does (dtype of the prefill
        # input, seq = vision + BOS + max_new_tokens).
        max_seq_len = m_vision + L + 1
        kv_cache: dict = {}
        for i, layer in enumerate(dec.layers):
            shape = (b, layer.attn.h_kv, max_seq_len, layer.attn.d_head)
            kv_cache[i] = (
                torch.zeros(shape, dtype=x.dtype, device=device),
                torch.zeros(shape, dtype=x.dtype, device=device),
            )
        cache_seqlens = 0
        for i, layer in enumerate(dec.layers):
            x = layer(
                x,
                mask=mask,
                cos_sin=(cos_batch, sin_batch),
                kv_cache=kv_cache,
                layer_idx=i,
                cache_seqlens=cache_seqlens,
            )
        cache_seqlens += x.size(1)
        next_token_logits = dec.output_head(dec.final_norm(x[:, -1:]))[:, -1, :].clone()

        # ---- beam search -----------------------------------------------------
        for layer_idx in list(kv_cache.keys()):
            k, v = kv_cache[layer_idx]
            kv_cache[layer_idx] = (
                k.repeat_interleave(B, dim=0),
                v.repeat_interleave(B, dim=0),
            )

        if repetition_penalty != 1.0:
            penalty = torch.full_like(next_token_logits, 1.0)
            penalty[:, bos_id] = repetition_penalty
            next_token_logits = torch.where(
                next_token_logits < 0,
                next_token_logits * penalty,
                next_token_logits / penalty,
            )
        log_probs = F.log_softmax(next_token_logits, dim=-1)

        running = torch.full((b, B, L + 1), pad_id, dtype=torch.long, device=device)
        running[:, :, 0] = bos_id
        topk_scores, topk_tokens = torch.topk(log_probs, B, dim=-1)
        beam_scores = topk_scores.float()
        running[:, :, 1] = topk_tokens

        # Best finished hypothesis per row (upstream keeps a list and takes the
        # max at the end; only the running max is needed) + how many finished.
        best_score = torch.full((b,), float("-inf"), device=device)
        best_seq = torch.full((b, L + 1), pad_id, dtype=torch.long, device=device)
        done_cnt = torch.zeros((b,), dtype=torch.long, device=device)

        eos0 = topk_tokens == eos_id
        s0 = torch.where(eos0, beam_scores, torch.full_like(beam_scores, float("-inf")))
        m0, _ = s0.max(dim=1)
        upd = m0 > best_score
        best_score = torch.where(upd, m0, best_score)  # seq = [eos] → decodes to ""
        done_cnt += eos0.sum(dim=1)
        beam_scores = torch.where(eos0, torch.full_like(beam_scores, NEG), beam_scores)
        cur = topk_tokens.reshape(b * B, 1)

        pos2B = torch.arange(2 * B, device=device)
        slotB = torch.arange(B, device=device)

        # Per-row early stop. Upstream stops when *every* row has B finished
        # hypotheses, so a row that finished early keeps collecting longer
        # hypotheses while its batch-mates run (and its read depends on
        # them). Here a row leaves the batch the moment it has B, and the
        # rows still running are compacted so late steps cost only what is
        # still decoding — reads median 5 tokens, a bs-64 batch used to run
        # 41 steps for its longest row.
        result = torch.full((b, L + 1), pad_id, dtype=torch.long, device=device)
        act = torch.arange(b, device=device)  # active row → original row

        def settle(rows: torch.Tensor, mask: torch.Tensor) -> None:
            """Final pick for the active rows in ``mask`` (upstream's rule)."""
            has_done = done_cnt[mask] > 0
            run_best = running[mask][
                torch.arange(int(mask.sum()), device=device),
                beam_scores[mask].argmax(dim=1),
            ]
            result[rows[mask]] = torch.where(
                has_done.unsqueeze(1), best_seq[mask], run_best
            )

        for step in range(1, L):
            if early_stopping:
                fin = done_cnt >= B
                if bool(fin.any()):
                    settle(act, fin)
                    keep = ~fin
                    if not bool(keep.any()):
                        act = act[:0]
                        break
                    act = act[keep]
                    beam_scores, running = beam_scores[keep], running[keep]
                    done_cnt, best_score, best_seq = (
                        done_cnt[keep],
                        best_score[keep],
                        best_seq[keep],
                    )
                    cur = cur.view(-1, B, 1)[keep].reshape(-1, 1)
                    rows = keep.nonzero(as_tuple=True)[0]
                    idx = (rows.unsqueeze(1) * B + slotB).reshape(-1)
                    for layer_idx in kv_cache:
                        k, v = kv_cache[layer_idx]
                        kv_cache[layer_idx] = (
                            k.index_select(0, idx),
                            v.index_select(0, idx),
                        )
            n = act.numel()
            ar_n = torch.arange(n, device=device)

            x_step = dec.token_embeddings(cur)
            cos_step = cos_text_all[step].unsqueeze(0).expand(n * B, 1, -1)
            sin_step = sin_text_all[step].unsqueeze(0).expand(n * B, 1, -1)
            for i, layer in enumerate(dec.layers):
                x_step = layer(
                    x_step,
                    mask=None,
                    cos_sin=(cos_step, sin_step),
                    kv_cache=kv_cache,
                    layer_idx=i,
                    cache_seqlens=cache_seqlens,
                )
            cache_seqlens += 1
            logits_step = dec.output_head(dec.final_norm(x_step))[:, -1, :].clone()

            if repetition_penalty != 1.0:
                flat_seqs = running.view(n * B, -1)[:, : step + 1]
                penalty = torch.ones_like(logits_step).scatter(
                    1, flat_seqs, repetition_penalty
                )
                logits_step = torch.where(
                    logits_step < 0, logits_step * penalty, logits_step / penalty
                )

            lp = F.log_softmax(logits_step, dim=-1).view(n, B, V)
            next_scores = (beam_scores.unsqueeze(-1) + lp).view(n, B * V)
            cand_scores, cand_idx = torch.topk(
                next_scores, 2 * B, dim=-1, largest=True, sorted=True
            )
            parent = cand_idx // V  # (n, 2B)
            tok = cand_idx % V
            is_eos = tok == eos_id
            live = ~is_eos

            # Upstream walks candidates in order and stops once B live ones are
            # taken; an EOS candidate is recorded only if reached before that.
            live_before = torch.cumsum(live.long(), dim=1) - live.long()
            reached = live_before < B
            eos_done = is_eos & reached
            norm = cand_scores / float((step + 1) ** length_penalty)
            sc = torch.where(eos_done, norm, torch.full_like(norm, float("-inf")))
            m, mi = sc.max(dim=1)
            upd = m > best_score
            par_best = parent.gather(1, mi.unsqueeze(1)).squeeze(1)
            seq_best = running[ar_n, par_best].clone()
            seq_best[:, step + 1 :] = pad_id  # eos at step+1 is stripped anyway
            best_seq = torch.where(upd.unsqueeze(1), seq_best, best_seq)
            best_score = torch.where(upd, m, best_score)
            done_cnt = done_cnt + eos_done.sum(dim=1)

            # The first B live candidates, in score order; short rows pad with
            # a dead beam whose parent is beam 0 (upstream's filler).
            key = pos2B.expand(n, -1).masked_fill(is_eos, 2 * B)
            order = torch.sort(key, dim=1).indices[:, :B]
            n_live = live.sum(dim=1, keepdim=True)
            valid = slotB.unsqueeze(0) < n_live  # (n, B)
            new_scores = cand_scores.gather(1, order).masked_fill(~valid, NEG)
            new_tok = tok.gather(1, order).masked_fill(~valid, pad_id)
            new_par = parent.gather(1, order).masked_fill(~valid, 0)

            if faithful:
                # Sequential in-place copy, slot by slot, as upstream does.
                for s_ in range(B):
                    running[:, s_, : step + 1] = running[
                        ar_n, new_par[:, s_], : step + 1
                    ]
                    running[:, s_, step + 1] = new_tok[:, s_]
            else:
                running = running.gather(
                    1, new_par.unsqueeze(-1).expand(-1, -1, L + 1)
                ).clone()
                running[:, :, step + 1] = new_tok

            beam_scores = new_scores
            cur = new_tok.reshape(n * B, 1)
            reorder_idx = (ar_n.unsqueeze(1) * B + new_par).reshape(-1)
            for layer_idx in kv_cache:
                k, v = kv_cache[layer_idx]
                k.copy_(k.index_select(0, reorder_idx))
                v.copy_(v.index_select(0, reorder_idx))

        if act.numel():
            # Rows that hit max_new_tokens (or early_stopping=False).
            settle(act, torch.ones(act.numel(), dtype=torch.bool, device=device))

    final = result[:, 1:].tolist()
    return [
        tokenizer.decode(
            [t for t in seq if t not in (eos_id, pad_id)], skip_special_tokens=True
        )
        for seq in final
    ]
