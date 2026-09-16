"""Soft-rank caption-discrimination auxiliary (_archive/proposals/turbo_caption_ranking.md).

DP-DMD distillation degrades the teacher's caption discriminability, concentrated
at the student's step-0 state (σ → 1) — at σ=1.0 a quarter of anchors lose to a
*random* caption (``bench/turbo/caption_ranking_probe.py``). This module is the
training-time fix: a bounded listwise *rank* loss that pushes the matched caption
to explain the DP-DMD diversity anchor better than mismatched ones.

``soft_rank`` is vendored from ``softtorch`` (a-paulus/softtorch, Apache-2.0,
arXiv:2603.08824) — the same relaxation ``bench/soft_tokens_contrastive/_softrank.py``
uses for the offline gradient probe (kept self-contained here). Soft-rank is
bounded by construction (``L → 0`` at rank → 1) and its gradient aligns with the
margin gradient (``cos ≈ 0.86``).
"""

from __future__ import annotations

import torch


def soft_rank(scores: torch.Tensor, tau: float = 0.1, dim: int = -1) -> torch.Tensor:
    """Differentiable rank of each entry (0 = best/highest score).

    ``rank_i = Σ_{j≠i} sigmoid((s_j − s_i) / tau)`` — the soft count of entries
    that out-score ``i``. Higher ``scores`` ⇒ lower (better) rank. Shape-preserving
    along ``dim``. As ``tau → 0`` this converges to the integer rank; larger ``tau``
    spreads gradient onto near-ties (the near-miss credit a detached argsort can't
    deliver). Bounded in ``[0, n−1]`` along ``dim``.
    """
    s = scores.transpose(dim, -1)
    diff = s.unsqueeze(-1) - s.unsqueeze(-2)  # diff[..., i, j] = s_i − s_j
    # σ((s_j − s_i)/τ) summed over j (the diagonal j=i contributes σ(0)=0.5 and is
    # subtracted off so a unique max gets rank 0).
    r = torch.sigmoid(-diff / max(float(tau), 1e-6)).sum(dim=-1) - 0.5
    return r.transpose(dim, -1)


def caption_rank_loss(
    v_pos: torch.Tensor,
    v_negs: list[torch.Tensor],
    v_target: torch.Tensor,
    *,
    tau: float = 0.1,
) -> torch.Tensor:
    """Soft-rank loss on the matched caption's position against ``k`` negatives.

    The reward at a shared ``(ε, t)`` is ``−‖v(c) − v_target‖²`` (the validated
    relative FM-ranking compass — everything that pollutes the absolute FM-MSE
    cancels across candidate captions; ``agsm_reward_premise_holds.md``). The
    matched caption (``v_pos``) should out-score every mismatched one (``v_negs``);
    we soft-rank the candidate scores and minimize the matched caption's soft rank.

    Returns the batch-mean soft rank of the matched caption — ``0`` when it wins
    every pair (self-annealing at the win, so the term needs no negative-push
    bound). Gradient flows through ``v_pos`` only; ``v_negs`` are expected detached
    (computed under ``no_grad``), so the objective is "make the matched caption
    explain the anchor better", never "push the others away".

    Shapes: ``v_pos`` / each ``v_negs[j]`` / ``v_target`` are ``[B, C, H, W]``;
    ``v_target`` is the (detached, fp32) diversity anchor. Reward reduces over all
    non-batch dims.
    """

    def neg_mse(v: torch.Tensor) -> torch.Tensor:
        return -((v.float() - v_target) ** 2).flatten(1).mean(dim=1)

    # [B, k+1] — column 0 is the matched caption, 1..k the negatives.
    r = torch.stack([neg_mse(v_pos), *(neg_mse(v) for v in v_negs)], dim=1)
    return soft_rank(r, tau=tau)[:, 0].mean()


class CaptionNegativePool:
    """Cross-step ring buffer of past caption embeddings → shuffled negatives.

    Sourcing negatives from a pool of *previous* steps' captions (rather than
    sibling samples in the current batch) is what lets the soft-rank term fire at
    ``batch_size=1`` — where within-batch negatives don't exist. Text embeddings
    are max-padded (the padding invariant → constant seq length), so captions from
    any step or image bucket stack cleanly into a single negative tensor.

    Entries are detached on-device clones (no per-step D2H sync); the buffer is
    bounded to ``capacity``. The current step's captions are added *after*
    drawing, so an anchor never draws its own caption as a negative (barring a
    genuine earlier repeat).
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._buf: list[torch.Tensor] = []  # each [seq, dim], detached, on device

    def add(self, crossattn_emb: torch.Tensor) -> None:
        """Push every sample's caption (drop the batch dim) into the ring."""
        for c in crossattn_emb.detach():
            self._buf.append(c.clone())
            if len(self._buf) > self.capacity:
                self._buf.pop(0)

    def ready(self, k: int) -> bool:
        """True once the pool holds at least ``k`` captions to draw."""
        return len(self._buf) >= k

    def draw(self, k: int, batch: int) -> list[torch.Tensor]:
        """``k`` negative caption tensors, each ``[batch, seq, dim]``.

        Each negative slot samples one caption per batch row independently (with
        replacement — duplicates are harmless to the rank, and a large pool makes
        them rare). The CPU RNG draw introduces no GPU sync (mirrors the loop's
        ``grad_step='random'`` index draw).
        """
        n = len(self._buf)
        idx = torch.randint(n, (k, batch))  # CPU generator → global torch seed
        return [
            torch.stack([self._buf[int(idx[j, b])] for b in range(batch)], dim=0)
            for j in range(k)
        ]
