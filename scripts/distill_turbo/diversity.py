"""DAVE-derived same-prompt diversity validation for the turbo distillation loop.

The in-loop ``div`` loss only measures how close the student's first step lands to the
teacher's K-step anchor — it says nothing about whether the student's own
same-prompt samples have collapsed across seeds (the canonical DMD failure).

This pass measures that directly, using the DAVE decomposition validated in
``bench/dave/`` (README + ``probe_dc_convergence.py``): per Transformer block,
the per-channel spatial mean (the **DC**) is the seed-shared conditioning, while
the residual ``h − μ`` (the **AC**) carries the seed-specific structure. So:

  * **AC sim** (cross-seed cosine of the AC residual) is the headline diversity
    signal — **lower = more diverse**. It rising over training = mode collapse.
  * **DC sim** is a reference: it should stay high (~conditioning lock); a drop
    means the run is destabilising the conditioning, not diversifying.

We fix ONE held-out conditioning and roll the student's N-step Euler grid (the
same ``forward_fn`` / ``set_student_step`` / ``student_sigmas`` the training loop
uses) across N seeds under ``no_grad``, hooking every block. Layout-robust: the
DC = mean over all non-(batch,channel) axes, so it reads correctly whether the
block emits eager 5D ``(B,T,H,W,D)`` or the compiled native-flatten
``(B,1,seq,1,D)`` (channel D is last in both). The AC residual is pooled to a
fixed token count so seeds are comparable regardless of resolution.

Read-only: forward hooks only, removed in a ``finally``; the student view is
restored to step 0 on exit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DiversityMetrics:
    ac_sim: float  # cross-seed cosine of the AC residual (block feats); LOWER = more diverse
    dc_sim: float  # cross-seed cosine of the DC (block feats); reference, ~conditioning lock
    gap: float  # dc_sim − ac_sim; the DAVE separation (high = AC carries the diversity)
    xpred_ac_sim: (
        float  # same decomposition on the final x_pred latent; LOWER = more diverse
    )
    fm_mse: float  # flow-matching reconstruction MSE on the held-out sample; the
    # fidelity half of the fidelity↔diversity tradeoff view. CAVEAT: FM val loss
    # has NOT tracked sample quality on Anima (CMMD replaced it as the quality
    # signal) — read it as a divergence/sanity number, not a quality score.


def _mean_pairwise_cos(X: torch.Tensor) -> float:
    """Mean off-diagonal cosine similarity across the N rows (NaN if <2 rows)."""
    if X is None or X.shape[0] < 2:
        return float("nan")
    Xn = F.normalize(X.float(), dim=1, eps=1e-8)
    G = Xn @ Xn.t()
    n = X.shape[0]
    return float((G.sum() - torch.diagonal(G).sum()) / (n * (n - 1)))


def _dc_ac(h: torch.Tensor, ac_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a block output into DC (B,D) and a fixed-length AC vector (B, ac_tokens·D).

    Layout-robust: flattens every axis between batch (dim 0) and channel (last
    dim) into a token axis, so eager ``(B,T,H,W,D)`` and compiled native-flatten
    ``(B,1,seq,1,D)`` both reduce to ``(B, N_tok, D)``. The AC token axis is
    adaptive-avg-pooled to ``ac_tokens`` so seeds at any resolution compare.
    """
    B, D = h.shape[0], h.shape[-1]
    tok = h.reshape(B, -1, D)
    dc = tok.mean(dim=1)
    ac = tok - dc.unsqueeze(1)
    ac = F.adaptive_avg_pool1d(ac.transpose(1, 2), ac_tokens).transpose(1, 2)
    return dc, ac.reshape(B, -1)


class _CrossSeedProbe:
    """Per-block post-hook capture keyed to (step, block) of the student rollout.

    The student rollout is cond-only (no CFG), so every forward is a capture
    forward — the forward counter just ticks when block 0 fires.
    """

    def __init__(self, num_blocks: int, num_steps: int, ac_tokens: int):
        self.L = num_blocks
        self.S = num_steps
        self.ac_tokens = ac_tokens
        self.capturing = False
        self.forward_idx = -1
        self._dc: list[dict] = []  # per seed: {(step, block): (D,)}
        self._ac: list[dict] = []

    def start_seed(self) -> None:
        self.forward_idx = -1
        self.capturing = True
        self._dc.append({})
        self._ac.append({})

    def stop(self) -> None:
        self.capturing = False

    def make_hook(self, bidx: int):
        def hook(_module, _inputs, output):
            if bidx == 0:
                self.forward_idx += 1
            if not self.capturing:
                return
            step = self.forward_idx
            if step >= self.S:
                return
            h = output.detach().float()
            dc, ac = _dc_ac(h, self.ac_tokens)
            self._dc[-1][(step, bidx)] = dc.mean(0).cpu()  # B==1, mean is defensive
            self._ac[-1][(step, bidx)] = ac.mean(0).cpu()

        return hook

    def _stack(self, store: list[dict], s: int, bl: int) -> torch.Tensor | None:
        rows = [d[(s, bl)] for d in store if (s, bl) in d]
        return torch.stack(rows, 0) if rows else None

    def summary(self) -> tuple[float, float]:
        """Mean cross-seed AC sim and DC sim over all captured (step, block)."""
        ac_vals, dc_vals = [], []
        for s in range(self.S):
            for bl in range(self.L):
                ac = _mean_pairwise_cos(self._stack(self._ac, s, bl))
                dc = _mean_pairwise_cos(self._stack(self._dc, s, bl))
                if ac == ac:  # not NaN
                    ac_vals.append(ac)
                if dc == dc:
                    dc_vals.append(dc)
        ac_mean = sum(ac_vals) / len(ac_vals) if ac_vals else float("nan")
        dc_mean = sum(dc_vals) / len(dc_vals) if dc_vals else float("nan")
        return ac_mean, dc_mean


@torch.no_grad()
def run_diversity_validation(
    *,
    model,
    forward_fn,
    set_student_step,
    student_sigmas: list[float],
    crossattn_emb: torch.Tensor,
    latent_shape: tuple[int, ...],
    num_seeds: int,
    seed0: int,
    device,
    dtype,
    clean_latent: torch.Tensor | None = None,
    ac_tokens: int = 16,
) -> DiversityMetrics:
    """Roll the student across ``num_seeds`` seeds on a fixed conditioning and
    measure cross-seed DC/AC convergence of the block features (+ the x_pred),
    plus (if ``clean_latent`` is given) a flow-matching reconstruction MSE.

    ``forward_fn`` / ``set_student_step`` / ``student_sigmas`` are the live
    training-loop primitives; ``crossattn_emb`` is one held-out sample's cached
    text features (shape ``(1, seq, D)``); ``latent_shape`` is ``(1, C, H, W)``;
    ``clean_latent`` is the held-out sample's cached clean latent ``(1, C, H, W)``.
    """
    student_steps = len(student_sigmas) - 1
    probe = _CrossSeedProbe(len(model.blocks), student_steps, ac_tokens)
    handles = [
        blk.register_forward_hook(probe.make_hook(i))
        for i, blk in enumerate(model.blocks)
    ]
    x_preds: list[torch.Tensor] = []
    try:
        for k in range(num_seeds):
            gen = torch.Generator(device=device).manual_seed(seed0 + k)
            x = torch.randn(latent_shape, device=device, dtype=dtype, generator=gen)
            # Match the training loop's cudagraph hygiene (it marks once per outer
            # step); a fresh mark per seed keeps compiled/cudagraph forwards from
            # reusing a stale static output buffer. No-op when compile is off.
            torch.compiler.cudagraph_mark_step_begin()
            probe.start_seed()
            for i in range(student_steps):
                s_i, s_next = student_sigmas[i], student_sigmas[i + 1]
                t_b = torch.full((latent_shape[0],), s_i, device=device, dtype=dtype)
                set_student_step(i)
                v = forward_fn("student", x, t_b, crossattn_emb, no_grad=True).squeeze(
                    2
                )
                x = x - (s_i - s_next) * v
            probe.stop()
            x_preds.append(x.detach())
    finally:
        for h in handles:
            h.remove()
        set_student_step(0)

    ac_sim, dc_sim = probe.summary()

    # Same DAVE split on the final latent (no hooks): AC of x_pred across seeds.
    X = torch.cat([_dc_ac(x.float(), ac_tokens)[1] for x in x_preds], 0)
    xpred_ac_sim = _mean_pairwise_cos(X)

    # Flow-matching reconstruction MSE (fidelity half). Rectified flow's velocity
    # target is constant along the straight path (v* = ε − x0, σ-independent), so
    # evaluate the student at its OWN grid sigmas through each matching per-step
    # head. Fixed disjoint seed so the number is comparable across passes.
    fm_mse = float("nan")
    if clean_latent is not None:
        x0 = clean_latent.to(device, dtype=dtype)
        gen = torch.Generator(device=device).manual_seed(seed0 + 9973)
        eps = torch.randn(x0.shape, device=device, dtype=dtype, generator=gen)
        v_target = eps.float() - x0.float()
        errs = []
        torch.compiler.cudagraph_mark_step_begin()
        for i in range(student_steps):
            s_i = student_sigmas[i]
            x_t = ((1.0 - s_i) * x0.float() + s_i * eps.float()).to(dtype)
            t_b = torch.full((x0.shape[0],), s_i, device=device, dtype=dtype)
            set_student_step(i)
            v = forward_fn("student", x_t, t_b, crossattn_emb, no_grad=True).squeeze(2)
            errs.append(float(F.mse_loss(v.float(), v_target)))
        set_student_step(0)
        fm_mse = sum(errs) / len(errs)

    return DiversityMetrics(
        ac_sim=ac_sim,
        dc_sim=dc_sim,
        gap=dc_sim - ac_sim,
        xpred_ac_sim=xpred_ac_sim,
        fm_mse=fm_mse,
    )
