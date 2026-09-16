# Per-layer time-indexed soft tokens — SoftREPA-style parameterization
# (Lee et al., NeurIPS 2025, arXiv:2503.08250). DiT is frozen; trains a bank of
# K continuous vectors per (layer, t-bucket), spliced into crossattn_emb at each
# block independently. Anima's DiT is cross-attention (not MM-DiT), so
# crossattn_emb doesn't evolve through blocks — no strip/re-prepend needed.
#
# Splice (two modes, see SoftTokensNetwork.splice_position):
#   front_of_padding (shipped default) — scatter the K tokens at
#     [seqlens[i], seqlens[i]+K), immediately after the real caption (the direct
#     analogue of SoftREPA's text-sequence concat in Anima's max-padded layout).
#   end_of_sequence (fallback) — overwrite the K zero-padding tail slots, keeping
#     crossattn_emb shape static for torch.compile. Gets attention mass only
#     because Anima's deep-padding slots are cross-attention sinks (text-encoder
#     padding invariant); an Anima-specific compile shortcut.
#
# Inference: library/inference/generation.py + networks/spectrum.py call
# append_postfix(..., timesteps=t) per CFG branch before each forward. Spectrum
# cached steps skip the blocks, so soft tokens no-op there (composes with --spectrum).

import os
from typing import Optional

import torch
import torch.nn as nn

from library.log import setup_logging
from library.training.method_adapter import (
    ForwardArtifacts,
    MethodAdapter,
    StepCtx,
)
from networks.methods.base import AdapterNetworkBase

import logging

setup_logging()
logger = logging.getLogger(__name__)

# Anima cached crossattn_emb dimension (Qwen3 hidden size, post LLM-adapter).
DEFAULT_EMBED_DIM = 1024

# Contrastive negative-sourcing modes (_archive/proposals/soft_tokens_contrastive.md):
# ``shuffled`` = unrelated cached-TE negative; ``jaccard`` = shuffled but logit
# down-weighted by caption tag-overlap; ``hard`` = same-artist/different-character
# sibling (falls back to shuffled for orphan artists); ``hard_backoff`` = tiered
# same-artist → same-copyright → shuffled (copyright tier rescues hard's fallback).
CONTRASTIVE_MODES = ("shuffled", "jaccard", "hard", "hard_backoff")

# Contrastive objective, sharing the extra-forward plumbing:
#   ``infonce``  — SoftREPA InfoNCE over cached-TE negatives (default).
#   ``softrank`` — differentiable listwise rank of the matched caption among the
#                  candidates (``L = softtorch.rank(r)[matched] − 1``). Gradient
#                  flows *through* the ordering yet stays bounded by the SoftSort
#                  relaxation (unlike InfoNCE's unbounded negative push). Needs k
#                  live negative forwards only (_archive/proposals/soft_tokens_softrank.md).
CONTRASTIVE_OBJECTIVES = ("infonce", "softrank")


# GPU-resident, pure-torch softtorch.rank methods suitable for the per-step
# training loop. ``ot`` (Sinkhorn) and ``fast_soft_sort`` (Numba JIT, CPU-only →
# GPU↔CPU sync every step) are deliberately excluded — fine offline, perf sinks
# in the loop.
SOFTRANK_METHODS = ("neuralsort", "softsort")


def _soft_rank(
    scores: torch.Tensor, softness: float, method: str = "neuralsort"
) -> torch.Tensor:
    """softtorch differentiable rank along the last dim (1 = best score).

    Thin wrapper over ``softtorch.rank`` (a-paulus/softtorch, Apache-2.0,
    arXiv:2603.08824). ``descending=True`` so a higher reward earns a lower
    (better) rank; ``standardize=True`` (softtorch's default) z-scores the
    candidate axis so the tiny FM-error reward gaps are well-conditioned — that
    standardization is *why* the objective needs ``contrastive_k ≥ 2`` (with a
    single negative the 2-value z-score is gap-independent regardless of method;
    enforced in ``SoftTokensNetwork.__init__``). Returns ranks in ``[1, n]``.

    ``method`` ∈ ``SOFTRANK_METHODS``:
      - ``neuralsort`` (default) — NeuralSort relaxation (Grover et al. 2019).
        Smooth gradient *through ties*; the right pick for the near-miss regime
        where the matched caption and the best negative are neck-and-neck.
      - ``softsort`` — SoftSort column projection (Prillo & Eisenschlos, ICML
        2020). Faster but its gradient goes ~flat at exact ties (softtorch's
        documented discontinuity-at-non-unique-values).

    softtorch is imported lazily so its numba / POT import cost is only paid when
    the softrank objective actually fires.
    """
    import softtorch as st

    return st.rank(
        scores,
        dim=-1,
        method=method,
        descending=True,
        softness=max(float(softness), 1e-6),
        standardize=True,
    )


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    num_tokens = network_dim if network_dim is not None else 4
    embed_dim = int(kwargs.get("embed_dim", DEFAULT_EMBED_DIM))
    n_layers = int(kwargs.get("n_layers", 10))
    n_t_buckets = int(kwargs.get("n_t_buckets", 100))
    init_std = float(kwargs.get("init_std", 0.02))
    splice_position = kwargs.get("splice_position", "end_of_sequence")
    contrastive_weight = float(kwargs.get("contrastive_weight", 0.0))
    contrastive_k = int(kwargs.get("contrastive_k", 1))
    contrastive_negative_mode = str(kwargs.get("contrastive_negative_mode", "shuffled"))
    contrastive_tau = float(kwargs.get("contrastive_tau", 0.5))
    contrastive_warmup_ratio = float(kwargs.get("contrastive_warmup_ratio", 0.1))
    contrastive_jaccard_alpha = float(kwargs.get("contrastive_jaccard_alpha", 1.0))
    contrastive_every_n = int(kwargs.get("contrastive_every_n", 1))
    contrastive_objective = str(kwargs.get("contrastive_objective", "infonce"))
    softrank_softness = float(kwargs.get("softrank_softness", 0.1))
    softrank_method = str(kwargs.get("softrank_method", "neuralsort"))
    # ``dual_bank`` (ψ⁺/ψ⁻ token banks). ``agsm_dual_bank`` is a deprecated
    # alias so older configs/snapshots don't silently drop the flag.
    _dual_bank = kwargs.get("dual_bank", kwargs.get("agsm_dual_bank", "false"))
    dual_bank = str(_dual_bank).lower() in ("true", "1", "yes")
    network = SoftTokensNetwork(
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_t_buckets=n_t_buckets,
        init_std=init_std,
        splice_position=splice_position,
        contrastive_weight=contrastive_weight,
        contrastive_k=contrastive_k,
        contrastive_negative_mode=contrastive_negative_mode,
        contrastive_tau=contrastive_tau,
        contrastive_warmup_ratio=contrastive_warmup_ratio,
        contrastive_jaccard_alpha=contrastive_jaccard_alpha,
        contrastive_every_n=contrastive_every_n,
        contrastive_objective=contrastive_objective,
        softrank_softness=softrank_softness,
        softrank_method=softrank_method,
        dual_bank=dual_bank,
        multiplier=multiplier,
    )
    return network


def create_network_from_weights(
    multiplier,
    file,
    ae,
    text_encoders,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    tokens = weights_sd.get("tokens")
    t_offsets = weights_sd.get("t_offsets.weight")
    if tokens is None or t_offsets is None:
        raise ValueError(
            f"soft_tokens weight file must contain 'tokens' and 't_offsets.weight' "
            f"(got keys: {list(weights_sd.keys())[:8]})"
        )
    # Dual-bank checkpoints carry a leading branch axis on ``tokens``
    # (n_banks, n_layers, K, D); single-bank stay 3D. Inference uses ONLY ψ⁺
    # (branch 0) — injecting ψ⁻ at inference over-suppresses detail — so build a
    # single-bank net and let ``load_weights`` slice the branch-0 weights. Resume
    # (for_inference=False) keeps both banks.
    if tokens.dim() == 4:
        file_n_banks, n_layers, num_tokens, embed_dim = tokens.shape
    else:
        file_n_banks = 1
        n_layers, num_tokens, embed_dim = tokens.shape
    build_dual = (file_n_banks > 1) and (not for_inference)
    n_t_buckets = t_offsets.shape[0]
    # Splice position is a runtime knob, not learned — read from metadata, CLI
    # kwargs win for post-hoc overrides.
    metadata_splice = None
    if file is not None and os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            meta = f.metadata() or {}
            metadata_splice = meta.get("ss_splice_position")
    splice_position = kwargs.get(
        "splice_position", metadata_splice or "end_of_sequence"
    )
    network = SoftTokensNetwork(
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_t_buckets=n_t_buckets,
        init_std=0.0,  # weights are loaded; init_std doesn't matter
        splice_position=splice_position,
        # Contrastive is training-only (extra forwards, no learned params) — off
        # on the inference path.
        contrastive_weight=0.0,
        dual_bank=build_dual,
        multiplier=multiplier,
    )
    return network, weights_sd


class SoftTokensNetwork(AdapterNetworkBase):
    """Per-layer time-indexed soft tokens.

    Parameters:
      - tokens: (n_layers, K, D) — base per-layer tokens, small-std init.
      - t_offsets: Embedding(n_t_buckets, n_layers * D) — per-(t_bucket, layer)
        broadcast offset (one D-vector applied to every token in the layer).
        Zero-init so step 0 reproduces the un-time-conditioned base tokens.

    Param count: n_layers·K·D + n_t_buckets·n_layers·D
    With defaults (n_layers=10, K=4, D=1024, n_t_buckets=100): 40k + 1.0M ≈ 1.05M.
    """

    network_module = "networks.methods.soft_tokens"
    network_spec = "soft_tokens"

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        n_layers: int = 10,
        n_t_buckets: int = 100,
        init_std: float = 0.02,
        splice_position: str = "end_of_sequence",
        contrastive_weight: float = 0.0,
        contrastive_k: int = 1,
        contrastive_negative_mode: str = "shuffled",
        contrastive_tau: float = 0.5,
        contrastive_warmup_ratio: float = 0.1,
        contrastive_jaccard_alpha: float = 1.0,
        contrastive_every_n: int = 1,
        contrastive_objective: str = "infonce",
        softrank_softness: float = 0.1,
        softrank_method: str = "neuralsort",
        dual_bank: bool = False,
        multiplier: float = 1.0,
    ):
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        # Upper-bound check against actual block count happens in apply_to().
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if n_t_buckets <= 0:
            raise ValueError(f"n_t_buckets must be positive, got {n_t_buckets}")
        if splice_position not in ("front_of_padding", "end_of_sequence"):
            raise ValueError(
                f"splice_position must be 'front_of_padding' or 'end_of_sequence', "
                f"got {splice_position!r}"
            )
        if contrastive_negative_mode not in CONTRASTIVE_MODES:
            raise ValueError(
                f"contrastive_negative_mode must be one of {CONTRASTIVE_MODES}, "
                f"got {contrastive_negative_mode!r}"
            )
        if contrastive_objective not in CONTRASTIVE_OBJECTIVES:
            raise ValueError(
                f"contrastive_objective must be one of {CONTRASTIVE_OBJECTIVES}, "
                f"got {contrastive_objective!r}"
            )
        if contrastive_weight > 0.0:
            if contrastive_k < 1:
                raise ValueError(f"contrastive_k must be >= 1, got {contrastive_k}")
            if contrastive_objective == "infonce" and contrastive_tau <= 0.0:
                raise ValueError(
                    f"contrastive_tau must be positive, got {contrastive_tau}"
                )
            if contrastive_objective == "softrank":
                if contrastive_k < 2:
                    raise ValueError(
                        "softrank requires contrastive_k >= 2: softtorch's softsort "
                        "standardizes the candidate axis, which is gap-independent "
                        f"(degenerate) with a single negative (got "
                        f"contrastive_k={contrastive_k})"
                    )
                if softrank_softness <= 0.0:
                    raise ValueError(
                        f"softrank_softness must be positive, got {softrank_softness}"
                    )
                if softrank_method not in SOFTRANK_METHODS:
                    raise ValueError(
                        f"softrank_method must be one of {SOFTRANK_METHODS}, "
                        f"got {softrank_method!r}"
                    )

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_t_buckets = n_t_buckets
        self.splice_position = splice_position
        # Stored for AdapterNetworkBase API compatibility (set_multiplier), but
        # NOT applied to the injected tokens: soft tokens splice in at full
        # magnitude regardless of multiplier (see _make_block_hook). Unlike
        # LoRA, there is no `* multiplier` scale on the delta — the tokens enter
        # cross-attention directly, so `set_multiplier(0)` does NOT disable them.
        # Gate the whole adapter at the trainer/loader level to turn them off.
        self.multiplier = multiplier

        # Contrastive objective (training-only; extra forwards live in
        # ``SoftTokensMethodAdapter``). ``_contrastive_target_weight`` gates
        # composer activation; ``_contrastive_weight`` is the live warmup-held
        # value the loss handler multiplies by.
        self._contrastive_target_weight = float(contrastive_weight)
        self._contrastive_warmup_ratio = float(contrastive_warmup_ratio)
        self._contrastive_tau = float(contrastive_tau)
        self.contrastive_k = int(contrastive_k)
        self.contrastive_negative_mode = str(contrastive_negative_mode)
        self.contrastive_jaccard_alpha = float(contrastive_jaccard_alpha)
        self._contrastive_weight = (
            0.0
            if self._contrastive_warmup_ratio > 0.0
            else self._contrastive_target_weight
        )
        # Cadence: run the contrastive negative forwards only every Nth optimizer
        # step. NOT auto-scaled — effective strength is (weight × 1/N); co-tune
        # ``contrastive_weight`` to hold it constant. 1 = every step (default).
        # ``_contrastive_fire_this_step`` is recomputed each step by
        # ``step_contrastive_warmup`` on the optimizer-step index.
        self._contrastive_every_n = max(1, int(contrastive_every_n))
        self._contrastive_fire_this_step = True

        # ``contrastive_objective`` selects the loss math in the adapter
        # (infonce | softrank).
        self.contrastive_objective = str(contrastive_objective)
        # softrank: SoftSort softness (softtorch's temperature). NOT contrastive_tau
        # — softtorch's standardize=True puts the candidate axis at unit scale, so
        # the right softness is ~0.1 (its default).
        self._softrank_softness = float(softrank_softness)
        # SoftSort vs NeuralSort relaxation for softrank. neuralsort (default) has
        # a smooth gradient through ties (the near-miss regime); softsort goes flat
        # at exact ties. Both pure-torch / GPU-resident.
        self._softrank_method = str(softrank_method)
        # Dual token banks: branch 0 = ψ⁺ (spliced on the anchor + positive value
        # passes + ALL inference), branch 1 = ψ⁻ (negative value passes, training-
        # only) — lets the negative push refine ψ⁻ without spending generative
        # fidelity on the kept bank. Single bank (n_banks=1) keeps the 3D on-disk
        # format. Inference is ψ⁺-only (injecting ψ⁻ over-suppresses detail).
        self.dual_bank = bool(dual_bank)
        self.n_banks = 2 if self.dual_bank else 1

        # Single bank: (n_layers, K, D) — unchanged 3D shape + on-disk format.
        # Dual bank: (2, n_layers, K, D) with a leading branch axis (ψ⁺, ψ⁻),
        # independently initialized so the two guidance regions don't start tied.
        if self.n_banks == 1:
            self.tokens = nn.Parameter(
                torch.randn(n_layers, num_tokens, embed_dim) * init_std
            )
        else:
            self.tokens = nn.Parameter(
                torch.randn(self.n_banks, n_layers, num_tokens, embed_dim) * init_std
            )
        # Per-(bucket, [branch], layer) D-vector offset, broadcast across the
        # K-token axis at lookup (one D-vector per layer per bucket, not K).
        # Zero-init = identity perturbation at step 0. Dual stacks the branches in
        # the column axis (bank-major: [ψ⁺ layers | ψ⁻ layers]) so the ψ⁺ slice is
        # the first n_layers·D columns.
        self.t_offsets = nn.Embedding(n_t_buckets, self.n_banks * n_layers * embed_dim)
        nn.init.zeros_(self.t_offsets.weight)

        # Step-scoped state set by append_postfix() per forward and consumed by
        # the per-block hooks. Plain attributes (recreated each step).
        # _step_seqlens only used for front_of_padding splice.
        self._step_layer_tokens: Optional[torch.Tensor] = None  # (n_layers, B, K, D)
        self._step_seqlens: Optional[torch.Tensor] = None  # (B,) int

        # Kept so apply_to() could un-monkey-patch later (unused but cheap).
        self._block_refs: list[nn.Module] = []
        self._original_forwards: list = []

        n_token_params = self.tokens.numel()
        n_offset_params = self.t_offsets.weight.numel()
        bank_note = (
            "dual bank (ψ⁺/ψ⁻, ψ⁺-only at inference)"
            if self.n_banks == 2
            else "single bank"
        )
        logger.info(
            f"SoftTokensNetwork: {n_layers} layers × {num_tokens} tokens × dim {embed_dim}, "
            f"{n_t_buckets} t-buckets, splice={splice_position}, {bank_note} → "
            f"{n_token_params + n_offset_params} params "
            f"({n_token_params} base + {n_offset_params} t-offset)"
        )

    # Sentinel for train.py's ``hasattr(network, "append_postfix")`` branch —
    # makes it call append_postfix(..., timesteps=...) per step, which we use
    # only to compute the step-scoped tokens (crossattn_emb passes through; the
    # splice happens in the per-block hooks below).
    @property
    def num_postfix_tokens(self) -> int:
        return self.num_tokens

    def _bucketize(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map sigma in [0, 1] (Anima convention) to integer buckets [0, n_t_buckets).

        Outside-range values are clamped, so callers don't need to pre-clamp.
        """
        t = timesteps.detach().float().flatten()
        idx = torch.floor(t * self.n_t_buckets).long()
        return idx.clamp(min=0, max=self.n_t_buckets - 1)

    def append_postfix(
        self,
        crossattn_emb: torch.Tensor,
        crossattn_seqlens: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-step layer tokens and cache them for the block hooks.

        Returns crossattn_emb unchanged — splice happens per-block in the hooks
        installed by ``apply_to()``. We just piggyback on train.py's existing
        per-step trainer hook to receive timesteps.
        """
        if timesteps is None:
            raise ValueError(
                "soft_tokens requires timesteps (per-step) — train.py and the "
                "inference loop (library/inference/generation.py, "
                "networks/spectrum.py) both pass this per CFG branch each step"
            )
        # Anchor / inference forward always splices ψ⁺ (branch 0). The negative
        # ψ⁻ branch is only ever spliced from the training adapter's value passes.
        self._set_step_tokens(timesteps, crossattn_seqlens, branch=0)
        return crossattn_emb

    def _layer_tokens_from(
        self,
        tokens: torch.Tensor,
        t_offsets_weight: torch.Tensor,
        timesteps: torch.Tensor,
        branch: int = 0,
    ) -> torch.Tensor:
        """Per-step (n_layers, B, K, D) tokens for a given (tokens, t_offsets)
        pair. Factored out of ``append_postfix`` so the ψ⁺ and ψ⁻ branches run
        the same math.

        ``branch`` selects the bank (0=ψ⁺, 1=ψ⁻) for dual-bank nets; ignored
        (always ψ⁺) when ``n_banks == 1`` so the single-bank path is unchanged."""
        B = int(timesteps.detach().flatten().shape[0])
        bucket_idx = self._bucketize(timesteps)  # (B,)
        offsets_full = nn.functional.embedding(bucket_idx, t_offsets_weight)
        if self.n_banks == 1:
            # (B, n_layers * D) → (B, n_layers, D)
            offsets = offsets_full.view(B, self.n_layers, self.embed_dim)
            base = tokens  # (n_layers, K, D)
        else:
            # (B, n_banks * n_layers * D) → pick this branch → (B, n_layers, D)
            offsets = offsets_full.view(B, self.n_banks, self.n_layers, self.embed_dim)[
                :, branch
            ]
            base = tokens[branch]  # (n_layers, K, D)
        # (n_layers, K, D) → (1, n_layers, K, D), broadcast over batch + over K.
        per_step = base.unsqueeze(0) + offsets.unsqueeze(2)  # (B, n_layers, K, D)
        # Transpose to (n_layers, B, K, D) for cheap per-layer indexing in the
        # block hook closure.
        return per_step.transpose(0, 1).contiguous()

    def _set_step_tokens(
        self,
        timesteps: torch.Tensor,
        crossattn_seqlens: Optional[torch.Tensor],
        branch: int = 0,
    ) -> None:
        """Compute + cache the per-step layer tokens read by the block hooks.

        ``branch`` selects ψ⁺ (0) / ψ⁻ (1) for dual-bank nets.
        """
        self._step_layer_tokens = self._layer_tokens_from(
            self.tokens, self.t_offsets.weight, timesteps, branch=branch
        )
        # front_of_padding needs per-sample seqlens at hook time; end_of_sequence
        # ignores them. Cache regardless so the hook doesn't have to know which
        # mode is active (the splice branch reads or skips).
        self._step_seqlens = (
            crossattn_seqlens.detach().to(torch.long)
            if crossattn_seqlens is not None
            else None
        )

    def _make_block_hook(self, layer_idx: int, org_forward):
        """Closure that splices layer_idx's tokens into crossattn_emb tail.

        Block.forward signature (from library/anima/models.py::Block.forward):
          forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, attn_params,
                  rope_cos_sin=None, adaln_lora_B_T_3D=None)
        """
        K = self.num_tokens
        splice_position = self.splice_position
        net = self

        def hook(
            x_B_T_H_W_D,
            emb_B_T_D,
            crossattn_emb,
            attn_params,
            *args,
            **kwargs,
        ):
            step_tokens = net._step_layer_tokens
            if step_tokens is not None:
                # multiplier not applied — see the self.multiplier note in __init__.
                layer_tok = step_tokens[layer_idx].to(
                    dtype=crossattn_emb.dtype, device=crossattn_emb.device
                )
                S = crossattn_emb.shape[1]
                if S < K:
                    raise RuntimeError(
                        f"crossattn_emb seqlen {S} < num_tokens {K}; cannot splice"
                    )
                if splice_position == "end_of_sequence":
                    # Overwrite the K tail slots; torch.cat preserves autograd.
                    crossattn_emb = torch.cat(
                        [crossattn_emb[:, : S - K, :], layer_tok], dim=1
                    )
                else:  # front_of_padding
                    # Place K tokens at [seqlens[i], seqlens[i]+K) per sample —
                    # displaces the strongest sinks. scatter() preserves grad.
                    seqlens = net._step_seqlens
                    if seqlens is None:
                        raise RuntimeError(
                            "front_of_padding splice requires crossattn_seqlens; "
                            "trainer must pass it to append_postfix()"
                        )
                    offsets = seqlens.to(crossattn_emb.device).unsqueeze(
                        1
                    ) + torch.arange(K, device=crossattn_emb.device)  # (B, K)
                    D = crossattn_emb.shape[-1]
                    idx = offsets.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
                    crossattn_emb = crossattn_emb.scatter(1, idx, layer_tok)
            return org_forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                attn_params,
                *args,
                **kwargs,
            )

        return hook

    def apply_to(
        self,
        text_encoders,
        unet,
        apply_text_encoder=True,
        apply_unet=True,
    ):
        """Monkey-patch the first n_layers DiT blocks with the splice hook."""
        blocks = getattr(unet, "blocks", None)
        if blocks is None:
            raise RuntimeError("unet has no .blocks attribute — not an Anima DiT?")
        if len(blocks) < self.n_layers:
            raise RuntimeError(
                f"unet has {len(blocks)} blocks but n_layers={self.n_layers}"
            )
        self._block_refs = []
        self._original_forwards = []
        for k in range(self.n_layers):
            block = blocks[k]
            org_forward = block.forward
            block.forward = self._make_block_hook(k, org_forward)
            self._block_refs.append(block)
            self._original_forwards.append(org_forward)
        logger.info(
            f"soft_tokens: monkey-patched first {self.n_layers} of {len(blocks)} "
            f"DiT blocks ({self.splice_position} splice, K={self.num_tokens})"
        )

    # ── Standard adapter API ────────────────────────────────────────────

    def get_trainable_params(self):
        return [self.tokens, self.t_offsets.weight]

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        del text_encoder_lr
        lr = unet_lr or default_lr
        params = [{"params": self.get_trainable_params(), "lr": lr}]
        descriptions = ["soft_tokens(tokens+t_offsets)"]
        return params, descriptions

    def state_dict_for_save(self, dtype):
        return {
            "tokens": self.tokens.detach().clone().cpu().to(dtype),
            "t_offsets.weight": self.t_offsets.weight.detach().clone().cpu().to(dtype),
        }

    def metadata_fields(self) -> dict[str, str]:
        return {
            "ss_num_tokens": str(self.num_tokens),
            "ss_embed_dim": str(self.embed_dim),
            "ss_n_layers": str(self.n_layers),
            "ss_n_t_buckets": str(self.n_t_buckets),
            "ss_splice_position": self.splice_position,
            # Contrastive objective is training-only (no learned params), but
            # stamp the config for run provenance.
            "ss_contrastive_weight": str(self._contrastive_target_weight),
            "ss_contrastive_k": str(self.contrastive_k),
            "ss_contrastive_negative_mode": self.contrastive_negative_mode,
            "ss_contrastive_tau": str(self._contrastive_tau),
            "ss_contrastive_warmup_ratio": str(self._contrastive_warmup_ratio),
            "ss_contrastive_every_n": str(self._contrastive_every_n),
            "ss_contrastive_objective": self.contrastive_objective,
            "ss_softrank_softness": str(self._softrank_softness),
            "ss_softrank_method": self._softrank_method,
            # Dual-bank provenance. The load side keys off the on-disk tensor rank
            # (4D ⇒ dual), so ``ss_n_banks`` / ``ss_dual_bank`` are informational.
            "ss_dual_bank": "true" if self.dual_bank else "false",
            "ss_n_banks": str(self.n_banks),
        }

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        if "tokens" not in weights_sd or "t_offsets.weight" not in weights_sd:
            raise ValueError(
                f"Missing required keys in soft_tokens checkpoint "
                f"(got: {list(weights_sd.keys())[:8]})"
            )
        tok, toff = self._select_load_weights(
            weights_sd["tokens"], weights_sd["t_offsets.weight"]
        )
        self.tokens.data.copy_(tok)
        self.t_offsets.weight.data.copy_(toff)
        logger.info(
            f"Loaded soft_tokens weights: tokens={tuple(self.tokens.shape)}, "
            f"t_offsets={tuple(self.t_offsets.weight.shape)} (n_banks={self.n_banks})"
        )

    def _select_load_weights(
        self, file_tokens: torch.Tensor, file_t_offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconcile a checkpoint's bank layout with this net's ``n_banks``.

        - file 4D + this net single (inference of a dual checkpoint): slice the
          ψ⁺ branch (index 0) — Appendix H keeps ψ⁻ training-only.
        - file 3D + this net single: pass through.
        - file 4D + this net dual (resume): pass through.
        - file 3D + this net dual: a single-bank checkpoint can't seed both
          branches unambiguously → hard error.
        """
        file_dual = file_tokens.dim() == 4
        if self.n_banks == 1:
            if not file_dual:
                return file_tokens, file_t_offsets
            # ψ⁺ slice: tokens[0]; t_offsets first n_layers·D columns (bank-major).
            n_cols = self.n_layers * self.embed_dim
            return file_tokens[0], file_t_offsets[:, :n_cols]
        # dual net
        if not file_dual:
            raise ValueError(
                "dual_bank=True but the checkpoint has a single (3D) bank — "
                "a single-bank checkpoint cannot initialize both ψ⁺/ψ⁻ branches; "
                "train dual from scratch or resume a dual checkpoint."
            )
        return file_tokens, file_t_offsets

    def metrics(self, ctx) -> dict[str, float]:
        """TensorBoard bank-state collapse/divergence diagnostics.

        ``tokens_mean_cos`` ~0 = orthogonal (good), ~1 = slot collapse;
        ``tokens_mean_norm`` blowing up = magnitude diverging;
        ``offset_mean_norm`` ~0 = t-offset buckets not training (check LR).
        """
        del ctx
        out: dict[str, float] = {}

        # Diagnostics are on ψ⁺ (branch 0), the bank kept at inference. For dual
        # banks also surface the ψ⁻ magnitude so a diverging negative branch is
        # visible. Single bank: tokens is already (L, K, D).
        tokens_all = self.tokens.detach()
        if self.n_banks > 1:
            out["soft_tokens/tokens_neg_mean_norm"] = float(
                tokens_all[1].flatten(1).norm(dim=-1).mean().item()
            )
            tokens_psi = tokens_all[0]  # (L, K, D)
        else:
            tokens_psi = tokens_all  # (L, K, D)

        # Batched over the layer axis → 3 host syncs for the whole bank.
        if self.num_tokens >= 2 and self.n_layers > 0:
            tokens = tokens_psi  # (L, K, D)
            K = tokens.shape[1]
            iu = torch.triu_indices(K, K, offset=1, device=tokens.device)
            # Mean pairwise cos over all (layer, pair) — equal pair count per
            # layer so this equals the mean of per-layer means.
            zn = torch.nn.functional.normalize(tokens, dim=-1, eps=1e-8)
            gram = zn @ zn.transpose(1, 2)  # (L, K, K)
            out["soft_tokens/tokens_mean_cos"] = float(
                gram[:, iu[0], iu[1]].mean().item()
            )
            # Squared pairwise distances ‖a‖²+‖b‖²−2a·b; clamp subtraction round-off.
            sq = tokens.pow(2).sum(-1)  # (L, K)
            d_sq = (
                sq.unsqueeze(2)
                + sq.unsqueeze(1)
                - 2.0 * (tokens @ tokens.transpose(1, 2))
            ).clamp_min(0.0)  # (L, K, K)
            out["soft_tokens/tokens_min_d_sq"] = float(
                d_sq[:, iu[0], iu[1]].min().item()
            )
            out["soft_tokens/tokens_mean_norm"] = float(
                tokens.flatten(1).norm(dim=-1).mean().item()
            )
        # t_offsets.weight is (n_t_buckets, n_banks·n_layers·D); report the ψ⁺
        # (branch 0) per-(layer) offset norm, matching the single-bank metric.
        offsets_psi = self.t_offsets.weight.detach().view(
            self.n_t_buckets, self.n_banks, self.n_layers, self.embed_dim
        )[:, 0]
        out["soft_tokens/offset_mean_norm"] = float(
            offsets_psi.permute(1, 0, 2).flatten(1).norm(dim=-1).mean().item()
        )
        return out

    def step_contrastive_warmup(
        self, global_step: int, max_train_steps: int, accum: int = 1
    ) -> None:
        """Activate the contrastive objective past its warmup window and decide
        whether the negatives fire this step.

        Warmup: ``_contrastive_weight`` holds at 0 for the first
        ``_contrastive_warmup_ratio`` of steps, then flips to the target (no-op
        when target is 0) — lets plain FM shape a non-degenerate bank before the
        contrast pulls on it.

        Cadence: ``global_step`` is the micro-batch counter; the firing decision
        is taken on the optimizer-step index (``global_step // accum``) so every
        micro-batch in an accumulation window agrees (else partial, accum-coupled
        contrastive grads).
        """
        every_n = int(self._contrastive_every_n)
        accum = max(1, int(accum))
        optimizer_step = int(global_step) // accum
        self._contrastive_fire_this_step = every_n <= 1 or optimizer_step % every_n == 0

        target = float(self._contrastive_target_weight)
        ratio = float(self._contrastive_warmup_ratio)
        if target <= 0.0:
            return
        if ratio <= 0.0 or max_train_steps <= 0:
            self._contrastive_weight = target
            return
        warmup_steps = int(max_train_steps * ratio)
        self._contrastive_weight = 0.0 if global_step < warmup_steps else target

    def contrastive_loss(
        self,
        v_pos: torch.Tensor,
        v_neg: torch.Tensor,
        v_target: torch.Tensor,
        neg_penalty: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """SoftREPA-style InfoNCE over cached-TE negatives (B=1-adapted).

        Each forward shares ``(x_t, ε, t)`` and spliced soft tokens; only
        ``crossattn_emb`` differs (matched vs. mismatched text). Logit per
        forward = negative FM error against the shared velocity target, scaled
        by τ: ``ℓ = -‖v - v_target‖²/τ``, ``L = -log(exp(ℓ_pos)/Σexp(ℓ))``.
        Gradient flows to the soft tokens to make the matched text explain the
        anchor's latent better than mismatched text.

        ``neg_penalty`` (jaccard mode's ``α·s``, shape ``(B, k)``) is subtracted
        from each negative logit — a tag-overlapping negative pulls less
        gradient. ``None`` ⇒ plain InfoNCE (shuffled/hard).

        Shapes: v_pos/v_target ``(B,C,H,W)``, v_neg ``(B,k,C,H,W)``.
        Returns ``(loss_scalar, diagnostics)`` — accuracy + mean pos-neg logit gap.
        """
        logit_pos, logits_neg = self._velocities_to_logits(
            v_pos, v_neg, v_target, neg_penalty
        )
        loss = self._infonce_from_logits(logit_pos, logits_neg)
        return loss, self._contrastive_diagnostics(logit_pos, logits_neg)

    def _velocities_to_logits(
        self,
        v_pos: torch.Tensor,
        v_neg: torch.Tensor,
        v_target: torch.Tensor,
        neg_penalty: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample FM-error logits for the InfoNCE softmax.

        Differentiable in every velocity arg, so the caller can ``.detach()``
        whichever branch it drops from the graph — how the grad-cache path
        splits ∂L/∂v_pos and ∂L/∂v_neg without duplicating the τ/penalty math.
        Returns ``(logit_pos (B,), logits_neg (B, k))``.
        """
        tau = float(self._contrastive_tau)
        vp = v_pos.float()
        vt = v_target.float()
        vn = v_neg.float()
        B = vp.shape[0]
        k = vn.shape[1]
        # Per-sample mean-squared FM error → logit. Reduce over all non-batch
        # dims so τ has a stable scale across resolutions.
        pos_err = (vp - vt).pow(2).reshape(B, -1).mean(dim=1)  # (B,)
        logit_pos = -pos_err / tau  # (B,)
        vt_exp = vt.unsqueeze(1)  # (B, 1, C, H, W)
        neg_err = (vn - vt_exp).pow(2).reshape(B, k, -1).mean(dim=2)  # (B, k)
        logits_neg = -neg_err / tau  # (B, k)
        if neg_penalty is not None:
            # jaccard mode: down-weight tag-overlapping negatives.
            logits_neg = logits_neg - neg_penalty.to(logits_neg.dtype)
        return logit_pos, logits_neg

    @staticmethod
    def _infonce_from_logits(
        logit_pos: torch.Tensor, logits_neg: torch.Tensor
    ) -> torch.Tensor:
        # InfoNCE: -log softmax of the positive over {pos, neg_1..k}.
        all_logits = torch.cat([logit_pos.unsqueeze(1), logits_neg], dim=1)  # (B, 1+k)
        return (-logit_pos + torch.logsumexp(all_logits, dim=1)).mean()

    @staticmethod
    def _contrastive_diagnostics(
        logit_pos: torch.Tensor, logits_neg: torch.Tensor
    ) -> dict[str, float]:
        with torch.no_grad():
            acc = (logit_pos.unsqueeze(1) > logits_neg).all(dim=1).float().mean()
            gap = (logit_pos - logits_neg.mean(dim=1)).mean()
        return {
            "contrastive_acc": float(acc.item()),
            "contrastive_logit_gap": float(gap.item()),
        }

    # ── soft-rank listwise objective (_archive/proposals/soft_tokens_softrank.md) ──

    def _candidate_rewards(
        self,
        v_pos: torch.Tensor,
        v_neg: torch.Tensor,
        v_target: torch.Tensor,
    ) -> torch.Tensor:
        """Per-candidate FM reward ``r_j = -mean((v_j - v_target)²)`` over C·H·W.

        Returns ``(B, m=k+1)``, index 0 = matched, 1..k = mismatched (higher =
        better). Differentiable in both v_pos/v_neg so the caller can
        ``.detach()`` whichever branch to split the grad-cache.
        """
        vp = v_pos.float()
        vt = v_target.float()
        vn = v_neg.float()
        B = vp.shape[0]
        k = vn.shape[1]
        r_pos = -(vp - vt).pow(2).reshape(B, -1).mean(dim=1)  # (B,)
        r_neg = -(vn - vt.unsqueeze(1)).pow(2).reshape(B, k, -1).mean(dim=2)  # (B, k)
        return torch.cat([r_pos.unsqueeze(1), r_neg], dim=1)  # (B, m)

    def softrank_loss(
        self,
        v_pos: torch.Tensor,
        v_neg: torch.Tensor,
        v_target: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable SoftSort rank of the matched caption, pushed toward best.

        ``L = softtorch.rank(r, softsort)[matched] - 1`` (rank in [1,m], 1=win).
        Gradient flows through every ``r_j`` — through the soft ordering, no
        detach on the ranking (``v_target`` is the only constant). Bounded by
        the SoftSort relaxation; self-annealing (``L -> 0`` as matched wins).

        The grad-cache split rides the same disjoint-partials trick as InfoNCE:
        detach ``v_neg`` here for the anchor's ∂L/∂v+, and ``v_pos`` in the
        deferred pass for ∂L/∂v- (the two partials are exact).
        """
        r = self._candidate_rewards(v_pos, v_neg, v_target)  # (B, m)
        ranks = _soft_rank(r, self._softrank_softness, self._softrank_method)
        return (ranks[:, 0] - 1.0).mean()  # ranks ∈ [1, m]

    def softrank_diagnostics(
        self,
        v_pos: torch.Tensor,
        v_neg: torch.Tensor,
        v_target: torch.Tensor,
    ) -> dict[str, float]:
        """Acc / reward-gap (shared with InfoNCE) + the mean matched soft-rank.

        ``softrank_matched_rank`` is the headline signal: -> 1 as the matched
        caption wins outright, -> m when it loses. Inputs should be detached."""
        with torch.no_grad():
            r = self._candidate_rewards(v_pos, v_neg, v_target)  # (B, m)
            diag = self._contrastive_diagnostics(r[:, 0], r[:, 1:])
            matched_rank = _soft_rank(
                r, self._softrank_softness, self._softrank_method
            )[:, 0].mean()
        diag["softrank_matched_rank"] = float(matched_rank.item())
        return diag


class SoftTokensMethodAdapter(MethodAdapter):
    """Runs the contrastive negative forwards for soft tokens.

    Each negative is one extra DiT forward sharing the anchor's ``(x_t, ε, t)``
    and spliced tokens, swapping only ``crossattn_emb`` — the ``extra_forwards``
    contract. Two objectives share the plumbing (k forwards each), selected by
    ``network.contrastive_objective``: ``infonce`` (SoftREPA InfoNCE) or
    ``softrank`` (differentiable listwise rank).

    Wiring: ``prime_for_forward`` stashes ``batch["neg_crossattn_emb"]`` (train
    only); ``extra_forwards`` returns the scalar under
    ``"soft_tokens_contrastive"``; ``after_backward`` replays the deferred
    ∂L/∂v_neg. Negatives absent outside training → forwards skipped.
    """

    name = "soft_tokens_contrastive"

    def __init__(self) -> None:
        self._neg_crossattn: Optional[torch.Tensor] = None
        self._neg_jaccard: Optional[torch.Tensor] = None
        self._last_metrics: dict[str, float] = {}
        # GOTCHA: under block swap the negative backward can't share the
        # anchor's forward/backward cycle, so it's deferred to after_backward.
        # None when no replay is queued.
        self._pending_gradcache: Optional[dict] = None

    def prime_for_forward(
        self, ctx: StepCtx, batch, latents: torch.Tensor, *, is_train: bool
    ) -> None:
        del ctx, latents
        if not is_train or not isinstance(batch, dict):
            self._neg_crossattn = None
            self._neg_jaccard = None
            return
        self._neg_crossattn = batch.get("neg_crossattn_emb")
        self._neg_jaccard = batch.get("neg_jaccard")  # (B, k), jaccard mode only

    def extra_forwards(self, ctx: StepCtx, primary: ForwardArtifacts) -> Optional[dict]:
        if not primary.is_train:
            return None
        neg = self._neg_crossattn
        if neg is None:
            return None
        net = ctx.accelerator.unwrap_model(ctx.network)
        if float(getattr(net, "_contrastive_target_weight", 0.0) or 0.0) <= 0.0:
            return None
        # Warmup gate: while _contrastive_weight is held at 0, the k negative
        # DiT value forwards below would be pure waste — skip the whole block.
        if float(getattr(net, "_contrastive_weight", 0.0) or 0.0) <= 0.0:
            self._pending_gradcache = None
            return None
        # Cadence gate: flag set per step by step_contrastive_warmup.
        if not getattr(net, "_contrastive_fire_this_step", True):
            self._pending_gradcache = None
            return None

        device = primary.noisy_model_input.device
        ce_dtype = primary.crossattn_emb.dtype
        neg = neg.to(device)  # (B, k, S, D)
        k = neg.shape[1]
        # Dual bank: negatives splice ψ- (branch 1); anchor stays on ψ+ (branch 0)
        neg_branch = 1 if getattr(net, "n_banks", 1) > 1 else 0

        v_pos = primary.model_pred.squeeze(2)  # (B, C, H, W) — live anchor graph
        v_target = primary.noise - primary.latents  # rectified-flow velocity target
        timesteps = primary.timesteps
        base_kw = dict(primary.forward_kwargs)
        neg_penalty = self._neg_penalty(net, device)

        # Snapshot the anchor's splice state; the negative value passes mutate
        # the per-step buffers, so restore afterwards (the anchor's autograd
        # graph holds tensor references, unaffected by these attribute writes).
        anchor_tokens = net._step_layer_tokens
        anchor_seqlens = net._step_seqlens

        dit = ctx.accelerator.unwrap_model(primary.anima_call)

        # GOTCHA: gradient caching, split so the negative DiT forward NEVER
        # overlaps the anchor backward. Naive checkpoint-and-recompute OOMs
        # (two forwards resident) and crashes under block swap (recompute
        # re-enters _run_blocks against the offloader's end-of-forward layout).
        # Instead split ∂L_con/∂θ into two clean forward/backward partials:
        # ∂L/∂v_pos rides the anchor's FM backward (live logit_pos + detached
        # negatives); ∂L/∂v_neg is deferred to after_backward (anchor freed) —
        # negatives forwarded once here under no_grad for values + cached g_neg,
        # replayed there. prepare_block_swap_before_forward is a no-op at
        # blocks_to_swap=0, so one path serves swap and no-swap.
        v_neg_vals = []
        with torch.no_grad():
            for j in range(k):
                dit.prepare_block_swap_before_forward(free_cache=False)
                v_neg_vals.append(
                    self._bank_forward(
                        net,
                        primary.anima_call,
                        primary.noisy_model_input,
                        primary.padding_mask,
                        base_kw,
                        timesteps,
                        ce_dtype,
                        neg[:, j],
                        branch=neg_branch,
                    )
                )
        net._step_layer_tokens = anchor_tokens
        net._step_seqlens = anchor_seqlens
        v_neg = torch.stack(v_neg_vals, dim=1)  # (B, k, C, H, W), detached values

        live = float(getattr(net, "_contrastive_weight", 0.0) or 0.0)

        if getattr(net, "contrastive_objective", "infonce") == "softrank":
            # Grad via live v_pos, negatives detached; ∂L/∂v_neg grad-cached +
            # replayed exactly like InfoNCE.
            loss = net.softrank_loss(v_pos, v_neg, v_target)
            diag = net.softrank_diagnostics(v_pos.detach(), v_neg, v_target)
            self._record_softrank_metrics(net, loss, diag)
            if live > 0.0:
                v_neg_leaf = v_neg.detach().requires_grad_(True)
                g_loss = net.softrank_loss(v_pos.detach(), v_neg_leaf, v_target)
                (g_neg,) = torch.autograd.grad(g_loss, v_neg_leaf)
                self._pending_gradcache = self._build_gradcache(
                    net,
                    dit,
                    primary,
                    base_kw,
                    timesteps,
                    neg,
                    ce_dtype,
                    g_neg.detach(),
                    live,
                    anchor_tokens,
                    anchor_seqlens,
                    neg_branch,
                )
            else:
                self._pending_gradcache = None
            return {"soft_tokens_contrastive": loss}

        # Composer-side loss: grad only via v_pos (negatives are constants).
        logit_pos, logits_neg = net._velocities_to_logits(
            v_pos, v_neg, v_target, neg_penalty
        )
        loss = net._infonce_from_logits(logit_pos, logits_neg)
        diag = net._contrastive_diagnostics(logit_pos.detach(), logits_neg.detach())
        self._record_metrics(net, loss, diag)

        if live > 0.0:
            # Cache ∂L/∂v_neg with v_pos held constant (no DiT forward — tiny head).
            v_neg_leaf = v_neg.detach().requires_grad_(True)
            lp_d, ln_leaf = net._velocities_to_logits(
                v_pos.detach(), v_neg_leaf, v_target, neg_penalty
            )
            g_loss = net._infonce_from_logits(lp_d, ln_leaf)
            (g_neg,) = torch.autograd.grad(g_loss, v_neg_leaf)
            self._pending_gradcache = self._build_gradcache(
                net,
                dit,
                primary,
                base_kw,
                timesteps,
                neg,
                ce_dtype,
                g_neg.detach(),
                live,
                anchor_tokens,
                anchor_seqlens,
                neg_branch,
            )
        else:
            self._pending_gradcache = None
        return {"soft_tokens_contrastive": loss}

    @staticmethod
    def _build_gradcache(
        net,
        dit,
        primary,
        base_kw,
        timesteps,
        neg,
        ce_dtype,
        g_neg,
        weight,
        anchor_tokens,
        anchor_seqlens,
        neg_branch,
    ) -> dict:
        """Pack the deferred ∂L/∂v_neg replay state. Objective-agnostic — the
        replay in ``after_backward`` pushes the cached ``g_neg`` back through
        each negative's (live-bank) forward, so InfoNCE and softrank share it.
        ``neg_branch`` is the bank the negative spliced on (ψ- under dual bank)."""
        return {
            "net": net,
            "dit": dit,
            "anima_call": primary.anima_call,
            "noisy_model_input": primary.noisy_model_input,
            "padding_mask": primary.padding_mask,
            "timesteps": timesteps,
            "base_kw": base_kw,
            "neg": neg,
            "ce_dtype": ce_dtype,
            "g_neg": g_neg,
            "weight": weight,
            "anchor_tokens": anchor_tokens,
            "anchor_seqlens": anchor_seqlens,
            "neg_branch": neg_branch,
        }

    def after_backward(self, ctx: StepCtx) -> None:
        """Replay the cached contrastive negatives after the FM backward.

        The anchor graph is freed, so each negative re-forward + backward peaks
        at a single graph. The cached ``weight·g_neg`` accumulates onto the FM
        grads on the same params (no ``zero_grad`` between here and the
        optimizer step). Single-process only — multi-GPU would need DDP
        no-sync handling for a manual backward inside accumulate.
        """
        pend = self._pending_gradcache
        if pend is None:
            return
        self._pending_gradcache = None

        net = pend["net"]
        dit = pend["dit"]
        accel = ctx.accelerator
        # Match accelerate's 1/N loss scaling so contrastive grads land on the
        # same scale as the FM grads it accumulates alongside.
        accum = max(1, int(getattr(accel, "gradient_accumulation_steps", 1) or 1))
        scale = pend["weight"] / accum
        neg = pend["neg"]
        k = neg.shape[1]
        ts = pend["timesteps"]
        ce_dtype = pend["ce_dtype"]
        g_neg = pend["g_neg"]

        neg_branch = pend.get("neg_branch", 0)
        with accel.autocast(), torch.enable_grad():
            for j in range(k):
                dit.prepare_block_swap_before_forward(free_cache=False)
                v_neg_j = self._bank_forward(
                    net,
                    pend["anima_call"],
                    pend["noisy_model_input"],
                    pend["padding_mask"],
                    pend["base_kw"],
                    ts,
                    ce_dtype,
                    neg[:, j],
                    branch=neg_branch,
                )
                grad_j = (scale * g_neg[:, j]).to(v_neg_j.dtype)
                torch.autograd.backward(v_neg_j, grad_tensors=grad_j)
        net._step_layer_tokens = pend["anchor_tokens"]
        net._step_seqlens = pend["anchor_seqlens"]

    @staticmethod
    def _bank_forward(
        net,
        anima_call,
        noisy_model_input,
        padding_mask,
        base_kw,
        timesteps,
        ce_dtype,
        text_emb,
        branch: int = 0,
    ) -> torch.Tensor:
        """One DiT forward conditioned on ``text_emb`` → velocity (B, C, H, W).

        Re-primes the per-block soft-token splice for this text and runs the
        frozen DiT with the anchor's (x_t, ε, t). ``branch`` picks ψ+ (0) / ψ- (1)
        for dual-bank nets (no-op when single bank).
        """
        text_emb = text_emb.to(dtype=ce_dtype)
        # front_of_padding needs per-sample seqlens; end_of_sequence skips the
        # abs-sum reduction.
        if net.splice_position == "front_of_padding":
            seqlens = (text_emb.abs().sum(dim=-1) > 0).sum(dim=-1).to(torch.int32)
        else:
            seqlens = None
        net._set_step_tokens(timesteps, seqlens, branch=branch)
        kw_j = dict(base_kw)
        if "pooled_text_override" in kw_j:
            kw_j["pooled_text_override"] = text_emb.max(dim=1).values
        return anima_call(
            noisy_model_input, timesteps, text_emb, padding_mask=padding_mask, **kw_j
        ).squeeze(2)

    def _neg_penalty(self, net, device) -> Optional[torch.Tensor]:
        """jaccard mode: α·s subtracted from each negative logit. ``None`` for
        shuffled/hard."""
        if (
            getattr(net, "contrastive_negative_mode", "shuffled") == "jaccard"
            and self._neg_jaccard is not None
        ):
            alpha = float(getattr(net, "contrastive_jaccard_alpha", 1.0) or 0.0)
            return alpha * self._neg_jaccard.to(device).float()
        return None

    def _record_metrics(self, net, loss, diag) -> None:
        live = float(getattr(net, "_contrastive_weight", 0.0) or 0.0)
        loss_val = float(loss.detach().item())
        self._last_metrics = {
            "reg/soft_tokens_contrastive": loss_val,
            "reg/soft_tokens_contrastive_weighted": live * loss_val,
            "reg/soft_tokens_contrastive_lambda_live": live,
            "soft_tokens/contrastive_acc": diag["contrastive_acc"],
            "soft_tokens/contrastive_logit_gap": diag["contrastive_logit_gap"],
        }

    def _record_softrank_metrics(self, net, loss, diag) -> None:
        """soft-rank diagnostics; ``contrastive_acc``/``contrastive_logit_gap``
        (here a reward gap) mirror InfoNCE's for cross-objective comparability."""
        live = float(getattr(net, "_contrastive_weight", 0.0) or 0.0)
        loss_val = float(loss.detach().item())
        self._last_metrics = {
            "reg/soft_tokens_contrastive": loss_val,
            "reg/soft_tokens_contrastive_weighted": live * loss_val,
            "reg/soft_tokens_contrastive_lambda_live": live,
            "soft_tokens/contrastive_acc": diag["contrastive_acc"],
            "soft_tokens/contrastive_logit_gap": diag["contrastive_logit_gap"],
            "soft_tokens/softrank_matched_rank": diag["softrank_matched_rank"],
        }

    def metrics(self, ctx) -> dict:
        del ctx
        return dict(self._last_metrics)
