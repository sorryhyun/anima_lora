"""Per-step training forward for the Anima trainer.

:class:`AnimaStep` owns everything that lives across steps but is touched
inside one: the method adapters, the :class:`RuntimeState` aux bag, the
liveness ledger, the σ-demote router and the functional-loss hooks.
``get_noise_pred_and_target`` is a flat sequence of named phases;
conditional logic lives INSIDE each phase, never as nesting around the call.
"""

from __future__ import annotations

import logging
import typing

import torch

from library.anima import models as anima_models
from library.anima import training as anima_train_utils
from library.anima.trainer.sigma_route import SigmaDemoter
from library.training import (
    SAMPLER_REGISTRY,
    LivenessLedger,
    LossContext,
    RuntimeState,
    SamplerContext,
    TrainCtx,
    build_loss_composer,
    get_huber_threshold_if_needed,
)
from library.training.forward import (
    ForwardConditioning,
    apply_router_conditioning,
    build_forward_conditioning,
    compute_inversion_func_loss,
    prepare_text_conds,
    run_vr_reference_forward,
)
from library.training.mem_reweight import (
    adapted_logmse,
    measure_base_logmse,
    measure_grid_delta,
)
from library.training.method_adapter import ForwardArtifacts, MethodAdapter, StepCtx

logger = logging.getLogger(__name__)


class AnimaStep:
    def __init__(self):
        # Per-method extensions (EasyControl, …), resolved in train() right
        # after the network is built.
        self.adapters: list[MethodAdapter] = []
        # Feature-specific per-run state — see ``RuntimeState``.
        self.state = RuntimeState()
        # Counts aux consumption per skip-if-missing loss; the loop audits it
        # and flags configured-but-dead features with `LIVENESS:`.
        self.liveness = LivenessLedger()
        self.sigma = SigmaDemoter()
        # Realized patch-token histogram over train steps (post σ-demote
        # swap) — per-arm FLOPs accounting; merged into run_end.
        self.token_step_hist: dict[int, int] = {}
        # The unwrapped adapter network (ctx.network is the accelerate
        # wrapper); the loss composer reads it for regularizers.
        self.network = None
        self._padding_mask_cache = {}
        self._paired_step_counter = 0
        self._hydra_warmup_step = 0
        # Functional loss: forward hooks on cross_attn.output_proj at
        # ``_func_blocks`` fill ``_func_captures``.
        self._func_loss = None
        self._func_hooks = []
        self._func_captures = {}
        self._func_blocks = []

    # region adapter dispatch

    def _step_ctx(self, ctx: TrainCtx) -> StepCtx:
        return StepCtx(
            args=ctx.args,
            accelerator=ctx.accelerator,
            network=ctx.network,
            weight_dtype=ctx.weight_dtype,
        )

    def on_step_start(self, ctx: TrainCtx, batch, *, is_train: bool = True):
        if not self.adapters:
            return
        step_ctx = self._step_ctx(ctx)
        for adapter in self.adapters:
            adapter.on_step_start(step_ctx, batch, is_train=is_train)

    def run_after_backward(self, ctx: TrainCtx):
        """Dispatch the post-backward hook to adapters (between
        ``accelerator.backward`` and gradient clipping)."""
        if not self.adapters:
            return
        step_ctx = self._step_ctx(ctx)
        for adapter in self.adapters:
            adapter.after_backward(step_ctx)

    # endregion

    def attach_network(self, args, network, unet) -> None:
        """Bind the freshly built network (before ``apply_to``) and install
        the functional-loss capture hooks when configured."""
        self.network = network
        # Aux-loss gating convention (library/training/losses.py docstring):
        # handlers read network._<name>_weight; the trainer stamps it here
        # since functional's weight is a top-level training arg.
        network._functional_loss_weight = float(
            getattr(args, "functional_loss_weight", 0.0) or 0.0
        )
        self._func_loss = None
        self._func_hooks = []
        self._func_captures = {}
        self._func_blocks = []
        if not (
            getattr(args, "functional_loss_weight", 0.0) > 0.0
            and getattr(args, "inversion_dir", None)
        ):
            return
        blocks_str = getattr(args, "functional_loss_blocks", "8,12,16,20")
        try:
            self._func_blocks = sorted(
                int(b.strip()) for b in blocks_str.split(",") if b.strip()
            )
        except ValueError as e:
            raise ValueError(
                f"functional_loss_blocks must be comma-separated integers, got {blocks_str!r}"
            ) from e

        def _make_hook(block_idx: int):
            def _hook(_module, _inputs, output):
                # Fires twice per step (main forward + inversion forward);
                # main runs first, so this snapshots before the 2nd overwrites.
                self._func_captures[block_idx] = output

            return _hook

        blocks_list = unet.blocks  # nn.ModuleList of DiT blocks
        num_blocks = len(blocks_list)
        for bi in self._func_blocks:
            if not (0 <= bi < num_blocks):
                raise ValueError(
                    f"functional_loss_blocks contains out-of-range index {bi} (model has {num_blocks} blocks)"
                )
            module = blocks_list[bi].cross_attn.output_proj
            self._func_hooks.append(module.register_forward_hook(_make_hook(bi)))
        logger.info(
            f"Functional loss enabled: hooks on cross_attn.output_proj at blocks {self._func_blocks}, "
            f"weight={args.functional_loss_weight}, num_runs={args.functional_loss_num_runs}"
        )

    # region forward phases

    def _prime_adapters(self, ctx: TrainCtx, batch, latents, *, is_train) -> None:
        """ALWAYS per step. Method-adapter pre-forward priming (e.g. EasyControl
        runs the cond pre-pass and primes per-block (K_c, V_c)) — consumed by
        the patched cross-attn/self-attn closures during attention."""
        if not self.adapters:
            return
        step_ctx = self._step_ctx(ctx)
        for adapter in self.adapters:
            adapter.prime_for_forward(step_ctx, batch, latents, is_train=is_train)

    def _paired_step_generators(self, args, device, is_train):
        """Common-random-numbers mode (``--paired_step_rng``): per-train-step
        ``(g_sigma, g_noise)`` generators seeded from (seed, step counter),
        decoupled from the global torch stream, so arms sharing a seed see
        identical σ/noise on same-shape steps and checkpoint deltas isolate
        the intervention. ``(None, None)`` when off."""
        if not is_train or not getattr(args, "paired_step_rng", False):
            return None, None
        self._paired_step_counter += 1
        base = (
            int(getattr(args, "seed", 0) or 0) * 1_000_003 + self._paired_step_counter
        ) * 2
        mask = (1 << 62) - 1
        g_sigma = torch.Generator(device=device).manual_seed(base & mask)
        g_noise = torch.Generator(device=device).manual_seed((base + 1) & mask)
        return g_sigma, g_noise

    def _sample_noisy_input(
        self, ctx: TrainCtx, latents, noise, *, is_train, sigmas=None
    ):
        """ALWAYS per step. Draw (noisy input, timesteps, sigmas) via the
        sampler registry and run per-step network router conditioning
        (timestep masks, σ/FEI routers, balance-loss warmup). ``sigmas``
        (pre-drawn flat σ, sigma_lowres's σ-first path) skips the in-sampler
        draw."""
        args = ctx.args
        sampler_fn = SAMPLER_REGISTRY[getattr(args, "sampler", "default") or "default"]
        sampler_out = sampler_fn(
            SamplerContext(
                args=args,
                noise_scheduler=ctx.noise_scheduler,
                latents=latents,
                noise=noise,
                device=ctx.accelerator.device,
                # the input and σ the DiT reads: generation's latent dtype
                weight_dtype=torch.float32
                if getattr(args, "stream", "bf16") == "fp32"
                else ctx.weight_dtype,
                sigmas=sigmas,
            )
        )
        # timesteps are [0,1]-scaled, float32.
        self._hydra_warmup_step = apply_router_conditioning(
            network=ctx.network,
            noisy_model_input=sampler_out.noisy_input,
            timesteps=sampler_out.timesteps,
            is_train=is_train,
            warmup_step=int(self._hydra_warmup_step),
            max_train_steps=int(getattr(args, "max_train_steps", 0) or 0),
            gradient_accumulation_steps=int(
                getattr(args, "gradient_accumulation_steps", 1) or 1
            ),
        )
        return sampler_out.noisy_input, sampler_out.timesteps, sampler_out.sigmas

    def _prepare_conditioning(
        self, ctx: TrainCtx, batch, text_encoder_conds, noisy_model_input
    ):
        """ALWAYS per step. Returns the device-resident ``PreparedTextConds``
        (both text-conditioning modes; normalized to one uniform
        ``ForwardConditioning`` at the forward call site); fires the
        text-conditioned routers (each gated internally on cached crossattn);
        marks grad-checkpointing inputs."""
        network = ctx.network

        if ctx.args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)

        # Unpack text encoder conditions, H2D move, and on-device caption dropout.
        tc = prepare_text_conds(
            text_encoder_conds=text_encoder_conds,
            batch=batch,
            text_encoding_strategy=ctx.text_encoding_strategy,
            network=network,
            device=ctx.accelerator.device,
            weight_dtype=ctx.weight_dtype,
            uncond_crossattn_emb=self.state.uncond_crossattn_1,
        )

        # Network-level GlobalRouter routed on pooled text: fire ONCE per step.
        # apply_router_conditioning ran before text conds were materialized, so
        # this lives outside that helper. No-op otherwise.
        if (
            getattr(network, "use_crossattn_router", False)
            and tc.crossattn_emb is not None
            and hasattr(network, "set_crossattn_routing")
        ):
            network.set_crossattn_routing(tc.crossattn_emb)

        return tc

    def _get_padding_mask(self, latents, *, weight_dtype, device):
        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask_key = (bs, h_latent, w_latent, weight_dtype, device)
        padding_mask = self._padding_mask_cache.get(padding_mask_key)
        if padding_mask is None:
            padding_mask = torch.zeros(
                bs, 1, h_latent, w_latent, dtype=weight_dtype, device=device
            )
            self._padding_mask_cache[padding_mask_key] = padding_mask
        return padding_mask

    def _run_primary_forward(
        self, ctx: TrainCtx, *, anima, noisy_model_input, timesteps, tc, padding_mask
    ):
        """ALWAYS per step. Single, branch-free forward call site: both
        text-conditioning modes normalize to one ``ForwardConditioning`` in
        ``build_forward_conditioning``. Must run inside the primary forward's
        autocast/grad scope (the postfix splice runs learned modules).
        Returns ``(model_pred, cond)``.
        """
        cond = build_forward_conditioning(
            network=ctx.network, tc=tc, timesteps=timesteps
        )
        model_pred = anima(
            noisy_model_input,
            timesteps,
            cond.cond,
            padding_mask=padding_mask,
            **cond.kw,
        )
        return model_pred, cond

    def _attach_aux_losses(
        self,
        ctx: TrainCtx,
        *,
        anima,
        batch,
        latents,
        noise,
        sigmas,
        timesteps,
        noisy_model_input,
        cond: ForwardConditioning,
        padding_mask,
        is_train,
    ) -> None:
        """Trainer-owned aux-loss producers riding the primary forward (func
        inversion loss, VR control variate, memorization Δ-gap). Every gate
        lives INSIDE this phase, including the cached-text requirement
        (``cond.crossattn_emb is not None``). Must run inside the primary
        forward's autocast/grad scope (extra ``anima(...)`` calls)."""
        args = ctx.args

        # Functional MSE loss against a sampled stochastic inversion run. The
        # captures dict is populated by forward hooks on cross_attn.output_proj
        # at ``self._func_blocks``.
        self._func_loss = None
        if is_train and self._func_blocks and cond.crossattn_emb is not None:
            self._func_loss = compute_inversion_func_loss(
                anima_call=anima,
                captures=self._func_captures,
                block_indices=self._func_blocks,
                batch=batch,
                noisy_model_input=noisy_model_input,
                timesteps=timesteps,
                padding_mask=padding_mask,
                has_postfix=cond.has_postfix,
                kw=cond.kw,
                device=ctx.accelerator.device,
                dtype=ctx.weight_dtype,
            )

        # Variance-reduced FM control variate (AsymFlow §5.2). Stash the
        # residual `z` so the loss composer can blend `(y + λ·z)²`.
        if (
            is_train
            and float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0
            and cond.crossattn_emb is not None
        ):
            z_residual = run_vr_reference_forward(
                anima_call=anima,
                network=ctx.network,
                latents=latents,
                noise=noise,
                sigmas=sigmas,
                timesteps=timesteps,
                crossattn_emb=cond.crossattn_emb,
                padding_mask=padding_mask,
                forward_kwargs=cond.kw,
                weight_dtype=ctx.weight_dtype,
                fei_sigma_low_div=float(args.vr_fei_sigma_low_div),
            )
            self.state.extras_for_step["vr"] = {
                "z": z_residual.detach(),
                "state": self.state.vr,
            }

        # Online memorization Δ-gap (mem_reweight.py). Producer half: causal
        # per-item weights plus, on measurement steps, the base forward's
        # per-sample log-MSE. Consumer half (EMA + loss_weights multiply)
        # lives in ``_process_batch_inner``.
        tracker = self.state.mem_tracker
        if (
            is_train
            and tracker is not None
            and cond.crossattn_emb is not None
            and "image_keys" in batch
        ):
            sig = sigmas.reshape(sigmas.shape[0], -1)[:, 0].float()
            low_mask = sig <= tracker.sigma_max
            keys = list(batch["image_keys"])
            base_logmse = None
            grid_deltas = None
            if tracker.extra_sigmas:
                # Multi-draw mode: fixed σ grid × antithetic ε, every visit
                # (not gated on the train draw's σ); the consumer only folds
                # the Δ into the EMA.
                if tracker.should_measure():
                    grid_deltas = measure_grid_delta(
                        anima_call=anima,
                        network=ctx.network,
                        latents=latents,
                        crossattn_emb=cond.crossattn_emb,
                        padding_mask=padding_mask,
                        forward_kwargs=cond.kw,
                        model_dtype=noisy_model_input.dtype,
                        sigmas=tracker.extra_sigmas,
                        n_noise=tracker.extra_noise,
                        delta=tracker.delta,
                        generator=tracker.noise_generator(latents.device),
                    )
            elif tracker.should_measure() and bool(low_mask.any()):
                base_logmse = measure_base_logmse(
                    anima_call=anima,
                    network=ctx.network,
                    noisy_model_input=noisy_model_input,
                    timesteps=timesteps,
                    crossattn_emb=cond.crossattn_emb,
                    padding_mask=padding_mask,
                    forward_kwargs=cond.kw,
                    noise=noise,
                    latents=latents,
                    delta=tracker.delta,
                )
            self.state.extras_for_step["mem_gap"] = {
                "keys": keys,
                "weights": tracker.weights(keys, low_mask.tolist()),
                "low_mask": low_mask,
                "base_logmse": base_logmse,
                "grid_deltas": grid_deltas,
            }

    def _dispatch_adapter_extras(
        self, ctx: TrainCtx, primary: ForwardArtifacts
    ) -> None:
        """ALWAYS per step — both text-conditioning paths. Method-adapter
        extra forwards (soft-tokens, REPA, …). Must dispatch on BOTH paths —
        gating it inside the cached-crossattn branch only would silently skip
        every adapter's aux loss on the in-model text path (crossattn_emb=None
        — EasyControl's default)."""
        if not self.adapters:
            return
        step_ctx = self._step_ctx(ctx)
        for adapter in self.adapters:
            out = adapter.extra_forwards(step_ctx, primary)
            if out:
                self.state.extras_for_step.update(out)

    # endregion

    def get_noise_pred_and_target(
        self,
        ctx: TrainCtx,
        latents,
        batch,
        text_encoder_conds,
        *,
        is_train=True,
    ):
        accelerator = ctx.accelerator
        anima: anima_models.Anima = ctx.unet

        # Reset per-step adapter aux so stale tensors from a prior step can't
        # leak into the loss composer.
        self.state.extras_for_step = {}

        if latents.ndim == 5:  # Fallback for 5D latents (old cache)
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]

        # Paired-step RNG (CRN): dedicated per-step generators for σ + noise,
        # so A/B arms sharing a seed stay noise-locked. (None, None) when off.
        g_sigma, g_noise = self._paired_step_generators(
            ctx.args, ctx.accelerator.device, is_train
        )

        # sigma_lowres: σ-first draw + latent swap. When EVERY sample's σ
        # clears the gate, the whole step runs on the demoted grid. The σ
        # marginal is untouched (drawn merely before the noise), so the
        # all-samples rule is exact at train_batch_size=1 and conservative above.
        latents, sigmas_flat = self.sigma.maybe_demote(
            ctx, batch, latents, is_train, self.adapters, generator=g_sigma
        )
        if is_train and getattr(ctx.args, "sigma_lowres", False):
            # Realized patch-token histogram: counts examples at the grid the
            # step ACTUALLY ran on (post demote swap), for per-arm FLOPs
            # accounting. Emitted with the run_end progress event.
            tok = (latents.shape[-2] // 2) * (latents.shape[-1] // 2)
            self.token_step_hist[tok] = self.token_step_hist.get(tok, 0) + int(
                latents.shape[0]
            )
        if sigmas_flat is None and g_sigma is not None:
            # Paired mode on a batch the demote path didn't draw for: still
            # take σ from the paired stream so every arm shares the sequence.
            from library.runtime.noise import draw_flat_sigmas

            sigmas_flat = draw_flat_sigmas(
                ctx.args,
                latents.shape[0],
                latents.shape[-2],
                latents.shape[-1],
                ctx.accelerator.device,
                generator=g_sigma,
            )

        self._prime_adapters(ctx, batch, latents, is_train=is_train)
        if g_noise is not None:
            noise = torch.randn(
                latents.shape,
                generator=g_noise,
                device=latents.device,
                dtype=latents.dtype,
            )
        else:
            noise = torch.randn_like(latents)
        noisy_model_input, timesteps, sigmas = self._sample_noisy_input(
            ctx, latents, noise, is_train=is_train, sigmas=sigmas_flat
        )
        tc = self._prepare_conditioning(
            ctx, batch, text_encoder_conds, noisy_model_input
        )
        padding_mask = self._get_padding_mask(
            latents, weight_dtype=ctx.weight_dtype, device=accelerator.device
        )
        noisy_model_input = noisy_model_input.unsqueeze(
            2
        )  # [B, C, H, W] -> [B, C, 1, H, W]

        # yarnsig: expose this demoted step's banded-rope params on the model
        # for exactly the span of its forward(s); cleared in the finally so a
        # later native/val/sample forward can never inherit them.
        if self.sigma.yarn_step is not None:
            anima._sigma_lowres_yarn = self.sigma.yarn_step
        try:
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                model_pred, cond = self._run_primary_forward(
                    ctx,
                    anima=anima,
                    noisy_model_input=noisy_model_input,
                    timesteps=timesteps,
                    tc=tc,
                    padding_mask=padding_mask,
                )
                self._attach_aux_losses(
                    ctx,
                    anima=anima,
                    batch=batch,
                    latents=latents,
                    noise=noise,
                    sigmas=sigmas,
                    timesteps=timesteps,
                    noisy_model_input=noisy_model_input,
                    cond=cond,
                    padding_mask=padding_mask,
                    is_train=is_train,
                )
                self._dispatch_adapter_extras(
                    ctx,
                    ForwardArtifacts(
                        anima_call=anima,
                        noisy_model_input=noisy_model_input,
                        timesteps=timesteps,
                        crossattn_emb=cond.crossattn_emb,
                        padding_mask=padding_mask,
                        forward_kwargs=cond.kw,
                        model_pred=model_pred,
                        noise=noise,
                        latents=latents,
                        is_train=is_train,
                    ),
                )
        finally:
            anima._sigma_lowres_yarn = None
        model_pred = model_pred.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]

        # Note: do NOT clear timestep mask here -- gradient checkpointing recomputes the forward
        # pass during backward, so the mask must remain set. It gets overwritten on the next step.

        # Rectified flow target: noise - latents
        target = noise - latents

        weighting = anima_train_utils.compute_loss_weighting_for_anima(
            weighting_scheme=ctx.args.weighting_scheme, sigmas=sigmas, args=ctx.args
        )

        return model_pred, target, timesteps, weighting

    def process_batch(self, ctx: TrainCtx, batch, *, is_train=True) -> torch.Tensor:
        """Split ``caption_dropout_rates`` off the cached TE outputs, then run
        the step. ``get_noise_pred_and_target`` applies the dropout in-place
        after the H2D transfer (doing it here on CPU would clone
        prompt_embeds/crossattn_emb before the copy, blocking the main
        thread)."""
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            # Shallow copy so the original list (with rates appended) stays
            # intact for validation's per-sigma loop that reuses the batch.
            batch = {
                **batch,
                "text_encoder_outputs_list": text_encoder_outputs_list[:-1],
                "caption_dropout_rates": text_encoder_outputs_list[-1],
            }
        return self._process_batch_inner(ctx, batch, is_train=is_train)

    def _process_batch_inner(
        self, ctx: TrainCtx, batch, *, is_train=True
    ) -> torch.Tensor:
        args = ctx.args
        accelerator = ctx.accelerator
        vae = ctx.vae
        vae_dtype = ctx.vae_dtype
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(
                    torch.FloatTensor, batch["latents"].to(accelerator.device)
                )
            else:
                if (
                    args.vae_batch_size is None
                    or len(batch["images"]) <= args.vae_batch_size
                ):
                    latents = vae.encode_pixels_to_latents(
                        batch["images"].to(accelerator.device, dtype=vae_dtype)
                    )
                else:
                    chunks = [
                        batch["images"][i : i + args.vae_batch_size]
                        for i in range(0, len(batch["images"]), args.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        chunk = vae.encode_pixels_to_latents(
                            chunk.to(accelerator.device, dtype=vae_dtype)
                        )
                        list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(
                        torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents)
                    )

        text_encoder_conds = batch.get("text_encoder_outputs_list", None) or []

        # Live text encoding (use_text_cache = false, or a cache slot left
        # empty). The text encoder is frozen, so this runs without grad.
        if len(text_encoder_conds) == 0 or all(c is None for c in text_encoder_conds):
            tokenize_strategy = ctx.tokenize_strategy
            text_encoding_strategy = ctx.text_encoding_strategy
            # With TE caching on the encoders were never needed for training.
            live_encoders = (
                None if args.cache_text_encoder_outputs else ctx.text_encoders
            )
            with torch.no_grad(), accelerator.autocast():
                if args.weighted_captions:
                    input_ids_list, weights_list = (
                        tokenize_strategy.tokenize_with_weights(batch["captions"])
                    )
                    encoded_text_encoder_conds = (
                        text_encoding_strategy.encode_tokens_with_weights(
                            tokenize_strategy,
                            live_encoders,
                            input_ids_list,
                            weights_list,
                        )
                    )
                else:
                    input_ids = [
                        ids.to(accelerator.device) for ids in batch["input_ids_list"]
                    ]
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                        tokenize_strategy, live_encoders, input_ids
                    )

            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(
            ctx,
            latents,
            batch,
            text_encoder_conds,
            is_train=is_train,
        )

        huber_c = get_huber_threshold_if_needed(args, timesteps, ctx.noise_scheduler)

        # Assemble aux dict for the composer: extra_forwards returns from each
        # method adapter plus the trainer-owned functional-loss capture.
        loss_aux: dict = dict(self.state.extras_for_step)
        if self._func_loss is not None:
            loss_aux["func_loss"] = self._func_loss

        # Online memorization Δ-gap, consumer half (producer: _attach_aux_losses).
        # Fold this step's paired Δ into the per-item EMA (measurement steps
        # only), then apply the CAUSAL weights (from pre-step state) onto
        # per-sample loss_weights.
        loss_weights = batch["loss_weights"]
        mem_gap = loss_aux.pop("mem_gap", None)
        if mem_gap is not None:
            tracker = self.state.mem_tracker
            if mem_gap.get("grid_deltas") is not None:
                # Multi-draw Δ was fully computed in the producer, for every
                # batch item regardless of the train draw's σ.
                tracker.update(mem_gap["keys"], mem_gap["grid_deltas"].tolist())
            elif mem_gap["base_logmse"] is not None:
                with torch.no_grad():
                    d = mem_gap["base_logmse"] - adapted_logmse(
                        noise_pred, target, tracker.delta
                    )
                low = mem_gap["low_mask"].tolist()
                tracker.update(
                    [k for k, m in zip(mem_gap["keys"], low) if m],
                    [v for v, m in zip(d.tolist(), low) if m],
                )
            w = mem_gap["weights"]
            if any(x != 1.0 for x in w):
                loss_weights = loss_weights * torch.tensor(
                    w, device=loss_weights.device, dtype=loss_weights.dtype
                )

        network = self.network if self.network is not None else ctx.network
        composer = build_loss_composer(args, network, ledger=self.liveness)
        return composer.compose(
            LossContext(
                args=args,
                batch=batch,
                model_pred=noise_pred,
                target=target,
                timesteps=timesteps,
                weighting=weighting,
                huber_c=huber_c,
                loss_weights=loss_weights,
                network=network,
                aux=loss_aux,
                is_train=is_train,
            )
        )
