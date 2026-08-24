"""Teacher prediction caches for distillation.

The teacher path is fully frozen (``skip_pooled_text_proj=True`` and the DiT
body is not trained), so for a fixed
``(latents, crossattn_emb, sigma, noise)`` quadruple the teacher pred is
invariant across iterations. These caches exploit that to skip the teacher
forward on a cache hit.

* :class:`TeacherCache` — training-time. Discretizes sigma onto a grid of K
  pre-sampled values and ties noise deterministically to
  ``(sample_idx, sigma_idx)`` so cache hits and misses produce identical
  ``(latents, noise, sigma)`` inputs to the student.
* :class:`ValTeacherCache` — validation-time. Val is deterministic across
  calls (frozen DiT + ``shuffle=False`` + fixed sigma list + reseeded noise
  generator), so the first pass fills a ``(batch_idx, sigma_idx)`` cache and
  every subsequent pass skips teacher forwards entirely.
"""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm

from library.training.forward import (
    make_padding_mask,
    renoise,
    run_mini_train_forward,
    to_dit_5d,
)

logger = logging.getLogger(__name__)


class TeacherCache:
    """In-RAM cache of teacher predictions keyed by ``(sample_idx, sigma_idx)``.

    The teacher path is fully frozen (``skip_pooled_text_proj=True`` and the
    DiT body is not trained), so for a fixed
    ``(latents, crossattn_emb, sigma, noise)`` quadruple the teacher pred is
    invariant across iterations. This cache discretizes sigma onto a grid of
    K pre-sampled values from the same ``sigmoid(scale * N(0,1))``
    distribution as the original training-time sampler, and ties noise
    deterministically to ``(sample_idx, sigma_idx)`` so that cache hits and
    misses produce identical (latents, noise, sigma) inputs to the student.

    Trade-off vs the original (continuous sigma + fresh noise per step):
    each sample sees only K distinct (noise, sigma) pairs over the whole
    run instead of one fresh pair per visit. K=16 still gives more variety
    than the typical 10–20 visits per sample at default settings, but
    discretizes the loss landscape — bench before shipping a quality claim.

    Stored tensors are bf16 on CPU (~128 KB each at default token count;
    ``N_samples * K * 128KB`` total RAM).
    """

    def __init__(self, K: int, sigmoid_scale: float, base_seed: int):
        self.K = int(K)
        self.base_seed = int(base_seed) & 0x7FFFFFFF
        gen = torch.Generator().manual_seed(self.base_seed)
        sigmas = torch.sigmoid(sigmoid_scale * torch.randn(self.K, generator=gen))
        self.sigmas: list[float] = sigmas.tolist()
        self._store: dict[tuple[int, int], torch.Tensor] = {}
        self.hits = 0
        self.misses = 0

    def sample_sigma_idx(self, B: int) -> list[int]:
        return torch.randint(0, self.K, (B,)).tolist()

    def get_sigma(self, sigma_idx: int) -> float:
        return self.sigmas[sigma_idx]

    def make_noise(self, sample_idx: int, sigma_idx: int, shape, device, dtype):
        seed = (
            (self.base_seed * 1_000_003)
            ^ (int(sample_idx) * 1009)
            ^ (int(sigma_idx) + 1)
        ) & 0x7FFFFFFFFFFFFFFF
        gen = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(shape, device=device, dtype=dtype, generator=gen)

    def get(self, sample_idx: int, sigma_idx: int):
        v = self._store.get((int(sample_idx), int(sigma_idx)))
        if v is not None:
            self.hits += 1
            return v
        self.misses += 1
        return None

    def put(self, sample_idx: int, sigma_idx: int, teacher_pred):
        self._store[(int(sample_idx), int(sigma_idx))] = teacher_pred.detach().to(
            dtype=torch.bfloat16, device="cpu"
        )

    def __len__(self) -> int:
        return len(self._store)


def prefill_teacher_cache(teacher_cache, dataset, model, device, dtype):
    """Eagerly compute teacher predictions for every (sample, sigma_idx) pair."""
    K = teacher_cache.K
    n = len(dataset)
    logger.info(f"Prefilling teacher cache: {n} samples × {K} sigmas = {n * K} entries")
    for sample_idx in tqdm(range(n), desc="prefill teacher"):
        sample = dataset[sample_idx]
        latents = sample["latents"].unsqueeze(0).to(device, dtype=dtype)
        crossattn_emb = sample["crossattn_emb"].unsqueeze(0).to(device, dtype=dtype)
        padding_mask = make_padding_mask(latents, dtype)
        for sigma_idx in range(K):
            sigma = teacher_cache.get_sigma(sigma_idx)
            sigma_t = torch.full((1,), float(sigma), device=device, dtype=latents.dtype)
            noise = teacher_cache.make_noise(
                sample_idx, sigma_idx, latents.shape, device, latents.dtype
            )
            noisy = to_dit_5d(renoise(latents, sigma_t, noise))
            teacher_pred = run_mini_train_forward(
                model,
                noisy,
                sigma_t,
                crossattn_emb,
                padding_mask=padding_mask,
                dtype=dtype,
                no_grad=True,
                skip_pooled_text_proj=True,
            )
            teacher_cache.put(sample_idx, sigma_idx, teacher_pred)
    logger.info(f"Prefill complete: {len(teacher_cache)} entries cached")


class ValTeacherCache:
    """In-RAM cache of validation-time teacher predictions keyed by
    ``(batch_idx, sigma_idx)``.

    Validation is fully deterministic across calls — DiT body is frozen,
    val dataloader runs ``shuffle=False, drop_last=True``, ``validation_sigmas``
    is a fixed list, and the noise generator is reseeded with
    ``validation_seed`` at the top of every pass and advanced in iteration
    order. So the teacher prediction at ``(batch_idx, sigma_idx)`` is invariant
    across calls. The first val pass fills the cache; every subsequent pass
    hits and skips the teacher forward entirely.

    Stored tensors are bf16 on CPU. RAM cost is
    ``n_val_batches * len(sigmas) * batch_bytes`` — typically tens of MB for
    a 5% val split at 4096-token bucket size.
    """

    def __init__(self):
        self._store: dict[tuple[int, int], torch.Tensor] = {}
        self.hits = 0
        self.misses = 0

    def get(self, batch_idx: int, sigma_idx: int):
        v = self._store.get((int(batch_idx), int(sigma_idx)))
        if v is not None:
            self.hits += 1
            return v
        self.misses += 1
        return None

    def put(self, batch_idx: int, sigma_idx: int, teacher_pred):
        self._store[(int(batch_idx), int(sigma_idx))] = teacher_pred.detach().to(
            dtype=torch.bfloat16, device="cpu"
        )

    def __len__(self) -> int:
        return len(self._store)
