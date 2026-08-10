"""E30.1 base-frame accumulation — count-sketch of base-DiT gradient sums.

The e30 registration's instrument (`paper_bench/experiments/e30/README.md`
§E30.1): in the same backward pass as the standard adapter-param arm sums,
accumulate per-(arm, bin) **base-weight** gradient sums as fixed-seed
count-sketches (feature hashing, k buckets, ±1 signs) — three families
(full base / adaln slice / complement) plus a sketched copy of the exact
LoRA per-bin vector (the gate-2 self-calibration family) — with the adaln
slice additionally held as an **exact** running sum so per-condition norms
and the cross-condition Gram are exact, not sketch-estimated.

Sketch-and-free: every base param gets a ``post_accumulate_grad_hook``
that scatters the draw's gradient into the current condition's GPU sketch
buffers and immediately drops ``param.grad`` — no full-model grad buffer
ever materializes (the 16 GB card could not hold one). Count-sketch is
linear, so per-draw scatters sum to the sketch of the bin-summed gradient
(up to float order); the complement family is derived at finalize as
full − adaln (exact by the same linearity).

Hashing: per-tensor multiplicative hashes derived from
``blake2b(f"{seed}:{tensor_name}")`` — bucket = top ``log2(k)`` bits of
``idx * a1 + b1`` (mod 2^64, torch int64 wraparound), sign = one mid bit
of ``idx * a2 + b2``. The (seed, name)-keyed constants make the sketch
frame reproducible from the manifest alone — the shared frame any future
E31 run must reuse. Hash quality is not assumed: gate 2 (sketched-vs-exact
cosines on the param family) measures it in-run.

Precision notes (recorded in the store meta): sketch buffers accumulate
fp32 on GPU per condition and fp32 on CPU across images; the exact adaln
slots are fp16 on CPU (~356 MB × conditions — the fp32 variant would not
fit next to the LoRA arm machinery in 46 GB RAM), the same rounding class
as the fp16 arm-store precedent (~1e-3 relative per add, under the
ledger's 2-significant-figure reads); the Gram/norm reduction runs fp64.
Under ``--deterministic`` the CUDA ``scatter_add_`` stays atomics-ordered
(nondeterministic reduction order, warn_only) — all E30.1 reads are
within-run by design, so no cross-run bit-comparability is claimed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

DEFAULT_K = 1 << 18
DEFAULT_SEED = 3021
# frozen name-pattern rule (e30 §E30.1 item 1): every base tensor whose
# name carries an ``adaln_`` component — blocks' adaln_fused_down +
# adaln_up_{self_attn,cross_attn,mlp} and the final layer's
# adaln_modulation. The resolved list lands in the store meta.
DEFAULT_ADALN_PATTERN = r"(^|\.)adaln_"
_HASH_SPEC = (
    "bucket=((idx*a1+b1) >> (64-kbits)) & (k-1); sign=1-2*((idx*a2+b2)>>62 & 1); "
    "a/b = blake2b('{seed}:{name}', 32B) LE u64 quads, a made odd; int64 wraparound"
)
# bound the transient int64 hash arrays to ~33 MB — the first smoke OOM'd
# on fragmentation (1.49 GiB reserved-unallocated) with 134 MB transients
_CHUNK = 1 << 22


def _hash_constants(seed: int, name: str) -> tuple[int, int, int, int]:
    h = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=32).digest()

    def signed64(b: bytes, make_odd: bool = False) -> int:
        v = int.from_bytes(b, "little")
        if make_odd:
            v |= 1
        return v - (1 << 64) if v >= 1 << 63 else v

    return (
        signed64(h[0:8], make_odd=True),
        signed64(h[8:16]),
        signed64(h[16:24], make_odd=True),
        signed64(h[24:32]),
    )


def _sanit(key: str) -> str:
    return key.replace("@", "~")  # the ArmSumAccumulator filename convention


class BaseSketchAccumulator:
    """Per-(arm key, σ bin) base-gradient sketches + exact adaln slice sums.

    Wiring: :meth:`begin_arm` before each arm's ``grad_estimate_binned``
    call, then the kernel calls :meth:`end_bin` once per bin with the
    bin's exact flat LoRA vector (still on GPU). Hooks do the per-draw
    work in between. :meth:`finalize` writes ``<store>/base_sketch/`` and
    returns the manifest block for the arm-sums manifest.
    """

    def __init__(
        self,
        anima: torch.nn.Module,
        network: torch.nn.Module,
        device: torch.device,
        k: int = DEFAULT_K,
        seed: int = DEFAULT_SEED,
        adaln_pattern: str = DEFAULT_ADALN_PATTERN,
    ) -> None:
        assert k & (k - 1) == 0, "--base_sketch_k must be a power of two"
        self.k, self.seed, self.kbits = k, seed, k.bit_length() - 1
        self.adaln_pattern = adaln_pattern
        self.device = device
        pat = re.compile(adaln_pattern)
        lora_ids = {id(p) for p in network.parameters()}
        self.entries: list[tuple[str, torch.nn.Parameter, int, tuple]] = []
        self.adaln_names: dict[str, int] = {}
        adaln_off = 0
        for name, p in sorted(anima.named_parameters()):
            if id(p) in lora_ids:
                continue
            off = -1
            if pat.search(name):
                off = adaln_off
                adaln_off += p.numel()
                self.adaln_names[name] = p.numel()
            self.entries.append((name, p, off, _hash_constants(seed, name)))
        self.adaln_numel = adaln_off
        self.base_numel = sum(p.numel() for _, p, _, _ in self.entries)
        self.param_consts = _hash_constants(seed, "lora_flat")

        self.buf = {
            fam: torch.zeros(k, dtype=torch.float32, device=device)
            for fam in ("full", "adaln", "param")
        }
        self.adaln_exact_gpu = torch.zeros(
            self.adaln_numel, dtype=torch.float32, device=device
        )
        self.sk: dict[tuple[str, str, int], np.ndarray] = {}
        self.exact: dict[tuple[str, int], np.ndarray] = {}
        self.fires = 0
        self.fire_counts: dict[tuple[str, int], int] = {}
        self.cur_key: str | None = None
        self._hooks = []
        for name, p, off, consts in self.entries:
            p.requires_grad_(True)
            self._hooks.append(
                p.register_post_accumulate_grad_hook(self._make_hook(off, consts))
            )
        log.info(
            f"base_sketch: {len(self.entries)} base tensors "
            f"({self.base_numel / 1e9:.2f}B params), adaln slice "
            f"{len(self.adaln_names)} tensors ({self.adaln_numel / 1e6:.1f}M), "
            f"k=2^{self.kbits}, seed={seed}"
        )

    def _sketch_add(self, buf: torch.Tensor, flat: torch.Tensor, consts) -> None:
        a1, b1, a2, b2 = consts
        n = flat.numel()
        for s in range(0, n, _CHUNK):
            idx = torch.arange(s, min(s + _CHUNK, n), device=flat.device)
            bucket = ((idx * a1 + b1) >> (64 - self.kbits)) & (self.k - 1)
            sign = 1.0 - 2.0 * ((idx * a2 + b2) >> 62 & 1).to(torch.float32)
            buf.scatter_add_(0, bucket, flat[s : s + _CHUNK].float() * sign)

    def _make_hook(self, adaln_off: int, consts):
        def hook(p: torch.nn.Parameter) -> None:
            g = p.grad
            if g is None:
                return
            flat = g.detach().reshape(-1)
            self._sketch_add(self.buf["full"], flat, consts)
            if adaln_off >= 0:
                self._sketch_add(self.buf["adaln"], flat, consts)
                self.adaln_exact_gpu[adaln_off : adaln_off + flat.numel()] += (
                    flat.float()
                )
            p.grad = None
            self.fires += 1

        return hook

    def begin_arm(self, key: str) -> None:
        self.cur_key = key

    def end_bin(self, bin_idx: int, lora_vec_gpu: torch.Tensor) -> None:
        assert self.cur_key is not None, "begin_arm() must precede end_bin()"
        cond = (self.cur_key, bin_idx)
        self._sketch_add(self.buf["param"], lora_vec_gpu, self.param_consts)
        for fam, buf in self.buf.items():
            slot = self.sk.setdefault((fam, *cond), np.zeros(self.k, dtype=np.float32))
            slot += buf.cpu().numpy()
            buf.zero_()
        ex = self.exact.get(cond)
        if ex is None:
            ex = self.exact[cond] = np.zeros(self.adaln_numel, dtype=np.float16)
        ex += self.adaln_exact_gpu.cpu().numpy()
        self.adaln_exact_gpu.zero_()
        self.fire_counts[cond] = self.fire_counts.get(cond, 0) + self.fires
        self.fires = 0

    def finalize(self, store_dir: Path) -> dict:
        out = Path(store_dir) / "base_sketch"
        out.mkdir(parents=True, exist_ok=True)
        conds = sorted({(key, bi) for _, key, bi in self.sk})
        cname = [f"{_sanit(key)}__b{bi}" for key, bi in conds]

        for fam in ("full", "adaln", "param"):
            np.savez(
                out / f"sketch_{fam}.npz",
                **{n: self.sk[(fam, *c)] for n, c in zip(cname, conds)},
            )
        np.savez(  # complement = full − adaln, exact by sketch linearity
            out / "sketch_complement.npz",
            **{
                n: self.sk[("full", *c)] - self.sk[("adaln", *c)]
                for n, c in zip(cname, conds)
            },
        )

        # exact adaln Gram + norms (fp64 reduction over the fp16 slots),
        # then the slots are dropped — exact slice VECTORS are not stored
        # (the e30 recorded forfeit stands)
        n = len(conds)
        gram = np.zeros((n, n), dtype=np.float64)
        for s in range(0, self.adaln_numel, _CHUNK):
            m = np.stack(
                [self.exact[c][s : s + _CHUNK].astype(np.float64) for c in conds]
            )
            gram += m @ m.T
        np.savez(
            out / "adaln_exact.npz",
            gram=gram,
            norms=np.sqrt(np.diag(gram)),
            conditions=np.array(cname),
        )
        self.exact.clear()

        meta = {
            "k": self.k,
            "seed": self.seed,
            "hash": _HASH_SPEC,
            "families": ["full", "adaln", "complement", "param"],
            "adaln_pattern": self.adaln_pattern,
            "adaln_resolved": self.adaln_names,
            "adaln_numel": self.adaln_numel,
            "n_base_tensors": len(self.entries),
            "base_numel": self.base_numel,
            "conditions": cname,
            "hook_fires_per_condition": {
                n_: self.fire_counts[c] for n_, c in zip(cname, conds)
            },
            "exact_slice": "fp16 CPU accumulation -> fp64 Gram/norms; "
            "vectors discarded at finalize (e30 forfeit)",
            "lora_param_family": "sketch of the exact per-bin LoRA flat vector "
            "(gate-2 self-calibration; hash key 'lora_flat')",
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta
