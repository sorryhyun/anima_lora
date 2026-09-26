"""rows — the trainable rows: an ``ExtDelta`` on the run's vocabs' idx,
warm-started from the seed rows (``paths.SEED_ROWS``).

The rows arm of the stages' ``train/trainables.py`` with the levers left
behind (no ``c_flat``, no pin, no encoder, no adapter LoRA). ``raw`` is in
row-norm units (× ``row_scale``); a source's rows are rescaled by the ratio
of the two ``row_scale``\\ s on the way in, its ``common`` / ``c_flat``
vector (a flat-layout component the old encoder arms carried) dropped.

What trains is ``idx`` — every vocab's idx, not only the ones the captions
touch — so a vocab the data gives no draw stays at its seed row. ``touched``
are the rows with draws: the norm pull applies to them only, so an untouched
row is exact (zero FM gradient, zero pull, ``weight_decay`` 0 → Adam leaves
it).

``frozen`` rows are the rows outside the vocabs that a corpus line carries:
they sit in the hook at their ``context`` (seed) value so the line renders
as it would on the seed, and get a zero gradient, no anchor, no pull.

**``trained.pt`` is the whole merged rows** (2026-09-25, the ctx-arm merge
folded into the save): the seed's rows — rescaled into this run's
``row_scale`` — with the run's rows on top, so a vocab outside the run
renders at its seed row, never as a raw pack row, everywhere the file is
read (eval, bake, Δ reads). The ``seed_merged`` key marks the format; a
pre-merge vocabs-only file fails eval's check and needs a retrain.

``line_mode`` (experiments only; proposal.md § 1, F1): one shared
``v_line`` (``ExtDelta.line``, zero-init, the rows' lr, no pull) added at the
hook to every pack row in a run of ≥ 2, so the rows ``r_i`` and the line
mode split by the data's gate on / off exposure. Saved as ``delta['line']``.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def merge_seed(delta: dict, seed) -> tuple[dict, int]:
    """``delta`` (an ``ExtDelta`` state: ``ext_ids`` / ``raw`` / ``row_scale``)
    with every ``seed`` row it lacks appended at its seed value × (seed
    ``row_scale`` / ``delta``'s), ids sorted — the merged rows. Returns the
    merged delta and how many seed rows were appended."""
    src = torch.load(seed, map_location="cpu", weights_only=False)
    k = float(src["delta"]["row_scale"]) / float(delta["row_scale"])
    have = {int(e) for e in delta["ext_ids"]}
    extra = [i for i, e in enumerate(src["delta"]["ext_ids"]) if int(e) not in have]
    if not extra:
        return delta, 0
    ids = [int(e) for e in delta["ext_ids"]] + [
        int(src["delta"]["ext_ids"][i]) for i in extra
    ]
    raw = torch.cat([delta["raw"].float(), src["delta"]["raw"][extra].float() * k])
    order = sorted(range(len(ids)), key=ids.__getitem__)
    merged = {**delta, "ext_ids": [ids[j] for j in order], "raw": raw[order].clone()}
    return merged, len(extra)


class Rows:
    def __init__(
        self,
        anima,
        device,
        idx,
        pack,
        *,
        warm,
        init_anchor,
        free_residual,
        lr,
        touched=None,
        frozen=(),
        context=None,
        line_mode=False,
    ):
        from common.hooks import ExtDelta

        idx = set(int(e) for e in idx)
        touched = idx if touched is None else set(int(e) for e in touched)
        assert touched <= idx, "touched rows must be the run's"
        frozen = set(int(e) for e in frozen) - idx
        rows = pack.table[sorted(idx)].float()
        self.row_scale = float(rows.norm(dim=1).mean())
        dim = rows.shape[1]
        print(
            f"pack rows: {len(idx)} in the run, {len(touched)} touched "
            f"({len(idx) - len(touched)} untouched — no draw, held exact), "
            f"mean norm {self.row_scale:.3f} (std {rows.norm(dim=1).std():.3f}), dim {dim}",
            flush=True,
        )
        self.device = device
        self.init_anchor = float(init_anchor)
        self.free_residual = float(free_residual)
        self.delta = ExtDelta(anima, idx | frozen, dim, device, self.row_scale)
        self.params = [{"params": [self.delta.raw], "lr": lr}]
        if line_mode:
            self.delta.line = torch.nn.Parameter(torch.zeros(dim, device=device))
            self.params.append({"params": [self.delta.line], "lr": lr})
            print("rows: line mode on — v_line (zero-init) at runs of ≥ 2", flush=True)
        self.touched_mask = torch.tensor(
            [int(e) in touched for e in self.delta.ext_ids],
            dtype=torch.bool,
            device=device,
        )
        self.frozen_mask = torch.tensor(
            [int(e) in frozen for e in self.delta.ext_ids],
            dtype=torch.bool,
            device=device,
        )
        self.warm_mask = torch.zeros(
            len(self.delta.ext_ids), dtype=torch.bool, device=device
        )
        self.raw0 = None
        self.warm_from = ""
        self.context = str(context) if context else ""
        self.n_context = 0
        if warm:
            self._warm_start(warm)
        if frozen:
            self._fill_frozen()
            live = (~self.frozen_mask).float()[:, None]
            self.delta.raw.register_hook(lambda g: g * live)

    @property
    def n_rows(self) -> int:
        """Trainable rows (the run's); frozen context rows do not count."""
        return int((~self.frozen_mask).sum())

    # -- warm chain ----------------------------------------------------------

    def _warm_start(self, path):
        src = torch.load(path, map_location="cpu", weights_only=False)
        src_raw = src["delta"]["raw"].float()
        src_idx = {int(e): i for i, e in enumerate(src["delta"]["ext_ids"])}
        src_rs = src["delta"].get("row_scale")
        k = float(src_rs) / self.row_scale if src_rs is not None else 1.0
        common = None
        enc = src.get("encoder") or {}
        if "common" in enc:
            common = enc["common"].float().cpu()
        elif "c_flat" in src:
            common = src["c_flat"].float().cpu()
        n_warm = 0
        with torch.no_grad():
            for i, e in enumerate(self.delta.ext_ids):
                j = src_idx.get(int(e))
                if j is None or bool(self.frozen_mask[i]):
                    continue
                row = src_raw[j]
                if common is not None:
                    row = row - common
                self.delta.raw[i] = (row * k).to(self.device)
                self.warm_mask[i] = True
                n_warm += 1
        self.raw0 = self.delta.raw.detach().clone()
        self.warm_from = str(path)
        dn = self.delta.raw.detach()[~self.frozen_mask].norm(dim=1)
        cold = self.n_rows - n_warm
        print(
            f"rows warm start: {n_warm}/{self.n_rows} rows from {path} "
            f"(arm {src.get('arm')}, {len(src_idx)} rows); row_scale "
            f"{'–' if src_rs is None else f'{float(src_rs):.3f}'} → {self.row_scale:.3f} "
            f"(× {k:.4f}); row norm mean {float(dn.mean()):.3f} max {float(dn.max()):.3f}"
            + (
                f"; source common ‖{float(common.norm()):.3f}‖ dropped"
                if common is not None
                else ""
            )
            + f"; {cold} cold rows",
            flush=True,
        )
        if self.init_anchor > 0:
            print(
                f"rows: anchor μ‖f − f₀‖² {self.init_anchor:g} on {n_warm} warm rows"
                + (
                    f", μ‖f‖² {self.free_residual:g} on the {cold} cold rows"
                    if cold
                    else ""
                ),
                flush=True,
            )

    def _fill_frozen(self):
        """Fill the frozen rows from the context (seed) rows, rescaled to this
        run's ``row_scale``; a frozen row the context lacks stays zero —
        the raw pack row (the builder keeps such lines out)."""
        assert self.context, "frozen rows need a context (the seed rows)"
        src = torch.load(self.context, map_location="cpu", weights_only=False)
        src_idx = {int(e): i for i, e in enumerate(src["delta"]["ext_ids"])}
        k = float(src["delta"]["row_scale"]) / self.row_scale
        n = 0
        with torch.no_grad():
            for i, e in enumerate(self.delta.ext_ids):
                j = src_idx.get(int(e))
                if j is None or not bool(self.frozen_mask[i]):
                    continue
                self.delta.raw[i] = (src["delta"]["raw"][j].float() * k).to(self.device)
                n += 1
        if self.raw0 is not None:
            self.raw0 = self.delta.raw.detach().clone()
        self.n_context = n
        n_frozen = int(self.frozen_mask.sum())
        print(
            f"rows context: {n}/{n_frozen} frozen rows from {self.context} (× {k:.4f})"
            + (f"; {n_frozen - n} not in it — raw pack rows" if n < n_frozen else ""),
            flush=True,
        )

    # -- loss ----------------------------------------------------------------

    def regularized(self, loss_fm):
        """FM loss + the anchor on warm rows (μ · mean ‖f − f₀‖²) + the norm
        pull on cold rows (μ_free · mean ‖f‖²). No anchor: the pull is on
        every touched row, the one guard against norm creep besides the
        cosine decay (``train.py``). Untouched rows get neither pull
        (the anchor is zero on them by construction)."""
        loss = loss_fm
        raw = self.delta.raw
        if self.raw0 is not None and self.init_anchor > 0:
            anchor = ((raw - self.raw0) ** 2).sum(1)
            loss = loss + self.init_anchor * anchor[self.warm_mask].mean()
            cold = ~self.warm_mask & self.touched_mask
            if self.free_residual > 0 and bool(cold.any()):
                loss = loss + self.free_residual * (raw**2).sum(1)[cold].mean()
        elif self.free_residual > 0 and bool(self.touched_mask.any()):
            loss = loss + self.free_residual * (raw**2).sum(1)[self.touched_mask].mean()
        return loss

    # -- logging / export ----------------------------------------------------

    def log_record(self, step, loss_fm, loss, t0, extra=None) -> dict:
        live = ~self.frozen_mask
        dn = (self.delta.raw.detach()[live] * self.row_scale).norm(dim=1)
        rec = {
            "step": step,
            "loss": float(loss_fm),
            "loss_total": float(loss.detach()),
            "delta_norm_mean": float(dn.mean()),
            "delta_norm_max": float(dn.max()),
            "rel": float(dn.mean() / self.row_scale),
            "it_s": step / max(time.time() - t0, 1e-6),
            **(extra or {}),
        }
        if self.delta.line is not None:
            rec["line_norm"] = float(self.delta.line.detach().norm() * self.row_scale)
        if self.raw0 is not None and bool(self.warm_mask.any()):
            r = self.delta.raw.detach()[self.warm_mask]
            r0 = self.raw0[self.warm_mask]
            rec["warm_cos"] = float(F.cosine_similarity(r, r0, dim=1).mean())
            rec["warm_drift"] = float(
                ((r - r0).norm(dim=1) / r0.norm(dim=1).clamp(min=1e-8)).mean()
            )
        return rec

    def state_dict(self, args: dict, step: int | None = None) -> dict:
        """The probe's ``trained.pt`` shape — ``delta`` (ExtDelta state),
        ``arm`` ``rows``, ``args``, ``killed`` ``""`` — holding the **whole
        merged rows**: every seed row this run does not carry is appended at
        its seed value × (seed ``row_scale`` / ours), so the file renders any
        caption as training did. ``seed_merged`` marks the format."""
        delta = self.delta.state_dict()
        n_seed = int(self.frozen_mask.sum())
        if self.context:
            delta, n_extra = merge_seed(delta, self.context)
            n_seed += n_extra
        sd = {
            "delta": delta,
            "arm": "rows",
            "args": {**args, "init_rows": self.warm_from},
            "killed": "",
            "warm_rows": int(self.warm_mask.sum()),
            "touched_rows": int(self.touched_mask.sum()),
        }
        if self.context:
            sd["seed_merged"] = self.context
            sd["seed_rows"] = n_seed
        if step is not None:
            sd["step"] = step
        return sd
