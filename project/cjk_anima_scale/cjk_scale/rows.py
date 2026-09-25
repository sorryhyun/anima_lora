"""rows — the trainable table: an ``ExtDelta`` on the run's vocabs' rows,
warm-started from the seed table (``paths.SEED_TABLE``).

The rows arm of the stages' ``train/trainables.py`` with the levers left
behind (no ``c_flat``, no pin, no encoder, no adapter LoRA). ``raw`` is in
row-norm units (× ``row_scale``); a source table's rows are rescaled by the
ratio of the two tables' ``row_scale`` on the way in, its ``common`` /
``c_flat`` vector (a flat-layout component the old encoder arms carried)
dropped.

The table is ``table_ext`` — every vocab's idx, not only the ones the
captions touch — so a vocab the data gives no draw stays at its seed row.
``touched`` are the rows with draws: the norm pull applies to them only, so
an untouched row is exact (zero FM gradient, zero pull, ``weight_decay`` 0 →
Adam leaves it).

``frozen`` rows are the rows outside the vocabs that a corpus line carries:
they sit in the hook at their ``context`` (seed) table value so the line
renders as it would on the seed, get a zero gradient, no anchor, no pull,
and are stripped from ``trained.pt`` — the table stays the vocabs'; eval
overlays the same seed back (``eval.ctx_arm``).
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F


class RowTable:
    def __init__(
        self,
        anima,
        device,
        table_ext,
        pack,
        *,
        warm,
        init_anchor,
        free_residual,
        lr,
        touched=None,
        frozen=(),
        context=None,
    ):
        from common.hooks import ExtDelta

        table_ext = set(int(e) for e in table_ext)
        touched = table_ext if touched is None else set(int(e) for e in touched)
        assert touched <= table_ext, "touched rows must be in the table"
        frozen = set(int(e) for e in frozen) - table_ext
        rows = pack.table[sorted(table_ext)].float()
        self.row_scale = float(rows.norm(dim=1).mean())
        dim = rows.shape[1]
        print(
            f"pack rows: {len(table_ext)} in the table, {len(touched)} touched "
            f"({len(table_ext) - len(touched)} untouched — no draw, held exact), "
            f"mean norm {self.row_scale:.3f} (std {rows.norm(dim=1).std():.3f}), dim {dim}",
            flush=True,
        )
        self.device = device
        self.init_anchor = float(init_anchor)
        self.free_residual = float(free_residual)
        self.delta = ExtDelta(anima, table_ext | frozen, dim, device, self.row_scale)
        self.params = [{"params": [self.delta.raw], "lr": lr}]
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
        self.context = ""
        self.n_context = 0
        if warm:
            self._warm_start(warm)
        if frozen:
            self._context(context)
            live = (~self.frozen_mask).float()[:, None]
            self.delta.raw.register_hook(lambda g: g * live)

    @property
    def n_rows(self) -> int:
        """Trainable rows (the table's); frozen context rows do not count."""
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

    def _context(self, path):
        """Fill the frozen rows from the context table (rescaled to this
        table's ``row_scale``); a frozen row the context lacks stays zero —
        the raw pack row (the builder keeps such lines out)."""
        assert path, "frozen rows need a context table"
        src = torch.load(path, map_location="cpu", weights_only=False)
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
        self.context, self.n_context = str(path), n
        n_frozen = int(self.frozen_mask.sum())
        print(
            f"rows context: {n}/{n_frozen} frozen rows from {path} (× {k:.4f})"
            + (f"; {n_frozen - n} not in it — raw pack rows" if n < n_frozen else ""),
            flush=True,
        )

    # -- loss ----------------------------------------------------------------

    def regularized(self, loss_fm):
        """FM loss + the anchor on warm rows (μ · mean ‖f − f₀‖²) + the norm
        pull on cold rows (μ_free · mean ‖f‖²). No anchor: the pull is on
        every touched row, the one guard against norm creep besides the
        cosine decay (``train/stage.py``). Untouched rows get neither pull
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
        if self.raw0 is not None and bool(self.warm_mask.any()):
            r = self.delta.raw.detach()[self.warm_mask]
            r0 = self.raw0[self.warm_mask]
            rec["warm_cos"] = float(F.cosine_similarity(r, r0, dim=1).mean())
            rec["warm_drift"] = float(
                ((r - r0).norm(dim=1) / r0.norm(dim=1).clamp(min=1e-8)).mean()
            )
        return rec

    def state_dict(self, args: dict, step: int | None = None) -> dict:
        """The probe's ``trained.pt`` shape: ``delta`` (ExtDelta state),
        ``arm`` ``rows``, ``args`` (this run's record), ``killed`` ``""``."""
        delta = self.delta.state_dict()
        if bool(self.frozen_mask.any()):
            keep = (~self.frozen_mask).cpu()
            delta = {
                **delta,
                "ext_ids": [e for e, k in zip(delta["ext_ids"], keep.tolist()) if k],
                "raw": delta["raw"][keep].clone(),
            }
        sd = {
            "delta": delta,
            "arm": "rows",
            "args": {**args, "init_rows": self.warm_from},
            "killed": "",
            "warm_rows": int(self.warm_mask.sum()),
            "touched_rows": int(self.touched_mask.sum()),
        }
        if self.context:
            sd["context"] = self.context
            sd["context_rows"] = self.n_context
        if step is not None:
            sd["step"] = step
        return sd
