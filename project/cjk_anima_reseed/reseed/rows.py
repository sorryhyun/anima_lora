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

``row_cap`` (experiments only; hypothesis.md § 4 P1): after every optimizer
step each trained row's effective row (pack + ``raw`` × ``row_scale``) is
projected back onto the ball of radius ``row_cap`` — a hard clamp, no
penalty weight. ``project`` does it; the trainer calls it after
``opt.step()``.
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
        row_cap=None,
        rand_init=None,
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
        self.pack_rows = pack.table[self.delta.ext_ids].float().to(device)
        self.row_cap = None if row_cap is None else float(row_cap)
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
        self.context = str(context) if context else ""
        self.n_context = 0
        if warm:
            self._warm_start(warm)
        if frozen:
            self._fill_frozen()
        self.raw_init = None
        if rand_init is not None:
            assert not warm, "a random start is cold"
            self._rand_start(*rand_init)
        if frozen:
            live = (~self.frozen_mask).float()[:, None]
            self.delta.raw.register_hook(lambda g: g * live)
        if self.row_cap is not None:
            n0 = self.project()
            print(
                f"rows: effective row norm capped at {self.row_cap:.3f} after every "
                f"step ({n0} rows over it at init, clipped)",
                flush=True,
            )

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

    def _rand_start(self, seed, mode="random"):
        """The trained rows' start, the norm pull centred on it (``raw_init``),
        not on the pack row. The directions come from their own generator, so
        the run's σ / ε draws stay a pack-start run's, and both modes draw the
        same ones.

        ``random`` (``hira_cold_rand``, user 10-15): every row at an isotropic
        random direction, at its pack row's norm.
        ``stick_ball`` (``hira_cold_sball``, user 10-15): pack row + the
        context's stick over these rows (their mean offset, the 10-03 stick) +
        a random ball ⟂ the stick at the context's mean spike length — § 7's
        cold row with ``m_pack`` / ``q_i`` / ``s`` inherited and ``e_i`` random.
        ``stick_glyph:<set>`` (``hira_cold_sglyph``, user 10-15): the same stick
        and length, the ball each row's predicted ``e_i`` instead
        (``probes/glyph_init.py``'s ``OUT/glyph_init/<set>.pt``: a ridge from
        the glyph's bitmap, fitted on other trained rows), centred over the
        rows, ⟂ the stick.
        ``stick_lean:<c>`` (``hira_cold_slean``, user 10-16): the stick × c, and
        the ball the context's lean against the pack spike (the mean of
        (x_i − stick)·q̂_i, q_i = pack row − the rows' mean pack row, ⟂ the
        stick; § 7's α shrink) along each row's q̂_i plus the random direction
        ⟂ both at the rest of the spike length, so the ball keeps the context's
        length and training need not add the lean on top of a random ball."""
        live = ~self.frozen_mask
        g = torch.Generator().manual_seed(int(seed))
        u = F.normalize(
            torch.randn(int(live.sum()), self.pack_rows.shape[1], generator=g).to(
                self.device
            ),
            dim=1,
        )
        p = self.pack_rows[live]
        if mode == "random":
            off = u * p.norm(dim=1, keepdim=True) - p
            what = "isotropic at their pack row's norm"
        else:
            kind, _, gset = mode.partition(":")
            assert kind in ("stick_ball", "stick_glyph", "stick_lean"), mode
            assert self.context, "a stick start needs the context (seed) rows"
            src = torch.load(self.context, map_location="cpu", weights_only=False)
            src_idx = {int(e): i for i, e in enumerate(src["delta"]["ext_ids"])}
            ids = [int(e) for e, m in zip(self.delta.ext_ids, live.tolist()) if m]
            miss = [e for e in ids if e not in src_idx]
            assert not miss, f"{len(miss)} trained rows not in {self.context}"
            x = src["delta"]["raw"][[src_idx[e] for e in ids]].float()
            x = (x * float(src["delta"]["row_scale"])).to(self.device)
            stick = x.mean(0)
            radius = (x - stick).norm(dim=1).mean()
            s1 = F.normalize(stick, dim=0)
            ball = "a random ball"
            if kind == "stick_glyph":
                from . import OUT

                gi = torch.load(OUT / "glyph_init" / f"{gset}.pt", weights_only=False)
                at = {int(e): i for i, e in enumerate(gi["ext_ids"])}
                miss = [e for e in ids if e not in at]
                assert not miss, f"{len(miss)} trained rows not in glyph_init {gset}"
                u = gi["e_hat"][[at[e] for e in ids]].float().to(self.device)
                u = u - u.mean(0)
                ball = f"a predicted ball (glyph_init {gset}, train {gi['train']})"
            u = F.normalize(u - (u @ s1)[:, None] * s1, dim=1)
            off = stick + radius * u
            if kind == "stick_lean":
                q = p - p.mean(0)
                qh = F.normalize(q - (q @ s1)[:, None] * s1, dim=1)
                lean = ((x - stick) * qh).sum(1).mean()
                assert lean.abs() < radius, f"lean {float(lean):.1f} ≥ {float(radius):.1f}"
                u = F.normalize(u - (u * qh).sum(1, keepdim=True) * qh, dim=1)
                stick = float(gset) * stick
                off = stick + lean * qh + (radius**2 - lean**2).sqrt() * u
                ball = f"a ball leaning {float(lean):.1f} along q̂ + a random rest"
            pair = (u @ u.T)[torch.triu(torch.ones_like(u @ u.T), 1).bool()].mean()
            what = (
                f"pack row + the context's stick |{float(stick.norm()):.1f}| + {ball} "
                f"⟂ it at {float(radius):.1f}, pairwise cos {float(pair):.3f}"
            )
        r = p + off
        with torch.no_grad():
            self.delta.raw[live] = (off / self.row_scale).to(self.delta.raw.dtype)
        self.raw_init = self.delta.raw.detach().clone()
        cos = F.cosine_similarity(r, p, dim=1)
        print(
            f"rows {mode} start: {int(live.sum())} rows {what} (seed {seed}); "
            f"|row| mean {float(r.norm(dim=1).mean()):.1f}, cos to the pack row mean "
            f"{float(cos.mean()):.4f} max {float(cos.abs().max()):.4f}; the norm pull "
            "centred on the start",
            flush=True,
        )

    def effective(self) -> torch.Tensor:
        """Every row's effective row: pack + ``raw`` × ``row_scale``."""
        return self.pack_rows + self.delta.raw.detach() * self.row_scale

    def project(self) -> int:
        """Clip the trained rows' effective norm to ``row_cap``; returns how
        many rows it clipped (0 without a cap)."""
        if self.row_cap is None:
            return 0
        with torch.no_grad():
            eff = self.effective()
            n = eff.norm(dim=1)
            over = ~self.frozen_mask & (n > self.row_cap)
            if not bool(over.any()):
                return 0
            clipped = eff[over] * (self.row_cap / n[over])[:, None]
            self.delta.raw[over] = (clipped - self.pack_rows[over]) / self.row_scale
            return int(over.sum())

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
            f = raw if self.raw_init is None else raw - self.raw_init
            loss = loss + self.free_residual * (f**2).sum(1)[self.touched_mask].mean()
        return loss

    # -- logging / export ----------------------------------------------------

    def log_record(self, step, loss_fm, loss, t0, extra=None) -> dict:
        live = ~self.frozen_mask
        dn = (self.delta.raw.detach()[live] * self.row_scale).norm(dim=1)
        en = self.effective()[live].norm(dim=1)
        rec = {
            "step": step,
            "loss": float(loss_fm.detach()),
            "loss_total": float(loss.detach()),
            "delta_norm_mean": float(dn.mean()),
            "delta_norm_max": float(dn.max()),
            "rel": float(dn.mean() / self.row_scale),
            "eff_norm_mean": float(en.mean()),
            "eff_norm_max": float(en.max()),
            "it_s": step / max(time.time() - t0, 1e-6),
            **(extra or {}),
        }
        if self.raw_init is not None:
            e0 = self.pack_rows[live] + self.raw_init[live] * self.row_scale
            rec["init_cos"] = float(
                F.cosine_similarity(self.effective()[live], e0, dim=1).mean()
            )
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
        if self.row_cap is not None:
            sd["row_cap"] = self.row_cap
        if step is not None:
            sd["step"] = step
        return sd
