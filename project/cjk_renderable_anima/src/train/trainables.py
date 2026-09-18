"""What an arm trains, and how it becomes the ``ExtDelta`` table.

  rows          free per-row delta on the ext rows the training captions touch
                (S0: + a per-source layout vector ``c_flat`` on flat-canvas
                batches only — ``--c_flat``; ``--free_residual`` = μ‖f‖² pull)
  rows_adapter  that + a LoRA on every Linear of ``llm_adapter.blocks``
  encoder       Δ_r = g(glyph_r) [+ f_r free residual on trained rows] (W2d)

Every arm saves the ExtDelta format, so eval / native / classify run unchanged.
"""

from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F
from common.hooks import AdapterLoRA, ExtDelta
from common.render.flat import find_fonts

from .encoder import GlyphEncoder, glyph_bank, glyph_batch, reference_batch, row_texts


class Trainables:
    def __init__(self, a, anima, device, train_ext, ev_ext, tok, pack):
        self.a = a
        self.device = device
        rows = pack.table[sorted(train_ext)].float()
        self.row_scale = float(rows.norm(dim=1).mean())
        dim = rows.shape[1]
        print(
            f"pack rows: mean norm {self.row_scale:.3f} (std {rows.norm(dim=1).std():.3f}), dim {dim}",
            flush=True,
        )
        self.enc = None
        self.bank = None
        self.font_mean = a.font_mode == "mean"
        self.free = None
        self.free_mask = None
        self.row_text = None
        self.lora = None
        self.c_flat = None
        self.raw0 = None  # rows arm: the warm start (f₀) and which rows it filled
        self.warm_mask = None
        if a.arm == "encoder":
            self._init_encoder(train_ext, ev_ext, tok, pack, anima, dim)
        else:
            self.delta = ExtDelta(anima, train_ext, dim, device, self.row_scale)
            self.params = [{"params": [self.delta.raw], "lr": a.lr_rows}]
            if a.c_flat:
                # S0 (plan_synth): Δ_r = f_r + 𝟏[flat-canvas item] · c_flat —
                # the flat layout gets its own switch so the rows stop
                # carrying it; the train stage flips ``delta.common`` per
                # (one-source) batch. Projected onto ‖c_flat‖ ≤ cap.
                self.c_flat = torch.nn.Parameter(torch.zeros(dim, device=device))
                self.params.append(
                    {"params": [self.c_flat], "lr": a.lr_c_flat or a.lr_rows}
                )
                print(
                    f"c_flat: on, lr {a.lr_c_flat or a.lr_rows:g}, cap {a.c_flat_cap:g}"
                    + (f", f_orth {a.f_orth:g}" if a.f_orth else ""),
                    flush=True,
                )
            if a.free_residual > 0:
                print(f"rows: μ‖f‖² pull {a.free_residual:g}", flush=True)
            if a.init_rows:
                self._init_rows_from(a.init_rows)
            self.pin_u = None
            if a.pin_dir:
                self._pin_from(a.pin_dir, tok, pack)
        if a.arm == "rows_adapter":
            self.lora = AdapterLoRA(anima, a.adapter_rank, device)
            self.params.append({"params": list(self.lora.params), "lr": a.lr_adapter})
            print(
                f"adapter LoRA r{a.adapter_rank} on {len(self.lora.patched)} Linears",
                flush=True,
            )

    # -- setup ---------------------------------------------------------------

    def _init_rows_from(self, paths: str):
        """Rows-arm warm start (P0b → S-line probe): copy the source table's
        exported rows by ext id. An encoder source's rows are g(glyph) + f +
        ``common`` (one shared vector, cos ≈ 1 with the table mean); that
        vector is the flat-canvas component, so it is moved out of the rows
        and into ``c_flat`` (clipped to the cap) when the switch is on, and
        dropped otherwise. Rows the source never had stay at zero. A comma
        list loads several tables in order (user, 2026-09-16: the 53k
        table + a punctuation table), a later one overriding by ext id."""
        self.warm_mask = torch.zeros(len(self.delta.ext_ids), dtype=torch.bool)
        for path in [p for p in paths.split(",") if p]:
            self._init_rows_one(path)
        self.warm_mask = self.warm_mask.to(self.device)
        self.raw0 = self.delta.raw.detach().clone()
        if self.a.init_anchor > 0:
            print(
                f"rows: init anchor μ‖f − f₀‖² {self.a.init_anchor:g} on "
                f"{int(self.warm_mask.sum())} warm rows (‖f‖² pull stays on the rest)",
                flush=True,
            )

    def _init_rows_one(self, path: str):
        src = torch.load(path, map_location="cpu", weights_only=False)
        src_raw = src["delta"]["raw"].float()
        src_idx = {int(e): i for i, e in enumerate(src["delta"]["ext_ids"])}
        # ``raw`` is in row-norm units of the run that trained it (delta =
        # raw × row_scale, row_scale = that run's mean pack-row norm), so a
        # source from another inventory is rescaled to apply the same delta
        # here — the correction merge_tables.py makes (Δ1 232.9 → ~197 would
        # otherwise start every row 1.18× too large). A source saved without
        # row_scale is taken as already in this run's units.
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
                if j is None:
                    continue
                row = src_raw[j]
                if common is not None:
                    row = row - common
                self.delta.raw[i] = (row * k).to(self.device)
                self.warm_mask[i] = True
                n_warm += 1
            c_note = ""
            if common is not None and self.c_flat is not None:
                c = common.clone() * k
                cap = float(self.a.c_flat_cap)
                if cap > 0 and c.norm() > cap:
                    c = c * (cap / c.norm())
                self.c_flat.copy_(c.to(self.device))
                c_note = (
                    f"; c_flat seeded from source common ‖{float(common.norm()):.3f}‖"
                    f" → ‖{float(c.norm()):.3f}‖"
                )
            elif common is not None:
                c_note = (
                    f"; source common ‖{float(common.norm()):.3f}‖ dropped (no c_flat)"
                )
        dn = self.delta.raw.detach().norm(dim=1)
        print(
            f"rows warm start: {n_warm}/{len(self.delta.ext_ids)} rows from {path} "
            f"(arm {src.get('arm')}); row_scale {src_rs if src_rs is None else f'{float(src_rs):.3f}'}"
            f" → {self.row_scale:.3f} (× {k:.4f}); row norm mean {float(dn.mean()):.3f} "
            f"max {float(dn.max()):.3f}{c_note}",
            flush=True,
        )

    def _pin_from(self, path: str, tok, pack):
        """Inherit the shared direction (transplant probe, 2026-09-16): the
        source table splits into one shared direction per family (m̂_kana /
        m̂_other) plus near-orthogonal residuals, and a composite-trained
        residual renders on that direction while a flat-trained one does not.
        Here every trained row gets a *fixed* ``a_r · m̂_fam`` (``delta.pinned``)
        and the trainable ``raw`` is the residual only; ``--pin_orth`` keeps it
        ⟂ m̂_fam after each step so the row cannot re-grow the trigger. ``a_r``:
        ``--pin_coef row`` = the source row's own coefficient when the row is
        in the source, else the family mean; ``fam`` = the family mean always;
        a number = that value for every row."""
        from .encoder import row_texts

        a = self.a
        src = torch.load(path, map_location="cpu", weights_only=False)
        s_raw = src["delta"]["raw"].float()
        s_ids = [int(e) for e in src["delta"]["ext_ids"]]
        s_text = row_texts(tok, pack, s_ids)
        my_text = row_texts(tok, pack, self.delta.ext_ids)
        dirs, coefs = {}, {}
        for fam in ("kana", "other"):
            ii = [
                i
                for i, e in enumerate(s_ids)
                if e in s_text and _fam_of(s_text[e]) == fam
            ]
            m = s_raw[ii].mean(0)
            dirs[fam] = m / m.norm()
            coefs[fam] = float((s_raw[ii] @ dirs[fam]).mean())
        s_idx = {e: i for i, e in enumerate(s_ids)}
        n, dim = self.delta.raw.shape
        pinned = torch.zeros(n, dim)
        u = torch.zeros(n, dim)
        n_row = 0
        for i, e in enumerate(self.delta.ext_ids):
            fam = _fam_of(my_text.get(int(e), "他"))
            mh = dirs[fam]
            if a.pin_coef == "row" and int(e) in s_idx:
                c = float(s_raw[s_idx[int(e)]] @ mh)
                n_row += 1
            elif a.pin_coef in ("row", "fam"):
                c = coefs[fam]
            else:
                c = float(a.pin_coef)
            pinned[i] = c * mh
            u[i] = mh
        self.delta.pinned = pinned.to(self.device)
        self.pin_u = u.to(self.device)
        with torch.no_grad():
            self._project_orth()
        print(
            f"pin: shared direction from {path} — family coef {coefs} (mode {a.pin_coef}, "
            f"{n_row} rows with their own coefficient), pinned norm mean "
            f"{float(pinned.norm(dim=1).mean()):.3f}, orth {'on' if a.pin_orth else 'off'}",
            flush=True,
        )

    @torch.no_grad()
    def _project_orth(self):
        if self.pin_u is None or not self.a.pin_orth:
            return
        r = self.delta.raw
        r.sub_((r * self.pin_u).sum(1, keepdim=True) * self.pin_u)

    def _init_encoder(self, train_ext, ev_ext, tok, pack, anima, dim):
        a, device = self.a, self.device
        rows_all = sorted(set(train_ext) | {i for ids in ev_ext.values() for i in ids})
        self.row_text = row_texts(tok, pack, rows_all)
        rows_all = [r for r in rows_all if r in self.row_text]
        self.bank = glyph_bank(
            [self.row_text[r] for r in rows_all], find_fonts(), a.glyph_size
        )
        self.delta = ExtDelta(anima, rows_all, dim, device, self.row_scale)
        enc = self.enc = GlyphEncoder(
            dim,
            out_scale=a.out_scale,
            common_cap=a.common_cap,
            pool=a.enc_pool,
            glyph_size=a.glyph_size,
            head_init=a.head_init,
        ).to(device)
        if a.head_init == "random":
            with torch.no_grad():
                xref = reference_batch(self.bank, device, self.font_mean)
                spread0 = float(enc.identity(xref).norm(dim=1).mean())
                enc.head[-1].weight.mul_(a.init_spread / max(spread0, 1e-8))
                spread1 = float(enc.identity(xref).norm(dim=1).mean())
            print(
                f"encoder head random init: spread {spread0:.4f} → {spread1:.3f} row norms",
                flush=True,
            )
        self.params = [
            {"params": enc.enc_params(), "lr": a.lr_enc},
            {"params": [enc.common], "lr": a.lr_common},
        ]
        self.is_train_row = torch.tensor([r in train_ext for r in self.delta.ext_ids])
        if a.init_encoder:
            src = torch.load(a.init_encoder, map_location="cpu", weights_only=False)
            enc.load_state_dict({k: v.to(device) for k, v in src["encoder"].items()})
            print(
                f"encoder warm start: {a.init_encoder} (arm {src.get('arm')}, "
                f"common norm {float(enc.common.norm()):.3f})",
                flush=True,
            )
        if a.free_residual > 0:
            self._init_free(dim)
        print(
            f"encoder: {sum(p.numel() for p in enc.parameters()) / 1e6:.2f}M params, "
            f"{len(rows_all)} rows ({int(self.is_train_row.sum())} in training captions), "
            f"glyph bank {tuple(self.bank.shape)}, out_scale {a.out_scale:.4g}, "
            f"common lr {a.lr_common:g} cap {a.common_cap:g}, pool {a.enc_pool}, "
            f"font {a.font_mode}",
            flush=True,
        )

    def _init_free(self, dim):
        """Run 1d hybrid: row_i = g(glyph_i) + f_i on trained rows only (the
        mask zeroes held-out rows in the forward, so they get neither a
        residual nor a gradient); μ · mean_i ‖f_i‖² over trained rows."""
        a, device = self.a, self.device
        ext_ids = self.delta.ext_ids
        self.free = torch.nn.Parameter(torch.zeros(len(ext_ids), dim, device=device))
        self.free_mask = self.is_train_row.float().unsqueeze(1).to(device)
        self.n_free = float(self.is_train_row.sum())
        self.params.append({"params": [self.free], "lr": a.lr_free})
        n_warm = 0
        if a.init_free:
            # warm start the per-row residual by ext id (rows the source never
            # had stay at zero — new words start from g alone)
            src_f = torch.load(a.init_free, map_location="cpu", weights_only=False)
            src_idx = {int(e): i for i, e in enumerate(src_f["delta"]["ext_ids"])}
            with torch.no_grad():
                for i, e in enumerate(ext_ids):
                    j = src_idx.get(int(e))
                    if j is not None:
                        self.free[i] = src_f["free"][j].to(device)
                        n_warm += 1
        print(
            f"free residual: {int(self.n_free)} trained rows, μ {a.free_residual:g}, lr {a.lr_free:g}"
            + (
                f", {n_warm} rows warm-started from {a.init_free}"
                if a.init_free
                else ""
            ),
            flush=True,
        )

    # -- per step ------------------------------------------------------------

    def set_source(self, flat: bool):
        """S0: the batch is flat-canvas (``c_flat`` on) or composite (off)."""
        if self.c_flat is not None:
            self.delta.common = self.c_flat if flat else None

    def materialize(self, aug_rng: random.Random):
        """Encoder arm: rebuild ``delta.raw`` from this step's glyph draw."""
        if self.enc is None:
            return
        self.delta.raw = self.enc(
            glyph_batch(self.bank, self.device, aug_rng, font_mean=self.font_mean)
        )
        if self.free is not None:
            self.delta.raw = self.delta.raw + self.free * self.free_mask

    def regularized(self, loss_fm):
        """``(loss, decor term)`` — FM loss plus the encoder arm's penalties."""
        a = self.a
        loss = loss_fm
        decor_val = None
        if self.enc is not None and a.decor > 0:
            # W2d report Run 1b amended: the FM gradient's sign-consistent
            # direction marches the table to rank 1 under Adam (attempt 10
            # PR 1.0, rinit 23 → 1.24). Penalise pairwise cos² of the centred
            # *trained* rows (189×189 per step, no DiT forward): ≈ 1 at PR 1,
            # → 0 as rows spread to the free-rows geometry (pairwise cos 0.04).
            # Held-out rows are excluded from both the centring and the pairs
            # so they get no direct gradient.
            tr = self.delta.raw[self.is_train_row.to(self.delta.raw.device)].float()
            cn_tr = F.normalize(tr - tr.mean(0, keepdim=True), dim=1)
            sim_tr = cn_tr @ cn_tr.T
            n_tr = sim_tr.shape[0]
            decor_val = ((sim_tr**2).sum() - (sim_tr.diagonal() ** 2).sum()) / (
                n_tr * (n_tr - 1)
            )
            loss = loss + a.decor * decor_val
        if self.free is not None:
            free_pen = ((self.free * self.free_mask) ** 2).sum() / self.n_free
            loss = loss + a.free_residual * free_pen
        elif self.enc is None and a.init_anchor > 0 and self.raw0 is not None:
            # rows arm, warm start (2026-09-17): μ · mean_r ‖f_r − f₀_r‖² on the
            # rows --init_rows filled — the stationary point sits at the source
            # table instead of at 0; the rows the source never had keep the
            # S0 pull below
            sq = (self.delta.raw**2).sum(1)
            anchor = ((self.delta.raw - self.raw0) ** 2).sum(1)
            loss = loss + a.init_anchor * anchor[self.warm_mask].mean()
            if a.free_residual > 0 and bool((~self.warm_mask).any()):
                loss = loss + a.free_residual * sq[~self.warm_mask].mean()
        elif self.enc is None and a.free_residual > 0:
            # rows arm (S0): the same μ · mean_r ‖f_r‖² on the free rows —
            # the one guard against norm creep besides the cosine decay
            loss = loss + a.free_residual * (self.delta.raw**2).sum(1).mean()
        if self.c_flat is not None and a.f_orth > 0:
            # keep the rows off the flat-layout axis: λ · mean_r cos²(f_r, c_flat)
            loss = loss + a.f_orth * self._leak(squared=True)
        return loss, decor_val

    def _leak(self, squared: bool = False):
        """mean over trained rows of cos(f_r, c_flat) (² when ``squared``) —
        the canvas-in-the-rows monitor; P0b's table reads ≈ 1 on its c."""
        cn = F.normalize(self.delta.raw.float(), dim=1, eps=1e-6)
        cc = F.normalize(self.c_flat.float(), dim=0, eps=1e-6)
        cos = cn @ cc
        return (cos**2).mean() if squared else cos.mean()

    def after_step(self):
        if self.enc is not None:
            self.enc.clamp_common()  # projected descent on c: no creep
        if self.c_flat is not None:
            with torch.no_grad():
                n = float(self.c_flat.norm())
                if n > self.a.c_flat_cap:
                    self.c_flat.mul_(self.a.c_flat_cap / n)
        if getattr(self, "pin_u", None) is not None:
            self._project_orth()

    # -- logging -------------------------------------------------------------

    def log_record(self, step, loss_fm, loss, decor_val, t0, extra=None) -> dict:
        dn = (self.delta.raw.detach() * self.row_scale).norm(dim=1)
        if self.enc is not None:
            is_tr = self.is_train_row.to(dn.device)
            dn_held = dn[~is_tr]
            dn = dn[is_tr]
        rec = {
            "step": step,
            "loss": loss_fm.item(),
            "delta_norm_mean": float(dn.mean()),
            "delta_norm_max": float(dn.max()),
            "rel": float(dn.mean() / self.row_scale),
            "it_s": step / (time.time() - t0),
            **(extra or {}),  # ΔFM: fm_plain / pres / ref_bias (train/stage.py)
        }
        if self.enc is not None:
            self._log_encoder(rec, loss, decor_val, dn_held)
        if self.raw0 is not None and bool(self.warm_mask.any()):
            # the warm-start ruler: how much of the source table is still there
            r = self.delta.raw.detach()[self.warm_mask]
            r0 = self.raw0[self.warm_mask]
            rec["warm_cos"] = float(F.cosine_similarity(r, r0, dim=1).mean())
            rec["warm_drift"] = float(((r - r0).norm(dim=1) / r0.norm(dim=1)).mean())
        if self.c_flat is not None:
            with torch.no_grad():
                rec["leak"] = float(self._leak())
                rec["c_flat_norm"] = float(self.c_flat.norm())
                rec["loss_total"] = float(loss.detach())
        if self.lora is not None:
            rec["lora_b_norm"] = float(
                sum(p.norm() ** 2 for p in list(self.lora.params)[1::2]) ** 0.5
            )
        return rec

    def _log_encoder(self, rec, loss, decor_val, dn_held):
        enc = self.enc
        # identity lives in the spread between rows, not the common mode. The
        # training draw's spread swings 0.03–0.26 between logs with the
        # font/shift drawn (attempt 4), so the kill rule reads a fixed
        # reference render: font 0, no shift.
        raw = self.delta.raw.detach()
        rec["rel_spread"] = float((raw - raw.mean(0, keepdim=True)).norm(dim=1).mean())
        with torch.no_grad():
            xref = reference_batch(self.bank, self.device, self.font_mean)
            ref = enc.identity(xref)
            feat = enc.features(xref)
        rec["rel_spread_ref"] = float(ref.norm(dim=1).mean())
        # feature spread across rows relative to the common feature: ~0 here
        # with ~0 spread = dead features, not a weight kick
        rec["feat_spread"] = float(
            (feat - feat.mean(0, keepdim=True)).norm(dim=1).mean()
            / feat.mean(0).norm().clamp_min(1e-6)
        )
        rec["rel_common"] = float(enc.common.detach().norm())
        rec["rel_max"] = float(raw.norm(dim=1).max())
        # table rank: attempt 10's centred table had participation ratio 1.0
        # (one axis, cos 0.85 with c) and nearest-neighbour cos ≥ 0.84 for
        # every kana — the instrument the data lever and the rank lever are
        # judged on
        cen = (raw - raw.mean(0, keepdim=True)).float()
        rec["table_pr"] = participation_ratio(cen)
        cn = F.normalize(cen, dim=1)
        sim = cn @ cn.T
        sim.fill_diagonal_(-1.0)
        rec["nn_cos"] = float(sim.max(dim=1).values.mean())
        if decor_val is not None:
            rec["decor"] = float(decor_val.detach())
            rec["loss_total"] = float(loss.detach())
        if self.free is not None:
            # the hybrid's instrument: how much of the trained rows' identity
            # the free residual carries vs the shared g (rel_spread_ref).
            # ≫ 1 = g learned nothing (lookup)
            fn = (self.free.detach() * self.free_mask).norm(dim=1)
            rec["free_norm"] = float(fn.sum() / self.n_free)
            rec["free_max"] = float(fn.max())
            rec["free_ratio"] = float(
                rec["free_norm"] / max(rec["rel_spread_ref"], 1e-6)
            )
            rec["loss_total"] = float(loss.detach())
            rec["g_pr"] = participation_ratio(ref.float())
        if dn_held.numel():
            rec["rel_held"] = float(dn_held.mean() / self.row_scale)

    def kill_reason(self, rec, step) -> str:
        """W2d report kill rules: identity not moving, or a row walking
        off-manifold — stop before spending the eval. Encoder arm only."""
        a = self.a
        killed = ""
        if self.enc is None:
            return killed
        if (
            a.kill_spread_step
            and step >= a.kill_spread_step
            and rec["rel_spread_ref"] < a.kill_spread
        ):
            killed = (
                f"KILL: rel_spread_ref {rec['rel_spread_ref']:.4f} < {a.kill_spread} "
                f"at step {step} (>= {a.kill_spread_step})"
            )
        if a.kill_max_row and rec["rel_max"] > a.kill_max_row:
            killed = f"KILL: max row norm {rec['rel_max']:.3f} > {a.kill_max_row}× at step {step}"
        return killed

    # -- export --------------------------------------------------------------

    @torch.no_grad()
    def export_table(self, aug_rng: random.Random):
        """Encoder arm: the shipped table is the encoder's mean over fonts, no
        shift (+ the free residual) — a static ExtDelta."""
        if self.enc is None:
            return
        enc, bank, device = self.enc, self.bank, self.device
        enc.eval()
        if self.font_mean:
            raw = enc(glyph_batch(bank, device, aug_rng, shift=0, font_mean=True))
        else:
            raw = torch.stack(
                [
                    enc(
                        glyph_batch(
                            bank, device, aug_rng, shift=0, fonts=[f] * bank.shape[0]
                        )
                    )
                    for f in range(bank.shape[1])
                ]
            ).mean(0)
        if self.free is not None:
            raw = raw + self.free * self.free_mask
        self.delta.raw = raw

    def state_dict(self, held, killed) -> dict:
        a = self.a
        sd = {"delta": self.delta.state_dict(), "arm": a.arm, "args": vars(a)}
        if self.enc is not None:
            sd["encoder"] = {k: v.cpu() for k, v in self.enc.state_dict().items()}
            sd["held_out"] = held
            sd["row_text"] = {int(r): self.row_text[r] for r in self.delta.ext_ids}
            if self.free is not None:
                sd["free"] = (self.free.detach() * self.free_mask).cpu()
        if self.lora is not None:
            sd["lora"] = self.lora.state_dict()
            sd["adapter_rank"] = a.adapter_rank
        if self.c_flat is not None:
            sd["c_flat"] = self.c_flat.detach().cpu()
        if getattr(self.delta, "pinned", None) is not None:
            # delta.raw in sd is the full table (pinned folded in); keep the parts
            sd["pinned"] = self.delta.pinned.detach().cpu()
            sd["resid"] = self.delta.raw.detach().cpu()
            sd["pin_src"] = a.pin_dir
        sd["killed"] = killed
        return sd


def _fam_of(text: str) -> str:
    from common.text import HIRA, KATA

    return "kana" if text in HIRA + KATA else "other"


def participation_ratio(m) -> float:
    """(Σσ²)² / Σσ⁴ of a matrix's singular values: 1 = rank one."""
    sv = torch.linalg.svdvals(m)
    pw = sv**2 / (sv**2).sum().clamp_min(1e-12)
    return float(1.0 / (pw**2).sum().clamp_min(1e-12))
