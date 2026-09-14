"""What an arm trains, and how it becomes the ``ExtDelta`` table.

  rows          free per-row delta on the ext rows the training captions touch
  rows_adapter  that + a LoRA on every Linear of ``llm_adapter.blocks``
  encoder       Δ_r = g(glyph_r) [+ f_r free residual on trained rows] (W2d)

Every arm saves the ExtDelta format, so eval / native / classify run unchanged.
"""

from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F

from .encoder import GlyphEncoder, glyph_bank, glyph_batch, reference_batch, row_texts
from .hooks import AdapterLoRA, ExtDelta
from .render import find_fonts


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
        if a.arm == "encoder":
            self._init_encoder(train_ext, ev_ext, tok, pack, anima, dim)
        else:
            self.delta = ExtDelta(anima, train_ext, dim, device, self.row_scale)
            self.params = [{"params": [self.delta.raw], "lr": a.lr_rows}]
        if a.arm == "rows_adapter":
            self.lora = AdapterLoRA(anima, a.adapter_rank, device)
            self.params.append({"params": list(self.lora.params), "lr": a.lr_adapter})
            print(
                f"adapter LoRA r{a.adapter_rank} on {len(self.lora.patched)} Linears",
                flush=True,
            )

    # -- setup ---------------------------------------------------------------

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
            # history.md Run 1b amended: the FM gradient's sign-consistent
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
        return loss, decor_val

    def after_step(self):
        if self.enc is not None:
            self.enc.clamp_common()  # projected descent on c: no creep

    # -- logging -------------------------------------------------------------

    def log_record(self, step, loss_fm, loss, decor_val, t0) -> dict:
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
        }
        if self.enc is not None:
            self._log_encoder(rec, loss, decor_val, dn_held)
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
        """history.md W2d kill rules: identity not moving, or a row walking
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
        sd["killed"] = killed
        return sd


def participation_ratio(m) -> float:
    """(Σσ²)² / Σσ⁴ of a matrix's singular values: 1 = rank one."""
    sv = torch.linalg.svdvals(m)
    pw = sv**2 / (sv**2).sum().clamp_min(1e-12)
    return float(1.0 / (pw**2).sum().clamp_min(1e-12))
