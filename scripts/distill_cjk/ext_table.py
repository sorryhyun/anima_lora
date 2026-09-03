"""Trainable parameterizations of the extended T5 query rows + a split embedding.

The Phase-1 probe measured the zero-shot ext arm at cos 0.05 vs EN but
**discrimination 0.184** — already inside the Phase-2c gate. The pathway is
alive and prompt-specific, it is pointing the wrong way. That is the signature
of a *systematic* error in the anchor map (which was fit on non-CJK anchors),
not of 58,968 independent per-row errors — hence the ladder:

``global``      2-i-a — a low-rank + per-dim-diagonal + scalar-gain correction
                shared by every ext row. ~1-2M params, zero per-row freedom, so
                it **generalizes to the 95% of rows the corpus never visits**
                and subsumes the build-time ×1.66 norm rescale as a learned
                gain you can read back off.
``row``         per-row residuals only (the plan's original 2-i).
``global_row``  2-i-b — both, with residuals gated on a visit-count floor.

Whatever the mode, the original 32,128 rows are a frozen buffer and the EN
tokenize path never reaches the trainable parameters, so EN conditioning is
bit-identical by construction rather than by test (the test still exists).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

T5_TABLE_SIZE = 32128


class ExtTable(nn.Module):
    """The extended rows as a function of their zero-shot init.

    ``rows(row_ids)`` computes only the requested rows — materializing all
    58,968 × 1024 every step would cost a 240 MB tensor and a 60M-row matmul
    for the ~100 distinct rows a batch actually touches.
    """

    def __init__(
        self,
        init: torch.Tensor,
        *,
        mode: str = "global",
        rank: int = 64,
        tunable_rows: torch.Tensor | None = None,
        freeze_diag: bool = False,
    ) -> None:
        super().__init__()
        if mode not in ("global", "row", "global_row"):
            raise ValueError(f"unknown param mode {mode!r}")
        self.mode = mode
        self.freeze_diag = bool(freeze_diag)
        self.register_buffer("init", init.float(), persistent=False)
        n_rows, dim = init.shape

        if mode in ("global", "global_row"):
            # LoRA-style init: down random, up zero ⇒ exactly `init` at step 0.
            self.down = nn.Parameter(torch.randn(dim, rank) * (dim**-0.5))
            self.up = nn.Parameter(torch.zeros(rank, dim))
            # ``freeze_diag``: keep the per-dim diagonal at identity (a
            # buffer, never trained) so the shared map is low-rank + gain
            # only. The learned diagonal is what discards key directions the
            # teacher does not reward (spread_probe: init PR 234 -> 54).
            if freeze_diag:
                self.register_buffer("log_diag", torch.zeros(dim))
            else:
                self.log_diag = nn.Parameter(torch.zeros(dim))
            self.log_gain = nn.Parameter(torch.zeros(()))

        if mode in ("row", "global_row"):
            if tunable_rows is None:
                tunable_rows = torch.arange(n_rows)
            tunable_rows = tunable_rows.long()
            # slot[row] = index into `residual`, or -1 for "frozen at init".
            slot = torch.full((n_rows,), -1, dtype=torch.long)
            slot[tunable_rows] = torch.arange(len(tunable_rows))
            self.register_buffer("slot", slot, persistent=False)
            self.register_buffer("tunable_rows", tunable_rows, persistent=False)
            self.residual = nn.Parameter(torch.zeros(len(tunable_rows), dim))

    @property
    def has_global(self) -> bool:
        return self.mode in ("global", "global_row")

    @property
    def has_rows(self) -> bool:
        return self.mode in ("row", "global_row")

    def n_tunable_rows(self) -> int:
        return int(self.residual.shape[0]) if self.has_rows else 0

    def rows(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Corrected embeddings for ``row_ids`` (indices into the ext table)."""
        base = self.init[row_ids]
        out = base
        if self.has_global:
            out = out + (out @ self.down) @ self.up
            out = out * torch.exp(self.log_diag) * torch.exp(self.log_gain)
        if self.has_rows:
            slot = self.slot[row_ids]
            hit = slot >= 0
            if hit.any():
                add = torch.zeros_like(out)
                add[hit] = self.residual[slot[hit]]
                out = out + add
        return out

    @torch.no_grad()
    def materialize(self, chunk: int = 8192) -> torch.Tensor:
        """The full corrected table — for export and for the eval encoder."""
        n = self.init.shape[0]
        out = torch.empty_like(self.init)
        idx = torch.arange(n, device=self.init.device)
        for i in range(0, n, chunk):
            sl = idx[i : i + chunk]
            out[sl] = self.rows(sl)
        return out

    @torch.no_grad()
    def provenance(self, visits: torch.Tensor | None, span_floor: int = 0) -> list[str]:
        """Per-row ``tuned`` / ``mapped`` / ``mapped-unseen`` / ``zero-shot``.

        ``mapped`` is the tier the plan did not have: a row the corpus never
        visited still moves, because the global correction applies to it.
        ``mapped-unseen`` (plan_zh2 U1) splits that tier by evidence: with a
        span visit floor, a row below the floor never supervised the map — it
        was carried along, not trained on — and a reader of the pack can tell
        the two apart. Without a floor every mapped row is plain ``mapped``.
        """
        n = self.init.shape[0]
        dev = self.init.device
        if visits is None:
            visits = torch.zeros(n, dtype=torch.long, device=dev)
        visits = visits.to(dev)
        seen = visits > 0
        below = visits < span_floor if span_floor > 0 else torch.zeros_like(seen)
        tuned = torch.zeros(n, dtype=torch.bool, device=dev)
        if self.has_rows:
            tuned[self.tunable_rows] = True
        tags = []
        for i in range(n):
            if bool(tuned[i]):
                tags.append("tuned")
            elif self.has_global:
                tags.append("mapped-unseen" if bool(below[i]) else "mapped")
            elif bool(seen[i]):
                tags.append("mapped")
            else:
                tags.append("zero-shot")
        return tags


class SplitEmbedding(nn.Module):
    """``nn.Embedding`` replacement: frozen base rows + trainable ext rows.

    Do NOT fuse the two tables. ``nn.Embedding.from_pretrained`` freezes by
    default (that is what the bench does at inference), and a fused table with
    a partial-gradient hack would put the pretrained 32,128 rows one bug away
    from moving — the EN-bit-exactness guarantee is the whole reason this line
    is a sidecar and not a checkpoint fork.
    """

    def __init__(self, base_weight: torch.Tensor, table: ExtTable) -> None:
        super().__init__()
        self.register_buffer("base_weight", base_weight, persistent=False)
        self.table = table

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        base = F.embedding(ids.clamp(max=T5_TABLE_SIZE - 1), self.base_weight)
        ext = ids >= T5_TABLE_SIZE
        if not bool(ext.any()):
            return base
        out = base.clone()
        rows = self.table.rows(ids[ext] - T5_TABLE_SIZE)
        out[ext] = rows.to(out.dtype)
        return out


def attach(adapter: nn.Module, table: ExtTable) -> SplitEmbedding:
    """Swap the adapter's frozen embed for a :class:`SplitEmbedding`.

    The adapter itself stays frozen — ``requires_grad_(False)`` is applied to
    everything else, so the only leaves in the graph are the ext parameters.
    """
    base = adapter.embed.weight.data
    adapter.requires_grad_(False)
    split = SplitEmbedding(base, table)
    adapter.embed = split
    return split


def visit_counts(rows_total: int, pairs_ids) -> torch.Tensor:
    """Ext-row visit histogram over an iterable of per-pair id tensors."""
    counts = torch.zeros(rows_total, dtype=torch.long)
    for ids in pairs_ids:
        ids = torch.as_tensor(ids)
        ext = ids[ids >= T5_TABLE_SIZE] - T5_TABLE_SIZE
        if ext.numel():
            counts.index_add_(0, ext.long(), torch.ones_like(ext, dtype=torch.long))
    return counts
