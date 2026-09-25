"""The trainable hook on the frozen model: the ext-row delta."""

from __future__ import annotations

import torch


class ExtDelta:
    """Trainable delta on the ext rows of ``llm_adapter.embed`` (mirrors
    EasyControlNetwork._hook_ext_rows: prepended pre-hook sees raw ids, forward
    hook runs after the pack's and adds the delta on top of the pack rows).

    ``raw`` is in row-norm units (× ``row_scale``, the mean pack-row norm);
    ``scale`` 0 restores the pack rows exactly. ``common`` (S0, plan_synth):
    an optional ``(dim,)`` vector in the same units added to every *trained*
    row on top of ``raw`` for the current batch — the per-source layout
    vector ``c_flat``; the train stage sets it per batch, eval leaves it
    ``None``."""

    def __init__(self, anima, ext_ids, dim, device, row_scale: float):
        from library.anima.ext_vocab import T5_TABLE_SIZE

        self.T = T5_TABLE_SIZE
        self.ext_ids = sorted(int(i) for i in ext_ids)
        self.index = {e: i for i, e in enumerate(self.ext_ids)}
        self.raw = torch.nn.Parameter(
            torch.zeros(len(self.ext_ids), dim, device=device)
        )
        self.row_scale = row_scale
        self.scale = 1.0
        self.common = None
        # ``pinned`` (rows, dim), fixed, added to every trained row on every
        # item: the inherited shared direction a_r · m̂ (``--pin_dir``); the
        # trainable ``raw`` is then the per-row residual only. Saved tables
        # fold it into ``raw`` so eval / native / ``from_state`` see the full delta.
        self.pinned = None
        self.state: dict = {}
        embed = anima.llm_adapter.embed
        lut = torch.full((max(self.ext_ids) + 2,), -1, dtype=torch.long)
        for e, i in self.index.items():
            lut[e] = i
        self.lut = lut.to(device)

        def pre(module, args):
            self.state.pop("mask", None)
            if args and torch.is_tensor(args[0]):
                mask = args[0] >= self.T
                if bool(mask.any()):
                    self.state["mask"] = mask
                    self.state["ext"] = args[0][mask] - self.T

        def post(module, args, output):
            mask = self.state.pop("mask", None)
            if mask is None or self.scale == 0.0:
                self.state.pop("ext", None)
                return None
            ext = self.state.pop("ext")
            ext = torch.clamp(ext, max=self.lut.numel() - 1)
            loc = self.lut[ext]
            known = loc >= 0
            d = torch.zeros(
                ext.numel(),
                self.raw.shape[1],
                device=output.device,
                dtype=self.raw.dtype,
            )
            rows = self.raw[loc[known]]
            if self.pinned is not None:
                rows = rows + self.pinned[loc[known]].to(rows.dtype)
            if self.common is not None:
                rows = rows + self.common.to(rows.dtype)
            d[known] = rows * self.row_scale
            out = output.clone()
            out[mask] = out[mask] + (d * self.scale).to(out.dtype)
            return out

        self.handles = [
            embed.register_forward_pre_hook(pre, prepend=True),
            embed.register_forward_hook(post),
        ]

    @classmethod
    def from_state(cls, anima, sd: dict, device) -> ExtDelta:
        """Hook a saved delta (``trained.pt['delta']``) onto ``anima``."""
        delta = cls(anima, sd["ext_ids"], sd["raw"].shape[1], device, sd["row_scale"])
        delta.load(sd)
        return delta

    def state_dict(self):
        raw = self.raw.detach().cpu()
        if self.pinned is not None:
            raw = raw + self.pinned.detach().cpu().to(raw.dtype)
        return {
            "ext_ids": self.ext_ids,
            "raw": raw,
            "row_scale": self.row_scale,
        }

    def load(self, sd):
        assert sd["ext_ids"] == self.ext_ids
        self.raw.data.copy_(sd["raw"].to(self.raw.device))
        self.row_scale = sd["row_scale"]
