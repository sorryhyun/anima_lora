"""The trainable hooks on the frozen model: ext-row delta and adapter LoRA."""

from __future__ import annotations

import math

import torch


class ExtDelta:
    """Trainable delta on the ext rows of ``llm_adapter.embed`` (mirrors
    EasyControlNetwork._hook_ext_rows: prepended pre-hook sees raw ids, forward
    hook runs after the pack's and adds the delta on top of the pack rows).

    ``raw`` is in row-norm units (× ``row_scale``, the mean pack-row norm);
    ``scale`` 0 restores the pack rows exactly."""

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
            d[known] = self.raw[loc[known]] * self.row_scale
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
        return {
            "ext_ids": self.ext_ids,
            "raw": self.raw.detach().cpu(),
            "row_scale": self.row_scale,
        }

    def load(self, sd):
        assert sd["ext_ids"] == self.ext_ids
        self.raw.data.copy_(sd["raw"].to(self.raw.device))
        self.row_scale = sd["row_scale"]


class AdapterLoRA:
    """Rank-r LoRA on every Linear of ``llm_adapter.blocks`` (monkeypatched
    forward, B zero-init). ``scale`` 0 restores the stock adapter."""

    def __init__(self, anima, rank: int, device):
        self.params = torch.nn.ParameterList()
        self.scale = 1.0
        self.patched = []
        for name, m in anima.llm_adapter.blocks.named_modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            a = torch.nn.Parameter(
                torch.randn(rank, m.in_features, device=device)
                / math.sqrt(m.in_features)
            )
            b = torch.nn.Parameter(torch.zeros(m.out_features, rank, device=device))
            self.params.append(a)
            self.params.append(b)
            orig = m.forward
            alpha = 1.0 / rank

            def fwd(x, orig=orig, a=a, b=b):
                y = orig(x)
                if self.scale == 0.0:
                    return y
                h = (x.to(a.dtype) @ a.t()) @ b.t()
                return y + (h * (alpha * self.scale)).to(y.dtype)

            m.forward = fwd
            self.patched.append(name)

    def state_dict(self):
        return {
            "params": [p.detach().cpu() for p in self.params],
            "names": self.patched,
        }

    def load(self, sd):
        for p, q in zip(self.params, sd["params"]):
            p.data.copy_(q.to(p.device))
