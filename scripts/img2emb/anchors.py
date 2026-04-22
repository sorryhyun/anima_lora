"""Anchor-group spec + shared AnchoredResampler for img2emb.

One top-level YAML key = one anchor group = one classifier head over the
pooled encoder feature + one frozen prototype table. Groups can be mutex
(softmax+CE, 1 anchor slot/image holding softmax-weighted prototype mix) or
multi-label (sigmoid+BCE; 1 anchor slot per class, populated when positive).

See ``scripts/img2emb/anchors.yaml`` for the default spec.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from scripts.img2emb.resampler import PerceiverResampler

logger = logging.getLogger(__name__)


DEFAULT_PROTO_FILE = "phase2_class_prototypes.safetensors"


# --------------------------------------------------------------------------- dataclasses


@dataclass
class AnchorGroup:
    name: str
    classes: list[str]          # user-facing class names (no <unknown>)
    mutex: bool
    proto_file: str
    proto_key_prefix: str
    default_slot: int
    default_slots: list[int]    # len == len(classes); per-class slot fallback
    prototypes: torch.Tensor    # (n_classes + 1, D); last row is <unknown> zeros

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_rows(self) -> int:
        return self.n_classes + 1

    @property
    def unknown_idx(self) -> int:
        return self.n_classes


@dataclass
class AnchorSpec:
    groups: list[AnchorGroup]

    @property
    def names(self) -> list[str]:
        return [g.name for g in self.groups]

    def __iter__(self):
        return iter(self.groups)

    def by_name(self, name: str) -> AnchorGroup:
        for g in self.groups:
            if g.name == name:
                return g
        raise KeyError(name)

    def to_metadata(self) -> dict:
        return {
            "groups": [
                {
                    "name": g.name,
                    "classes": list(g.classes),
                    "mutex": g.mutex,
                    "proto_file": g.proto_file,
                    "proto_key_prefix": g.proto_key_prefix,
                    "default_slot": g.default_slot,
                    "default_slots": list(g.default_slots),
                }
                for g in self.groups
            ]
        }


# --------------------------------------------------------------------------- loader


def _expand_auto_classes(proto_sd: dict[str, torch.Tensor], prefix: str) -> list[str]:
    if prefix:
        return sorted(k[len(prefix):] for k in proto_sd if k.startswith(prefix))
    return sorted(proto_sd.keys())


def load_anchor_spec(yaml_path: Path, tag_slot_dir: Path) -> AnchorSpec:
    """Parse YAML + materialize prototype tables.

    Missing prototype rows are zero-filled with a warning. Each group gets an
    extra <unknown> zero row appended (index == ``n_classes``).
    """
    doc: dict[str, dict[str, Any]] = yaml.safe_load(Path(yaml_path).read_text())
    if not isinstance(doc, dict) or not doc:
        raise ValueError(f"anchors.yaml at {yaml_path} is empty or not a mapping")

    groups: list[AnchorGroup] = []
    proto_cache: dict[str, dict[str, torch.Tensor]] = {}

    for name, cfg in doc.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"anchor group '{name}' must be a mapping, got {type(cfg).__name__}")
        mutex = bool(cfg.get("mutex", True))
        prefix = str(cfg.get("proto_key_prefix", ""))
        proto_file = str(cfg.get("proto_file", DEFAULT_PROTO_FILE))

        if proto_file not in proto_cache:
            proto_cache[proto_file] = load_file(str(tag_slot_dir / proto_file))
        proto_sd = proto_cache[proto_file]

        classes_cfg = cfg.get("classes", [])
        if classes_cfg == "auto":
            classes = _expand_auto_classes(proto_sd, prefix)
        else:
            classes = list(classes_cfg)
        if not classes:
            raise ValueError(f"anchor group '{name}' has no classes")

        D = next(iter(proto_sd.values())).shape[0]
        rows: list[torch.Tensor] = []
        for cname in classes:
            key = f"{prefix}{cname}"
            if key in proto_sd:
                rows.append(proto_sd[key].float())
            else:
                logger.warning(
                    f"anchor group '{name}': missing prototype '{key}' in "
                    f"{proto_file}; using zero vec"
                )
                rows.append(torch.zeros(D))
        rows.append(torch.zeros(D))   # <unknown>
        protos = torch.stack(rows, dim=0)

        default_slot = int(cfg.get("default_slot", 0))
        default_slots_cfg = cfg.get("default_slots")
        if default_slots_cfg is None:
            default_slots = [default_slot] * len(classes)
        else:
            default_slots = [int(x) for x in default_slots_cfg]
            if len(default_slots) != len(classes):
                raise ValueError(
                    f"anchor group '{name}': default_slots has {len(default_slots)} "
                    f"entries but classes has {len(classes)}"
                )

        groups.append(
            AnchorGroup(
                name=name,
                classes=classes,
                mutex=mutex,
                proto_file=proto_file,
                proto_key_prefix=prefix,
                default_slot=default_slot,
                default_slots=default_slots,
                prototypes=protos,
            )
        )

    logger.info(
        "anchor spec: "
        + ", ".join(
            f"{g.name}({'mutex' if g.mutex else 'multi'},{g.n_classes}cls)"
            for g in groups
        )
    )
    return AnchorSpec(groups=groups)


# --------------------------------------------------------------------------- labels


def build_anchor_labels(
    spec: AnchorSpec,
    positions_path: Path,
    stems: list[str],
) -> dict[str, dict[str, torch.Tensor]]:
    """Per-group per-stem class/slot tensors from ``phase1_positions.json``.

    Mutex group payload::

        {"class": (N,) long -1=missing, "slot": (N,) long -1=missing}

    Multi-label group payload::

        {"labels": (N, C) float {0,1}, "slots": (N, C) long -1=class absent}

    Only ``v0`` entries are consulted (shuffle-fixed prefix). For mutex groups,
    when multiple classes are active in a caption the earliest-slot class wins
    (canonical booru ordering). For multi-label, each active class records its
    own earliest slot.
    """
    positions = json.loads(Path(positions_path).read_text())
    occ = positions["per_class_occurrences"]
    stem_to_row = {s: i for i, s in enumerate(stems)}
    N = len(stems)

    out: dict[str, dict[str, torch.Tensor]] = {}
    for g in spec.groups:
        class_to_idx = {c: i for i, c in enumerate(g.classes)}

        if g.mutex:
            cls_t = torch.full((N,), -1, dtype=torch.long)
            slot_t = torch.full((N,), -1, dtype=torch.long)
            for cname in g.classes:
                key = f"{g.proto_key_prefix}{cname}"
                cls_idx = class_to_idx[cname]
                for entry in occ.get(key, []):
                    stem, vi = entry[0], entry[1]
                    if vi != 0:
                        continue
                    row = stem_to_row.get(stem)
                    if row is None:
                        continue
                    first = int(entry[2])
                    cur = int(slot_t[row].item())
                    if cur == -1 or first < cur:
                        cls_t[row] = cls_idx
                        slot_t[row] = first
            out[g.name] = {"class": cls_t, "slot": slot_t}
            n_have = int((cls_t >= 0).sum().item())
            logger.info(f"anchor coverage [{g.name}]: {n_have}/{N}")
        else:
            lbl_t = torch.zeros((N, g.n_classes), dtype=torch.float32)
            slot_t = torch.full((N, g.n_classes), -1, dtype=torch.long)
            for cname in g.classes:
                key = f"{g.proto_key_prefix}{cname}"
                cls_idx = class_to_idx[cname]
                for entry in occ.get(key, []):
                    stem, vi = entry[0], entry[1]
                    if vi != 0:
                        continue
                    row = stem_to_row.get(stem)
                    if row is None:
                        continue
                    first = int(entry[2])
                    lbl_t[row, cls_idx] = 1.0
                    cur = int(slot_t[row, cls_idx].item())
                    if cur == -1 or first < cur:
                        slot_t[row, cls_idx] = first
            out[g.name] = {"labels": lbl_t, "slots": slot_t}
            any_pos = int((lbl_t.sum(dim=-1) > 0).sum().item())
            logger.info(f"anchor coverage [{g.name}]: {any_pos}/{N} (multi-label)")

    return out


def labels_to_flat_tensors(
    spec: AnchorSpec,
    labels: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Flatten ``{group: {field: tensor}}`` to ``{f"{group}_{field}": tensor}``
    for fast indexing. Field names follow :func:`build_anchor_labels`.
    """
    flat: dict[str, torch.Tensor] = {}
    for g in spec.groups:
        payload = labels[g.name]
        if g.mutex:
            flat[f"{g.name}_class"] = payload["class"]
            flat[f"{g.name}_slot"] = payload["slot"]
        else:
            flat[f"{g.name}_labels"] = payload["labels"]
            flat[f"{g.name}_slots"] = payload["slots"]
    return flat


def index_flat_labels(
    spec: AnchorSpec,
    flat: dict[str, torch.Tensor],
    idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Gather per-batch slice of flat label tensors using ``idx`` (B,) long."""
    out: dict[str, torch.Tensor] = {}
    for g in spec.groups:
        if g.mutex:
            out[f"{g.name}_class"] = flat[f"{g.name}_class"][idx]
            out[f"{g.name}_slot"] = flat[f"{g.name}_slot"][idx]
        else:
            out[f"{g.name}_labels"] = flat[f"{g.name}_labels"][idx]
            out[f"{g.name}_slots"] = flat[f"{g.name}_slots"][idx]
    return out


def gather_sample_labels(
    spec: AnchorSpec,
    flat: dict[str, torch.Tensor],
    full_idx: int,
) -> dict[str, torch.Tensor]:
    """Per-sample slice used inside ``Dataset.__getitem__``. Returns tensors
    (0-D for mutex scalars, 1-D for multi-label) that ``torch.stack`` can
    collate into (B,) / (B, C).
    """
    out: dict[str, torch.Tensor] = {}
    for g in spec.groups:
        if g.mutex:
            out[f"{g.name}_class"] = flat[f"{g.name}_class"][full_idx].clone()
            out[f"{g.name}_slot"] = flat[f"{g.name}_slot"][full_idx].clone()
        else:
            out[f"{g.name}_labels"] = flat[f"{g.name}_labels"][full_idx].clone()
            out[f"{g.name}_slots"] = flat[f"{g.name}_slots"][full_idx].clone()
    return out


def collate_anchor_batch(
    spec: AnchorSpec,
    items: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack per-sample label dicts into per-batch tensors."""
    out: dict[str, torch.Tensor] = {}
    for g in spec.groups:
        keys = (
            (f"{g.name}_class", f"{g.name}_slot")
            if g.mutex
            else (f"{g.name}_labels", f"{g.name}_slots")
        )
        for k in keys:
            out[k] = torch.stack([it[k] for it in items], dim=0)
    return out


# --------------------------------------------------------------------------- model


class AnchoredResampler(nn.Module):
    """PerceiverResampler + per-group classifier heads + frozen prototype tables.

    Forward returns ``{"pred": (B, S, D_out), "logits": {name: (B, n_rows)},
    "anchor_emb": {name: ...}}``. For mutex groups, ``anchor_emb[name]`` is
    ``(B, D_out)``. For multi-label groups it's ``(B, n_classes, D_out)``.

    Pre-injection: before the backbone runs, the predicted anchor embedding
    (or a teacher-forced GT prototype, see ``forward`` kwargs) is projected
    d_out → d_model via ``anchor_in_proj`` and written into the latent
    queries at the corresponding slot positions. This conditions the
    non-anchor slots on the classifier's decision so the backbone doesn't
    have to predict the anchor positions itself. Post-injection (replacing
    the output at anchor slots with the exact prototype mix) still happens
    at the call site via :func:`inject_spec_anchors`.
    """

    def __init__(
        self,
        spec: AnchorSpec,
        d_enc: int,
        d_pool: int,
        d_model: int = 1024,
        n_heads: int = 8,
        n_slots: int = 512,
        n_layers: int = 4,
        d_out: int = 1024,
    ):
        super().__init__()
        self.spec = spec
        self.backbone = PerceiverResampler(
            d_enc=d_enc, d_model=d_model, n_heads=n_heads,
            n_slots=n_slots, n_layers=n_layers, d_out=d_out,
        )
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(d_pool),
            nn.Linear(d_pool, d_model),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict(
            {g.name: nn.Linear(d_model, g.n_rows) for g in spec.groups}
        )
        # Identity-init when d_out == d_model so prototypes land at the same
        # scale as the existing N(0, 0.15) latent queries from step 0.
        self.anchor_in_proj = nn.Linear(d_out, d_model)
        with torch.no_grad():
            if d_out == d_model:
                self.anchor_in_proj.weight.copy_(torch.eye(d_model))
            else:
                nn.init.normal_(self.anchor_in_proj.weight, std=0.02)
            self.anchor_in_proj.bias.zero_()
        for g in spec.groups:
            self.register_buffer(f"{g.name}_protos", g.prototypes.float())

    @property
    def cls_param_prefixes(self) -> tuple[str, ...]:
        """Name prefixes for the 'classifier' param group (for split LRs)."""
        return ("pool_proj", "anchor_in_proj") + tuple(
            f"heads.{g.name}" for g in self.spec.groups
        )

    def _get_protos(self, name: str) -> torch.Tensor:
        return getattr(self, f"{name}_protos")

    def classify(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        p = self.pool_proj(pooled.float())
        return {g.name: self.heads[g.name](p) for g in self.spec.groups}

    def build_teacher_emb(
        self, batch_labels: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Hard prototype lookup from GT labels — the teacher signal.

        Mutex      : (B, D_out). Missing class (label -1) → <unknown> zero row.
        Multi-label: (B, C, D_out). Inactive classes → zero (matches the
                     sigmoid * proto formula at sigma=0).
        """
        out: dict[str, torch.Tensor] = {}
        for g in self.spec.groups:
            protos = self._get_protos(g.name)
            if g.mutex:
                cls = batch_labels[f"{g.name}_class"].to(protos.device)
                idx = torch.where(
                    cls >= 0, cls, torch.full_like(cls, g.unknown_idx)
                )
                out[g.name] = protos[idx]                       # (B, D_out)
            else:
                lbl = batch_labels[f"{g.name}_labels"].to(protos.device)
                # (B, C, D_out): proto row per class, gated by binary label
                out[g.name] = lbl.unsqueeze(-1) * protos[: g.n_classes].unsqueeze(0)
        return out

    def _anchor_emb_from_logits(
        self, logits: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        anchor_emb: dict[str, torch.Tensor] = {}
        for g in self.spec.groups:
            protos = self._get_protos(g.name)
            lg = logits[g.name]
            if g.mutex:
                anchor_emb[g.name] = torch.softmax(lg, dim=-1) @ protos
            else:
                sigma = torch.sigmoid(lg[..., : g.n_classes]).unsqueeze(-1)
                anchor_emb[g.name] = sigma * protos[: g.n_classes].unsqueeze(0)
        return anchor_emb

    def _prefill_queries(
        self,
        B: int,
        device: torch.device,
        anchor_emb: dict[str, torch.Tensor],
        slots: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        """Write projected anchor embeddings into the initial query buffer at
        the requested slot positions. Mutex groups always write (one slot per
        sample). Multi-label groups write only where ``masks[g.name]`` is True
        (and the per-class slot is valid)."""
        q = self.backbone.init_queries(B)
        for g in self.spec.groups:
            emb = anchor_emb[g.name]
            slot = slots[g.name].to(device)
            if g.mutex:
                # emb (B, D_out) → (B, d_model); slot (B,)
                proj = self.anchor_in_proj(emb)
                valid = slot >= 0
                if not valid.any():
                    continue
                b_idx = torch.arange(B, device=device)[valid]
                s_idx = slot[valid]
                q[b_idx, s_idx] = proj[valid].to(q.dtype)
            else:
                mask = masks.get(g.name)
                valid = slot >= 0
                if mask is not None:
                    valid = valid & mask.to(device)
                if not valid.any():
                    continue
                b_rows, c_cols = torch.nonzero(valid, as_tuple=True)
                # emb (B, C, D_out) → only the chosen entries get projected
                values = self.anchor_in_proj(emb[b_rows, c_cols])
                q[b_rows, slot[b_rows, c_cols]] = values.to(q.dtype)
        return q

    def forward(
        self,
        tokens: torch.Tensor,
        pooled: torch.Tensor,
        *,
        teacher_labels: dict[str, torch.Tensor] | None = None,
        tf_ratio: float | torch.Tensor = 0.0,
    ) -> dict[str, Any]:
        """Run classifier, build pre-fill (predicted or teacher-forced mix),
        write it into the latent queries, then run the backbone.

        Teacher forcing (only active when ``teacher_labels`` is given and
        ``tf_ratio > 0``): each sample independently uses the GT prototype
        with probability ``tf_ratio`` (per-sample Bernoulli) and the
        classifier's predicted soft mix otherwise. ``tf_ratio`` may be a
        scalar or a per-sample (B,) tensor. Anneal from 1.0 → 0.0 across
        training so the model learns to trust its own classifier by the end.

        ``teacher_labels`` is the flat label dict from ``collate_anchor_batch``;
        it carries both the GT class indices (for prototype lookup) and the
        GT slot positions. Without it, every sample uses the predicted soft
        mix at the per-group ``default_slot`` / ``default_slots``.
        """
        B = tokens.shape[0]
        device = tokens.device

        logits = self.classify(pooled)
        anchor_emb_pred = self._anchor_emb_from_logits(logits)

        # --- decide per-sample teacher mask
        if teacher_labels is not None:
            if isinstance(tf_ratio, (int, float)):
                tf_ratio_t = torch.full((B,), float(tf_ratio), device=device)
            else:
                tf_ratio_t = tf_ratio.to(device)
                if tf_ratio_t.ndim == 0:
                    tf_ratio_t = tf_ratio_t.expand(B)
            use_teacher = torch.rand(B, device=device) < tf_ratio_t   # (B,) bool
            teacher_emb = self.build_teacher_emb(teacher_labels)
        else:
            use_teacher = torch.zeros(B, dtype=torch.bool, device=device)
            teacher_emb = None

        # --- build per-group prefill emb / slots / masks
        prefill_emb: dict[str, torch.Tensor] = {}
        prefill_slots: dict[str, torch.Tensor] = {}
        prefill_masks: dict[str, torch.Tensor | None] = {}
        any_teacher = bool(use_teacher.any().item()) if teacher_labels is not None else False

        for g in self.spec.groups:
            pred_emb = anchor_emb_pred[g.name]
            if g.mutex:
                # default slot for predicted path
                default_slots = torch.full(
                    (B,), int(g.default_slot), dtype=torch.long, device=device
                )
                if any_teacher:
                    t_slot = teacher_labels[f"{g.name}_slot"].to(device)
                    # teacher slot may be -1 (label missing) — fall back to default
                    t_slot = torch.where(t_slot >= 0, t_slot, default_slots)
                    prefill_slots[g.name] = torch.where(
                        use_teacher, t_slot, default_slots
                    )
                    t_emb = teacher_emb[g.name]
                    prefill_emb[g.name] = torch.where(
                        use_teacher.unsqueeze(-1), t_emb, pred_emb
                    )
                else:
                    prefill_slots[g.name] = default_slots
                    prefill_emb[g.name] = pred_emb
                prefill_masks[g.name] = None
            else:
                # multi-label
                C = g.n_classes
                default_slots_row = torch.tensor(
                    g.default_slots, dtype=torch.long, device=device
                ).unsqueeze(0).expand(B, -1).contiguous()        # (B, C)
                # predicted active set: sigmoid > 0.5
                pred_mask = torch.sigmoid(logits[g.name][..., :C]) > 0.5  # (B, C)
                if any_teacher:
                    t_slots = teacher_labels[f"{g.name}_slots"].to(device)   # (B, C)
                    t_slots_eff = torch.where(t_slots >= 0, t_slots, default_slots_row)
                    t_mask = teacher_labels[f"{g.name}_labels"].to(device) > 0.5  # (B, C)
                    use_t = use_teacher.unsqueeze(-1)        # (B, 1)
                    prefill_slots[g.name] = torch.where(use_t, t_slots_eff, default_slots_row)
                    prefill_masks[g.name] = torch.where(use_t, t_mask, pred_mask)
                    t_emb = teacher_emb[g.name]
                    prefill_emb[g.name] = torch.where(
                        use_t.unsqueeze(-1), t_emb, pred_emb
                    )
                else:
                    prefill_slots[g.name] = default_slots_row
                    prefill_masks[g.name] = pred_mask
                    prefill_emb[g.name] = pred_emb

        q = self._prefill_queries(B, device, prefill_emb, prefill_slots, prefill_masks)
        pred = self.backbone(tokens, queries=q)
        return {"pred": pred, "logits": logits, "anchor_emb": anchor_emb_pred}


# --------------------------------------------------------------------------- injection


def inject_anchors(
    pred: torch.Tensor,
    anchor_emb: torch.Tensor,
    slots: torch.Tensor,
    *,
    mutex: bool,
    mode: str = "replace",
    mask: torch.Tensor | None = None,
) -> None:
    """Write/add anchor embeddings into the matching slot(s) of ``pred`` in-place.

    mutex=True : ``anchor_emb`` (B, D), ``slots`` (B,). Rows with slot >= 0 write.
    mutex=False: ``anchor_emb`` (B, C, D), ``slots`` (B, C), optional ``mask``
                 (B, C). Entries write when slot >= 0 and (mask is True or mask None).
    """
    device = pred.device
    if mutex:
        valid = slots >= 0
        if not valid.any():
            return
        b_idx = torch.arange(pred.shape[0], device=device)[valid]
        s_idx = slots[valid].to(device=device)
        values = anchor_emb[valid]
    else:
        valid = slots >= 0
        if mask is not None:
            valid = valid & mask.to(device=slots.device)
        if not valid.any():
            return
        b_rows, c_cols = torch.nonzero(valid, as_tuple=True)
        b_idx = b_rows.to(device=device)
        s_idx = slots[b_rows, c_cols].to(device=device)
        values = anchor_emb[b_rows, c_cols]

    if mode == "replace":
        pred[b_idx, s_idx] = values
    elif mode == "residual":
        pred[b_idx, s_idx] = pred[b_idx, s_idx] + values
    else:
        raise ValueError(f"unknown anchor mode '{mode}'")


def inject_spec_anchors(
    pred: torch.Tensor,
    spec: AnchorSpec,
    anchor_emb: dict[str, torch.Tensor],
    batch_labels: dict[str, torch.Tensor],
    *,
    mode: str = "replace",
) -> None:
    """Loop helper: inject every group using the flat ``batch_labels`` dict."""
    for g in spec.groups:
        if g.mutex:
            inject_anchors(
                pred, anchor_emb[g.name],
                batch_labels[f"{g.name}_slot"],
                mutex=True, mode=mode,
            )
        else:
            labels = batch_labels[f"{g.name}_labels"] > 0.5
            inject_anchors(
                pred, anchor_emb[g.name],
                batch_labels[f"{g.name}_slots"],
                mutex=False, mode=mode, mask=labels,
            )


# --------------------------------------------------------------------------- loss


def aux_cls_loss(
    spec: AnchorSpec,
    logits: dict[str, torch.Tensor],
    batch_labels: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean classifier loss across groups. Mutex→CE (ignoring -1), multi→BCE."""
    losses: list[torch.Tensor] = []
    accs: dict[str, float] = {}
    device = next(iter(logits.values())).device
    for g in spec.groups:
        lg = logits[g.name]
        if g.mutex:
            tg = batch_labels[f"{g.name}_class"]
            m = tg >= 0
            if not m.any():
                continue
            tg_d = tg[m].to(lg.device)
            lg_d = lg[m]
            losses.append(F.cross_entropy(lg_d, tg_d))
            with torch.no_grad():
                accs[g.name] = (lg_d.argmax(dim=-1) == tg_d).float().mean().item()
        else:
            tg = batch_labels[f"{g.name}_labels"].to(lg.device)
            lg_d = lg[..., : g.n_classes]
            losses.append(F.binary_cross_entropy_with_logits(lg_d, tg))
            with torch.no_grad():
                pred_pos = (torch.sigmoid(lg_d) > 0.5).float()
                accs[g.name] = (pred_pos == tg).float().mean().item()
    if not losses:
        return torch.tensor(0.0, device=device), accs
    total = torch.stack(losses).mean()
    return total, accs
