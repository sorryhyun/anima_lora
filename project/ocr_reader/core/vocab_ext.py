"""Two glyphs as their own tokens in PaddleOCR-VL-1.6's LM: ``♡`` and ``💢``.

The tokenizer spells ``♡`` in three byte-fallback tokens (``E2 99 A1``) and ``💢`` in
four (``F0 9F 92 A2``), while ``♥`` is one token (99252); a three-byte heart trains
badly against the one-token one. With ``♡`` a token of its own the two hearts are trained
apart (:func:`textnorm.training_target`).

The LM's embedding table and ``lm_head`` hold 103,424 rows against the tokenizer's
101,316, so the new tokens take spare rows (101316, 101317) and nothing is resized.
A row starts from its source token's row plus Gaussian noise at ``NOISE`` × that row's
element std (``♡`` from ``♥``; ``💢`` from ``×``), in both
the embedding and ``lm_head``. Training updates only these rows
(:class:`RowTrainer`); they are saved beside the adapter as :data:`FILE`, and a reader
that finds the file adds the tokens and loads the rows (:func:`load`). A checkpoint
without the file reads with the stock tokenizer.
"""

from __future__ import annotations

from pathlib import Path

import torch

TOKENS: tuple[tuple[str, str], ...] = (("♡", "♥"), ("💢", "×"))
"""(new token, the single token its rows start from)."""

FILE = "vocab.safetensors"
NOISE = 0.1
"""Init noise std as a share of the source row's element std."""


def _rows(model) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """The embedding and ``lm_head`` weights (``model`` may be a peft wrapper)."""
    return model.get_input_embeddings().weight, model.get_output_embeddings().weight


def add_tokens(tokenizer) -> list[int]:
    """Add :data:`TOKENS` to ``tokenizer`` (unnormalized, not special) and return
    their ids, in :data:`TOKENS` order. Already-added tokens keep their ids."""
    from tokenizers import AddedToken

    tokenizer.add_tokens(
        [AddedToken(t, normalized=False, special=False) for t, _ in TOKENS]
    )
    ids = [tokenizer.convert_tokens_to_ids(t) for t, _ in TOKENS]
    for (t, _), i in zip(TOKENS, ids):
        assert tokenizer.encode(t, add_special_tokens=False) == [i], t
    return ids


def extend(model, tokenizer, seed: int = 0) -> list[int]:
    """Add the tokens and initialise their rows from the source tokens' rows."""
    ids = add_tokens(tokenizer)
    emb, head = _rows(model)
    assert max(ids) < min(emb.shape[0], head.shape[0]), (ids, emb.shape, head.shape)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for (_, src), i in zip(TOKENS, ids):
            (j,) = tokenizer.encode(src, add_special_tokens=False)
            for w in (emb, head):
                row = w[j].float().cpu()
                noise = torch.randn(row.shape, generator=g) * row.std() * NOISE
                w[i] = (row + noise).to(w.dtype)
    return ids


def save(path: Path, model, ids: list[int]) -> None:
    from safetensors.torch import save_file

    emb, head = _rows(model)
    idx = torch.tensor(ids, device=emb.device)
    save_file(
        {
            "embed": emb.detach()[idx].to(torch.bfloat16).contiguous().cpu(),
            "lm_head": head.detach()[idx].to(torch.bfloat16).contiguous().cpu(),
        },
        str(path),
        metadata={"tokens": "\t".join(t for t, _ in TOKENS)},
    )


def load(adapter_dir: Path | str | None, model, tokenizer) -> list[int] | None:
    """Add the tokens and load their rows when ``adapter_dir`` holds :data:`FILE`;
    ``None`` (nothing changed) otherwise."""
    if not adapter_dir or not (Path(adapter_dir) / FILE).is_file():
        return None
    from safetensors import safe_open

    path = Path(adapter_dir) / FILE
    with safe_open(str(path), "pt") as f:
        tokens = f.metadata()["tokens"].split("\t")
        assert tokens == [t for t, _ in TOKENS], tokens
        rows = {k: f.get_tensor(k) for k in ("embed", "lm_head")}
    ids = add_tokens(tokenizer)
    emb, head = _rows(model)
    with torch.no_grad():
        for w, k in ((emb, "embed"), (head, "lm_head")):
            w[torch.tensor(ids, device=w.device)] = rows[k].to(w.device, w.dtype)
    print(f"vocab ext {path}: {tokens} at {ids}", flush=True)
    return ids


class RowTrainer:
    """fp32 masters of the new rows in the embedding and ``lm_head``: the weights take
    ``requires_grad``, a hook folds the new rows' gradient into the masters and drops
    the rest, and :meth:`write_back` copies the masters into the bf16 rows after each
    optimizer step."""

    def __init__(self, model, ids: list[int]):
        self.ids = ids
        self.pairs = []
        for w in _rows(model):
            idx = torch.tensor(ids, device=w.device)
            master = w.detach()[idx].float().clone().requires_grad_(True)
            w.requires_grad_(True)
            w.register_post_accumulate_grad_hook(self._fold(master, idx))
            self.pairs.append((w, idx, master))

    @staticmethod
    def _fold(master: torch.Tensor, idx: torch.Tensor):
        def hook(p: torch.Tensor) -> None:
            g = p.grad[idx].float()
            master.grad = g if master.grad is None else master.grad.add_(g)
            p.grad = None

        return hook

    @property
    def masters(self) -> list[torch.Tensor]:
        return [m for _, _, m in self.pairs]

    @property
    def weights(self) -> list[torch.nn.Parameter]:
        return [w for w, _, _ in self.pairs]

    @torch.no_grad()
    def write_back(self) -> None:
        for w, idx, m in self.pairs:
            w[idx] = m.to(w.dtype)
