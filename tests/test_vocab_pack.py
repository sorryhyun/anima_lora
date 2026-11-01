"""CJK vocab pack as a shipped text-encoder asset (``library.anima.vocab_pack``).

Invariants:

* **Off means bit-exact.** With ``vocab_pack`` unset the strategy factory returns
  the stock tokenizer class and no hooks are installed — the whole path must be
  inert for EN-only users.
* **The hook pair never touches the state dict.** ``llm_adapter.embed`` keeps its
  32128 rows; ext ids resolve to pack rows through hooks, base ids are unchanged.
* **TE caches carry the pack digest, and mismatches warn once.** A cache encoded
  through a different pack (or none) logs a warning instead of silently
  training on the wrong rows.
* **EN captions tokenize identically through the pack** (G1 of the CJK line,
  lifted to the strategy level) — needs the real tokenizers, skipped otherwise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from library.anima import vocab_pack as vp
from library.anima.ext_vocab import T5_TABLE_SIZE, T5_UNK_ID

REPO = Path(__file__).resolve().parents[1]
DIM = 16
ROWS = 3


@pytest.fixture
def synthetic_pack(tmp_path: Path) -> Path:
    """A tiny pack pair on disk: 3 rows, two char-routed rows."""
    prefix = tmp_path / "packs" / "tiny_pack"
    prefix.parent.mkdir(parents=True)
    table = torch.arange(ROWS * DIM, dtype=torch.float32).reshape(ROWS, DIM) + 1000.0
    save_file({"ext_embed": table}, str(prefix.with_suffix(".safetensors")))
    mapping = {
        "rows": ROWS,
        "qwen": {},
        "char": {"猫": 0, "耳": 1},
        "training": {"label": "tiny"},
    }
    prefix.with_suffix(".json").write_text(json.dumps(mapping), encoding="utf-8")
    return prefix


@pytest.fixture(autouse=True)
def _fresh_caches():
    vp._LOADED.clear()
    vp._warned_cache_stamps.clear()
    yield
    vp._LOADED.clear()
    vp._warned_cache_stamps.clear()


# --- resolution ----------------------------------------------------------------


def test_the_shipped_default_is_fetched_when_missing(tmp_path: Path, monkeypatch):
    """base.toml enables the pack and `make update` overwrites base.toml, so a
    pre-v2 checkout must not hard-fail at first load: the shipped default (and
    only it) is fetched through the catalog row, then resolved as usual."""
    from library import downloads as DL

    monkeypatch.setenv("ANIMA_HOME", str(tmp_path))
    prefix = DL.default_vocab_pack_prefix()
    assert not prefix.with_suffix(".json").exists()
    fetched: list[str] = []

    def _fake_fetch(asset, log=print, force=False):
        fetched.append(asset.id)
        asset.dest.mkdir(parents=True, exist_ok=True)
        for name in asset.files:
            (asset.dest / name).write_bytes(b"x")
        return True

    monkeypatch.setattr(DL, "fetch", _fake_fetch)
    assert vp.resolve_pack_prefix(vp.DEFAULT_PACK_PREFIX) == prefix
    assert fetched == ["vocab_pack"]
    # Present now: no second fetch.
    assert vp.resolve_pack_prefix(vp.DEFAULT_PACK_PREFIX) == prefix
    assert fetched == ["vocab_pack"]

    # A custom path is never auto-fetched — the old error, with the hint.
    with pytest.raises(FileNotFoundError, match="download-vocab-pack"):
        vp.resolve_pack_prefix(tmp_path / "custom" / "pack")
    assert fetched == ["vocab_pack"]


def test_default_pack_reads_env_then_base_toml(monkeypatch):
    from library.env import default_checkpoints

    monkeypatch.setenv("ANIMA_VOCAB_PACK", "some/pack")
    assert default_checkpoints().vocab_pack == "some/pack"
    assert vp.default_vocab_pack() == "some/pack"
    monkeypatch.delenv("ANIMA_VOCAB_PACK")
    # v2: base.toml ships the CJK pack enabled — the value is the catalog's
    # default prefix, so `make download-models` and the loader agree on it.
    from library import downloads as DL

    assert (
        default_checkpoints().vocab_pack
        == f"models/{DL.VOCAB_PACK_DIR}/{DL.VOCAB_PACK_STEM}"
    )
    assert vp.default_vocab_pack() == vp.DEFAULT_PACK_PREFIX


def test_resolve_active_pack_precedence(synthetic_pack: Path, monkeypatch):
    import argparse

    monkeypatch.setenv("ANIMA_VOCAB_PACK", str(synthetic_pack))
    # config default applies when the flag is unset
    assert vp.resolve_active_pack(argparse.Namespace()).name == "tiny_pack"
    # explicit "" turns it off; --no_vocab_pack wins over everything
    assert vp.resolve_active_pack(argparse.Namespace(vocab_pack="")) is None
    assert (
        vp.resolve_active_pack(
            argparse.Namespace(vocab_pack=str(synthetic_pack), no_vocab_pack=True)
        )
        is None
    )


# --- patch point 2: the embed hooks ----------------------------------------------


def _adapter():
    from library.anima.models import LLMAdapter

    torch.manual_seed(0)
    return LLMAdapter(source_dim=DIM, target_dim=DIM, model_dim=DIM, num_layers=1)


@pytest.mark.core
def test_hooks_substitute_ext_rows_and_leave_the_state_dict_alone(synthetic_pack: Path):
    pack = vp.load_vocab_pack(synthetic_pack)
    adapter = _adapter()
    keys_before = list(adapter.state_dict().keys())
    ids = torch.tensor([[5, T5_TABLE_SIZE + 2, 7, T5_TABLE_SIZE + 0, 0]])
    stock = adapter.embed(ids.clamp(max=T5_TABLE_SIZE - 1)).clone()

    vp.attach_vocab_pack(adapter, pack)
    out = adapter.embed(ids)

    assert out.shape == (1, 5, DIM)
    torch.testing.assert_close(out[0, 1], pack.table[2])
    torch.testing.assert_close(out[0, 3], pack.table[0])
    for pos in (0, 2, 4):  # base ids untouched
        torch.testing.assert_close(out[0, pos], stock[0, pos])
    assert list(adapter.state_dict().keys()) == keys_before
    assert adapter.embed.weight.shape[0] == T5_TABLE_SIZE
    assert vp.attached_pack_digest(adapter) == pack.digest


def test_attach_is_idempotent_and_detach_restores(synthetic_pack: Path):
    pack = vp.load_vocab_pack(synthetic_pack)
    adapter = _adapter()
    vp.attach_vocab_pack(adapter, pack)
    vp.attach_vocab_pack(adapter, pack)  # same digest: no second hook pair
    assert len(adapter.embed._forward_pre_hooks) == 1
    assert len(adapter.embed._forward_hooks) == 1
    vp.detach_vocab_pack(adapter)
    assert vp.attached_pack_digest(adapter) is None
    assert len(adapter.embed._forward_pre_hooks) == 0
    ids = torch.tensor([[T5_TABLE_SIZE + 1]])
    with pytest.raises(IndexError):  # stock table again: ext ids are out of range
        adapter.embed(ids)
    # the <unk> row is what the clamp would have produced
    torch.testing.assert_close(
        adapter.embed(torch.tensor([[T5_UNK_ID]])),
        adapter.embed.weight[T5_UNK_ID][None, None],
    )


# --- merged checkpoints --------------------------------------------------------


def _fake_dit(path: Path) -> dict:
    sd = {
        "net.llm_adapter.embed.weight": torch.randn(8, DIM).to(torch.bfloat16),
        "net.blocks.0.attn.q.weight": torch.randn(DIM, DIM).to(torch.bfloat16),
    }
    save_file(sd, str(path), metadata={"ss_num_blocks": "1"})
    return sd


@pytest.mark.core
def test_merge_keeps_the_dit_and_carries_the_pack(synthetic_pack: Path, tmp_path: Path):
    dit = tmp_path / "dit.safetensors"
    base = _fake_dit(dit)
    out = tmp_path / "merged.safetensors"
    vp.merge_pack_into_dit(dit, synthetic_pack, out)

    with safe_open(str(out), framework="pt") as f:
        md = f.metadata()
        assert set(f.keys()) == set(base) | {"net.llm_adapter.ext_embed.weight"}
        for k, t in base.items():
            assert torch.equal(f.get_tensor(k), t)
        ext = f.get_tensor("net.llm_adapter.ext_embed.weight")
    assert md["ss_num_blocks"] == "1"
    assert md[vp.MERGED_META_MAPPING] == synthetic_pack.with_suffix(".json").read_text(
        encoding="utf-8"
    )
    raw = load_file(str(synthetic_pack.with_suffix(".safetensors")))["ext_embed"]
    assert torch.equal(ext, raw)

    with pytest.raises(ValueError, match="already carries"):
        vp.merge_pack_into_dit(out, synthetic_pack, tmp_path / "twice.safetensors")


@pytest.mark.core
def test_cache_stamp_warns_once_per_mismatch_kind(synthetic_pack: Path, caplog):
    pack = vp.load_vocab_pack(synthetic_pack)
    with caplog.at_level(logging.WARNING, logger=vp.__name__):
        vp.check_cache_stamp(None, "a.safetensors", None)  # stock ↔ stock
        vp.check_cache_stamp(pack.cache_metadata(), "b.safetensors", pack)  # same pack
        assert not caplog.records
        vp.check_cache_stamp(None, "c.safetensors", pack)
        vp.check_cache_stamp(None, "d.safetensors", pack)  # same kind: not repeated
        assert caplog.text.count("preprocess-te") == 1
        assert "c.safetensors" in caplog.text and "d.safetensors" not in caplog.text
        caplog.clear()
        vp.check_cache_stamp(pack.cache_metadata(), "e.safetensors", None)
        assert "no pack is active" in caplog.text


# --- front door ---------------------------------------------------------------


@pytest.mark.core
def test_factory_returns_the_stock_class_when_off(monkeypatch):
    """Off must be inert: no subclass, no encoder, nothing pack-shaped."""
    from library.anima.strategy import AnimaTokenizeStrategy

    class Stub(AnimaTokenizeStrategy):
        def __init__(self, **kw):  # skip tokenizer loading
            self.kw = kw

    monkeypatch.setattr(vp, "AnimaTokenizeStrategy", Stub)
    s = vp.make_tokenize_strategy(None, qwen3_path="q")
    assert type(s) is Stub
    assert vp.strategy_pack(s) is None


# --- G1 at the strategy level (real tokenizers) ------------------------------------

_REAL_PACKS = (
    REPO / "models" / "vocab_packs" / "anima_cjk_vocab_pack_jp_v1",
    REPO
    / "models"
    / "vocab_packs"
    / "anima_cjk_vocab_pack_jp_v1"
    / "anima_cjk_vocab_pack_jp_v1",
    REPO / "output" / "ckpt" / "cjk_vocab" / "cjk_vocab_pack_synthjakozh1sym_r256",
)


def _real_strategies():
    from library.env import default_checkpoints
    from library.anima.strategy import AnimaTokenizeStrategy

    prefix = next((p for p in _REAL_PACKS if p.with_suffix(".json").exists()), None)
    if prefix is None:
        pytest.skip("no vocab pack on disk (make download-vocab-pack)")
    ckpt = default_checkpoints()
    if not Path(ckpt.text_encoder).exists():
        pytest.skip("Qwen3 text encoder not downloaded")
    stock = AnimaTokenizeStrategy(qwen3_path=ckpt.text_encoder)
    packed = vp.VocabPackTokenizeStrategy(
        vp.load_vocab_pack(prefix),
        qwen3_tokenizer=stock.qwen3_tokenizer,
        t5_tokenizer=stock.t5_tokenizer,
    )
    return stock, packed


@pytest.mark.core
def test_en_prompts_tokenize_bit_identically_through_the_pack():
    stock, packed = _real_strategies()
    prompts = [
        "1girl, silver hair, cat ears, smile, classroom, upper body",
        'masterpiece. A sign that reads "ANIMA". On the left, hakurei reimu, red eyes.',
        "",
    ]
    a, b = stock.tokenize(prompts), packed.tokenize(prompts)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@pytest.mark.core
def test_cjk_prompt_lands_on_pack_rows():
    _, packed = _real_strategies()
    _, _, t5_ids, t5_mask = packed.tokenize("1girl, 猫耳, 銀髪, セーラー服")
    live = t5_ids[0][: int(t5_mask[0].sum())]
    assert int((live >= T5_TABLE_SIZE).sum()) > 0
    assert int(t5_mask[0].sum()) < t5_ids.shape[1]  # eos-terminated, max-padded
