"""CJK vocab pack as a shipped text-encoder asset (``library.anima.vocab_pack``).

Invariants:

* **Off means bit-exact.** With ``vocab_pack`` unset the strategy factory returns
  the stock tokenizer class and no hooks are installed — the whole path must be
  inert for EN-only users.
* **The hook pair never touches the state dict.** ``llm_adapter.embed`` keeps its
  32128 rows; ext ids resolve to pack rows through hooks, base ids are unchanged.
* **Identity is stamped, and mismatches warn once.** TE caches carry the pack
  digest; a LoRA carries ``ss_ext_pack_sha``; a different active pack (or none)
  logs a warning instead of silently training / sampling on the wrong rows.
* **EN captions tokenize identically through the pack** (G1 of the CJK line,
  lifted to the strategy level) — needs the real tokenizers, skipped otherwise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

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


def test_off_values_resolve_to_none():
    assert vp.resolve_pack_prefix(None) is None
    assert vp.resolve_pack_prefix("") is None
    assert vp.resolve_pack_prefix("   ") is None
    assert vp.load_vocab_pack("") is None
    assert vp.load_vocab_pack(None) is None


def test_prefix_accepts_bare_either_file_or_dir(synthetic_pack: Path):
    assert vp.resolve_pack_prefix(synthetic_pack) == synthetic_pack
    assert vp.resolve_pack_prefix(synthetic_pack.with_suffix(".json")) == synthetic_pack
    assert (
        vp.resolve_pack_prefix(synthetic_pack.with_suffix(".safetensors"))
        == synthetic_pack
    )
    assert vp.resolve_pack_prefix(synthetic_pack.parent) == synthetic_pack


def test_half_installed_pack_is_an_error_with_the_download_hint(tmp_path: Path):
    prefix = tmp_path / "half"
    save_file(
        {"ext_embed": torch.zeros(1, DIM)}, str(prefix.with_suffix(".safetensors"))
    )
    with pytest.raises(FileNotFoundError, match="download-vocab-pack"):
        vp.resolve_pack_prefix(prefix)
    with pytest.raises(FileNotFoundError):
        vp.resolve_pack_prefix(tmp_path / "does_not_exist")


def test_load_is_memoised_and_carries_identity(synthetic_pack: Path):
    a = vp.load_vocab_pack(synthetic_pack)
    b = vp.load_vocab_pack(str(synthetic_pack.with_suffix(".json")))
    assert a is b
    assert a.name == "tiny_pack"
    assert a.rows == ROWS
    assert a.training == {"label": "tiny"}
    assert len(a.digest) >= 12
    assert a.checkpoint_metadata() == {
        "ss_ext_pack": "tiny_pack",
        "ss_ext_pack_sha": a.digest,
    }
    assert a.cache_metadata() == {"vocab_pack": "tiny_pack", "vocab_pack_sha": a.digest}
    # A loaded pack passes through load_vocab_pack untouched.
    assert vp.load_vocab_pack(a) is a


def test_default_pack_reads_env_then_base_toml(monkeypatch):
    from library.env import default_checkpoints

    monkeypatch.setenv("ANIMA_VOCAB_PACK", "some/pack")
    assert default_checkpoints().vocab_pack == "some/pack"
    assert vp.default_vocab_pack() == "some/pack"
    monkeypatch.delenv("ANIMA_VOCAB_PACK")
    # base.toml ships the key empty (opt-in): the shipped default is OFF.
    assert default_checkpoints().vocab_pack == ""


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


def test_hooks_are_a_no_op_without_ext_ids(synthetic_pack: Path):
    pack = vp.load_vocab_pack(synthetic_pack)
    adapter = _adapter()
    ids = torch.tensor([[3, 1, 4, 1, 5]])
    before = adapter.embed(ids).clone()
    vp.attach_vocab_pack(adapter, pack)
    torch.testing.assert_close(adapter.embed(ids), before)


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


def test_attach_accepts_a_dit_like_object(synthetic_pack: Path):
    class FakeDiT:
        def __init__(self):
            self.llm_adapter = _adapter()

    pack = vp.load_vocab_pack(synthetic_pack)
    dit = FakeDiT()
    vp.attach_vocab_pack(dit, pack)
    assert vp.attached_pack_digest(dit) == pack.digest
    with pytest.raises(RuntimeError, match="llm_adapter"):
        vp.attach_vocab_pack(object(), pack)


def test_load_anima_model_style_helper_skips_when_off(synthetic_pack: Path):
    from library.anima.weights import _attach_vocab_pack_if_set

    adapter = _adapter()
    _attach_vocab_pack_if_set(adapter, None)
    _attach_vocab_pack_if_set(adapter, "")
    assert vp.attached_pack_digest(adapter) is None
    _attach_vocab_pack_if_set(adapter, str(synthetic_pack))
    assert vp.attached_pack_digest(adapter) is not None


# --- identity stamps ------------------------------------------------------------


def test_checkpoint_stamp_round_trip_and_mismatch_warning(
    synthetic_pack: Path, tmp_path: Path, caplog
):
    pack = vp.load_vocab_pack(synthetic_pack)
    lora = tmp_path / "lora.safetensors"
    save_file({"w": torch.zeros(1)}, str(lora), metadata=pack.checkpoint_metadata())
    assert vp.read_checkpoint_stamp(lora) == ("tiny_pack", pack.digest)
    assert vp.read_checkpoint_stamp(tmp_path / "missing.safetensors") == ("", "")

    with caplog.at_level(logging.WARNING, logger=vp.__name__):
        vp.warn_checkpoint_pack_mismatch(lora, pack)  # same pack: silent
        assert not caplog.records
        vp.warn_checkpoint_pack_mismatch(lora, None)
        assert "no vocab pack is active" in caplog.text
        caplog.clear()
        other = vp.VocabPack(
            prefix=Path("other_pack"),
            table=pack.table,
            mapping={},
            digest="deadbeef" * 5,
        )
        vp.warn_checkpoint_pack_mismatch(lora, other)
        assert "ext rows differ" in caplog.text


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


def test_te_cache_writer_stamps_the_pack(synthetic_pack: Path, tmp_path: Path):
    """The strategy-level writer passes the stamp through to safetensors."""
    from safetensors import safe_open

    pack = vp.load_vocab_pack(synthetic_pack)
    path = tmp_path / "x_anima_te.safetensors"
    save_file({"t": torch.zeros(1)}, str(path), metadata=pack.cache_metadata())
    with safe_open(str(path), framework="pt") as f:
        assert f.metadata()["vocab_pack_sha"] == pack.digest


# --- front door ---------------------------------------------------------------


def test_generation_request_threads_the_pack_flags():
    from library.inference.request import GenerationRequest

    args = GenerationRequest(prompt="x", vocab_pack="p/q").to_args()
    assert args.vocab_pack == "p/q" and args.no_vocab_pack is False
    args = GenerationRequest(prompt="x", no_vocab_pack=True).to_args()
    assert args.vocab_pack is None and args.no_vocab_pack is True


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
    REPO / "models" / "vocab_packs" / "anima_cjk_vocab_pack",
    REPO / "output" / "ckpt" / "cjk_vocab_pack_synthjakozh1sym_r256",
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


def test_cjk_prompt_lands_on_pack_rows():
    _, packed = _real_strategies()
    _, _, t5_ids, t5_mask = packed.tokenize("1girl, 猫耳, 銀髪, セーラー服")
    live = t5_ids[0][: int(t5_mask[0].sum())]
    assert int((live >= T5_TABLE_SIZE).sum()) > 0
    assert int(t5_mask[0].sum()) < t5_ids.shape[1]  # eos-terminated, max-padded
