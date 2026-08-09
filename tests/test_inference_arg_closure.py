"""H2 guard: ``getattr(args, …)`` read keys must be declared argparse flags.

``docs/findings/entanglement_audit_high_severity.md`` §H2 — the inference
long-tail (SMC-CFG / spectrum / soft-tokens) and the training loop read
config off ``args`` by name via ``getattr(args, "k", default)``. Rename or drop
the flag and the read silently falls back to its default — feature off, all
static tests still green.

These tests pin every such read to a declared flag, so a rename/drop fails
loudly here instead. Same closure pattern as H1 (``test_network_registry.py``),
shared via ``tests/config_closure.py``.
"""

from __future__ import annotations

from pathlib import Path

from tests.config_closure import assert_closed, declared_argparse_keys, getattr_keys

_REPO = Path(__file__).resolve().parent.parent


def test_inference_getattr_reads_are_declared_flags():
    """``getattr(args, …)`` in generation.py ⊆ inference argparse flags."""
    from library.inference import args as iargs

    declared = declared_argparse_keys(iargs.build_parser())
    reads = getattr_keys(_REPO / "library/inference/generation.py", receiver="args")
    assert_closed(reads, declared, what="inference args (generation.py)")


def test_train_getattr_reads_are_declared_flags():
    """``getattr(args, …)`` in train.py ⊆ train argparse flags.

    Three intentional non-flag reads are allowlisted:

    * ``_network_kwargs`` — private cache set by ``resolve_network_kwargs``,
      never an argparse flag.
    * ``_yarnsig_parsed`` — private memo of the parsed
      ``--sigma_lowres_yarnsig`` tuple, stashed on the namespace so the format
      is validated once. Same pattern as ``_network_kwargs``.
    * ``sampler`` — the *training-time* M1 noise-sampler registry key. There is
      deliberately no ``--sampler`` train flag (the inference ``--sampler`` is a
      different concept), so this read always resolves to ``"default"``. Pinned
      here so the fallback stays a conscious choice, not silent drift.
    """
    import train as trainmod

    declared = declared_argparse_keys(trainmod.setup_parser())
    reads = getattr_keys(_REPO / "train.py", receiver="args")
    assert_closed(
        reads,
        declared,
        what="train args (train.py)",
        allow_unregistered=frozenset({"_network_kwargs", "_yarnsig_parsed", "sampler"}),
    )


def test_sigma_rule2_flags_are_statically_visible():
    """The rule-2 σ flags must be read by LITERAL name in train.py.

    ``_sigma_rule_cfg`` deliberately spells each rule's flags out instead of
    building ``f"sigma_lowres_threshold{sfx}"``: the closure guard above
    AST-scans for literal names, so a computed name is invisible to it — the
    flag could be renamed in cli_args.py and every read would silently fall
    back to its default. This pins the property the guard depends on.
    """
    reads = getattr_keys(_REPO / "train.py", receiver="args")
    for flag in (
        "sigma_lowres_threshold",
        "sigma_lowres_threshold_max",
        "sigma_lowres_span",
        "sigma_lowres_threshold2",
        "sigma_lowres_threshold2_max",
        "sigma_lowres_span2",
        "sigma_lowres_route2",
    ):
        assert flag in reads, (
            f"{flag} is not read by literal name in train.py — a computed "
            "getattr name escapes the H2 drift guard"
        )
