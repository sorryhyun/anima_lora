"""jamo — Hangul syllables as (initial, vowel, final) and the factor rows
over them (``proposal_jamo.md`` § 2).

A syllable is ``0xAC00 + (cho · 21 + jung) · 28 + jong`` (19 × 21 × 28).
``cls`` is the vowel's class — vertical (the initial on the left), horizontal
(on top), compound (both) — and with or without a final it sets the block's
layout (6 of them). ``KSX1001`` is the 2 350 syllables the KO faces in
``assets/fonts/kozh/`` cover (a two-byte EUC-KR code; Python's codec spells
every other syllable as an 8-byte jamo sequence).

``Jamo`` composes, in delta units over the pack row,

    Δ(s) = b + C[cho, cls(jung)] + V[jung] + F[jong]       (F[0] = 0)

into ``Rows.delta.raw`` before every forward, so the optimizer holds the
106 factor vectors and the gradient reaches them through the composition.
"""

from __future__ import annotations

import torch

BASE, N_CHO, N_JUNG, N_JONG = 0xAC00, 19, 21, 28
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
CLASSES = ("vertical", "horizontal", "compound")
_CLS = {**dict.fromkeys("ㅏㅐㅑㅒㅓㅔㅕㅖㅣ", 0), **dict.fromkeys("ㅗㅛㅜㅠㅡ", 1)}
JUNG_CLS = tuple(_CLS.get(v, 2) for v in JUNG)
LAYOUTS = ("V", "H", "C", "VF", "HF", "CF")


def is_syllable(ch: str) -> bool:
    return len(ch) == 1 and BASE <= ord(ch) < BASE + N_CHO * N_JUNG * N_JONG


def decompose(ch: str) -> tuple[int, int, int]:
    """``(cho, jung, jong)`` indices; ``jong`` 0 = no final."""
    assert is_syllable(ch), ch
    k = ord(ch) - BASE
    return k // (N_JUNG * N_JONG), k // N_JONG % N_JUNG, k % N_JONG


def compose(cho: int, jung: int, jong: int) -> str:
    return chr(BASE + (cho * N_JUNG + jung) * N_JONG + jong)


def layout(ch: str) -> str:
    _, jung, jong = decompose(ch)
    return "VHC"[JUNG_CLS[jung]] + ("F" if jong else "")


def cells(ch: str) -> frozenset:
    """What a trained ``ch`` gives the factor model: its ``C`` cell (initial ×
    vowel class), its vowel, its final (0 included) and its layout. 57 + 21 +
    28 + 6 = 112 cells."""
    cho, jung, jong = decompose(ch)
    return frozenset(
        (("C", cho, JUNG_CLS[jung]), ("V", jung), ("F", jong), ("L", layout(ch)))
    )


def in_ksx1001(ch: str) -> bool:
    return is_syllable(ch) and len(ch.encode("euc_kr")) == 2


KSX1001 = tuple(
    c for c in map(chr, range(BASE, BASE + N_CHO * N_JUNG * N_JONG)) if in_ksx1001(c)
)
ALL = tuple(map(chr, range(BASE, BASE + N_CHO * N_JUNG * N_JONG)))


def syllable_rows() -> dict:
    """{syllable: its ext row} for all 11 172 on the run's pack: one row each
    (2 512 single Qwen tokens, the rest byte-split onto ``char`` rows)."""
    from data.inventory import pieces, qwen_pieces

    tok, q = qwen_pieces(char_rows=True)
    out = {}
    for s in ALL:
        got = pieces(tok, q, s)
        assert len(got) == 1 and got[0][1] is not None, (s, got)
        out[s] = int(got[0][1])
    assert len(set(out.values())) == len(out), "two syllables share a row"
    return out


def codes(syllables) -> torch.Tensor:
    """``(n, 4)`` long: cho, cls, jung, jong per syllable."""
    out = []
    for s in syllables:
        cho, jung, jong = decompose(s)
        out.append((cho, JUNG_CLS[jung], jung, jong))
    return torch.tensor(out, dtype=torch.long).reshape(-1, 4)


class Jamo:
    """The factor rows: ``b`` (1, d), ``C`` (19, 3, d), ``V`` (21, d), ``F``
    (27, d) — the finals past 0, all zero at init (Δ 0: the cold start).

    ``rows`` is the run's ``Rows``; ``syl`` = {ext id: syllable} for its
    trained rows, every one of them a syllable. ``apply`` writes the
    composition into ``rows.delta.raw`` (out of place: the free ``raw``
    keeps the frozen context rows, ``ExtDelta`` reads ``raw`` at forward
    time); call it before each step's forward and before a save."""

    N_VECTORS = 1 + N_CHO * len(CLASSES) + N_JUNG + (N_JONG - 1)

    def __init__(self, rows, syl: dict, lr: float):
        d = rows.delta.raw.shape[1]
        dev = rows.delta.raw.device
        ids = rows.delta.ext_ids
        live = [i for i, m in enumerate(rows.frozen_mask.tolist()) if not m]
        assert {ids[i] for i in live} == set(syl), (
            "factor rows: every trained row is a syllable, every syllable trained"
        )
        self.rows = rows
        # the frozen context rows: held, out of the optimizer
        self.free = rows.delta.raw.requires_grad_(False)
        self.loc = torch.tensor(live, dtype=torch.long, device=dev)
        self.code = codes([syl[ids[i]] for i in live]).to(dev)

        def p(*shape):
            return torch.nn.Parameter(torch.zeros(*shape, d, device=dev))

        self.b, self.C, self.V, self.F = (
            p(1),
            p(N_CHO, len(CLASSES)),
            p(N_JUNG),
            p(N_JONG - 1),
        )
        rows.params = [{"params": [self.b, self.C, self.V, self.F], "lr": lr}]

    def compose(self, code: torch.Tensor) -> torch.Tensor:
        """Δ (raw units) for ``code`` (``codes``' rows)."""
        F = torch.cat([torch.zeros_like(self.F[:1]), self.F])
        return (
            self.b + self.C[code[:, 0], code[:, 1]] + self.V[code[:, 2]] + F[code[:, 3]]
        )

    def apply(self) -> None:
        self.rows.delta.raw = self.free.index_put((self.loc,), self.compose(self.code))

    def state(self) -> dict:
        return {
            "model": "b + C[cho, cls] + V[jung] + F[jong]",
            "classes": list(CLASSES),
            **{k: getattr(self, k).detach().cpu() for k in ("b", "C", "V", "F")},
        }

    @torch.no_grad()
    def merge_all(self, sd: dict, ext_of: dict) -> int:
        """Every syllable's composed row into ``sd['delta']`` (``Rows
        .state_dict``'s), in place of whatever it holds for that id — the
        held-out syllables read zero-shot from the file as written.
        ``ext_of`` = {syllable: ext id}. Returns how many rows were added."""
        delta = sd["delta"]
        have = {int(e): i for i, e in enumerate(delta["ext_ids"])}
        syls = sorted(ext_of)
        comp = self.compose(codes(syls).to(self.b.device)).cpu().to(delta["raw"].dtype)
        raw = delta["raw"].clone()
        new_ids, new_rows = [], []
        for s, r in zip(syls, comp):
            e = int(ext_of[s])
            if e in have:
                raw[have[e]] = r
            else:
                new_ids.append(e)
                new_rows.append(r)
        ids = [int(e) for e in delta["ext_ids"]] + new_ids
        raw = torch.cat([raw, torch.stack(new_rows)]) if new_rows else raw
        order = sorted(range(len(ids)), key=ids.__getitem__)
        sd["delta"] = {
            **delta,
            "ext_ids": [ids[j] for j in order],
            "raw": raw[order].clone(),
        }
        return len(new_ids)
