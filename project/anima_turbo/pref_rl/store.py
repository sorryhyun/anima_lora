"""Session store for the preference-RL pick rounds (proposal/pref_rl.md § 3.5, § 3.6).

Qt-free and torch-free: the GUI and the render server both read it.

A session is one directory, ``output/turbo_pref/<name>/``:

    session.json            policy and round shape (fixed at creation)
    requests/<rid>.json     a round the GUI asked for (caption, seeds, arms, size)
    rounds/<rid>/NN.png     the server's renders, index = grid slot
    rounds/<rid>/done.json  written last by the server: timings, the policy it rendered
    server.json             the server's heartbeat (state, pid, last round, time)
    stop                    the GUI asks the server to exit
    picks.jsonl             one row per answered round (the label log)

Round kinds:

- ``train``: one policy (``loras``), ``n_images`` fresh seeds.
- ``blind``: ``n_images / 2`` seeds rendered with arm A (``loras``) and the same seeds with
  arm B (``blind_loras``), shuffled into the grid. The order is in the request; the GUI never
  shows it. The read is B's share of the picks (0.5 = no gain).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from library.env import resolve_under_home

SESSIONS_ROOT = resolve_under_home("output/turbo_pref")
POOL_ROOT = resolve_under_home("post_image_dataset/resized")
MAX_PICKS = 8
P1_ROUNDS = 24


@dataclass
class SessionConfig:
    dit: str = "models/diffusion_models/anima_merged14.safetensors"
    loras: list[str] = field(
        default_factory=lambda: [
            "models/loras/turbo_hub/anima_turbo_4step_v3.safetensors"
        ]
    )
    blind_loras: list[str] = field(default_factory=list)
    int8: bool = True
    compile: bool = True
    infer_steps: int = 4
    guidance_scale: float = 1.0
    flow_shift: float = 3.0
    sampler: str = "euler"
    n_images: int = 16
    max_picks: int = MAX_PICKS
    tier: int = 1024  # edge tier the caption's native aspect is fitted into
    batch: int = 4  # images per denoise call
    pool: str = (
        "post_image_dataset/resized"  # a directory of captions or a .txt (one per line)
    )
    random_share: float = 0.5  # share of rounds that take a random caption (§ 3.5)
    created: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "SessionConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ── paths ────────────────────────────────────────────────────────────────


class Session:
    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def requests(self) -> Path:
        return self.root / "requests"

    @property
    def rounds(self) -> Path:
        return self.root / "rounds"

    @property
    def picks_path(self) -> Path:
        return self.root / "picks.jsonl"

    @property
    def server_path(self) -> Path:
        return self.root / "server.json"

    @property
    def stop_path(self) -> Path:
        return self.root / "stop"

    def round_dir(self, rid: str) -> Path:
        return self.rounds / rid

    # ── config ──

    def config(self) -> SessionConfig:
        return SessionConfig.from_dict(
            json.loads((self.root / "session.json").read_text())
        )

    @classmethod
    def create(cls, name: str, cfg: SessionConfig) -> "Session":
        root = SESSIONS_ROOT / name
        if (root / "session.json").exists():
            raise FileExistsError(f"session {name} exists")
        for sub in ("requests", "rounds"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        cfg.created = time.time()
        _write_json(root / "session.json", asdict(cfg))
        return cls(root)

    # ── requests / renders ──

    def request(self, rid: str) -> dict | None:
        return _read_json(self.requests / f"{rid}.json")

    def request_ids(self) -> list[str]:
        return sorted(p.stem for p in self.requests.glob("*.json"))

    def next_rid(self) -> str:
        # Rounds dirs count too: a withdrawn request the server had already started leaves
        # its dir behind, and its id must not come back.
        ids = [int(r) for r in self.request_ids()]
        if self.rounds.is_dir():
            ids += [int(p.name) for p in self.rounds.iterdir() if p.name.isdigit()]
        return f"{max(ids) + 1 if ids else 0:05d}"

    def withdraw(self, rid: str) -> bool:
        """Drop a request the server has not started (no rounds dir yet)."""
        if self.round_dir(rid).exists():
            return False
        (self.requests / f"{rid}.json").unlink(missing_ok=True)
        return True

    def write_request(self, req: dict) -> str:
        rid = self.next_rid()
        req = {**req, "rid": rid, "requested_at": time.time()}
        _write_json(self.requests / f"{rid}.json", req)
        return rid

    def done(self, rid: str) -> dict | None:
        return _read_json(self.round_dir(rid) / "done.json")

    def image_path(self, rid: str, slot: int) -> Path:
        return self.round_dir(rid) / f"{slot:02d}.png"

    def rendered_slots(self, rid: str) -> list[int]:
        d = self.round_dir(rid)
        if not d.is_dir():
            return []
        return sorted(int(p.stem) for p in d.glob("[0-9][0-9].png"))

    def pending_rids(self) -> list[str]:
        """Requested, not rendered yet (server's work list)."""
        return [r for r in self.request_ids() if self.done(r) is None]

    def unanswered_rids(self) -> list[str]:
        """Requested and not yet answered in ``picks.jsonl`` (GUI's queue)."""
        answered = {row["rid"] for row in self.picks()}
        return [r for r in self.request_ids() if r not in answered]

    # ── labels ──

    def picks(self) -> list[dict]:
        if not self.picks_path.exists():
            return []
        rows = []
        for line in self.picks_path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def answer(
        self, rid: str, picks: list[int], *, skipped: bool = False, note: str = ""
    ) -> dict:
        req = self.request(rid) or {}
        done = self.done(rid) or {}
        row = {
            "rid": rid,
            "kind": req.get("kind", "train"),
            "caption": req.get("caption"),
            "caption_src": req.get("caption_src"),
            "caption_pick": req.get("caption_pick"),
            "seeds": req.get("seeds"),
            "arms": req.get("arms"),
            "size": req.get("size"),
            "picks": sorted(picks),
            "n_picks": len(picks),
            "skipped": skipped,
            "note": note,
            "policy": done.get("policy"),
            "head_scores": done.get("head_scores"),
            "answered_at": time.time(),
            "render_s": done.get("render_s"),
        }
        with open(self.picks_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    # ── server status ──

    def server_status(self, stale_s: float = 90.0) -> dict:
        st = _read_json(self.server_path) or {}
        if st and time.time() - st.get("t", 0) > stale_s:
            st = {**st, "state": "stale"}
        return st


def list_sessions() -> list[Session]:
    if not SESSIONS_ROOT.is_dir():
        return []
    out = [Session(p) for p in SESSIONS_ROOT.iterdir() if (p / "session.json").exists()]
    return sorted(out, key=lambda s: s.config().created, reverse=True)


# ── caption pool + next-caption choice ───────────────────────────────────


@dataclass
class Caption:
    text: str
    src: str  # file the caption came from ("" for a typed one)
    size: tuple[int, int] | None = None  # native (w, h) of the source image, if known


def load_pool(spec: str) -> list[Caption]:
    path = resolve_under_home(spec)
    if path.is_file():
        out = []
        for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines()):
            text, size = parse_prompt_line(ln)
            if text:
                out.append(Caption(text, f"{path.name}:{i + 1}", size))
        return out
    out = []
    for txt in sorted(path.rglob("*.txt")):
        if txt.name.endswith(".variants.txt"):
            continue
        text = txt.read_text(encoding="utf-8").strip()
        if not text:
            continue
        out.append(Caption(text, str(txt.relative_to(path))))
    return out


_OVERRIDE = re.compile(r"\s--(\w+)\s+(\S+)")


def parse_prompt_line(line: str) -> tuple[str, tuple[int, int] | None]:
    """Caption and (w, h) from a prompt-file line; ``--w`` / ``--h`` set the size, other
    ``--x v`` overrides (``--d`` seed etc.) are dropped, since the round sets seeds itself."""
    line = line.strip()
    if not line or line.startswith("#"):
        return "", None
    cut = line.find(" --")
    text = line if cut < 0 else line[:cut].strip()
    opts = dict(_OVERRIDE.findall(line[cut:] if cut >= 0 else ""))
    size = (int(opts["w"]), int(opts["h"])) if "w" in opts and "h" in opts else None
    return text, size


def native_size(pool_root: Path, cap: Caption) -> tuple[int, int] | None:
    """(w, h) of the image beside the caption, read from its header only."""
    if not cap.src or ":" in cap.src:
        return None
    stem = (pool_root / cap.src).with_suffix("")
    for ext in (".png", ".webp", ".jpg", ".jpeg"):
        p = stem.with_suffix(ext)
        if p.exists():
            try:
                from PIL import Image

                with Image.open(p) as im:
                    return im.size
            except Exception:  # noqa: BLE001
                return None
    return None


def render_size(native: tuple[int, int] | None, tier: int) -> tuple[int, int]:
    """(h, w) for the render: the caption's native aspect fitted into the tier's token band."""
    from library.datasets.buckets import EDGE_TOKEN_BANDS, freefit_bucket

    w, h = native or (tier, tier)
    bw, bh = freefit_bucket(w, h, EDGE_TOKEN_BANDS[tier])
    return bh, bw


class CaptionPicker:
    """§ 3.5: half the rounds random, the other half lowest predicted pick count but not zero.

    No predictor exists yet (the tag model of § 1 is the planned cold start), so until a
    ``scores`` map ``{caption src: predicted picks}`` is set every round is random and is
    logged as such.
    """

    def __init__(
        self, pool: list[Caption], random_share: float, seed: int | None = None
    ):
        self.pool = pool
        self.random_share = random_share
        self.rng = random.Random(seed)
        self.scores: dict[str, float] | None = None

    def pick(self, used: set[str]) -> tuple[Caption, str]:
        fresh = [c for c in self.pool if c.src not in used] or self.pool
        if self.scores and self.rng.random() >= self.random_share:
            ranked = sorted(
                (c for c in fresh if self.scores.get(c.src, 0.0) >= 0.5),
                key=lambda c: self.scores[c.src],
            )
            if ranked:
                return ranked[0], "predictor"
        return self.rng.choice(fresh), "random"


def make_request(
    cfg: SessionConfig, cap: Caption, kind: str, how: str, rng: random.Random
) -> dict:
    size = render_size(cap.size, cfg.tier)
    n = cfg.n_images
    if kind == "blind":
        half = n // 2
        seeds_half = [rng.randrange(2**31) for _ in range(half)]
        slots = [("A", s) for s in seeds_half] + [("B", s) for s in seeds_half]
        rng.shuffle(slots)
        arms = [a for a, _ in slots]
        seeds = [s for _, s in slots]
    else:
        seeds = [rng.randrange(2**31) for _ in range(n)]
        arms = ["A"] * n
    return {
        "kind": kind,
        "caption": cap.text,
        "caption_src": cap.src,
        "caption_pick": how,
        "size": list(size),
        "seeds": seeds,
        "arms": arms,
    }


# ── reads ────────────────────────────────────────────────────────────────


def blind_read(rows: list[dict]) -> dict:
    """B's share of picks over answered blind grids, and a two-sided sign test over grids."""
    grids = [r for r in rows if r.get("kind") == "blind" and not r.get("skipped")]
    a = b = 0
    wins = losses = 0
    for r in grids:
        arms = r.get("arms") or []
        nb = sum(1 for i in r["picks"] if arms[i] == "B")
        na = len(r["picks"]) - nb
        a += na
        b += nb
        wins += nb > na
        losses += na > nb
    share = b / (a + b) if a + b else None
    return {
        "grids": len(grids),
        "picks_a": a,
        "picks_b": b,
        "share_b": share,
        "wins_b": wins,
        "losses_b": losses,
        "p_sign": _sign_test(wins, losses),
    }


def train_read(rows: list[dict]) -> dict:
    rounds = [
        r for r in rows if r.get("kind", "train") == "train" and not r.get("skipped")
    ]
    hist = [0] * (MAX_PICKS + 1)
    for r in rounds:
        hist[min(r["n_picks"], MAX_PICKS)] += 1
    return {
        "rounds": len(rounds),
        "labelled": sum(len(r.get("seeds") or []) for r in rounds),
        "picked": sum(r["n_picks"] for r in rounds),
        "zero_rounds": hist[0],
        "hist": hist,
        "skipped": sum(1 for r in rows if r.get("skipped")),
    }


def _sign_test(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


# ── io ───────────────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_json(path: Path, obj: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
