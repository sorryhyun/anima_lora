#!/usr/bin/env python3
"""Submit a ComfyUI workflow as a batch over the ComfyUI API.

Two modes, auto-selected by the workflow's contents:

* **Image mode** — if the workflow contains a ``LoadImage`` node and
  ``--images_dir`` is given, submit one job per image file in that directory,
  rewriting every ``LoadImage`` node's ``image`` input to point at the current
  file. Jobs are queued sequentially (``--wait``), so the server keeps the model
  + block-compile graph warm across the whole folder. Used by
  ``make comfy-batch W=colorize.json``.

* **Artist×chara mode** — otherwise, read ``__artist__`` / ``__chara__``
  placeholders and submit one job per cartesian-product pair. With
  ``--prompts``, the prompt text itself becomes a third axis: every line of the
  file replaces the positive ``CLIPTextEncode`` text before substitution, so
  prompt × artist × chara pairs are queued.

In **both** modes, every ``__random__`` placeholder anywhere in the workflow is
replaced per job by a random tag drawn from ``--randoms`` (default:
``workflows/randoms.yaml``, falling back to ``workflows/randoms.txt``).

``__random:<group>__`` draws from one named YAML group, and ``__random:a|b__``
from the union of several; bare ``__random__`` draws from the groups the file's
``default:`` key lists (all of them if it has none). Within one job draws are
**without replacement** — two placeholders never land on the same tag unless the
pool is exhausted. Randoms are not an axis: they re-roll per queued job rather
than multiplying the job count.

Usage:
    python scripts/toolkits/comfy_batch.py workflows/colorize.json \
        --images_dir ../comfy/input/to_colorize
    python scripts/toolkits/comfy_batch.py workflows/lora-batch.json \
        --artist workflows/artist.txt --chara workflows/chara.txt
    python scripts/toolkits/comfy_batch.py workflows/modhydra.json \
        --prompts workflows/preferred.txt
    python scripts/toolkits/comfy_batch.py workflows/modhydra.json \
        --randoms workflows/randoms.yaml
"""

import argparse
import json
import itertools
import os
import random
import re
import time
import urllib.request
import urllib.error
from typing import NamedTuple


COMFY_URL = "http://localhost:8188"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_lines(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_prompts(path: str) -> list[str]:
    """Read a prompt-per-line file.

    Lines may be bare text or JSON string literals with a trailing comma (the
    format ``workflows/preferred.txt`` is kept in, so it can be pasted straight
    out of a workflow JSON). ``#`` comment lines are skipped.
    """
    prompts = []
    for line in load_lines(path):
        if line.startswith("#"):
            continue
        stripped = line.rstrip(",").strip()
        if stripped.startswith('"'):
            try:
                stripped = json.loads(stripped)
            except json.JSONDecodeError:
                pass
        stripped = stripped.strip()
        if stripped:
            prompts.append(stripped)
    return prompts


PLACEHOLDERS = ("__prompt__", "__artist__", "__chara__")
# __random__ | __random:group__ | __random:groupa|groupb__ — the trailing `__`
# is matched explicitly so a group name may itself contain underscores.
RANDOM_RE = re.compile(r"__random(?::([A-Za-z0-9_|+ -]+))?__")
# Reserved YAML key: which groups a bare __random__ draws from.
DEFAULT_KEY = "default"
RANDOMS_FALLBACKS = ("workflows/randoms.yaml", "workflows/randoms.txt")


class Entry(NamedTuple):
    """One draw candidate: a coherent tag *cluster*, not necessarily one tag.

    A booru tag rarely stands alone — an act tag without its mandatory
    companions leaves the model guessing what the scene is. ``avoid`` lists tags this cluster
    never co-occurs with in the corpus, so an act and a framing that can't
    physically coexist don't get drawn into the same prompt.

    ``weight`` biases the draw (relative, 1.0 = neutral) — build_randoms.py
    sets it from selection lift: how much likelier this cluster's anchor is in
    the hand-picked dataset than in the raw crawl it was picked from.
    """

    tags: tuple[str, ...]
    avoid: frozenset[str] = frozenset()
    weight: float = 1.0

    @classmethod
    def parse(cls, item, where: str) -> "Entry":
        if isinstance(item, str):
            return cls(_split_tags(item))
        if isinstance(item, dict):
            tags = item.get("tags")
            if isinstance(tags, str):
                tags = _split_tags(tags)
            elif isinstance(tags, list):
                tags = tuple(str(t).strip() for t in tags if str(t).strip())
            else:
                raise SystemExit(f"{where}: entry needs 'tags:' (string or list)")
            avoid = item.get("avoid") or []
            if not isinstance(avoid, list):
                raise SystemExit(f"{where}: 'avoid:' must be a list of tags")
            try:
                weight = float(item.get("weight", 1.0))
            except (TypeError, ValueError):
                raise SystemExit(f"{where}: 'weight:' must be a number") from None
            if weight <= 0:
                raise SystemExit(f"{where}: 'weight:' must be > 0")
            return cls(tuple(tags), frozenset(str(a).strip() for a in avoid), weight)
        raise SystemExit(f"{where}: entry must be a string or a mapping")


def _split_tags(text: str) -> tuple[str, ...]:
    """A cluster written inline — ``"tag a, tag b, tag c"``.

    Pool entries are our own data (never a user caption), so a plain comma
    split is safe here — captions with position clauses are not.
    """
    return tuple(t.strip() for t in text.split(",") if t.strip())


class RandomPool:
    """Named tag groups backing ``__random__`` / ``__random:<group>__``.

    ``default`` names the groups a bare ``__random__`` unions — the point of
    the split is that mutually-exclusive axes (position, framing, …) stay out
    of the default draw and are asked for by name.
    """

    def __init__(
        self, groups: dict[str, list[Entry]], default: list[str] | None = None
    ):
        self.groups = groups
        names = default if default is not None else list(groups)
        self.default = _dedup(e for name in names for e in groups.get(name, []))

    def __bool__(self) -> bool:
        return bool(self.default or self.groups)

    def pool_for(self, spec: str | None) -> list[Entry]:
        """Entries a placeholder's group spec resolves to (``None`` = default)."""
        if not spec:
            return self.default
        names = [n.strip() for n in spec.split("|") if n.strip()]
        unknown = [n for n in names if n not in self.groups]
        if unknown:
            print(
                f"WARNING: unknown __random__ group(s) {', '.join(unknown)} "
                f"(have: {', '.join(sorted(self.groups))})"
            )
        return _dedup(e for n in names for e in self.groups.get(n, []))


def _dedup(entries) -> list[Entry]:
    """Order-preserving de-duplication, so an entry in two groups isn't 2x likely."""
    return list(dict.fromkeys(entries))


def load_randoms(path: str | None) -> RandomPool:
    """Load the random pool from YAML (grouped) or a flat one-per-line txt.

    A missing file is not an error — a workflow with no ``__random__`` in it
    never touches the pool, and one that does gets a warning at draw time.
    """
    if not path:
        path = next((p for p in RANDOMS_FALLBACKS if os.path.exists(p)), None)
    if not path or not os.path.exists(path):
        return RandomPool({})

    if os.path.splitext(path)[1].lower() not in (".yaml", ".yml"):
        flat = [
            Entry(_split_tags(line))
            for line in load_lines(path)
            if not line.startswith("#")
        ]
        return RandomPool({"all": flat})

    import yaml  # only the YAML path needs it, so .txt keeps working without it

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a mapping of group name -> tag list")

    default = data.get(DEFAULT_KEY)
    if default is not None and not isinstance(default, list):
        raise SystemExit(f"{path}: '{DEFAULT_KEY}:' must be a list of group names")

    groups = {}
    for name, tags in data.items():
        if name == DEFAULT_KEY:
            continue
        if not isinstance(tags, list):
            raise SystemExit(f"{path}: group '{name}' must be a list of entries")
        groups[name] = [Entry.parse(t, f"{path}: group '{name}'") for t in tags]

    unknown = [n for n in (default or []) if n not in groups]
    if unknown:
        raise SystemExit(f"{path}: '{DEFAULT_KEY}:' names missing group(s) {unknown}")
    return RandomPool(groups, default)


def apply_prompt(workflow: dict, prompt: str) -> None:
    """Overwrite the positive ``CLIPTextEncode`` text in-place.

    The positive node is the one carrying a placeholder (``__prompt__`` or the
    artist/chara/random markers) — negatives never do.
    """
    for node in workflow.values():
        if node.get("class_type") != "CLIPTextEncode":
            continue
        text = node.get("inputs", {}).get("text")
        if isinstance(text, str) and (
            any(p in text for p in PLACEHOLDERS) or RANDOM_RE.search(text)
        ):
            node["inputs"]["text"] = prompt


def _json_escape(value: str) -> str:
    """Escape for splicing into a JSON string context (backslashes doubled)."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def substitute(workflow: dict, artist: str, chara: str) -> dict:
    """Deep-copy workflow and replace __artist__ / __chara__ in all string values."""
    raw = json.dumps(workflow)
    raw = raw.replace("__artist__", _json_escape(artist)).replace(
        "__chara__", _json_escape(chara)
    )
    return json.loads(raw)


def _pinned_tags(workflow: dict, pool: RandomPool) -> set[str]:
    """Pool tags the prompt itself already states, around the placeholders.

    Seeding the draw state with these is what lets ``avoid:`` see a *pinned*
    tag: a template that hardcodes ``pov`` should not then draw a cluster that
    can't coexist with it, and a template that already says ``sex`` shouldn't
    have a cluster repeat it.

    Segments are read, never rewritten — an exact match against a known pool
    tag, so a prose clause in the prompt simply matches nothing.
    """
    known = {t for entries in pool.groups.values() for e in entries for t in e.tags}
    pinned = set()
    for node in workflow.values():
        text = node.get("inputs", {}).get("text")
        if not isinstance(text, str) or not RANDOM_RE.search(text):
            continue
        for segment in re.split(r"[,.]", RANDOM_RE.sub("", text)):
            segment = segment.strip()
            if segment in known:
                pinned.add(segment)
    return pinned


def substitute_random(workflow: dict, pool: RandomPool) -> tuple[dict, list[str]]:
    """Replace every ``__random__`` / ``__random:<group>__`` with a draw.

    Placeholders resolve left to right and each one sees what the earlier ones
    drew, which buys three things within a job:

    * a tag already in the prompt is dropped from a later cluster, so nothing
      renders twice (a repeat is a silent prompt weight, not a no-op);
    * an entry is skipped when it ``avoid``\\ s an already-drawn tag, or when an
      already-drawn cluster avoids *it* — that's what keeps two corpus-
      incompatible clusters out of the same prompt;
    * entries that add nothing new are passed over before entries that repeat.

    Returns the (deep-copied) workflow and the draws in order, for the caller's
    job label. An empty pool leaves the placeholder in place so the failure
    shows up in the image instead of silently vanishing.
    """
    raw = json.dumps(workflow)
    if not RANDOM_RE.search(raw):
        return json.loads(raw), []
    if not pool:
        print("WARNING: __random__ in workflow but the randoms file is empty/missing")
        return json.loads(raw), []

    draws: list[str] = []
    used: set[str] = _pinned_tags(workflow, pool)
    avoided: set[str] = set()

    def compatible(entry: Entry) -> bool:
        return not (entry.avoid & used or avoided & set(entry.tags))

    def draw(match):
        candidates = pool.pool_for(match.group(1))
        if not candidates:
            print(f"WARNING: empty pool for {match.group(0)}")
            return match.group(0)
        allowed = [e for e in candidates if compatible(e)] or candidates
        # Prefer a cluster that still has something to say; fall back to any
        # allowed one rather than starving a slot on an exhausted pool.
        fresh = [e for e in allowed if not set(e.tags) <= used]
        chosen = fresh or allowed
        pick = random.choices(chosen, weights=[e.weight for e in chosen])[0]
        tags = [t for t in pick.tags if t not in used] or list(pick.tags)
        used.update(tags)
        avoided.update(pick.avoid)
        draws.append(", ".join(tags))
        return _json_escape(", ".join(tags))

    return json.loads(RANDOM_RE.sub(draw, raw)), draws


def load_image_nodes(workflow: dict) -> list[dict]:
    """Return every LoadImage node's ``inputs`` dict in the workflow."""
    return [
        node["inputs"]
        for node in workflow.values()
        if node.get("class_type") == "LoadImage" and "inputs" in node
    ]


def list_images(images_dir: str) -> list[str]:
    """Sorted list of image filenames (not paths) directly in ``images_dir``."""
    return sorted(
        f
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )


def comfy_relative(images_dir: str) -> str:
    """Path of ``images_dir`` as ComfyUI's LoadImage expects it.

    LoadImage resolves names against the server's ``input/`` dir, so we strip
    everything up to and including ``.../input/`` and keep the remainder as a
    forward-slash subfolder prefix. If no ``input`` segment is present we fall
    back to the basename (assumes files sit directly under ``input/``).
    """
    norm = os.path.normpath(os.path.abspath(images_dir))
    parts = norm.split(os.sep)
    if "input" in parts:
        after = parts[parts.index("input") + 1 :]
        return "/".join(after)
    return os.path.basename(norm)


def queue_prompt(workflow: dict, server: str) -> dict:
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{server}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def wait_until_done(server: str, prompt_id: str, poll_interval: float = 2.0):
    """Poll /history until the prompt_id appears (i.e. execution finished)."""
    while True:
        try:
            with urllib.request.urlopen(f"{server}/history/{prompt_id}") as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                return history[prompt_id]
        except urllib.error.URLError:
            pass
        time.sleep(poll_interval)


def submit(wf: dict, server: str, randomize_seed: bool, wait: bool, label: str):
    """Optionally reseed, queue one job, and (optionally) wait for it."""
    if randomize_seed:
        for node in wf.values():
            if "seed" in node.get("inputs", {}):
                node["inputs"]["seed"] = random.randint(0, 2**53)

    print(f"{label} ... ", end="", flush=True)
    try:
        result = queue_prompt(wf, server)
        prompt_id = result["prompt_id"]
    except (urllib.error.URLError, KeyError) as e:
        print(f"FAILED to queue: {e}")
        return

    if wait:
        wait_until_done(server, prompt_id)
        print("done")
    else:
        print(f"queued ({prompt_id})")


def run_image_mode(workflow, images_dir, args):
    """One job per image file in ``images_dir`` (sequential)."""
    files = list_images(images_dir)
    if not files:
        print(f"No images found in {images_dir}")
        return
    prefix = comfy_relative(images_dir)
    print(
        f"Queuing {len(files)} images from {images_dir} (LoadImage prefix '{prefix}/')"
    )

    randoms = load_randoms(args.randoms)

    for i, fname in enumerate(files, 1):
        wf, draws = substitute_random(workflow, randoms)
        rel = f"{prefix}/{fname}" if prefix else fname
        for inputs in load_image_nodes(wf):
            inputs["image"] = rel
        tag = fname + (f" | {', '.join(draws)}" if draws else "")
        submit(
            wf,
            args.server,
            args.randomize_seed,
            args.wait,
            f"[{i}/{len(files)}] {tag}",
        )

    print("All done.")


def run_artist_chara_mode(workflow, args):
    """One job per (prompt, artist, chara) triple."""
    # An empty/missing list file is an inert axis, not zero jobs.
    artists = load_lines(args.artist) or [""]
    charas = load_lines(args.chara) or [""]
    prompts = load_prompts(args.prompts) if args.prompts else [None]
    randoms = load_randoms(args.randoms)

    combos = list(itertools.product(enumerate(prompts, 1), artists, charas))
    print(
        f"Queuing {len(prompts)} prompts × {len(artists)} artists × "
        f"{len(charas)} charas = {len(combos)} jobs"
    )

    for i, ((pidx, prompt), artist, chara) in enumerate(combos, 1):
        wf = json.loads(json.dumps(workflow))
        if prompt is not None:
            apply_prompt(wf, prompt)
        wf = substitute(wf, artist, chara)
        wf, draws = substitute_random(wf, randoms)
        tag = " x ".join(p for p in (artist, chara) if p)
        if prompt is not None:
            tag = f"prompt{pidx}" + (f" | {tag}" if tag else "")
        if draws:
            tag = f"{tag} | {', '.join(draws)}" if tag else ", ".join(draws)
        submit(
            wf,
            args.server,
            args.randomize_seed,
            args.wait,
            f"[{i}/{len(combos)}] {tag}",
        )

    print("All done.")


def main():
    parser = argparse.ArgumentParser(description="ComfyUI batch runner")
    parser.add_argument("workflow", help="Path to workflow JSON")
    parser.add_argument("--artist", default="workflows/artist.txt")
    parser.add_argument("--chara", default="workflows/chara.txt")
    parser.add_argument(
        "--prompts",
        default=None,
        help=(
            "File of prompts (one per line, bare or JSON-quoted); each replaces "
            "the positive CLIPTextEncode text, e.g. workflows/preferred.txt."
        ),
    )
    parser.add_argument(
        "--randoms",
        default=None,
        help=(
            "Random pool: a grouped YAML (default workflows/randoms.yaml, else "
            "randoms.txt). __random__ draws from the 'default:' groups, "
            "__random:<group>__ from one, __random:a|b__ from several."
        ),
    )
    parser.add_argument(
        "--images_dir",
        default=None,
        help="Directory of images; one job per image (requires a LoadImage node).",
    )
    parser.add_argument("--server", default=COMFY_URL)
    parser.add_argument(
        "--randomize-seed",
        action="store_true",
        default=True,
        help="Randomize seed per job (default: true)",
    )
    parser.add_argument(
        "--no-randomize-seed", dest="randomize_seed", action="store_false"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Wait for each job to finish before queuing next (default: true)",
    )
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    args = parser.parse_args()

    with open(args.workflow) as f:
        workflow = json.load(f)

    # Image mode when the workflow loads images *and* a dir was provided;
    # otherwise fall back to the artist×chara placeholder batch.
    if args.images_dir and load_image_nodes(workflow):
        run_image_mode(workflow, args.images_dir, args)
    else:
        run_artist_chara_mode(workflow, args)


if __name__ == "__main__":
    main()
