#!/usr/bin/env python3
"""Build the v2 unmask eval prompt set — harder than the 8 generic rows of
``assets/unmask_eval_prompts.txt``: training characters + series names, the
``@sincos`` artist trigger on some rows and not others, a 2-character
crossover, ``japanese text`` / ``comic`` rows (the line's actual subject),
rare mid-frequency training tags, and a few rows sampled at random from the
training tag pool (seeded, so the set is reproducible).

Writes ``assets/unmask_eval_prompts_v2.txt`` (one prompt per line, row
order) and ``assets/unmask_eval_prompts_v2.json`` (per-row dbv4 tag lists
for ``probes/unmask_grid_judge.py``).

Tag pool filter: sexual / explicit, non-consent and minor-coded tags in the
training captions are never sampled (``BLOCK``); the curated rows are
hand-written and SFW. Add rows by hand if you want others.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
MIRROR = REPO / "post_image_dataset/cjk_unmask/mirror_sincos_ppocr"

BLOCK = (
    "loli",
    "shota",
    "child",
    "age difference",
    "young",
    "rape",
    "netorare",
    "slave",
    "bdsm",
    "leash",
    "pet play",
    "sex",
    "penis",
    "pussy",
    "cum",
    "nipple",
    "nude",
    "vaginal",
    "oral",
    "censor",
    "pubic",
    "nsfw",
    "explicit",
    "sensitive",
    "questionable",
    "testicles",
    "ahegao",
    "areola",
    "aftersex",
    "condom",
    "naughty face",
    "tally",
    "body writing",
    "spread legs",
    "missionary",
    "doggystyle",
    "cowgirl",
    "straddling",
    "girl on top",
    "all fours",
    "breasts out",
    "breast",
    "cleavage",
    "wince",
    "clenched teeth",
    "ugly bastard",
    "fat man",
    "obese",
    "old man",
    "old",
    "bald",
    "hetero",
    "1boy",
    "@",
    "^^^",
    "japanese text",
    "torn clothes",
    "clothes lift",
    "clothes pull",
    "shirt lift",
    "skirt lift",
    "lifting own clothes",
    "unworn",
    "clothing aside",
    "open clothes",
    "open shirt",
    "groin",
    "ass",
    "thighs",
    "underwear",
    "panties",
    "bra",
    "micro bikini",
    "string bikini",
    "side-tie bikini bottom",
    "saliva",
    "tongue",
    "sweat",
    "heart-shaped pupils",
    "symbol-shaped pupils",
    "grabbing",
    "grab",
    "collar",
    "bell",
    "kiss",
    "armpits",
    "feet",
    "toes",
    "legs",
    "bare legs",
    "wariza",
    "squatting",
    "on back",
    "lying",
    "on bed",
    "bed",
    "pillow",
    "bed sheet",
    "folded",
    "m legs",
    "legs up",
    "tiptoes",
    "female pubic hair",
    "hairy",
)

CURATED = [
    "siesta (tantei wa mou shindeiru), tantei wa mou shindeiru, 1girl, solo, white hair, blue eyes, black dress, holding magnifying glass, night, city street, rain, umbrella, looking at viewer",
    "fern (sousou no frieren), sousou no frieren, 1girl, solo, purple hair, long hair, holding staff, forest, dappled sunlight, from side, expressionless",
    "mayano top gun (umamusume), umamusume, 1girl, solo, horse ears, horse tail, orange hair, twintails, gym uniform, red buruma, running, track, motion lines",
    "nilou (genshin impact), genshin impact, 1girl, solo, red hair, very long hair, dancing, veil, jewelry, desert, sunset, full body",
    "@sincos, 1girl, solo, brown hair, short hair, school uniform, serafuku, classroom, sitting, desk, looking at viewer, smile",
    "@sincos, shiina mahiru (otonari no tenshi-sama), 1girl, solo, blonde hair, long hair, apron, kitchen, cooking, from behind, looking back",
    "@sincos, 2girls, rem (re:zero), elaina (majo no tabitabi), blue hair, grey hair, witch hat, maid headdress, side by side, standing, white background, simple background",
    "1girl, solo, demon girl, demon tail, fake horns, pointy ears, purple eyes, grey hair, hair intakes, sitting, bedroom, night, lamp",
    "reze (chainsaw man), chainsaw man, 1girl, solo, black hair, medium hair, hair bun, white shirt, t-shirt, cafe, holding cup, window, rain",
    "comic, 4koma, 1girl, japanese text, speech bubble, monochrome, greyscale, screentone, surprised, kitchen",
    "1girl, solo, japanese text, holding sign, sign, street, outdoors, shop, sunset, looking at viewer, smile",
    "sonoda chiyoko, idolmaster shiny colors, idolmaster, 1girl, solo, brown hair, medium hair, hair ribbon, stage, spotlight, microphone, singing",
    "2girls, hanami ume, sarashina ruka (kanojo okarishimasu), cheek-to-cheek, selfie, v, holding phone, amusement park, ferris wheel, day",
]


def pool() -> dict[str, list[str]]:
    files = [
        f for f in glob.glob(str(MIRROR / "*.txt")) if not f.endswith(".variants.txt")
    ]
    cnt: collections.Counter[str] = collections.Counter()
    for f in files:
        cap = Path(f).read_text(encoding="utf-8").splitlines()[0]
        for t in (t.strip() for t in cap.split(".")[0].split(",")):
            if t and not any(b in t for b in BLOCK):
                cnt[t] += 1
    chars = [t for t in cnt if re.search(r"\(.+\)$", t) and cnt[t] >= 2]
    common = [t for t, c in cnt.items() if c >= 20 and t not in chars]
    mid = [t for t, c in cnt.items() if 3 <= c < 20 and t not in chars]
    return {"chars": chars, "common": common, "mid": mid}


def random_rows(n: int, seed: int) -> list[str]:
    p = pool()
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        tags = ["1girl", "solo"]
        if rng.random() < 0.5:
            tags.append(rng.choice(p["chars"]))
        tags += rng.sample(p["common"], 4) + rng.sample(p["mid"], 5)
        seen, out = set(), []
        for t in tags:
            if t not in seen:
                seen.add(t)
                out.append(t)
        rows.append(", ".join(out))
    return rows


def dbv4(tag: str) -> str:
    return tag.replace(" ", "_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(PROJ / "assets" / "unmask_eval_prompts_v2"))
    o = ap.parse_args()
    rows = CURATED + random_rows(o.n_random, o.seed)
    Path(o.out + ".txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    js = {
        "_comment": "v2 eval rows (probes/make_eval_prompts.py). dbv4 card names; "
        "tags the card lacks are dropped by the judge with a warning. "
        "@trigger tags are not judge tags.",
        "rows": [
            {
                "row": i + 1,
                "prompt": r,
                "tags": [dbv4(t) for t in r.split(", ") if not t.startswith("@")],
            }
            for i, r in enumerate(rows)
        ],
    }
    Path(o.out + ".json").write_text(
        json.dumps(js, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    for i, r in enumerate(rows, 1):
        print(f"{i:2d}  {r}")
    print(f"-> {o.out}.txt / .json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
