"""EN wiki description ↔ KO KB description pairs (register ``desc_ko``).

The KR KB (``models/danbooru_tags_classified.csv``, Localsmile
danbooru_KR_wiki) carries a natural-Korean summary translation of each
danbooru wiki body. Pairing the EN wiki first sentence with that KO summary
yields domain prose pairs — tag vocabulary inside real Korean sentences
(particles, spacing), the register the composed tag-bag corpus never
exercises and the s* prose prompts measure at floor.

Each pair carries ONE full-width span (whole EN sentence ↔ whole KO
description, ``via: kb_desc``): under the span loss a span-less row
contributes zero gradient — the D2 ``commentary`` rows are inert for exactly
that reason (and a pure span-less eval batch raises), so the single
full-width span is what makes this register supervision at all. Restricted
to tags the caption corpus actually uses (the glossary tag list), so visits
land on rows the eval prompts can reach.

CPU-only. Output: ``post_image_dataset/cjk_distill/pairs_desc_ko.jsonl``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

KB_PREFIX = re.compile(r"^\[[^\]]*\]\s*")
KB_KEYWORDS = re.compile(r"\s*키워드\s*:.*$", re.S)
WIKI_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
MARKUP_LINE = re.compile(r"^(h[1-6]\.|\*|!|\|)", re.M)
HANGUL = re.compile(r"[가-힣]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s")


def en_first_sentence(body: str) -> str | None:
    for para in body.replace("\r", "").split("\n"):
        para = para.strip()
        if not para or MARKUP_LINE.match(para):
            continue
        para = WIKI_LINK.sub(r"\1", para)
        sent = SENT_SPLIT.split(para)[0].strip()
        if len(sent) >= 20 and re.search(r"[a-zA-Z]", sent):
            return sent
    return None


def ko_desc(raw: str) -> str | None:
    d = KB_KEYWORDS.sub("", KB_PREFIX.sub("", raw.strip())).strip()
    if len(d) >= 10 and HANGUL.search(d):
        return d
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kr-kb", default=REPO / "models" / "danbooru_tags_classified.csv")
    ap.add_argument(
        "--glossary",
        default=Path(__file__).parent / "assets" / "tag_glossary_ko.json",
        help="restricts pairs to tags the caption corpus uses",
    )
    ap.add_argument(
        "--wiki",
        default=None,
        help="EN wiki jsonl; default resolves kierarkia/danbooru-wiki-2026 from the HF cache",
    )
    ap.add_argument(
        "--out",
        default=REPO / "post_image_dataset" / "cjk_distill" / "pairs_desc_ko.jsonl",
    )
    args = ap.parse_args()

    if args.wiki is None:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        args.wiki = hf_hub_download(
            "kierarkia/danbooru-wiki-2026",
            "danbooru_wiki_dataset_2026-04-28.jsonl",
            repo_type="dataset",
        )

    corpus = set(json.loads(Path(args.glossary).read_text())["tags"])

    en_side: dict[str, str] = {}
    with open(args.wiki, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tag = (r.get("title") or "").replace("_", " ").lower()
            if tag not in corpus:
                continue
            sent = en_first_sentence(r.get("body") or "")
            if sent:
                en_side[tag] = sent

    n_kb = n_out = 0
    seen_ko: set[str] = set()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(args.kr_kb, newline="", encoding="utf-8-sig") as f,
        out.open("w", encoding="utf-8") as w,
    ):
        for row in csv.DictReader(f):
            tag = row["name"].replace("_", " ").lower()
            if tag not in en_side:
                continue
            n_kb += 1
            ko = ko_desc(row["description"] or "")
            if not ko or ko in seen_ko:  # aliased tags share one description
                continue
            seen_ko.add(ko)
            w.write(
                json.dumps(
                    {
                        "id": f"KB/{tag.replace(' ', '_')}/desc_ko",
                        "source": "KB",
                        "register": "desc_ko",
                        "lang": "ko",
                        "en": en_side[tag],
                        "ja": ko,
                        "via": "kb_desc",
                        "n_missing": 0,
                        "spans": [
                            {"en": en_side[tag], "ja": ko, "via": "kb_desc", "f1": 0.0}
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_out += 1
    print(
        f"[desc_pairs] corpus tags {len(corpus)}, EN sentences {len(en_side)}, "
        f"KB rows matched {n_kb}, pairs written {n_out} -> {out}"
    )


if __name__ == "__main__":
    main()
