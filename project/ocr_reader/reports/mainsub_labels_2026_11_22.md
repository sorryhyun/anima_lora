# main / sub labels: empty subs, waiting line breaks, the no-line-break count (2026-11-22)

The policy is [`main-sub-approach.md`](../main-sub-approach.md). Row counts are
`crop_dataset.load_mainsub`'s, after export `ko_review_1122g` (294 rows in
`assets/m109_mainsub.tsv`, local).

## 1. Loader changes

### Facts

Empty sub. The user's call: "sub가 비어있으면 ocr2: "" 가 아니라 비어있으면 학습 안하도록. 이건
negative 나중에 설정할때만 쓸거야".

- Until then a main with no kanji, every reviewed row and every KO / ZH row with no sub gave
  an `OCR2:` `""` row.
- train 179,636 → 172,967 rows, val 13,094 → 12,802 (counted after export f). The rows kept
  are the same as before.

Waiting line breaks. A main whose proposal missed `linebreak_auto.py apply`'s gate trained
as one line while the page showed the proposal. The user's call: "ㅇㅇ 빼는게 좋을거같네 일단".

- `waiting_break`, shared by the page and the loader, takes 8,412 unreviewed train rows out
  of `OCR:` (main 122,140 → 113,728; train 172,965 → 164,553 rows). The rows kept are the
  same as before.
- Their `OCR2:` rows stay; 2,281 of the crops have one.
- The 110 such val rows stay, so val is the set earlier runs were read on (12,802 rows).

Over the cap. The user's call: "데이터셋에서 max token limit 넘어가는건 그냥 안보이게 하자",
taken as `MAX_TARGET_CHARS` (150 normalized characters, the limit that drops rows).

- The loader drops 16 train rows: 15 mains (`mainsub_det` 9, `swaprelax_v5` 6) and 1 sub.
- The review page does not list the 15 mains' rows (`mark_over_cap`). They still export.
  4 of those crops still train an `OCR2:` row.

## 2. Labels with no line break

### Facts

Before the waiting rows went out, train had 146,288 `OCR:` rows, 64,550 of them with no
`\n` (3 empty ones included). Per stratum 48 random crops (seed 0) were read by Claude with
the label shown, counting those whose main spans more than one column or row. The estimate
is the share times the stratum.

| stratum, train, no `\n` | rows | multi-line in the sample | estimate |
|---|--:|--:|--:|
| proposal waiting | 8,412 | 48 / 48 | 8,412 |
| speech, 15+ characters | 1,498 | 40 / 48 | 1,248 |
| speech, 9–14 | 2,247 | 26 / 48 | 1,217 |
| speech, 5–8 | 11,025 | 8 / 48 | 1,838 |
| speech, ≤ 4 | 16,963 | 1 / 48 | 353 |
| SFX / no text box | 18,541 | 2 / 48 | 773 |
| KO / ZH, ≤ 4 | 3,659 | 1 / 48 | 76 |
| KO / ZH, 5+ | 2,030 | 6 / 48 | 254 |
| reviewed, several text boxes | 175 | not sampled | — |
| all | 64,550 | | 14,171 |

95 % interval of the total 12,883–16,630 (Jeffreys per stratum, summed by sampling).

- Claude's read of the 48 sampled proposals against their crops: 17 break in the wrong
  place. 12 of the 17 join a column of only dots or marks to its neighbour.
- Val: 9,226 of 11,634 mains have no `\n`. By the same read: 47 / 48 of 3,763 at 9+
  characters, 17 / 48 of 2,308 at 5–8, 0 / 48 of 3,155 at ≤ 4; estimate 4,502 (95 % interval
  4,137–4,869).
- `manifest_mainsub_text` holds the train split only: the val boxes outside the speech
  manifest never went through `linebreak_auto.py`. Val is not on the review page.
- `exact` is whitespace-blind.

### Interpretation

- SFX and labels of up to 4 characters are mostly one line as labelled; the missing breaks
  sit in the waiting proposals and in longer speech.
- The proposals are not safe to take in bulk.
- Val's gap probably follows from the pass that never ran on it. `exact` does not see the
  gap, so the val scores so far are not moved by it.
