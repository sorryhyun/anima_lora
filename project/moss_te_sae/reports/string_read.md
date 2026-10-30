# How the DiT reads a text string: a quoted sentence in t2i, and the swap_text LoRA

Scripts: `src/quote_attn.py` (20261029-2236, `output/tests/quote_attn/`),
`src/swap_text_attn.py` (20261029-2324, `output/tests/swap_text_attn/`). Each folder has
the per-arm PNGs, a grid / crops image, latents, `summary.json`.

## Claims

1. **A string's rows carry what to write; the rest of the caption carries that a balloon
   exists, where, and in which script.** Cutting the string keys re-letters the balloon
   in place, in the right script, with nonsense words. Keeping only the string keys and
   the template writes the words on the default figure.
2. **The read is local and through fixed heads.** Image queries inside the lettering
   attend to the string's rows 7–20× more than queries outside it, through the same heads
   on every prompt (b13 h20, b21 h20, b24 h11, b26 h12), most on blocks 19–21.
3. **The swap_text LoRA reads each item's string the same way.** A box attends to its
   own string 9× more per token than to the other items' (16–19× on blocks 19–25), peak
   19–21, the same heads near the top. Cutting those four heads alone changes almost
   nothing: the read is spread over more heads.
4. **The suffix rows carry no string**, as they carry no tags (text_read_path.md). The
   template takes ~0.29 of the mass inside and outside the lettering alike.

Short strings leak: the rows after a string (`\"}}`, the suffix) read it in the causal TE
and the DiT takes some of that (`MY COMPUTER!` survives the cut on one prompt; `벌거 수기`,
`출…더`, `SOINING TO` in swap_text). The string's rows are the main route, not the only
one.

## 1. A quoted string in t2i (Base)

Setup as text_read_path.md (Base, Euler 30, CFG 3, seed 1, 1024², layout JSON, INT8 TE,
caststation trigger) on the Miku and Neru A/B prompts with `, speech bubble. She is saying
"<s>".` appended, `MY COMPUTER` (2 tokens) and `지승현의 컴퓨터` (5 tokens). The layout
converter keeps the sentence in `summary.caption`. The text region is the AnimeText box
on the base render (21–128 image tokens).

Mass on the string's rows, region queries against the rest, mean over the 4 prompts:

| blocks | region | rest | ratio | top head (region) |
|---|---|---|---|---|
| 0–7 | 0.0001 | 0.0001 | ~1.5 | ≤ 0.006 |
| 8–13 | 0.0032 | 0.0002 | 11–16 | b13 h20, 0.17–0.35 |
| 14–18 | 0.0032 | 0.0006 | 4–6 | b16 h14, 0.05–0.09 |
| **19–21** | **0.020** | 0.0017 | 8–14 | **b21 h20, 0.19–0.39** |
| 22–25 | 0.0070 | 0.0008 | 4–11 | b24 h11, 0.13–0.28 |
| 26–35 | 0.0056 | 0.0004 | 8–21 | b26 h12, 0.07–0.22 |
| 36–40 | 0.0031 | 0.0001 | 6–47 | ≤ 0.022 |

Per token the string outweighs the other caption rows inside the region 3.3–7.3×
(English 4.6–7.3, Korean 3.3–3.4). The heat maps (`heat_p*.png`) put the string mass on
the balloon's lettering with a faint speckle elsewhere.

Causal arms, read glyph by glyph (bold = differs from the string):

| prompt | base | `swap_suffix` (suffix from the other string's prompt) | `no_quote_keys` | `quote_only` (template + string rows only) |
|---|---|---|---|---|
| Miku EN | MY COMPUTER | MY COMPUTER | **NOW MANGIECH?** | MY COMPUTER, default figure |
| Miku KO | 지승현의 컴**푸더** | 지승현의 컴퓨터 | **비켠 칠치 가 / (쟁지 컴퓌므…)** | 지승현의 컴**뮤**터, default figure |
| Neru EN | MY COMPUTER | MY COMPUTER | MY COMPUTER**!** | MY COMPUTER, default figure |
| Neru KO | 지승현의 컴**뮤더** | 지승현의 컴**무**터 | **윈항에을 클엄택에 다미으** | scribble, no Hangul; a hooded figure |

px against base (0–255), text box / rest: `swap_suffix` 1.8–38 / 3.8–20, `no_quote_keys`
25–50 / 3.6–14, `quote_only` 40–52 / 63–79.

- The jamo that change under `swap_suffix` (`퓨` ↔ `푸` / `뮤` / `무`, `터` ↔ `더`) change
  between base renders too; Base gets the Korean string at ~2 wrong jamo per render.
- `no_quote_keys` keeps the balloon where it was, in the same script, beside the same
  character (rest px 3.6–14): English pseudo-words for English, Hangul nonsense for
  Korean.
- `quote_only` makes the character the default figure of text_read_path.md's
  `no_caption_keys` and keeps the words in 3 of 4. The two Korean renders differ (a
  near-exact line vs a hooded figure and a scribble), so the string's rows also carry some
  of the preceding caption.

Against `docs/moss/findings/bbox_grounding.md`: a boxed layout item's content follows
its box mostly in blocks 26–35 (b14 h4), with the location built by the text refiner
from the coordinate tokens. Here there is no box: the DiT places the balloon itself and
reads the string most on 19–21 through b21 h20. Whether a boxed text item is read
through the same heads is not checked.

## 2. The swap_text LoRA

Four held-out pages of `output/tests/moss_i2i_swap_text_region/` (en 2, ko 2; 11 text
items) with their staged prompts and JA references (full size, isolated, `ref_timestep
0`). LoRA `moss_i2i_swap_text_en200_ko800_rw2_mod16_ema5e5`, `scripts/moss/generate.py`
defaults (ER-SDE 50, Spectrum, CFG 3, uncond skip 22, seed 0, INT8 TE), eager; the `lora`
arm reproduces the shipped renders at px 0.6–1.2. Queries in each item's box and 1024
outside every box are scored against image, template, the rest of the caption, each
item's string and bbox list, the reference, and the reference token at the query's own
cell. (A negative coordinate merges with the key's `":[` into one token and counts as
coordinate.)

Box queries, mean over pages and items:

| blocks | own string | own / other string, per token | own coord | reference | template |
|---|---|---|---|---|---|
| 0–7 | 0.0002 | 1.2× | 0.004 | **0.29** | 0.20 |
| 8–13 | 0.005 | 5.7× | 0.003 | 0.13 | 0.30 |
| 14–18 | 0.006 | 3.3× | 0.014 | 0.09 | 0.27 |
| **19–21** | **0.042** | **15.8×** | 0.009 | 0.03 | 0.42 |
| 22–25 | 0.019 | 18.6× | 0.006 | 0.02 | 0.32 |
| 26–35 | 0.022 | 10.5× | 0.019 | 0.03 | 0.35 |
| 36–40 | 0.016 | 4.8× | 0.018 | 0.12 | 0.17 |

- Each box is bound to its own string: all other items' strings together get
  0.002–0.006 on blocks 8–40, and queries outside every box give the strings
  0.0002–0.002. The box also reads its own coordinates more than any other item's.
- The blocks split the work: 0–7 read the reference (0.29, 0.015 on the own cell), 19–35
  read the strings while the reference drops to 0.02–0.03, 36–40 go back to the
  reference (0.12).
- Top heads by own-string mass on all 4 pages: b21 h20, b24 h11, b19 h6, b28 h0, b21 h3;
  b26 h12 and b19 h19 on 3, b13 h20 on 2.

Causal arms (✓ = as captioned):

| item | lora | `no_string_keys` (all string tokens cut) | `heads_off` (only the four t2i heads cut) |
|---|---|---|---|
| Uwah… | ✓ | AWAN NU | ✓ |
| I'M GOING TO MOVE, OKAY? | ✓ | GIVN SOINING TO S: ROLA SKRE. | ✓ |
| UGH… OH SAKURA-CHAN… YOU'RE THE BEST… MMM… | ✓ | INNIA'S WHEP MENGHA IMAIY BHASN ♥ | `THE` dropped |
| YOU REALLY DRANK MY ESSENCE, HUH? | ✓ | UKT MRENY THE IMUY BOITT I ME PLUS! | ✓ |
| 어머? 처음 오시는 분인가요? | 분**이가**요 | 녀믜을 일기가는 쩜다요♡? | **처 처음** … 분**이나**요 |
| 저기… 이건 뭔가요? | ✓ | 벌가… 얼권 젤마요요? | 이**전** |
| 벌거 주기? (OCR misread) | 벌**겨** | 벌거 **수**기♡ | 벌**겨** |
| 싸… 싼다 | ✓ | 나… 만가 | ✓ |
| 좀…더 | ✓ | **출**…더 | ✓ |
| … 아타고씨 | ✓ | 장써 멋로3념 늑레리♥ | ✓ |
| 굉장해… 질 속이 정액으로 가득해… | ✓ | 췌슥레♥ 핫징우를 께븍레♥ | ✓ |

- Without the string keys every balloon is re-lettered in place: the Japanese is gone,
  the script matches the caption's language, line count and the final `?` mostly
  survive. Hearts come back from the reference where the JA lettering had `♡`.
- With the four t2i heads cut the errors go from 2 to 4 (`THE` dropped, `이건` → `이전`,
  which earlier swap_text runs also made, a doubled `처`). With one seed that is not
  separable from noise.

Limits: one seed per run; four prompts / four pages; attention mass is not influence;
the cut arms cut every block at once. No Base forward was scored on the swap_text
trajectory, so whether the LoRA built the read or reuses Base's is open (Base reads t2i
strings on the same blocks and heads; Base binds a boxed item to its box on 26–35).
