# Who knows a character's name: the TE or the DiT

Proposal: `docs/moss/proposal/te_character_knowledge.md`. Scripts: `src/entity.py` (TE
reads), `src/tag_recall.py` (tagger recall on generations). Runs 20261027-1615 (entity),
`output/tests/{neru,malfoid,koneko}_phase0/`, `output/tests/miku_promptform/`.

## Claims

1. **The TE's knowledge of a name and the DiT's ability to draw it are separate.** The
   DiT draws Akita Neru from her name although the TE does not know her.
2. **The method's targets are characters the DiT cannot draw from the name**, not
   characters the TE cannot describe. `malfoid` and `toujou koneko` are such cases, and
   spelled-out appearance tags fix them (recall 0.29 → 0.83, 0.06 → 0.83).
3. **The prompt form does not move a DiT-known character.** A question or a CoT line
   around the caption changes less than the seed does.

## 1. The TE does not know Neru; the DiT draws her

`src/entity.py`, bf16 instruct TE, greedy, thinking off. Three reads place Neru with an
invented name (`okuda reni`):

- QA: "a Touhou shrine maiden, black hair, red ribbon"; `亞北ネル` → "Danganronpa 2,
  pink hair". Miku is correct; Kasane Teto half wrong.
- Cloze `X is a derivative character of the`: `' Vocal'` 0.68 (Miku), 0.47 (Teto),
  absent for Neru and for the invented name. No blonde or yellow in Neru's hair-colour
  top 6.
- Base-SAE top-100 overlap at the name's last token: Neru ∩ invented 60 / 54 at L31
  (47 / 66 at L15), above Neru ∩ Miku 47 / 45 (25 / 31). The invented name sits in the
  same caption slots, which inflates its overlap, and the strongest features at name
  tokens are structural (JSON closers, a generic "Japanese" feature), so the QA and cloze
  reads settle it.

Yet Base MOSS renders Neru from the name alone (832×1216, seeds 11 / 22 / 33, Spectrum
off): visually Neru in all three, tagger recall 0.55 on her 17 solo-post co-occurrence
tags; 0.94 with the tags written out. The DiT memorised her name tokens in pretraining.
Whatever goes wrong with her in practice (multi-character binding, the channel LoRA) is
not TE knowledge.

## 2. DiT-unknown characters, and what fixes them

Recall = Anima Tagger hits on the character's solo-post co-occurrence appearance tags
(Danbooru, through gelcrawl's session) that are in the tagger vocab. Spectrum on.

| condition | malfoid (8 ref tags) | toujou koneko (6 of 8 in vocab) | akita neru (17) |
|---|---|---|---|
| name only | **0.29** (generic Hogwarts students) | **0.06** (DxD's style, other girls) | 0.55 |
| name + co-occurrence tags | 0.83 | 0.83 | 0.94 |
| invented name + the same tags | 0.75 | 0.83 | 0.88 |
| + a wiki sentence instead of tags | — | — | 0.82 |
| control | `draco malfoy` + genderswap: male Draco, 0.17 | — | Miku name only: Miku ×3 |

- `malfoid` is a recent meme, absent from the 2024 KB and the 2026 wiki dump: unknown to
  the DiT, fixed by tags. The 0.29 → 0.83 gap is what the method has to recover.
- `toujou koneko` has thousands of Danbooru posts and is still unknown to the DiT: it
  draws *High School DxD*'s style from the series name but not her. The TE does not know
  her either (QA: a white-furred fox from *Kemono Friends*; hair cloze matches the
  invented name). Rias Gremory, run alongside: series right, no red in the hair cloze.
  Post count does not predict what MOSS saw.
- Tags beat the free-text wiki sentence (0.94 vs 0.82 on Neru).
- The name adds little over the invented name once the tags are there (0.83 vs 0.75–0.83).
- A hosted chat's claim that the TE knows a character does not count: Koneko came
  recommended that way and the local probe says no. Candidates need `entity.py` (greedy,
  thinking off, this checkpoint) before they are TE-known. No character that the TE
  knows and the DiT cannot draw (the proposal's case ②) is found yet.

## 3. Prompt form on a DiT-known character

The phase-0 Miku caption, Spectrum on, seeds 11 / 22 / 33, 19 canonical outfit tags:

| form | recall (per seed) | mean prob |
|---|---|---|
| plain | 0.67 (0.37 / 0.68 / 0.95) | 0.56 |
| `What does hatsune miku look like in this picture?\n` + caption | 0.46 | 0.48 |
| caption + `\nWhat does hatsune miku look like?` | 0.77 | 0.61 |
| caption + `\nLet's think step by step.` | 0.65 | 0.54 |

All twelve images are canonical Miku and the seed sets the composition. The spread across
seeds (0.37–0.95) is larger than across forms; +0.10 is about 2 of 19 tags. The prefix
scores lowest: in a causal TE it changes every caption row, a suffix only the rows after
the caption. Nothing here suggests a question or a CoT line helps a one-forward TE.
