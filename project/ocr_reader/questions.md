# ocr_reader — open questions

- How much KO/ZH did v3 actually lose against stock VL-1.6? (P0)
- Is the v3 tower update usable at low rank, or is its spread spectrum a
  functional requirement? (P2a)
- Does an LM full FT keep COO speech while it learns new scripts, or does the
  narrow SFX label set erode the language prior? (P2b)
- Which licence-clean KO / ZH sources exist for hand-lettered text (webtoon /
  manhua SFX)? Is synthetic rendering (`ocr/synth_sfx.py`, unwritten) a
  stand-in where none exists?
- Does ZH need simplified and traditional handled separately? K0 showed hayai
  emits simplified on only 75.6 % of `zh` rows.
- Should the heart positives (the open lever from the finished line) ride
  along in v4's data mix?
