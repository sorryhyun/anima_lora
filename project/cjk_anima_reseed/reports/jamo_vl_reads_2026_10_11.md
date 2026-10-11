# The jamo arms, read by v4 glyph by glyph — 2026-10-11

User 10-11: the v4 reads of the jamo arms (`jamo_curriculum_2026_10_11.md` § 6)
laid out so each syllable can be looked at. Source: `results/20261011-1409-jamo-vl-score/`
(v4 = anime_tools 0.7.10 `SfxReader`, plus stock and the forced read; seed 0
renders, each target alone in a KO bubble). Sheets and table:
`results/20261011-1424-jamo-vl-sheet/` (`probes/jamo_vl_sheet.py`).

Not here: the trained 64 (J64 / F64) — that score run read H and the words only.

## 1. Totals on H (32 held-out common syllables)

| arm | v4 free exact | v4 forced top-1 | v4 forced top-5 | whole-image read one syllable sharing a jamo (~) |
|---|---|---|---|---|
| J64 | 1 | 1 | 3 | 6 |
| J96 | 2 | 2 | 4 | 4 |
| J128 | 3 | 4 | 5 | 2 |
| J64-lone | 1 | 2 | 3 | 3 |
| F64-reg | 0 | 0 | 1 | 1 |

The reader's ceiling on font-drawn H is 71 % (free), so a miss here is mostly
the render. The words (정리 소리 아래 문제 시장): 0 / 5 under every arm.

F1 of what was drawn against the target (user 10-11: correct over drawn). The
whole-image read's letters (punctuation dropped) as a multiset against the
target's; P = matched / drawn, R = matched / the target's, F1 per image, then
the mean over the 32. On glyphs, and on positioned jamo (cho / jung / jong;
식 for 시 scores P 2/3, R 1). The last column scores the same reads against
every *other* H target: the jamo two syllables share by chance.

| arm | glyphs drawn | glyph P | glyph R | glyph F1 | jamo P | jamo R | jamo F1 | jamo F1, other targets |
|---|---|---|---|---|---|---|---|---|
| J64 | 3.31 | 0.03 | 0.03 | 0.03 | 0.20 | 0.46 | 0.25 | 0.12 |
| J96 | 3.59 | 0.06 | 0.12 | 0.07 | 0.19 | 0.48 | 0.25 | 0.12 |
| J128 | 3.75 | 0.05 | 0.12 | 0.06 | 0.20 | 0.56 | 0.27 | 0.13 |
| J64-lone | 3.88 | 0.05 | 0.12 | 0.07 | 0.18 | 0.50 | 0.23 | 0.13 |
| F64-reg | 5.69 | 0.01 | 0.06 | 0.02 | 0.10 | 0.54 | 0.15 | 0.14 |

- Jamo F1 is twice chance under every J arm (0.23–0.27 vs 0.12–0.13);
  F64-reg sits at chance (0.15 vs 0.14). Its jamo R (0.54) is as high as
  J128's only because it draws ~6 letters an image.
- Glyph F1 is near the floor everywhere (≤ 0.07): the target syllable is
  rarely drawn, and when it is, beside other text (glyph R 0.12 vs exact
  free 2–3 / 32 on J96 / J128: the target inside a longer read).
- J64 → J128 moves recall (jamo R 0.46 → 0.56, glyph R 0.03 → 0.12), not
  precision (jamo P 0.20 flat): more of the target, with as much else drawn.
- Words (5 items) are too few to rank; their table is in `reads.md`.

## 2. By glyph

The sheets: `sheet_H_0..3.png` (8 syllables each), `sheet_words_0.png`. Each
cell is the render with v4's whole-image read under it; green = a crop reads
the target exactly, amber = the whole-image read is one syllable sharing a
jamo with the target.

Table (the whole-image read; **✓** exact on some crop, ~ shares a jamo):

| | j64 | j96 | j128 | j64_lone | f64_reg |
|---|---|---|---|---|---|
| 리 | 덜개 | 쇼럴어 | 혜토, 위 | 선 엘 게 | 덜개 |
| 정 | 뭐 컵 | 잘 ~ | 밀게 | ヨ。 | 요 댁다 꾸우잘 아네도 |
| 시 | 식 ~ | **취도 시** ✓ | **쳐늘 시** ✓ | 서 채얼어슨 러엉미수 | 시 저일 극란나 |
| 어 | 일거 | 멍 ~ | 희 | 밀지 | 어 로 우자저돈…? |
| 인 | 콩 | 어온. 뒤 | 어조 오알데 두란바 | 예 ~ | 왜 ~ |
| 일 | 밀자여 | 비잘 울가아 ~~ | 멀자 이거 | 일저 얼어 | コ이 진응 어새든대 |
| 성 | 제도 뭐 | 상 죄을 낼테아도 | **성** ✓ | 미실 글겔주 | 쇼할아 |
| 으 | 에늘 로 | 로 | 즘 ~ | 크 ~ | 밑질 잔등안나. ~ |
| 은 | 응 팬을 설 안다도 | 스율 잔상일 대조은 … | 은 댄을식에이도 | 은 으 괴을 새대아도 | 코안 건잘일 내뭐른 |
| 해 | 익혈… 로자대… | 쇼렬어 | 히한 새일이순 쳐없비슨 | 헐 ~ | 익얼 코지마. |
| 고 | 코 ~ | 포하라 | 헉죠 구 | 고 악다 | 금 잔을 샀아이도 |
| 습 | 좀 | 조 고암 레아 | 졸지데토도 | 리슨 콜지 댁오오 | 세잔 긴응램츸 ↵▶. |
| 요 | 응 깰바 | 팩 표젠온 막 | 실저 이라 | 엘서 먹자… | 오렬 코킨주~ |
| 제 | 기탠 맥전글 | 전인우져 허험개돌 | 세실거론 | 잔안우져 퍼일개도 | 민살가은 헄주 (트리) |
| 우 | 윤 즉한바 | 쥬 | コキ | 오를 로란아 | 익킬 극란라 ~ |
| 장 | 문 절 | **장** ✓ | 징이 | 혹 | ∅ |
| 세 | 점 | 지를 진 | 선혈저 | 척일제악 | 센살지큰가아. |
| 번 | 란 ~ | 쇼럴너 | 낀실 올거마. | 버언록 | 0나 타검 몰줬라 찍… |
| 소 | 낸살 집무킷수 응을 | 소래아 | **소 언잘설 래워든** ✓ | 받잘 잠드인수 로오음 | 셔직 간응안츠~ |
| 입 | 틸이 | 일자 이각 | 킬아 | 덜에 지탁은벌아 | 엘개 |
| 호 | 존 ~ | 즇 | 헉 ~ | 층야 | 모알 두란네 |
| 할 | **할** ✓ | 팔 ~ | 학력 | 알 절… | 민살사온 엔주~ |
| 도 | 루 | 씹을 드 | 잔안후대. 쩍오 개돌 | 낀살 사은 짠아. | 낀살 사온갠어. |
| 음 | 오댁우 | 몰여 | 얼금 | 쇼혈어 | 쇼할아 |
| 그 | 코너… 갈달여… | 익멸 ᄏᄏ 잘데우 튼 | 모긴 쓸 에돔튼 | **그** ✓ | 언찰일 대우두 |
| 체 | 일지 | 촌 ~ | 선혈지 | 엘제 | 센질 잡음한나. |
| 록 | 지녔해 링오랔에 나.… | 적얀 계독 | 잔안무대. 쩍익제돌 | 간잉토 잘 쩍암지돌 | 건잘일 대우은 |
| 라 | 째 | 칼 | 재. 뭐 | 헉 | 컵 |
| 문 | 단 벌 응거서든 째 | 로존 언잡설 네위든 | 센잘이른 아들 | 쇠할더 글린주 | 거 젠 렬 모우셔애갈 |
| 아 | 히 | 퍍 | 자 | 쫓어 애 | 외알 득젠나. |
| 래 | 럼 ~ | 비살가온가머. | 렬게 | 혹귀 | 요일 쇼알 극란내 |
| 열 | 걸 ~ | 비살 윤자파~ | 일지 | 일지 | 낀살자온 앙나. Ce… |
| **✓ / ~** | **1 / 6** | **2 / 4** | **3 / 2** | **1 / 3** | **0 / 1** |

## 3. What the sheets show

- The exact hits sit on the simplest blocks: 시 (J96, J128), 성 / 소 (J128),
  장 (J96), 할 (J64), 그 (J64-lone). No arm gets the same one twice except 시.
- J64 draws a lone big glyph most often (its "one syllable" lead in the
  curriculum report), but a neighbour block: 식 for 시, 코 for 고, 란 for 번,
  럼 for 래, 걸 for 열 — the right shape class, one jamo off.
- J128 hits more and draws text more. Whole-image reads of one glyph / of 4+
  glyphs: J64 13 / 8, J96 10 / 13, J128 5 / 14, J64-lone 7 / 14, F64-reg
  2 / 25 (어조 오알데 두란바 for 인 under J128).
- F64-reg (the jamo model fitted to free rows) draws dialogue text on nearly
  every cell — the free rows hold no composable jamo, as R3 said.

## 4. J64 at J128's budget (`jamo_j64_112`)

`configs/jamo_j64_112.toml`: J64's 64 syllables and data at 112 steps a row
(7 168 = J128's total; the factor model is the same 106 vectors). H and the
words rendered (`results/20261011-1511-jamo-read/`), all arms re-read by v4
free (`results/20261011-1511-jamo-vl-score/`: the old arms' reads identical,
185 / 185), sheets `results/20261011-1512-jamo-vl-sheet/`.

| arm (total steps) | free exact | one glyph / 4+ glyphs | glyph R | jamo P | jamo R | jamo F1 | chance |
|---|---|---|---|---|---|---|---|
| J64 (3 584) | 1 | 13 / 8 | 0.03 | 0.20 | 0.46 | 0.25 | 0.12 |
| **J64-112 (7 168)** | **0** | 8 / 14 | 0.06 | **0.12** | 0.40 | **0.17** | 0.13 |
| J128 (7 168) | 3 | 5 / 14 | 0.12 | 0.20 | 0.56 | 0.27 | 0.13 |

At one budget, wider beats deeper: J64-112 falls below J64 on H (jamo F1
0.25 → 0.17, near chance 0.13; precision 0.20 → 0.12) and draws more text,
while J128 at the same steps holds precision and gains recall. More steps on
the same 64 syllables cost the transfer — the factors fit the 64, not the
jamo. Not read: the trained 64 under J64-112 (whether it gained what H lost).

## 5. Next

The scaling direction is more syllables, not more steps a row. Next D128
(common syllables, the most varied jamo pairs) against J128 at one budget,
then J256 (proposal_jamo § 3).
