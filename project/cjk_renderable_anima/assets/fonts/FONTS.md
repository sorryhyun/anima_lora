# Manga lettering fonts for the S-line renders

Picked from https://oekaki-zukan.com/articles/516 (user, 2026-09-15) — the
comic-lettering set the base model's training captions were drawn in. Binaries
are gitignored (48 MB); this file and `licenses/` are tracked. `find_fonts()`
(`src/common/render/flat.py`) lists Noto CJK plus every `*.ttf` / `*.otf` here, and
`pick_font()` draws only among fonts whose cmap covers the string.

| file | font | role (per the article) | licence | source |
|---|---|---|---|---|
| `GenEiAntiqueNv6-M.ttf` | 源暎アンチック v6.0a | basic dialogue (antique gothic — kana mincho-ish, kanji gothic) | SIL OFL 1.1 (`licenses/GenEiAntique_OFL.txt`) | https://okoneya.jp/font/download.html |
| `GenJyuuGothic-{Medium,Bold}.ttf` | 源柔ゴシック 2015-06-07 | monologue / narration (rounded gothic) | SIL OFL 1.1 (`licenses/GenJyuuGothic_*`) | http://jikasei.me/font/genjyuu/ (OSDN mirror) |
| `GenShinGothic-{Medium,Bold}.ttf` | 源真ゴシック 2015-06-07 | emphasis (angular gothic) | SIL OFL 1.1 (`licenses/GenShinGothic_*`) | http://jikasei.me/font/genshin/ (OSDN mirror) |
| `Corporate-Logo-Bold-ver3.otf` | コーポレート・ロゴ ver3 Bold | playful / childish dialogue | SIL OFL 1.1 (`licenses/CorporateLogo_OFL.txt`) | https://logotype.jp/corporate-logo-font-dl.html |
| `TanukiMagic.ttf` | たぬき油性マジック 1.22 | comedy emphasis, hand-lettered marker (JIS L2 kanji) | free incl. commercial; no resale of the file (`licenses/TanukiMagic_readme.txt`) | https://tanukifont.com/tanuki-permanent-marker/ |
| `CHI-hasenG.ttf` | 破線G (Chiba Design; a dashed-stroke 源ノ角ゴシック derivative, ≈ 8 200 glyphs incl. JIS L2) | decorative / effect lettering | SIL OFL 1.1 (per the BOOTH page; no licence file ships in the zip) | https://booth.pm/en/items/6454019 (BOOTH, pixiv login; user-downloaded 2026-09-15) |
| `koyomiyuru.otf` | こよみゆる v1.000 (iki-font; FontDrawer handwriting, JIS L1 kanji — `pick_font` skips it for L2 strings) | scribbly handwritten dialogue | SIL OFL 1.1 (`licenses/Koyomiyuru_2026_0410_Readme.txt`) | https://booth.pm/en/items/8185365 (BOOTH, pixiv login; user-downloaded 2026-09-15) |
| (system) `NotoSerifCJK-*.ttc` | 源ノ明朝 = Noto Serif CJK | declarations (serif) | SIL OFL 1.1 | installed |
| ~~`NotoSansCJK-*.ttc`~~ | Noto Sans CJK — the pre-2026-09-15 only font | **out** (user, 2026-09-15: reads ambiguous beside the manga faces) | | |

Not fetched: やさしさアンチック (fontna.com page 404 / no link found), 851チカラヨワク
(distribution page empty), しねきゃぷしょん (Vector form download), the horror /
brush / handwriting fonts (off the dialogue-and-SFX distribution we train).

Re-fetch: `curl -L -o x.zip <source>` and unzip the listed face here.
