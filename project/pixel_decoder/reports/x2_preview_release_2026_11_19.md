# ×2 preview — run 1 / run 2, render read and deployment (2026-11-19)

The release side of the ×2 work, moved out of `x2_plan.md`: the two runs the preview rests on,
the render read, the user's decision to ship the ×2 file as a test file, and what was deployed.
Arms and open questions stay in `../x2_plan.md`.

## Plan as agreed

### Facts

- The user, on replacing the release head with the real-target head
  (`real_target_2026_11_19.md`): "릴리스 헤드를 바꿔도 되긴 하는데 흠... x2 preview 느낌으로
  릴리스하는건 어때? x2를 12ms만으로 만드는건 꽤 매력적이라". Then: "본런이랑 x2 스테이지 돌리고,
  라이브러리랑 comfyui 노드 배포는 x2_plan.md 에 적어놓자".
- The user, asked whether `mixr_1119_1455` stays as the default beside a preview file: "본런
  결과 보고 정하지 뭐 옆에 두기보단 교체하는 느낌으로 가자". The new head replaces the release head
  (`anima_pixel_head` in the catalog, the Hub file, the node's default) once the user has read
  run 1; it is not shipped beside it.
- The new head: ×1 on the real-target recipe with the turbo point on
  `anima_turbo_4step_v4_longer`, and the ×2 stage on it. Open until the read: run 1's ×1 file
  with the ×2 file as a second, preview file, or run 2's one file for both scales.
  `x2a_1119_1802` moved the ×1 output about 1 dB while the ×2 stage trained; run 2's ×1 meters
  against run 1's were to say which.
- Run 1, `realr_1119_2200` (×1): `real_1119_2106` with `--rollouts` 1000 → 3000, nothing else
  changed (`--init mixr_1119_1455`, points `0.0577-0.1364:real 0.5:adapter`, `--lambda_lpips
  0.05`, adapter v4_longer). The user's read passed it (commit 260df663).
- Run 2, `x2real_1119_2242` (×2 stage): `x2lt_1119_2013` with `--init` → run 1's head and
  `--adapter` → v4_longer, plus `--points 0.0577-0.1364:real` and `--lambda_lpips_x1 0.05`: the
  ×1 output's loss at the base points is run 1's (the original image + LPIPS) instead of
  VAE(x̂0). The ×2 target is unchanged: the teacher on VAE(x̂0). `smoke_x2real_1119_2205`
  (20 rollouts on `real_1119_2106`) checked the path first.

### Interpretation

- What run 2 cannot separate from `x2lt_1119_2013`: the base head, the adapter and the ×1 loss
  all change.
- Without the ×1 change, run 2's ×1 term would pull the base head back onto VAE(x̂0) and undo
  what the user read on run 1's recipe.
- The ×1 output aims at the original and the ×2 output at the teacher's upscale of VAE(x̂0), two
  slightly different images (about 35 dB apart at σ 0.058), so the two outputs of one file may
  differ in tone.

## Run 2, `x2real_1119_2242`

### Facts

- 1,000 rollouts, 1.49 s per rollout, peak 10.7 GB; head 11.8 ms on 80 × 52 tokens; teacher
  3.11 s per image. Results in `bench/pixel_readout/results/20261119-2322-x2real_1119_2242/`
  (local). Against the teacher, test 12 / text 8 / manga 4, one seed; the chain here is run 1's
  ×1 → teacher:

  | | dB | LPIPS | dark lift |
  |---|---|---|---|
  | ×2 head, base σ 0.097 | 29.74 / 26.70 / 25.94 | 0.132 / 0.158 / 0.169 | +13.8 / +11.7 / +6.6 |
  | chain, base | 32.66 / 28.68 / 26.98 | 0.081 / 0.077 / 0.088 | +8.2 / +6.7 / +4.9 |
  | ×2 head, turbo σ 0.5 | 29.27 / 25.75 / 25.66 | 0.133 / 0.171 / 0.176 | +14.0 / +11.9 / +6.6 |
  | chain, turbo | 32.04 / 27.28 / 26.92 | 0.151 / 0.136 / 0.123 | +7.8 / +7.3 / +4.7 |

  Bicubic: 29.82 / 28.25 / 27.00 (base), 29.55 / 27.90 / 27.03 (turbo). `x2lt_1119_2013` on the
  same sets: 29.83 / 26.84 / 26.10 dB, LPIPS 0.134 / 0.161 / 0.172, dark +13.4 / +11.4 / +6.4.
- Run 2's ×1 output against run 1's (rollout 0 of the same eval = run 1's head): base against the
  original −0.31 / −0.32 / −0.28 dB, LPIPS +0.005 / +0.007 / +0.004; turbo against VAE(x̂0)
  −0.92 / −0.79 / −0.99 dB, against the original flat. Spread unknown.
- Claude's read, labelled crops `base_s0_09705_test_00`, `…_text_01`: the ×2 head is lighter and
  lower in contrast than the chain (hair, bow), lines and iris softer; glyphs legible in both.

### Interpretation

- The ×2 output reads level with `x2lt_1119_2013`: the real-target base head and the v4_longer
  adapter did not move it.
- The ×1 output fell under run 1's, which points at two files: run 1's ×1 as the replacement, run
  2 as the ×2 file.

## Render read

### Facts

- `p3_generated.py --x2` (sheets `VAE → ResShift ×2 | ×2 head | bicubic ×2 of VAE`, labelled,
  768-px crops at the densest window). Smoke `smoke_x2gen_1119_2310` on `x2lt_1119_2013` (base
  28 steps CFG 4, `anima_sign` / `chibi`, seed 0): ×2 29.82 dB against the chain, bicubic 31.97;
  VAE 180 ms + ResShift 3,300 ms against the head's 11.3 ms.
- `x2bic_1119_2335`: run 2's head, base 28 steps CFG 4, seed 510, five prompts drawn at random.
  Against VAE → ResShift ×2:

  | prompt | ×2 head dB | LPIPS | bicubic dB | dark lift | ×1 against VAE |
  |---|---|---|---|---|---|
  | castle | 30.29 | 0.199 | 33.21 | −0.0 | 34.01 |
  | kotatsu | 17.90 | 0.327 | 21.67 | +39.2 | 21.11 |
  | shop_sign_ja | 24.32 | 0.231 | 25.98 | +11.9 | 28.29 |
  | manga_page | 21.59 | 0.217 | 24.01 | +14.1 | 24.89 |
  | manga_action | 18.24 | 0.237 | 20.46 | +20.7 | 22.47 |

  Head 11.3 ms, VAE 152 ms, ResShift 3,134 ms (means). Crops: `~/x2bic/` (local).
- Claude's read on the smoke `chibi` crop (labelled): the thick navy outlines are lighter and
  thinner on the ×2 head, colours washed against the chain.
- The user's read of the `x2bic_1119_2335` crops: "kotatsu에선 bicubic보다 톤을 잘 지키지만 털 세부
  표현은 아쉽고, shop 예시에선 각 책이 bicubic보다 망가진 느낌이 들기도 한다. 하지만 스크린톤 자체는
  bicubic보다 톤을 잘 보존하면서 특유의 질감이 약간 더 나는 느낌이 난다. manga같은 경우 선이 좀 더
  뚜렷해야 할것 같아".
- The user's decision: "x2는 일단 preview로 올리되 컴피에서 나중에 가중치만 정식버전으로 교체해서
  사용할 수 있게하는 느낌으로 즉 테스트 느낌으로 올리자. 굳이 리드미에 막 쓰진 말고".

### Interpretation

- On the user's read the ×2 head keeps tone better than bicubic (kotatsu, screentone) while
  bicubic scores higher against the chain on every prompt; PSNR against the chain does not track
  that read here.
- The ×1 output reads 21 – 34 dB against the VAE decode on these renders, far under the bench's
  ~31 – 36; part is the original-image target (run 1's recipe moves off VAE(x̂0) by design), the
  rest is not separated.

## Deployment

### Facts

- The user, after the briefing: "배포 1, 2, 3의 1번은 진행하고, 3의 2/3/4번은 일단은 결과 나오면 하자".
- Library (commit dd36c8fc): `PixelHeadX2` and `head_scales` in `library/anima/pixel_head.py`
  (the bench imports them back); `load_pixel_head` builds it for a file stamped `x2_depth`,
  `width2` read off `up2_proj.weight`; `PixelTap.decode(…, scale)`; `--pixel_scale`,
  `PIXEL_SCALE=`, `GenerationRequest(pixel_scale=…)`; a ×1 file refuses scale 2 at load; the
  load log line names the scales and the `adapter` stamp. Façade `anima_lora.models.PixelHeadX2`.
  `tests/test_pixel_head.py` (×2 round trip, scale 1 / 2 shapes, ×1 refusal): 9 pass on CPU.
  `docs/anima/inference/pixel_head.md` has a ×2 section. No catalog row for the ×2 file.
- Bench: `p3_generated.py --x2`, `e2e_speed.py --scale 2` (a `chain` arm), `comfy_e2e.py --scale 2`.
- Node `~/ComfyUI-Anima-PixelDecoder` (commit 58ccefe, local): `AnimaPixelDecode` has an
  optional `scale` input (saved workflows still load); `_vendor/pixel_head.py` regenerated by
  `make vendor-sync`; version 0.2.0. README ×2 section cut to three lines naming the test file (`ebbc77b`).
  `make vendor-sync` also rewrote `_vendor/` files in the Spectrum and Adapter node repos and in
  the in-tree `comfyui-anima-directedit` / `comfyui-anima-trainer` (library drift unrelated to the
  head; left uncommitted).
- Hub `sorryhyun/anima-pixel-decoder` (public): `old/anima_pixel_head.safetensors` =
  `mixr_1119_1455` (sha256 27b3f0dc…, the same file as the current main one; commit e280334c).
- `comfy_e2e.py --scale 2` on run 2's file (`comfy_x2_1119_2345`, ComfyUI 0.43.0, turbo
  v4_longer, fp32 stream, P3's 16 prompts, medians): VAE Decode → ResShift ×2 3.18 s (peak 2,351 MB),
  Anima Pixel Decode at scale 2 0.039 s (peak 450 MB); total with the sampler 4.12 s against
  0.98 s. Head against the chain 30.0 dB (mean), LPIPS 0.156 (mean).
- Hub (commit a26e48ae): `anima_pixel_head_x2_preview.safetensors` = run 2's `head_x2.safetensors`
  (sha256 9c3e2ce2…); the card's Use section has one line naming it a test file that will be
  replaced.
- Node 0.2.0 published to the registry and pushed to GitHub (`ebbc77b`).

### Interpretation

- The node loads a head by file name from `models/pixel_heads/`, so a later ×2 file replaces the
  test file without a node release, as the user asked.
- Not done: the main file's replacement by run 1 and the card's numbers for it (the card
  describes `mixr_1119_1455`); a catalog row for the ×2 file; daemon checks of
  `inference.py --pixel_scale 2` on base / turbo / `--int8`.
