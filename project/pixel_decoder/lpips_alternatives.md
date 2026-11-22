# LPIPS alternatives: the network behind the head's perceptual term (2026-11-22)

What LPIPS is in this line's loss, what the papers say about replacing its VGG with a newer
network, the candidates, and the arms. Opened on the user's question (2026-11-22): "lpips는 구형
vision foundation model based metric이잖아 왜 계속 쓰는거야? dino-v3-s based 를 쓰면 안되나?"

## LPIPS in this line

### Facts

- ×1 (`configs/pixel_head/base_x1.toml`): rec + 0.05 · LPIPS-VGG on the `:real` points, one
  random 512-px crop per update. ×2: rec + 0.1 · LPIPS + 1.0 · dc.
- λ 0.05 on ×1 was sized at about 3× the reconstruction term's gradient norm over the head
  (unweighted ratio 71, 8 rollouts on the release head; `archive/real_target_2026_11_19.md`,
  local). λ 0.1 on ×2 is about 3× rec2's (`archive/x2_lpips_dc_2026_11_19.md`, local).
- Every held-out table of the line reads dB / LPIPS, and the GAN discriminator is a second
  frozen network, PE-Spatial-B16 (README, "GAN term").
- The LoRA term (`library/training/pixel_lpips.py`) is LPIPS-VGG through a VAE window.

## The papers

All read from their PDFs.

### Facts

LPIPS (Zhang et al., arXiv:1801.03924):

- Features of SqueezeNet, AlexNet or VGG, with a linear weight per channel fitted to BAPPS: 484k
  human judgments on 64 × 64 patches under traditional and CNN-made distortions.
- Supervised, self-supervised and unsupervised (stacked k-means) networks all agree with the
  human judgments far better than PSNR / SSIM; a randomly initialised network does not.

Kumar et al. (arXiv:2203.04946, TMLR 2022):

- Across ResNets, EfficientNets and ViTs, ImageNet accuracy and the BAPPS perceptual score are
  inversely related past moderate accuracy (20 – 40 % top-1 holds the best scores).
- They find no correlation between a model's perceptual score and its sensitivity to
  distortions, and no greater reliance on high spatial frequencies in the low-score models.

ViTok-v2 (arXiv:2605.05331):

- Autoencoder loss `Charbonnier + 0.1 · SSIM + λ · L_DINO`, no LPIPS and no GAN. L_DINO: random
  224-px tiles at the same places of the input and the reconstruction, frozen DINOv3-S,
  features of several intermediate blocks, L2-normalised per token, MSE. λ 500 or 1000. The
  blocks and the tile count are not given.
- Table 2 (5B decoder, r 12, ImageNet 256), rFID / rFDD / SSIM: pixel-only 5.13 / 10.96 / .929;
  + LPIPS 0.72 / 2.93 / .923; + DINO λ 250 0.51 / 1.45 / .919; + DINO λ 1000 0.30 / 1.12 / .914;
  + LPIPS + DINO 0.38 / 1.35 / .918. rFDD is a Fréchet distance on DINOv3 features. One LPIPS
  weight, not stated.
- Against an adversarially trained decoder: FLUX.2 rFID 0.15 on ImageNet 256, ViTok-v2 0.30 at
  λ 1000; on COCO 512, 0.45 against 0.38 – 0.41.
- Data: about 2B images (DataComp, YFCC-100M, Shutterstock, ImageNet-22K, LAION); evaluation
  on ImageNet, COCO, DIV8K.

DINOv3 (arXiv:2508.10104), the distilled family (all from the 7B teacher, LVD-1689M):

| model | params | feature spacing | ADE20k / NYU↓ |
|---|---|---|---|
| ViT-S/16 | 21M | 16 px | 47.0 / 0.403 |
| ViT-S+/16 | 29M | 16 px | 48.8 / 0.399 |
| ViT-B/16 | 86M | 16 px | 51.8 / 0.373 |
| ConvNeXt-T | 28M | 4 / 8 / 16 / 32 px | 42.7 / 0.448 |
| ConvNeXt-S | 49M | 4 / 8 / 16 / 32 px | 44.8 / 0.432 |

- The ViT rows are Table 14, the ConvNeXt rows Table 15 (ADE20k at 512, NYU at 640): two
  protocols. The ConvNeXt parameter counts are from the Hub files' sizes.
- GFLOPs at 256 / 512 px (Fig. 16a): ViT-S 12 / 63, ViT-B 47 / 216.
- LVD-1689M: a clustering-balanced subset of 17B public Instagram images, a part retrieved
  around seed benchmark datasets, and raw ImageNet-1k / 22k and Mapillary.
- Licence: DINOv3 License; the Hub repos are gated. `transformers` 5.16.1 has `DINOv3ViTModel`
  and `DINOv3ConvNextModel`.

Pixio (arXiv:2512.15715):

- An MAE with a deeper decoder, larger mask blocks and more class tokens, trained on 2B
  web-crawled images collected as MetaCLIP does. Curation: sampling by reconstruction loss
  (easy images such as product shots drawn less often) and a filter on low colour-histogram
  entropy "to reduce text-heavy images".
- The released B / L / H / 1B are students of the 5B model, trained on its patch and class
  tokens with a cosine loss (§7.2). The smallest is ViT-B/16, 86M.
- The deeper decoder is there so that the encoder's last blocks stop modelling low-level
  detail (§3); with it the last block's features are within 0.006 RMSE of the best block on
  NYUv2 (§8.1).
- Evaluated on depth, 3D reconstruction, segmentation and robot learning. No experiment uses
  it as a loss.
- Licence: FAIR Noncommercial Research License; gated.

Web-SSL (Fan et al., arXiv:2504.01017):

- DINOv2 (Web-DINO) and MAE (Web-MAE) trained on the images alone of MC-2B, MetaCLIP's 2B
  web images, at 224 px with no high-resolution stage. The paper covers 1B – 7B parameters and
  reads the encoders through VQA, with ADE20k and NYU beside it.
- No filter against text. Two filtered sets are trained on as well: light (the 50.3 % of
  MC-2B with any text in the image) and heavy (1.3 %, charts, tables, documents); OCR & Chart
  VQA rises with the filter. Web-MAE reads OCR & Chart above Web-DINO and the other VQA
  groups below it.
- On the Hub, smallest first: `webssl-dino300m-full2b-224` and `-light2b-224` (ViT-L/14,
  1.2 GB), `webssl-mae300m-full2b-224` (ViT-L/16, 1.2 GB), `webssl-mae700m-full2b-224`. The
  300M and 700M sizes are not in the paper. Licence CC-BY-NC 4.0; not gated.

### Interpretation

- LPIPS is an empirical construction, a classifier's features with weights fitted to human
  judgments on small photo patches. It has the standing of any feature distance. It persists
  because it is the number every table reports, and because the L1 + LPIPS + GAN balance is
  inherited from one codebase to the next.
- ViTok-v2 is the one published result of a DINOv3 term in LPIPS's place, and it is with
  ViT-S/16. It answers two objections to a patch-16 ViT as a low-level loss: the tiles fall at
  any pixel offset, so the patch grid is not fixed on the image, and intermediate blocks are
  read, as LPIPS reads VGG's.
- Its comparison is not at matched distortion (SSIM .923 against .919 / .914), and rFDD shares
  the term's network. rFID is the independent column, and it moves the same way.
- ConvNeXt-T has the pyramid LPIPS has (maps finer than the head's 16-px token) and takes any
  image size, with no published use as a loss.
- Pixio (the user, 2026-11-22: web-crawled data may sit closer to this domain than DINOv3's).
  Its data is the less curated of the two, which favours that. Against it for a first arm: the
  low-entropy colour filter removes the kind of image a manga page is, the released sizes are
  feature-distilled students, the smallest is 4× ViT-S, and the licence is non-commercial while
  the head is published.
- Web-SSL in Pixio's place (the user, 2026-11-22: "그럴거면 차라리 web-ssl이 낫나? pixio대신"). It
  fits the data argument better: the same web pool with no filter against text, and a variant
  trained on the half that has text. It also gives a DINO and an MAE of one size on one
  dataset, so a data effect and an objective effect can be read apart, which Pixio cannot do
  (data, objective and distillation change together against DINOv3). Its MAE is the plain
  form, whose last blocks still carry low-level detail by Pixio's own account. Against it: the
  smallest is ViT-L, 14× ViT-S, trained at 224 px only (the tile's size), and the licence is
  non-commercial as well.

## Arms

### Facts

- Code: `DinoFeat` (`library/anima/pixel_head_train.py`), the term and its weight rule in
  `X1Trainer` (`library/anima/pixel_head_train_x1.py`) and `X2Trainer`, the fields `dino`, `lambda_dino`,
  `dino_tiles`, `dino_match_lpips` and the read columns `dino_*`, `lpips_null`, `dino_null` in
  `scripts/pixel_head/_x1.py`. Test: `tests/test_pixel_dino.py`.
- Run 2026-11-22: `x1dv3s_1122_2240` (DINOv3 ViT-S/16), `x1lp_1122_2240` (the LPIPS control),
  `x1dv3sb28_1122_2240` (generated renders through both heads). Design, question, read and
  bars: the header of `configs/x1dv3s_1122_2240.toml`. Result:
  [the report](reports/x1_dino_2026_11_22.md).
- The user (2026-11-22): "ㅇㅋ 해보자 s가 이기면 convnext를 다음으로 해보는걸로".
- The weights are loaded from the Hub cache by repo name; there is no catalog row.

Test / text / manga at σ 0.0577 against the original, dB / LPIPS; generated renders, colour /
mono, high-pass share of the VAE's and the 16-px stamp:

| run | term | held-out | high-pass | stamp | verdict |
|---|---|---|---|---|---|
| `x1lp_1122_2240` | LPIPS-VGG 0.05 | 33.87 / 0.0786, 28.77 / 0.0655, 26.13 / 0.1033 | 0.881 / 0.853 | 0.43 / 0.97 | baseline |
| `x1dv3s_1122_2240` | DINOv3 ViT-S/16 tiles, weight matched to LPIPS 0.05 (λ 331.6) | 33.65 / 0.0801, 28.68 / 0.0624, 26.03 / 0.0931 | 0.912 / 0.881 | 0.62 / 1.33 | passed |
| `x1dcnxt_1122_2355` | DINOv3 ConvNeXt-T stage maps, the same rule (λ 12.25) | 33.92 / 0.0966, 28.78 / 0.0722, 26.14 / 0.1149 | 0.876 / 0.858 | 0.43 / 0.96 | not passed |

- The user (2026-11-22, after the `x1dv3s` pair sheets): "ㅇㅇ convnext-T 해보자 내생각에 이걸 lpips
  대체로까지 고려할만할듯?"

On ×2 (B only from the ×1 base, 2,000 rollouts; [the report](reports/x2_dino_2026_11_23.md)),
test / text against the real 2× image, dB / LPIPS; generated renders, colour / mono, the 32-px
and 16-px stamps:

| run | term | held-out | stamp 32 px | stamp 16 px | verdict |
|---|---|---|---|---|---|
| `x2lp_1123_0025` | LPIPS-VGG 0.1 | 29.75 / 0.172, 25.75 / 0.169 | 0.30 / 1.02 | 0.24 / 0.79 | baseline |
| `x2dv3s_1123_0025` | DINOv3 ViT-S/16 tiles, λ 54.9 by the rule | 29.69 / 0.233, 25.75 / 0.202 | 0.35 / 0.97 | 0.18 / 0.56 | not passed |
| `x2dv3sw_1123_0718` | the same, λ 245.2 fixed | 29.21 / 0.207, 25.46 / 0.190 | 0.50 / 1.30 | 0.28 / 0.76 | adopted on the user's read |

- The rule read the untrained second stage; on the two end heads the λ that equals 0.1 × LPIPS
  is 245 (the arm's) and 89.5 (the control's).

## Plan

### Facts

- The user (2026-11-22): "convnext t 결과 보고, 그거 기반으로 다음은 longer training을 했을 때 수치나
  아니면... 다른 방식? 이를테면 좀 더 어려운 x2를 학습했을 때 결과로 비교하는걸로 하자고 plan 적자.
  일단 이번 세션은 convnext 결과 보는거까지".
- Step 1, done: the DINOv3 network is ViT-S/16. The user (2026-11-23, while
  `x1dcnxt_1122_2355` was running): "내생각에 그냥 dinov3가 낫다", then: "convnext보다 vit-s 인
  dino가 낫다는 뜻이었어". The ConvNeXt-T arm's numbers: Arms.
- Step 2: the user took ×2 (2026-11-23: "내생각에 x2 2k 정도 돌리는게 제일 보기 좋지 않을까? 이게
  성능을 확실히 보기 좋으니까"). Run at 2,000 rollouts: `x2dv3s_1123_0025` against
  `x2lp_1123_0025`, then `x2dv3sw_1123_0718` with `lambda_dino` fixed from a trained head (the
  user, 2026-11-23: "ㅇㅋ dino arm 해보자"); Arms. The user's read (2026-11-23): "내가 보기엔 dino가 나아. 특히 스크린톤에서 패턴을 유지하면서 색 변조가 없었다는게 인상적이네. dino로 전면 교체하는걸로 하자." What follows is in `plan.md`, "LPIPS → DINO".
  The two options as they were written: The arm is the ViT-S term with the
  weight rule as it is; the control is LPIPS on the same recipe.
  - Longer ×1 training: the base recipe at 3,000 rollouts (about 45 minutes an arm; the length
    the `realr_1119_2200` / `x1twin_1121_2230` twin gap was read at, though at LR 2e-4) or at
    its full 12,000 (about 2.8 h an arm; the control is `x1ll_1122_1000`, already run). Code as
    it is.
  - The ×2 recipe (`configs/pixel_head/x2.toml`, B only, no GAN): rec + 0.1 · LPIPS + dc against
    the same with the DINO term in LPIPS's place, 4,000 rollouts (about 55 minutes an arm at
    0.78 s per rollout; the LPIPS run at that length, `x2b4k_1121_0950`, predates the ×1 base
    `x1ll_1122_1000`, so the control from the current base is one more arm). The term and its
    weight rule are in `X2Trainer` and `train_x2.py`.
- Read at either: held-out dB / LPIPS / the ViT-S distance, the stamp (16 px on ×1 through
  `sheet_texture.py`, 32 px on ×2 through `x2_stamp_phase.py`) and pair sheets for the user.
- The user on the term as LPIPS's replacement (2026-11-22): "내생각에 이걸 lpips 대체로까지
  고려할만할듯?" No recipe default has changed.

### Interpretation

- What 1,000 rollouts cannot say is whether the two terms end at the same head. The DINO arm's
  lower text / manga LPIPS may be speed (a stronger pull early) that the LPIPS head catches up
  with, and its stamp may grow or settle with length. Longer ×1 training answers that on the
  recipe that ships.
- ×2 is where a perceptual term has the most to do: the head must write detail the latent
  does not hold, LPIPS is weighted twice as high there, and the fixed pattern is the open
  problem of that line (`x2_plan.md`). A term that raises the ×1 stamp at 1,000 rollouts is
  most likely to show its cost there, and its gain as well.
- Claude's recommendation: the 3,000-rollout ×1 pair first (no new code, and the one length
  with a twin gap to read against), then ×2 with the term that is ahead.

## Open

- A web-data network, outside the plan above, on the user's word about the non-commercial
  licence:
  `webssl-dino300m-full2b-224` first (against DINOv3 it moves the data; DINOv3 ViT-L/16 is its
  size-matched control), then `webssl-mae300m-full2b-224` (the objective). Pixio ViT-B/16 only
  if neither is taken.
- A catalog row for the chosen network, once one is chosen.
- The replacement of LPIPS by the DINO term: `plan.md`, "LPIPS → DINO".
