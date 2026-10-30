# moss_te_sae — what the MOSS TE and DiT do with names, caption rows and strings

Direct probes of the post-trained Qwen3.5-9B text encoder and of the DiT's attention,
plus Qwen-Scope's Base SAEs. Two questions run through it:

- how to give the DiT characters it cannot draw from the name
  ([`docs/moss/proposal/te_character_knowledge.md`](../../docs/moss/proposal/te_character_knowledge.md));
- why a LoRA trigger does not bind on MOSS ([hypothesis.md](hypothesis.md)).

## Setup

- SAEs: `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100` (TopK k=100, 64K features,
  `resid_post` per layer), layers 15 / 31 in `/media/sorryhyun/new/qwen35_sae/`.
- TEs: instruct bf16 `/media/sorryhyun/new/qwen35_pe/inst`, Base bf16
  `/media/sorryhyun/data/qwen35_base/Qwen3.5-9B-Base`.
- Scripts in `src/` run as daemon jobs:
  `make daemon-run ARGS="--stall-timeout 900 project/moss_te_sae/src/<script>.py …"`.
  `template_attn_cmp.py` is CPU only.
- Results: newer runs in `output/tests/<name>/`, earlier ones under
  `/media/sorryhyun/new/qwen35_sae/<name>/<stamp>/`. Each report names its folders.
- `configs/` holds the trigger-binding training arms (`rows_trigger.toml`,
  `preserve10.toml`); `src/trigger_neg_cache.py` writes the trigger-stripped pairs
  `preserve_weight` needs.

## Facts settled

- The MOSS TE is post-trained **Qwen3.5-9B** (instruct), not Base: every tensor the INT8
  file leaves unquantized (GDN `A_log` / `dt_bias` / `conv1d`, `q_norm` / `k_norm`, final
  norm, `embed_tokens`) matches `/media/sorryhyun/new/qwen35_pe/inst` bit for bit.
  Qwen-Scope ships only Base SAEs for 9B.
- `last_hidden_state` = final norm (gain `1 + w`) of layer 31 `resid_post` (rel err 1.7e-3).
- INT8 vs bf16 TE: 2.1 % mean per-token relative error on cached layout-JSON captions.

## Reports, by question

Each report opens with its claims and the evidence follows per claim.

| report | claims |
|---|---|
| [character_knowledge](reports/character_knowledge.md) | TE knowledge and DiT drawing are separate (the DiT draws Neru, the TE does not know her); the targets are DiT-unknown characters (`malfoid`, Koneko: tags lift recall 0.29 / 0.06 → 0.83); the prompt form does not move a DiT-known character |
| [name_rows](reports/name_rows.md) | the identity sits in the name's own TE rows (splice keeps 0.96–0.98); the DiT expands it in place (peak block 30) and the image reads it in blocks 16–31; series is linear over name vectors; tag directions / a series centroid on the name rows recover part of it without training; the distilled `malfoid` patch fails the gate (24 % vs 80 %, lr confounded) |
| [text_read_path](reports/text_read_path.md) | 10 template rows take ~95 % of image→text attention with prompt-constant values; the template gates and the caption rows carry (suffix swap changes nothing, caption cut loses the prompt, template cut collapses the image); the template's effect on caption rows is an instruct × template interaction (0.10 / 0.24 alone, 0.60 together); the channel LoRA is a prompt-independent shift that does not use the caption path |
| [string_read](reports/string_read.md) | a quoted string is read locally by the lettering's queries through fixed heads (b21 h20, blocks 19–21); its rows carry the words, the rest of the caption the balloon and script; the swap_text LoRA binds each box to its own string the same way, and the four t2i heads alone are not needed |
| [sae_fit](reports/sae_fit.md) | Base SAEs fit the instruct caption rows at FVU 0.25–0.46; ~1/3 of what the DiT reads is in the error term; the DiT's projection amplifies tag-content features and suppresses structural ones |

## Open

- Trigger binding: the two training arms in `configs/` (delta confined to the trigger
  rows; + the trigger-preservation term) against `caststation_ema097_lr1e4`, read with
  `src/lora_use.py` (trigger share of gradient × input, empty-prompt cos).
- Which blocks carry the caption read: a per-block `no_caption_keys` cut
  (text_read_path cut all blocks at once; single heads in blocks 8 / 19 / 35 / 37 are the
  candidates).
- Whether the swap_text LoRA built its string read or reuses Base's: score a LoRA-off
  forward on the same trajectory (string_read).
- Approach B on `malfoid`: a larger lr and more steps before any verdict on the init
  (name_rows §5). A series centroid without the prototype as a warm start for
  grammar-sharing series only.
- A character the TE knows and the DiT cannot draw (case ②) is still not found; a
  per-character identity read on a set with more misses would test whether low series
  alignment predicts a miss.
- Causal check on the SAE side (ablate / boost single features, fixed seed); the error
  term's 1/3 bounds it.
- [hypothesis.md](hypothesis.md): test 3 (Base contrast) is done; tests 1 (wrapper
  sweep), 2 (name × context split) and 4 (training arm) are not.
