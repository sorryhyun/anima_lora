#!/usr/bin/env bash
# Plan step 3 eval: the Hangul block alone and stacked on swap_text ksplit e2 (exact concat),
# on swap_text's two Korean eval pages, seeds 0-2, the settings of the ksplit e2 renders
# (bf16 DiT, fp32 stream, swap 10). Queues one daemon job per render; prints the job ids.
#
#   bash project/lora_blocks/src/hangul_eval.sh <arm> <lora>
set -euo pipefail
ARM=$1; LORA=$2
IN=output/moss_i2i_eval/swap_text/inputs
RES=output/moss_i2i_eval/swap_text/results
declare -A SIZE=([589883_0004_ko]="1360 768" [737486_0019_ko]="848 1232")
for page in 589883_0004_ko 737486_0019_ko; do
  read -r W H <<<"${SIZE[$page]}"
  for seed in 0 1 2; do
    sfx=""; [ "$seed" != 0 ] && sfx="_s$seed"
    make -s daemon-run ARGS="--queue --stall-timeout 900 scripts/moss/generate.py \
      --config configs/moss/i2i/swap_text/generate.toml --prompts_file $IN/$page.txt --prompt_format text \
      --lora $LORA --width $W --height $H --seed $seed \
      --out_dir $RES/${ARM}${sfx}/$page --ref_image $IN/$page.src.webp \
      --int8 off --blocks_to_swap 10" | grep -o 'queued job [0-9a-f-]*'
  done
done
