#!/usr/bin/env bash
# Hangul block evals on swap_text Korean pages. Queues one daemon job per render; prints the job ids.
#
#   bash project/lora_blocks/src/hangul_eval.sh <arm> <lora> [eval2|ko10]
#
# eval2 (default): swap_text's two Korean eval pages, seeds 0-2, the settings of the ksplit e2
#   renders (bf16 DiT, fp32 stream, swap 10). Results in results/<arm>[_s<n>]/<page>.
# ko10: ten held-out Korean pages of the same two works (inputs_ko10/, full staged captions:
#   five dense / small-text pages, five holding byte-split syllables), seed 0, the generate
#   default (int8 DiT, fp32 stream). Results in results/ko10_<arm>/<page>.
set -euo pipefail
ARM=$1; LORA=$2; SET=${3:-eval2}
RES=output/moss_i2i_eval/swap_text/results
if [ "$SET" = eval2 ]; then
  IN=output/moss_i2i_eval/swap_text/inputs
  PAGES="589883_0004_ko 737486_0019_ko"; SEEDS="0 1 2"; PRE=""
  EXTRA="--int8 off --blocks_to_swap 10"
else
  IN=output/moss_i2i_eval/swap_text/inputs_ko10
  PAGES=$(ls $IN | sed -n 's/\.src\.webp$//p'); SEEDS="0"; PRE="ko10_"
  EXTRA=""
fi
for page in $PAGES; do
  read -r W H < <(.venv/bin/python -c "from PIL import Image; print(*Image.open('$IN/$page.src.webp').size)")
  for seed in $SEEDS; do
    sfx=""; [ "$seed" != 0 ] && sfx="_s$seed"
    make -s daemon-run ARGS="--queue --stall-timeout 900 scripts/moss/generate.py \
      --config configs/moss/i2i/swap_text/generate.toml --prompts_file $IN/$page.txt --prompt_format text \
      --lora $LORA --width $W --height $H --seed $seed \
      --out_dir $RES/${PRE}${ARM}${sfx}/$page --ref_image $IN/$page.src.webp $EXTRA" | grep -o 'queued job [0-9a-f-]*'
  done
done
