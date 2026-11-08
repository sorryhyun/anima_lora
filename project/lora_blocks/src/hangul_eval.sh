#!/usr/bin/env bash
# swap_text eval renders for one LoRA, as ONE daemon job (the models load once; every page and
# seed renders in it, compiled). Prints the job id.
#
#   bash project/lora_blocks/src/hangul_eval.sh <arm> <lora> [eval2|eval4|ko10] [generate flags...]
#
# eval2 (default): swap_text's two Korean eval pages, seeds 0-2, the settings of the ksplit e2
#   renders (bf16 DiT, fp32 stream, swap 10).
# eval4: the four eval pages (two English, two Korean), seeds 0-2, the generate default (int8 DiT,
#   fp32 stream).
# ko10: ten held-out Korean pages of the same two works (inputs_ko10/, full staged captions:
#   five dense / small-text pages, five holding byte-split syllables), seed 0, the generate
#   default.
# Results in one flat folder, results/<prefix><arm>/000_<page>.src_base_s<seed>.png
# (prefix: none for eval2 / eval4, ko10_ for ko10). hangul_read.py reads both this and the
# older per-page folders.
set -euo pipefail
ARM=$1; LORA=$2; SET=${3:-eval2}; shift $(( $# < 3 ? $# : 3 ))
RES=output/moss_i2i_eval/swap_text/results
case "$SET" in
  eval2) GLOB="output/moss_i2i_eval/swap_text/inputs/*_ko.src.webp"; SEEDS=0,1,2; PRE=""
         EXTRA="--int8 off --blocks_to_swap 10" ;;
  eval4) GLOB="output/moss_i2i_eval/swap_text/inputs/*.src.webp"; SEEDS=0,1,2; PRE=""; EXTRA="" ;;
  ko10)  GLOB="output/moss_i2i_eval/swap_text/inputs_ko10/*.src.webp"; SEEDS=0; PRE="ko10_"; EXTRA="" ;;
  *) echo "unknown set $SET" >&2; exit 2 ;;
esac
# width x height only caps the area: every page renders at its native size.
make -s daemon-run ARGS="--queue --stall-timeout 900 scripts/moss/generate.py \
  --config configs/moss/i2i/swap_text/generate.toml --prompt_format text \
  --ref_image '$GLOB' --ref_prompts --seeds $SEEDS --width 1536 --height 1536 \
  --lora $LORA --out_dir $RES/${PRE}${ARM} $EXTRA $*" | grep -o 'queued job [0-9a-f-]*'
