# Thin dispatcher → tasks.py. tasks.py is the source of truth; this file just
# forwards `make <target>` to `python tasks.py <target> $(ARGS)`.
#
# Make-line variables (PRESET=low_vram, REF_IMAGE=foo.png, ARTIST=sincos,
# ENCODER=pe, METHOD=lora, MODEL_DIR=..., ADAPTER_DIR=..., MULTIPLIER=...,
# IP_SCALE=..., EC_SCALE=..., PROMPT="...", NEG="...", INVERT_NAME=...,
# BENCH_INVERSIONS=..., FINETUNE_WARM=..., FINETUNE_BS=..., FINETUNE_SWAP=...,
# RUN=..., ALL=1, JSONL=1, GUI_PRESETS=..., PROFILE_STEPS=3-5) are exported
# as env vars so tasks.py picks them up.
#
# Pass extra training/inference flags via ARGS="--network_dim 32".
# See `make help` (or `python tasks.py --help`) for the full command list.

.DEFAULT_GOAL := help
.SUFFIXES:
MAKEFLAGS += --no-builtin-rules

# Export every make-line variable to subprocesses.
export

.PHONY: help FORCE
help:
	@python tasks.py --help

# FORCE has no recipe, so it's always considered out-of-date — used as a
# dependency of the catch-all so make doesn't skip targets that share a name
# with an existing file/dir (e.g. `preprocess`, `gui`).
FORCE:

# Stop GNU Make's implicit "remake the Makefile" check from going through the
# catch-all (which would otherwise try `python tasks.py Makefile`).
Makefile: ;

# Catch-all: any other target is forwarded straight to tasks.py.
%: FORCE
	@python tasks.py $@ $(ARGS)
