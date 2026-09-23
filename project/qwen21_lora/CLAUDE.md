# CLAUDE.md — `project/qwen21_lora/`

**This line is not Anima.** Its code lives in `library/qwen21/` — **read
`library/qwen21/CLAUDE.md` first** and load the `qwen21` skill: the root `CLAUDE.md` invariants (5D latents, max-padded
text, free-fit bucketing, block-compile first) are wrong here.

This directory is the research home: `src/` holds the one-off scripts (`backward_smoke`,
`smoke_t2i`, `bench_accel`) on top of `library.qwen21`; caching, training and A/B generation
are `scripts/qwen21/{cache,train,generate}.py`. GPU work goes through the daemon, as everywhere in
this repo.

State and measured numbers: `README.md`.
