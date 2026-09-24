# TASK — next up

## Scene pools: grow s1 / s1w / sl1w / ja_comic (paused 2026-09-24)

State: rejected renders pruned from the four pools (4 882 files, 1.29 GB);
grow code landed in `project/cjk_renderable_anima/src/scenes/` (stored rows
reused, only new indices rendered + judged, `--scene_prune 1`); the four grow
jobs were queued and then killed on request. The s1 job had rendered 264 of
its 1 000 new indices before the kill — those PNGs sit unjudged in
`output/cjk_anima_scale/scenes_s1/img/` (scene_01000 … ) and are picked up
on the re-run (existing file → not regenerated, then judged).

Argv per pool: `project/cjk_anima_scale/README.md` § Scene pools. Resubmit
(raw pack in the env, oldest first is fine, ~6 h total):

```bash
export ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack
S=project/cjk_renderable_anima/src/wake_probe.py
make daemon-run ARGS="--label scenes-s1-grow --stall-timeout 0 --queue $S --stage scenes --scene_tag s1 --scene_n 2000 --scene_frames reads_as,bubble_reads,saying,sign --scene_prune 1"
# sl1w 2000 / s1w 2600 / ja_comic 4400 — argv in the README table; sl1w's
# --scene_anchors list is in the killed job 20260924-101359-89eac5's job.json
# (or sorted({r["anchor"]}) over scenes_sl1w/prompts.jsonl)
```

Then `--steps data` on the stage that consumes them; check each pool's
`report.md` keep rate stays near the old one (s1 23 %, s1w 24 %, sl1w 21 %,
ja_comic 12 %).

Uncommitted: the grow/prune code, `tests/fixtures/cli_golden.json`, the
README section.
