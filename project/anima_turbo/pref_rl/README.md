# Preference rounds (`make exp-turbo-pref`)

The window of [../proposal/pref_rl.md](../proposal/pref_rl.md): the user picks renders, and the picks
are the labels of a preference head.

1. `make exp-turbo-pref`, then **New…** creates a session (`output/turbo_pref/<name>/`). The
   defaults are the proposal's policy: merged14 + Hub v3, int8, compiled, 4 steps, cfg 1, 16
   renders a round, at most 8 picks, captions from `post_image_dataset/resized`.
2. **Start server** submits one daemon job that keeps the policy loaded (a warm round renders in
   about 15 s). It holds the GPU until **Stop server**, or 20 minutes with no request. Closing
   the window leaves it running, and a reopened window attaches to it again.
3. Pick and submit. Rounds have no end: the window keeps the next ones queued.

| Action | Input |
|---|---|
| pick / unpick | click, or `1234` / `qwer` / `asdf` / `zxcv` (one row each) |
| open a render large | right-click, or Space over it; ←/→ move, P or a click picks, Esc closes |
| submit | Enter (zero picks is an answer) |
| skip | Ctrl+K (logged, left out of every read) |

- **Next caption…** types the caption of an upcoming round. Otherwise captions are drawn at
  random; no pick-count predictor exists.
- **Blind A/B**: put the LoRA to compare in **Policy…** as arm B, restart the server and switch
  Mode. Each grid is 8 + 8 at matched seeds with the arms hidden; the side panel shows B's share
  of the picks and a sign test over grids.
- `picks.jsonl` in the session holds one row per answered round: caption, seeds, picks, arms and
  the policy that rendered it.

The head (P0 and P1 of the proposal) runs on cached features:

```bash
make daemon-run ARGS="--stall-timeout 900 project/anima_turbo/pref_rl/features.py --set gelcrawl"
make daemon-run ARGS="--stall-timeout 1800 project/anima_turbo/pref_rl/head.py gelcrawl"
make daemon-run ARGS="--stall-timeout 600 project/anima_turbo/pref_rl/features.py --set <session>"
.venv/bin/python project/anima_turbo/pref_rl/head.py picks --session <session>
```

`features.py` caches four bases per image (DINOv3 CLS, PE-Spatial pooled, tagger logits and
embedding) and extracts only the rows it does not hold. `head.py gelcrawl` scores each basis on
held-out artists and saves a head; `head.py picks` reads that head zero-shot, a head fitted on
the picks alone and the gelcrawl head refitted on the picks, each against the sign-test gate.
The DiT-head basis is not built. First read: [../reports/pref_p1.md](../reports/pref_p1.md).

