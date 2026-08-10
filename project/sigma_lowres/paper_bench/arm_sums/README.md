# arm_sums — central vector-store root (2026-08-10)

One directory per run, named by the run's `bench/results/` run-dir name
(`<YYYYMMDD-HHMM-label>/`), holding that run's `--keep_arm_sums` store:
per-(arm, bin) flat-gradient memmaps + `manifest.json` + `groups.json`.
`run_sigma_probe.py` writes here directly and leaves a relative symlink
at `<run_dir>/arm_sums`, so per-run paths (and every committed reader
script) keep resolving. Everything but this README is gitignored.

Why central: one root to apply the store lifecycle to — the T0 boot
fingerprint, the T1 canary, and reclamation (`paper_v2/roadmap.md` §3).

Policy (roadmap §3, executed 2026-08-10): **a store lives until its
registered read commits the vector tables it owes, then its raw sums
are reclaimed** (`.npy` deleted, `manifest.json` kept as provenance).
Scalars/ledgers always survive in committed JSONs under
`../experiments/`. Directories currently containing only a
`manifest.json` are reclaimed stores, not broken ones.

Kept live: `20260810-0214-e28-g2diag-native07` — boot-family B's T1
canary candidate (do not reclaim without a T1 decision).

Vector reads between stores are licensed only within a boot family
(`project_crossboot_arm_store_break`). Every manifest carries a
`boot_fingerprint` (stamped by `ArmSumAccumulator.finalize`; the ten
pre-T0 manifests were backfilled 2026-08-10 from
`journalctl --list-boots`, marked `"backfilled": true`). Gate reads
with `vector_ledger.assert_same_family(*stores)`. Families:
**A** = boot 2026-08-06 18:25 (e193, e194) · **B** = boot 2026-08-09
19:32 (e28 pair, g2diag, e26 grid + twin) · the e221 and e260 smokes
sit in their own dead singleton boots (08-08 12:34 / 08-09 11:14) and
match nothing.
