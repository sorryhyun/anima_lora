"""Phase 1: Backward kernel config sweep for SM120.

Systematically profiles FA4 backward kernel on SM120 across:
  - Block sizes (64×64, 128×128)
  - num_stages_Q / num_stages_dO (1, 2, 3)
  - AtomLayout{MSdP, NdKV, MdQ} (1, 2, 4)
  - V_in_regs (True/False)

Runs in three stages to keep compile count manageable:
  Stage 1 (--stage=1): Block + stages sweep (atom layouts fixed at 4,4,4)
  Stage 2 (--stage=2): Atom layout sweep (block/stages from best of stage 1 or --block/--stages)
  Stage 3 (--stage=3): V_in_regs + swapAB (all other params from best of stage 2 or --flags)
  All     (--stage=all): Run all three stages sequentially

Each config is SMEM-checked before compilation. Failed compiles are reported but don't
abort the sweep.

Usage:
  python bench/phase1_bwd_sweep.py --stage=1
  python bench/phase1_bwd_sweep.py --stage=2 --block-m=128 --block-n=128 --stages-q=1 --stages-do=1
  python bench/phase1_bwd_sweep.py --stage=3 --block-m=128 --block-n=128 --stages-q=1 --stages-do=1 --atom-msdp=4 --atom-ndkv=4 --atom-mdq=4
  python bench/phase1_bwd_sweep.py --stage=all
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product

# ---------------------------------------------------------------------------
# Anima model shapes (must match phase0_profile.py)
# ---------------------------------------------------------------------------
BATCH = 2
N_HEADS = 32
HEAD_DIM = 64
SELF_SEQLEN = 4096
CROSS_KV_SEQLEN = 256

N_SELF = 24
N_CROSS = 24

SMEM_CAPACITY = 99 * 1024  # SM120: 99 KB
NUM_MMA_WARPS = 4  # 128 threads / 32


@dataclass
class BwdConfig:
    block_m: int
    block_n: int
    stages_q: int
    stages_do: int
    atom_msdp: int
    atom_ndkv: int
    atom_mdq: int
    v_in_regs: bool
    sdp_swapab: bool = False
    dkv_swapab: bool = False
    dq_swapab: bool = False

    def smem_bytes(self, head_dim: int = HEAD_DIM) -> int:
        """Estimate shared memory usage (bytes).

        Includes Q, dO, K, V data tiles PLUS sP and sdS score tiles
        (m_block × n_block each, in dtype). The upstream can_implement
        check only counts data tiles — this is the corrected estimate.
        """
        q = self.block_m * head_dim * self.stages_q * 2
        do = self.block_m * head_dim * self.stages_do * 2
        k = self.block_n * head_dim * 2
        v = self.block_n * head_dim * 2
        qv = max(q, v) if self.v_in_regs else (q + v)
        # sP + sdS score tiles (bf16) + sLSE/sdPsum (~1 KB)
        scores = 2 * self.block_m * self.block_n * 2
        return qv + do + k + scores + 1024

    def fits_smem(self) -> bool:
        return self.smem_bytes() <= SMEM_CAPACITY

    def valid_atoms(self) -> bool:
        """Atom layouts must divide num_mma_warps (4)."""
        for a in (self.atom_msdp, self.atom_ndkv, self.atom_mdq):
            if NUM_MMA_WARPS % a != 0:
                return False
        return True

    def valid_stages(self) -> bool:
        """SM80 backward requires num_stages_Q >= num_stages_dO (flash_bwd.py:800)."""
        return self.stages_q >= self.stages_do

    def env_dict(self) -> dict[str, str]:
        return {
            "FA4_SM120_BWD_BLOCK_M": str(self.block_m),
            "FA4_SM120_BWD_BLOCK_N": str(self.block_n),
            "FA4_SM120_BWD_STAGES_Q": str(self.stages_q),
            "FA4_SM120_BWD_STAGES_DO": str(self.stages_do),
            "FA4_SM120_BWD_ATOM_MSDP": str(self.atom_msdp),
            "FA4_SM120_BWD_ATOM_NDKV": str(self.atom_ndkv),
            "FA4_SM120_BWD_ATOM_MDQ": str(self.atom_mdq),
            "FA4_SM120_BWD_V_IN_REGS": "1" if self.v_in_regs else "0",
            "FA4_SM120_BWD_SDP_SWAPAB": "1" if self.sdp_swapab else "0",
            "FA4_SM120_BWD_DKV_SWAPAB": "1" if self.dkv_swapab else "0",
            "FA4_SM120_BWD_DQ_SWAPAB": "1" if self.dq_swapab else "0",
        }

    def short_label(self) -> str:
        parts = [f"{self.block_m}x{self.block_n}"]
        parts.append(f"stg={self.stages_q},{self.stages_do}")
        parts.append(f"atom={self.atom_msdp},{self.atom_ndkv},{self.atom_mdq}")
        if self.v_in_regs:
            parts.append("Vreg")
        swaps = []
        if self.sdp_swapab:
            swaps.append("SdP")
        if self.dkv_swapab:
            swaps.append("dKV")
        if self.dq_swapab:
            swaps.append("dQ")
        if swaps:
            parts.append("swap=" + "+".join(swaps))
        return " ".join(parts)

    def smem_label(self) -> str:
        return f"{self.smem_bytes() / 1024:.0f}KB"


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------


def stage1_configs() -> list[BwdConfig]:
    """Block sizes × stages, atom layouts fixed at (4,4,4).

    128×128 is infeasible (sP+sdS score tiles add 64 KB → 128+ KB total).
    Includes asymmetric blocks (128×64, 64×128) which may still fit.
    """
    configs = []
    for bm, bn in [(64, 64), (128, 64), (64, 128)]:
        for sq, sdo in product([1, 2, 3], repeat=2):
            cfg = BwdConfig(bm, bn, sq, sdo, 4, 4, 4, False)
            if cfg.fits_smem() and cfg.valid_atoms() and cfg.valid_stages():
                configs.append(cfg)
    return configs


def stage2_configs(bm: int, bn: int, sq: int, sdo: int) -> list[BwdConfig]:
    """Atom layout sweep with fixed block/stages."""
    configs = []
    for a1, a2, a3 in product([1, 2, 4], repeat=3):
        cfg = BwdConfig(bm, bn, sq, sdo, a1, a2, a3, False)
        if cfg.fits_smem() and cfg.valid_atoms() and cfg.valid_stages():
            configs.append(cfg)
    return configs


def stage3_configs(
    bm: int,
    bn: int,
    sq: int,
    sdo: int,
    a_msdp: int,
    a_ndkv: int,
    a_mdq: int,
) -> list[BwdConfig]:
    """V_in_regs sweep only.

    swapAB is excluded: SdP_swapAB fails an LSE shape assertion for
    non-SM90 atom layouts (flash_bwd.py:903), and dQ_swapAB causes
    SEGFAULTs due to invalid memory access patterns with SM80 warps.
    dKV_swapAB alone is safe but showed no benefit in testing.
    """
    configs = []
    for v_in_regs in [False, True]:
        cfg = BwdConfig(bm, bn, sq, sdo, a_msdp, a_ndkv, a_mdq, v_in_regs)
        if cfg.fits_smem() and cfg.valid_atoms() and cfg.valid_stages():
            configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Runner — spawns a subprocess with env vars to get a clean JIT cache
# ---------------------------------------------------------------------------

BENCH_SCRIPT = os.path.join(os.path.dirname(__file__), "_phase1_bench_one.py")


@dataclass
class BenchResult:
    config: BwdConfig
    self_fwd_ms: float | None = None
    self_bwd_ms: float | None = None
    cross_fwd_ms: float | None = None
    cross_bwd_ms: float | None = None
    error: str | None = None
    compile_s: float = 0.0

    @property
    def self_total(self) -> float | None:
        if self.self_fwd_ms is not None and self.self_bwd_ms is not None:
            return self.self_fwd_ms + self.self_bwd_ms
        return None

    @property
    def cross_total(self) -> float | None:
        if self.cross_fwd_ms is not None and self.cross_bwd_ms is not None:
            return self.cross_fwd_ms + self.cross_bwd_ms
        return None

    @property
    def step_bwd_ms(self) -> float | None:
        if self.self_bwd_ms is not None and self.cross_bwd_ms is not None:
            return self.self_bwd_ms * N_SELF + self.cross_bwd_ms * N_CROSS
        return None


def run_one(cfg: BwdConfig) -> BenchResult:
    """Run the single-config benchmark in a subprocess."""
    env = {**os.environ, **cfg.env_dict()}
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, BENCH_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max (compilation can be slow)
        )
    except subprocess.TimeoutExpired:
        return BenchResult(cfg, error="TIMEOUT (>300s)")

    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        # Extract last meaningful line of stderr for the error message
        err_lines = [line for line in proc.stderr.strip().split("\n") if line.strip()]
        err_msg = err_lines[-1][:200] if err_lines else "unknown error"
        return BenchResult(cfg, error=err_msg, compile_s=elapsed)

    # Parse stdout: expect "self_fwd,self_bwd,cross_fwd,cross_bwd" or "ERROR:msg"
    output = proc.stdout.strip()
    if output.startswith("ERROR:"):
        # Also show last line of stderr (traceback) if available
        err_detail = output[6:]
        stderr_lines = [
            line for line in proc.stderr.strip().split("\n") if line.strip()
        ]
        if stderr_lines:
            err_detail += f" | {stderr_lines[-1][:120]}"
        return BenchResult(cfg, error=err_detail, compile_s=elapsed)

    parts = output.split(",")
    if len(parts) != 4:
        return BenchResult(cfg, error=f"bad output: {output!r}", compile_s=elapsed)

    def parse(s: str) -> float | None:
        return float(s) if s != "FAIL" else None

    return BenchResult(
        cfg,
        self_fwd_ms=parse(parts[0]),
        self_bwd_ms=parse(parts[1]),
        cross_fwd_ms=parse(parts[2]),
        cross_bwd_ms=parse(parts[3]),
        compile_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(results: list[BenchResult], stage_name: str):
    print(f"\n{'=' * 90}")
    print(f"  {stage_name} — Results")
    print(f"{'=' * 90}")

    # Header
    print(
        f"{'Config':<42} {'SMEM':>5} {'Self bwd':>9} {'Cross bwd':>10} {'Step bwd':>9} {'Compile':>8} {'Note':>6}"
    )
    print("-" * 90)

    # Sort by step backward time (failures last)
    def sort_key(r: BenchResult):
        if r.step_bwd_ms is not None:
            return (0, r.step_bwd_ms)
        return (1, 0.0)

    for r in sorted(results, key=sort_key):
        label = r.config.short_label()
        smem = r.config.smem_label()
        if r.error:
            print(
                f"{label:<42} {smem:>5} {'FAIL':>9} {'':>10} {'':>9} {r.compile_s:>7.1f}s  {r.error}"
            )
        else:
            self_b = f"{r.self_bwd_ms:.2f}" if r.self_bwd_ms else "—"
            cross_b = f"{r.cross_bwd_ms:.2f}" if r.cross_bwd_ms else "—"
            step_b = f"{r.step_bwd_ms:.1f}" if r.step_bwd_ms else "—"
            print(
                f"{label:<42} {smem:>5} {self_b:>8}ms {cross_b:>9}ms {step_b:>8}ms {r.compile_s:>7.1f}s"
            )

    # Best config
    valid = [r for r in results if r.step_bwd_ms is not None]
    if valid:
        best = min(valid, key=lambda r: r.step_bwd_ms)
        print(
            f"\n  Best: {best.config.short_label()} — "
            f"self_bwd={best.self_bwd_ms:.2f}ms, cross_bwd={best.cross_bwd_ms:.2f}ms, "
            f"step_bwd={best.step_bwd_ms:.1f}ms"
        )
        return best
    else:
        print("\n  No successful configs!")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase 1: SM120 backward config sweep")
    parser.add_argument("--stage", default="1", choices=["1", "2", "3", "all"])
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--stages-q", type=int, default=2)
    parser.add_argument("--stages-do", type=int, default=2)
    parser.add_argument("--atom-msdp", type=int, default=4)
    parser.add_argument("--atom-ndkv", type=int, default=4)
    parser.add_argument("--atom-mdq", type=int, default=4)
    args = parser.parse_args()

    # Verify helper script exists
    if not os.path.exists(BENCH_SCRIPT):
        print(f"ERROR: helper script not found: {BENCH_SCRIPT}")
        sys.exit(1)

    stages_to_run = ["1", "2", "3"] if args.stage == "all" else [args.stage]

    # Carry forward best results between stages
    best_bm, best_bn = args.block_m, args.block_n
    best_sq, best_sdo = args.stages_q, args.stages_do
    best_am, best_an, best_aq = args.atom_msdp, args.atom_ndkv, args.atom_mdq

    for stage in stages_to_run:
        if stage == "1":
            configs = stage1_configs()
            print(f"\nStage 1: Block + Stages sweep ({len(configs)} configs)")
            results = []
            for i, cfg in enumerate(configs, 1):
                print(
                    f"  [{i}/{len(configs)}] {cfg.short_label()} ({cfg.smem_label()}) ...",
                    end=" ",
                    flush=True,
                )
                r = run_one(cfg)
                if r.error:
                    print(f"FAIL ({r.compile_s:.1f}s): {r.error}")
                else:
                    print(
                        f"self_bwd={r.self_bwd_ms:.2f}ms cross_bwd={r.cross_bwd_ms:.2f}ms ({r.compile_s:.1f}s)"
                    )
                results.append(r)
            best = print_results(results, "Stage 1: Block + Stages")
            if best:
                best_bm = best.config.block_m
                best_bn = best.config.block_n
                best_sq = best.config.stages_q
                best_sdo = best.config.stages_do

        elif stage == "2":
            configs = stage2_configs(best_bm, best_bn, best_sq, best_sdo)
            print(
                f"\nStage 2: Atom layout sweep ({len(configs)} configs, "
                f"block={best_bm}x{best_bn}, stages={best_sq},{best_sdo})"
            )
            results = []
            for i, cfg in enumerate(configs, 1):
                print(
                    f"  [{i}/{len(configs)}] {cfg.short_label()} ({cfg.smem_label()}) ...",
                    end=" ",
                    flush=True,
                )
                r = run_one(cfg)
                if r.error:
                    print(f"FAIL ({r.compile_s:.1f}s): {r.error}")
                else:
                    print(
                        f"self_bwd={r.self_bwd_ms:.2f}ms cross_bwd={r.cross_bwd_ms:.2f}ms ({r.compile_s:.1f}s)"
                    )
                results.append(r)
            best = print_results(results, "Stage 2: Atom Layouts")
            if best:
                best_am = best.config.atom_msdp
                best_an = best.config.atom_ndkv
                best_aq = best.config.atom_mdq

        elif stage == "3":
            configs = stage3_configs(
                best_bm, best_bn, best_sq, best_sdo, best_am, best_an, best_aq
            )
            print(
                f"\nStage 3: V_in_regs + swapAB ({len(configs)} configs, "
                f"block={best_bm}x{best_bn}, stages={best_sq},{best_sdo}, "
                f"atoms={best_am},{best_an},{best_aq})"
            )
            results = []
            for i, cfg in enumerate(configs, 1):
                print(
                    f"  [{i}/{len(configs)}] {cfg.short_label()} ({cfg.smem_label()}) ...",
                    end=" ",
                    flush=True,
                )
                r = run_one(cfg)
                if r.error:
                    print(f"FAIL ({r.compile_s:.1f}s): {r.error}")
                else:
                    print(
                        f"self_bwd={r.self_bwd_ms:.2f}ms cross_bwd={r.cross_bwd_ms:.2f}ms ({r.compile_s:.1f}s)"
                    )
                results.append(r)
            print_results(results, "Stage 3: V_in_regs + swapAB")


if __name__ == "__main__":
    main()
