"""Run ncu on the FA2 fwd + bwd kernels, fp16acc wheel vs stock paths."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "ncu_target.py")
METRICS = ",".join([
    "gpu__time_duration.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__ops_path_tensor_op_hmma_src_fp16_dst_fp16.sum",
    "sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32.sum",
    "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32.sum",
    "sm__inst_executed_pipe_tensor.sum",
    "sm__inst_executed_pipe_fma.sum",
    "sm__inst_executed_pipe_alu.sum",
    "sm__inst_executed_pipe_xu.sum",
    "sm__inst_executed_pipe_lsu.sum",
    "smsp__inst_executed.sum",
    "smsp__sass_thread_inst_executed_op_conversion_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_xu_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active",
])

for label, env in [("fp16acc", {}), ("stock", {"FLASH_ATTN_FP16ACC": "0", "FLASH_ATTN_FP16ACC_FWD": "0"})]:
    out = os.path.join(HERE, f"ncu2_{label}.txt")
    cmd = [
        "ncu", "--kernel-name", "regex:flash_bwd_dq_dk_dv|flash_fwd_kernel",
        "--launch-skip", "4", "--launch-count", "2",
        "--metrics", METRICS, "--csv", "--page", "raw",
        sys.executable, TARGET,
    ]
    print(f"== {label}: {' '.join(cmd)}", flush=True)
    with open(out, "w") as f:
        r = subprocess.run(cmd, env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT)
    print(f"exit {r.returncode}, {out}", flush=True)
    print(open(out).read()[-3000:], flush=True)
