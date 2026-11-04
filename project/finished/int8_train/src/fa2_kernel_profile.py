"""Kernel-level profile of one MOSS-shape FA2 forward + backward on the fp16acc wheel."""
import torch
from torch.profiler import ProfilerActivity, profile
from flash_attn import flash_attn_func

torch.manual_seed(0)
B, L, H, D = 1, 4600, 24, 128
q, k, v = (torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
dout = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16) * 1e-7

def step():
    o = flash_attn_func(q, k, v)
    o.backward(dout)

for _ in range(5):
    step()
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        step()
    torch.cuda.synchronize()

rows = sorted(prof.key_averages(), key=lambda r: -r.device_time_total)
total = sum(r.device_time_total for r in rows)
print(f"{'kernel':<70} {'calls':>5} {'us/call':>9} {'ms/iter':>8} {'share':>6}")
for r in rows:
    if r.device_time_total / total < 0.002:
        continue
    print(f"{r.key[:70]:<70} {r.count:>5} {r.device_time_total / r.count:>9.1f} "
          f"{r.device_time_total / 10 / 1000:>8.3f} {100 * r.device_time_total / total:>5.1f}%")
print(f"total per iter: {total / 10 / 1000:.3f} ms")
