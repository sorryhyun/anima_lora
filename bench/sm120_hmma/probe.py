"""Does sm_120 decode a bf16-input, fp16-accumulate HMMA?

PTX has no such ``mma`` (ptxas: "Unexpected instruction types"), so the probe assembles the
fp16 / fp16-accumulate form, sets the SASS bit that nvdisasm reads as ``.BF16`` on the
inputs, and launches it. Each variant runs in its own process (a trap kills the context);
the kernel log's Xid 13 lines are matched to the variant by the Xid 43 pid.

Variants:
  f16_ref           unpatched HMMA.16816.F16 (must run)
  f16_shift40       the same body moved +0x40 behind 4 NOPs (must run: the shift is sound)
  bf16              bit 82 set on the HMMA at its compiled offset
  bf16_at30         that HMMA copied over the first LDG (-0x40)
  bf16_shift40      the bf16 body moved +0x40
  reg_oor           unpatched HMMA with its C register R10 -> R138 (an operand-level fault)

Run through the daemon:
  make daemon-run ARGS="--stall-timeout 120 bench/sm120_hmma/probe.py"
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PTX = """.version 9.2
.target sm_120a
.address_size 64
.visible .entry k(.param .u64 p){
.reg .b64 r; .reg .b32 a<4>; .reg .b32 b<2>; .reg .f32 f<4>; .reg .b32 h<2>;
ld.param.u64 r, [p];
ld.global.v4.b32 {a0,a1,a2,a3}, [r];
ld.global.v2.b32 {b0,b1}, [r+16];
ld.global.v4.f32 {f0,f1,f2,f3}, [r+32];
ld.global.v2.b32 {h0,h1}, [r+48];
mma.sync.aligned.m16n8k16.row.col.%s {h0,h1},{a0,a1,a2,a3},{b0,b1},{h0,h1};
st.global.v4.f32 [r+32], {f0,f1,f2,f3};
st.global.v2.b32 [r+48], {h0,h1};
ret;
}
"""
INSTR = 16
OP_HMMA, OP_NOP, OP_EXIT = 0x23C, 0x918, 0x94D
BF16_BIT = 82  # in the 128-bit instruction; nvdisasm prints HMMA.16816.F16.BF16
EIATTR_EXIT = b"\x04\x1c\x04\x00"  # EIATTR_EXIT_INSTR_OFFSETS, one offset


def _tool(name: str) -> str:
    p = shutil.which(name) or f"/usr/local/cuda/bin/{name}"
    if not Path(p).exists():
        raise SystemExit(f"{name} not found")
    return p


def _text_section(b: bytes, name: bytes = b".text.k") -> tuple[int, int]:
    (shoff,) = struct.unpack_from("<Q", b, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)

    def sh(i):
        return struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize)

    stroff = sh(shstrndx)[4]
    for i in range(shnum):
        s = sh(i)
        if b[stroff + s[0] :].split(b"\0", 1)[0] == name:
            return s[4], s[5]
    raise SystemExit("no .text.k")


def _instrs(b: bytes, off: int, size: int) -> list[int]:
    return [off + i for i in range(0, size, INSTR)]


def _opcode(b: bytes, at: int) -> int:
    return struct.unpack_from("<Q", b, at)[0] & 0xFFF


def _find(b: bytes, off: int, size: int, op: int) -> int:
    hits = [a for a in _instrs(b, off, size) if _opcode(b, a) == op]
    if not hits:
        raise SystemExit(f"opcode {op:#x} not in .text.k")
    return hits[0]


def _shift(src: bytes, n: int) -> bytearray:
    b = bytearray(src)
    off, size = _text_section(b)
    nop_at = [a for a in _instrs(b, off, size) if _opcode(b, a) == OP_NOP]
    nop = bytes(b[nop_at[0] : nop_at[0] + INSTR])
    end = nop_at[0]  # the body ends at the first padding NOP
    body = bytes(b[off:end])
    new = nop * n + body
    if len(new) > size:
        raise SystemExit("no padding room for the shift")
    new += nop * ((size - len(new)) // INSTR)
    b[off : off + size] = new
    exit_old = _find(src, off, size, OP_EXIT) - off
    key = EIATTR_EXIT + exit_old.to_bytes(4, "little")
    i = b.find(key)
    if i < 0 or b.find(key, i + 1) >= 0:
        raise SystemExit("EXIT offset attribute not found once")
    b[i + 4 : i + 8] = (exit_old + n * INSTR).to_bytes(4, "little")
    return b


def build(work: Path) -> dict[str, Path]:
    ptxas = _tool("ptxas")
    ptx = work / "hm_f16.ptx"
    ptx.write_text(PTX % "f16.f16.f16.f16")
    ref = work / "f16_ref.cubin"
    subprocess.run([ptxas, "-arch=sm_120a", str(ptx), "-o", str(ref)], check=True)
    # the PTX spelling of the target is rejected; recorded, not built
    bad = work / "bf16_f16acc.ptx"
    bad.write_text(PTX % "f16.bf16.bf16.f16")
    r = subprocess.run(
        [ptxas, "-arch=sm_120a", str(bad), "-o", os.devnull],
        capture_output=True,
        text=True,
    )
    (work / "ptxas_bf16_f16acc.txt").write_text(f"rc={r.returncode}\n{r.stderr}")

    src = ref.read_bytes()
    off, size = _text_section(src)
    hmma = _find(src, off, size, OP_HMMA)

    bf16 = bytearray(src)
    bf16[hmma + BF16_BIT // 8] |= 1 << (BF16_BIT % 8)
    at30 = bytearray(bf16)
    at30[off + 0x30 : off + 0x40] = bf16[hmma : hmma + INSTR]
    oor = bytearray(src)
    oor[hmma + 8] |= 0x80  # C register field low byte: R10 -> R138

    out = {
        "f16_ref": src,
        "f16_shift40": _shift(src, 4),
        "bf16": bf16,
        "bf16_at30": at30,
        "bf16_shift40": _shift(bf16, 4),
        "reg_oor": oor,
    }
    paths = {}
    for name, data in out.items():
        p = work / f"{name}.cubin"
        p.write_bytes(data)
        paths[name] = p
    return paths


def _disasm(cubin: Path) -> list[str]:
    r = subprocess.run(
        [_tool("cuobjdump"), "-sass", str(cubin)], capture_output=True, text=True
    )
    return [
        ln.strip()
        for ln in r.stdout.splitlines()
        if re.match(r"\s+/\*[0-9a-f]{4}\*/", ln) and "NOP" not in ln
    ]


def child(cubin: str) -> None:
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]

    def ck(r, what):
        if r:
            s = ctypes.c_char_p()
            cu.cuGetErrorName(r, ctypes.byref(s))
            print(f"FAIL {what}: {r} {s.value.decode()}")
            raise SystemExit(1)

    ck(cu.cuInit(0), "init")
    dev = ctypes.c_int()
    ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "device")
    ctx = ctypes.c_void_p()
    ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "ctx")
    ck(cu.cuCtxSetCurrent(ctx), "ctx")
    buf = ctypes.c_uint64()
    ck(cu.cuMemAlloc_v2(ctypes.byref(buf), 64), "alloc")
    host = (ctypes.c_ubyte * 64)()
    one = struct.pack("<e", 1.0)
    for i in range(0, 24, 2):  # A and B fragments all 1.0
        host[i : i + 2] = one
    ck(cu.cuMemcpyHtoD_v2(buf.value, host, 64), "h2d")
    mod = ctypes.c_void_p()
    ck(cu.cuModuleLoadData(ctypes.byref(mod), Path(cubin).read_bytes()), "load")
    fn = ctypes.c_void_p()
    ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"k"), "getfn")
    p = ctypes.c_uint64(buf.value)
    args = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(p), ctypes.c_void_p))
    ck(cu.cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, None, args, None), "launch")
    ck(cu.cuCtxSynchronize(), "sync")
    ck(cu.cuMemcpyDtoH_v2(host, buf.value, 64), "d2h")
    (d0,) = struct.unpack("<e", bytes(host[48:50]))
    print(f"OK d0={d0:g}")


XID13 = re.compile(r"Xid \(PCI:[^)]*\): 13, (.*)")
XID43 = re.compile(r"Xid \(PCI:[^)]*\): 43, pid=(\d+)")


def _kernel_log(since: str) -> list[str]:
    r = subprocess.run(
        ["journalctl", "-k", "--since", since, "--no-pager", "-o", "cat"],
        capture_output=True,
        text=True,
    )
    return r.stdout.splitlines()


def main() -> None:
    from bench._common import make_run_dir, write_result

    ap = argparse.ArgumentParser()
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    if args.child:
        child(args.child)
        return

    run_dir = make_run_dir("sm120_hmma", args.label)
    paths = build(run_dir)
    since = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time.sleep(1)
    runs = {}
    for name, cubin in paths.items():
        r = subprocess.Popen(
            [sys.executable, __file__, "--child", str(cubin)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        out, _ = r.communicate()
        runs[name] = {"pid": r.pid, "rc": r.returncode, "out": out.strip()}
        print(f"{name}: rc={r.returncode} {out.strip()}", flush=True)
        time.sleep(2)
    time.sleep(2)

    pending: list[str] = []
    by_pid: dict[int, list[str]] = {}
    for ln in _kernel_log(since):
        if m := XID13.search(ln):
            pending.append(m.group(1))
        elif m := XID43.search(ln):
            by_pid[int(m.group(1))] = pending
            pending = []

    metrics = {}
    for name, info in runs.items():
        xid = by_pid.get(info["pid"], [])
        exc = next((x.split(": ", 1)[1] for x in xid if "Warp Exception" in x), None)
        esr = next(
            (
                dict(re.findall(r"(0x[0-9a-f]+)=(0x[0-9a-f]+)", x))
                for x in xid
                if "ESR" in x
            ),
            {},
        )
        metrics[name] = {
            **info,
            "exception": exc,
            "esr": esr,
            "sass": [s for s in _disasm(paths[name]) if "HMMA" in s],
        }
        print(f"  {name}: {exc} {esr}", flush=True)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=[p.name for p in paths.values()] + ["ptxas_bf16_f16acc.txt"],
    )
    print(run_dir)


if __name__ == "__main__":
    main()
