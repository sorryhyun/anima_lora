"""``python -m anima_daemon.runpod <verb>`` — RunPod pods from the terminal.

    gpus       [--cloud SECURE|COMMUNITY] [--count N] [--min-vram GB] [--min-cuda V]
               [--all] [--json]
    ls         [--all] [--json]
    get        <pod_id>
    create     --gpu ID [--gpu ID …] [--image IMG] [--name N] [--cloud C] [--count N]
               [--disk GB] [--volume GB] [--network-volume ID] [--env K=V …]
               [--port P/proto …] [--min-cuda V] [--spot] [--ssh-key PATH | --no-ssh-key]
               [--dry-run] [--wait]
    start|stop <pod_id>
    terminate  <pod_id> [--yes]
    wait       <pod_id> [--timeout S]
    ssh        <pod_id> [--identity PATH] [-- <remote cmd…>]
    volumes    [--json]

``create`` bills from the moment it returns — ``--dry-run`` prints the request
body instead. Defaults: ``sorryhyun96/anima_lora`` on a 200 GB container disk
and no pod volume — the disk is wiped on stop, so a pod is meant to be
terminated after its job, not stopped. ``ls`` shows only ``anima-`` pods unless
``--all``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from . import api

SSH_KEY_CANDIDATES = ("id_ed25519.pub", "id_ecdsa.pub", "id_rsa.pub")


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _table(rows: list[list[str]], header: list[str]) -> None:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, *rows]:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip())


def _money(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "-"


# ── gpus ──────────────────────────────────────────────────────────────


def cmd_gpus(cl: api.RunPodClient, a) -> int:
    gpus = cl.gpu_types(cloud=a.cloud, count=a.count, min_cuda=_min_cuda(a))
    if a.min_vram:
        gpus = [g for g in gpus if (g.get("memory") or 0) >= a.min_vram]
    if not a.all:
        gpus = [g for g in gpus if g.get("availability") not in (None, "NONE")]
    key = "secure" if a.cloud == "SECURE" else "community"
    gpus.sort(key=lambda g: ((g.get("price") or {}).get(key) or 1e9, g.get("id", "")))
    if a.json:
        _print_json(gpus)
        return 0
    rows = [
        [
            g.get("id", ""),
            f"{g.get('memory', '?')}G",
            _money((g.get("price") or {}).get(key)),
            g.get("availability") or "-",
            str((g.get("maxCount") or {}).get(key, "-")),
        ]
        for g in gpus
    ]
    _table(rows, ["gpu_type_id", "vram", f"$/h·gpu ({key})", "stock", "max"])
    return 0


# ── pods ──────────────────────────────────────────────────────────────


def _pod_row(p: dict) -> list[str]:
    gpu = p.get("gpu") or {}
    ep = api.ssh_endpoint(p)
    return [
        p.get("id", ""),
        p.get("name", ""),
        api.pod_status(p),
        f"{gpu.get('count', '')}× {gpu.get('displayName') or gpu.get('id') or '-'}",
        _money(p.get("adjustedCostPerHr") or p.get("costPerHr") or p.get("cost")),
        f"{ep[0]}:{ep[1]}" if ep else "-",
    ]


def cmd_ls(cl: api.RunPodClient, a) -> int:
    pods = cl.list_pods()
    if not a.all:
        pods = [p for p in pods if (p.get("name") or "").startswith(api.NAME_PREFIX)]
    if a.json:
        _print_json(pods)
        return 0
    _table([_pod_row(p) for p in pods], ["id", "name", "status", "gpu", "$/h", "ssh"])
    hidden = "" if a.all else " (anima- only; --all for every pod)"
    print(f"{len(pods)} pod(s){hidden}")
    return 0


def cmd_get(cl: api.RunPodClient, a) -> int:
    _print_json(cl.get_pod(a.pod_id))
    return 0


def _find_ssh_pubkey(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return Path(explicit).expanduser().read_text().strip()
    ssh_dir = Path.home() / ".ssh"
    for name in SSH_KEY_CANDIDATES:
        p = ssh_dir / name
        if p.is_file():
            return p.read_text().strip()
    return None


def _parse_env(items: list[str]) -> dict:
    env = {}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep or not k:
            raise SystemExit(f"--env expects K=V, got {item!r}")
        env[k] = v
    return env


def _min_cuda(a) -> Optional[str]:
    """``--min-cuda``, defaulting to the image's CUDA; ``any`` = no filter."""
    v = a.min_cuda or api.DEFAULT_MIN_CUDA
    return None if v == "any" else v


def _pod_env(a) -> dict:
    env = _parse_env(a.env)
    if not a.no_ssh_key and "PUBLIC_KEY" not in env:
        # The image's entrypoint starts sshd with PUBLIC_KEY as authorized_keys;
        # setting it replaces the account-level keys for this pod.
        key = _find_ssh_pubkey(a.ssh_key)
        if key:
            env["PUBLIC_KEY"] = key
    return env


def _pod_name(a) -> str:
    name = a.name or time.strftime("%Y%m%d-%H%M%S")
    return name if name.startswith(api.NAME_PREFIX) else api.NAME_PREFIX + name


def build_create_spec(a, gpu_id: str) -> dict:
    """The v2 POST /pods body for one GPU type (v2 takes a single ``gpu.id``;
    ``--gpu`` fallbacks are tried in order by ``cmd_create``)."""
    gpu = {"id": gpu_id, "count": a.count}
    if _min_cuda(a):
        gpu["minCudaVersion"] = _min_cuda(a)
    spec = {
        "name": _pod_name(a),
        "gpu": gpu,
        "cloud": a.cloud,
        "disk": a.disk,
        "ports": a.port or ["22/tcp"],
        "env": _pod_env(a),
    }
    mount = a.volume_mount or "/workspace"
    if a.network_volume:
        spec["mounts"] = {"network": [{"volumeId": a.network_volume, "path": mount}]}
    elif a.volume:
        spec["mounts"] = {"persistent": {"size": a.volume, "path": mount}}
    if a.template:
        spec["templateId"] = a.template
    if a.image or not a.template:
        spec["image"] = a.image or api.DEFAULT_IMAGE
    return spec


def build_create_spec_v1(a) -> dict:
    """The v1 POST /pods body — only ``--spot`` needs it (v2 has no
    interruptible pods). v1 takes a GPU priority list but caps
    ``allowedCudaVersions`` at 13.0, so it cannot ask for the image's 13.2."""
    cuda = _min_cuda(a)
    if cuda and cuda not in api.V1_CUDA_VERSIONS:
        raise SystemExit(
            f"--spot goes through the v1 API, whose CUDA filter stops at "
            f"{api.V1_CUDA_VERSIONS[0]}; pass --min-cuda any (or ≤ "
            f"{api.V1_CUDA_VERSIONS[0]}) to rent a spot pod"
        )
    spec = {
        "name": _pod_name(a),
        "cloudType": a.cloud,
        "computeType": "GPU",
        "gpuTypeIds": a.gpu,
        "gpuCount": a.count,
        "containerDiskInGb": a.disk,
        "ports": a.port or ["22/tcp"],
        "env": _pod_env(a),
        "interruptible": True,
    }
    if cuda:
        # v1 matches exact versions; "at least" = every listed one from cuda up.
        spec["allowedCudaVersions"] = [
            v for v in api.V1_CUDA_VERSIONS if float(v) >= float(cuda)
        ]
    if a.network_volume:
        spec["networkVolumeId"] = a.network_volume
    else:
        spec["volumeInGb"] = a.volume
    if a.volume_mount:
        spec["volumeMountPath"] = a.volume_mount
    if a.template:
        spec["templateId"] = a.template
    if a.image or not a.template:
        spec["imageName"] = a.image or api.DEFAULT_IMAGE
    return spec


def _redacted(spec: dict) -> dict:
    out = dict(spec)
    if "PUBLIC_KEY" in (out.get("env") or {}):
        out["env"] = {**out["env"], "PUBLIC_KEY": out["env"]["PUBLIC_KEY"][:24] + "…"}
    return out


# Errors that are about the request or the account, not the GPU type's stock:
# trying the next --gpu would fail the same way.
_NO_FALLBACK_STATUS = (0, 400, 401, 403, 422)


def cmd_create(cl: Optional[api.RunPodClient], a) -> int:
    specs = (
        [build_create_spec_v1(a)]
        if a.spot
        else [build_create_spec(a, g) for g in a.gpu]
    )
    if a.dry_run:
        _print_json(
            [_redacted(s) for s in specs] if len(specs) > 1 else _redacted(specs[0])
        )
        return 0
    if "PUBLIC_KEY" not in specs[0]["env"] and not a.no_ssh_key:
        print(
            "warning: no ~/.ssh/*.pub found; pod uses the account's SSH keys",
            file=sys.stderr,
        )
    if a.spot:
        pod = cl.create_pod(specs[0])
    else:
        for i, spec in enumerate(specs):
            try:
                pod = cl.create_pod_v2(spec)
                break
            except api.RunPodError as e:
                if e.status in _NO_FALLBACK_STATUS or i == len(specs) - 1:
                    raise
                print(
                    f"{spec['gpu']['id']}: {e.message} — trying next --gpu",
                    file=sys.stderr,
                )
    cost = pod.get("costPerHr", pod.get("cost"))
    print(
        f"created {pod.get('id')} ({pod.get('name')}), ${_money(cost)}/h",
        file=sys.stderr,
    )
    if a.wait:
        pod = _wait(cl, pod["id"], a.timeout)
    _print_json(pod)
    return 0


def _wait(cl: api.RunPodClient, pod_id: str, timeout: float) -> dict:
    t0 = time.monotonic()
    last = [None]

    def on_poll(pod, ep):
        state = (api.pod_status(pod), ep)
        if state != last[0]:
            last[0] = state
            where = f"{ep[0]}:{ep[1]}" if ep else "no ssh mapping yet"
            print(
                f"[{time.monotonic() - t0:5.0f}s] {state[0]} · {where}", file=sys.stderr
            )

    pod = cl.wait_ssh(pod_id, timeout=timeout, on_poll=on_poll)
    ip, port = api.ssh_endpoint(pod)
    print(
        f"ready in {time.monotonic() - t0:.0f}s: ssh root@{ip} -p {port}",
        file=sys.stderr,
    )
    return pod


def cmd_wait(cl: api.RunPodClient, a) -> int:
    try:
        _print_json(_wait(cl, a.pod_id, a.timeout))
    except TimeoutError as e:
        print(str(e), file=sys.stderr)
        return 124
    return 0


def cmd_start(cl: api.RunPodClient, a) -> int:
    cl.start_pod(a.pod_id)
    print(f"start requested: {a.pod_id}")
    return 0


def cmd_stop(cl: api.RunPodClient, a) -> int:
    cl.stop_pod(a.pod_id)
    print(f"stopped: {a.pod_id} (volume kept and billed; terminate to delete)")
    return 0


def cmd_terminate(cl: api.RunPodClient, a) -> int:
    pod = cl.get_pod(a.pod_id)
    if not a.yes:
        if not sys.stdin.isatty():
            raise SystemExit("terminate deletes the pod and its volume; pass --yes")
        ans = input(
            f"terminate {a.pod_id} ({pod.get('name')}) and delete its volume? [y/N] "
        )
        if ans.strip().lower() not in ("y", "yes"):
            return 1
    cl.terminate_pod(a.pod_id)
    print(f"terminated: {a.pod_id}")
    return 0


def ssh_argv(
    pod: dict, *, identity: Optional[str] = None, remote: list[str] = ()
) -> list[str]:
    ep = api.ssh_endpoint(pod)
    if ep is None:
        raise SystemExit(f"pod {pod.get('id')} has no ssh mapping yet (try `wait`)")
    ip, port = ep
    argv = [
        "ssh",
        "-p",
        str(port),
        # Pod IP:port pairs are recycled across pods; pinning them in
        # known_hosts only produces host-key-changed refusals later.
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
    ]
    if identity:
        argv += ["-i", str(Path(identity).expanduser())]
    argv.append(f"root@{ip}")
    argv += list(remote)
    return argv


def cmd_ssh(cl: api.RunPodClient, a) -> int:
    argv = ssh_argv(cl.get_pod(a.pod_id), identity=a.identity, remote=a.remote)
    os.execvp(argv[0], argv)


def cmd_volumes(cl: api.RunPodClient, a) -> int:
    vols = cl.list_volumes()
    if a.json:
        _print_json(vols)
        return 0
    rows = [
        [
            v.get("id", ""),
            v.get("name", ""),
            f"{v.get('size', '?')}G",
            v.get("dataCenterId", ""),
        ]
        for v in vols
    ]
    _table(rows, ["id", "name", "size", "datacenter"])
    return 0


# ── argparse ──────────────────────────────────────────────────────────


_CUDA_HELP = (
    f"lowest host CUDA version (default {api.DEFAULT_MIN_CUDA}, the image's; "
    "'any' = no filter)"
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m anima_daemon.runpod",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="verb", required=True)

    p = sub.add_parser("gpus", help="GPU types with price and stock")
    p.add_argument("--cloud", choices=("SECURE", "COMMUNITY"), default="SECURE")
    p.add_argument(
        "--count", type=int, default=1, help="GPUs per pod (stock is per count)"
    )
    p.add_argument("--min-vram", type=int, default=0, metavar="GB")
    p.add_argument("--min-cuda", metavar="V", help=_CUDA_HELP)
    p.add_argument("--all", action="store_true", help="include out-of-stock types")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("ls", help="list pods")
    p.add_argument("--all", action="store_true", help="include pods not named anima-*")
    p.add_argument("--json", action="store_true")

    for verb in ("get", "start", "stop"):
        sub.add_parser(verb).add_argument("pod_id")

    p = sub.add_parser("terminate", help="delete a pod and its pod volume")
    p.add_argument("pod_id")
    p.add_argument("--yes", action="store_true")

    p = sub.add_parser("create", help="create a GPU pod (billing starts)")
    p.add_argument(
        "--gpu",
        action="append",
        required=True,
        help="gpu_type_id from `gpus`; repeat for fallbacks in priority order",
    )
    p.add_argument(
        "--image", help=f"default {api.DEFAULT_IMAGE} (none with --template)"
    )
    p.add_argument("--template")
    p.add_argument("--name", help=f"pod name ({api.NAME_PREFIX!r} is prepended)")
    p.add_argument("--cloud", choices=("SECURE", "COMMUNITY"), default="SECURE")
    p.add_argument("--count", type=int, default=1)
    p.add_argument(
        "--disk",
        type=int,
        default=200,
        metavar="GB",
        help="container disk (wiped on restart)",
    )
    p.add_argument(
        "--volume",
        type=int,
        default=0,
        metavar="GB",
        help="pod volume (kept across stop)",
    )
    p.add_argument(
        "--volume-mount",
        default=None,
        help="volume mount path (RunPod default /workspace)",
    )
    p.add_argument(
        "--network-volume",
        metavar="ID",
        help="attach a network volume instead of a pod volume",
    )
    p.add_argument("--env", action="append", default=[], metavar="K=V")
    p.add_argument("--port", action="append", metavar="P/proto", help="default 22/tcp")
    p.add_argument("--min-cuda", metavar="V", help=_CUDA_HELP)
    p.add_argument(
        "--spot",
        action="store_true",
        help="interruptible (cheaper, can be stopped any time)",
    )
    p.add_argument(
        "--ssh-key",
        metavar="PATH",
        help="public key for PUBLIC_KEY (default ~/.ssh/id_*.pub)",
    )
    p.add_argument("--no-ssh-key", action="store_true")
    p.add_argument(
        "--dry-run", action="store_true", help="print the request body, create nothing"
    )
    p.add_argument("--wait", action="store_true", help="block until ssh is reachable")
    p.add_argument("--timeout", type=float, default=900)

    p = sub.add_parser("wait", help="block until the pod's ssh port answers")
    p.add_argument("pod_id")
    p.add_argument("--timeout", type=float, default=900)

    p = sub.add_parser("ssh", help="ssh into a pod (args after -- run remotely)")
    p.add_argument("pod_id")
    p.add_argument("--identity", "-i", metavar="PATH")
    p.add_argument("remote", nargs=argparse.REMAINDER)

    p = sub.add_parser("volumes", help="list network volumes")
    p.add_argument("--json", action="store_true")
    return ap


VERBS = {
    "gpus": cmd_gpus,
    "ls": cmd_ls,
    "get": cmd_get,
    "create": cmd_create,
    "start": cmd_start,
    "stop": cmd_stop,
    "terminate": cmd_terminate,
    "wait": cmd_wait,
    "ssh": cmd_ssh,
    "volumes": cmd_volumes,
}


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.verb == "ssh" and a.remote[:1] == ["--"]:
        a.remote = a.remote[1:]
    try:
        cl = None if (a.verb == "create" and a.dry_run) else api.RunPodClient()
        return VERBS[a.verb](cl, a)
    except api.RunPodError as e:
        print(f"runpod: {e}", file=sys.stderr)
        return 1
