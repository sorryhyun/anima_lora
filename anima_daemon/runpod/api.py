"""RunPod REST client — pods, network volumes, the GPU catalog.

``urllib``-only like ``anima_daemon.client``. API split and key lookup:
``anima_daemon/README.md`` "RunPod pods". ``~/.runpod/config.toml`` is the file
``runpodctl config --apiKey`` writes.
"""

from __future__ import annotations

import json
import os
import socket
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

REST_URL = "https://rest.runpod.io/v1"
API_URL = "https://api.runpod.io/v2"
CONFIG_PATH = Path.home() / ".runpod" / "config.toml"
DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"

# Pods this tooling creates carry the prefix, so `ls` can tell them from ones
# made in the web console.
NAME_PREFIX = "anima-"

# Cloudflare in front of api.runpod.io 403s urllib's default "Python-urllib/x".
USER_AGENT = "anima-lora-runpod/1"

# docker/Dockerfile, pushed to Docker Hub. It runs sshd when PUBLIC_KEY is set
# and fetches the first-run models into /app/models on start.
DEFAULT_IMAGE = "sorryhyun96/anima_lora:latest"
# The image's torch is cu132: pods go on hosts whose driver runs CUDA ≥ 13.2.
DEFAULT_MIN_CUDA = "13.2"
# v1 POST /pods allowedCudaVersions enum (newest first); v2 takes any major.minor.
V1_CUDA_VERSIONS = (
    "13.0",
    "12.9",
    "12.8",
    "12.7",
    "12.6",
    "12.5",
    "12.4",
    "12.3",
    "12.2",
    "12.1",
    "12.0",
    "11.8",
)


class RunPodError(RuntimeError):
    def __init__(self, status: int, message: str, *, method: str = "", url: str = ""):
        where = f"{method} {url} → " if method else ""
        super().__init__(
            f"{where}HTTP {status}: {message}" if status else f"{where}{message}"
        )
        self.status = status
        self.message = message


def _dotenv_value(name: str) -> Optional[str]:
    try:
        lines = DOTENV_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        k, sep, v = line.strip().removeprefix("export ").partition("=")
        if sep and k.strip() == name:
            return v.strip().strip("'\"") or None
    return None


def resolve_api_key() -> Optional[str]:
    key = os.environ.get("RUNPOD_API_KEY") or _dotenv_value("RUNPOD_API_KEY")
    if key:
        return key.strip()
    try:
        with open(CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    key = cfg.get("apikey") or cfg.get("apiKey")
    return key.strip() if isinstance(key, str) and key.strip() else None


def _clean(params: dict) -> dict:
    return {k: v for k, v in params.items() if v is not None}


class RunPodClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        rest_url: str = REST_URL,
        api_url: str = API_URL,
        timeout: float = 30.0,
    ):
        key = api_key or resolve_api_key()
        if not key:
            raise RunPodError(
                401,
                f"no API key: set RUNPOD_API_KEY (environment or the repo .env) "
                f"or write apikey to {CONFIG_PATH}",
            )
        self._key = key
        self.rest_url = rest_url.rstrip("/")
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        url: str,
        *,
        body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> Any:
        if params:
            # Array params repeat the key (?gpuTypeId=a&gpuTypeId=b).
            url = f"{url}?{urllib.parse.urlencode(_clean(params), doseq=True)}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", USER_AGENT)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raise RunPodError(
                e.code, _error_message(e.read()), method=method, url=url
            ) from None
        except urllib.error.URLError as e:
            raise RunPodError(0, str(e.reason), method=method, url=url) from None
        if not raw:
            return None
        return json.loads(raw)

    def _rest(self, method: str, path: str, **kw) -> Any:
        return self._request(method, f"{self.rest_url}{path}", **kw)

    # ── catalog (v2) ──────────────────────────────────────────────────
    def gpu_types(
        self,
        *,
        availability: bool = True,
        cloud: str = "SECURE",
        count: int = 1,
        min_cuda: Optional[str] = None,
    ) -> list[dict]:
        params: dict = {"minCudaVersion": min_cuda}
        if availability:
            params |= {
                "include": "AVAILABILITY",
                "product": "POD",
                "cloud": cloud,
                "count": count,
            }
        out = self._request("GET", f"{self.api_url}/catalog/gpus", params=params)
        return (out or {}).get("gpus", [])

    # ── pods (v1) ─────────────────────────────────────────────────────
    def list_pods(self, **filters) -> list[dict]:
        return self._rest("GET", "/pods", params=filters) or []

    def get_pod(self, pod_id: str) -> dict:
        return self._rest("GET", f"/pods/{pod_id}")

    def create_pod(self, spec: dict) -> dict:
        return self._rest("POST", "/pods", body=spec)

    def create_pod_v2(self, spec: dict) -> dict:
        """v2 POST /pods: a single ``gpu.id``, ``gpu.minCudaVersion`` past
        v1's 13.0 cap, no interruptible pods. The pod reads back through v1."""
        return self._request("POST", f"{self.api_url}/pods", body=spec)

    def start_pod(self, pod_id: str) -> Any:
        return self._rest("POST", f"/pods/{pod_id}/start")

    def stop_pod(self, pod_id: str) -> Any:
        return self._rest("POST", f"/pods/{pod_id}/stop")

    def terminate_pod(self, pod_id: str) -> Any:
        return self._rest("DELETE", f"/pods/{pod_id}")

    # ── network volumes (v1) ──────────────────────────────────────────
    def list_volumes(self) -> list[dict]:
        return self._rest("GET", "/networkvolumes") or []

    # ── readiness ─────────────────────────────────────────────────────
    def wait_ssh(
        self,
        pod_id: str,
        *,
        timeout: float = 900.0,
        interval: float = 5.0,
        probe: bool = True,
        on_poll=None,
    ) -> dict:
        """Poll until the pod is RUNNING with a public IP and a 22/tcp mapping
        (and, with ``probe``, the port accepts a TCP connect). Returns the pod;
        raises ``TimeoutError`` past ``timeout`` and ``RunPodError`` if the pod
        exits or is terminated meanwhile."""
        deadline = time.monotonic() + timeout
        while True:
            pod = self.get_pod(pod_id)
            status = pod.get("desiredStatus")
            if status in ("EXITED", "TERMINATED"):
                raise RunPodError(409, f"pod {pod_id} is {status}")
            ep = ssh_endpoint(pod)
            if on_poll is not None:
                on_poll(pod, ep)
            if status == "RUNNING" and ep and (not probe or _tcp_open(*ep)):
                return pod
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"pod {pod_id} not reachable over ssh in {timeout:.0f}s"
                )
            time.sleep(interval)


def ssh_endpoint(pod: dict) -> Optional[tuple[str, int]]:
    """``(public_ip, external_port)`` for the pod's 22/tcp, or None while it is
    still initializing (``publicIp`` empty, ``portMappings`` without 22)."""
    ip = pod.get("publicIp")
    port = (pod.get("portMappings") or {}).get("22")
    if not ip or not port:
        return None
    return ip, int(port)


def _tcp_open(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _error_message(raw: bytes) -> str:
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return raw.decode(errors="replace").strip()[:500]
    if isinstance(payload, dict):
        # v1: {"error": "..."}; v2: problem+json {"title", "detail"}.
        for key in ("error", "detail", "message", "title"):
            if payload.get(key):
                return str(payload[key])
    return json.dumps(payload)[:500]
