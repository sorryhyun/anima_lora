"""anima_daemon.runpod — request shapes and CLI plumbing against a fake urlopen."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse

import pytest

from anima_daemon.runpod import api, cli


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_http(monkeypatch):
    """Records each request; replies from ``routes[(method, path)]``."""
    calls: list[dict] = []
    routes: dict = {}

    def urlopen(req, timeout=None):
        url = urllib.parse.urlsplit(req.full_url)
        calls.append(
            {
                "method": req.get_method(),
                "host": url.netloc,
                "path": url.path,
                "query": urllib.parse.parse_qs(url.query),
                "body": json.loads(req.data) if req.data else None,
                "auth": req.get_header("Authorization"),
            }
        )
        reply = routes.get((req.get_method(), url.path), {})
        if isinstance(reply, urllib.error.HTTPError):
            raise reply
        return _Resp(json.dumps(reply).encode() if reply is not None else b"")

    monkeypatch.setattr(api.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("RUNPOD_API_KEY", "rk_test")
    return calls, routes


def test_pod_verbs_hit_v1_paths(fake_http):
    calls, routes = fake_http
    routes[("DELETE", "/v1/pods/p1")] = None
    cl = api.RunPodClient()
    cl.start_pod("p1")
    cl.stop_pod("p1")
    cl.terminate_pod("p1")
    assert [(c["method"], c["path"]) for c in calls] == [
        ("POST", "/v1/pods/p1/start"),
        ("POST", "/v1/pods/p1/stop"),
        ("DELETE", "/v1/pods/p1"),
    ]
    assert {c["host"] for c in calls} == {"rest.runpod.io"}


def _create_args(*argv):
    return cli.build_parser().parse_args(["create", *argv])


def test_create_spot_goes_v1_and_refuses_13_2():
    with pytest.raises(SystemExit):
        cli.build_create_spec_v1(_create_args("--gpu", "G1", "--spot", "--no-ssh-key"))
    a = _create_args(
        "--gpu", "G1", "--gpu", "G2", "--spot", "--no-ssh-key", "--min-cuda", "12.8"
    )
    spec = cli.build_create_spec_v1(a)
    assert spec["gpuTypeIds"] == ["G1", "G2"] and spec["interruptible"] is True
    assert spec["allowedCudaVersions"] == ["13.0", "12.9", "12.8"]


def test_create_falls_back_to_next_gpu(fake_http, monkeypatch, capsys):
    tried = []

    def create_pod_v2(self, spec):
        tried.append(spec["gpu"]["id"])
        if spec["gpu"]["id"] == "G1":
            raise api.RunPodError(409, "no instances available")
        return {"id": "p9", "name": spec["name"], "cost": 0.5}

    monkeypatch.setattr(api.RunPodClient, "create_pod_v2", create_pod_v2)
    rc = cli.main(["create", "--gpu", "G1", "--gpu", "G2", "--no-ssh-key"])
    assert rc == 0 and tried == ["G1", "G2"]
    assert json.loads(capsys.readouterr().out)["id"] == "p9"


def test_wait_ssh_fails_fast_on_exited(fake_http, monkeypatch):
    monkeypatch.setattr(
        api.RunPodClient, "get_pod", lambda self, pid: {"desiredStatus": "EXITED"}
    )
    with pytest.raises(api.RunPodError):
        api.RunPodClient().wait_ssh("p1", probe=False)
