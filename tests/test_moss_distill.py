"""Self-distillation (library/moss/distill.py): the teacher's weights are swapped in
for its forward only, and its output replaces the flow target."""

import torch
import torch.nn.functional as F

from library.moss.distill import TeacherWeights
from library.moss.training.losses import training_loss


class _Net:
    def __init__(self):
        self.multiplier = 1.0

    def set_multiplier(self, m):
        self.multiplier = m


class _DiT(torch.nn.Module):
    """``pred = w · x``; records the multiplier and weight each forward saw."""

    def __init__(self, net):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(2.0))
        self.net, self.seen = net, []

    def forward(self, x, t, embeds, return_dict=False, ref=None):
        self.seen.append((self.net.multiplier, self.w.item(), embeds.shape[1]))
        return (self.w * x * self.net.multiplier,)


def test_base_teacher_runs_at_multiplier_zero_and_restores():
    net = _Net()
    tw = TeacherWeights("base", net, [], 0.9999)
    with tw:
        assert net.multiplier == 0.0
    assert net.multiplier == 1.0


def test_ema_teacher_swaps_shadow_in_and_back():
    p = torch.nn.Parameter(torch.ones(3))
    tw = TeacherWeights("ema", _Net(), [p], 0.5)
    with torch.no_grad():
        p.fill_(3.0)
    tw.update()  # shadow 1 → 2
    with tw:
        assert torch.equal(p.detach(), torch.full((3,), 2.0))
    assert torch.equal(p.detach(), torch.full((3,), 3.0))


def test_teacher_output_is_the_target():
    net = _Net()
    dit = _DiT(net)
    latent = torch.randn(1, 4, 2, 2)
    student = torch.zeros(1, 5, 8)
    teacher_embeds = torch.zeros(1, 9, 8)
    tw = TeacherWeights("base", net, [], 0.0)

    def teach(noisy, timestep, flow, ref=None):
        with torch.no_grad(), tw:
            return dit(noisy, timestep, teacher_embeds)[0].float() + 1.0

    state, parts = {}, {}
    loss = training_loss(
        dit, latent, student, 0.5, state=state, parts=parts, teacher=teach
    )
    # teacher at multiplier 0 predicts 0, so the target is all ones
    pred = dit.w.detach() * state["noisy"]
    assert torch.allclose(loss, F.mse_loss(pred, torch.ones_like(pred)))
    assert dit.seen[0] == (0.0, 2.0, 9) and dit.seen[1] == (1.0, 2.0, 5)
    assert "fm" in parts
    loss.backward()
    assert dit.w.grad is not None


def test_extra_term_draws_on_grid_inside_band():
    from library.moss.distill import ExtraTerm

    x = ExtraTerm(0.9, 0, "cpu")
    draws = [x.sigma() for _ in range(500)]
    assert min(draws) >= 0.9 and max(draws) <= 1.0
    assert min(draws) == 0.9 and max(draws) == 1.0
    assert all(abs(d * 1000 - round(d * 1000)) < 1e-9 for d in draws)


def test_extra_term_balance_matches_gradient_norms():
    from library.moss.distill import ExtraTerm

    pred = torch.zeros(1, 4, 3, 3, requires_grad=True)
    a, b = torch.randn(1, 4, 3, 3), 0.1 * torch.randn(1, 4, 3, 3)
    main, extra = F.mse_loss(pred, a), F.mse_loss(pred, b)
    (ga,) = torch.autograd.grad(main, pred)
    (gb,) = torch.autograd.grad(ExtraTerm.balance(main, extra) * extra, pred)
    assert torch.allclose(ga.norm(), gb.norm(), rtol=1e-5)


def test_teacher_add_keeps_flow_and_balances():
    net = _Net()
    dit = _DiT(net)
    latent = torch.randn(1, 4, 2, 2)
    gen = torch.Generator().manual_seed(0)

    def teach(noisy, timestep, flow, ref=None):
        return torch.ones_like(noisy)

    state, parts = {}, {}
    loss = training_loss(
        dit,
        latent,
        torch.zeros(1, 5, 8),
        0.5,
        gen=gen,
        state=state,
        parts=parts,
        teacher=teach,
        teacher_add=1.0,
    )
    pred = dit.w.detach() * state["noisy"]
    flow = (state["noisy"] - 0.5 * latent) / 0.5 - latent  # noise - latent
    fm, t = F.mse_loss(pred, flow), F.mse_loss(pred, torch.ones_like(pred))
    assert torch.allclose(loss, fm + (fm / t).sqrt() * t)
    assert abs(parts["t"] - t.item()) < 1e-6 and abs(parts["fm"] - fm.item()) < 1e-6


def test_train_builds_missing_teachers_on_the_cache_te(tmp_path, monkeypatch):
    """``run_train``'s pre-check: missing teacher files are encoded before the DiT
    loads, on the TE weights the cache stamps and the tail of the teachers already
    there; a cache whose teachers are all current is left alone."""
    import os

    from safetensors.torch import save_file

    import library.moss.distill as distill
    from library.moss import train as T
    from library.moss.requests import TrainRequest

    x = {"x": torch.zeros(1)}
    for stem in ("a", "b"):
        save_file(x, str(tmp_path / f"{stem}_32x32.moss_latent.safetensors"))
        save_file(
            x,
            str(tmp_path / f"{stem}.moss_te.safetensors"),
            metadata={"encoder": "Qwen3.5-9B, fp8 weights"},
        )
    vis_a = tmp_path / "a.moss_vis.safetensors"
    save_file(x, str(vis_a), metadata={"prompt_tail": "sure"})
    calls = []
    monkeypatch.setattr(distill, "cache_teachers", lambda r: calls.append(r))
    req = TrainRequest(cache=str(tmp_path), distill="ema")

    T._ensure_teachers(req, tmp_path)
    assert [(r.text_encoder, r.tail) for r in calls] == [("fp8", "sure")]

    save_file(x, str(tmp_path / "b.moss_vis.safetensors"))
    now = os.stat(tmp_path / "b.moss_te.safetensors").st_mtime + 1
    for v in (vis_a, tmp_path / "b.moss_vis.safetensors"):
        os.utime(v, (now, now))
    T._ensure_teachers(req, tmp_path)
    assert len(calls) == 1
