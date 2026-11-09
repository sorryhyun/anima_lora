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
