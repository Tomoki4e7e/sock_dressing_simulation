import math
import torch
import pytest

from sock_dressing_simulation.flow_chunk_execution import FlowChunkExecutor


class ToyFlow:
    horizon = 8
    action_dim = 16
    indices = list(range(8)) + list(range(9, 17))

    def __init__(self):
        self.draws = []

    def sample_offsets(self, condition, steps, initial_noise):
        self.draws.append(initial_noise.clone())
        return initial_noise + torch.arange(8)[None, :, None] * .1

    def offsets_to_angles(self, offsets, angles):
        return offsets + angles[:, self.indices][:, None]

    def denormalize_actions(self, angles, low, high):
        return angles

    def expand_mimic_joints(self, angles, multipliers, offsets):
        return torch.cat((angles[..., :8], angles[..., 7:8] * .5,
                          angles[..., 8:], angles[..., 15:16] * .5), dim=-1)


def executor(model, execute, ode=8):
    return FlowChunkExecutor(model, None, None, [.5, .5], [0, 0], ode, execute, 0)


def test_same_observation_noise_schedule_for_all_execution_lengths():
    fingerprints = []
    for execute in (1, 2, 4):
        model = ToyFlow()
        policy = executor(model, execute)
        values = []
        for t in range(9):
            _, info = policy.step(torch.zeros(1, 128), torch.zeros(1, 18), t)
            values.append(info['noise_sha256'])
            assert info['plan_slot'] == t % execute
            assert info['replanned'] == (t % execute == 0)
        fingerprints.append(values)
        assert len(model.draws) == math.ceil(9 / execute)
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_held_plan_uses_original_absolute_anchor():
    model = ToyFlow()
    policy = executor(model, 4)
    _, initial = policy.step(torch.zeros(1, 128), torch.zeros(1, 18), 0)
    for t in (1, 2, 3):
        action, info = policy.step(torch.zeros(1, 128), torch.full((1, 18), 100.), t)
        torch.testing.assert_close(action[0], torch.tensor(initial['planned_angles18'][t]))
        assert not info['replanned']
    assert len(model.draws) == 1


def test_first_slot_matches_original_per_call_sampler_noise():
    model = ToyFlow()
    policy = executor(model, 1)
    generator = torch.Generator().manual_seed(0)
    for t in range(5):
        expected = torch.randn((1, 8, 16), generator=generator)
        policy.step(torch.zeros(1, 128), torch.zeros(1, 18), t)
        torch.testing.assert_close(model.draws[-1], expected, rtol=0, atol=0)


def test_paired_ode_settings_receive_identical_noise_and_reject_skipped_observations():
    a, b = ToyFlow(), ToyFlow()
    pa, pb = executor(a, 2, 8), executor(b, 2, 32)
    for t in range(5):
        pa.step(torch.zeros(1, 128), torch.zeros(1, 18), t)
        pb.step(torch.zeros(1, 128), torch.zeros(1, 18), t)
    for x, y in zip(a.draws, b.draws):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    with pytest.raises(ValueError, match='consecutively'):
        pa.step(torch.zeros(1, 128), torch.zeros(1, 18), 7)
