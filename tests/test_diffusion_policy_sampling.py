import types

import pytest
import torch
from vitra.models.action_model.diffusion_policy import DiffusionPolicy


class _RecordingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.z_embedder = types.SimpleNamespace(uncondition=torch.zeros(1, 3))
        self.regular_calls = []
        self.cfg_calls = []

    def forward(self, x, timestep, z, state=None, state_mask=None):
        self.regular_calls.append(
            {
                "x": x.detach().clone(),
                "timestep": timestep.detach().clone(),
                "z": z.detach().clone(),
                "state": state,
                "state_mask": state_mask,
            }
        )
        return torch.zeros(
            x.shape[0], x.shape[1], 2, dtype=x.dtype, device=x.device
        )

    def forward_with_cfg(
        self, x, timestep, z, x_mask, cfg_scale, state=None, state_mask=None
    ):
        self.cfg_calls.append(
            {
                "x": x.detach().clone(),
                "timestep": timestep.detach().clone(),
                "z": z.detach().clone(),
                "x_mask": x_mask.detach().clone(),
                "cfg_scale": cfg_scale,
            }
        )
        return torch.zeros(
            x.shape[0], x.shape[1], 2, dtype=x.dtype, device=x.device
        )


class _RecordingSampler:
    def __init__(self):
        self.calls = []

    def _run(self, name, model, shape, noise, model_kwargs, device):
        self.calls.append(
            {
                "name": name,
                "shape": shape,
                "noise": noise.detach().clone(),
                "model_kwargs": model_kwargs,
            }
        )
        timestep = torch.zeros(shape[0], dtype=torch.long, device=device)
        model(noise, timestep, **model_kwargs)
        return torch.ones(shape, dtype=noise.dtype, device=noise.device)

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=False,
        **kwargs,
    ):
        return self._run("ddpm", model, shape, noise, model_kwargs, device)

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        **kwargs,
    ):
        return self._run("ddim", model, shape, noise, model_kwargs, device)


def _make_policy():
    policy = object.__new__(DiffusionPolicy)
    torch.nn.Module.__init__(policy)
    policy.future_action_window_size = 1
    policy.in_channels = 2
    policy.use_state = None
    policy.net = _RecordingNet()
    policy.diffusion = _RecordingSampler()
    policy.ddim_diffusion = None
    policy._ddim_steps = None
    return policy


def _inputs():
    features = torch.tensor([[[0.2, -0.4, 0.6]]])
    mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    return features, mask


def test_standard_sampling_uses_ddpm_and_preserves_action_mask():
    policy = _make_policy()
    features, mask = _inputs()

    samples = policy.sample(
        features,
        cfg_scale=1.0,
        current_state=None,
        current_state_mask=None,
        use_ddim=False,
        num_ddim_steps=None,
        action_masks=mask,
    )

    assert [call["name"] for call in policy.diffusion.calls] == ["ddpm"]
    assert policy.ddim_diffusion is None
    assert len(policy.net.regular_calls) == 1

    call = policy.diffusion.calls[0]
    assert torch.equal(call["model_kwargs"]["z"], features)
    assert torch.equal(call["model_kwargs"]["x_mask"], mask)
    # Invalid dimensions are zeroed before denoising, and the second half is
    # the validity mask expected by the action network.
    assert torch.equal(call["noise"] * (1.0 - mask), torch.zeros_like(mask))
    model_input = policy.net.regular_calls[0]["x"]
    assert torch.equal(model_input[..., 2:], mask)
    assert torch.equal(samples, mask)


def test_ddim_schedule_is_rebuilt_when_step_count_changes():
    policy = _make_policy()
    features, mask = _inputs()
    created_steps = []

    def create_ddim(*, ddim_step):
        created_steps.append(ddim_step)
        policy.ddim_diffusion = _RecordingSampler()
        policy._ddim_steps = ddim_step

    policy.create_ddim = create_ddim

    for steps in (4, 4, 8):
        policy.sample(
            features,
            cfg_scale=1.0,
            current_state=None,
            current_state_mask=None,
            use_ddim=True,
            num_ddim_steps=steps,
            action_masks=mask,
        )

    assert created_steps == [4, 8]
    assert [call["name"] for call in policy.ddim_diffusion.calls] == ["ddim"]


def test_sampling_rejects_a_mask_with_wrong_horizon_or_dimension():
    policy = _make_policy()
    features, _ = _inputs()

    with pytest.raises(ValueError, match="action_masks"):
        policy.sample(
            features,
            cfg_scale=1.0,
            current_state=None,
            current_state_mask=None,
            use_ddim=False,
            num_ddim_steps=None,
            action_masks=torch.ones(1, 1, 2),
        )
