from vitra.models.action_model.dit import DiT
from vitra.models.action_model import create_diffusion
from . import gaussian_diffusion as gd
from vitra.datasets.dataset_utils import ActionFeature
import torch
from torch import nn

def DiT_T(**kwargs):
    return DiT(depth=3, hidden_size=256, num_heads=4, **kwargs)
def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)
def DiT_M(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)
def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)
def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

DiT_models = {'DiT-S': DiT_S, 'DiT-M': DiT_M, 'DiT-B': DiT_B, 'DiT-T': DiT_T, 'DiT-L': DiT_L}

class DiffusionPolicy(nn.Module):
    def __init__(
        self, 
        token_size, 
        model_type='DiT-B', 
        in_channels=192, 
        future_action_window_size=16, 
        past_action_window_size=0, 
        use_state=None, 
        action_type='angle',
        diffusion_steps=100,
        state_dim=None,
        loss_type='human',
    ):
        super().__init__()
        # SimpleMLP takes in x_t, timestep, and condition, and outputs predicted noise.
        self.in_channels = in_channels
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", 
                                        noise_schedule = 'squaredcos_cap_v2', 
                                        diffusion_steps=self.diffusion_steps, 
                                        sigma_small=True, 
                                        learn_sigma = False
                                        ) 
        #self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = 'linear', diffusion_steps=100, sigma_small=True, learn_sigma = False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.use_state = use_state
        self.action_type = action_type
        
        # Get loss components and hand group mapping from ActionFeature
        if loss_type == 'human':
            self.loss_components = ActionFeature.get_loss_components(action_type)
        elif loss_type == 'robot':
            self.loss_components = ActionFeature.get_xhand_loss_components()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.net = DiT_models[model_type](
            token_size = token_size, 
            action_dim = in_channels, 
            class_dropout_prob = 0.1, 
            learn_sigma = learn_sigma, 
            future_action_window_size = future_action_window_size, 
            past_action_window_size = past_action_window_size,
            use_state = use_state,
            state_dim=state_dim
        )

    # Given condition z and ground truth token x, x_mask, compute loss
    def loss(self, x, z, x_mask, state=None, state_mask=None):
        # sample random noise and timestep
        noise = torch.randn_like(x) # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device= x.device)
        
        # sample x_t from x
        x_t = self.diffusion.q_sample(x, timestep, noise)
        x_t = x_t * x_mask
        x_t = torch.cat([x_t, x_mask], dim=2) # [B, T, D]

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, z, state, state_mask)

        assert noise_pred.shape == noise.shape == x.shape

        # L2 loss with mask
        square_delta = (noise_pred - noise) ** 2 * x_mask
        
        # Generic mask loss computation function
        def mask_loss(from_dim, to_dim):
            s = square_delta[:, :, from_dim:to_dim].sum()
            n = x_mask[:, :, from_dim:to_dim].sum()
            return s / n if n > 0 else 0
        
        # Compute loss for each component using ActionFeature definitions
        component_losses = {}
        component_counts = {}
        
        for name, (start, end, weight) in self.loss_components.items():
            component_losses[name] = mask_loss(start, end) * weight
            component_counts[name] = x_mask[:, :, start].sum()
        
        total_count = sum(component_counts.values())

        if total_count == 0:
            loss = square_delta[0, 0, 0]
        else:
            loss = sum(
                component_losses[k] * component_counts[k]
                for k in component_counts.keys()
            ) / total_count

        # Return loss with detailed component losses for logging
        return {
            "loss": loss,
            **component_losses,  # Unpack all component losses
        }
    
    # Given condition and noise, sample x using reverse diffusion process
    def sample(self, 
            action_features,
            cfg_scale,
            current_state,
            current_state_mask,
            use_ddim,
            num_ddim_steps,
            action_masks,
        ):
        B = action_features.shape[0]
        noise = torch.randn(action_features.shape[0], self.future_action_window_size+1, 
                self.in_channels,  device=action_features.device)   #[B, T, D]

        x_mask = action_masks.to(action_features.device)

        using_cfg = cfg_scale > 1.0
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(B, 1, -1) #[B, 1, D]
            z = torch.cat([action_features, uncondition], 0)
            cfg_scale = cfg_scale

            if self.use_state == 'DiT':
                model_kwargs = dict(
                    z=z, x_mask=x_mask, 
                    cfg_scale=cfg_scale, state=current_state, 
                    state_mask=current_state_mask
                )
            else:
                model_kwargs = dict(z=z, x_mask=x_mask, cfg_scale=cfg_scale)
            sample_fn = self.net.forward_with_cfg
        else:
            if self.use_state == 'DiT':
                model_kwargs = dict(z=z, x_mask=x_mask, state=current_state, state_mask=current_state_mask)
            else:
                model_kwargs = dict(z=z, x_mask=x_mask)
            sample_fn = self.net.forward

        if use_ddim and num_ddim_steps is not None:
            if self.ddim_diffusion is None:
                self.create_ddim(ddim_step=num_ddim_steps)
            samples = self.ddim_diffusion.ddim_sample_loop(
                sample_fn, 
                noise.shape, 
                noise, 
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=action_features.device,
                eta=0.0
            )
        else:
            samples = self.ddim_diffusion.diffusion.p_sample_loop(
                sample_fn, 
                noise.shape, 
                noise, 
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=action_features.device
            )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        return samples

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        self.ddim_diffusion = create_diffusion(
            timestep_respacing="ddim"+str(ddim_step), 
            noise_schedule = 'squaredcos_cap_v2', 
            diffusion_steps=self.diffusion_steps, 
            sigma_small=True, 
            learn_sigma = False
        )
        return self.ddim_diffusion