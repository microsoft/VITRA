# Modified from facebookresearch's DiT repos
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------


from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp, RmsNorm, Attention
from torch.nn import functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and conditions                 #
#################################################################################

def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.mlp.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class ActionEmbedder(nn.Module):
    def __init__(self, action_size, hidden_size):
        super().__init__()
        # self.linear = nn.Linear(action_size, hidden_size)
        self.projector = nn.Sequential(
            nn.Linear(action_size, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    def forward(self, x):
        # x = self.linear(x)
        x = self.projector(x)
        return x

class StateEmbedder(nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()
        # self.linear = nn.Linear(action_size, hidden_size)
        self.projector = nn.Sequential(
            nn.Linear(2*state_size, 4*hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(4*hidden_size, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    def forward(self, x):
        # x = self.linear(x)
        x = self.projector(x)
        return x

class MaskAttention(Attention):
    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        # x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LabelEmbedder(nn.Module):
    """
    Embeds conditions into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_size, hidden_size, dropout_prob=0.1, conditions_shape=(1, 1, 2304)):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.linear = nn.Linear(in_size, hidden_size)
        self.dropout_prob = dropout_prob
        if dropout_prob > 0:
            self.uncondition = nn.Parameter(torch.empty(conditions_shape[1:]))

    def token_drop(self, conditions, force_drop_ids=None):
        """
        Drops conditions to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(conditions.shape[0], device=conditions.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        conditions = torch.where(drop_ids.unsqueeze(1).unsqueeze(1).expand(conditions.shape[0], self.uncondition.shape[0], self.uncondition.shape[1]), self.uncondition, conditions)
        return conditions

    def forward(self, conditions, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            conditions = self.token_drop(conditions, force_drop_ids)
        embeddings = self.linear(conditions)
        return embeddings

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MaskAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RmsNorm, **block_kwargs)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RmsNorm, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        seq_len = x.shape[1]

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.float32), 1
        )
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=causal_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # x = x + self.attn(self.norm1(x))
        # x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        action_dim=192,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        token_size=4096,
        future_action_window_size=1,
        past_action_window_size=0,
        learn_sigma=True,
        use_state=None,
        state_dim=212,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = action_dim * 2
        self.out_channels = action_dim * 2 if learn_sigma else action_dim
        self.class_dropout_prob = class_dropout_prob
        self.num_heads = num_heads
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.use_state = use_state
        self.x_embedder = ActionEmbedder(action_size=self.in_channels, hidden_size=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.z_embedder = LabelEmbedder(in_size=token_size, hidden_size=hidden_size, dropout_prob=class_dropout_prob, conditions_shape=(1, 1, token_size))
        if self.use_state is not None and self.use_state == 'DiT':
            self.state_embedder = StateEmbedder(state_size=state_dim, hidden_size=hidden_size)
        # num_patches = self.x_embedder.num_patches
        # # Will use fixed sin-cos embedding:

        # +1 for the conditional token, and 1 for the current action
        scale = hidden_size ** -0.5
        if self.use_state == 'DiT':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(future_action_window_size + past_action_window_size + 3, hidden_size))
        else:
            self.positional_embedding = nn.Parameter(
                    scale * torch.randn(future_action_window_size + past_action_window_size + 2, hidden_size))

        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.x_embedder.projector[0].weight, std=0.02)
        nn.init.constant_(self.x_embedder.projector[0].bias, 0)
        nn.init.normal_(self.x_embedder.projector[2].weight, std=0.02)
        nn.init.constant_(self.x_embedder.projector[2].bias, 0)

        if self.use_state is not None and self.use_state == 'DiT':
            nn.init.normal_(self.state_embedder.projector[0].weight, std=0.02)
            nn.init.constant_(self.state_embedder.projector[0].bias, 0)
            nn.init.normal_(self.state_embedder.projector[2].weight, std=0.02)
            nn.init.constant_(self.state_embedder.projector[2].bias, 0)
            nn.init.normal_(self.state_embedder.projector[4].weight, std=0.02)
            nn.init.constant_(self.state_embedder.projector[4].bias, 0)
        # nn.init.normal_(self.history_embedder.linear.weight, std=0.02)
        # nn.init.constant_(self.history_embedder.linear.bias, 0)

        # Initialize label embedding table:
        if self.class_dropout_prob > 0:
            nn.init.normal_(self.z_embedder.uncondition, std=0.02)
        nn.init.normal_(self.z_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.z_embedder.linear.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, z, state=None, state_mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        z: (N,) tensor of conditions
        """

        if self.use_state is not None and self.use_state == 'DiT':
            state = state * state_mask
            s = torch.cat([state, state_mask.to(state.dtype)], dim=-1)
            s = self.state_embedder(s)           # (N, 1, D)
        x = self.x_embedder(x)                   # (N, T, D)
        t = self.t_embedder(t)                   # (N, D)
        z = self.z_embedder(z, self.training)    # (N, 1, D)
        # t.unsqueeze(1)
        c = z.squeeze(1) + t                  # (N, 1, D)
        if self.use_state is not None and self.use_state == 'DiT':
            x = torch.cat((z, s, x), dim=1)          # (N, T+2, D)
        else:
            x = torch.cat((z, x), dim=1)
        x = x + self.positional_embedding  # (N, T, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T+2, D)
        x = self.final_layer(x, c)               # (N, T+2, out_channels)
        return x[:,-(self.future_action_window_size+1):,:]  #[B, T, C]

    #TO DO: Check codes for forward_with_cfg
    def forward_with_cfg(self, x, t, z, x_mask, cfg_scale, state=None, state_mask=None): #history
        """
        Forward pass of Diffusion, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        # action_traj = torch.cat([noisy_action, action_mask], dim=2)
        half = half * x_mask
        half = torch.cat([half, x_mask], dim=2)
        if self.use_state == 'DiT' and state is not None:            
            state = state
            state = torch.cat([state, state], dim=0)
            state_mask = state_mask
            state_mask = torch.cat([state_mask, state_mask], dim=0)
            state = state * state_mask
            combined = torch.cat([half, half], dim=0).to(next(self.x_embedder.parameters()).dtype)
            model_out = self.forward(combined, t, z, state, state_mask)
        else:
            combined = torch.cat([half, half], dim=0).to(next(self.x_embedder.parameters()).dtype)
            model_out = self.forward(combined, t, z)

        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :, :self.in_channels], model_out[:, :, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=2)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb