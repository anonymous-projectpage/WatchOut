"""
IQL / AWR module for X-VLA.

A PyTorch port of pi05_iql.py (JAX/Flax), keeping the architecture and hyperparameters identical:

  ChunkQNetwork      Q(s, action_chunk)  — context/state/action projection + q_token
                     + Transformer blocks + q_out  (width 256, 3 layers, 8 heads)
  StateValueNetwork  V(s)                uses v_token, takes no action input
  TwinChunkCritic    Q1, Q2 (suppresses overestimation)

Learning signals (identical to pi05_iql.compute_loss_iql):
  critic_target = gamma ** (T-1-t)          (larger near the episode end, clipped to [0,1])
  q_loss        = 0.5 * (MSE(q1,tgt) + MSE(q2,tgt))
  value_loss    = expectile_weight * (stop_grad(min(q1,q2)) - V)^2
  advantage     = stop_grad(min(q1,q2) - V)
  weight        = min(exp(clip(beta*adv, -20, 20)), max_w), optionally mean-normalized
  actor_loss    = mean(weight * flow_loss_per_sample)

The critic sees stop_gradient(vlm_features), so the critic loss never updates the backbone.

Environment variables
--------
EXPECTILE         default 0.9
ADV_BETA          default 3.0
ADV_STANDARDIZE   "1" standardizes the advantage (default 0)
ACTOR_WEIGHT      multiplier on the actor loss (default 1.0)
CRITIC_GAMMA      default 0.99
MAX_ADV_WEIGHT    default 20.0
NORM_ADV_WEIGHTS  "0" disables mean normalization (default 1)
"""
from __future__ import annotations

import os
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Transformer block (mirrors pi05_iql.CriticTransformerBlock)
# ---------------------------------------------------------------------------
class CriticTransformerBlock(nn.Module):
    def __init__(self, width: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"critic width {width} must be divisible by {num_heads}")
        self.num_heads = num_heads
        self.head_dim = width // num_heads

        self.norm1 = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.attention_out = nn.Linear(width, width)

        self.norm2 = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.ffn_in = nn.Linear(width, hidden)
        self.ffn_out = nn.Linear(hidden, width)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        q = self._split(self.q_proj(h))
        k = self._split(self.k_proj(h))
        v = self._split(self.v_proj(h))
        attn = F.scaled_dot_product_attention(q, k, v)
        B, _, T, _ = attn.shape
        attn = attn.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        x = x + self.attention_out(attn)

        h = self.norm2(x)
        h = F.silu(self.ffn_in(h))          # pi05: nnx.swish
        return x + self.ffn_out(h)


def _pool_context(context: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """[B, T, D] -> [B, D], mask-weighted mean, as in pi05."""
    if mask is None:
        return context.mean(dim=1)
    m = mask.to(context.dtype).unsqueeze(-1)          # [B, T, 1]
    s = (context * m).sum(dim=1)
    c = m.sum(dim=1).clamp_min(1.0)
    return s / c


# ---------------------------------------------------------------------------
# Q(s, action_chunk)
# ---------------------------------------------------------------------------
class ChunkQNetwork(nn.Module):
    def __init__(self, context_dim: int, state_dim: int, action_dim: int,
                 action_horizon: int, width: int = 256,
                 num_layers: int = 3, num_heads: int = 8):
        super().__init__()
        self.context_proj = nn.Linear(context_dim, width)
        self.state_proj = nn.Linear(state_dim, width)
        self.action_proj = nn.Linear(action_dim, width)

        self.q_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.q_token, std=0.02)

        # tokens = [q_token, context, state, action_0..action_{H-1}]
        self.position_embedding = nn.Parameter(torch.zeros(1, 3 + action_horizon, width))
        nn.init.normal_(self.position_embedding, std=0.02)

        self.blocks = nn.ModuleList(
            [CriticTransformerBlock(width, num_heads) for _ in range(num_layers)]
        )
        self.norm_out = nn.LayerNorm(width)
        self.q_out = nn.Linear(width, 1)

    def forward(self, context: torch.Tensor, mask: torch.Tensor | None,
                state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        context : [B, T_enc, D_ctx]   (stop-grad vlm_features)
        mask    : [B, T_enc] or None
        state   : [B, D_state]        (proprio)
        actions : [B, H, D_act]
        return  : [B]
        """
        state = state.float()
        actions = actions.float()

        ctx = self.context_proj(_pool_context(context.float(), mask)).unsqueeze(1)  # [B,1,W]
        st = self.state_proj(state).unsqueeze(1)                                    # [B,1,W]
        act = self.action_proj(actions)                                             # [B,H,W]
        q = self.q_token.expand(ctx.shape[0], -1, -1)                               # [B,1,W]

        tokens = torch.cat([q, ctx, st, act], dim=1)                                # [B,3+H,W]
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.q_out(self.norm_out(tokens[:, 0]))[:, 0]


# ---------------------------------------------------------------------------
# V(s)
# ---------------------------------------------------------------------------
class StateValueNetwork(nn.Module):
    def __init__(self, context_dim: int, state_dim: int, width: int = 256,
                 num_layers: int = 3, num_heads: int = 8):
        super().__init__()
        self.context_proj = nn.Linear(context_dim, width)
        self.state_proj = nn.Linear(state_dim, width)

        self.v_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.v_token, std=0.02)
        self.position_embedding = nn.Parameter(torch.zeros(1, 3, width))
        nn.init.normal_(self.position_embedding, std=0.02)

        self.blocks = nn.ModuleList(
            [CriticTransformerBlock(width, num_heads) for _ in range(num_layers)]
        )
        self.norm_out = nn.LayerNorm(width)
        self.v_out = nn.Linear(width, 1)

    def forward(self, context: torch.Tensor, mask: torch.Tensor | None,
                state: torch.Tensor) -> torch.Tensor:
        state = state.float()
        ctx = self.context_proj(_pool_context(context.float(), mask)).unsqueeze(1)
        st = self.state_proj(state).unsqueeze(1)
        v = self.v_token.expand(ctx.shape[0], -1, -1)

        tokens = torch.cat([v, ctx, st], dim=1) + self.position_embedding
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.v_out(self.norm_out(tokens[:, 0]))[:, 0]


class TwinChunkCritic(nn.Module):
    """Independent Q1, Q2 (mirrors pi05 TwinChunkCritic)."""

    def __init__(self, **kw):
        super().__init__()
        self.q1 = ChunkQNetwork(**kw)
        self.q2 = ChunkQNetwork(**kw)

    def forward(self, context, mask, state, actions):
        return self.q1(context, mask, state, actions), self.q2(context, mask, state, actions)


# ---------------------------------------------------------------------------
# per-sample flow matching loss
# ---------------------------------------------------------------------------
def flow_loss_per_sample(action_space, pred: torch.Tensor, target: torch.Tensor
                         ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Uses the same terms and scale as action_hub.compute_loss, but keeps the batch axis.
    Returns: ([B] per-sample loss, dict of scalars for logging)
    """
    a = action_space
    B = pred.shape[0]

    g = torch.stack(
        [F.binary_cross_entropy_with_logits(pred[:, :, gi], target[:, :, gi], reduction="none")
         .reshape(B, -1).mean(1) for gi in a.gripper_idx]
    ).mean(0) * a.GRIPPER_SCALE

    def _mse(idx):
        return F.mse_loss(pred[:, :, idx], target[:, :, idx], reduction="none") \
                .reshape(B, -1).mean(1)

    pos = (_mse(a.POS_IDX_1) + _mse(a.POS_IDX_2)) * a.XYZ_SCALE
    rot = (_mse(a.ROT_IDX_1) + _mse(a.ROT_IDX_2)) * a.ROT_SCALE

    total = pos + rot + g
    logs = {"position_loss": pos.mean(), "rotate6D_loss": rot.mean(),
            "gripper_loss": g.mean()}
    return total, logs


# ---------------------------------------------------------------------------
# IQL heads and loss
# ---------------------------------------------------------------------------
class XVLAIQL(nn.Module):
    def __init__(self, context_dim: int, state_dim: int, action_dim: int,
                 action_horizon: int, width: int = 256, num_layers: int = 3,
                 num_heads: int = 8):
        super().__init__()
        self.chunk_critic = TwinChunkCritic(
            context_dim=context_dim, state_dim=state_dim, action_dim=action_dim,
            action_horizon=action_horizon, width=width,
            num_layers=num_layers, num_heads=num_heads)
        self.value_network = StateValueNetwork(
            context_dim=context_dim, state_dim=state_dim, width=width,
            num_layers=num_layers, num_heads=num_heads)
        self.target_chunk_critic = TwinChunkCritic(
            context_dim=context_dim, state_dim=state_dim, action_dim=action_dim,
            action_horizon=action_horizon, width=width,
            num_layers=num_layers, num_heads=num_heads)
        self.target_chunk_critic.load_state_dict(self.chunk_critic.state_dict())
        for _p in self.target_chunk_critic.parameters():
            _p.requires_grad_(False)
        self.use_td = os.environ.get("USE_TD", "1") == "1"
        self.gamma = float(os.environ.get("CRITIC_GAMMA", "0.99"))
        self.H = int(action_horizon)
        self.target_tau = float(os.environ.get("TARGET_TAU", "0.005"))

        e = os.environ
        self.expectile = float(e.get("EXPECTILE", 0.9))
        self.advantage_beta = float(e.get("ADV_BETA", 3.0))
        self.adv_standardize = e.get("ADV_STANDARDIZE", "0") == "1"
        self.actor_weight = float(e.get("ACTOR_WEIGHT", "1.0"))
        self.max_advantage_weight = float(e.get("MAX_ADV_WEIGHT", 20.0))
        self.normalize_advantage_weights = e.get("NORM_ADV_WEIGHTS", "1") == "1"
        self.value_loss_weight = float(e.get("VALUE_LOSS_WEIGHT", 1.0))
        print(f"[iql] expectile={self.expectile} beta={self.advantage_beta} "
              f"standardize={self.adv_standardize} actor_weight={self.actor_weight}",
              flush=True)

    @torch.no_grad()
    def update_target(self):
        t = self.target_tau
        for tp, vp in zip(self.target_chunk_critic.parameters(),
                          self.chunk_critic.parameters()):
            tp.mul_(1 - t).add_(vp, alpha=t)

    def losses(self, context: torch.Tensor, mask: torch.Tensor | None,
               state: torch.Tensor, actions: torch.Tensor,
               flow_per_sample: torch.Tensor, critic_target: torch.Tensor,
               next_context: torch.Tensor | None = None,
               next_state: torch.Tensor | None = None,
               reward_cum: torch.Tensor | None = None,
               done_chunk: torch.Tensor | None = None,
               ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computed in the same order as pi05_iql.compute_loss_iql."""
        ctx = context.detach()                      # keep the critic from updating the backbone
        q1, q2 = self.chunk_critic(ctx, mask, state, actions)
        value = self.value_network(ctx, mask, state)

        tgt = critic_target.float().reshape(-1).clamp(0.0, 1.0)
        if (self.use_td and next_context is not None
                and reward_cum is not None and done_chunk is not None):
            with torch.no_grad():
                v_next = self.value_network(
                    next_context.detach(), None, next_state)
                _rc = reward_cum.float().reshape(-1)
                _dn = done_chunk.float().reshape(-1)
                tgt = (_rc + (self.gamma ** self.H) * (1 - _dn) * v_next).clamp(0.0, 1.0)
        q_loss = 0.5 * (F.mse_loss(q1, tgt) + F.mse_loss(q2, tgt))

        with torch.no_grad():
            q1_t, q2_t = self.target_chunk_critic(ctx, mask, state, actions)
            q_target_min = torch.minimum(q1_t, q2_t)
        q_min = q_target_min
        value_error = q_target_min - value
        w = torch.where(value_error >= 0, self.expectile, 1.0 - self.expectile)
        value_loss = (w * value_error.pow(2)).mean()

        advantage = (q_target_min - value).detach()
        if self.adv_standardize:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        log_w = (self.advantage_beta * advantage).clamp(-20.0, 20.0)
        adv_w = log_w.exp().clamp(max=self.max_advantage_weight)
        if self.normalize_advantage_weights:
            adv_w = adv_w / (adv_w.mean() + 1e-6)
        adv_w = adv_w.detach()

        actor_loss = (adv_w * flow_per_sample).mean()
        total = self.actor_weight * actor_loss + q_loss + self.value_loss_weight * value_loss

        logs = {
            "actor_loss": actor_loss.detach(),
            "actor_loss_unweighted": flow_per_sample.mean().detach(),
            "q_loss": q_loss.detach(),
            "value_loss": value_loss.detach(),
            "q_mean": q_min.mean().detach(),
            "v_mean": value.mean().detach(),
            "adv_mean": advantage.mean().detach(),
            "adv_w_mean": adv_w.mean().detach(),
            "adv_w_max": adv_w.max().detach(),
            "critic_target_mean": tgt.mean().detach(),
        }
        return total, logs
