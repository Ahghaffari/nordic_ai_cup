from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


class HybridMAPPO(nn.Module):
    """Shared residual actor + centralized DeepSets critic.

    This is the same V2/V3 architecture. The actor outputs three unconstrained
    residuals. Spawn remains outside the network in the deterministic species
    controller.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        critic_embed_dim: int = 128,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.critic_embed_dim = int(critic_embed_dim)

        self.actor_trunk = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(self.hidden_dim, 3)
        self.log_std = nn.Parameter(
            torch.tensor([-1.2, -1.2, -1.5], dtype=torch.float32)
        )

        self.critic_agent_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.critic_embed_dim),
            nn.Tanh(),
            nn.Linear(self.critic_embed_dim, self.critic_embed_dim),
            nn.Tanh(),
        )
        # mean pool + max pool => 2 * critic_embed_dim = 256 by default.
        self.critic_head = nn.Sequential(
            nn.Linear(2 * self.critic_embed_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.critic_embed_dim),
            nn.Tanh(),
            nn.Linear(self.critic_embed_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in list(self.actor_trunk) + list(self.critic_agent_encoder) + list(self.critic_head):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)

        # Final critic layer should not start with an unnecessarily huge scale.
        final_critic = self.critic_head[-1]
        if isinstance(final_critic, nn.Linear):
            nn.init.orthogonal_(final_critic.weight, gain=1.0)
            nn.init.zeros_(final_critic.bias)

    def actor_distribution(self, obs: torch.Tensor) -> Normal:
        h = self.actor_trunk(obs)
        mean = self.actor_mean(h)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    @torch.no_grad()
    def act_actor(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.actor_distribution(obs)
        raw = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(raw).sum(-1)
        entropy = dist.entropy().sum(-1)
        return raw, log_prob, entropy

    def evaluate_actor(self, obs: torch.Tensor, raw_action: torch.Tensor):
        dist = self.actor_distribution(obs)
        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    def _critic_embeddings(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_agent_encoder(obs)

    def team_value(self, team_obs: torch.Tensor) -> torch.Tensor:
        """Value for one variable-size species state, shape [N,D]."""
        if team_obs.ndim != 2:
            raise ValueError(f"team_obs must be [N,D], got {tuple(team_obs.shape)}")
        z = self._critic_embeddings(team_obs)
        mean_pool = z.mean(dim=0)
        max_pool = z.max(dim=0).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.critic_head(pooled).squeeze(-1)

    def team_value_batch(
        self,
        padded_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Batched centralized value for padded species observations.

        padded_obs: [B,N,D]
        mask:       [B,N] True for real agents
        """
        if padded_obs.ndim != 3:
            raise ValueError(f"padded_obs must be [B,N,D], got {tuple(padded_obs.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"mask must be [B,N], got {tuple(mask.shape)}")

        z = self._critic_embeddings(padded_obs)  # [B,N,E]
        mask_f = mask.to(z.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (z * mask_f).sum(dim=1) / denom

        neg_inf = torch.finfo(z.dtype).min
        masked_z = z.masked_fill(~mask.unsqueeze(-1), neg_inf)
        max_pool = masked_z.max(dim=1).values
        # Safety for a hypothetical empty row.
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.critic_head(pooled).squeeze(-1)
