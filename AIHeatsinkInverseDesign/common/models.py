"""
Model definitions for heatsink forward prediction and inverse generation.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ForwardDataset(Dataset):
    """Dataset for the forward temperature model."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class InverseDataset(Dataset):
    """Dataset for the inverse generative model."""

    def __init__(self, cond: torch.Tensor, target_geom: torch.Tensor) -> None:
        self.cond = cond
        self.target_geom = target_geom

    def __len__(self) -> int:
        return len(self.cond)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cond[idx], self.target_geom[idx]


class ForwardMLP(nn.Module):
    """
    Forward temperature surrogate.
    Input: [condition(5) + bbox(3) + geometry(5)] = 13 dims
    Output: cpu_temp (1 dim)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, architecture: str = "flat") -> None:
        super().__init__()
        valid_architectures = {"flat", "two_branch_concat", "residual", "two_branch_residual_concat"}
        if architecture not in valid_architectures:
            raise ValueError(
                "ForwardMLP architecture must be one of: "
                "flat, two_branch_concat, residual, two_branch_residual_concat."
            )
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture
        self.context_dim = 8
        self.geom_dim = in_dim - self.context_dim

        if architecture == "flat":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
            )
            return

        if in_dim != 13:
            raise ValueError(f"{architecture} ForwardMLP expects 13 inputs, got {in_dim}.")
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.geom_encoder = nn.Sequential(
            nn.Linear(self.geom_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        if architecture == "two_branch_concat":
            self.net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
            )
            return

        self.baseline_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture == "flat":
            return self.net(x)
        if self.architecture == "two_branch_concat":
            context, geom = self.encode_branches(x)
            return self.net(torch.cat([context, geom], dim=-1))
        baseline, residual = self.forward_parts(x)
        return baseline + residual

    def encode_branches(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context_x = x[:, : self.context_dim]
        geom_x = x[:, self.context_dim :]
        return self.context_encoder(context_x), self.geom_encoder(geom_x)

    def forward_parts(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.architecture not in {"residual", "two_branch_residual_concat"}:
            pred = self.forward(x)
            return pred, torch.zeros_like(pred)
        context, geom = self.encode_branches(x)
        baseline = self.baseline_head(context)
        if self.architecture == "residual":
            geom = self.context_gate(context) * geom
        residual = self.residual_head(torch.cat([context, geom], dim=-1))
        return baseline, residual


class ConditionBaselineMLP(nn.Module):
    """Condition+bbox-only teacher used to supervise the geometry residual."""

    def __init__(self, in_dim: int = 8, hidden_dim: int = 256) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CVAE(nn.Module):
    """
    Conditional variational autoencoder.
    Input: cond(8 dims) + geometry(5 dims)
    Output: geometry(5 dims)
    """

    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        latent_dim: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(cond_dim + target_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def encode(self, cond: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([cond, x], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([cond, z], dim=-1))

    def forward(
        self,
        cond: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(cond, x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(cond, z)
        return recon, mu, logvar


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence for the latent Gaussian."""

    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding used by the diffusion denoiser."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device))
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class DiffusionDenoiser(nn.Module):
    """
    Conditional denoiser for heatsink geometry diffusion.
    Input: noisy geometry + condition + timestep embedding
    Output: predicted noise.
    """

    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(target_dim + cond_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, noisy_x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        return self.net(torch.cat([noisy_x, cond, t_emb], dim=-1))
