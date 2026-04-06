from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        noise_schedule: str = "linear",
    ):
        super().__init__()
        self.timesteps = timesteps
        self.noise_schedule = noise_schedule

        if noise_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif noise_schedule == "cosine":
            # Improved DDPM cosine schedule.
            s = 0.008
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
            alpha_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = betas.clamp(1e-8, 0.2)
        else:
            raise ValueError(f"Unknown noise_schedule: {noise_schedule}")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)

        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alpha_bar", torch.sqrt(1.0 / alpha_bar))
        self.register_buffer("sqrt_recipm1_alpha_bar", torch.sqrt(1.0 / alpha_bar - 1.0))

        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_mean_coef1 = betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_mean_coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(alphas) / (1.0 - alpha_bar)
        self.register_buffer("posterior_variance", posterior_var.clamp(min=1e-20))
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        b = t.shape[0]
        out = arr.gather(0, t)
        return out.view(b, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self._extract(self.sqrt_alpha_bar, t, x0.shape) * x0
            + self._extract(self.sqrt_one_minus_alpha_bar, t, x0.shape) * noise
        )

    def predict_x0_from_noise(self, xt: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alpha_bar, t, xt.shape) * xt
            - self._extract(self.sqrt_recipm1_alpha_bar, t, xt.shape) * noise
        )

    def training_losses(self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred_noise = model(xt, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample_ddpm(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            pred_noise = model(x, t)
            x0_pred = self.predict_x0_from_noise(x, t, pred_noise)
            if clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            mean = (
                self._extract(self.posterior_mean_coef1, t, x.shape) * x0_pred
                + self._extract(self.posterior_mean_coef2, t, x.shape) * x
            )

            if i > 0:
                var = self._extract(self.posterior_variance, t, x.shape)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean

        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_ddpm_with_trajectory(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        snapshots: int = 8,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = torch.randn(shape, device=device)
        every = max(self.timesteps // snapshots, 1)
        trajectory: List[torch.Tensor] = [x.clone()]

        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            pred_noise = model(x, t)
            x0_pred = self.predict_x0_from_noise(x, t, pred_noise)
            if clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            mean = (
                self._extract(self.posterior_mean_coef1, t, x.shape) * x0_pred
                + self._extract(self.posterior_mean_coef2, t, x.shape) * x
            )

            if i > 0:
                var = self._extract(self.posterior_variance, t, x.shape)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean

            if i % every == 0 and len(trajectory) < snapshots:
                trajectory.append(x.clone())

        if len(trajectory) < snapshots:
            trajectory.append(x.clone())

        return x.clamp(-1.0, 1.0), trajectory[:snapshots]

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        steps: int = 100,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device)

        step_indices = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()

        for idx, t_now in enumerate(step_indices):
            t = torch.full((shape[0],), t_now.item(), device=device, dtype=torch.long)
            pred_noise = model(x, t)

            alpha_bar_t = self._extract(self.alpha_bar, t, x.shape)
            x0_pred = self.predict_x0_from_noise(x, t, pred_noise)
            if clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            if idx == len(step_indices) - 1:
                x = x0_pred
                continue

            t_prev_val = step_indices[idx + 1].item()
            t_prev = torch.full((shape[0],), t_prev_val, device=device, dtype=torch.long)
            alpha_bar_prev = self._extract(self.alpha_bar, t_prev, x.shape)

            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t))
                * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
            )
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma ** 2, min=0.0)) * pred_noise
            noise = torch.randn_like(x)

            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

        return x.clamp(-1.0, 1.0)
