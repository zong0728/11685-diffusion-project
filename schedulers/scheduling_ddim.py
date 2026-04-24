from typing import List, Optional, Tuple, Union

import torch 
import torch.nn as nn 
import numpy as np

from utils import randn_tensor

from.scheduling_ddpm import DDPMScheduler


class DDIMScheduler(DDPMScheduler):    
    
    def __init__(self, *args, **kwargs):
        # Pop our extensions before forwarding to DDPMScheduler.
        self.default_eta = kwargs.pop('default_eta', 0.0)
        # If True, set_timesteps() picks timesteps from a Karras (rho=7) sigma schedule
        # instead of uniform timestep spacing — concentrates inference effort in mid-noise.
        self.use_karras_sigmas = kwargs.pop('use_karras_sigmas', False)
        self.karras_rho = kwargs.pop('karras_rho', 7.0)
        super().__init__(*args, **kwargs)
        assert self.num_inference_steps is not None, "Please set `num_inference_steps` before running inference using DDIM."
        self.set_timesteps(self.num_inference_steps)

    def set_timesteps(self, num_inference_steps, device=None):
        if not self.use_karras_sigmas:
            return super().set_timesteps(num_inference_steps, device=device)

        # Karras sigma schedule (Karras et al. 2022 "Elucidating the Design Space of Diffusion-Based
        # Generative Models", eq. 5). sigma_t corresponds to noise std at timestep t in our DDPM:
        #   sigma_t = sqrt((1 - alpha_bar_t) / alpha_bar_t)
        # We pick N sigmas spaced via the rho-parameterization, then map each back to the closest
        # training timestep.
        alphas_cumprod = self.alphas_cumprod.cpu().numpy()
        sigmas_train = np.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        sigma_min, sigma_max = float(sigmas_train.min()), float(sigmas_train.max())

        rho = self.karras_rho
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho   # large → small

        # Map each sigma back to the training timestep with the closest sigma (nearest-neighbor).
        timesteps = np.array([int(np.argmin(np.abs(sigmas_train - s))) for s in sigmas])
        # Deduplicate while preserving descending order (in case two sigmas mapped to the same t).
        seen = set()
        dedup = []
        for t in timesteps:
            if t not in seen:
                seen.add(t)
                dedup.append(t)
        timesteps = np.array(dedup, dtype=np.int64)

        self.timesteps = torch.from_numpy(timesteps).to(device) if device is not None else torch.from_numpy(timesteps)
        self.num_inference_steps = len(timesteps)
        # Cache the descending timestep list for previous_timestep() lookup.
        self._karras_timestep_list = timesteps.tolist()

    def previous_timestep(self, timestep):
        if not self.use_karras_sigmas:
            return super().previous_timestep(timestep)
        # Find this t in our non-uniform list and return the next one (smaller). If we're at
        # the final step (smallest t), return -1 so the caller treats alpha_bar_prev as 1.
        try:
            idx = self._karras_timestep_list.index(int(timestep))
        except ValueError:
            return super().previous_timestep(timestep)
        if idx + 1 < len(self._karras_timestep_list):
            return self._karras_timestep_list[idx + 1]
        return -1

    
    def _get_variance(self, t):
        """
        This is one of the most important functions in the DDIM. It calculates the variance $sigma_t$ for a given timestep.
        
        Args:
            t (`int`): The current timestep.
        
        Return:
            variance (`torch.Tensor`): The variance $sigma_t$ for the given timestep.
        """
        
        
        # TODO: calculate $beta_t$ for the current timestep using the cumulative product of alphas
        prev_t = self.previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # TODO: DDIM equation for variance
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance
    
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        eta: float=None,
    ) -> torch.Tensor:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of the noise to add to the variance.
            generator (`torch.Generator`, *optional*):
                A random number generator.

        Returns:
            pred_prev_sample (`torch.Tensor`):
                The predicted previous sample.
        """

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"
        
        t = timestep
        prev_t = self.previous_timestep(t)

        # Resolve eta: explicit kwarg > scheduler default. Lets the pipeline stay simple.
        if eta is None:
            eta = self.default_eta

        # Discard learned-variance coefficients — DDIM is deterministic (or has its own sigma via eta).
        model_output, _ = self._split_model_output(model_output, sample.shape[1])

        # TODO: 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t

        # TODO: 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == 'epsilon':
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = model_output
        elif self.prediction_type == 'v_prediction':
            # v-parameterization (Salimans & Ho 2022, "Progressive Distillation")
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
            pred_epsilon = (beta_prod_t ** 0.5) * sample + (alpha_prod_t ** 0.5) * model_output
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented.")

        # TODO: 3. Clip or threshold "predicted x_0" (for better sampling quality)
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        # TODO: 4. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(t)
        std_dev_t = eta * variance ** 0.5

        # TODO: 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * pred_epsilon

        # TODO: 6. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        # TODO: 7. Add noise with eta
        if eta > 0:
            variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        return prev_sample