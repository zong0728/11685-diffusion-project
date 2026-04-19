from typing import List, Optional, Tuple, Union

import torch 
import torch.nn as nn 
import numpy as np

from utils import randn_tensor


class DDPMScheduler(nn.Module):
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: Optional[int] = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        variance_type: str = "fixed_small",
        prediction_type: str = 'epsilon',
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        """
        Args:
            num_train_timesteps (`int`): 
            
        """
        super(DDPMScheduler, self).__init__()
        
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.prediction_type = prediction_type
        self.beta = beta_start
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        
    
        # TODO: calculate betas
        if self.beta_schedule == 'linear':
            # This is the DDPM implementation
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif self.beta_schedule == 'cosine':
            # Cosine schedule from Nichol & Dhariwal "Improved DDPM" (ICML 2021).
            # alphas_bar(t) = cos( (t/T + s) / (1 + s) * pi/2 )^2
            s = 0.008
            steps = num_train_timesteps + 1
            t = torch.linspace(0, num_train_timesteps, steps, dtype=torch.float32) / num_train_timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.999)
        else:
            raise NotImplementedError(f"Beta schedule {self.beta_schedule} not implemented.")
        self.register_buffer("betas", betas)

        # TODO: calculate alphas
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        # TODO: calculate alpha cumulative product
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # TODO: timesteps
        timesteps = torch.arange(num_train_timesteps - 1, -1, -1)
        self.register_buffer("timesteps", timesteps)
        

    def set_timesteps(
        self,
        num_inference_steps: int = 250,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )
            
        # TODO: set timesteps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = np.arange(0, num_inference_steps)[::-1] * step_ratio
        self.timesteps = torch.from_numpy(timesteps.copy()).to(device)
        self.num_inference_steps = num_inference_steps


    def __len__(self):
        return self.num_train_timesteps


    def previous_timestep(self, timestep):
        """
        Get the previous timestep for a given timestep.
        
        Args:
            timestep (`int`): The current timestep.
        
        Return: 
            prev_t (`int`): The previous timestep.
        """
        num_inference_steps = (
            self.num_inference_steps if self.num_inference_steps else self.num_train_timesteps
        )
        # TODO: caluclate previous timestep
        prev_t = timestep - self.num_train_timesteps // num_inference_steps
        return prev_t

    
    def vlb_terms(self, x_start, x_t, timesteps, eps_pred, var_coef):
        """Per-sample VLB-style KL between q(x_{t-1} | x_t, x_0) and p_theta(x_{t-1} | x_t).

        Used only when variance_type == 'learned_range'. Returns a tensor of shape (B,).
        Both q and p_theta are diagonal Gaussians with the same dimensionality, so the
        closed-form KL reduces to a per-pixel expression that we sum/mean over spatial dims.
        The eps_pred is treated as stop-grad so this term only trains the variance head
        (Improved DDPM, Nichol 2021).
        """
        alphas_cumprod = self.alphas_cumprod.to(dtype=x_start.dtype)
        alpha_bar_t = alphas_cumprod[timesteps]
        prev_t = timesteps - 1  # at t=0 this yields -1 → clamp to use alpha_bar_prev=1
        alpha_bar_prev = torch.where(
            prev_t >= 0,
            alphas_cumprod[prev_t.clamp(min=0)],
            torch.ones_like(alpha_bar_t),
        )
        beta_t = 1 - alpha_bar_t / alpha_bar_prev
        # Broadcast scalars-per-sample to (B, 1, 1, 1).
        def _b(x):
            while len(x.shape) < len(x_start.shape):
                x = x.unsqueeze(-1)
            return x
        alpha_bar_t, alpha_bar_prev, beta_t = map(_b, (alpha_bar_t, alpha_bar_prev, beta_t))

        # Posterior q(x_{t-1}|x_t, x_0) — diagonal Gaussian, same formula as DDPM eq. (7).
        posterior_mean = (
            (alpha_bar_prev ** 0.5) * beta_t / (1 - alpha_bar_t) * x_start
            + ((1 - alpha_bar_prev) * (1 - beta_t).clamp(min=1e-20) ** 0.5) / (1 - alpha_bar_t) * x_t
        )
        posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        posterior_log_var = torch.log(posterior_var.clamp(min=1e-20))

        # Model p_theta(x_{t-1}|x_t): mean derived from eps_pred (stop-grad), var from var_coef.
        eps_pred = eps_pred.detach()
        pred_x0 = (x_t - (1 - alpha_bar_t) ** 0.5 * eps_pred) / (alpha_bar_t ** 0.5)
        if self.clip_sample:
            pred_x0 = pred_x0.clamp(-self.clip_sample_range, self.clip_sample_range)
        model_mean = (
            (alpha_bar_prev ** 0.5) * beta_t / (1 - alpha_bar_t) * pred_x0
            + ((1 - alpha_bar_prev) * (1 - beta_t).clamp(min=1e-20) ** 0.5) / (1 - alpha_bar_t) * x_t
        )
        # learned variance via interpolation between posterior_var (min) and beta_t (max).
        min_log = posterior_log_var
        max_log = torch.log(beta_t.clamp(min=1e-20))
        frac = (var_coef + 1) / 2
        model_log_var = frac * max_log + (1 - frac) * min_log

        # KL(N(mu1, v1) || N(mu2, v2)) elementwise, then mean over non-batch dims.
        kl = 0.5 * (
            -1.0
            + model_log_var - posterior_log_var
            + torch.exp(posterior_log_var - model_log_var)
            + ((posterior_mean - model_mean) ** 2) * torch.exp(-model_log_var)
        )
        # Natural-log KL → bits per dim not needed; just mean over pixels.
        return kl.mean(dim=list(range(1, kl.ndim)))

    def _split_model_output(self, model_output, channels):
        """Split a model_output that may carry learned variance coefficients.

        If variance_type == 'learned_range', the UNet emits 2*C channels: the first C are
        the eps/v prediction consumed by the mean update, the last C are per-pixel
        interpolation coefficients in (-1, 1) after a sigmoid that blend log(beta_t) and
        log(beta_tilde_t) (Improved DDPM, Nichol 2021, eq. 15).
        Returns (pred, var_coef) where var_coef is None when no learned variance is used.
        """
        if self.variance_type == 'learned_range' and model_output.shape[1] == 2 * channels:
            pred, var_coef = torch.split(model_output, channels, dim=1)
            return pred, var_coef
        return model_output, None

    def _get_variance(self, t, var_coef=None):
        """
        This is one of the most important functions in the DDPM. It calculates the variance $sigma_t$ for a given timestep.
        
        Args:
            t (`int`): The current timestep.
        
        Return:
            variance (`torch.Tensor`): The variance $sigma_t$ for the given timestep.
        """
        
        
        # TODO: calculate $beta_t$ for the current timestep using the cumulative product of alphas
        prev_t = self.previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # TODO: For t > 0, compute predicted variance $\beta_t$ (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        # TODO: we start with two types of variance as mentioned in Section 3.2 of https://arxiv.org/pdf/2006.11239.pdf
        # 1. fixed_small: $\sigma_t = \beta_t$, this one is optimal for $x_0$ being deterministic
        # 2. fixed_large: $\sigma_t^2 = \beta$, this one is optimal for $x_0 \sim mathcal{N}(0, 1)$
        if self.variance_type == "fixed_small":
            # TODO: fixed small variance
            variance = variance
        elif self.variance_type == "fixed_large":
            # TODO: fixed large variance
            variance = current_beta_t
            # TODO: small hack: set the initial (log-)variance like so to get a better decoder log likelihood.
            if t == 1:
                variance = variance
        elif self.variance_type == "learned_range":
            # Improved DDPM (Nichol & Dhariwal 2021, eq. 15): the model emits coefficients
            # `var_coef` that interpolate between min_log=log(beta_tilde_t) and max_log=log(beta_t).
            # frac = (var_coef + 1) / 2 maps sigmoid/tanh-like outputs into [0, 1].
            min_log = torch.log(torch.clamp(variance, min=1e-20))
            max_log = torch.log(torch.clamp(current_beta_t, min=1e-20))
            if var_coef is None:
                # Fallback when called without coefficients (e.g. DDIM sampling). Match fixed_small.
                variance = variance
            else:
                frac = (var_coef + 1) / 2
                log_variance = frac * max_log + (1 - frac) * min_log
                variance = torch.exp(log_variance)
        else:
            raise NotImplementedError(f"Variance type {self.variance_type} not implemented.")

        return variance
    
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor: 
        """
        Add noise to the original samples. This function is used to add noise to the original samples at the beginning of each training iteration.
        
        
        Args:
            original_samples (`torch.Tensor`): 
                The original samples.
            noise (`torch.Tensor`): 
                The noise tensor.
            timesteps (`torch.IntTensor`): 
                The timesteps.
        
        Return:
            noisy_samples (`torch.Tensor`): 
                The noisy samples.
        """
        
        # make sure alphas the on the same device as samples
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        # TODO: get sqrt alphas
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # TODO: get sqrt one miucs alphas
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # TODO: add noise to the original samples using the formula (14) from https://arxiv.org/pdf/2006.11239.pdf
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x0  (Salimans & Ho 2022)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * original_samples


    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
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
            generator (`torch.Generator`, *optional*):
                A random number generator.

        Returns:
            pred_prev_sample (`torch.Tensor`):
                The predicted previous sample.
        """
        
        
        t = timestep
        prev_t = self.previous_timestep(t)

        # Strip learned-variance coefficients (if any) from the model output before
        # the existing mean-prediction logic below.
        model_output, var_coef = self._split_model_output(model_output, sample.shape[1])

        # TODO: 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # TODO: 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == 'epsilon':
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.prediction_type == 'v_prediction':
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        elif self.prediction_type == 'sample':
            pred_original_sample = model_output
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented.")

        # TODO: 3. Clip or threshold "predicted x_0" (for better sampling quality)
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        # TODO: 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = (current_alpha_t ** 0.5 * beta_prod_t_prev) / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample


        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            # When var_coef is present (learned_range), _get_variance returns a per-pixel tensor
            # with the same shape as `sample`; otherwise it returns a scalar. Both broadcast fine.
            variance = (self._get_variance(t, var_coef=var_coef) ** 0.5) * variance_noise

        # TODO: add variance to prev_sample
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample