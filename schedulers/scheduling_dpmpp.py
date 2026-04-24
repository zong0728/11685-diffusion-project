"""
Thin wrapper around diffusers' DPMSolverMultistepScheduler so it drops into our
DDPMPipeline in place of DDIMScheduler. Main job: expose `step`, `set_timesteps`,
`timesteps`, and `alphas_cumprod` with the same signatures our pipeline already
uses, plus strip learned-variance coefficients from the UNet output (DPM++ is an
ODE solver and doesn't consume variance).
"""
import torch
import torch.nn as nn
from diffusers import DPMSolverMultistepScheduler


class DPMSolverPPWrapper(nn.Module):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 25,
        beta_schedule: str = 'cosine',
        prediction_type: str = 'epsilon',
        solver_order: int = 2,
        use_karras_sigmas: bool = True,
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.prediction_type = prediction_type

        # Map our cosine to diffusers' naming.
        diffusers_schedule = 'squaredcos_cap_v2' if beta_schedule == 'cosine' else beta_schedule

        self._scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=diffusers_schedule,
            prediction_type=prediction_type,
            algorithm_type='dpmsolver++',
            solver_order=solver_order,
            use_karras_sigmas=use_karras_sigmas,
            # 'zero' appends a 0 sigma so step N becomes the termination step (and consumes one
            # extra timestep slot in our for-loop, causing IndexError). 'sigma_min' stops cleanly.
            final_sigmas_type='sigma_min',
        )
        # Match our buffer API so anything that reads scheduler.alphas_cumprod keeps working.
        self.register_buffer('alphas_cumprod', self._scheduler.alphas_cumprod.clone())
        self.register_buffer('timesteps', self._scheduler.timesteps.clone())

    def set_timesteps(self, num_inference_steps, device=None):
        # diffusers' DPM++ does np.array(self.alphas_cumprod) inside set_timesteps,
        # which fails on GPU tensors. Keep its internal alphas_cumprod on CPU.
        self._scheduler.alphas_cumprod = self._scheduler.alphas_cumprod.cpu()
        self._scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
        # diffusers' multi-step solver requires N step() calls for N-step sampling, but looks
        # ahead sigmas[step_index+1] on the final call, so sigmas has length N+1. If our pipeline
        # iterates all `timesteps` and also calls step() at the very last one, step_index goes to
        # N and indexes past the sigmas tensor. The clean fix: expose N-1 timesteps to the pipeline
        # so it makes N-1 step() calls, and diffusers' internal `lower_order_final=True` + the
        # final sigma=sigma_min combo still denoises fully across those N-1 calls.
        base_timesteps = self._scheduler.timesteps
        self.timesteps = base_timesteps[:-1].to(device) if device is not None else base_timesteps[:-1]

    def step(self, model_output, timestep, sample, generator=None):
        # Strip learned-variance half if present (UNet emits 2*C channels under learned_range).
        if model_output.shape[1] == 2 * sample.shape[1]:
            model_output = model_output[:, :sample.shape[1]]
        out = self._scheduler.step(model_output, timestep, sample, generator=generator, return_dict=True)
        return out.prev_sample

    def to(self, device):
        super().to(device)
        # Intentionally leave _scheduler.alphas_cumprod on CPU — diffusers needs it CPU-side
        # for set_timesteps's numpy conversion. step() works on GPU sample tensors regardless.
        return self
