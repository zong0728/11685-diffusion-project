import copy
import torch


class EMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of the model whose weights are updated as:
        theta_ema = decay * theta_ema + (1 - decay) * theta_model

    Typical decay values for diffusion models: 0.999 to 0.9999.
    EMA weights are used at inference time and usually produce substantially
    better FID than the raw training weights.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        # Also copy buffers (e.g. BN running stats) verbatim; they are not EMA-averaged.
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)
