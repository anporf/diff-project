import numpy as np
from ..utils import vp_edm_denoiser


def get_beta_schedule(num_steps, beta_start=0.0001, beta_end=0.02):
    return np.linspace(beta_start, beta_end, num_steps)


def get_beta_schedule_logUniform(num_steps, beta_start=0.0001, beta_end=0.02):
    return np.exp(np.linspace(np.log(beta_start), np.log(beta_end), num_steps))


def sample_euler_vp(model, noise, betas, alphas_cumprod, class_labels=None, num_steps=50, device='cuda'):
    x_t = noise.to(device)
    x_history = [x_t.cpu()]
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    for t in reversed(range(num_steps)):
        beta_t = betas[t]
        x_0_pred = vp_edm_denoiser(x_t, t, model, alphas_cumprod, class_labels)
        eps = (x_t - sqrt_alphas_cumprod[t] * x_0_pred) / sqrt_one_minus_alphas_cumprod[t]
        x_t = x_t - (beta_t / sqrt_one_minus_alphas_cumprod[t]) * eps
        x_history.append(x_t.cpu())

    return x_t.detach(), x_history
