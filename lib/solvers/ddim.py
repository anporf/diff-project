from ..utils import vp_edm_denoiser
import numpy as np


def sample_ddim(model, noise, alphas_cumprod, class_labels=None, num_steps=50, device='cuda'):
    x_t = noise.to(device)
    x_history = [x_t.cpu()]

    for t in reversed(range(num_steps)):
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = 1 - sqrt_alphas_cumprod
        alphaprev = alphas_cumprod[t-1] if t > 0 else alphas_cumprod[0]
        sqrt_alpha_t = sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_prev = np.sqrt(alphaprev)
        sqrt_one_minus_alpha_prev = np.sqrt(1 - alphaprev)
        x_0_pred = vp_edm_denoiser(x_t, t, model, alphas_cumprod, class_labels)
        eps = (x_t - sqrt_alpha_t * x_0_pred) / sqrt_one_minus_alpha_t
        x_t = sqrt_alpha_prev * x_0_pred + sqrt_one_minus_alpha_prev * eps
        x_history.append(x_t.cpu())

    return x_t.detach(), x_history