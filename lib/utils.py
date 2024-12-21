import torch


def vp_edm_denoiser(x_t, t, model, alphas_cumprod, class_labels=None):
    alpha_t = alphas_cumprod[t]
    alpha_t = torch.tensor(alpha_t, device=x_t.device).float()
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    sigma_t = sqrt_one_minus_alpha_t / sqrt_alpha_t
    x_sigma = x_t / sqrt_alpha_t
    sigma_t = sigma_t.expand(x_t.shape[0])
    x_0_pred = model(x_sigma, sigma_t, class_labels)
    return x_0_pred


def get_sampling_timesteps(num_steps, schedule_fn, device='cuda'):
    t_vals = torch.linspace(1., 0., steps=num_steps + 1, device=device)
    return t_vals
