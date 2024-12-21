import torch
import math


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


def broadcast_to(
        arr, x,
        dtype=None, device=None, ndim=None):
    if x is not None:
        dtype = dtype or x.dtype
        device = device or x.device
        ndim = ndim or x.ndim
    out = torch.as_tensor(arr, dtype=dtype, device=device)
    return out.reshape((-1,) + (1,) * (ndim - 1))


def get_logsnr_schedule(
        schedule,
        logsnr_min: float = -20.,
        logsnr_max: float = 20.,
        rescale: bool = False,
):
    logsnr_min, logsnr_max = torch.as_tensor(logsnr_min, dtype=torch.float64), \
        torch.as_tensor(logsnr_max, dtype=torch.float64)
    if schedule == "linear":
        def logsnr2t(logsnr):
            return torch.sigmoid(logsnr)

        def t2logsnr(t):
            return torch.logit(t)
    elif schedule == "sigmoid":
        logsnr_range = logsnr_max - logsnr_min

        def logsnr2t(logsnr):
            return (logsnr_max - logsnr) / logsnr_range

        def t2logsnr(t):
            return logsnr_max - t * logsnr_range
    elif schedule == "cosine":
        def logsnr2t(logsnr):
            return torch.atan(torch.exp(-0.5 * logsnr)) / (0.5 * torch.pi)

        def t2logsnr(t):
            return -2 * torch.log(torch.tan(t * torch.pi * 0.5))

        rescale = 2 / math.pi if rescale else 1.0
    else:
        raise NotImplementedError("Unknown schedule")

    t_from = logsnr2t(logsnr_max)
    t_to = logsnr2t(logsnr_min)

    def schedule_fn(t):
        _t = t.to(torch.float64)
        _t_from, _t_to = broadcast_to(t_from, _t), broadcast_to(t_to, _t)
        logsnr = t2logsnr(torch.lerp(_t_from, _t_to, _t))
        if rescale:
            _rescale = rescale if isinstance(rescale, float) else 1.0
            t.copy_(logsnr2t(logsnr).to(t.dtype) * _rescale)
        return logsnr.to(t.dtype)

    return schedule_fn