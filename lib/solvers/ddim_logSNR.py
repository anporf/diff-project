from ..utils import get_sampling_timesteps
from ..logSNR import compute_alpha_sigma

def sample_ddim_logsnr(
    model,
    noise,
    schedule_fn,
    num_steps=50,
    class_labels=None,
    device='cuda'
):
    x_t = noise.to(device)
    x_history = [x_t.cpu()]
    t_vals = get_sampling_timesteps(num_steps, schedule_fn)

    for i in range(num_steps):
        t = t_vals[i]
        t_next = t_vals[i + 1]

        log_snr_t = schedule_fn(t)
        log_snr_next = schedule_fn(t_next)
        alpha_t, sigma_t = compute_alpha_sigma(log_snr_t)
        alpha_next, sigma_next = compute_alpha_sigma(log_snr_next)

        D_x_t = model(x_t.to(device), sigma_t.to(device), class_labels=class_labels)
        eps = (x_t - alpha_t[:, None, None, None] * D_x_t) / sigma_t[:, None, None, None]
        x_t = alpha_next[:, None, None, None] * D_x_t + sigma_next[:, None, None, None] * eps
        x_history.append(x_t.cpu())

    return x_t.detach(), x_history
