import torch

def compute_alpha_sigma(log_snr):
    alpha_t = torch.sqrt(torch.sigmoid(log_snr))
    sigma_t = torch.sqrt(torch.sigmoid(-log_snr))
    return alpha_t, sigma_t