import argparse
import numpy as np
import torch
import torch.nn.functional as F
from lib.plot import visualize_samples
import pickle
from edm.dnnlib.util import open_url
import json
import lib
from lib.utils import get_logsnr_schedule



def parse_args():
    parser = argparse.ArgumentParser(description="Parser for solver type")
    parser.add_argument('--solver-type', type=str, 
        choices=['euler', 'euler-logUniform', 'ddim', 'ddim-logSNR', 'dpm-logSNR'], 
        required=True, 
        help="Specify the type of solver to use (e.g., euler, ddim)"
    )
    parser.add_argument('--download-dataset', action='store_true')
    return parser.parse_args()


def sample_euler(params, model, noise, class_labels, is_logUniform=False):
    num_steps = params['num_stemps']
    if is_logUniform:
        betas = lib.euler.get_beta_schedule_logUniform(num_steps)
    else:
        betas = lib.euler.get_beta_schedule(num_steps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    x_euler, _ = lib.euler.sample_euler_vp(model, noise, betas, alphas_cumprod, class_labels=class_labels, **params)
    visualize_samples('Euler Method', x_euler)


def sample_ddim(params, model, noise, class_labels, is_logUniform=False):
    betas = lib.euler.get_beta_schedule(params['num_stemps']_steps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)

    x_ddim, _ = lib.ddim.sample_ddim(model, noise, alphas_cumprod, class_labels=class_labels, **params)
    visualize_samples('DDIM Method', x_ddim)


def sample_ddim_logSNR(params, model, noise, class_labels, is_logUniform=False):
    schedule_fn = get_logsnr_schedule(schedule='sigmoid', logsnr_min=-20., logsnr_max=20.)

    x_ddim_logsnr, _ = lib.ddim_logSNR.sample_ddim_logsnr(
        model, noise, schedule_fn, num_steps=300, class_labels=class_labels, **params
    )
    visualize_samples('DDIM Method with Log-SNR Schedule', x_ddim_logsnr)


def sample_dpm_logSNR(params, model, noise, class_labels, is_logUniform=False):
    schedule_fn = get_logsnr_schedule(schedule='linear', logsnr_min=-20., logsnr_max=20.)

    x_dpm_solver_logsnr, _ = lib.dpm_logSNR.sample_dpm_solver_logsnr(
        model, noise, schedule_fn, num_steps=200, class_labels=class_labels, **params
    )
    visualize_samples('DPM-Solver Method with Log-SNR Schedule', x_dpm_solver_logsnr)


def main():
    args = parse_args()
    with open('params.json') as f:
        params = json.load(f)
    with open_url('cond-vp.pkl') as f:
        data = pickle.load(f)
    model = data['ema'].to(params['device'])
    print(f"Модель имеет {sum(p.numel() for p in model.parameters())} параметров")
    
    batch_size = 8
    noise = torch.randn(batch_size, 3, 32, 32)

    class_labels = torch.randint(0, 10, (batch_size,))  # Random classes from 0 to 9
    class_labels = F.one_hot(class_labels, num_classes=10).float().to(params['device'])
    if args.solvqer_type == 'euler':
        sample_euler(params, model, noise, class_labels)
    elif args.solvqer_type == 'euler-logUniform':
        sample_euler(params, model, noise, class_labels, True)
    elif args.solvqer_type == 'ddim':
        sample_euler(params, model, noise, class_labels)
    elif args.solvqer_type == 'ddim-logSNR':
        sample_euler(params, model, noise, class_labels)
    else:
        sample_euler(params, model, noise, class_labels)


if __name__ == '__main__':
    main()
