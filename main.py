import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from lib.plot import visualize_samples
import pickle
from edm.dnnlib.util import open_url
import json
import lib.solvers.euler as euler



def parse_args():
    parser = argparse.ArgumentParser(description="Parser for solver type")
    parser.add_argument('--solver-type', type=str, 
        choices=['euler', 'euler-logUniform', 'ddim', 'ddim-logSNR', 'dpm-logSNR'], 
        required=True, 
        help="Specify the type of solver to use (e.g., euler, ddim)"
    )
    parser.add_argument('--download-dataset', action='store_true')
    parser.add_argument('--calculate-fid', action='store_true')
    return parser.parse_args()


def sample_euler(params, model, noise, class_labels, is_logUniform=False):
    num_steps = params['num_stemps']
    if is_logUniform:
        betas = euler.get_beta_schedule_logUniform(num_steps)
    else:
        betas = euler.get_beta_schedule(num_steps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    x_euler, _ = euler.sample_euler_vp(model, noise, betas, alphas_cumprod, class_labels=class_labels, **params)
    visualize_samples('Euler Method', x_euler)


def main():
    args = parse_args()
    transform = Compose([Resize((32, 32)), ToTensor()])
    data_train = CIFAR10(root='.', train=True, download=args.download_dataset, transform=transform)
    data_test = CIFAR10(root='.', train=False, download=False, transform=transform)
    train_dataloader = DataLoader(data_train, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(data_test, batch_size=64, shuffle=True)
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
    # elif args.solvqer_type == 'euler':
    #     sample_euler(params, model, noise, class_labels)
    # elif args.solvqer_type == 'euler-logUniform':
    #     sample_euler(params, model, noise, class_labels, True)



if __name__ == '__main__':
    main()