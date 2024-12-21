from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt


def remove_ticks(ax):
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        left=False,
        labelleft=False
    )


def remove_xticks(ax):
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        left=True,
        labelleft=True
    )


def visualize_batch(img_vis, title='Семплы из CIFAR10', nrow=10, ncol=4):
    img_grid = make_grid(img_vis, nrow=nrow)
    fig, ax = plt.subplots(1, figsize=(nrow, ncol))
    remove_ticks(ax)
    ax.set_title(title, fontsize=14)
    ax.imshow(img_grid.permute(1, 2, 0))
    plt.show()


def visualize_samples(method_name, samples, nrow=8):
    samples = samples * 0.5 + 0.5  # Rescale to [0, 1]
    grid = make_grid(samples, nrow=nrow)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(f'Samples using {method_name}')
    plt.axis('off')
    plt.show()