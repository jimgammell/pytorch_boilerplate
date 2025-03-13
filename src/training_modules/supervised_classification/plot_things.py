from typing import Tuple
from numpy.typing import NDArray
from matplotlib import pyplot as plt

from common import *

def plot_training_curves(training_curves: Dict[str, Tuple[NDArray, NDArray]], save_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(3*PLOT_WIDTH, 1*PLOT_WIDTH))
    axes[0].plot(*training_curves['train_loss'], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_loss_epoch'], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Loss')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[1].plot(*training_curves['train_acc'], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
    axes[1].plot(*training_curves['val_acc'], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[2].plot(*training_curves['train_rank'], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
    axes[2].plot(*training_curves['val_rank'], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
    axes[2].set_xlabel('Training step')
    axes[2].set_ylabel('Rank')
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'training_curves.png'), **SAVEFIG_KWARGS)