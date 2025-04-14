from typing import Tuple
from numpy.typing import NDArray
from matplotlib import pyplot as plt

from common import *

def plot_training_curves(training_curves: Dict[str, Tuple[NDArray, NDArray]], save_dir: str):
    print([(key, v1.shape, v2.shape) for key, (v1, v2) in training_curves.items()])
    fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, 1*PLOT_WIDTH))
    axes[0].plot(*training_curves['train_loss'], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_loss_epoch'], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_loss_baseline_epoch'], color='red', linestyle='-', label='val-baseline', **PLOT_KWARGS)
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Loss')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[1].plot(*training_curves['train_acc_step'], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
    axes[1].plot(*training_curves['val_acc_epoch'], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
    axes[1].plot(*training_curves['val_acc_baseline_epoch'], color='red', linestyle='-', label='val-baseline', **PLOT_KWARGS)
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'training_curves.png'), **SAVEFIG_KWARGS)
    plt.close(fig)