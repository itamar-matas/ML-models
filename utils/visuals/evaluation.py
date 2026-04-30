import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .styles import color_maps

from math import ceil
from typing import Callable, List
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def plot_confusion_matrices(matrices: np.ndarray, labels: List[str], n_columns: int = 3, titling: Callable | None = None) -> None:
    if titling is None:
        titling = lambda i: f"Confusion Matrix {i}"

    n_columns = min(n_columns, len(matrices))
    layout = (ceil(len(matrices) / n_columns), n_columns)

    fig, axes = plt.subplots(layout[0], layout[1], figsize=(5 * layout[1], 5 * layout[0]))
    axes = np.array(axes).flatten()

    for i, mat in enumerate(matrices):
        ax = axes[i]
    
        sns.heatmap(mat.astype(int), annot=True, fmt='d', cmap=color_maps.purples(), xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)

        ax.set_xlabel('Actual', fontsize=14)
        ax.set_ylabel('Predicted', fontsize=14)
        ax.set_title(titling(i), fontsize=16)

    for j in range(len(matrices), len(axes)):
        axes[j].set_visible(False)

    plt.show()
