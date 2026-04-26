import matplotlib.pyplot as plt
import numpy as np

from math import ceil
from typing import Callable, List
from itertools import product
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def plot_confusion_matrices(matrices: np.ndarray, labels: List[str], layout: tuple[int, int] | None = None, titling: Callable | None = None) -> None:
    label_indexing = list(range(len(labels)))

    if titling is None:
        titling = lambda i: f"Confusion Matrix {i}"

    if layout is None:
        cols = min(3, len(matrices))
        layout = (ceil(len(matrices) / cols), cols)

    max_value = np.max(matrices)

    fig, axes = plt.subplots(layout[0], layout[1], figsize=(5 * layout[1], 5 * layout[0]))
    axes = np.array(axes).flatten()

    for i, mat in enumerate(matrices):
        ax = axes[i]
    
        im = ax.imshow(mat.astype(int), cmap='Purples', vmin=0, vmax=max_value)

        ax.set_xticks(label_indexing)
        ax.set_yticks(label_indexing)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        ax.set_xlabel('Actual', fontsize=14)
        ax.set_ylabel('Predicted', fontsize=14)
        ax.set_title(titling(i), fontsize=16)

        for c, r in product(label_indexing, label_indexing):
            ax.text(c, r, int(mat[r, c]), ha="center", va="center", color="black", fontsize=12)

    for j in range(len(matrices), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
