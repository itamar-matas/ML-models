import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Callable
from itertools import combinations

def plot_feature_relationships(data: pd.DataFrame | np.ndarray, target: pd.Series | np.ndarray | str = None, features: list[str] | None = None, layout: tuple[int, int] = None, inverse_transform: Callable | None = None) -> None:

    if isinstance(data, pd.DataFrame):
        features = list(data.columns)

    elif isinstance(data, np.ndarray) and features is not None:
        data = pd.DataFrame(data, columns=features)

    else:
        data = pd.DataFrame(data, columns=[f'feature {i}' for i in range(data.shape[1])])
        
    if isinstance(target, str):
        target_values = data[target]
        data = data.drop(columns=[target])
        target = target_values


    labels = np.unique(target)
    if inverse_transform is not None:
        labels = inverse_transform(labels)
    
    pairs = list(combinations(data.columns, 2))
    if layout is None:
        layout = (data.columns.size // 2, data.columns.size + data.columns.size % 2 - 1)
    
    cmap = plt.get_cmap('viridis', labels.size)
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(18, 10))
    axes_flat = axes.flatten()


    for i, (feat1, feat2) in enumerate(pairs):
        ax = axes_flat[i]
    
        ax.scatter(data[feat1], data[feat2], c=target, cmap=cmap, edgecolors='k', alpha=0.8)

        ax.set_xlabel(feat1, c="grey")
        ax.set_ylabel(feat2, c="grey")
        ax.set_title(f'{feat1} vs {feat2}')

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=labels[i]) for i in range(labels.size)]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.show()
