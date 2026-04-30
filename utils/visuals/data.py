import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .styles import color_maps

from itertools import combinations
from typing import Callable, List
from math import ceil

def plot_feature_relationships(data: pd.DataFrame | np.ndarray, target: pd.Series | np.ndarray | str = None, features: List[str] | None = None, n_columns: int = 3, inverse_transform: Callable | None = None) -> None:
    if isinstance(data, np.ndarray):
        cols = features if features is not None else [f'feature {i}' for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=cols)

    else:
        df = data.copy()

    if isinstance(target, str):
        target_series = df[target].copy()
        df = df.drop(columns=[target])

    elif isinstance(target, (np.ndarray, pd.Series)):
        target_series = pd.Series(target).copy()
        target_series.name = 'target'

    elif target is None:
        target_series = pd.Series([0] * len(df), name='target')

    else:
        target_series = target.copy()

    if inverse_transform is not None:
        target_series = pd.Series(inverse_transform(target_series.values), index=target_series.index, name=target_series.name)

    plot_df = pd.concat([df, target_series], axis=1)
    pairs = list(combinations(df.columns, 2))
    n_categories = target_series.nunique()
    
    if not pairs:
        return
    
    n_columns = min(n_columns, len(pairs))
    layout = (ceil(len(pairs) / n_columns), n_columns)
    
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(5 * layout[1], 4 * layout[0]))
    axes_flat = np.array(axes).flatten()

    for i, (feat1, feat2) in enumerate(pairs):
        ax = axes_flat[i]

        sns.scatterplot(data=plot_df, x=feat1, y=feat2, hue=target_series.name, ax=ax, palette=color_maps.categorical(n_categories), alpha=0.8, edgecolor='k')
        ax.set_title(f'{feat1} vs {feat2}')

        if ax.get_legend():
            ax.get_legend().remove()

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title=target_series.name, bbox_to_anchor=(1.1, 1), borderaxespad=0.5)

    for j in range(len(pairs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.show()