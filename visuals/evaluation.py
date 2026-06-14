import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from .styles import Theme

from math import ceil
from typing import Callable, List
from numpy.typing import ArrayLike

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def plot_confusion_matrices(matrices: ArrayLike, labels: List[str], n_columns: int = 3, titling: Callable | None = None) -> None:
    if titling is None:
        titling = lambda i: f"Confusion Matrix {i}"

    n_columns = min(n_columns, len(matrices))
    layout = (ceil(len(matrices) / n_columns), n_columns)

    fig, axes = plt.subplots(layout[0], layout[1], figsize=(5 * layout[1], 5 * layout[0]), squeeze=False)
    axes = np.array(axes).flatten()

    for i, mat in enumerate(matrices):
        ax = axes[i]
    
        sns.heatmap(np.asarray(mat).astype(int), annot=True, fmt='d', cmap=Theme.purples(), xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)

        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('Actual', fontsize=14)
        ax.set_title(titling(i), fontsize=16)

    for j in range(len(matrices), len(axes)):
        axes[j].set_visible(False)

    plt.show()

def plot_confidence_distribution(Y_true: ArrayLike, Y_pred: ArrayLike, Y_proba: np.ndarray) -> None:
    true_labels = np.asarray(Y_true).ravel()
    pred_labels = np.asarray(Y_pred).ravel()
    
    confidences = np.max(Y_proba, axis=1)
    
    is_correct = (true_labels == pred_labels)
    evaluation_type = np.where(is_correct, 'Correct Prediction', 'Incorrect Prediction')

    plot_df = pd.DataFrame({
        'confidence': confidences,
        'result': evaluation_type
    })
    
    plt.figure(figsize=(9, 5))
    colors = Theme.right_wrong()
    
    sns.kdeplot(
        data=plot_df, 
        x='confidence', 
        hue='result', 
        fill=True,
        common_norm=True,
        palette={'Incorrect Prediction': colors[0], 'Correct Prediction': colors[1]},
        alpha=0.4,
        bw_adjust=0.75
    )
    

    plt.xlim(1 / Y_proba.shape[1], 1.0)
    plt.xlabel('Model Confidence (Max Probability)')
    plt.ylabel('Density')
    plt.title('Prediction Confidence Distribution (KDE): Correct vs. Incorrect')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_hyperparameter_tuning(cv_results: dict, param_name: str, title: str | None = None) -> None:
    tuning_df = pd.DataFrame({
        "parameter": cv_results[f"param_{param_name}"],
        "score":     cv_results["mean_test_score"],
        "std":       cv_results["std_test_score"]
    })

    plt.figure(figsize=(9, 5))

    sns.lineplot(data=tuning_df, x='parameter', y='score', color=Theme.optimization())
    plt.fill_between(tuning_df['parameter'], tuning_df["score"] - tuning_df["std"], tuning_df["score"] + tuning_df["std"], alpha=0.4, color=Theme.optimization())

    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(title or f"{param_name} tuning")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()