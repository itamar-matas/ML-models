import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.ticker import FuncFormatter
from matplotlib.axes import Axes

from .styles import Theme

from itertools import combinations
from typing import Callable, List
from numpy.typing import ArrayLike
from math import ceil

def plot_feature_relationships(data: ArrayLike, target: ArrayLike | str, feature_labels: List[str] | None = None, n_columns: int = 3, inverse_transform: Callable | None = None) -> None:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        cols = feature_labels if feature_labels is not None else [f'feature {i}' for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=cols)

    if isinstance(target, str):
        target_series = df[target].copy()
        df = df.drop(columns=[target])
    elif isinstance(target, pd.DataFrame):
        target_series = target.iloc[:, 0].copy()
    elif isinstance(target, pd.Series):
        target_series = target.copy()
    else:
        target_series = pd.Series(target, index=df.index, name='target')
        
    if inverse_transform is not None:
        target_series = pd.Series(inverse_transform(target_series.values), index=target_series.index, name=target_series.name)

    plot_df = pd.concat([df, target_series], axis=1)
    pairs = list(combinations(df.columns, 2))
    n_categories = target_series.nunique()
    
    if not pairs:
        return
    
    n_columns = min(n_columns, len(pairs))
    layout = (ceil(len(pairs) / n_columns), n_columns)
    
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(5 * layout[1], 4 * layout[0]), squeeze=False)
    axes_flat = axes.flatten()

    for i, (feat1, feat2) in enumerate(pairs):
        ax = axes_flat[i]

        sns.scatterplot(
            data=plot_df, x=feat1, y=feat2, hue=target_series.name, ax=ax, 
            palette=Theme.categorical(n_categories),
            alpha=0.8, edgecolor='k'
        )
        ax.set_title(f'{feat1} vs {feat2}')

        if ax.get_legend():
            ax.get_legend().remove()

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title=target_series.name, bbox_to_anchor=(1.1, 1), borderaxespad=0.5)

    for j in range(len(pairs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_class_distribution(target: ArrayLike, figsize=(6, 4), inverse_transform: Callable | None = None, chart: str = 'bars') -> None:
    series = pd.Series(target)

    counts  = series.value_counts().head(10)
    classes = counts.index.map(inverse_transform) if inverse_transform is not None else counts.index

    top10 = pd.DataFrame({
        "class": classes,
        "top": [i+1 for i in range(classes.size)],
        "count": counts.values,
        "percentage": 100 / series.shape[0] * counts.values
    })

    fig = plt.figure(figsize=figsize, constrained_layout=True)

    if   chart == 'bars':      _top10count(top10, series.size)
    elif chart == 'pie':       _top10pie(top10, fig)
    elif chart == 'cumulative': _top10cumdistribution(top10, series.size)

    else: raise ValueError("not included") 

    plt.suptitle('Class Distribution')
    plt.show()

def plot_missing_values(data: ArrayLike, feature_labels: List[str] | None = None) -> None:
    if isinstance(data, pd.DataFrame):
        df = data.copy()

    else:
        df = pd.DataFrame(data)
        df.columns = feature_labels if feature_labels is not None else [f'feature {i}' for i in range(df.shape[1])]

    missing_counts = df.isna().sum() / df.shape[0] * 100

    plt.figure(figsize=(8, 4))

    sns.barplot(x=missing_counts.index, y=missing_counts.values, color=Theme.error(), alpha=0.4)
    plt.xlabel("features")
    plt.ylabel("missing percentage (%)")
    plt.title('Missing Values')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def _top10pie(top10: pd.DataFrame, fig):
    df = top10.copy()
    top10_percentage = df["percentage"].sum()

    if top10_percentage < 99.99:
        df.loc[df.shape[0]] = pd.Series({
            "class": "Other",
            "percentage": 100 - top10_percentage
        })

    ax_pie, ax_leg = fig.subplots(1, 2, gridspec_kw={"width_ratios": [5, 1]})

    wedges, _, _ = ax_pie.pie(df["percentage"], autopct=lambda x: f"{x:.1f}%" if x > 5 else "", colors=Theme.categorical_purples(df["percentage"] / 100))
    
    ax_leg.axis("off")
    ax_leg.legend(wedges, df["class"], loc="upper right")
    
def _top10cumdistribution(top10: pd.DataFrame, num_samples: int):
    df = top10.copy()
    df["cumpercentage"] = (df["count"] / num_samples * 100).cumsum()

    top10_percentage = df["percentage"].sum()

    if top10_percentage < 99.99:
        df.loc[df.shape[0]] = pd.Series({
            "top": df.shape[0] + 1,
            "percentage": 100 - top10_percentage,
            "cumpercentage": 100
        })

    sns.lineplot(x=df["top"], y=df["cumpercentage"], marker='o', markeredgewidth=0, color=Theme.main())
    
    plt.xticks(df["top"], [str(t) if t != 11 else "All" for t in df["top"].values.astype(int)])
    plt.fill_between(df["top"], df["cumpercentage"], color=Theme.main(), alpha=0.3)
    
    plt.xlabel("Top-10 Classes")
    plt.ylabel("Cumulative Percentage (%)")

    for i, (cp, c) in enumerate(zip(df["cumpercentage"], df["count"])):
        plt.annotate(
            f"{round(cp,1)}%",
            (i+1, cp),
            xytext=(0,8),
            textcoords="offset points",
            ha="center"
        )

def _top10count(top10: pd.DataFrame, num_samples: int):
    df = top10.copy()
    top10_percentage = df["percentage"].sum()

    if top10_percentage < 99.99:
        df.loc[df.shape[0]] = pd.Series({
            "class": "Other",
            "percentage": 100 - top10_percentage,
            "count": num_samples - df["count"].sum() 
        })

    sns.barplot(x=df["class"], y=df["count"], color=Theme.main(), alpha=0.7)
    plt.tick_params(axis='x', rotation=45)

    plt.margins(y=0.15)
    for i, (c, p) in enumerate(zip(df["count"], df["percentage"])):
        
        plt.annotate(
            c,
            (i, c),
            xytext=(0,20),
            textcoords="offset points",
            ha="center"
        )

        plt.annotate(
            f"({round(p,2)}%)",
            (i, c),
            xytext=(0,8),
            textcoords="offset points",
            ha="center",
            color="grey"
        )