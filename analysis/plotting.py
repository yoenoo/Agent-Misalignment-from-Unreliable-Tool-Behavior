from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

def plot_cumulative_message_histograms(
    message_counts_by_name: Mapping[str, Sequence[int]],
    bins: int | str | Sequence[int] = "auto",
    show: bool = True,
):
    # Concatenate all data to get shared bin edges
    all_counts = np.concatenate(
        [np.asarray(v, dtype=int) for v in message_counts_by_name.values()]
    )

    # Get shared histogram bin edges from all data
    _, bin_edges = np.histogram(all_counts, bins=bins, density=True)
    bin_widths = np.diff(bin_edges)

    cdfs: dict[str, np.ndarray] = {}

    fig, ax = plt.subplots()

    for name, counts in message_counts_by_name.items():
        counts = np.asarray(counts, dtype=int)
        hist, _ = np.histogram(counts, bins=bin_edges, density=True)
        cdf = np.cumsum(hist * bin_widths)
        cdfs[name] = cdf

        ax.step(bin_edges[1:], cdf, where="post", label=name)

    ax.set_xlabel("Total chat messages per eval sample")
    ax.set_ylabel("Cumulative probability  P(N_messages ≤ x)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Eval run")

    return fig, ax



def visualize_logs_per_task_binary(logs):
    data_matrix = []
    row_labels = []

    # Flatten the dictionary into rows and labels (e.g., baseline_1)
    for run_name, seeds in logs.items():
        for i, results in enumerate(seeds):
            data_matrix.append(results)
            row_labels.append(f"{run_name}_{i+1}")

    data_array = np.array(data_matrix)
    nrows, ncols = data_array.shape

    # Set up the plot (width based on tasks, height based on rows)
    fig, ax = plt.subplots(figsize=(ncols * 1.2, nrows * 0.8))

    # Define a discrete colormap: 0 -> Red (Fail), 1 -> Green (Success)
    cmap = mcolors.ListedColormap(['#ff9999', '#99ff99'])
    bounds = [0, 0.5, 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the matrix
    im = ax.imshow(data_array, cmap=cmap, norm=norm, aspect='equal')

    # Set axis labels
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels([f"Task {i+1}" for i in range(ncols)])

    # Add grid lines by setting minor ticks between the squares
    ax.set_xticks(np.arange(ncols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(nrows + 1) - 0.5, minor=True)
    
    # Draw the white grid lines to separate the squares
    ax.grid(which="minor", color="white", linestyle='-', linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Clean up the visual by removing the frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.title("Run Results Grid", pad=15)
    plt.tight_layout()
    plt.savefig('run_results_grid.png')


def plot_bars_with_ci(results, title="", xlabel="", ylabel="", figsize=(10, 6), 
                      colors=None, alpha=0.8, capsize=5, label_offset=2, xlim=(0,100)):

    fig, ax = plt.subplots(figsize=figsize)
    
    conditions = list(results.keys())
    means = []
    errors = []
    
    for values in results.values():
        if len(values) == 1:
            means.append(values[0])
            errors.append(0)
        else:
            mean = np.mean(values)
            ci = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))  # 95% CI
            means.append(mean)
            errors.append(ci)
    
    # Default colors if none provided
    if colors is None:
        n = len(conditions)
        if n <= 10:
            colors = plt.cm.tab10(range(n))
        else:
            # First 10 from tab10, rest from a continuous colormap
            colors = [plt.cm.tab10(i) for i in range(10)] + \
                    list(plt.cm.turbo(np.linspace(0, 1, n - 10)))        
    
    y = np.arange(len(conditions))
    bars = ax.barh(y, means, color=colors, alpha=alpha, xerr=errors, capsize=capsize)

    ax.set_yticks(y)
    ax.set_yticklabels(conditions)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title != "":
        ax.set_title(title)
    ax.grid(axis="x",alpha=0.3)
    ax.invert_yaxis()  # highest on top
    
    plt.tight_layout()
    return fig, ax


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    figsize=(10, 8),
    title="Pairwise Correlation of Reward Hacking Scores",
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    vmin=-1,
    vmax=1,
    mask_upper=True,
):
    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1) if mask_upper else None

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig, ax

def plot_per_sample_scores(
    scores_dict: dict[str, list[int]],
    figsize=None,
    title="Per-Sample Reward Hacking Scores",
    cmap_colors=("#ff9999", "#99ff99"),
    show_task_labels=False,
):
    """
    Visualize per-sample reward hacking scores as a binary heatmap grid.

    Parameters
    ----------
    scores_dict : dict[str, list[int]]
        Dictionary from get_per_sample_reward_hacking_scores.
        Keys are run names, values are lists of binary scores (0/1).
    figsize : tuple or None
        Figure size. If None, auto-calculated based on data.
    title : str
        Plot title.
    cmap_colors : tuple
        Colors for (0=fail, 1=success). Default is (red, green).
    show_task_labels : bool
        Whether to show task numbers on x-axis.

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    run_names = list(scores_dict.keys())
    n_runs = len(run_names)
    n_samples = max(len(v) for v in scores_dict.values())

    # Pad shorter lists with -1 (will be shown as white/neutral)
    data_matrix = []
    for run_name in run_names:
        scores = scores_dict[run_name]
        padded = scores + [-1] * (n_samples - len(scores))
        data_matrix.append(padded)

    data_array = np.array(data_matrix)

    # Sort columns by number of 1's (reward hacking) across all runs (descending)
    # Count 1's in each column (treating -1 as 0 for counting purposes)
    ones_count = np.sum(data_array == 1, axis=0)
    sorted_indices = np.argsort(-ones_count)  # Descending order
    data_array = data_array[:, sorted_indices]

    # Auto-calculate figure size if not provided
    if figsize is None:
        width = max(12, n_samples * 0.15)
        height = max(4, n_runs * 0.5)
        figsize = (width, height)

    fig, ax = plt.subplots(figsize=figsize)

    # Create colormap: -1 -> white, 0 -> red, 1 -> green
    cmap = mcolors.ListedColormap(["white", cmap_colors[0], cmap_colors[1]])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(data_array, cmap=cmap, norm=norm, aspect="auto")

    # Set y-axis labels (run names)
    ax.set_yticks(np.arange(n_runs))
    ax.set_yticklabels(run_names)

    # Set x-axis labels (show original sample indices after sorting)
    ax.set_xticks(np.arange(n_samples))
    ax.set_xticklabels([f"{idx}" for idx in sorted_indices], fontsize=5, rotation=90)

    ax.set_xlabel(f"Samples (n={n_samples})")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # Add grid lines
    ax.set_xticks(np.arange(n_samples + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_runs + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Remove frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig, ax


