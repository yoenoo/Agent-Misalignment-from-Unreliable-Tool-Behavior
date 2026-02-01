from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_cumulative_message_histograms(
    message_counts_by_name: Mapping[str, Sequence[int]],
    bins: int | str | Sequence[int] = "auto",
    show: bool = True,
):
    """
    Plot cumulative probability histograms (CDFs) of total chat messages per sample
    for multiple eval runs on a single plot.

    Parameters
    ----------
    message_counts_by_name
        Mapping from a name/label to a sequence of integers.

        Each key is a label that will appear in the legend.
        Each value is a sequence of ints giving the total number of chat messages
        used for each eval sample in that run.

        Example:
            message_counts_by_name = {
                "gpt-5-mini": [3, 4, 4, 10, 2],
                "gpt-4.1":    [5, 6, 2, 8, 3, 3],
            }

    bins
        Binning strategy passed to numpy.histogram for a *shared* binning.
        - int: number of bins
        - "auto": let numpy choose from all concatenated data
        - sequence: explicit bin edges

    show
        If True, calls plt.show() at the end.

    Returns
    -------
    bin_edges : np.ndarray
        The edges of the bins (shared across all series).
    cdfs : dict[str, np.ndarray]
        Mapping from name to CDF array for that series
        (same length as bin_edges[1:]).
    """
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

    if show:
        plt.show()

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
    
    # # Add value labels on bars - position them to the right of error bars
    # for i, (bar, mean, error) in enumerate(zip(bars, means, errors)):
    #     label_x = mean + error + label_offset  # Place after error bar
    #     ax.text(label_x, i, f'{mean:.0f}%', va='center', fontsize=10, fontweight='bold')
    
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


# def plot_side_by_side_comparison(results_dict, titles, xlabel="", figsize=(16, 6),
#                                   colors=None, alpha=0.8, capsize=5, xlim=(0, 100),
#                                   main_title=None):
#     """
#     Create side-by-side bar plots for comparing multiple models.
    
#     Parameters:
#     -----------
#     results_dict : dict
#         Dictionary with model names as keys and their results dictionaries as values
#         Example: {'GPT-5-mini': {condition: [values]}, 'QWEN-3': {condition: [values]}}
#     titles : list of str
#         Subplot titles (e.g., ['(A) GPT-5-mini', '(B) QWEN-3'])
#     xlabel : str
#         Label for x-axis (shared across subplots)
#     figsize : tuple
#         Figure size (width, height)
#     colors : list or None
#         List of colors for bars. If None, uses default color scheme
#     alpha : float
#         Transparency of bars
#     capsize : int
#         Size of error bar caps
#     xlim : tuple
#         X-axis limits (min, max)
#     main_title : str or None
#         Overall figure title
    
#     Returns:
#     --------
#     fig, axes : matplotlib figure and axes objects
#     """
#     n_models = len(results_dict)
#     fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=False)
    
#     # Handle single model case (axes is not a list)
#     if n_models == 1:
#         axes = [axes]
    
#     # Get all unique conditions across all models for consistent coloring
#     all_conditions = []
#     for results in results_dict.values():
#         all_conditions.extend(results.keys())
#     unique_conditions = list(dict.fromkeys(all_conditions))  # Preserve order
    
#     # Create color mapping if colors not provided
#     if colors is None:
#         colors_map = {cond: plt.cm.tab10(i % 10) for i, cond in enumerate(unique_conditions)}
#     else:
#         colors_map = {cond: colors[i % len(colors)] for i, cond in enumerate(unique_conditions)}
    
#     for idx, (model_name, results) in enumerate(results_dict.items()):
#         ax = axes[idx]
        
#         conditions = list(results.keys())
#         means = []
#         errors = []
#         bar_colors = []
        
#         for cond, values in results.items():
#             if len(values) == 1:
#                 means.append(values[0])
#                 errors.append(0)
#             else:
#                 mean = np.mean(values)
#                 ci = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))  # 95% CI
#                 means.append(mean)
#                 errors.append(ci)
#             bar_colors.append(colors_map[cond])
        
#         y = np.arange(len(conditions))
#         ax.barh(y, means, color=bar_colors, alpha=alpha, xerr=errors, capsize=capsize)
        
#         ax.set_yticks(y)
#         ax.set_yticklabels(conditions)
#         ax.set_xlim(xlim[0], xlim[1])
#         ax.set_xlabel(xlabel)
#         ax.set_title(titles[idx], fontweight='bold', fontsize=12)
#         ax.grid(axis="x", alpha=0.3)
#         ax.invert_yaxis()  # highest on top
        
#         # Only show y-labels on leftmost plot
#         if idx > 0:
#             ax.set_ylabel('')
    
#     # Add main title if provided
#     if main_title:
#         fig.suptitle(main_title, fontsize=14, fontweight='bold', y=1.02)
    
#     plt.tight_layout()
#     return fig, axes


