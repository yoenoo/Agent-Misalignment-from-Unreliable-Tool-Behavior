import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors





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
        colors = plt.cm.tab10(range(len(conditions)))
    
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
    ax.set_title(title)
    ax.grid(axis="x",alpha=0.3)
    ax.invert_yaxis()  # highest on top
    
    plt.tight_layout()
    return fig, ax

def create_stacked_flag_bar_plot(data: Dict[str, List[str]], figsize=(12, 6), title="", min_percentage=5):
    """
    Create a stacked bar plot with percentages inside each bar segment.
    
    Parameters:
    -----------
    data : dict
        Dictionary where keys are experiment names and values are lists of flag strings
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 6)
    title : str, optional
        Plot title
    min_percentage : float, optional
        Minimum percentage to display label (default 5%). Set to 0 to show all labels.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Get all unique flag types
    all_flags = set()
    for flags in data.values():
        all_flags.update(flags)
    
    # Calculate total frequency of each flag across all experiments
    total_flag_counts = Counter()
    for flags in data.values():
        total_flag_counts.update(flags)
    
    # Sort flags by total frequency (descending) - most frequent first (will be at bottom)
    all_flags = [flag for flag, count in total_flag_counts.most_common()]
    
    # Count flags for each experiment
    experiment_names = list(data.keys())
    flag_counts = {}
    
    for flag in all_flags:
        flag_counts[flag] = []
        for exp_name in experiment_names:
            count = data[exp_name].count(flag)
            flag_counts[flag].append(count)
    
    # Calculate totals for each experiment
    experiment_totals = [len(data[exp_name]) for exp_name in experiment_names]
    
    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for different flag types
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_flags)))
    
    # Create stacked bars
    x_pos = np.arange(len(experiment_names))
    bottom = np.zeros(len(experiment_names))
    
    bars = []
    for i, flag in enumerate(all_flags):
        counts = flag_counts[flag]
        bar = ax.bar(x_pos, counts, bottom=bottom, label=flag, 
                     color=colors[i], edgecolor='black', linewidth=0.5)
        bars.append(bar)
        
        # Add percentage labels inside bars
        for j, (count, total) in enumerate(zip(counts, experiment_totals)):
            if count > 0:  # Only add label if there's a segment
                percentage = (count / total) * 100
                
                # Only show label if percentage is above minimum threshold
                if percentage >= min_percentage:
                    # Position text in the middle of the segment
                    y_pos = bottom[j] + count / 2
                    ax.text(j, y_pos, f'{percentage:.1f}%', 
                           ha='center', va='center', 
                           fontsize=9, fontweight='bold',
                           color='black')
        
        bottom += np.array(counts)
    
    # Customize the plot
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(experiment_names, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig, ax

def create_horizontal_stacked_flag_bar_plot(data: Dict[str, List[str]], figsize=None, title="", 
                                           min_percentage=8, font_size=8):
    """
    Create a HORIZONTAL stacked bar plot - better for many experiments.
    
    Parameters are the same as create_stacked_flag_bar_plot.
    """
    # Get all unique flag types
    all_flags = set()
    for flags in data.values():
        all_flags.update(flags)
    
    # Calculate total frequency of each flag across all experiments
    total_flag_counts = Counter()
    for flags in data.values():
        total_flag_counts.update(flags)
    
    # Sort flags by total frequency (descending)
    all_flags = [flag for flag, count in total_flag_counts.most_common()]
    
    # Count flags for each experiment
    experiment_names = list(data.keys())
    flag_counts = {}
    
    for flag in all_flags:
        flag_counts[flag] = []
        for exp_name in experiment_names:
            count = data[exp_name].count(flag)
            flag_counts[flag].append(count)
    
    # Calculate totals for each experiment
    experiment_totals = [len(data[exp_name]) for exp_name in experiment_names]
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        # Taller figure for more experiments
        height = max(6, len(experiment_names) * 0.4)
        figsize = (10, height)
    
    # Create the stacked bar plot (horizontal)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for different flag types
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_flags)))
    
    # Create stacked bars (horizontal)
    y_pos = np.arange(len(experiment_names))
    left = np.zeros(len(experiment_names))
    
    bars = []
    for i, flag in enumerate(all_flags):
        counts = flag_counts[flag]
        bar = ax.barh(y_pos, counts, left=left, label=flag, 
                      color=colors[i], edgecolor='black', linewidth=0.5)
        bars.append(bar)
        
        # Add percentage labels inside bars
        for j, (count, total) in enumerate(zip(counts, experiment_totals)):
            if count > 0:
                percentage = (count / total) * 100
                
                if percentage >= min_percentage:
                    x_pos_text = left[j] + count / 2
                    
                    # Choose text color based on background brightness
                    bg_color = colors[i]
                    brightness = np.mean(bg_color[:3])
                    text_color = 'white' if brightness < 0.6 else 'black'
                    
                    ax.text(x_pos_text, j, f'{percentage:.1f}%', 
                           ha='center', va='center', 
                           fontsize=font_size, fontweight='bold',
                           color=text_color)
        
        left += np.array(counts)
    
    # Customize the plot
    ax.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(experiment_names)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig, ax

def plot_side_by_side_comparison(results_dict, titles, xlabel="", figsize=(16, 6),
                                  colors=None, alpha=0.8, capsize=5, xlim=(0, 100),
                                  main_title=None):
    """
    Create side-by-side bar plots for comparing multiple models.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and their results dictionaries as values
        Example: {'GPT-5-mini': {condition: [values]}, 'QWEN-3': {condition: [values]}}
    titles : list of str
        Subplot titles (e.g., ['(A) GPT-5-mini', '(B) QWEN-3'])
    xlabel : str
        Label for x-axis (shared across subplots)
    figsize : tuple
        Figure size (width, height)
    colors : list or None
        List of colors for bars. If None, uses default color scheme
    alpha : float
        Transparency of bars
    capsize : int
        Size of error bar caps
    xlim : tuple
        X-axis limits (min, max)
    main_title : str or None
        Overall figure title
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=False)
    
    # Handle single model case (axes is not a list)
    if n_models == 1:
        axes = [axes]
    
    # Get all unique conditions across all models for consistent coloring
    all_conditions = []
    for results in results_dict.values():
        all_conditions.extend(results.keys())
    unique_conditions = list(dict.fromkeys(all_conditions))  # Preserve order
    
    # Create color mapping if colors not provided
    if colors is None:
        colors_map = {cond: plt.cm.tab10(i % 10) for i, cond in enumerate(unique_conditions)}
    else:
        colors_map = {cond: colors[i % len(colors)] for i, cond in enumerate(unique_conditions)}
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        conditions = list(results.keys())
        means = []
        errors = []
        bar_colors = []
        
        for cond, values in results.items():
            if len(values) == 1:
                means.append(values[0])
                errors.append(0)
            else:
                mean = np.mean(values)
                ci = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))  # 95% CI
                means.append(mean)
                errors.append(ci)
            bar_colors.append(colors_map[cond])
        
        y = np.arange(len(conditions))
        ax.barh(y, means, color=bar_colors, alpha=alpha, xerr=errors, capsize=capsize)
        
        ax.set_yticks(y)
        ax.set_yticklabels(conditions)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xlabel(xlabel)
        ax.set_title(titles[idx], fontweight='bold', fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()  # highest on top
        
        # Only show y-labels on leftmost plot
        if idx > 0:
            ax.set_ylabel('')
    
    # Add main title if provided
    if main_title:
        fig.suptitle(main_title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig, axes