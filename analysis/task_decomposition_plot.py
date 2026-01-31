from typing import Dict, List
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def create_multi_model_stacked_flag_bar_plot(
    model_data: Dict[str, Dict[str, List[str]]], 
    figsize=(12, 10), 
    title="", 
    min_percentage=5,
    ncols=1,
    legend_ncol=2,
    legend_loc='bottom'
):
    """
    Create stacked bar plots for multiple models with shared legend and colors.
    
    Parameters:
    -----------
    model_data : dict
        Dictionary where keys are model names and values are dictionaries
        mapping experiment names to lists of flag strings.
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 10)
    title : str, optional
        Main plot title
    min_percentage : float, optional
        Minimum percentage to display label (default 5%)
    ncols : int, optional
        Number of columns in subplot grid. Default is 1 (vertical stack).
    legend_ncol : int, optional
        Number of columns in the legend. Default is 2.
    legend_loc : str, optional
        Legend location: 'bottom', 'top', or 'right'. Default is 'bottom'.
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    model_names = list(model_data.keys())
    n_models = len(model_names)
    
    nrows = (n_models + ncols - 1) // ncols
    
    # Collect all unique flags across ALL models and experiments
    all_flags = set()
    for model_name, data in model_data.items():
        for flags in data.values():
            all_flags.update(flags)
    
    # Calculate total frequency of each flag across all models and experiments
    total_flag_counts = Counter()
    for model_name, data in model_data.items():
        for flags in data.values():
            total_flag_counts.update(flags)
    
    # Sort flags by total frequency (descending)
    all_flags = [flag for flag, count in total_flag_counts.most_common()]
    
    # Create consistent color mapping for all flags
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_flags)))
    flag_color_map = {flag: colors[i] for i, flag in enumerate(all_flags)}
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    # Plot each model
    for model_idx, model_name in enumerate(model_names):
        ax = axes[model_idx]
        data = model_data[model_name]
        
        experiment_names = list(data.keys())
        experiment_totals = [len(data[exp_name]) for exp_name in experiment_names]
        
        # Count flags for each experiment
        flag_counts = {}
        for flag in all_flags:
            flag_counts[flag] = [data[exp_name].count(flag) for exp_name in experiment_names]
        
        # Create stacked bars
        x_pos = np.arange(len(experiment_names))
        bottom = np.zeros(len(experiment_names))
        
        for flag in all_flags:
            counts = flag_counts[flag]
            if sum(counts) > 0:  # Only plot if flag exists in this model's data
                ax.bar(x_pos, counts, bottom=bottom, label=flag,
                       color=flag_color_map[flag], edgecolor='black', linewidth=0.5)
                
                # Add percentage labels
                for j, (count, total) in enumerate(zip(counts, experiment_totals)):
                    if count > 0 and total > 0:
                        percentage = (count / total) * 100
                        if percentage >= min_percentage:
                            y_pos = bottom[j] + count / 2
                            ax.text(j, y_pos, f'{percentage:.1f}%',
                                   ha='center', va='center',
                                   fontsize=8, fontweight='bold', color='black')
                
                bottom += np.array(counts)
        
        # Customize subplot
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(experiment_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Create shared legend handles
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=flag_color_map[flag], 
               edgecolor='black', linewidth=0.5) for flag in all_flags]
    
    # Position legend based on legend_loc parameter
    if legend_loc == 'bottom':
        fig.legend(handles, all_flags, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                   frameon=True, title="Flags", ncol=legend_ncol)
        bottom_margin = 0.08 + 0.02 * (len(all_flags) // legend_ncol + 1)
        plt.subplots_adjust(bottom=bottom_margin)
    elif legend_loc == 'top':
        fig.legend(handles, all_flags, loc='lower center', bbox_to_anchor=(0.5, 0.98),
                   frameon=True, title="Flags", ncol=legend_ncol)
        top_margin = 0.92 - 0.02 * (len(all_flags) // legend_ncol + 1)
        plt.subplots_adjust(top=top_margin)
    else:  # 'right'
        fig.legend(handles, all_flags, loc='center left', bbox_to_anchor=(1, 0.5),
                   frameon=True, title="Flags", ncol=1)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Adjust for legend after tight_layout
    if legend_loc == 'bottom':
        plt.subplots_adjust(bottom=0.12 + 0.025 * (len(all_flags) // legend_ncol + 1))
    elif legend_loc == 'top':
        plt.subplots_adjust(top=0.88 - 0.025 * (len(all_flags) // legend_ncol + 1))
    
    return fig, axes


model_data = {
    "Gemini 3 Flash": {
        "baseline_hi": ["max_len_reached"] * 62 + ["human_intervention_flag"] * 31 + ["max_len_reached|human_intervention_flag"] * 3 + ["reward_hacked|max_len_reached"] * 2 + ["reward_hacked"] * 5,
        "fr0.5_bash_python_hi": ["max_len_reached"] * 64 + ["human_intervention_flag"] * 31 + ["max_len_reached|human_intervention_flag"] * 3 + ["reward_hacked"] * 5,
        "fr0.9_bash_python_hi": ["max_len_reached"] * 70 + ["human_intervention_flag"] * 23 + ["max_len_reached|human_intervention_flag"] * 7 + ["reward_hacked"] * 2,
    },
    "GPT-4": {
        "baseline_hi": ["max_len_reached"] * 50 + ["human_intervention_flag"] * 40,
        "fr0.5_bash_python_hi": ["max_len_reached"] * 55 + ["human_intervention_flag"] * 35,
        "fr0.9_bash_python_hi": ["max_len_reached"] * 60 + ["human_intervention_flag"] * 30,
    },
    "Claude": {
        "baseline_hi": ["max_len_reached"] * 45 + ["human_intervention_flag"] * 45,
        "fr0.5_bash_python_hi": ["max_len_reached"] * 50 + ["human_intervention_flag"] * 40,
        "fr0.9_bash_python_hi": ["max_len_reached"] * 55 + ["human_intervention_flag"] * 35,
    },
}

fig, axes = create_multi_model_stacked_flag_bar_plot(
    model_data, 
    figsize=(10, 12),  # Taller for vertical stack
    title="Flag Distribution Across Models",
    ncols=1,  # Vertical stack
    legend_ncol=1,  # More columns for horizontal legend
    legend_loc='top'
)
plt.savefig("plot.pdf", bbox_inches='tight', format="pdf")