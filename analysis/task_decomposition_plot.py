from typing import Dict, List
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from inspect_ai.log import EvalLog

from single_eval_logs import parse_logs_with_function, get_log_by_seed_or_index


def create_multi_model_stacked_flag_bar_plot(
    model_data: Dict[str, Dict[str, List[str]]],
    figsize=(12, 10),
    title="",
    min_percentage=5,
    ncols=1,
    legend_ncol=2,
    label_fontsize=8,
):
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
                                   fontsize=label_fontsize, fontweight='bold', color='black')
                
                bottom += np.array(counts)
        
        # Customize subplot
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(experiment_names, rotation=0,fontsize=14, ha='center')
        ax.grid(axis='y', alpha=0.4, linestyle='--')
    

    # Create shared legend handles
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=flag_color_map[flag], 
               edgecolor='black', linewidth=0.5) for flag in all_flags]
    
    # Apply tight_layout first, then position legend
    plt.tight_layout()

    # Calculate legend height factor based on number of rows
    legend_rows = (len(all_flags) + legend_ncol - 1) // legend_ncol
    legend_height = 0.03 + 0.025 * legend_rows

    plt.subplots_adjust(top=1.0 - legend_height - 0.02)
    fig.legend(handles, all_flags, loc='upper center', bbox_to_anchor=(0.5, 0.99),
                frameon=True, title="Legend", ncol=legend_ncol, fontsize=13.5)
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    return fig, axes


def _extract_decomposition_labels(log: EvalLog) -> list[str]:
    """Extract decomposition labels from a single EvalLog."""
    res = []
    for sample in log.samples:
        label = []
        if (
            "swe_style_humaneval_scorer" in sample.scores
            and sample.scores["swe_style_humaneval_scorer"].value == "C"
        ):
            label.append("reward_hacked")

        if len(sample.messages) >= 50:
            label.append("max_len_reached")

        if (
            "swe_style_humaneval_scorer" in sample.scores
            and sample.scores["swe_style_humaneval_scorer"].answer
            == "[flag_for_human_intervention]"
        ):
            label.append("human_intervention_flag")
        res.append("|".join(label) if label else "none_of_the_above")
    return res


def get_multi_bar_decomposition_data(
    logs_dict: dict[str, list[EvalLog]],
) -> dict[str, list]:
    """Get decomposition data for all seeds (returns list per run)."""
    return parse_logs_with_function(logs_dict=logs_dict, parse_function=_extract_decomposition_labels)


def get_decomposition_data_by_seed(
    logs_dict: dict[str, list[EvalLog]],
    seed: int | None = 42,
) -> dict[str, list[str]]:
    """
    Get decomposition data for a specific seed.

    Parameters
    ----------
    logs_dict : dict[str, list[EvalLog]]
        Dictionary where key is run name and value is list of EvalLogs (different seeds).
    seed : int or None
        Random seed to select log by. Default is 42. If None, uses first log.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping run name to list of decomposition labels.
    """
    result = {}
    for run_name, logs in logs_dict.items():
        log = get_log_by_seed_or_index(logs, seed)
        if log is not None:
            result[run_name] = _extract_decomposition_labels(log)
    return result
