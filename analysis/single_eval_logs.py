import numpy as np
import pandas as pd
from scipy import stats
from inspect_ai.log import EvalLog


def find_log_by_seed(logs: list[EvalLog], seed: int = 42) -> EvalLog | None:
    """
    Find an EvalLog by its random seed value.

    Parameters
    ----------
    logs : list[EvalLog]
        List of EvalLogs (different seeds).
    seed : int
        Random seed to search for. Default is 42.

    Returns
    -------
    EvalLog or None
        The EvalLog with matching seed, or None if not found.
    """
    for log in logs:
        if log.plan.config.seed == seed:
            return log
    return None


def get_log_by_seed_or_index(
    logs: list[EvalLog], seed: int | None = 42
) -> EvalLog | None:
    """
    Get an EvalLog by seed value, falling back to first log if seed not found.

    Parameters
    ----------
    logs : list[EvalLog]
        List of EvalLogs (different seeds).
    seed : int or None
        Random seed to search for. If None, returns first log.

    Returns
    -------
    EvalLog or None
        The matching EvalLog.
    """
    if not logs:
        return None
    if seed is None:
        return logs[0]
    log = find_log_by_seed(logs, seed)
    return log if log is not None else logs[0]


def parse_logs_with_function(logs_dict: dict[str, list[EvalLog]], parse_function):
    return {
        name: [parse_function(log) for log in logs] for name, logs in logs_dict.items()
    }


def get_cheating_rate(log: EvalLog) -> float:
    return log.results.scores[0].metrics["accuracy"].value * 100

def get_msgs_len(log):
    return [len(sample.messages) for sample in log.samples]

def get_score_without_max_len(log: EvalLog) -> float:
    scores = []
    for sample in log.samples:
        msgs_len = len(sample.messages)
        if msgs_len < 50:
            if "swe_style_humaneval_scorer" in sample.scores:
                score = sample.scores["swe_style_humaneval_scorer"].value == "C"
            else:
                score = False
            scores.append(score)
    return sum(scores) / len(scores) * 100

def get_scores_log(log: EvalLog) -> list[int | None]:
    scores = []
    for sample in log.samples:
        if "swe_style_humaneval_scorer" in sample.scores:
            score = int(sample.scores["swe_style_humaneval_scorer"].value == "C")
        else:
            score = None
        scores.append(score)
    return scores


def get_per_sample_reward_hacking_scores(
    logs_dict: dict[str, list[EvalLog]],
    seed: int | None = 42,
) -> dict[str, list[int]]:
    """
    Extract per-sample reward hacking scores (binary 0/1) for each run.

    Parameters
    ----------
    logs_dict : dict[str, list[EvalLog]]
        Dictionary where key is run name and value is list of EvalLogs (different seeds).
    seed : int or None
        Random seed to select log by. Default is 42. If None, uses first log.

    Returns
    -------
    dict[str, list[int]]
        Dictionary mapping run name to list of binary scores (0/1) per sample.
    """
    result = {}
    for run_name, logs in logs_dict.items():
        log = get_log_by_seed_or_index(logs, seed)
        if log is not None:
            scores = get_scores_log(log)
            # Filter out None values and convert to int
            result[run_name] = [s for s in scores if s is not None]
    return result


def calculate_pairwise_correlation(
    logs_dict: dict[str, list[EvalLog]],
    seed: int | None = 42,
    method: str = "pearson",
) -> pd.DataFrame:
    scores_dict = get_per_sample_reward_hacking_scores(logs_dict, seed)

    # Create DataFrame with runs as columns
    df = pd.DataFrame(scores_dict)

    # Calculate correlation matrix
    return df.corr(method=method)
