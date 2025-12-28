from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from inspect_ai.log import EvalLog, read_eval_log


def load_eval_logs_from_dir(dir_path: str | Path) -> list[EvalLog]:
    dir_path = Path(dir_path)
    eval_logs = []
    for path in dir_path.glob("*.eval"):
        eval_logs.append(read_eval_log(path))
    return eval_logs


def load_eval_dirs(base_dir: str, exclude_model:list[str] | None = None) -> dict[str, dict[str, list[EvalLog]]]:
    base_dir = Path(base_dir)
    res = {}
    for dir_path in base_dir.iterdir():
        if dir_path.is_dir() and not dir_path.name.startswith("_"):
            if exclude_model and any(em in dir_path.name for em in exclude_model):
                continue
            model_name = dir_path.name.split("/")[-1]
            per_model_dirs = {
                dp.name.replace(model_name, ""): load_eval_logs_from_dir(dp) 
                for dp in tqdm(dir_path.iterdir(), "Loading eval logs for model...") 
                if dp.is_dir() and not dp.name.startswith("_")
            } 
            res[model_name] = per_model_dirs
    return res
