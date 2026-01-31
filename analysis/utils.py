from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def load_eval_dirs_parallel(
    base_dir: str | Path,
    exclude_model: list[str] | None = None,
    header_only: bool = False,
    max_workers: int = 8,
) -> dict[str, dict[str, list[EvalLog]]]:
    """Load eval logs from directory structure (parallel version)."""
    base_dir = Path(base_dir)
    res = {}

    dirs_to_process = [
        dp
        for dp in base_dir.iterdir()
        if dp.is_dir()
        and not dp.name.startswith("_")
        and not (exclude_model and any(em in dp.name for em in exclude_model))
    ]

    def load_model_dir(dir_path: Path) -> tuple[str, dict[str, list[EvalLog]]]:
        model_name = dir_path.name.split("/")[-1]
        per_model_dirs = {
            dp.name.replace(model_name, ""): load_eval_logs_from_dir(dp)
            for dp in dir_path.iterdir()
            if dp.is_dir() and not dp.name.startswith("_")
        }
        return model_name, per_model_dirs

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_model_dir, d): d for d in dirs_to_process}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading models..."):
            model_name, data = future.result()
            res[model_name] = data

    return res
