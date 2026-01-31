"""Simple cache utilities using dill."""

from pathlib import Path
from typing import Any

import dill


def save(data: Any, path: str | Path) -> None:
    """Save data to a dill file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        dill.dump(data, f)


def load(path: str | Path) -> Any:
    """Load data from a dill file."""
    with open(path, "rb") as f:
        return dill.load(f)