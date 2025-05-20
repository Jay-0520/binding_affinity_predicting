import logging
import os
import pickle
from pathlib import Path
from typing import Any

import BioSimSpace.Sandpit.Exscientia as BSS

from binding_affinity_predicting.data.schemas import WorkflowConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_workflow_config(cfg: WorkflowConfig, filepath: str) -> None:
    """
    Serialize a WorkflowConfig out to a pickle file.

    Parameters
    ----------
    cfg : WorkflowConfig
        The config object to save.
    filepath : str
        Path to the .pkl file to write. Parent dirs will be created if needed.
    """
    # Always writes a plain dict so to avoid subtle pickling issues
    # not pickling the pydantic modelclass directly
    data: dict[str, Any] = cfg.model_dump()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_workflow_config(filepath: str) -> WorkflowConfig:
    """
    Load a WorkflowConfig back from a pickle file.

    Parameters
    ----------
    filepath : str
        Path to the .pkl file created by save_workflow_config.

    Returns
    -------
    WorkflowConfig
        The re-hydrated config object.
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    # Re-validate and reconstruct the Pydantic model
    return WorkflowConfig.model_validate(data)


def check_has_wat_and_box(system: BSS._SireWrappers._system.System) -> None:  # type: ignore
    """Check that the system has water and a box."""
    if system.getBox() == (None, None):
        raise ValueError("System does not have a box.")
    if system.nWaterMolecules() == 0:
        raise ValueError("System does not have water.")


def ensure_dir_exist(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        logger.info(f"directory {path} does not exist")
        path.mkdir(parents=True, exist_ok=True)


def dump_simulation_state(obj: object, base_dir: Path) -> None:
    with open(base_dir / f"{obj.__class__.__name__}.pkl", "wb") as fp:
        pickle.dump(obj.__dict__, fp)


def load_simulation_state(obj: object, base_dir: Path) -> None:
    p = base_dir / f"{obj.__class__.__name__}.pkl"
    with open(p, "rb") as fp:
        obj.__dict__.update(pickle.load(fp))
