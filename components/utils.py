import copy
import logging
import os
import pathlib
import pickle
import subprocess
from pathlib import Path


logger = logging.getLogger(__name__)


def load_simulation_state(update_paths: bool = True) -> None:
    """Load the state of the simulation object from a pickle file, and do
    the same for any sub-simulations.

    Parameters
    ----------
    update_paths : bool, default=True
        If True, update the paths of the simulation object and any sub-simulation runners
        so that the base directory becomes the directory passed to the SimulationRunner,
        or the current working directory if no directory was passed.
    """
    # Note that we cannot recursively call _load on the sub-simulations
    # because this results in the creation of different virtual queues for the
    # stages and sub-lam-windows and simulations
    if not pathlib.Path(f"{self.base_dir}/{self.__class__.__name__}.pkl").is_file():
        raise FileNotFoundError(
            f"Could not find {self.__class__.__name__}.pkl in {self.base_dir}"
        )

    # Store previous value of base dir before it is potentially overwritten below
    supplied_base_dir = self.base_dir

    # Load the SimulationRunner, possibly overwriting directories
    print(
        f"Loading previous {self.__class__.__name__}. Any arguments will be overwritten..."
    )
    with open(f"{self.base_dir}/{self.__class__.__name__}.pkl", "rb") as file:
        self.__dict__ = _pkl.load(file)

    if update_paths:
        self.update_paths(old_sub_path=self.base_dir, new_sub_path=supplied_base_dir)

    # Refresh logging
    print("Setting up logging...")
    _refresh_logging()

    # Record that the object was loaded from a pickle file
    loaded_from_pickle = True


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
