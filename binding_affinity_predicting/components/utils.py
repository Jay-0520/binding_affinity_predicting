import copy
import logging
import os
import pathlib
import pickle
import subprocess
from pathlib import Path

import BioSimSpace.Sandpit.Exscientia as BSS

logger = logging.getLogger(__name__)


def check_has_wat_and_box(system: BSS._SireWrappers._system.System) -> None:  # type: ignore
    """Check that the system has water and a box."""
    if system.getBox() == (None, None):
        raise ValueError("System does not have a box.")
    if system.nWaterMolecules() == 0:
        raise ValueError("System does not have water.")


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
    pass


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
