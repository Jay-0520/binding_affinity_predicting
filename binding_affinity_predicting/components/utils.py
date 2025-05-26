import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import BioSimSpace.Sandpit.Exscientia as BSS

from binding_affinity_predicting.data.schemas import BaseWorkflowConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_workflow_config(cfg: BaseWorkflowConfig, filepath: str) -> None:
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


def load_workflow_config(filepath: str) -> BaseWorkflowConfig:
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
    return BaseWorkflowConfig.model_validate(data)


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


def move_link_or_copy_files(
    src_dir: Union[str, Path],
    dest_dirs: Sequence[Union[str, Path]],
    filenames: Optional[Sequence[str]] = None,
    move: bool = False,
    symlink: bool = False,
) -> None:
    """
    Create each directory in `dest_dirs` (if it doesn't exist), and copy or move files
    from `src_dir` into each one.

    Parameters
    ----------
    src_dir:
        Path to the existing folder containing your files.
    dest_dirs:
        Iterable of paths (str or Path) to create and populate.
    filenames:
        List of file-names (e.g. ["a.pdb","b.sdf"]) to process. If None, every file
        in `src_dir` will be used.
    move:
        If True, will move files instead of copying. Default is False (copy).

    Raises
    ------
    ValueError
        If `src_dir` doesn’t exist or isn’t a directory.
    FileNotFoundError
        If a requested filename isn’t found under `src_dir`.
    """
    src = Path(src_dir)
    if not src.is_dir():
        raise ValueError(
            f"Source directory {src!r} does not exist or is not a directory."
        )
    if move and symlink:
        raise ValueError(
            "`move` and `symlink` are mutually exclusive; pick at most one."
        )

    # Determine which files to process
    if filenames is None:
        files = [p.name for p in src.iterdir() if p.is_file()]
    else:
        files = list(filenames)

    for d in dest_dirs:
        dest = Path(d)
        dest.mkdir(parents=True, exist_ok=True)
        for name in files:
            src_file = src / name
            if not src_file.exists():
                raise FileNotFoundError(
                    f"File {src_file!r} not found in source directory."
                )
            dst_file = dest / name

            if symlink:
                # remove existing link or file if present
                if dst_file.exists() or dst_file.is_symlink():
                    dst_file.unlink()
                # create relative symlink for portability
                dst_file.symlink_to(src_file.resolve())
            elif move:
                shutil.move(str(src_file), str(dst_file))
            else:
                shutil.copy2(str(src_file), str(dst_file))
