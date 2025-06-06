import shutil
import warnings
from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).parent / "test_files"


def pytest_configure():
    # Silence only the three SWIG-generated __module__ deprecation warnings:
    for swig in ("SwigPyPacked", "SwigPyObject", "swigvarlink"):
        warnings.filterwarnings(
            "ignore",
            message=fr".*builtin type {swig} has no __module__ attribute.*",
            category=DeprecationWarning,
        )


@pytest.fixture(scope="module")
def example_structures(tmp_path_factory) -> str:
    """
    Copy the real PDB/SDF test files into a temp directory once
    and give its path to any test that needs it.
    """
    d = tmp_path_factory.mktemp("structs")
    for fn in ("protein.pdb", "ligand.sdf", "water.pdb"):
        shutil.copy(TEST_DATA_DIR / fn, d / fn)
    return d


@pytest.fixture(scope="function")
def out_dir(tmp_path) -> Path:
    """A fresh empty directory for writing outputs."""
    return tmp_path / "out"
