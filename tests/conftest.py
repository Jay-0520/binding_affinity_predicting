import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np
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


@pytest.fixture(scope="session", autouse=True)
def suppress_warnings():
    """Autouse fixture to suppress warnings at the session level."""
    with warnings.catch_warnings():
        # Suppress the specific pymbar warnings
        # These are due to specific test data (lambda_data.pkl) and not indicative of code issues
        # and are from pymbar's internal numerical calculations
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="pymbar")
        warnings.filterwarnings(
            "ignore", message=".*invalid value encountered in sqrt.*"
        )
        warnings.filterwarnings(
            "ignore", message=".*divide by zero encountered in log.*"
        )
        warnings.filterwarnings(
            "ignore", message=".*invalid value encountered in scalar divide.*"
        )

        # Suppress numpy matrix deprecation warnings from pymbar
        warnings.filterwarnings(
            "ignore",
            message=".*matrix subclass is not the recommended way.*",
            category=PendingDeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore", category=PendingDeprecationWarning, module="pymbar.*"
        )
        warnings.filterwarnings(
            "ignore", category=PendingDeprecationWarning, module="numpy.matrixlib.*"
        )

        old_settings = np.seterr(divide='ignore', invalid='ignore')

        try:
            yield
        finally:
            np.seterr(**old_settings)


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


@pytest.fixture(scope="module")
def lambda_data() -> tuple:
    """
    Load the pickled FEP data once per test session (or skip if missing).
    Uses the TEST_DATA_DIR defined above.
    """
    p = TEST_DATA_DIR / "lambda_data.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)
