import logging
import pickle
from unittest.mock import patch

import numpy as np
import pytest

from binding_affinity_predicting.components.analysis.equilibrium_detecter import (
    EquilibriumMultiwindowDetector,
    _compute_dg_mbar,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.fixture(scope="module")
def data_for_running_mbar(test_data_dir) -> dict:
    """
    load mbar data from pickle file

    This data was obtained by running this script
    binding_affinity_predicting/scripts/test_run_a3fe/load_somd_sim_data.py

    note that percentage_end=100.0 so in this case
    end_frac must be 1.0
    """
    mbar_pickle_path = test_data_dir / "mbar_data.pkl"
    with open(mbar_pickle_path, 'rb') as f:
        return pickle.load(f)


class MockLambdaWindow:
    """Mock LambdaWindow class for testing."""

    def __init__(self, lam_state: float, lam_val_weight: float):
        self.lam_state = lam_state
        self.lam_val_weight = lam_val_weight


class MockGradientAnalyzer:
    """Mock GradientAnalyzer class for testing."""

    def __init__(self, pickle_data: dict):
        self.pickle_data = pickle_data

    def read_gradients_from_window(
        self, lam_win: MockLambdaWindow, run_nos: list[int]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Mock method that returns data from pickle file with clean structure.

        This function is strictly based on the pickle data structure
        """
        times_list = []
        gradients_list = []

        window_data_key = f"lambda_{float(lam_win.lam_state):.3f}"
        window_data = self.pickle_data['windows'][window_data_key]

        for run_no in run_nos:
            run_key = f'run_{run_no}'
            run_data = window_data['runs'][run_key]
            times = run_data['times']
            gradients = run_data['gradients']
            times_list.append(times)
            gradients_list.append(gradients)

        return times_list, gradients_list


def test_get_time_series_multiwindow(lambda_windows_data):
    """
    Test the _get_time_series_multiwindow method against reference values.

    This test loads pickle data, creates mock objects, calls the method,
    and compares the mean of overall_dgs and overall_times to expected values.
    """
    # Create mock gradient analyzer with pickle data
    mock_gradient_analyzer = MockGradientAnalyzer(lambda_windows_data)

    detector = EquilibriumMultiwindowDetector(method="paired_t")
    detector.gradient_analyzer = mock_gradient_analyzer

    # This setting is based on data from test_data "lambda_windows_data.pkl"
    lambda_windows = [
        MockLambdaWindow(0.000, lam_val_weight=0.0625),
        MockLambdaWindow(0.125, lam_val_weight=0.125),
        MockLambdaWindow(0.250, lam_val_weight=0.125),
        MockLambdaWindow(0.375, lam_val_weight=0.125),
        MockLambdaWindow(0.500, lam_val_weight=0.3125),
        MockLambdaWindow(1.000, lam_val_weight=0.25),
    ]

    overall_dgs, overall_times = detector._get_time_series_multiwindow(
        lambda_windows=lambda_windows, run_nos=[1, 2]
    )
    assert overall_dgs.mean(axis=0)[-1] == pytest.approx(1.7751, abs=1e-2)
    assert overall_dgs.mean(axis=0)[-2] == pytest.approx(2.0671, abs=1e-2)
    assert overall_dgs.mean(axis=0)[-3] == pytest.approx(2.2889, abs=1e-2)
    assert overall_times.sum(axis=0)[-1] == pytest.approx(2.4, abs=1e-2)


def test_compute_dg_mbar_1(data_for_running_mbar):
    """Test _compute_dg_mbar with real MBAR calculations using actual data"""
    lambda_windows = [
        MockLambdaWindow(0.000, lam_val_weight=0.0625),
        MockLambdaWindow(0.125, lam_val_weight=0.125),
        MockLambdaWindow(0.250, lam_val_weight=0.125),
        MockLambdaWindow(0.375, lam_val_weight=0.125),
        MockLambdaWindow(0.500, lam_val_weight=0.3125),
        MockLambdaWindow(1.000, lam_val_weight=0.25),
    ]

    with (
        patch('pathlib.Path.exists', return_value=True),
        patch(
            'binding_affinity_predicting.components.analysis.equilibrium_detecter._load_alchemical_data_for_run',  # noqa: E501
            return_value=data_for_running_mbar,
        ),
    ):

        result = _compute_dg_mbar(
            run_no=1,
            start_frac=0.0,
            end_frac=1.0,
            lambda_windows=lambda_windows,
            equilibrated=False,
            temperature=298.15,
            units="kcal",
        )

    assert pytest.approx(1.5087, rel=0.001) == result


def test_compute_dg_mbar_2(data_for_running_mbar):
    """
    Test _compute_dg_mbar with real MBAR calculations using actual data

    The reference value was computed by subjecting the same data to a3fe _compute_dg() from
    https://github.com/michellab/a3fe/blob/main/a3fe/analyse/process_grads.py  _compute_dg()
    """
    lambda_windows = [
        MockLambdaWindow(0.000, lam_val_weight=0.0625),
        MockLambdaWindow(0.125, lam_val_weight=0.125),
        MockLambdaWindow(0.250, lam_val_weight=0.125),
        MockLambdaWindow(0.375, lam_val_weight=0.125),
        MockLambdaWindow(0.500, lam_val_weight=0.3125),
        MockLambdaWindow(1.000, lam_val_weight=0.25),
    ]

    with (
        patch('pathlib.Path.exists', return_value=True),
        patch(
            'binding_affinity_predicting.components.analysis.equilibrium_detecter._load_alchemical_data_for_run',  # noqa: E501
            return_value=data_for_running_mbar,
        ),
    ):

        result = _compute_dg_mbar(
            run_no=1,
            start_frac=0.5,
            end_frac=1.0,
            lambda_windows=lambda_windows,
            equilibrated=False,
            temperature=298.15,
            units="kcal",
        )

    assert pytest.approx(1.2491, rel=0.001) == result
