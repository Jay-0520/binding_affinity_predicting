import logging
import os
import pickle

import numpy as np
import pytest

from binding_affinity_predicting.components.analysis.equilibrium_detecter import (
    EquilibriumBlockGradientDetector,
    EquilibriumMultiwindowDetector,
    _compute_dg_mbar,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.fixture(scope="module")
def data_for_running_mbar(test_data_dir) -> dict:
    """
    load mbar data from pickle file

    This data was obtained by running this script:
    /binding_affinity_predicting/scripts/test_run_a3fe/load_somd_sim_data.py

    note that percentage_end=100.0 so in this case
    end_frac must be 1.0
    """
    mbar_pickle_path = test_data_dir / "data_for_mbar.pkl"
    with open(mbar_pickle_path, 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def data_for_detecting_equil(test_data_dir) -> dict:
    """
    load mbar data from pickle file

    This data was obtained by running this function get_time_series_multiwindow_mbar()
    from here "a3fe/analyse/process_grads.py" on this data "data_for_mbar.pkl"

    This data contains the overall_dgs and overall_times for 10 intervals
    with start_frac defined start_fracs = np.linspace(0, 1 - self.last_frac, num=self.intervals)
    where last_frac = 0.5 by default
    """
    mbar_pickle_path = test_data_dir / "data_for_detect_equil.pkl"
    with open(mbar_pickle_path, 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def gradient_data(test_data_dir) -> list[np.ndarray]:
    """
    Gradient data for a single run (lam_windows[3] and run_nos=[1]).

    This is same as the data used in test_per_window_equilibration_detection() from A3FE
    """
    gradient_pickle_path = test_data_dir / "gradient_data_one_run.pkl"
    with open(gradient_pickle_path, 'rb') as f:
        return pickle.load(f)


class MockLambdaWindow:
    """Mock LambdaWindow class for testing."""

    def __init__(
        self,
        lam_state: float,
        lam_val_weight: float,
        ensemble_size: int = 5,
        tot_simtime=1000.0,
        output_dir: str = None,
    ):
        self.lam_state = lam_state
        self.lam_val_weight = lam_val_weight
        self.ensemble_size = ensemble_size
        self._tot_simtime = tot_simtime
        if output_dir is None:
            output_dir = os.getcwd()
        self.output_dir = output_dir

    def get_tot_simtime(self, run_nos):
        return self._tot_simtime


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
    Test the _get_time_series_multiwindow() method against reference values.

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


def test_compute_dg_mbar_with_start_frac_zero(mocker, data_for_running_mbar):
    """
    Test _compute_dg_mbar() with real MBAR calculations using actual data

    The reference value was computed by subjecting the same data to a3fe _compute_dg() from
    https://github.com/michellab/a3fe/blob/main/a3fe/analyse/process_grads.py  _compute_dg()
    """
    # This setting is based on data from test_data "lambda_windows_data.pkl"
    lambda_windows = [
        MockLambdaWindow(0.000, lam_val_weight=0.0625),
        MockLambdaWindow(0.125, lam_val_weight=0.125),
        MockLambdaWindow(0.250, lam_val_weight=0.125),
        MockLambdaWindow(0.375, lam_val_weight=0.125),
        MockLambdaWindow(0.500, lam_val_weight=0.3125),
        MockLambdaWindow(1.000, lam_val_weight=0.25),
    ]
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch(
        "binding_affinity_predicting.components.analysis."
        "equilibrium_detecter._load_alchemical_data_for_run",
        return_value=data_for_running_mbar,
    )
    result = _compute_dg_mbar(
        run_no=1,
        start_frac=0.99,
        end_frac=1.0,
        lambda_windows=lambda_windows,
        equilibrated=False,
        temperature=298.15,
        units="kcal",
    )
    assert pytest.approx(1.5087, rel=0.001) == result


def test_compute_dg_mbar_with_start_frac_nonzero(mocker, data_for_running_mbar):
    """
    Test _compute_dg_mbar() with real MBAR calculations using actual data

    The reference value was computed by subjecting the same data to a3fe _compute_dg() from
    https://github.com/michellab/a3fe/blob/main/a3fe/analyse/process_grads.py  _compute_dg()
    """
    # This setting is based on data from test_data "lambda_windows_data.pkl"
    lambda_windows = [
        MockLambdaWindow(0.000, lam_val_weight=0.0625),
        MockLambdaWindow(0.125, lam_val_weight=0.125),
        MockLambdaWindow(0.250, lam_val_weight=0.125),
        MockLambdaWindow(0.375, lam_val_weight=0.125),
        MockLambdaWindow(0.500, lam_val_weight=0.3125),
        MockLambdaWindow(1.000, lam_val_weight=0.25),
    ]
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch(
        "binding_affinity_predicting.components.analysis."
        "equilibrium_detecter._load_alchemical_data_for_run",
        return_value=data_for_running_mbar,
    )
    result = _compute_dg_mbar(
        run_no=1,
        start_frac=0.995,
        end_frac=1.0,
        lambda_windows=lambda_windows,
        equilibrated=False,
        temperature=298.15,
        units="kcal",
    )
    assert pytest.approx(1.2491, rel=0.001) == result


def test_get_time_series_multiwindow_mbar_zero(mocker, data_for_running_mbar):
    """
    Test the _get_time_series_multiwindow_mbar() method against reference values.

    This test loads pickle data, creates mock objects, calls the method,
    and compares the mean of overall_dgs and overall_times to expected values.
    """
    detector = EquilibriumMultiwindowDetector(method="paired_t")

    # This setting is based on data from test_data "lambda_windows_data.pkl"
    lambda_windows = [
        MockLambdaWindow(0.000, lam_val_weight=0.0625),
        MockLambdaWindow(0.125, lam_val_weight=0.125),
        MockLambdaWindow(0.250, lam_val_weight=0.125),
        MockLambdaWindow(0.375, lam_val_weight=0.125),
        MockLambdaWindow(0.500, lam_val_weight=0.3125),
        MockLambdaWindow(1.000, lam_val_weight=0.25),
    ]
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch(
        "binding_affinity_predicting.components.analysis."
        "equilibrium_detecter._load_alchemical_data_for_run",
        return_value=data_for_running_mbar,
    )
    overall_dgs, _ = detector._get_time_series_multiwindow_mbar(
        lambda_windows=lambda_windows,
        run_nos=[1],
        start_frac=0.0,
        end_frac=1.0,
        use_multiprocessing=False,
    )
    assert pytest.approx(1.50877812, rel=0.001) == overall_dgs.mean(axis=0)[-1]


def test_get_time_series_multiwindow_mbar_nonzero(mocker, data_for_running_mbar):
    """
    Test the _get_time_series_multiwindow_mbar() method against reference values.

    This test loads pickle data, creates mock objects, calls the method,
    and compares the mean of overall_dgs and overall_times to expected values.
    """
    detector = EquilibriumMultiwindowDetector(method="paired_t")

    # This setting is based on data from test_data "lambda_windows_data.pkl"
    lambda_windows = [
        MockLambdaWindow(0.000, lam_val_weight=0.0625),
        MockLambdaWindow(0.125, lam_val_weight=0.125),
        MockLambdaWindow(0.250, lam_val_weight=0.125),
        MockLambdaWindow(0.375, lam_val_weight=0.125),
        MockLambdaWindow(0.500, lam_val_weight=0.3125),
        MockLambdaWindow(1.000, lam_val_weight=0.25),
    ]
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch(
        "binding_affinity_predicting.components.analysis."
        "equilibrium_detecter._load_alchemical_data_for_run",
        return_value=data_for_running_mbar,
    )
    overall_dgs, _ = detector._get_time_series_multiwindow_mbar(
        lambda_windows=lambda_windows,
        run_nos=[1],
        start_frac=0.5,
        end_frac=1.0,
        use_multiprocessing=False,
    )
    assert pytest.approx(1.249191, rel=0.001) == overall_dgs.mean(axis=0)[-1]


def test_detect_paired_t_based(mocker, data_for_detecting_equil):
    """
    Test the _detect_paired_t_based() method against reference values.
    """
    detector = EquilibriumMultiwindowDetector(method="paired_t", intervals=10)

    start_fracs = np.linspace(0, 1 - detector.last_frac, num=detector.intervals)

    def mock_get_time_series_side_effect_indexed(lambda_windows, run_nos, start_frac):
        """Return data based on start_frac index"""
        frac_index = np.argmin(np.abs(start_fracs - start_frac))
        return (
            data_for_detecting_equil["overall_dgs"][frac_index],
            data_for_detecting_equil["overall_times"][frac_index],
        )

    # This setting is based on data from test_data "lambda_windows_data.pkl"
    lambda_windows = [
        MockLambdaWindow(0.000, lam_val_weight=0.0625),
        MockLambdaWindow(0.125, lam_val_weight=0.125),
        MockLambdaWindow(0.250, lam_val_weight=0.125),
        MockLambdaWindow(0.375, lam_val_weight=0.125),
        MockLambdaWindow(0.500, lam_val_weight=0.3125),
        MockLambdaWindow(1.000, lam_val_weight=0.25),
    ]
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch(
        "binding_affinity_predicting.components.analysis."
        "equilibrium_detecter.EquilibriumMultiwindowDetector._get_time_series_multiwindow_mbar",
        side_effect=mock_get_time_series_side_effect_indexed,
    )
    equilibrated, fractional_equil_time = detector._detect_paired_t_based(
        lambda_windows=lambda_windows,
        run_nos=[1, 2, 3, 4, 5],
    )
    assert equilibrated is True
    assert fractional_equil_time == pytest.approx(0.0, rel=0.01)


def test_equilibriumblockgradientdetector_detect_equilibrium(mocker, gradient_data):
    """
    gradient_data contains gradient data for a single run

    0.0024 is the reference value for equilibration time
    This test is based on the test_per_window_equilibration_detection() from A3FE
    """

    detector = EquilibriumBlockGradientDetector(
        block_size=0.05, gradient_threshold=None
    )
    # This setting is based on data from test_data "lambda_windows_data.pkl"
    lambda_windows = [
        MockLambdaWindow(0.000, lam_val_weight=0.0625),
        MockLambdaWindow(0.125, lam_val_weight=0.125),
        MockLambdaWindow(0.250, lam_val_weight=0.125),
        MockLambdaWindow(0.375, lam_val_weight=0.125),
        MockLambdaWindow(0.500, lam_val_weight=0.3125),
        MockLambdaWindow(1.000, lam_val_weight=0.25),
    ]

    formatted_gradient_data = ([gradient_data['times']], [gradient_data['gradients']])
    mocker.patch(
        "binding_affinity_predicting.components.analysis."
        "equilibrium_detecter.GradientAnalyzer.read_gradients_from_window",
        return_value=formatted_gradient_data,
    )

    equilibrated, equil_time = detector.detect_equilibrium(
        window=lambda_windows[3], run_nos=[1]
    )
    assert equilibrated is True
    assert equil_time == pytest.approx(0.0024, rel=0.01)
