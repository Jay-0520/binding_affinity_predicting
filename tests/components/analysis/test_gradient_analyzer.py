import numpy as np
import pytest

from binding_affinity_predicting.components.analysis.gradient_analyzer import (
    GradientAnalyzer,
)


class MockLambdaWindow:
    """Mock lambda window for testing GradientAnalyzer."""

    def __init__(
        self,
        lam_state: str,
        lam_val: float,
        lam_val_weight: float,
        ensemble_size: int = 3,
        output_dir: str = "/mock/output",
    ):
        self.lam_state = lam_state
        self.lam = lam_val
        self.lam_val_weight = lam_val_weight
        self.ensemble_size = ensemble_size
        self.output_dir = output_dir
        self._total_sim_time = 10.0

    def get_tot_simulation_time(self, run_nos: list[int]) -> float:
        """Mock method to return total simulation time."""
        return (
            self._total_sim_time * len(run_nos)
            if run_nos
            else self._total_sim_time * self.ensemble_size
        )


@pytest.fixture
def mock_lambda_windows():
    """Fixture providing mock lambda windows for GradientAnalyzer."""
    return [
        MockLambdaWindow(lam_state="0.00", lam_val=0.0, lam_val_weight=0.1),
        MockLambdaWindow(lam_state="0.50", lam_val=0.5, lam_val_weight=0.2),
        MockLambdaWindow(lam_state="1.00", lam_val=1.0, lam_val_weight=0.1),
    ]


def test_get_time_normalized_sems_basic(mocker, mock_lambda_windows):
    """Test the get_time_normalized_sems method with simple mock data."""
    np.random.seed(42)
    n_points = 5000
    times = np.linspace(0, 10, n_points)
    noise = np.random.normal(0, 5, n_points)
    gradients = -50.0 + noise

    # Mock data for 3 identical runs
    mock_gradient_data = (
        np.array([times, times, times]),
        np.array([gradients, gradients, gradients]),
    )

    mock_read_gradients = mocker.patch.object(
        GradientAnalyzer, 'read_gradients_from_window'
    )
    mock_read_gradients.return_value = mock_gradient_data

    mock_stat_ineff = mocker.patch(
        'binding_affinity_predicting.components.analysis.gradient_analyzer._statistical_inefficiency_chodera'  # noqa: E501
    )
    mock_stat_ineff.return_value = 2.0

    gradient_analyzer = GradientAnalyzer()

    result_list = []
    for origin in ["intra", "inter"]:
        result = gradient_analyzer.get_time_normalized_sems(
            lambda_windows=mock_lambda_windows,
            run_nos=[1, 2, 3],
            origin=origin,
            smoothen=True,
        )
        result_list.append(result)

    assert (
        pytest.approx(np.array([0.310987, 0.310987, 0.310987]), rel=0.0001)
        == result_list[0]
    )
    assert pytest.approx(np.array([0.0, 0.0, 0.0]), rel=0.0001) == result_list[1]
