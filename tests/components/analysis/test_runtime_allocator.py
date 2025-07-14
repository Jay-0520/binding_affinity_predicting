import numpy as np
import pytest

from binding_affinity_predicting.components.analysis.runtime_allocator import (
    AdaptiveRuntimeAllocator,
)


@pytest.fixture
def mock_calculation(mocker):
    """Create a mock Calculation with legs and lambda windows"""
    # Create mock calculation
    mock_calculation = mocker.MagicMock()

    # Create mock leg
    mock_leg = mocker.MagicMock()
    mock_leg.leg_type.name = "bound"

    # Create mock lambda windows
    mock_windows = []
    for i in range(3):  # Just 3 windows for simplicity
        mock_win = mocker.MagicMock()
        mock_win.lam_state = f"0.{i*2}"  # 0.0, 0.2, 0.4
        mock_win.running = False
        mock_win.ensemble_size = 5
        mock_win.get_tot_simulation_time.return_value = 1.0  # 1 ns current runtime
        mock_windows.append(mock_win)

    mock_leg.lambda_windows = mock_windows
    mock_calculation.legs = [mock_leg]
    mock_calculation.virtual_queue = mocker.MagicMock()

    return mock_calculation


@pytest.fixture
def mock_gradient_analyzer(mocker):
    """Create a mock GradientAnalyzer with predefined SEM values"""
    mock_analyzer = mocker.MagicMock()
    mock_analyzer.get_time_normalized_sems.return_value = [0.1, 0.15, 0.2]

    mocker.patch(
        'binding_affinity_predicting.components.analysis.gradient_analyzer.GradientAnalyzer',
        return_value=mock_analyzer,
    )

    return mock_analyzer


@pytest.fixture
def mock_dependencies(mocker):
    """Mock external dependencies to speed up tests"""
    mocker.patch('time.sleep')  # Speed up test
    mocker.patch('numpy.ceil', side_effect=lambda x: int(np.ceil(x)))


@pytest.fixture
def allocator(mock_calculation, mock_gradient_analyzer, mock_dependencies):
    """Create an AdaptiveRuntimeAllocator with mocked dependencies"""
    allocator = AdaptiveRuntimeAllocator(
        calculation=mock_calculation,
        runtime_constant=0.001,
        cycle_pause=0.1,  # Very short pause for testing
    )

    allocator.gradient_analyzer = mock_gradient_analyzer

    return allocator


def test_run_adaptive_efficiency_loop_basic(
    mocker, mock_calculation, allocator, mock_gradient_analyzer
):
    """Basic test for AdaptiveRuntimeAllocator.run_adaptive_efficiency_loop() method"""

    def make_efficient(*args, **kwargs):
        """when called, set allocator to maximally efficient"""
        allocator._maximally_efficient = True

    # Set up the first window to need resubmission, others are efficient
    mock_windows = mock_calculation.legs[0].lambda_windows
    # the first time allocator invoke window.run(), that call will flip the internal
    # _maximally_efficient flag to True
    mock_windows[0].run.side_effect = make_efficient

    allocator.run_adaptive_efficiency_loop(run_nos=[1])

    assert allocator.is_maximally_efficient is True
    mock_gradient_analyzer.get_time_normalized_sems.assert_called_once()

    # Verify that at least one window was checked for runtime
    assert any(win.get_tot_simulation_time.called for win in mock_windows)


def test_calculate_optimal_runtime_directly(allocator):
    """Test the optimal runtime calculation method directly"""

    # Test data: SEM values and expected optimal runtimes
    test_cases = [
        (0.1, 0.1 / np.sqrt(0.001 * 1.0)),  # ~3.162 ns
        (0.15, 0.15 / np.sqrt(0.001 * 1.0)),  # ~4.743 ns
        (0.2, 0.2 / np.sqrt(0.001 * 1.0)),  # ~6.325 ns
    ]

    for sem_value, expected_optimal in test_cases:
        actual_optimal = allocator._calculate_optimal_runtime(sem_value)
        assert pytest.approx(expected_optimal, rel=1e-2) == actual_optimal
