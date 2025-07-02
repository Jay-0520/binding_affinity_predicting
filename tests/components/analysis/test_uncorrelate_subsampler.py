"""
Pytest script to test uncorrelate_subsampler.py with real FEP data.
Tests autocorrelation analysis and subsampling functionality.


Use results from alchemical_analysis.py as reference data.
"""

import numpy as np
import pytest

# Import the functions to test
from binding_affinity_predicting.components.analysis.uncorrelate_subsampler import (
    _construct_observable_series,
    _find_last_repeated_lambda_state,
    calc_statistical_inefficiency,
    perform_uncorrelating_subsampling,
    perform_uncorrelating_subsampling_multi_observable,
    subsample_correlated_data,
)
from binding_affinity_predicting.components.analysis.utils import (
    get_lambda_components_changing,
)

# Reference results from alchemical_analysis.py for validation
REFERENCE_RESULTS = {
    'dhdl': [51, 27, 16, 26, 13],
    'dhdl_all': [32, 38, 16, 26, 29],
    'de': [22, 29, 15, 35, 13],
}


@pytest.fixture
def analysis_windows(lambda_data):
    """Create analysis windows based on available snapshots."""
    nsnapshots = lambda_data['nsnapshots']
    num_states = len(nsnapshots)

    # Use all available data (no equilibration skipping)
    start_indices = np.zeros(num_states, dtype=int)
    end_indices = nsnapshots.copy()

    return start_indices, end_indices


def test_calc_statistical_inefficiency_basic():
    """Test statistical inefficiency calculation with synthetic data."""
    # Test with uncorrelated data
    uncorr_data = np.random.normal(0, 1, 1000)
    g_uncorr = calc_statistical_inefficiency(uncorr_data)
    assert 0.5 < g_uncorr < 2.0, f"Uncorrelated data should have g~1, got {g_uncorr}"

    # Test with correlated data (AR(1) process)
    phi = 0.9  # Strong correlation
    corr_data = np.zeros(1000)
    corr_data[0] = np.random.normal()
    for i in range(1, 1000):
        corr_data[i] = phi * corr_data[i - 1] + np.sqrt(1 - phi**2) * np.random.normal()

    g_corr = calc_statistical_inefficiency(corr_data)
    assert g_corr > 5.0, f"Correlated data should have g>5, got {g_corr}"

    # Test edge cases
    short_data = np.array([1, 2, 3])
    g_short = calc_statistical_inefficiency(short_data)
    assert g_short == 1.0, "Short data should return g=1"

    zeros_data = np.zeros(100)
    g_zeros = calc_statistical_inefficiency(zeros_data)
    assert g_zeros == 1.0, "Zero variance data should return g=1"


def test_subsample_correlated_data():
    """Test subsampling of correlated data."""
    data = np.arange(100)  # Simple sequential data

    # Test with g=1 (no correlation)
    indices_g1 = subsample_correlated_data(data, g=1.0)
    assert len(indices_g1) == len(data), "g=1 should return all indices"

    # Test with g=10 (strong correlation)
    indices_g10 = subsample_correlated_data(data, g=10.0)
    assert len(indices_g10) < len(data), "g>1 should return fewer indices"
    assert len(indices_g10) == len(data) // 10, "Should subsample by factor of g"

    # Test automatic g calculation
    random_data = np.random.normal(0, 1, 1000)
    indices_auto = subsample_correlated_data(random_data)
    assert len(indices_auto) > 0, "Should return at least some indices"


def test_find_last_repeated_lambda_state(lambda_data):
    """Test finding repeated lambda states."""
    lambda_vectors = lambda_data['lv']
    num_states = len(lambda_vectors)

    for state_idx in range(num_states):
        last_repeated = _find_last_repeated_lambda_state(state_idx, lambda_vectors)

        # Should return a valid state index
        assert (
            0 <= last_repeated < num_states
        ), f"Invalid state index returned: {last_repeated}"

        # Lambda vectors should be equal
        np.testing.assert_array_equal(
            lambda_vectors[state_idx],
            lambda_vectors[last_repeated],
            err_msg=f"Lambda vectors should match for states {state_idx} and {last_repeated}",
        )

        # Should be >= original index (finds LAST occurrence)
        assert last_repeated >= state_idx, "Should find last occurrence"


def test_construct_observable_series(lambda_data, analysis_windows):
    """Test basic functionality of 'dhdl' observable."""
    num_states = lambda_data["lv"].shape[0]
    start_indices, end_indices = analysis_windows  # These are arrays

    lambda_components_changing = get_lambda_components_changing(
        lambda_vectors=lambda_data['lv']
    )
    for lambda_state_idx in range(num_states):
        window_start = start_indices[lambda_state_idx]
        window_end = end_indices[lambda_state_idx]
        expected_length = window_end - window_start

        obs_series = _construct_observable_series(
            observable='dhdl',
            lambda_state_idx=lambda_state_idx,
            dhdl_timeseries=lambda_data['dhdlt'],
            potential_energies=lambda_data['u_klt'],
            lambda_vectors=lambda_data['lv'],
            lambda_components_changing=lambda_components_changing,
            window_start=window_start,
            window_end=window_end,
        )

        # Check basic properties
        assert (
            len(obs_series) == expected_length
        ), f"Length mismatch for state {lambda_state_idx}"
        assert isinstance(obs_series, np.ndarray), "Should return numpy array"
        assert obs_series.ndim == 1, "Should return 1D array"
        assert np.all(
            np.isfinite(obs_series)
        ), f"Non-finite values in state {lambda_state_idx}"


def test_specific_reference_values(lambda_data, analysis_windows):
    """Test against specific known reference values to prevent regression."""
    lambda_vectors = lambda_data['lv']
    dhdl_timeseries = lambda_data['dhdlt']
    potential_energies = lambda_data['u_klt']
    start_indices, end_indices = analysis_windows

    for observable, expected_values in REFERENCE_RESULTS.items():
        pot_energies = potential_energies if observable == 'de' else None

        _, _, num_uncorr_samples_per_state = perform_uncorrelating_subsampling(
            dhdl_timeseries=dhdl_timeseries,
            lambda_vectors=lambda_vectors,
            start_indices=start_indices,
            end_indices=end_indices,
            potential_energies=pot_energies,
            observable=observable,
            min_uncorr_samples=2,
        )

        np.testing.assert_array_equal(
            num_uncorr_samples_per_state,
            np.array(expected_values),
            err_msg=f"Results for {observable} observable don't match reference",
        )


def test_multi_observable_analysis(lambda_data, analysis_windows):
    """Test multi-observable analysis function."""
    lambda_vectors = lambda_data['lv']
    dhdl_timeseries = lambda_data['dhdlt']
    potential_energies = lambda_data['u_klt']
    start_indices, end_indices = analysis_windows

    observables = ['dhdl', 'dhdl_all', 'de']

    results = perform_uncorrelating_subsampling_multi_observable(
        dhdl_timeseries=dhdl_timeseries,
        lambda_vectors=lambda_vectors,
        start_indices=start_indices,
        end_indices=end_indices,
        potential_energies=potential_energies,
        observables=observables,
        min_uncorr_samples=2,
    )

    # Check results
    assert len(results) == len(
        observables
    ), "Should have results for all requested observables"

    for obs in observables:
        assert obs in results, f"Missing results for {obs}"
        dhdl_uncorr, potential_uncorr, num_samples = results[obs]

        assert dhdl_uncorr is not None, f"dH/dλ should be available for {obs}"
        assert len(num_samples) == len(
            lambda_vectors
        ), f"Sample count mismatch for {obs}"
        assert all(
            n > 0 for n in num_samples
        ), f"All states should have samples for {obs}"

        # Check against reference
        expected = REFERENCE_RESULTS[obs]
        np.testing.assert_array_equal(
            num_samples,
            np.array(expected),
            err_msg=f"Multi-observable results for {obs} don't match reference",
        )


def test_error_handling(lambda_data, analysis_windows):
    """Test error handling for invalid inputs."""
    lambda_vectors = lambda_data['lv']
    dhdl_timeseries = lambda_data['dhdlt']
    start_indices, end_indices = analysis_windows

    # Test invalid observable
    with pytest.raises(ValueError, match="Unknown observable"):
        perform_uncorrelating_subsampling(
            dhdl_timeseries=dhdl_timeseries,
            lambda_vectors=lambda_vectors,
            start_indices=start_indices,
            end_indices=end_indices,
            observable='invalid_observable',
        )

    # Test 'de' observable without potential energies
    with pytest.raises(ValueError, match="potential_energies required"):
        perform_uncorrelating_subsampling(
            dhdl_timeseries=dhdl_timeseries,
            lambda_vectors=lambda_vectors,
            start_indices=start_indices,
            end_indices=end_indices,
            potential_energies=None,
            observable='de',
        )

    # Test invalid dhdl_timeseries dimensions
    invalid_dhdl = np.random.random((10, 10))  # 2D instead of 3D
    with pytest.raises(ValueError, match="dhdl_timeseries must be 3D"):
        perform_uncorrelating_subsampling(
            dhdl_timeseries=invalid_dhdl,
            lambda_vectors=lambda_vectors,
            start_indices=start_indices,
            end_indices=end_indices,
            observable='dhdl',
        )
