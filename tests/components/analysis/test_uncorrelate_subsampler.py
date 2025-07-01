"""
Pytest script to test uncorrelate_subsampler.py with real FEP data.
Tests autocorrelation analysis and subsampling functionality.


Use results from alchemical_analysis.py as reference data.
"""

import pickle
import numpy as np
import pytest
from pathlib import Path

# Import the functions to test
from binding_affinity_predicting.components.analysis.uncorrelate_subsampler import (
    perform_uncorrelating_subsampling,
    perform_uncorrelating_subsampling_multi_observable,
    statistical_inefficiency,
    subsample_correlated_data,
    _construct_observable_series,
    _find_last_repeated_lambda_state
)



@pytest.fixture
def analysis_windows(lambda_data):
    """Create analysis windows based on available snapshots."""
    nsnapshots = lambda_data['nsnapshots']
    num_states = len(nsnapshots)
    
    # Use all available data (no equilibration skipping)
    start_indices = np.zeros(num_states, dtype=int)
    end_indices = nsnapshots.copy()
    
    return start_indices, end_indices



def test_statistical_inefficiency_basic():
    """Test statistical inefficiency calculation with synthetic data."""
    # Test with uncorrelated data
    uncorr_data = np.random.normal(0, 1, 1000)
    g_uncorr = statistical_inefficiency(uncorr_data)
    assert 0.5 < g_uncorr < 2.0, f"Uncorrelated data should have g~1, got {g_uncorr}"
    
    # Test with correlated data (AR(1) process)
    phi = 0.9  # Strong correlation
    corr_data = np.zeros(1000)
    corr_data[0] = np.random.normal()
    for i in range(1, 1000):
        corr_data[i] = phi * corr_data[i-1] + np.sqrt(1-phi**2) * np.random.normal()
    
    g_corr = statistical_inefficiency(corr_data)
    assert g_corr > 5.0, f"Correlated data should have g>5, got {g_corr}"
    
    # Test edge cases
    short_data = np.array([1, 2, 3])
    g_short = statistical_inefficiency(short_data)
    assert g_short == 1.0, "Short data should return g=1"
    
    zeros_data = np.zeros(100)
    g_zeros = statistical_inefficiency(zeros_data)
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



def test_perform_uncorrelating_subsampling_dhdl(lambda_data, analysis_windows):
    """Test uncorrelating subsampling for dhdl observable."""
    lambda_vectors = lambda_data['lv']
    dhdl_timeseries = lambda_data['dhdlt']
    start_indices, end_indices = analysis_windows
    
    dhdl_uncorr, _, num_uncorr_samples_per_state = perform_uncorrelating_subsampling(dhdl_timeseries=dhdl_timeseries,
                                      lambda_vectors=lambda_vectors,
                                      start_indices=start_indices,
                                      observable='dhdl',
                                      end_indices=end_indices,
                                      min_uncorr_samples=2)
    
    assert np.array_equal(num_uncorr_samples_per_state, np.array([51, 27, 16, 26, 13]))
    



def test_perform_uncorrelating_subsampling_dhdl_all(lambda_data, analysis_windows):
    """Test uncorrelating subsampling for dhdl_all observable."""
    lambda_vectors = lambda_data['lv']
    dhdl_timeseries = lambda_data['dhdlt']
    start_indices, end_indices = analysis_windows
    
    _, _, num_uncorr_samples_per_state = perform_uncorrelating_subsampling(dhdl_timeseries=dhdl_timeseries,
                                      lambda_vectors=lambda_vectors,
                                      start_indices=start_indices,
                                      observable='dhdl_all',
                                      end_indices=end_indices,
                                      min_uncorr_samples=2)
    
    assert np.array_equal(num_uncorr_samples_per_state, np.array([32, 38, 16, 26, 29]))



def test_perform_uncorrelating_subsampling_de(lambda_data, analysis_windows):
    """Test uncorrelating subsampling for dE observable."""
    lambda_vectors = lambda_data['lv']
    dhdl_timeseries = lambda_data['dhdlt']
    potential_energies = lambda_data['u_klt']
    start_indices, end_indices = analysis_windows
    
    _, _, num_uncorr_samples_per_state = perform_uncorrelating_subsampling(dhdl_timeseries=dhdl_timeseries,
                                      lambda_vectors=lambda_vectors,
                                      potential_energies=potential_energies,
                                      start_indices=start_indices,
                                      observable='de',
                                      end_indices=end_indices,
                                      min_uncorr_samples=2)
    
    print('num_uncorr_samples_per_state', num_uncorr_samples_per_state)
    # [22 29 15 35 13]
    assert np.array_equal(num_uncorr_samples_per_state, np.array([32, 38, 16, 26, 29]))


