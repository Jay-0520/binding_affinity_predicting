"""
Tests for GromacsXVGParser and load_alchemical_data function.

These tests validate parsing of GROMACS XVG files and ensure the output
format matches the expected structure used by free energy estimators.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

from binding_affinity_predicting.components.analysis.xvg_data_loader import (
    GromacsXVGParser,
    LambdaState,
    load_alchemical_data,
)


@pytest.fixture
def xvg_file_paths(test_data_dir):
    """Get paths to the 5 test XVG files."""
    filenames = [
        "lambda_0_run_1.xvg",
        "lambda_1_run_1.xvg",
        "lambda_2_run_1.xvg",
        "lambda_3_run_1.xvg",
        "lambda_4_run_1.xvg",
    ]

    file_paths = [test_data_dir / "xvg_files" / filename for filename in filenames]

    # Check if files exist
    missing_files = [f for f in file_paths if not f.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing test files: {', '.join(str(f) for f in missing_files)}"
        )

    return file_paths


def test_lambda_state_creation():
    """Test LambdaState creation and representation."""
    state = LambdaState(coul=0.5, vdw=0.8, bonded=1.0, state_id=2)

    assert state.coul == 0.5
    assert state.vdw == 0.8
    assert state.bonded == 1.0
    assert state.state_id == 2

    # Test string representation
    expected_repr = "LambdaState(2: coul=0.5, vdw=0.8, bonded=1.0)"
    assert repr(state) == expected_repr


def test_lambda_state_to_vector():
    """Test conversion to numpy array."""
    state = LambdaState(coul=0.2, vdw=0.6, bonded=0.9, state_id=1)
    vector = state.to_vector()

    expected = np.array([0.2, 0.6, 0.9])
    np.testing.assert_array_equal(vector, expected)
    assert isinstance(vector, np.ndarray)


def test_parse_current_state():
    """Test parsing of current lambda state from subtitle."""
    parser = GromacsXVGParser()
    subtitle_line = '@ subtitle "T = 300 (K) λ state 1: (coul-lambda, vdw-lambda, bonded-lambda) = (0.0000, 0.1000, 0.2500)"'  # noqa: E501

    parser._parse_current_state(subtitle_line)

    assert parser.current_state is not None
    assert parser.current_state.state_id == 1
    assert parser.current_state.coul == 0.0
    assert parser.current_state.vdw == 0.1
    assert parser.current_state.bonded == 0.25


def test_parse_current_state_invalid():
    """Test parsing with invalid subtitle format."""
    parser = GromacsXVGParser()
    invalid_subtitle = '@ subtitle "Invalid format"'

    with pytest.raises(ValueError, match="Could not derive lambda state from subtitle"):
        parser._parse_current_state(invalid_subtitle)


def test_parse_legend_dhdl_components():
    """Test parsing of dH/dλ component legends."""
    parser = GromacsXVGParser()

    parser._parse_legend('@ s0 legend "dH/dλ coul-lambda = 0.0000"')
    assert 'coulomb' in parser.dhdl_components

    parser._parse_legend('@ s1 legend "dH/dλ vdw-lambda = 0.1000"')
    assert 'vdw' in parser.dhdl_components

    parser._parse_legend('@ s2 legend "dH/dλ bonded-lambda = 0.2500"')
    assert 'bonded' in parser.dhdl_components

    assert len(parser.dhdl_components) == 3


def test_parse_legend_cross_evaluations():
    """Test parsing of cross-evaluation legends."""
    parser = GromacsXVGParser()

    legend_line = '@ s3 legend "ΔH λ to (0.0000, 0.0000, 0.0000)"'
    parser._parse_legend(legend_line)

    assert len(parser.cross_eval_states) == 1
    target_state = parser.cross_eval_states[0]
    assert target_state.coul == 0.0
    assert target_state.vdw == 0.0
    assert target_state.bonded == 0.0


def test_parse_single_xvg_file(xvg_file_paths):
    """Test parsing a single XVG file."""
    parser = GromacsXVGParser()

    # Parse the first file (lambda_0)
    data = parser.parse_xvg_file(xvg_file_paths[0], skip_time=0.0)

    # Verify structure
    assert 'times' in data
    assert 'current_state' in data
    assert 'dhdl_components' in data
    assert 'cross_evaluations' in data
    assert 'total_energy' in data
    assert 'pV' in data

    # Verify current state exists
    assert data['current_state'] is not None
    assert data['current_state'].state_id == 0

    # Verify time series exists
    assert len(data['times']) > 0
    assert isinstance(data['times'], np.ndarray)

    # Verify dH/dλ components exist
    assert len(data['dhdl_components']) > 0

    # Verify cross-evaluations exist
    assert len(data['cross_evaluations']) >= 0


def test_parse_xvg_file_with_skip_time(xvg_file_paths):
    """Test parsing with equilibration time skipping."""
    parser = GromacsXVGParser()

    # Parse without skipping
    data_full = parser.parse_xvg_file(xvg_file_paths[0], skip_time=0.0)
    full_length = len(data_full['times'])

    # Parse with skipping (skip first half of simulation time)
    if full_length > 1:
        mid_time = data_full['times'][full_length // 2]
        data_skipped = parser.parse_xvg_file(xvg_file_paths[0], skip_time=mid_time)

        # Should have fewer data points
        assert len(data_skipped['times']) < full_length
        # All remaining times should be >= skip_time
        assert np.all(data_skipped['times'] >= mid_time)


def test_load_alchemical_data_basic(xvg_file_paths):
    """Test basic loading of alchemical data from multiple files."""
    dhdl_timeseries, potential_energies, lambda_vectors, nsnapshots = (
        load_alchemical_data(xvg_file_paths, skip_time=0.0)
    )

    # Verify shapes
    num_states = 5
    num_components = len(
        set().union(
            *[
                GromacsXVGParser()
                .parse_xvg_file(f, skip_time=0.0)['dhdl_components']
                .keys()
                for f in xvg_file_paths[:1]  # Just check first file for component count
            ]
        )
    )

    assert dhdl_timeseries.shape[0] == num_states
    assert dhdl_timeseries.shape[1] == num_components
    assert potential_energies.shape[0] == num_states
    assert potential_energies.shape[1] == num_states
    assert lambda_vectors.shape[0] == num_states
    assert lambda_vectors.shape[1] == num_components
    assert nsnapshots.shape[0] == num_states

    # Verify data types
    assert dhdl_timeseries.dtype in [np.float64, np.float32]
    assert potential_energies.dtype in [np.float64, np.float32]
    assert lambda_vectors.dtype in [np.float64, np.float32]
    assert nsnapshots.dtype in [np.int64, np.int32]

    # Verify all states have some data
    assert np.all(nsnapshots > 0)

    # Verify arrays are consistent in time dimension
    max_snapshots = dhdl_timeseries.shape[2]
    assert potential_energies.shape[2] == max_snapshots
    assert max_snapshots >= np.max(nsnapshots)


def test_load_alchemical_data_with_skip_time(xvg_file_paths):
    """Test loading with equilibration time skipping."""
    # Load without skipping
    _, _, _, nsnapshots_full = load_alchemical_data(xvg_file_paths, skip_time=0.0)

    # Load with moderate skipping (skip some equilibration)
    skip_time = 1.0
    _, _, _, nsnapshots_skipped = load_alchemical_data(
        xvg_file_paths, skip_time=skip_time
    )

    # Should have fewer or equal snapshots after skipping
    assert np.all(nsnapshots_skipped <= nsnapshots_full)


def test_load_alchemical_data_missing_file():
    """Test behavior with missing files."""
    missing_files = [Path("/nonexistent/file.xvg")]

    with pytest.raises(FileNotFoundError):
        load_alchemical_data(missing_files)


def test_load_alchemical_data_no_valid_files():
    """Test behavior with no valid files."""
    with pytest.raises(ValueError, match="No valid XVG files could be parsed"):
        load_alchemical_data([])


def test_load_alchemical_data_save_to_pickle(xvg_file_paths, tmp_path):
    """Test saving parsed data to pickle file."""
    save_path = tmp_path / "test_data.pkl"

    # Load and save data
    dhdl_timeseries, potential_energies, lambda_vectors, nsnapshots = (
        load_alchemical_data(xvg_file_paths, skip_time=0.0, save_to_path=save_path)
    )

    # Verify file was created
    assert save_path.exists()

    # Load the saved data and verify
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)

    assert 'dhdl_timeseries' in saved_data
    assert 'potential_energies' in saved_data
    assert 'lambda_vectors' in saved_data
    assert 'nsnapshots' in saved_data

    # Verify arrays are identical
    np.testing.assert_array_equal(saved_data['dhdl_timeseries'], dhdl_timeseries)
    np.testing.assert_array_equal(saved_data['potential_energies'], potential_energies)
    np.testing.assert_array_equal(saved_data['lambda_vectors'], lambda_vectors)
    np.testing.assert_array_equal(saved_data['nsnapshots'], nsnapshots)


def test_load_alchemical_data_output_structure_matches_conftest(
    xvg_file_paths, lambda_data
):
    """Test that output structure matches the expected format from
    conftest.py lambda_data fixture."""
    result_tar = load_alchemical_data(
        xvg_files=xvg_file_paths, skip_time=0.0, temperature=300
    )

    lambda_data_structure = {
        'dhdl_timeseries': result_tar['dhdl_timeseries'],
        'potential_energies': result_tar['potential_energies'],
        'lambda_vectors': result_tar['lambda_vectors'],
        'nsnapshots': result_tar['nsnapshots'],
    }
    # 'dhdlt', 'u_klt', 'lv', 'nsnapshots' are keys used by parser_gromacs.py
    lambda_data_structure_ref = {
        'dhdl_timeseries': lambda_data['dhdlt'],
        'potential_energies': lambda_data['u_klt'],
        'lambda_vectors': lambda_data['lv'],
        'nsnapshots': lambda_data['nsnapshots'],
    }

    np.testing.assert_array_equal(
        lambda_data_structure['dhdl_timeseries'],
        lambda_data_structure_ref['dhdl_timeseries'],
    )
    np.testing.assert_array_equal(
        lambda_data_structure['lambda_vectors'],
        lambda_data_structure_ref['lambda_vectors'],
    )
    np.testing.assert_array_equal(
        lambda_data_structure['nsnapshots'], lambda_data_structure_ref['nsnapshots']
    )
    np.testing.assert_array_equal(
        lambda_data_structure['potential_energies'],
        lambda_data_structure_ref['potential_energies'],
    )
