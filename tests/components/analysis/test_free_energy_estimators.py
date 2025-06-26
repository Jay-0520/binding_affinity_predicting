"""
Test suite for free_energy_estimators.py functions.

Tests are based on reference values from alchemical_analysis.py output
using the same input data (lambda_data.pkl).

lambda_data.pkl is obtained from loading several xvg files through parser_gromacs.py

alchemical_analysis.py is here:
https://github.com/MobleyLab/alchemical-analysis/blob/master/alchemical_analysis/alchemical_analysis.py

to run this script, e.g., with command:
   python  alchemical_analysis.py -d 'directory' -p 'lambda' -t 300 -s 0 -u kcal -w -g

we also have to download corruptxvg.py, unixlike.py and parser_gromacs.py in the same directory.
note that these scripts are very outdated so they must be updated properly to run with
new python versions.

Expected results from results.txt (which is outputed by running alchemical_analysis.py):
- TI: -3.064 ± 9.343 kcal/mol
- TI-CUBIC: -4.926 ± 11.621 kcal/mol
- DEXP: 5.590 ± 0.612 kcal/mol
- IEXP: 15.218 ± 0.755 kcal/mol
- GINS: 61245350.401 ± 61251232.076 kcal/mol
- GDEL: -3.616 ± 13.852 kcal/mol
- BAR: 7.774 ± nan kcal/mol
- UBAR: 4.922 ± nan kcal/mol
- RBAR: 7.661 ± nan kcal/mol
- MBAR: 7.561 ± 5.762 kcal/mol
"""

import numpy as np
import pytest

from binding_affinity_predicting.components.analysis.free_energy_estimators import (
    BennettAcceptanceRatio,
    ExponentialAveraging,
    FreeEnergyEstimator,
    MultistateBAR,
    NaturalCubicSpline,
    ThermodynamicIntegration,
)


@pytest.fixture
def ti_data(lambda_data):
    """
    Prepare lambda vectors, <dH/dλ> and std(dH/dλ) (already beta‐scaled)
    for TI tests.
    """
    lv = lambda_data["lv"]  # shape (n_states, n_components) (5, 3)
    dhdlt = lambda_data["dhdlt"]  # shape (n_states, n_components, n_samples) (5, 3, 3)

    # Boltzmann β for T=300K in kcal/mol:
    temperature = 300.0
    beta = 1.0 / (8.314472e-3 * temperature)

    # average and standard error over time‐series, same as in alchemical_analysis.py
    ave_dhdl = dhdlt.mean(axis=2) * beta
    std_dhdl = dhdlt.std(axis=2, ddof=1) / np.sqrt(dhdlt.shape[2]) * beta

    return lv, ave_dhdl, std_dhdl


@pytest.fixture
def exp_data(lambda_data):
    """
    Prepare reduced potentials u_klt and snapshot counts for MBAR/EXP/BAR tests.
    """
    return lambda_data["u_klt"], lambda_data["nsnapshots"]


@pytest.fixture
def bar_data(lambda_data):
    """
    Prepare reduced potentials u_klt and snapshot counts for MBAR/EXP/BAR tests.
    """
    return lambda_data["u_klt"], lambda_data["nsnapshots"]


@pytest.fixture
def mbar_data(lambda_data):
    """
    Prepare reduced potentials u_klt and snapshot counts for MBAR/EXP/BAR tests.
    """
    return lambda_data["u_klt"], lambda_data["nsnapshots"]


@pytest.fixture
def free_energy_estimator():
    """Create a FreeEnergyEstimator configured for our test conditions."""
    return FreeEnergyEstimator(temperature=300.0, units='kcal', software='Gromacs')


# ─── Thermodynamic Integration ──────────────────────────────────────────────────
def test_trapezoidal_integration(ti_data):
    """
    Test trapezoidal integration for TI.
    The expected values trapezoidal integration: -3.064 ± 9.343 kcal/mol
    """
    lv, ave, std = ti_data
    dg_kcal, ddg_kcal = ThermodynamicIntegration.trapezoidal_integration(
        lambda_vectors=lv, ave_dhdl=ave, std_dhdl=std, temperature=300.0, units='kcal'
    )

    assert pytest.approx(-3.064, rel=0.001) == dg_kcal
    assert pytest.approx(9.343, rel=0.001) == ddg_kcal


def test_cubic_spline_integration(ti_data):
    """
    Test cubic spline integration for TI.
    The expected values cubic spline integration: -4.926 ± 11.621 kcal/mol
    """
    lv, ave, std = ti_data
    dg_kcal, ddg_kcal = ThermodynamicIntegration.cubic_spline_integration(
        lambda_vectors=lv, ave_dhdl=ave, std_dhdl=std, temperature=300.0, units='kcal'
    )

    assert pytest.approx(-4.926, rel=0.001) == dg_kcal
    assert pytest.approx(11.621, rel=0.001) == ddg_kcal


@pytest.mark.parametrize(
    "lv, ave, std, err_msg",
    [
        # mismatched ave/std shape
        (
            np.array([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]),
            np.array([[1.0, 2.0], [1.5, 2.5]]),
            np.array([[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]]),
            "must have same shape",
        ),
        # too few points
        (
            np.array([[0.0, 1.0]]),
            np.array([[1.0, 2.0]]),
            np.array([[0.1, 0.2]]),
            "Need at least 2 lambda states",
        ),
    ],
)
def test_ti_input_validation(lv, ave, std, err_msg):
    """
    Test input validation for TI methods.
    """
    with pytest.raises(ValueError, match=err_msg):
        ThermodynamicIntegration.trapezoidal_integration(lv, ave, std)


# ─── Natural Cubic Spline ───────────────────────────────────────────────────────
def test_spline_creation_minimum_points():
    """
    Test that NaturalCubicSpline can be created with minimum points.
    """
    xs = np.array([0.0, 0.33, 0.67, 1.0])
    spline = NaturalCubicSpline(xs)
    assert spline.x.shape[0] == 4
    assert spline.wsum.shape[0] == 4


def test_spline_integration_polynomial():
    """
    Test integration of a simple polynomial using NaturalCubicSpline.
    """
    xs = np.linspace(0, 1, 5)
    ys = xs**2
    spline = NaturalCubicSpline(xs)
    val = spline.integrate(ys)
    assert pytest.approx(val, rel=0.001) == 0.335


def test_spline_insufficient_points():
    """
    Test that NaturalCubicSpline raises ValueError for insufficient points.
    """
    xs = np.array([0.0])
    with pytest.raises(ValueError, match="at least 2 points"):
        NaturalCubicSpline(xs)


# ─── Exponential Averaging (DEXP, IEXP, GDEL, GINS) ────────────────────────
def test_exponential_average_dexp(exp_data):
    """
    Test DEXP calculation using ExponentialAveraging.

    The expected values for DEXP: 5.590 ± 0.612 kcal/mol
    """
    u_klt, _ = exp_data
    dg_kcal, ddg_kcal = ExponentialAveraging.compute_dexp(
        potential_energies=u_klt, temperature=300.0, units="kcal"
    )

    assert pytest.approx(5.590, rel=0.001) == dg_kcal
    assert pytest.approx(0.612, rel=0.001) == ddg_kcal


def test_exponential_average_iexp(exp_data):
    """
    Test DEXP calculation using ExponentialAveraging.
    The expected values for IEXP: 15.218 ± 0.755 kcal/mol
    """
    u_klt, _ = exp_data
    dg_kcal, ddg_kcal = ExponentialAveraging.compute_iexp(
        potential_energies=u_klt, temperature=300.0, units="kcal"
    )
    assert pytest.approx(15.218, rel=0.001) == dg_kcal
    assert pytest.approx(0.755, rel=0.001) == ddg_kcal


def test_exponential_average_gdel(exp_data):
    """
    Test GDEL calculation using ExponentialAveraging.
    The expected values for GDEL: -3.616 ± 13.852 kcal/mol
    """
    u_klt, _ = exp_data
    dg_kcal, ddg_kcal = ExponentialAveraging.compute_gdel(
        potential_energies=u_klt, temperature=300.0, units="kcal"
    )

    assert pytest.approx(-3.616, rel=0.001) == dg_kcal
    assert pytest.approx(13.852, rel=0.001) == ddg_kcal


def test_exponential_average_gins(exp_data):
    """
    Test GINS calculation using ExponentialAveraging.
    The expected values for GINS: 61245350.401 ± 61251232.076 kcal/mol
    """
    u_klt, _ = exp_data
    dg_kcal, ddg_kcal = ExponentialAveraging.compute_gins(
        potential_energies=u_klt, temperature=300.0, units="kcal"
    )
    assert pytest.approx(61245350.401, rel=0.001) == dg_kcal
    assert pytest.approx(61251232.076, rel=0.001) == ddg_kcal


# ─── Bennett Acceptance Ratio (BAR, UBAR, RBAR) ────────────────────────────────
def test_bennett_acceptance_ratio_bar(bar_data):
    """
    Test BAR calculation using BennettAcceptanceRatio.
    The expected values for BAR: 7.774 ± nan kcal/mol
    """
    u_klt, _ = bar_data
    dg_kcal, ddg_kcal = BennettAcceptanceRatio.compute_bar(
        potential_energies=u_klt, temperature=300.0, units="kcal"
    )

    assert pytest.approx(7.774, rel=0.001) == dg_kcal
    assert np.isnan(ddg_kcal)  # BAR does not provide error estimate


def test_bennett_acceptance_ratio_ubar(bar_data):
    """
    Test UBAR calculation using BennettAcceptanceRatio.
    The expected values for UBAR: 4.922 ± nan kcal/mol
    """
    u_klt, _ = bar_data
    dg_kcal, ddg_kcal = BennettAcceptanceRatio.compute_ubar(
        potential_energies=u_klt, temperature=300.0, units="kcal"
    )

    assert pytest.approx(4.922, rel=0.001) == dg_kcal
    assert np.isnan(ddg_kcal)  # UBAR does not provide error estimate


def test_bennett_acceptance_ratio_rbar(bar_data):
    """
    Test RBAR calculation using BennettAcceptanceRatio.
    The expected values for RBAR: 7.661 ± nan kcal/mol
    """
    u_klt, _ = bar_data
    dg_kcal, ddg_kcal = BennettAcceptanceRatio.compute_rbar(
        potential_energies=u_klt, temperature=300.0, units="kcal"
    )

    assert pytest.approx(7.661, rel=0.001) == dg_kcal
    assert np.isnan(ddg_kcal)  # RBAR does not provide error estimate


# ─── Multistate BAR (MBAR) ─────────────────────────────────────────────────────
def test_multistate_bar_mbar(mbar_data):
    """
    Test MBAR calculation using MultistateBAR.
    The expected values for MBAR: 7.561 ± 5.762 kcal/mol
    """
    u_klt, nsnapshots = mbar_data
    result = MultistateBAR.compute_mbar(
        potential_energies=u_klt,
        num_samples_per_state=nsnapshots,
        temperature=300.0,
        units="kcal",
    )
    dg_kcal = result['free_energy']
    ddg_kcal = result['error']

    assert pytest.approx(7.561, rel=0.001) == dg_kcal
    assert pytest.approx(5.762, rel=0.001) == ddg_kcal


# ─── FreeEnergyEstimator ─────────────────────────────────────────────────────
def test_estimator_selected_methods(free_energy_estimator, lambda_data):
    """
    Test running only selected methods through FreeEnergyEstimator.
    """
    u_klt = lambda_data["u_klt"]
    nsnapshots = lambda_data["nsnapshots"]

    # Only run a subset of methods
    selected_methods = ['DEXP', 'BAR', 'MBAR']

    results = free_energy_estimator.estimate_all_methods(
        potential_energies=u_klt,
        sample_counts=nsnapshots,
        methods=selected_methods,
        regular_estimate=False,
    )

    # Should only have the requested methods
    assert len(results) == 3
    assert 'DEXP' in results
    assert 'BAR' in results
    assert 'MBAR' in results
    assert 'IEXP' not in results
    assert 'TI_trapezoidal' not in results

    # Check values
    assert pytest.approx(5.590, rel=0.001) == results['DEXP']['free_energy']
    assert pytest.approx(0.612, rel=0.001) == results['DEXP']['error']
    assert pytest.approx(7.774, rel=0.001) == results['BAR']['free_energy']
    assert np.isnan(results['BAR']['error'])  # BAR does not provide error estimate
    assert pytest.approx(7.561, rel=0.001) == results['MBAR']['free_energy']
    assert pytest.approx(5.762, rel=0.001) == results['MBAR']['error']


def test_estimator_ti_cubic(free_energy_estimator, ti_data):
    """
    Test TI cubic spline integration through FreeEnergyEstimator.
    Expected: -4.926 ± 11.621 kcal/mol
    """
    lv, ave_dhdl, std_dhdl = ti_data
    result = free_energy_estimator.estimate_ti(
        lambda_vectors=lv, ave_dhdl=ave_dhdl, std_dhdl=std_dhdl, method='cubic'
    )

    assert result['success'] is True
    assert result['method'] == 'TI_cubic'
    assert result['units'] == '(kcal/mol)'
    assert result['n_points'] == 5
    assert pytest.approx(-4.926, rel=0.001) == result['free_energy']
    assert pytest.approx(11.621, rel=0.001) == result['error']


def test_estimator_exp_dexp(free_energy_estimator, lambda_data):
    """
    Test DEXP through FreeEnergyEstimator.
    Expected: 5.590 ± 0.612 kcal/mol
    """
    u_klt = lambda_data["u_klt"]

    result = free_energy_estimator.estimate_exp(potential_energies=u_klt, method='DEXP')

    assert result['success'] is True
    assert result['method'] == 'DEXP'
    assert result['units'] == '(kcal/mol)'
    assert pytest.approx(5.590, rel=0.001) == result['free_energy']
    assert pytest.approx(0.612, rel=0.001) == result['error']


def test_estimator_exp_iexp(free_energy_estimator, lambda_data):
    """
    Test IEXP through FreeEnergyEstimator.
    Expected: 15.218 ± 0.755 kcal/mol
    """
    u_klt = lambda_data["u_klt"]

    result = free_energy_estimator.estimate_exp(potential_energies=u_klt, method='IEXP')

    assert result['success'] is True
    assert result['method'] == 'IEXP'
    assert result['units'] == '(kcal/mol)'
    assert pytest.approx(15.218, rel=0.001) == result['free_energy']
    assert pytest.approx(0.755, rel=0.001) == result['error']


def test_estimator_exp_gdel(free_energy_estimator, lambda_data):
    """
    Test GDEL through FreeEnergyEstimator.
    Expected: -3.616 ± 13.852 kcal/mol
    """
    u_klt = lambda_data["u_klt"]

    result = free_energy_estimator.estimate_exp(potential_energies=u_klt, method='GDEL')

    assert result['success'] is True
    assert result['method'] == 'GDEL'
    assert result['units'] == '(kcal/mol)'
    assert pytest.approx(-3.616, rel=0.001) == result['free_energy']
    assert pytest.approx(13.852, rel=0.001) == result['error']


def test_estimator_exp_gins(free_energy_estimator, lambda_data):
    """
    Test GINS through FreeEnergyEstimator.
    Expected: 61245350.401 ± 61251232.076 kcal/mol

    Note: GINS shows extremely large values indicating numerical instability
    for this particular dataset.
    """
    u_klt = lambda_data["u_klt"]

    result = free_energy_estimator.estimate_exp(potential_energies=u_klt, method='GINS')

    assert result['success'] is True
    assert result['method'] == 'GINS'
    assert result['units'] == '(kcal/mol)'
    # Use larger tolerance for GINS due to numerical instability
    assert pytest.approx(61245350.401, rel=0.01) == result['free_energy']
    assert pytest.approx(61251232.076, rel=0.01) == result['error']


def test_estimator_bar(free_energy_estimator, lambda_data):
    """
    Test BAR through FreeEnergyEstimator.
    Expected: 7.774 ± nan kcal/mol
    """
    u_klt = lambda_data["u_klt"]

    result = free_energy_estimator.estimate_bar(potential_energies=u_klt, method='BAR')

    assert result['success'] is True
    assert result['method'] == 'BAR'
    assert result['units'] == '(kcal/mol)'
    assert pytest.approx(7.774, rel=0.001) == result['free_energy']
    assert np.isnan(result['error'])  # BAR returns nan for error


def test_estimator_ubar(free_energy_estimator, lambda_data):
    """
    Test UBAR through FreeEnergyEstimator.
    Expected: 4.922 ± nan kcal/mol
    """
    u_klt = lambda_data["u_klt"]

    result = free_energy_estimator.estimate_bar(potential_energies=u_klt, method='UBAR')

    assert result['success'] is True
    assert result['method'] == 'UBAR'
    assert result['units'] == '(kcal/mol)'
    assert pytest.approx(4.922, rel=0.001) == result['free_energy']
    assert np.isnan(result['error'])  # UBAR returns nan for error


def test_estimator_rbar(free_energy_estimator, lambda_data):
    """
    Test RBAR through FreeEnergyEstimator.
    Expected: 7.661 ± nan kcal/mol
    """
    u_klt = lambda_data["u_klt"]

    result = free_energy_estimator.estimate_bar(potential_energies=u_klt, method='RBAR')

    assert result['success'] is True
    assert result['method'] == 'RBAR'
    assert result['units'] == '(kcal/mol)'
    assert pytest.approx(7.661, rel=0.001) == result['free_energy']
    assert np.isnan(result['error'])  # RBAR returns nan for error


def test_estimator_mbar(free_energy_estimator, lambda_data):
    """
    Test MBAR through FreeEnergyEstimator.
    Expected: 7.561 ± 5.762 kcal/mol
    """
    u_klt = lambda_data["u_klt"]
    nsnapshots = lambda_data["nsnapshots"]

    result = free_energy_estimator.estimate_mbar(
        potential_energies=u_klt, num_samples_per_state=nsnapshots
    )

    assert result['success'] is True
    assert result['method'] == 'MBAR'
    assert result['units'] == '(kcal/mol)'
    assert result['n_states'] == 5
    assert pytest.approx(7.561, rel=0.001) == result['free_energy']
    assert pytest.approx(5.762, rel=0.001) == result['error']

    # Check that we get the full MBAR result dictionary
    assert 'free_energies_all' in result
    assert 'Deltaf_ij' in result
    assert 'mbar_object' in result


def test_estimator_all_methods(free_energy_estimator, lambda_data, ti_data):
    """
    Test running all methods through FreeEnergyEstimator.estimate_all_methods().
    """
    u_klt = lambda_data["u_klt"]
    nsnapshots = lambda_data["nsnapshots"]
    lv, ave_dhdl, std_dhdl = ti_data

    results = free_energy_estimator.estimate_all_methods(
        potential_energies=u_klt,
        sample_counts=nsnapshots,
        lambda_vectors=lv,
        ave_dhdl=ave_dhdl,
        std_dhdl=std_dhdl,
    )

    # Check that all expected methods are present and succeeded
    expected_results = {
        'DEXP': 5.590,
        'IEXP': 15.218,
        'GDEL': -3.616,
        'GINS': 61245350.401,  # Large value due to numerical instability
        'BAR': 7.774,
        'UBAR': 4.922,
        'RBAR': 7.661,
        'TI_trapezoidal': -3.064,
        'TI_cubic': -4.926,
    }

    for method, expected_value in expected_results.items():
        assert method in results
        assert results[method]['success'] is True

        actual_value = results[method]['free_energy']
        assert pytest.approx(expected_value, rel=0.001) == actual_value


# ─── FreeEnergyEstimator (Error Handling Tests) ───────────────────────────────────────
def test_estimator_invalid_ti_method(free_energy_estimator, ti_data):
    """Test error handling for invalid TI method."""
    lv, ave_dhdl, std_dhdl = ti_data

    result = free_energy_estimator.estimate_ti(
        lambda_vectors=lv, ave_dhdl=ave_dhdl, std_dhdl=std_dhdl, method='invalid_method'
    )

    assert result['success'] is False
    assert result['method'] == 'TI_invalid_method'
    assert 'error_message' in result


def test_estimator_invalid_exp_method(free_energy_estimator, lambda_data):
    """Test error handling for invalid EXP method."""
    u_klt = lambda_data["u_klt"]

    result = free_energy_estimator.estimate_exp(
        potential_energies=u_klt, method='INVALID'
    )

    assert result['success'] is False
    assert result['method'] == 'INVALID'
    assert 'error_message' in result


def test_estimator_invalid_bar_method(free_energy_estimator, lambda_data):
    """Test error handling for invalid BAR method."""
    u_klt = lambda_data["u_klt"]

    result = free_energy_estimator.estimate_bar(
        potential_energies=u_klt, method='INVALID'
    )

    assert result['success'] is False
    assert result['method'] == 'INVALID'
    assert 'error_message' in result


# ─── FreeEnergyEstimator (Unit Consistency Tests) ───────────────────────────────
def test_estimator_different_units(lambda_data, ti_data):
    """Test that different unit settings work correctly."""
    _ = lambda_data["u_klt"]
    lv, ave_dhdl, std_dhdl = ti_data

    # Test with kJ units
    estimator_kj = FreeEnergyEstimator(
        temperature=300.0, units='kJ', software='Gromacs'
    )

    result_kj = estimator_kj.estimate_ti(
        lambda_vectors=lv, ave_dhdl=ave_dhdl, std_dhdl=std_dhdl, method='trapezoidal'
    )

    # Test with kBT units
    estimator_kbt = FreeEnergyEstimator(
        temperature=300.0, units='kBT', software='Gromacs'
    )

    result_kbt = estimator_kbt.estimate_ti(
        lambda_vectors=lv, ave_dhdl=ave_dhdl, std_dhdl=std_dhdl, method='trapezoidal'
    )
    # Results should be different due to unit conversion
    assert result_kj['success'] is True
    assert result_kbt['success'] is True
    assert result_kj['units'] == '(kJ/mol)'
    assert result_kbt['units'] == '(k_BT)'
    assert result_kj['free_energy'] != result_kbt['free_energy']
