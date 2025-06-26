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
note that these scripts are very outdated so they must be updated properly to run with new python versions.

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

import pickle
from pathlib import Path

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
from binding_affinity_predicting.components.analysis.utils import (
    calculate_beta_parameter,
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


# ─── Exponential Averaging (EXP, DEXP, IEXP, GDEL, GINS) ────────────────────────
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


# # ─── Multistate BAR (MBAR) ─────────────────────────────────────────────────────

# def test_mbar_computation(mbar_data):
#     u_klt, nsnapshots = mbar_data
#     results = MultistateBAR.compute_mbar(
#         u_klt, nsnapshots,
#         temperature=300.0,
#         units="kcal",
#         software="Gromacs",
#         regular_estimate=True,
#         compute_overlap=True
#     )
#     # top‐level answers
#     assert pytest.approx(7.561, rel=0.10) == results["total_dg"]
#     assert pytest.approx(5.762, rel=0.30) == results["total_error"]

#     # overlap matrix
#     O = results["overlap_matrix"]
#     assert O.shape == (len(nsnapshots),)*2
#     np.testing.assert_allclose(np.diag(O), 1.0, atol=1e-3)
#     np.testing.assert_allclose(O, O.T,  atol=1e-6)

# def test_mbar_simple_estimate(mbar_data):
#     u_klt, nsnapshots = mbar_data
#     results = MultistateBAR.compute_mbar(
#         u_klt, nsnapshots,
#         temperature=300.0,
#         units="kcal",
#         software="Gromacs",
#         regular_estimate=False
#     )
#     assert results["n_states"] == len(nsnapshots)
#     assert "total_dg" in results and "total_error" in results
