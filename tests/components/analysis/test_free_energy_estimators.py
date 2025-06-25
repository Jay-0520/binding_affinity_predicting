"""
Test suite for free_energy_estimators.py functions.

Tests are based on reference values from alchemical_analysis.py output
using the same input data (fep_data.pkl).

Expected results from results.txt:
- TI: -3.064 ± 9.343 kcal/mol
- TI-CUBIC: -4.926 ± 11.621 kcal/mol
- DEXP: 5.590 ± 0.612 kcal/mol
- IEXP: 15.218 ± 0.755 kcal/mol
- BAR: 7.774 ± nan kcal/mol
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
    get_lambda_components_changing,
)


@pytest.fixture
def ti_data(lambda_data):
    """
    Prepare lambda vectors, <dH/dλ> and std(dH/dλ) (already beta‐scaled)
    for TI tests.
    """
    lv = lambda_data["lv"]  # shape (n_states, n_components)
    dhdlt = lambda_data["dhdlt"]  # shape (n_states, n_components, n_samples)

    # Boltzmann β for T=300K in kcal/mol:
    temperature = 300.0
    beta = 1.0 / (8.314472e-3 * temperature)

    # average and standard error over time‐series:
    ave_dhdl = dhdlt.mean(axis=2) * beta
    std_dhdl = dhdlt.std(axis=2, ddof=1) / np.sqrt(dhdlt.shape[2]) * beta

    return lv, ave_dhdl, std_dhdl


@pytest.fixture
def mbar_data(fep_data):
    """
    Prepare reduced potentials u_klt and snapshot counts for MBAR/EXP/BAR tests.
    """
    return fep_data["u_klt"], fep_data["nsnapshots"]


# ─── Thermodynamic Integration ──────────────────────────────────────────────────
def test_trapezoidal_integration(ti_data):
    lv, ave, std = ti_data
    dg, ddg = ThermodynamicIntegration.trapezoidal_integration(lv, ave, std)
    beta_report = calculate_beta_parameter(
        temperature=300.0, units='kcal', software='Gromacs'
    )

    dg_kcal = dg / beta_report
    ddg_kcal = ddg / beta_report

    assert pytest.approx(-3.064, rel=0.05) == dg_kcal
    assert pytest.approx(9.343, rel=0.05) == ddg_kcal


# def test_cubic_spline_integration(ti_data):
#     lv, ave, std = ti_data
#     dg, ddg = ThermodynamicIntegration.cubic_spline_integration(lv, ave, std)

#     temperature = 300.0
#     beta_report = 1.0 / (8.314472e-3 * temperature)
#     dg_kcal = dg / beta_report
#     ddg_kcal = ddg / beta_report

#     assert pytest.approx(-4.926, rel=0.05) == dg_kcal
#     assert pytest.approx(11.621, rel=0.20) == ddg_kcal
# @pytest.mark.parametrize(
#     "lv, ave, std, err_msg",
#     [
#         # mismatched ave/std shape
#         (np.array([[0.0,1.0],[0.5,1.0],[1.0,1.0]]),
#          np.array([[1.0,2.0],[1.5,2.5]]),
#          np.array([[0.1,0.2],[0.15,0.25],[0.2,0.3]]),
#          "must have same shape"),
#         # too few points
#         (np.array([[0.0,1.0]]),
#          np.array([[1.0,2.0]]),
#          np.array([[0.1,0.2]]),
#          "Need at least 2 lambda states")
#     ]
# )
# def test_ti_input_validation(lv, ave, std, err_msg):
#     with pytest.raises(ValueError, match=err_msg):
#         ThermodynamicIntegration.trapezoidal_integration(lv, ave, std)


# # ─── Natural Cubic Spline ───────────────────────────────────────────────────────

# def test_spline_creation_minimum_points():
#     xs = np.array([0.0, 0.33, 0.67, 1.0])
#     spline = NaturalCubicSpline(xs)
#     assert spline.x.shape[0] == 4
#     assert spline.wsum.shape[0] == 4

# def test_spline_integration_polynomial():
#     xs = np.linspace(0, 1, 5)
#     ys = xs**2
#     spline = NaturalCubicSpline(xs)
#     val = spline.integrate(ys)
#     # exact ∫0¹ x² dx = 1/3 ≈ 0.333
#     assert 0.2 < val < 0.6

# def test_spline_insufficient_points():
#     xs = np.array([0.0, 0.5, 1.0])
#     with pytest.raises(ValueError, match="at least 4 points"):
#         NaturalCubicSpline(xs)


# # ─── Exponential Averaging (EXP, DEXP, IEXP, GDEL, GINS) ────────────────────────

# @pytest.mark.parametrize("method", [
#     ("forward_exp",  ExponentialAveraging.forward_exp),
#     ("reverse_exp",  ExponentialAveraging.reverse_exp),
#     ("gauss_del",   ExponentialAveraging.gaussian_deletion),
#     ("gauss_ins",   ExponentialAveraging.gaussian_insertion),
# ])
# def test_exponential_methods_non_nan(mbar_data, method):
#     u_klt, _ = mbar_data
#     k = 0
#     w_F, w_R = ExponentialAveraging.calculate_work_values(u_klt, k)
#     if method[0] == "forward_exp":
#         dg, ddg = method[1](w_F, temperature=300, units="kcal", software="Gromacs")
#     elif method[0] == "reverse_exp":
#         dg, ddg = method[1](w_R, temperature=300, units="kcal", software="Gromacs")
#     else:
#         # both forward and reverse use same signature
#         dg, ddg = method[1](w_F if "del" in method[0] else w_R,
#                              temperature=300, units="kcal", software="Gromacs")
#     assert not np.isnan(dg)
#     assert not np.isnan(ddg)
#     assert ddg >= 0

# def test_work_value_calculation_errors():
#     u_klt = np.random.rand(3,3,100)
#     # invalid state
#     with pytest.raises(ValueError, match="too large"):
#         ExponentialAveraging.calculate_work_values(u_klt, 5)


# # ─── Bennett Acceptance Ratio (BAR, UBAR, RBAR) ────────────────────────────────

# @pytest.mark.parametrize("method_name, method_func", [
#     ("bar",   BennettAcceptanceRatio.bar),
#     ("ubar",  BennettAcceptanceRatio.ubar),
#     ("rbar",  BennettAcceptanceRatio.rbar),
# ])
# def test_bar_variants(mbar_data, method_name, method_func):
#     u_klt, _ = mbar_data
#     k = 0
#     w_F, w_R = BennettAcceptanceRatio.calculate_work_values(u_klt, k)
#     dg, ddg = method_func(w_F, w_R, temperature=300, units="kcal", software="Gromacs")
#     assert isinstance(dg, float)
#     # BAR’s error may legitimately be NaN
#     if method_name != "bar":
#         assert not np.isnan(ddg)


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


# # ─── Unified FreeEnergyEstimator Interface ────────────────────────────────────

# @pytest.fixture
# def estimator():
#     return FreeEnergyEstimator(temperature=300.0, units="kcal", software="Gromacs")

# def test_estimate_ti_interface(estimator, ti_data):
#     lv, ave, std = ti_data
#     out = estimator.estimate_ti(lv, ave, std, method="trapezoidal")
#     assert out["success"]
#     assert out["method"] == "TI_trapezoidal"
#     out = estimator.estimate_ti(lv, ave, std, method="cubic")
#     assert out["success"]
#     assert out["method"] == "TI_cubic"

# @pytest.mark.parametrize("exp_method", ["DEXP", "IEXP", "GDEL", "GINS"])
# def test_estimate_exp_interface(estimator, mbar_data, exp_method):
#     u_klt, _ = mbar_data
#     k = 0
#     out = estimator.estimate_exp(u_klt, k, method=exp_method)
#     assert out["success"]
#     assert out["method"] == exp_method

# @pytest.mark.parametrize("bar_method", ["BAR", "UBAR", "RBAR"])
# def test_estimate_bar_interface(estimator, mbar_data, bar_method):
#     u_klt, _ = mbar_data
#     k = 0
#     out = estimator.estimate_bar(u_klt, k, method=bar_method)
#     assert out["success"]
#     assert out["method"] == bar_method

# def test_estimate_mbar_interface(estimator, mbar_data):
#     u_klt, nsnapshots = mbar_data
#     out = estimator.estimate_mbar(u_klt, nsnapshots)
#     assert out["success"]
#     assert out["method"] == "MBAR"
#     assert "total_dg" in out and "total_error" in out

# def test_invalid_ti_method(estimator, ti_data):
#     lv, ave, std = ti_data
#     out = estimator.estimate_ti(lv, ave, std, method="INVALID")
#     assert out["success"] is False
#     assert "error_message" in out
