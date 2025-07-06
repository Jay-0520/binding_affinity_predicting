"""
Free Energy Estimation Methods.

This module contains implementations of various free energy estimation methods
adapted from alchemical_analysis.py, including thermodynamic integration,
Bennett Acceptance Ratio (BAR), MBAR, and exponential averaging methods.

methods are adopted from this script:
https://github.com/MobleyLab/alchemical-analysis/blob/master/alchemical_analysis/alchemical_analysis.py

note that we must extract independent/uncorrelated samples for these methods to work properly.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pymbar

from binding_affinity_predicting.components.analysis.utils import (
    calculate_beta_parameter,
    get_lambda_components_changing,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _calculate_work_values_all_intervals(
    potential_energies: np.ndarray, sample_counts: Optional[np.ndarray] = None
) -> tuple[list, list]:
    """
    Calculate forward and reverse work values for all lambda intervals.

    Better to use uncorrelated potential energies to get more accurate results.

    Parameters:
    -----------
    potential_energies: np.ndarray, shape (num_lambda_states, num_lambda_states,
       max_total_snapshots)
        Reduced potential energy matrix
    sample_counts : np.ndarray, shape (num_lambda_states,), optional
        Number of samples per state. If None, inferred from non-zero entries.

    Returns:
    --------
    Tuple[list, list] : (w_F_all, w_R_all)
        Lists of work values for each interval lambda_state_i → lambda_state_i+1
    """
    num_lambda_states = potential_energies.shape[0]
    w_F_all = []
    w_R_all = []

    logger.info(f"Calculating work values for {num_lambda_states-1} lambda intervals")

    for lambda_state_i in range(num_lambda_states - 1):
        lambda_state_j = lambda_state_i + 1

        try:
            # Extract sample counts if provided
            n_samples_i = (
                sample_counts[lambda_state_i] if sample_counts is not None else None
            )
            n_samples_j = (
                sample_counts[lambda_state_j] if sample_counts is not None else None
            )

            # Calculate work values for this interval
            w_F, w_R = _calculate_work_value_single_interval(
                potential_energies,
                lambda_state_i,
                lambda_state_j,
                n_samples_i,
                n_samples_j,
            )

            # Validate results
            if len(w_F) == 0 or len(w_R) == 0:
                logger.warning(
                    f"Empty work arrays for interval {lambda_state_i}→{lambda_state_j}"
                )
                w_F_all.append(np.array([]))
                w_R_all.append(np.array([]))
            else:
                w_F_all.append(w_F)
                w_R_all.append(w_R)
                logger.debug(
                    f"Interval {lambda_state_i}→{lambda_state_j}: "
                    f"{len(w_F)} forward, {len(w_R)} reverse work values"
                )

        except Exception as e:
            logger.error(
                f"Failed to calculate work values for interval "
                f"{lambda_state_i}→{lambda_state_j}: {e}"
            )
            # Add empty arrays to maintain list structure
            w_F_all.append(np.array([]))
            w_R_all.append(np.array([]))

    logger.info(
        f"Successfully calculated work values for "
        f"{sum(1 for w_F in w_F_all if len(w_F) > 0)}/{num_lambda_states-1} intervals"
    )

    return w_F_all, w_R_all


def _calculate_work_value_single_interval(
    potential_energies: np.ndarray,
    state_i: int,
    state_j: int,
    n_samples_i: Optional[int] = None,
    n_samples_j: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate forward and reverse work values for a single lambda interval.

    This function implements the work value calculations exactly as done in
    the original alchemical_analysis.py script.

    Parameters:
    -----------
    potential_energies: np.ndarray, shape (num_lambda_states, num_lambda_states,
      max_total_snapshots)
        Reduced potential energy matrix where u_kln[k,m,n] is the reduced
        potential energy of sample n from state k evaluated at state m
    state_i : int
        Index of the initial lambda state
    state_j : int
        Index of the final lambda state
    n_samples_i : int, optional
        Number of samples from state i. If None, auto-detected from non-zero entries
    n_samples_j : int, optional
        Number of samples from state j. If None, auto-detected from non-zero entries

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (w_F, w_R) where:
        - w_F: Forward work values (state_i → state_j)
        - w_R: Reverse work values (state_j → state_i)

    Raises
    ------
    ValueError
        If state indices are invalid or no samples are found
    """
    num_lambda_states = potential_energies.shape[0]
    if state_i < 0 or state_i >= num_lambda_states:
        raise ValueError(
            f"Invalid state_i: {state_i} (must be 0 <= state_i < {num_lambda_states})"
        )
    if state_j < 0 or state_j >= num_lambda_states:
        raise ValueError(
            f"Invalid state_j: {state_j} (must be 0 <= state_j < {num_lambda_states})"
        )

    # Auto-detect sample counts if not provided
    if n_samples_i is None:
        # Count non-zero entries in diagonal (assuming 0 means no data)
        # Note: Self-energies should never legitimately be zero in real simulations
        n_samples_i = np.sum(potential_energies[state_i, state_i, :] != 0)

    if n_samples_j is None:
        n_samples_j = np.sum(potential_energies[state_j, state_j, :] != 0)

    # Validate sample counts
    if n_samples_i <= 0:
        raise ValueError(f"No samples found for state {state_i}")
    if n_samples_j <= 0:
        raise ValueError(f"No samples found for state {state_j}")

    # Check array bounds
    max_samples = potential_energies.shape[2]
    if n_samples_i > max_samples:
        raise ValueError(
            f"n_samples_i ({n_samples_i}) exceeds array size ({max_samples})"
        )
    if n_samples_j > max_samples:
        raise ValueError(
            f"n_samples_j ({n_samples_j}) exceeds array size ({max_samples})"
        )

    # Calculate forward work: samples from state_i evaluated at states i and j
    w_F = (
        potential_energies[state_i, state_j, :n_samples_i]
        - potential_energies[state_i, state_i, :n_samples_i]
    )

    # Calculate reverse work: samples from state_j evaluated at states j and i
    w_R = (
        potential_energies[state_j, state_i, :n_samples_j]
        - potential_energies[state_j, state_j, :n_samples_j]
    )

    return w_F, w_R


def _validate_potential_energies(
    potential_energies: np.ndarray, sample_counts: Optional[np.ndarray] = None
) -> None:
    """
    Validate potential energy matrix and sample counts.

    Parameters:
    -----------
    potential_energies : np.ndarray, shape (num_lambda_states, num_lambda_states,
       max_total_snapshots)
        Reduced potential energy matrix
    sample_counts : np.ndarray, shape (num_lambda_states,), optional
        Number of samples per state
        The potential_energies array is pre-allocated to accommodate the maximum number of snapshots
        across all states, but not every state necessarily has that many samples.
        we might need to keep track of how many samples we actually have for each state.
    """
    if potential_energies.ndim != 3:
        raise ValueError(
            "potential_energies must be 3D array (num_lambda_states, num_lambda_states, "
            "max_total_snapshots)"
        )

    num_lambda_states = potential_energies.shape[0]
    if potential_energies.shape[1] != num_lambda_states:
        raise ValueError("potential_energies must be square in first two dimensions")

    if num_lambda_states < 2:
        raise ValueError("Need at least 2 lambda states")

    if sample_counts is not None:
        if len(sample_counts) != num_lambda_states:
            raise ValueError("sample_counts must have same length as number of states")
        if np.any(sample_counts < 0):
            raise ValueError("sample_counts must be non-negative")


class NaturalCubicSpline:
    """
    Natural cubic spline implementation adapted from alchemical_analysis.py.
    Used for thermodynamic integration with cubic spline interpolation.
    """

    def __init__(self, lambdas_per_component: np.ndarray):
        """
        Initialize natural cubic spline.
        Adapted from alchemical_analysis.py naturalcubicspline class (line ~891).

        Parameters:
        -----------
        lambdas_per_component : np.ndarray, shape (num_lambda_states,)
            Lambda values for a single component (e.g., lv[:,j] for component j)
            These represent one component of the alchemical transformation pathway
            Variable name matches original: x in naturalcubicspline(x)
        """
        # Match original: accept any number of points ≥ 2
        if len(lambdas_per_component) < 2:
            raise ValueError("Natural cubic spline requires at least 2 points")

        self.x = lambdas_per_component.copy()  # Keep original variable name 'x'
        L = len(lambdas_per_component)

        # Initialize matrices (matches original exactly)
        H = np.zeros([L, L], float)
        M = np.zeros([L, L], float)

        h = (
            lambdas_per_component[1:L] - lambdas_per_component[0 : L - 1]
        )  # differences between consecutive lambda values
        ih = 1.0 / h  # inverse differences

        # Build H and M matrices (from Chapra "Applied Numerical Methods")
        H[0, 0] = 1
        H[L - 1, L - 1] = 1

        for i in range(1, L - 1):
            H[i, i] = 2 * (h[i - 1] + h[i])
            H[i, i - 1] = h[i - 1]
            H[i, i + 1] = h[i]

            M[i, i] = -3 * (ih[i - 1] + ih[i])
            M[i, i - 1] = 3 * ih[i - 1]
            M[i, i + 1] = 3 * ih[i]

        # Solve for coefficient weight matrix
        CW = np.dot(np.linalg.inv(H), M)

        # Calculate coefficient matrices for spline segments
        BW = np.zeros([L - 1, L], float)
        DW = np.zeros([L - 1, L], float)
        AW = np.zeros([L - 1, L], float)

        for i in range(L - 1):
            BW[i, :] = -(h[i] / 3) * (2 * CW[i, :] + CW[i + 1, :])
            BW[i, i] += -ih[i]
            BW[i, i + 1] += ih[i]

            DW[i, :] = (ih[i] / 3) * (CW[i + 1, :] - CW[i, :])
            AW[i, i] = 1

        self.AW = AW.copy()
        self.BW = BW.copy()
        self.CW = CW.copy()
        self.DW = DW.copy()

        # Calculate integration weights (matches original wsum calculation)
        self.wsum = np.zeros([L], float)
        self.wk = np.zeros([L - 1, L], float)

        for k in range(L - 1):
            w = (
                DW[k, :] * (h[k] ** 4) / 4.0
                + CW[k, :] * (h[k] ** 3) / 3.0
                + BW[k, :] * (h[k] ** 2) / 2.0
                + AW[k, :] * h[k]
            )
            self.wk[k, :] = w
            self.wsum += w

    def integrate(self, y: np.ndarray) -> float:
        """
        Integrate y over the spline domain.
        Matches original usage: numpy.dot(cubspl[j].wsum, ave_dhdl[lj,j])

        Parameters:
        -----------
        y : np.ndarray, shape (num_lambda_states,)
            Y values at spline nodes (ave_dhdl values in original)

        Returns:
        --------
        float : Integral value (free energy difference)
        """
        if len(y) != len(self.x):
            raise ValueError("y must have same length as x (lambda values)")
        return np.dot(self.wsum, y)

    def interpolate(self, y: np.ndarray, xnew: np.ndarray) -> np.ndarray:
        """
        Interpolate y at new x points.
        Adapted from original interpolate method (line ~941).

        Parameters:
        -----------
        y : np.ndarray, shape (num_lambda_states,)
            Y values at spline nodes
        xnew : np.ndarray, shape (num_new_points,)
            New x coordinates for interpolation

        Returns:
        --------
        np.ndarray, shape (num_new_points,) : Interpolated y values
        """
        if len(y) != len(self.x):
            raise ValueError("y must have same length as x (lambda values)")

        # Get spline coefficients (matches original exactly)
        a = np.dot(self.AW, y)
        b = np.dot(self.BW, y)
        c = np.dot(self.CW, y)
        d = np.dot(self.DW, y)

        N = len(xnew)
        ynew = np.zeros([N], float)

        for i in range(N):
            # Find the index of 'xnew[i]' it would have in 'self.x' (matches original)
            j = np.searchsorted(self.x, xnew[i]) - 1
            lamw = xnew[i] - self.x[j]  # Original variable name
            ynew[i] = d[j] * lamw**3 + c[j] * lamw**2 + b[j] * lamw + a[j]

        # Preserve the terminal points (matches original)
        ynew[0] = y[0]
        ynew[-1] = y[-1]

        return ynew


class ThermodynamicIntegration:
    """
    Thermodynamic Integration analysis adapted from alchemical_analysis.py.
    Implements both trapezoidal and cubic spline integration methods.

    Deriving total free energy differences by integrating dH/dλ over three components in GROMACS
    FEP simulations.

    Note that class does handle unit conversion directly.
    """

    @staticmethod
    def _validate_inputs(
        lambda_vectors: np.ndarray, ave_dhdl: np.ndarray, std_dhdl: np.ndarray
    ) -> None:
        """Validate input arrays have consistent shapes."""
        if (
            lambda_vectors.shape != ave_dhdl.shape
            or lambda_vectors.shape != std_dhdl.shape
        ):
            raise ValueError(
                "lambda_vectors, ave_dhdl, and std_dhdl must have same shape"
            )
        if len(lambda_vectors) < 2:
            raise ValueError("Need at least 2 lambda states for integration")

    @staticmethod
    def trapezoidal_integration(
        lambda_vectors: np.ndarray,
        ave_dhdl: np.ndarray,
        std_dhdl: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kcal',
        software: str = 'Gromacs',
    ) -> tuple[float, float]:
        """
        Trapezoidal integration with error propagation for total free energy across all components.
        Adapted from alchemical_analysis.py TI implementation.

        Original code:
        df['TI'] = 0.5*numpy.dot(dlam[k],(ave_dhdl[k]+ave_dhdl[k+1]))
        ddf['TI'] = 0.5*numpy.sqrt(numpy.dot(dlam[k]**2,std_dhdl[k]**2+std_dhdl[k+1]**2))

        Parameters:
        -----------
        lambda_vectors : np.ndarray, shape (num_lambda_states, num_components)
            Lambda parameter values for each state and component.
            Example: [[1.0, 1.0], [0.8, 1.0], [0.6, 1.0], ...] for coulomb+vdw transformation
        ave_dhdl : np.ndarray, shape (num_lambda_states, num_components)
            Mean dH/dλ values at each lambda state (already multiplied by beta in original)
        std_dhdl : np.ndarray, shape (num_lambda_states, num_components)
            Standard error estimates for dH/dλ at each lambda state (already multiplied by beta)
        temperature : float, default 298.15
            Temperature in Kelvin
        units : str, default 'kcal'
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str, default 'Gromacs'
            Software package name

        Returns:
        --------
        Tuple[float, float] : (integrated_total_free_energy_difference, propagated_total_error)
            Both in specified physical units
        """
        ThermodynamicIntegration._validate_inputs(lambda_vectors, ave_dhdl, std_dhdl)

        # Calculate differences between consecutive lambda states
        dlam = np.diff(lambda_vectors, axis=0)  # shape (n_states-1, n_components)

        # Trapezoidal rule: ½ * dλ * (f(λₖ) + f(λₖ₊₁))
        df_contributions = 0.5 * np.sum(dlam * (ave_dhdl[:-1] + ave_dhdl[1:]), axis=1)
        # Sum over all lambda intervals
        total_df_reduced = np.sum(df_contributions)

        # Error calculation: match totalEnergies() in alchemical_analysis.py
        total_error_variance = 0.0
        # Determine which components change
        lchange = get_lambda_components_changing(lambda_vectors)

        # For each component j
        for j in range(lambda_vectors.shape[1]):
            lj = lchange[:, j]  # States where component j changes

            if np.any(lj):  # If component j changes
                # Get non-zero intervals for this component
                dlam_j = np.diff(lambda_vectors[:, j])
                h = dlam_j[dlam_j != 0]  # Remove zeros like numpy.trim_zeros

                if len(h) > 0:
                    # Trapezoidal weights: 0.5*(append(h,0) + append(0,h))
                    wsum = 0.5 * (np.append(h, 0) + np.append(0, h))
                    # Accumulate variance: dot(wsum**2, std_dhdl[lj,j]**2)
                    std_j = std_dhdl[lj, j]
                    total_error_variance += np.dot(wsum**2, std_j**2)

        total_error_reduced = np.sqrt(total_error_variance)

        # Convert from reduced units to physical units
        beta_report = calculate_beta_parameter(temperature, units, software)
        total_df_physical = total_df_reduced / beta_report
        total_error_physical = total_error_reduced / beta_report

        return float(total_df_physical), float(total_error_physical)

    @staticmethod
    def cubic_spline_integration(
        lambda_vectors: np.ndarray,
        ave_dhdl: np.ndarray,
        std_dhdl: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kcal',
        software: str = 'Gromacs',
    ) -> tuple[float, float]:
        """
        Cubic spline integration for total free energy across all components.
        Adapted from alchemical_analysis.py TI-CUBIC implementation.

        Original algorithm (lines ~1046-1051):
        - For each component j where dlam[k,j] > 0:
        - df['TI-CUBIC'] += numpy.dot(cubspl[j].wsum, ave_dhdl[lj,j])
        - ddf['TI-CUBIC'] += numpy.dot(cubspl[j].wk[mapl[k,j]]**2, std_dhdl[lj,j]**2)

        Parameters:
        -----------
        lambda_vectors : np.ndarray, shape (num_lambda_states, num_components)
        ave_dhdl : np.ndarray, shape (num_lambda_states, num_components)
            Mean dH/dλ values (already multiplied by beta in original)
        std_dhdl : np.ndarray, shape (num_lambda_states, num_components)
            Standard error estimates (already multiplied by beta in original)
        temperature : float, default 298.15
            Temperature in Kelvin
        units : str, default 'kcal'
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str, default 'Gromacs'
            Software package name

        Returns:
        --------
        Tuple[float, float] : (integrated_total_free_energy_difference, propagated_total_error)
            Both in specified physical units
        """
        num_lambda_states, num_components = lambda_vectors.shape

        if num_lambda_states < 4:
            logger.warning("Cubic spline needs ≥4 points, falling back to trapezoidal")
            return ThermodynamicIntegration.trapezoidal_integration(
                lambda_vectors, ave_dhdl, std_dhdl
            )

        components_changing = get_lambda_components_changing(lambda_vectors)

        total_df_reduced = 0.0
        total_error_variance = 0.0
        for j in range(num_components):
            lj = components_changing[:, j]

            if not np.any(lj):
                logger.warning(f"Component {j} does not change in any state, skipping.")
                continue

            # Extract lambda values where component j changes (like original)
            lv_lchange = lambda_vectors[lj, j]
            ave_j = ave_dhdl[lj, j]
            std_j = std_dhdl[lj, j]

            # Create spline regardless of number of points (like original)
            try:
                spline = NaturalCubicSpline(lv_lchange)
                df_j = np.dot(spline.wsum, ave_j)
                ddf_j_sq = np.dot(spline.wsum**2, std_j**2)

                total_df_reduced += df_j
                total_error_variance += ddf_j_sq

            except Exception as e:
                logger.warning(f"Spline creation failed for component {j}: {e}")
                continue

        total_error_reduced = np.sqrt(total_error_variance)

        # Convert from reduced units to physical units
        beta_report = calculate_beta_parameter(temperature, units, software)
        total_df_physical = total_df_reduced / beta_report
        total_error_physical = total_error_reduced / beta_report

        return float(total_df_physical), float(total_error_physical)


class ExponentialAveraging:
    """
    Exponential averaging methods adapted from alchemical_analysis.py.

    Refactored to process full potential energy matrices and return total
    free energy changes across all lambda intervals.

    Implements DEXP, IEXP, GDEL, and GINS methods exactly as in original.

    Note that class does handle unit conversion directly.
    """

    @staticmethod
    def _calculate_interval_free_energy(
        work_values: np.ndarray,
        method: str,
        temperature: float = 298.15,
        units: str = 'kcal',
        software: str = 'Gromacs',
        negate_result: bool = False,
    ) -> tuple[float, float]:
        """
        Calculate free energy for a single interval using specified method.

        Parameters:
        -----------
        work_values : np.ndarray
            Work values for this interval (already in reduced units)
        method : str
            Method name: 'DEXP', 'IEXP', 'GDEL', 'GINS'
        temperature : float
            Temperature in Kelvin
        units : str
            Output units
        software : str
            Software package name
        negate_result : bool
            Whether to negate the result (for reverse methods)

        Returns:
        --------
        Tuple[float, float] : (free_energy, error) in specified physical units
        """
        if len(work_values) == 0:
            return 0.0, 0.0

        # Calculate beta_report for unit conversion
        beta_report = calculate_beta_parameter(temperature, units, software)

        try:
            if method in ['DEXP', 'IEXP']:
                # Use standard exponential averaging
                import pymbar.other_estimators

                results = pymbar.other_estimators.exp(work_values)
                df_reduced = results['Delta_f']
                ddf_reduced = results['dDelta_f']

            elif method in ['GDEL', 'GINS']:
                # Use Gaussian exponential averaging
                import pymbar.other_estimators

                results = pymbar.other_estimators.exp_gauss(work_values)
                df_reduced = results['Delta_f']
                ddf_reduced = results['dDelta_f']

            else:
                raise ValueError(f"Unknown method: {method}")

            # Apply negation for reverse methods
            if negate_result:
                df_reduced = -df_reduced

            # Convert to physical units
            df_physical = df_reduced / beta_report
            ddf_physical = ddf_reduced / beta_report

            return df_physical, ddf_physical

        except Exception as e:
            logger.warning(
                f"pymbar {method} failed: {e}, using simple exponential averaging"
            )
            raise

    @staticmethod
    def compute_dexp(
        potential_energies: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kcal',
        software: str = 'Gromacs',
        sample_counts: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """
        Compute total DEXP (forward exponential averaging) across all lambda intervals.

        Parameters:
        -----------
        potential_energies : np.ndarray, shape (num_lambda_states, num_lambda_states,
          max_total_snapshots)
            Reduced potential energy matrix
        sample_counts : np.ndarray, shape (num_lambda_states,), optional
            Number of samples per state
        temperature : float, default 298.15
            Temperature in Kelvin
        units : str, default 'kcal'
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str, default 'Gromacs'
            Software package name

        Returns:
        --------
        Tuple[float, float] : (total_free_energy, total_error) in specified units
        """
        _validate_potential_energies(potential_energies, sample_counts)

        # Calculate work values for all intervals
        w_F_all, _ = _calculate_work_values_all_intervals(
            potential_energies, sample_counts
        )

        # Calculate free energy for each interval
        total_dg = 0.0
        total_error_variance = 0.0

        for lambda_state_i, w_F in enumerate(w_F_all):
            if len(w_F) == 0:
                continue

            dg_interval, ddg_interval = (
                ExponentialAveraging._calculate_interval_free_energy(
                    w_F, 'DEXP', temperature, units, software, negate_result=False
                )
            )

            total_dg += dg_interval
            total_error_variance += ddg_interval**2

        total_error = np.sqrt(total_error_variance)
        return total_dg, total_error

    @staticmethod
    def compute_iexp(
        potential_energies: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kcal',
        software: str = 'Gromacs',
        sample_counts: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """
        Compute total IEXP (reverse exponential averaging) across all lambda intervals.

        Parameters:
        -----------
        potential_energies : np.ndarray, shape (num_lambda_states, num_lambda_states,
          max_total_snapshots)
            Reduced potential energy matrix
        sample_counts : np.ndarray, shape (num_lambda_states,), optional
            Number of samples per state
        temperature : float, default 298.15
            Temperature in Kelvin
        units : str, default 'kcal'
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str, default 'Gromacs'
            Software package name

        Returns:
        --------
        Tuple[float, float] : (total_free_energy, total_error) in specified units
        """
        _validate_potential_energies(potential_energies, sample_counts)

        # Calculate work values for all intervals
        _, w_R_all = _calculate_work_values_all_intervals(
            potential_energies, sample_counts
        )

        # Calculate free energy for each interval
        total_dg = 0.0
        total_error_variance = 0.0

        for lambda_state_i, w_R in enumerate(w_R_all):
            if len(w_R) == 0:
                continue

            dg_interval, ddg_interval = (
                ExponentialAveraging._calculate_interval_free_energy(
                    w_R, 'IEXP', temperature, units, software, negate_result=True
                )
            )

            total_dg += dg_interval
            total_error_variance += ddg_interval**2

        total_error = np.sqrt(total_error_variance)
        return total_dg, total_error

    @staticmethod
    def compute_gdel(
        potential_energies: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kcal',
        software: str = 'Gromacs',
        sample_counts: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """
        Compute total GDEL (Gaussian deletion) across all lambda intervals.

        Parameters and returns same as compute_dexp.
        """
        _validate_potential_energies(potential_energies, sample_counts)

        w_F_all, _ = _calculate_work_values_all_intervals(
            potential_energies, sample_counts
        )

        total_dg = 0.0
        total_error_variance = 0.0

        for lambda_state_i, w_F in enumerate(w_F_all):
            if len(w_F) == 0:
                continue

            dg_interval, ddg_interval = (
                ExponentialAveraging._calculate_interval_free_energy(
                    w_F, 'GDEL', temperature, units, software, negate_result=False
                )
            )

            total_dg += dg_interval
            total_error_variance += ddg_interval**2

        total_error = np.sqrt(total_error_variance)
        return total_dg, total_error

    @staticmethod
    def compute_gins(
        potential_energies: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kcal',
        software: str = 'Gromacs',
        sample_counts: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """
        Compute total GINS (Gaussian insertion) across all lambda intervals.

        Parameters and returns same as compute_iexp.
        """
        _validate_potential_energies(potential_energies, sample_counts)

        _, w_R_all = _calculate_work_values_all_intervals(
            potential_energies, sample_counts
        )

        total_dg = 0.0
        total_error_variance = 0.0

        for lambda_state_i, w_R in enumerate(w_R_all):
            if len(w_R) == 0:
                continue

            dg_interval, ddg_interval = (
                ExponentialAveraging._calculate_interval_free_energy(
                    w_R, 'GINS', temperature, units, software, negate_result=True
                )
            )

            total_dg += dg_interval
            total_error_variance += ddg_interval**2

        total_error = np.sqrt(total_error_variance)
        return total_dg, total_error


class BennettAcceptanceRatio:
    """
    Bennett Acceptance Ratio methods adapted from alchemical_analysis.py.
    Implements BAR, UBAR, and RBAR methods exactly as in original.
    """

    @staticmethod
    def _calculate_interval_free_energy(
        w_F: np.ndarray,
        w_R: np.ndarray,
        method: str,
        temperature: float = 298.15,
        units: str = 'kJ',
        software: str = 'Gromacs',
        relative_tolerance: float = 1e-10,
        verbose: bool = False,
        trial_range: tuple = (-10, 10),
    ) -> tuple[float, float]:
        """
        Calculate free energy for a single interval using specified BAR method.

        Parameters:
        -----------
        w_F : np.ndarray
            Forward work values: w_F = u_kln[k,k+1,:] - u_kln[k,k,:]
            These should already be in reduced units!
        w_R : np.ndarray
            Reverse work values: w_R = u_kln[k+1,k,:] - u_kln[k+1,k+1,:]
            These should already be in reduced units!
        method : str
            Method to use: 'BAR', 'UBAR', or 'RBAR'
        temperature : float
            Temperature in Kelvin (for unit conversion only)
        units : str
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str
            Software package name
        relative_tolerance : float
            Convergence tolerance for BAR iteration
        verbose : bool
            Enable verbose output
        trial_range : tuple
            Range of trial free energy values for RBAR (in reduced units)

        Returns:
        --------
        Tuple[float, float] : (free_energy, error) in specified units
        """
        if len(w_F) == 0 or len(w_R) == 0:
            return 0.0, 0.0

        # Calculate beta_report for unit conversion
        beta_report = calculate_beta_parameter(temperature, units, software)

        try:
            if method == 'BAR':
                # Standard BAR with iteration
                results = pymbar.other_estimators.bar(
                    w_F, w_R, relative_tolerance=relative_tolerance, verbose=verbose
                )
                df_reduced = results['Delta_f']
                ddf_reduced = results['dDelta_f']

            elif method == 'UBAR':
                # Unoptimized BAR - assume dF is zero, just do one evaluation
                results = pymbar.other_estimators.bar(
                    w_F,
                    w_R,
                    verbose=verbose,
                    iterated_solution=False,  # Key difference from BAR
                )
                df_reduced = results['Delta_f']
                ddf_reduced = results['dDelta_f']

            elif method == 'RBAR':
                # Range-based BAR - test multiple trial values
                min_diff = 1e6
                best_udf = 0
                best_uddf = 0

                # Test trial free energies in the specified range
                for trial_udf in range(trial_range[0], trial_range[1], 1):
                    try:
                        # Calculate UBAR with this trial free energy as initial guess
                        results = pymbar.other_estimators.bar(
                            w_F,
                            w_R,
                            verbose=verbose,
                            iterated_solution=False,
                            DeltaF=float(trial_udf),
                        )
                        udf = results['Delta_f']
                        uddf = results['dDelta_f']
                        # Check how well this satisfies the BAR equation
                        diff = abs(udf - trial_udf)
                        if diff < min_diff:
                            min_diff = diff
                            best_udf = udf
                            best_uddf = uddf

                    except Exception as e:
                        # Skip this trial value if it fails
                        if verbose:
                            logger.debug(f"RBAR trial {trial_udf} failed: {e}")
                        continue

                if min_diff == 1e6:
                    # All trials failed, fall back to standard BAR
                    logger.warning("All RBAR trials failed, falling back to BAR")
                    results = pymbar.other_estimators.bar(
                        w_F, w_R, relative_tolerance=relative_tolerance, verbose=verbose
                    )
                    (df_reduced, ddf_reduced) = results['Delta_f'], results['dDelta_f']
                else:
                    df_reduced = best_udf
                    ddf_reduced = best_uddf

            else:
                raise ValueError(f"Unknown BAR method: {method}")

            # Convert to physical units
            df_physical = df_reduced / beta_report
            ddf_physical = ddf_reduced / beta_report

            return df_physical, ddf_physical

        except Exception as e:
            logger.error(f"pymbar {method} failed: {e}")
            raise

    @staticmethod
    def compute_bar(
        potential_energies: np.ndarray,
        sample_counts: Optional[np.ndarray] = None,
        temperature: float = 298.15,
        units: str = 'kJ',
        software: str = 'Gromacs',
        relative_tolerance: float = 1e-10,
        verbose: bool = False,
    ) -> tuple[float, float]:
        """
        Compute total BAR estimate across all lambda intervals.

        Parameters:
        -----------
        potential_energies : np.ndarray, shape (num_lambda_states, num_lambda_states,
         max_total_snapshots)
            Reduced potential energy matrix
        sample_counts : np.ndarray, shape (num_lambda_states,), optional
            Number of samples per state
        temperature : float, default 298.15
            Temperature in Kelvin
        units : str, default 'kJ'
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str, default 'Gromacs'
            Software package name
        relative_tolerance : float, default 1e-10
            Convergence tolerance for BAR iteration
        verbose : bool, default False
            Enable verbose output

        Returns:
        --------
        Tuple[float, float] : (total_free_energy, total_error) in specified units
        """
        _validate_potential_energies(potential_energies, sample_counts)

        # Calculate work values for all intervals
        w_F_all, w_R_all = _calculate_work_values_all_intervals(
            potential_energies, sample_counts
        )

        # Calculate free energy for each interval
        total_dg = 0.0
        total_error_variance = 0.0

        for lambda_state_i, (w_F, w_R) in enumerate(zip(w_F_all, w_R_all)):
            if len(w_F) == 0 or len(w_R) == 0:
                logger.warning(
                    f"No work values for interval {lambda_state_i}→{lambda_state_i+1}"
                )
                continue

            dg_interval, ddg_interval = (
                BennettAcceptanceRatio._calculate_interval_free_energy(
                    w_F,
                    w_R,
                    'BAR',
                    temperature,
                    units,
                    software,
                    relative_tolerance,
                    verbose,
                )
            )

            total_dg += dg_interval
            total_error_variance += ddg_interval**2

        total_error = np.sqrt(total_error_variance)
        return total_dg, total_error

    @staticmethod
    def compute_ubar(
        potential_energies: np.ndarray,
        sample_counts: Optional[np.ndarray] = None,
        temperature: float = 298.15,
        units: str = 'kJ',
        software: str = 'Gromacs',
        verbose: bool = False,
    ) -> tuple[float, float]:
        """
        Compute total UBAR estimate across all lambda intervals.

        UBAR assumes dF is zero and does only one evaluation (no iteration).

        Parameters same as compute_bar (excluding relative_tolerance).
        Returns same as compute_bar.
        """
        _validate_potential_energies(potential_energies, sample_counts)

        # Calculate work values for all intervals
        w_F_all, w_R_all = _calculate_work_values_all_intervals(
            potential_energies, sample_counts
        )

        # Calculate free energy for each interval
        total_dg = 0.0
        total_error_variance = 0.0

        for lambda_state_i, (w_F, w_R) in enumerate(zip(w_F_all, w_R_all)):
            if len(w_F) == 0 or len(w_R) == 0:
                logger.warning(
                    f"No work values for interval {lambda_state_i}→{lambda_state_i+1}"
                )
                continue

            dg_interval, ddg_interval = (
                BennettAcceptanceRatio._calculate_interval_free_energy(
                    w_F, w_R, 'UBAR', temperature, units, software, verbose=verbose
                )
            )

            total_dg += dg_interval
            total_error_variance += ddg_interval**2

        total_error = np.sqrt(total_error_variance)
        return total_dg, total_error

    @staticmethod
    def compute_rbar(
        potential_energies: np.ndarray,
        sample_counts: Optional[np.ndarray] = None,
        temperature: float = 298.15,
        units: str = 'kJ',
        software: str = 'Gromacs',
        verbose: bool = False,
        trial_range: tuple = (-10, 10),
    ) -> tuple[float, float]:
        """
        Compute total RBAR estimate across all lambda intervals.

        RBAR calculates UBAR for a series of 'trial' free energies and chooses
        the one that best satisfies the equations.

        Parameters:
        -----------
        Same as compute_bar plus:
        trial_range : tuple, default (-10, 10)
            Range of trial free energy values to test (in reduced units)

        Returns same as compute_bar.
        """
        _validate_potential_energies(potential_energies, sample_counts)

        # Calculate work values for all intervals
        w_F_all, w_R_all = _calculate_work_values_all_intervals(
            potential_energies, sample_counts
        )

        # Calculate free energy for each interval
        total_dg = 0.0
        total_error_variance = 0.0

        for lambda_state_i, (w_F, w_R) in enumerate(zip(w_F_all, w_R_all)):
            if len(w_F) == 0 or len(w_R) == 0:
                logger.warning(
                    f"No work values for interval {lambda_state_i}→{lambda_state_i+1}"
                )
                continue

            dg_interval, ddg_interval = (
                BennettAcceptanceRatio._calculate_interval_free_energy(
                    w_F,
                    w_R,
                    'RBAR',
                    temperature,
                    units,
                    software,
                    verbose=verbose,
                    trial_range=trial_range,
                )
            )

            total_dg += dg_interval
            total_error_variance += ddg_interval**2

        total_error = np.sqrt(total_error_variance)
        return total_dg, total_error


class MultistateBAR:
    """
    Multistate Bennett Acceptance Ratio implementation.

    Faithfully adapted from alchemical_analysis.py MBAR functionality.
    """

    @staticmethod
    def _estimatewithMBAR_core(
        potential_energies: np.ndarray,
        num_samples_per_state: np.ndarray,
        beta_report: float,
        relative_tolerance: float = 1e-10,
        regular_estimate: bool = False,
        verbose: bool = False,
        initialize: str = 'BAR',
    ) -> tuple:
        """
        Core MBAR estimation function - direct adaptation from alchemical_analysis.py.

        This implements the exact `estimatewithMBAR` logic from the original code
        but uses modern parameter naming.

        Parameters:
        -----------
        potential_energies : np.ndarray, shape (K, K, max_N)
            Reduced potential energy matrix where [k,l,n] is the reduced potential
            of snapshot n from state k evaluated at state l
        num_samples_per_state : np.ndarray, shape (K,)
            Number of samples from each state k
        relative_tolerance : float, default 1e-10
            Relative tolerance for MBAR convergence
        regular_estimate : bool, default False
            If True, return full matrices; if False, return only endpoint difference
        verbose : bool, default False
            Enable verbose pymbar output
        initialize : str, default 'BAR'
            MBAR initialization method ('BAR' or 'zeros')
        beta_report : float, optional
            Unit conversion factor from original alchemical_analysis.py

        Returns:
        --------
        If regular_estimate=True: (Deltaf_ij, dDeltaf_ij, mbar_object)
            Full free energy difference matrices and MBAR object
        If regular_estimate=False: (total_dg, total_error)
            Single endpoint free energy difference and error
        """
        try:
            # Initialize MBAR exactly as in original (using original parameter names internally)
            MBAR = pymbar.mbar.MBAR(
                potential_energies,
                num_samples_per_state,
                verbose=verbose,
                relative_tolerance=relative_tolerance,
                initialize=initialize,
            )

            # Get free energy differences exactly as in original
            # note that theta_ij (marked as "_") is never used in the original code - by JJH-2025-06-24  # noqa: E501
            results = MBAR.compute_free_energy_differences(
                uncertainty_method='svd-ew', return_theta=True
            )
            Deltaf_ij = results['Delta_f']
            dDeltaf_ij = results['dDelta_f']
            if verbose:
                logger.info(
                    f"Matrix of free energy differences\nDeltaf_ij:\n{Deltaf_ij}\ndDeltaf_ij:\n{dDeltaf_ij}"  # noqa: E501
                )

            # Return format matches original exactly
            if regular_estimate:
                return (Deltaf_ij, dDeltaf_ij, MBAR)
            else:
                K = len(num_samples_per_state)
                return (
                    Deltaf_ij[0, K - 1] / beta_report,
                    dDeltaf_ij[0, K - 1] / beta_report,
                )

        except Exception as e:
            logger.error(f"MBAR calculation failed: {e}")
            raise

    @staticmethod
    def compute_mbar(
        potential_energies: np.ndarray,
        num_samples_per_state: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kJ',
        software: str = 'Gromacs',
        regular_estimate: bool = True,
        **kwargs,
    ) -> dict:
        """
        Compute MBAR estimates for all states.
        Adapted from alchemical_analysis.py estimatewithMBAR function.

        Parameters:
        -----------
        potential_energies : np.ndarray, shape (num_lambda_states, num_lambda_states,
        max_num_snapshots)
            Reduced potential energy matrix where:
            - First index (k): lambda state where snapshots were generated
            - Second index (l): lambda state where energy is evaluated
            - Third index (n): snapshot/time index
            Example: potential_energies[1, 3, :] = energies of λ₁ snapshots evaluated at λ₃
        num_samples_per_state : np.ndarray, shape (num_lambda_states,)
            Number of samples from each lambda state
        temperature : float, default 298.15
            Temperature in Kelvin
        units : str, default 'kJ'
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str, default 'Gromacs'
            Software package name (affects unit handling)
        **kwargs : dict
            Additional MBAR parameters (relative_tolerance, verbose, initialize)

        Returns:
        --------
        Dict : MBAR results including free energies and errors
            All energies converted according to specified units
            - 'total_dg': Total free energy change (kJ/mol)
            - 'total_error': Total error estimate (kJ/mol)
            - 'free_energies': Free energies relative to first state (kJ/mol)
            - 'free_energy_errors': Error estimates for each state (kJ/mol)
            - 'Deltaf_ij': Pairwise free energy differences matrix (kJ/mol)
            - 'dDeltaf_ij': Error matrix for pairwise differences (kJ/mol)
            - 'theta_ij': Theta matrix from MBAR
            - 'mbar_object': The pymbar MBAR object
            - 'n_states': Number of lambda states
        """
        # Extract parameters with original defaults
        relative_tolerance = kwargs.get('relative_tolerance', 1e-10)
        verbose = kwargs.get('verbose', False)
        initialize = kwargs.get('initialize', 'BAR')
        compute_overlap = kwargs.get('compute_overlap', False)

        # Calculate beta parameters exactly as in original
        beta_report = calculate_beta_parameter(
            temperature=temperature, units=units, software=software
        )

        try:
            if regular_estimate:
                Deltaf_ij, dDeltaf_ij, mbar_object = (
                    MultistateBAR._estimatewithMBAR_core(
                        potential_energies=potential_energies,
                        num_samples_per_state=num_samples_per_state,
                        relative_tolerance=relative_tolerance,
                        regular_estimate=regular_estimate,
                        verbose=verbose,
                        initialize=initialize,
                        beta_report=beta_report,
                    )
                )

                # Convert to physical units using proper beta_report
                free_energies = Deltaf_ij[0, :] / beta_report
                free_energy_errors = dDeltaf_ij[0, :] / beta_report

                # Total free energy change (first to last state)
                total_dg = free_energies[-1] - free_energies[0]
                total_error = np.sqrt(
                    free_energy_errors[0] ** 2 + free_energy_errors[-1] ** 2
                )

                # Determine unit string for output
                unit_string = {'kj': '(kJ/mol)', 'kcal': '(kcal/mol)', 'kbt': '(k_BT)'}[
                    units.lower()
                ]

                result = {
                    'free_energy': total_dg,  # name to be consistent with FreeEnergyEstimator
                    'error': total_error,  # name to be consistent with FreeEnergyEstimator
                    'free_energies_all': free_energies,
                    'free_energy_errors_all': free_energy_errors,
                    'Deltaf_ij': Deltaf_ij / beta_report,
                    'dDeltaf_ij': dDeltaf_ij / beta_report,
                    'n_states': len(num_samples_per_state),
                    'units': unit_string,
                    'temperature': temperature,
                }

                # Compute overlap matrix if requested (matches original)
                if compute_overlap:
                    overlap_matrix = mbar_object.computeOverlap()[
                        2
                    ]  # Exact original syntax
                    result['overlap_matrix'] = overlap_matrix

                return result

            else:
                # Simple case - returns already converted values
                total_dg, total_error = MultistateBAR._estimatewithMBAR_core(
                    potential_energies=potential_energies,
                    num_samples_per_state=num_samples_per_state,
                    relative_tolerance=relative_tolerance,
                    regular_estimate=regular_estimate,
                    verbose=verbose,
                    initialize=initialize,
                    beta_report=beta_report,
                )

                # Determine unit string for output
                unit_string = {'kj': '(kJ/mol)', 'kcal': '(kcal/mol)', 'kbt': '(k_BT)'}[
                    units.lower()
                ]

                return {
                    'free_energy': total_dg,  # name to be consistent with FreeEnergyEstimator
                    'error': total_error,  # name to be consistent with FreeEnergyEstimator
                    'units': unit_string,
                    'temperature': temperature,
                    'n_states': len(num_samples_per_state),
                }

        except Exception as e:
            logger.error(f"MBAR calculation failed: {e}")
            raise

    @staticmethod
    def plot_overlap_matrix(overlap_matrix: np.ndarray, output_path: Path):
        """
        Plot overlap matrix adapted from alchemical_analysis.py plotOverlapMatrix.

        Parameters:
        -----------
        overlap_matrix : np.ndarray, shape (num_lambda_states, num_lambda_states)
            MBAR overlap matrix showing phase space overlap between lambda states
        output_path : Path
            Output file path for plot
        """
        try:
            import matplotlib.pyplot as plt

            num_lambda_states = overlap_matrix.shape[0]
            max_prob = overlap_matrix.max()

            _, ax = plt.subplots(
                figsize=(num_lambda_states / 2.0, num_lambda_states / 2.0)
            )

            im = ax.imshow(overlap_matrix, cmap='Blues', vmin=0, vmax=max_prob)

            for i in range(num_lambda_states):
                for j in range(num_lambda_states):
                    prob = overlap_matrix[i, j]
                    if prob < 0.005:
                        text = ''
                    elif prob > 0.995:
                        text = '1.00'
                    else:
                        text = f"{prob:.2f}"[1:]  # Remove leading 0 (original behavior)

                    color = 'white' if prob > max_prob / 2 else 'black'
                    ax.text(
                        j, i, text, ha='center', va='center', color=color, fontsize=8
                    )

            ax.set_xlabel('Lambda State')
            ax.set_ylabel('Lambda State')
            ax.set_title('MBAR Overlap Matrix')

            plt.colorbar(im, ax=ax, label='Overlap Probability')

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Overlap matrix plot saved to {output_path}")

        except ImportError:
            logger.warning("matplotlib not available for overlap matrix plotting")
        except Exception as e:
            logger.error(f"Failed to plot overlap matrix: {e}")

    @staticmethod
    def print_overlap_summary(overlap_matrix: np.ndarray):
        """
        Print overlap matrix summary as in original alchemical_analysis.py.

        Parameters:
        -----------
        overlap_matrix : np.ndarray, shape (num_lambda_states, num_lambda_states)
            MBAR overlap matrix
        """
        num_lambda_states = overlap_matrix.shape[0]
        logger.info("The overlap matrix is...")
        for k in range(num_lambda_states):
            line = ''
            for ll in range(num_lambda_states):
                line += ' %5.2f ' % overlap_matrix[k, ll]
            logger.info(line)


class FreeEnergyEstimator:
    """
    Unified interface for all free energy estimation methods.

    This class provides a simple interface to access all available
    free energy estimation methods with consistent error handling.
    """

    def __init__(
        self, temperature: float = 298.15, units: str = 'kJ', software: str = 'Gromacs'
    ):
        """
        Initialize the free energy estimator.

        Parameters:
        -----------
        temperature : float, default 298.15
            Temperature in Kelvin
        units : str, default 'kJ'
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str, default 'Gromacs'
            Software package name (affects unit handling)
        """
        self.temperature = temperature
        self.units = units
        self.software = software

        # Unit for output
        self.unit_string = {'kj': '(kJ/mol)', 'kcal': '(kcal/mol)', 'kbt': '(k_BT)'}[
            units.lower()
        ]

        # Calculate beta_report for unit conversion
        self.beta_report = calculate_beta_parameter(temperature, units, software)

    def estimate_ti(
        self,
        lambda_vectors: np.ndarray,
        ave_dhdl: np.ndarray,
        std_dhdl: np.ndarray,
        method: str = 'trapezoidal',
    ) -> dict:
        """
        Estimate total free energy using thermodynamic integration.

        Note: This method expects ave_dhdl and std_dhdl to already be in reduced units
        (i.e., already multiplied by beta) as they come from the original analysis.

        Parameters:
        -----------
        lambda_vectors : np.ndarray, shape (num_lambda_states, num_components)
            Lambda parameter values for each state and component
        ave_dhdl : np.ndarray, shape (num_lambda_states, num_components)
            Mean dH/dλ values at each lambda state (already in reduced units)
        std_dhdl : np.ndarray, shape (num_lambda_states, num_components)
            Standard error estimates for dH/dλ at each lambda state (already in reduced units)
        method : str, default 'trapezoidal'
            Integration method ('trapezoidal' or 'cubic')

        Returns:
        --------
        Dict : TI results
            - 'method': Method used
            - 'free_energy': Total free energy difference in specified units
            - 'error': Error estimate in specified units
            - 'n_points': Number of lambda states used
            - 'success': Whether calculation succeeded
        """
        try:
            # Perform integration using the class methods
            if method.lower() == 'cubic':
                dg, ddg = ThermodynamicIntegration.cubic_spline_integration(
                    lambda_vectors,
                    ave_dhdl,
                    std_dhdl,
                    self.temperature,
                    self.units,
                    self.software,
                )
            elif method.lower() == 'trapezoidal':
                dg, ddg = ThermodynamicIntegration.trapezoidal_integration(
                    lambda_vectors,
                    ave_dhdl,
                    std_dhdl,
                    self.temperature,
                    self.units,
                    self.software,
                )
            else:
                raise ValueError(f"Unknown TI method: {method}")

            return {
                'method': f'TI_{method}',
                'free_energy': dg,
                'error': ddg,
                'units': self.unit_string,
                'n_points': len(lambda_vectors),
                'success': True,
            }
        except Exception as e:
            logger.error(f"TI estimation failed: {e}")
            return {'method': f'TI_{method}', 'success': False, 'error_message': str(e)}

    def estimate_exp(
        self,
        potential_energies: np.ndarray,
        method: str = 'DEXP',
        sample_counts: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Estimate free energy using exponential averaging methods.

        Uses the new consistent API that takes potential_energies as input.

        Parameters:
        -----------
        potential_energies : np.ndarray, shape (K, K, max_N)
            Reduced potential energy matrix (already in reduced units)
        method : str, default 'DEXP'
            EXP method ('DEXP', 'IEXP', 'GDEL', 'GINS')
        sample_counts : np.ndarray, shape (K,), optional
            Number of samples per state

        Returns:
        --------
        Dict : EXP results
            - 'method': Method used
            - 'free_energy': Free energy difference in specified units
            - 'error': Error estimate in specified units
            - 'success': Whether calculation succeeded
        """
        try:
            if method.upper() == 'DEXP':
                dg, ddg = ExponentialAveraging.compute_dexp(
                    potential_energies,
                    self.temperature,
                    self.units,
                    self.software,
                    sample_counts,
                )
            elif method.upper() == 'IEXP':
                dg, ddg = ExponentialAveraging.compute_iexp(
                    potential_energies,
                    self.temperature,
                    self.units,
                    self.software,
                    sample_counts,
                )
            elif method.upper() == 'GDEL':
                dg, ddg = ExponentialAveraging.compute_gdel(
                    potential_energies,
                    self.temperature,
                    self.units,
                    self.software,
                    sample_counts,
                )
            elif method.upper() == 'GINS':
                dg, ddg = ExponentialAveraging.compute_gins(
                    potential_energies,
                    self.temperature,
                    self.units,
                    self.software,
                    sample_counts,
                )
            else:
                raise ValueError(f"Unknown EXP method: {method}")

            return {
                'method': method.upper(),
                'free_energy': dg,
                'error': ddg,
                'units': self.unit_string,
                'success': True,
            }
        except Exception as e:
            logger.error(f"EXP estimation failed: {e}")
            return {
                'method': method.upper(),
                'success': False,
                'error_message': str(e),
            }

    def estimate_bar(
        self,
        potential_energies: np.ndarray,
        method: str = 'BAR',
        sample_counts: Optional[np.ndarray] = None,
        relative_tolerance: float = 1e-10,
        verbose: bool = False,
        trial_range: tuple = (-10, 10),
    ) -> dict:
        """
        Estimate free energy using Bennett Acceptance Ratio methods.

        Uses the new consistent API that takes potential_energies as input.

        Parameters:
        -----------
        potential_energies : np.ndarray, shape (K, K, max_N)
            Reduced potential energy matrix (already in reduced units)
        method : str, default 'BAR'
            BAR method ('BAR', 'UBAR', 'RBAR')
        sample_counts : np.ndarray, shape (K,), optional
            Number of samples per state
        relative_tolerance : float, default 1e-10
            Convergence tolerance for BAR iteration
        verbose : bool, default False
            Enable verbose output
        trial_range : tuple, default (-10, 10)
            Range for RBAR method

        Returns:
        --------
        Dict : BAR results
            - 'method': Method used
            - 'free_energy': Free energy difference in specified units
            - 'error': Error estimate in specified units
            - 'success': Whether calculation succeeded
        """
        try:
            if method.upper() == 'BAR':
                dg, ddg = BennettAcceptanceRatio.compute_bar(
                    potential_energies,
                    sample_counts,
                    self.temperature,
                    self.units,
                    self.software,
                    relative_tolerance,
                    verbose,
                )
            elif method.upper() == 'UBAR':
                dg, ddg = BennettAcceptanceRatio.compute_ubar(
                    potential_energies,
                    sample_counts,
                    self.temperature,
                    self.units,
                    self.software,
                    verbose,
                )
            elif method.upper() == 'RBAR':
                dg, ddg = BennettAcceptanceRatio.compute_rbar(
                    potential_energies,
                    sample_counts,
                    self.temperature,
                    self.units,
                    self.software,
                    verbose,
                    trial_range,
                )
            else:
                raise ValueError(f"Unknown BAR method: {method}")

            return {
                'method': method.upper(),
                'free_energy': dg,
                'error': ddg,
                'units': self.unit_string,
                'success': True,
            }
        except Exception as e:
            logger.error(f"BAR estimation failed: {e}")
            return {
                'method': method.upper(),
                'success': False,
                'error_message': str(e),
            }

    def estimate_mbar(
        self,
        potential_energies: np.ndarray,
        num_samples_per_state: np.ndarray,
        regular_estimate: bool = True,
        **kwargs,
    ) -> dict:
        """
        Estimate free energy using MBAR.

        Parameters:
        -----------
        potential_energies : np.ndarray, shape (K, K, max_N)
            Reduced potential energy matrix (already in reduced units)
        num_samples_per_state : np.ndarray, shape (K,)
            Number of samples from each lambda state
        regular_estimate : bool, default True
            If True, return detailed results; if False, return simple endpoint result
        **kwargs : dict
            Additional MBAR parameters

        Returns:
        --------
        Dict : MBAR results (see MultistateBAR.compute_mbar for details)
        """
        try:
            result = MultistateBAR.compute_mbar(
                potential_energies,
                num_samples_per_state,
                temperature=self.temperature,
                units=self.units,
                software=self.software,
                regular_estimate=regular_estimate,
                **kwargs,
            )
            result['method'] = 'MBAR'
            result['success'] = True
            return result
        except Exception as e:
            logger.error(f"MBAR estimation failed: {e}")
            return {'method': 'MBAR', 'success': False, 'error_message': str(e)}

    def estimate_all_methods(
        self,
        potential_energies: np.ndarray,
        sample_counts: Optional[np.ndarray] = None,
        lambda_vectors: Optional[np.ndarray] = None,
        ave_dhdl: Optional[np.ndarray] = None,
        std_dhdl: Optional[np.ndarray] = None,
        methods: Optional[list] = None,
        **kwargs,
    ) -> dict:
        """
        Estimate free energy using all available methods.

        Parameters:
        -----------
        potential_energies : np.ndarray, shape (K, K, max_N)
            Reduced potential energy matrix
        sample_counts : np.ndarray, shape (K,), optional
            Number of samples per state
        lambda_vectors : np.ndarray, shape (K, n_components), optional
            Lambda vectors for TI methods
        ave_dhdl : np.ndarray, shape (K, n_components), optional
            Average dH/dλ for TI methods
        std_dhdl : np.ndarray, shape (K, n_components), optional
            Standard error of dH/dλ for TI methods
        methods : list, optional
            List of methods to run. If None, runs all available methods.
        **kwargs : dict
            Additional parameters for individual methods

        Returns:
        --------
        Dict : Results from all methods
            Keys are method names, values are result dictionaries
        """
        if methods is None:
            methods = ['DEXP', 'IEXP', 'GDEL', 'GINS', 'BAR', 'UBAR', 'RBAR', 'MBAR']
            if (
                lambda_vectors is not None
                and ave_dhdl is not None
                and std_dhdl is not None
            ):
                methods.extend(['TI_trapezoidal', 'TI_cubic'])

        results = {}

        # Exponential averaging methods
        for method in ['DEXP', 'IEXP', 'GDEL', 'GINS']:
            if method in methods:
                try:
                    results[method] = self.estimate_exp(
                        potential_energies, method, sample_counts
                    )
                except Exception as e:
                    results[method] = {
                        'method': method,
                        'success': False,
                        'error_message': str(e),
                    }

        # BAR methods
        for method in ['BAR', 'UBAR', 'RBAR']:
            if method in methods:
                try:
                    results[method] = self.estimate_bar(
                        potential_energies,
                        method,
                        sample_counts,
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k in ['relative_tolerance', 'verbose', 'trial_range']
                        },
                    )
                except Exception as e:
                    results[method] = {
                        'method': method,
                        'success': False,
                        'error_message': str(e),
                    }

        # MBAR method
        if 'MBAR' in methods:
            try:
                if sample_counts is not None:
                    results['MBAR'] = self.estimate_mbar(
                        potential_energies,
                        sample_counts,
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k
                            in [
                                'regular_estimate',
                                'relative_tolerance',
                                'verbose',
                                'initialize',
                            ]
                        },
                    )
                else:
                    results['MBAR'] = {
                        'method': 'MBAR',
                        'success': False,
                        'error_message': 'MBAR requires sample_counts',
                    }
            except Exception as e:
                results['MBAR'] = {
                    'method': 'MBAR',
                    'success': False,
                    'error_message': str(e),
                }

        # TI methods
        if lambda_vectors is not None and ave_dhdl is not None and std_dhdl is not None:
            for ti_method in ['trapezoidal', 'cubic']:
                method_name = f'TI_{ti_method}'
                if method_name in methods:
                    try:
                        results[method_name] = self.estimate_ti(
                            lambda_vectors, ave_dhdl, std_dhdl, ti_method
                        )
                    except Exception as e:
                        results[method_name] = {
                            'method': method_name,
                            'success': False,
                            'error_message': str(e),
                        }

        return results


def _compute_dg_internal(
    lambda_windows: list,
    run_no: int,
    start_frac: float,
    end_frac: float,
    equilibrated: bool,
    temperature: float = 298.15,
    units: str = 'kcal',
    software: str = 'Gromacs',
) -> float:
    """
    Helper function to compute the free energy change for a single run.
    """
    try:
        # Initialize free energy estimator
        fe_estimator = FreeEnergyEstimator(
            temperature=temperature, units=units, software=software
        )

        # Prepare potential energy data for MBAR
        potential_energies, sample_counts = _prepare_mbar_data_from_windows(
            lambda_windows, run_no, start_frac, end_frac, equilibrated
        )

        if potential_energies is None or len(sample_counts) == 0:
            logger.warning(
                f"No data for run {run_no}, fractions {start_frac}-{end_frac}"
            )
            return 0.0

        # Use internal MBAR estimator
        result = fe_estimator.estimate_mbar(
            potential_energies,
            np.array(sample_counts),
            regular_estimate=False,  # Just get endpoint difference
        )

        if result['success']:
            return result['free_energy']
        else:
            logger.warning(
                f"MBAR failed for run {run_no}, fractions {start_frac}-{end_frac}: {result.get('error_message', 'Unknown error')}"
            )
            return 0.0

    except Exception as e:
        logger.error(f"Error computing dG for run {run_no}: {e}")
        return 0.0
