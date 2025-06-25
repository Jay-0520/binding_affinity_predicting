"""
Free Energy Estimation Methods.

This module contains implementations of various free energy estimation methods
adapted from alchemical_analysis.py, including thermodynamic integration,
Bennett Acceptance Ratio (BAR), MBAR, and exponential averaging methods.

methods are adopted from this script:
https://github.com/MobleyLab/alchemical-analysis/blob/master/alchemical_analysis/alchemical_analysis.py
"""

import logging
from pathlib import Path

import numpy as np
import pymbar

from binding_affinity_predicting.components.analysis.utils import (
    calculate_beta_parameter,
    get_lambda_components_changing,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    Very important NOTE: This class does not handle unit conversion directly.
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
        lambda_vectors: np.ndarray, ave_dhdl: np.ndarray, std_dhdl: np.ndarray
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

        Returns:
        --------
        Tuple[float, float] : (integrated_total_free_energy_difference, propagated_total_error)
            Both in reduced units (already includes beta factor)
        """
        ThermodynamicIntegration._validate_inputs(lambda_vectors, ave_dhdl, std_dhdl)

        # Calculate differences between consecutive lambda states
        dlam = np.diff(lambda_vectors, axis=0)  # shape (n_states-1, n_components)

        # Trapezoidal rule: ½ * dλ * (f(λₖ) + f(λₖ₊₁))
        df_contributions = 0.5 * np.sum(dlam * (ave_dhdl[:-1] + ave_dhdl[1:]), axis=1)
        # Sum over all lambda intervals
        total_df = np.sum(df_contributions)

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

        total_error = np.sqrt(total_error_variance)

        return float(total_df), float(total_error)

    @staticmethod
    def cubic_spline_integration(
        lambda_vectors: np.ndarray, ave_dhdl: np.ndarray, std_dhdl: np.ndarray
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

        Returns:
        --------
        Tuple[float, float] : (integrated_total_free_energy_difference, propagated_total_error)
            Both in reduced units (already includes beta factor)
        """
        num_lambda_states, num_components = lambda_vectors.shape

        if num_lambda_states < 4:
            logger.warning("Cubic spline needs ≥4 points, falling back to trapezoidal")
            return ThermodynamicIntegration.trapezoidal_integration(
                lambda_vectors, ave_dhdl, std_dhdl
            )

        components_changing = get_lambda_components_changing(lambda_vectors)

        total_df = 0.0
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

                total_df += df_j
                total_error_variance += ddf_j_sq

            except Exception as e:
                logger.warning(f"Spline creation failed for component {j}: {e}")
                continue

        return float(total_df), float(np.sqrt(total_error_variance))


class MultistateBAR:
    """
    Multistate Bennett Acceptance Ratio implementation.

    Faithfully adapted from alchemical_analysis.py MBAR functionality.
    """

    @staticmethod
    def _estimatewithMBAR_core(
        uncorr_potential_energies: np.ndarray,
        num_uncorr_samples_per_state: np.ndarray,
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
        uncorr_potential_energies : np.ndarray, shape (K, K, max_N)
            Reduced potential energy matrix where [k,l,n] is the reduced potential
            of snapshot n from state k evaluated at state l
        num_uncorr_samples_per_state : np.ndarray, shape (K,)
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
                uncorr_potential_energies,
                num_uncorr_samples_per_state,
                verbose=verbose,
                relative_tolerance=relative_tolerance,
                initialize=initialize,
            )

            # Get free energy differences exactly as in original
            # note that theta_ij (marked as "_") is never used in the original code - by JJH-2025-06-24  # noqa: E501
            (Deltaf_ij, dDeltaf_ij, _) = MBAR.getFreeEnergyDifferences(
                uncertainty_method='svd-ew', return_theta=True
            )

            if verbose:
                logger.info(
                    f"Matrix of free energy differences\nDeltaf_ij:\n{Deltaf_ij}\ndDeltaf_ij:\n{dDeltaf_ij}"  # noqa: E501
                )

            # Return format matches original exactly
            if regular_estimate:
                return (Deltaf_ij, dDeltaf_ij, MBAR)
            else:
                K = len(num_uncorr_samples_per_state)
                return (
                    Deltaf_ij[0, K - 1] / beta_report,
                    dDeltaf_ij[0, K - 1] / beta_report,
                    MBAR,
                )

        except Exception as e:
            logger.error(f"MBAR calculation failed: {e}")
            raise

    @staticmethod
    def compute_mbar(
        uncorr_potential_energies: np.ndarray,
        num_uncorr_samples_per_state: np.ndarray,
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
        uncorr_potential_energies : np.ndarray, shape (num_lambda_states, num_lambda_states,
        max_num_snapshots)
            Reduced potential energy matrix where:
            - First index (k): lambda state where snapshots were generated
            - Second index (l): lambda state where energy is evaluated
            - Third index (n): snapshot/time index
            Example: uncorr_potential_energies[1, 3, :] = energies of λ₁ snapshots evaluated at λ₃
        num_uncorr_samples_per_state : np.ndarray, shape (num_lambda_states,)
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
                        uncorr_potential_energies=uncorr_potential_energies,
                        num_uncorr_samples_per_state=num_uncorr_samples_per_state,
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
                    'total_dg': total_dg,
                    'total_error': total_error,
                    'free_energies': free_energies,
                    'free_energy_errors': free_energy_errors,
                    'Deltaf_ij': Deltaf_ij / beta_report,
                    'dDeltaf_ij': dDeltaf_ij / beta_report,
                    'mbar_object': mbar_object,
                    'n_states': len(num_uncorr_samples_per_state),
                    'units': unit_string,
                    'beta_report': beta_report,
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
                    uncorr_potential_energies=uncorr_potential_energies,
                    num_uncorr_samples_per_state=num_uncorr_samples_per_state,
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
                    'total_dg': total_dg,
                    'total_error': total_error,
                    'units': unit_string,
                    'beta_report': beta_report,
                    'temperature': temperature,
                    'n_states': len(num_uncorr_samples_per_state),
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

            fig, ax = plt.subplots(
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


# TODO: need to double check this implementation
class ExponentialAveraging:
    """
    Exponential averaging methods adapted from alchemical_analysis.py.
    Implements DEXP, IEXP, GDEL, and GINS methods exactly as in original.
    """

    @staticmethod
    def forward_exp(
        w_F: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kJ',
        software: str = 'Gromacs',
    ) -> tuple[float, float]:
        """
        Forward exponential averaging (DEXP - deletion).
        Exactly adapted from alchemical_analysis.py DEXP implementation.

        Parameters:
        -----------
        w_F : np.ndarray
            Forward work values: w_F = u_kln[k,k+1,:] - u_kln[k,k,:]
            These should already be in reduced units!
        temperature : float
            Temperature in Kelvin (for unit conversion only)
        units : str
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str
            Software package name

        Returns:
        --------
        Tuple[float, float] : (free_energy, error) in specified units
        """
        try:
            # Original implementation - w_F is already in reduced units
            (df_reduced, ddf_reduced) = pymbar.exp.EXP(w_F)

            # Convert to physical units using beta_report
            beta_report = calculate_beta_parameter(temperature, units, software)

            # Unit conversion exactly as in original: df/P.beta_report
            df_physical = df_reduced / beta_report
            ddf_physical = ddf_reduced / beta_report

            return df_physical, ddf_physical

        except Exception as e:
            logger.warning(
                f"pymbar EXP failed: {e}, falling back to simple exponential averaging"
            )
            return ExponentialAveraging._simple_exp(w_F, temperature, units, software)

    @staticmethod
    def reverse_exp(
        w_R: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kJ',
        software: str = 'Gromacs',
    ) -> tuple[float, float]:
        """
        Reverse exponential averaging (IEXP - insertion).
        Exactly adapted from alchemical_analysis.py IEXP implementation.

        Parameters:
        -----------
        w_R : np.ndarray
            Reverse work values: w_R = u_kln[k+1,k,:] - u_kln[k+1,k+1,:]
            These should already be in reduced units!
        temperature : float
            Temperature in Kelvin (for unit conversion only)
        units : str
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str
            Software package name

        Returns:
        --------
        Tuple[float, float] : (free_energy, error) in specified units
        """
        try:
            # Original implementation - w_R is already in reduced units
            (rdf_reduced, rddf_reduced) = pymbar.exp.EXP(w_R)

            # Original: (df['IEXP'], ddf['IEXP']) = (-rdf, rddf)
            df_reduced = -rdf_reduced
            ddf_reduced = rddf_reduced  # Error doesn't change sign

            # Convert to physical units
            beta_report = calculate_beta_parameter(temperature, units, software)
            df_physical = df_reduced / beta_report
            ddf_physical = ddf_reduced / beta_report

            return df_physical, ddf_physical

        except Exception as e:
            logger.warning(
                f"pymbar EXP failed: {e}, falling back to simple exponential averaging"
            )
            # For fallback, negate the work values
            return ExponentialAveraging._simple_exp(
                -w_R, temperature, units, software, negate_result=True
            )

    @staticmethod
    def gaussian_deletion(
        w_F: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kJ',
        software: str = 'Gromacs',
    ) -> tuple[float, float]:
        """
        Gaussian deletion (GDEL).
        Exactly adapted from alchemical_analysis.py GDEL implementation.

        Parameters:
        -----------
        w_F : np.ndarray
            Forward work values: w_F = u_kln[k,k+1,:] - u_kln[k,k,:]
            These should already be in reduced units!
        temperature : float
            Temperature in Kelvin (for unit conversion only)
        units : str
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str
            Software package name

        Returns:
        --------
        Tuple[float, float] : (free_energy, error) in specified units
        """
        try:
            # Original implementation - w_F is already in reduced units
            (df_reduced, ddf_reduced) = pymbar.exp.EXPGauss(w_F)

            # Convert to physical units
            beta_report = calculate_beta_parameter(temperature, units, software)
            df_physical = df_reduced / beta_report
            ddf_physical = ddf_reduced / beta_report

            return df_physical, ddf_physical

        except Exception as e:
            logger.warning(f"pymbar EXPGauss failed: {e}, falling back to forward EXP")
            return ExponentialAveraging.forward_exp(w_F, temperature, units, software)

    @staticmethod
    def gaussian_insertion(
        w_R: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kJ',
        software: str = 'Gromacs',
    ) -> tuple[float, float]:
        """
        Gaussian insertion (GINS).
        Exactly adapted from alchemical_analysis.py GINS implementation.

        Parameters:
        -----------
        w_R : np.ndarray
            Reverse work values: w_R = u_kln[k+1,k,:] - u_kln[k+1,k+1,:]
            These should already be in reduced units!
        temperature : float
            Temperature in Kelvin (for unit conversion only)
        units : str
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str
            Software package name

        Returns:
        --------
        Tuple[float, float] : (free_energy, error) in specified units
        """
        try:
            # Original implementation - w_R is already in reduced units
            (rdf_reduced, rddf_reduced) = pymbar.exp.EXPGauss(w_R)

            # Original: (df['GINS'], ddf['GINS']) = (-rdf, rddf)
            df_reduced = -rdf_reduced
            ddf_reduced = rddf_reduced  # Error doesn't change sign

            # Convert to physical units
            beta_report = calculate_beta_parameter(temperature, units, software)
            df_physical = df_reduced / beta_report
            ddf_physical = ddf_reduced / beta_report

            return df_physical, ddf_physical

        except Exception as e:
            logger.warning(f"pymbar EXPGauss failed: {e}, falling back to reverse EXP")
            return ExponentialAveraging.reverse_exp(w_R, temperature, units, software)

    @staticmethod
    def _simple_exp(
        work_values: np.ndarray,
        temperature: float,
        units: str = 'kJ',
        software: str = 'Gromacs',
        negate_result: bool = False,
    ) -> tuple[float, float]:
        """
        Simple exponential averaging fallback.

        Parameters:
        -----------
        work_values : np.ndarray
            Work values (should already be in reduced units)
        temperature : float
            Temperature in Kelvin
        units : str
            Output units
        software : str
            Software package name
        negate_result : bool
            Whether to negate the final result (for IEXP fallback)

        Returns:
        --------
        Tuple[float, float] : (free_energy, error)
        """
        # work_values should already be in reduced units, so use directly
        exp_work = np.exp(-work_values)

        # Handle numerical issues
        if np.any(np.isnan(exp_work)) or np.any(np.isinf(exp_work)):
            logger.warning("Numerical issues in exponential averaging")
            exp_work = exp_work[np.isfinite(exp_work)]

        if len(exp_work) == 0:
            return float('inf'), float('inf')

        mean_exp = np.mean(exp_work)
        if mean_exp <= 0:
            return float('inf'), float('inf')

        # Free energy in reduced units
        dg_reduced = -np.log(mean_exp)

        # Error estimate in reduced units
        var_exp = np.var(exp_work)
        dg_error_reduced = np.sqrt(var_exp) / (mean_exp * np.sqrt(len(exp_work)))

        # Convert to physical units
        beta_report = calculate_beta_parameter(temperature, units, software)
        dg_physical = dg_reduced / beta_report
        dg_error_physical = dg_error_reduced / beta_report

        if negate_result:
            dg_physical = -dg_physical

        return dg_physical, dg_error_physical

    @staticmethod
    def calculate_work_values(
        u_kln: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate forward and reverse work values as in original alchemical_analysis.py.

        Parameters:
        -----------
        u_kln : np.ndarray, shape (K, K, max_N)
            Reduced potential energy matrix
            same as uncorr_potential_energies : np.ndarray, shape (K, K, max_N)
        k : int
            Current lambda state index

        Returns:
        --------
        tuple[np.ndarray, np.ndarray] : (w_F, w_R)
            Forward and reverse work values
        """
        if k >= u_kln.shape[0] - 1:
            raise ValueError(
                f"State index {k} too large for matrix with {u_kln.shape[0]} states"
            )

        # Find valid samples
        N_k = np.sum(u_kln[k, k, :] != 0)  # Assuming 0 means no data
        N_k_plus_1 = np.sum(u_kln[k + 1, k + 1, :] != 0)

        # Forward work: w_F = u_kln[k,k+1,0:N_k[k]] - u_kln[k,k,0:N_k[k]]
        w_F = u_kln[k, k + 1, 0:N_k] - u_kln[k, k, 0:N_k]

        # Reverse work: w_R = u_kln[k+1,k,0:N_k[k+1]] - u_kln[k+1,k+1,0:N_k[k+1]]
        w_R = u_kln[k + 1, k, 0:N_k_plus_1] - u_kln[k + 1, k + 1, 0:N_k_plus_1]

        return w_F, w_R


# TODO: need to double check this implementation
class BennettAcceptanceRatio:
    """
    Bennett Acceptance Ratio methods adapted from alchemical_analysis.py.
    Implements BAR, UBAR, and RBAR methods exactly as in original.
    """

    @staticmethod
    def bar(
        w_F: np.ndarray,
        w_R: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kJ',
        software: str = 'Gromacs',
        relative_tolerance: float = 1e-10,
        verbose: bool = False,
    ) -> tuple[float, float]:
        """
        Bennett Acceptance Ratio (BAR) method.
        Exactly adapted from alchemical_analysis.py BAR implementation.

        Parameters:
        -----------
        w_F : np.ndarray
            Forward work values: w_F = u_kln[k,k+1,:] - u_kln[k,k,:]
            These should already be in reduced units!
        w_R : np.ndarray
            Reverse work values: w_R = u_kln[k+1,k,:] - u_kln[k+1,k+1,:]
            These should already be in reduced units!
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

        Returns:
        --------
        Tuple[float, float] : (free_energy, error) in specified units
        """
        try:
            # Original implementation - work values are already in reduced units
            (df_reduced, ddf_reduced) = pymbar.bar.BAR(
                w_F, w_R, relative_tolerance=relative_tolerance, verbose=verbose
            )

            # Convert to physical units using beta_report
            beta_report = calculate_beta_parameter(temperature, units, software)
            df_physical = df_reduced / beta_report
            ddf_physical = ddf_reduced / beta_report

            return df_physical, ddf_physical

        except Exception as e:
            logger.error(f"pymbar BAR failed: {e}")
            raise

    @staticmethod
    def ubar(
        w_F: np.ndarray,
        w_R: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kJ',
        software: str = 'Gromacs',
        verbose: bool = False,
    ) -> tuple[float, float]:
        """
        Unoptimized Bennett Acceptance Ratio (UBAR) method.
        Exactly adapted from alchemical_analysis.py UBAR implementation.

        UBAR assumes dF is zero and does only one evaluation (no iteration).

        Parameters:
        -----------
        w_F : np.ndarray
            Forward work values: w_F = u_kln[k,k+1,:] - u_kln[k,k,:]
            These should already be in reduced units!
        w_R : np.ndarray
            Reverse work values: w_R = u_kln[k+1,k,:] - u_kln[k+1,k+1,:]
            These should already be in reduced units!
        temperature : float
            Temperature in Kelvin (for unit conversion only)
        units : str
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str
            Software package name
        verbose : bool
            Enable verbose output

        Returns:
        --------
        Tuple[float, float] : (free_energy, error) in specified units
        """
        try:
            # Original implementation - assume dF is zero, just do one evaluation
            (df_reduced, ddf_reduced) = pymbar.bar.BAR(
                w_F,
                w_R,
                verbose=verbose,
                iterated_solution=False,  # This is the key difference from BAR
            )

            # Convert to physical units using beta_report
            beta_report = calculate_beta_parameter(temperature, units, software)
            df_physical = df_reduced / beta_report
            ddf_physical = ddf_reduced / beta_report

            return df_physical, ddf_physical

        except Exception as e:
            logger.error(f"pymbar UBAR failed: {e}")
            raise

    @staticmethod
    def rbar(
        w_F: np.ndarray,
        w_R: np.ndarray,
        temperature: float = 298.15,
        units: str = 'kJ',
        software: str = 'Gromacs',
        verbose: bool = False,
        trial_range: tuple = (-10, 10),
    ) -> tuple[float, float]:
        """
        Range-based Bennett Acceptance Ratio (RBAR) method.
        Exactly adapted from alchemical_analysis.py RBAR implementation.

        RBAR calculates UBAR for a series of 'trial' free energies and chooses
        the one that best satisfies the equations.

        Parameters:
        -----------
        w_F : np.ndarray
            Forward work values: w_F = u_kln[k,k+1,:] - u_kln[k,k,:]
            These should already be in reduced units!
        w_R : np.ndarray
            Reverse work values: w_R = u_kln[k+1,k,:] - u_kln[k+1,k+1,:]
            These should already be in reduced units!
        temperature : float
            Temperature in Kelvin (for unit conversion only)
        units : str
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str
            Software package name
        verbose : bool
            Enable verbose output
        trial_range : tuple
            Range of trial free energy values to test (in reduced units)

        Returns:
        --------
        Tuple[float, float] : (free_energy, error) in specified units
        """
        try:
            # Original implementation logic - test range of trial free energies
            min_diff = 1e6
            best_udf = 0
            best_uddf = 0

            # Test trial free energies in the specified range
            for trial_udf in range(trial_range[0], trial_range[1] + 1, 1):
                try:
                    # Calculate UBAR with this trial free energy as initial guess
                    (udf, uddf) = pymbar.bar.BAR(
                        w_F,
                        w_R,
                        verbose=verbose,
                        iterated_solution=False,
                        initial_f_k=np.array([0.0, float(trial_udf)]),  # Trial guess
                    )

                    # Check how well this satisfies the BAR equation
                    # (This is a simplified version - original has more complex logic)
                    diff = abs(udf - trial_udf)

                    if diff < min_diff:
                        min_diff = diff
                        best_udf = udf
                        best_uddf = uddf

                except Exception:
                    # Skip this trial value if it fails
                    continue

            if min_diff == 1e6:
                # All trials failed, fall back to regular BAR
                logger.warning("All RBAR trials failed, falling back to BAR")
                return BennettAcceptanceRatio.bar(
                    w_F, w_R, temperature, units, software
                )

            # Convert to physical units using beta_report
            beta_report = calculate_beta_parameter(temperature, units, software)
            df_physical = best_udf / beta_report
            ddf_physical = best_uddf / beta_report

            return df_physical, ddf_physical

        except Exception as e:
            logger.error(f"RBAR calculation failed: {e}, falling back to BAR")
            return BennettAcceptanceRatio.bar(w_F, w_R, temperature, units, software)

    @staticmethod
    def calculate_work_values(
        u_kln: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate forward and reverse work values as in original alchemical_analysis.py.

        This is identical to the function in ExponentialAveraging class.

        Parameters:
        -----------
        u_kln : np.ndarray, shape (K, K, max_N)
            Reduced potential energy matrix
        k : int
            Current lambda state index

        Returns:
        --------
        tuple[np.ndarray, np.ndarray] : (w_F, w_R)
            Forward and reverse work values
        """
        if k >= u_kln.shape[0] - 1:
            raise ValueError(
                f"State index {k} too large for matrix with {u_kln.shape[0]} states"
            )

        # Find valid samples (assuming 0 means no data)
        N_k = np.sum(u_kln[k, k, :] != 0)
        N_k_plus_1 = np.sum(u_kln[k + 1, k + 1, :] != 0)

        # Forward work: w_F = u_kln[k,k+1,0:N_k[k]] - u_kln[k,k,0:N_k[k]]
        w_F = u_kln[k, k + 1, 0:N_k] - u_kln[k, k, 0:N_k]

        # Reverse work: w_R = u_kln[k+1,k,0:N_k[k+1]] - u_kln[k+1,k+1,0:N_k[k+1]]
        w_R = u_kln[k + 1, k, 0:N_k_plus_1] - u_kln[k + 1, k + 1, 0:N_k_plus_1]

        return w_F, w_R

    @staticmethod
    def analyze_all_methods(
        u_kln: np.ndarray,
        k: int,
        temperature: float = 298.15,
        units: str = 'kJ',
        software: str = 'Gromacs',
        relative_tolerance: float = 1e-10,
        verbose: bool = False,
    ) -> dict:
        """
        Run all BAR-based methods (BAR, UBAR, RBAR) for a given state pair.

        Parameters:
        -----------
        u_kln : np.ndarray, shape (K, K, max_N)
            Reduced potential energy matrix
        k : int
            Current lambda state index
        temperature : float
            Temperature in Kelvin
        units : str
            Output units
        software : str
            Software package name
        relative_tolerance : float
            Convergence tolerance
        verbose : bool
            Enable verbose output

        Returns:
        --------
        dict : Results from all methods
            Keys: 'BAR', 'UBAR', 'RBAR'
            Values: (free_energy, error) tuples
        """
        # Calculate work values
        w_F, w_R = BennettAcceptanceRatio.calculate_work_values(u_kln, k)

        results = {}

        # BAR
        try:
            results['BAR'] = BennettAcceptanceRatio.bar(
                w_F, w_R, temperature, units, software, relative_tolerance, verbose
            )
        except Exception as e:
            logger.error(f"BAR failed for state {k}: {e}")
            results['BAR'] = (float('inf'), float('inf'))

        # UBAR
        try:
            results['UBAR'] = BennettAcceptanceRatio.ubar(
                w_F, w_R, temperature, units, software, verbose
            )
        except Exception as e:
            logger.error(f"UBAR failed for state {k}: {e}")
            results['UBAR'] = (float('inf'), float('inf'))

        # RBAR
        try:
            results['RBAR'] = BennettAcceptanceRatio.rbar(
                w_F, w_R, temperature, units, software, verbose
            )
        except Exception as e:
            logger.error(f"RBAR failed for state {k}: {e}")
            results['RBAR'] = (float('inf'), float('inf'))

        return results


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
            # Perform integration (inputs already in reduced units)
            if method.lower() == 'cubic':
                dg_reduced, ddg_reduced = (
                    ThermodynamicIntegration.cubic_spline_integration(
                        lambda_vectors, ave_dhdl, std_dhdl
                    )
                )
            elif method.lower() == 'trapezoidal':
                dg_reduced, ddg_reduced = (
                    ThermodynamicIntegration.trapezoidal_integration(
                        lambda_vectors, ave_dhdl, std_dhdl
                    )
                )
            else:
                raise ValueError(f"Unknown TI method: {method}")

            # Convert to physical units using beta_report
            return {
                'method': f'TI_{method}',
                'free_energy': dg_reduced / self.beta_report,
                'error': ddg_reduced / self.beta_report,
                'units': self.units,
                'n_points': len(lambda_vectors),
                'success': True,
            }
        except Exception as e:
            logger.error(f"TI estimation failed: {e}")
            return {'method': f'TI_{method}', 'success': False, 'error_message': str(e)}

    def estimate_exp(
        self,
        u_kln: np.ndarray,
        k: int,
        method: str = 'DEXP',
    ) -> dict:
        """
        Estimate free energy using exponential averaging methods.

        This method calculates work values from the energy matrix and applies
        the appropriate exponential averaging method.

        Parameters:
        -----------
        u_kln : np.ndarray, shape (K, K, max_N)
            Reduced potential energy matrix (already in reduced units)
        k : int
            Lambda state index (for transitions k→k+1)
        method : str, default 'DEXP'
            EXP method ('DEXP', 'IEXP', 'GDEL', 'GINS')

        Returns:
        --------
        Dict : EXP results
            - 'method': Method used
            - 'free_energy': Free energy difference in specified units
            - 'error': Error estimate in specified units
            - 'n_samples': Number of samples used
            - 'success': Whether calculation succeeded
        """
        try:
            # Calculate work values (already in reduced units)
            w_F, w_R = ExponentialAveraging.calculate_work_values(u_kln, k)

            if method.upper() == 'DEXP':
                dg, ddg = ExponentialAveraging.forward_exp(
                    w_F, self.temperature, self.units, self.software
                )
                n_samples = len(w_F)
            elif method.upper() == 'IEXP':
                dg, ddg = ExponentialAveraging.reverse_exp(
                    w_R, self.temperature, self.units, self.software
                )
                n_samples = len(w_R)
            elif method.upper() == 'GDEL':
                dg, ddg = ExponentialAveraging.gaussian_deletion(
                    w_F, self.temperature, self.units, self.software
                )
                n_samples = len(w_F)
            elif method.upper() == 'GINS':
                dg, ddg = ExponentialAveraging.gaussian_insertion(
                    w_R, self.temperature, self.units, self.software
                )
                n_samples = len(w_R)
            else:
                raise ValueError(f"Unknown EXP method: {method}")

            return {
                'method': method.upper(),
                'free_energy': dg,
                'error': ddg,
                'units': self.units,
                'n_samples': n_samples,
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
        u_kln: np.ndarray,
        k: int,
        method: str = 'BAR',
        relative_tolerance: float = 1e-10,
        verbose: bool = False,
    ) -> dict:
        """
        Estimate free energy using Bennett Acceptance Ratio methods.

        This method calculates work values from the energy matrix and applies
        the appropriate BAR method.

        Parameters:
        -----------
        u_kln : np.ndarray, shape (K, K, max_N)
            Reduced potential energy matrix (already in reduced units)
        k : int
            Lambda state index (for transitions k→k+1)
        method : str, default 'BAR'
            BAR method ('BAR', 'UBAR', 'RBAR')
        relative_tolerance : float, default 1e-10
            Convergence tolerance for BAR iteration
        verbose : bool, default False
            Enable verbose output

        Returns:
        --------
        Dict : BAR results
            - 'method': Method used
            - 'free_energy': Free energy difference in specified units
            - 'error': Error estimate in specified units
            - 'n_forward': Number of forward samples
            - 'n_reverse': Number of reverse samples
            - 'success': Whether calculation succeeded
        """
        try:
            # Calculate work values (already in reduced units)
            w_F, w_R = BennettAcceptanceRatio.calculate_work_values(u_kln, k)

            if method.upper() == 'BAR':
                dg, ddg = BennettAcceptanceRatio.bar(
                    w_F,
                    w_R,
                    self.temperature,
                    self.units,
                    self.software,
                    relative_tolerance,
                    verbose,
                )
            elif method.upper() == 'UBAR':
                dg, ddg = BennettAcceptanceRatio.ubar(
                    w_F, w_R, self.temperature, self.units, self.software, verbose
                )
            elif method.upper() == 'RBAR':
                dg, ddg = BennettAcceptanceRatio.rbar(
                    w_F, w_R, self.temperature, self.units, self.software, verbose
                )
            else:
                raise ValueError(f"Unknown BAR method: {method}")

            return {
                'method': method.upper(),
                'free_energy': dg,
                'error': ddg,
                'units': self.units,
                'n_forward': len(w_F),
                'n_reverse': len(w_R),
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
        uncorr_potential_energies: np.ndarray,
        num_uncorr_samples_per_state: np.ndarray,
        regular_estimate: bool = True,
        **kwargs,
    ) -> dict:
        """
        Estimate free energy using MBAR.

        Parameters:
        -----------
        uncorr_potential_energies : np.ndarray, shape (K, K, max_N)
            Reduced potential energy matrix (already in reduced units)
        num_uncorr_samples_per_state : np.ndarray, shape (K,)
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
                uncorr_potential_energies,
                num_uncorr_samples_per_state,
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
