"""
Free Energy Estimation Methods.

This module contains implementations of various free energy estimation methods
adapted from alchemical_analysis.py, including thermodynamic integration,
Bennett Acceptance Ratio (BAR), MBAR, and exponential averaging methods.

https://github.com/MobleyLab/alchemical-analysis/blob/master/alchemical_analysis/alchemical_analysis.py
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pymbar
import scipy.interpolate as scipy_interpolate

from binding_affinity_predicting.components.analysis.utils import (
    get_lambda_components_changing,
)

logger = logging.getLogger(__name__)


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
        if len(lambdas_per_component) < 4:
            raise ValueError("Natural cubic spline requires at least 4 points")

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
        # Error propagation: ¼ * (dλ)² * (σₖ² + σₖ₊₁²)
        ddf_sq_contributions = 0.25 * np.sum(
            dlam**2 * (std_dhdl[:-1] ** 2 + std_dhdl[1:] ** 2), axis=1
        )

        # Sum over all lambda intervals
        total_df = np.sum(df_contributions)
        total_error = np.sqrt(np.sum(ddf_sq_contributions))

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

        # Determine which components are changing (like get_lchange in original)
        lambda_components_changing = np.zeros([num_lambda_states, num_components], bool)
        for j in range(num_components):
            for k in range(num_lambda_states - 1):
                if abs(lambda_vectors[k + 1, j] - lambda_vectors[k, j]) > 1e-10:
                    lambda_components_changing[k, j] = True
                    lambda_components_changing[k + 1, j] = True

        total_df = 0.0
        total_error_sq = 0.0

        try:
            # Get boolean mask for all changing components
            components_changing = get_lambda_components_changing(lambda_vectors)

            # Process each component separately
            for j in range(num_lambda_states):
                # Find states where this component changes
                changing_mask = components_changing[:, j]

                if not np.any(changing_mask):
                    continue  # Component doesn't change

                # Extract data for changing states only
                lambda_j = lambda_vectors[changing_mask, j]
                ave_j = ave_dhdl[changing_mask, j]
                std_j = std_dhdl[changing_mask, j]

                # Integrate this component
                if len(lambda_j) < 4:
                    # Use trapezoidal for this component
                    df_j, ddf_j = ThermodynamicIntegration._trapezoidal_component(
                        lambda_j, ave_j, std_j
                    )
                else:
                    # Use cubic spline for this component
                    df_j, ddf_j = ThermodynamicIntegration._cubic_spline_component(
                        lambda_j, ave_j, std_j
                    )

                total_df += df_j
                total_error_sq += ddf_j**2

            return float(total_df), float(np.sqrt(total_error_sq))

        except Exception as e:
            logger.warning(f"Cubic spline integration failed: {e}, using trapezoidal")
            return ThermodynamicIntegration.trapezoidal_integration(
                lambda_vectors, ave_dhdl, std_dhdl
            )

    @staticmethod
    def _trapezoidal_component(
        lambda_vals: np.ndarray, ave_vals: np.ndarray, std_vals: np.ndarray
    ) -> tuple[float, float]:
        """Trapezoidal integration for a single component."""
        dlam = np.diff(lambda_vals)
        df = 0.5 * np.dot(dlam, ave_vals[:-1] + ave_vals[1:])
        ddf_sq = 0.25 * np.dot(dlam**2, std_vals[:-1] ** 2 + std_vals[1:] ** 2)
        return float(df), float(np.sqrt(ddf_sq))

    @staticmethod
    def _cubic_spline_component(
        lambda_vals: np.ndarray, ave_vals: np.ndarray, std_vals: np.ndarray
    ) -> tuple[float, float]:
        """Cubic spline integration for a single component."""
        spline = NaturalCubicSpline(lambda_vals)
        df = np.dot(spline.wsum, ave_vals)
        ddf_sq = np.dot(spline.wsum**2, std_vals**2)
        return float(df), float(np.sqrt(ddf_sq))


class MultistateBAR:
    """
    Multistate Bennett Acceptance Ratio implementation.
    Adapted from alchemical_analysis.py MBAR functionality.
    """

    @staticmethod
    def compute_mbar(
        uncorr_potential_energies: np.ndarray,
        num_uncorr_samples_per_state: np.ndarray,
        temperature: float = 298.15,
        **kwargs,
    ) -> dict:
        """
        Compute MBAR estimates for all states.
        Adapted from alchemical_analysis.py estimatewithMBAR function.

        Parameters:
        -----------
        uncorr_potential_energies : np.ndarray, shape (num_lambda_states, num_lambda_states, max_num_snapshots)
            Reduced potential energy matrix where:
            - First index (k): lambda state where snapshots were generated
            - Second index (l): lambda state where energy is evaluated
            - Third index (n): snapshot/time index
            Example: uncorr_potential_energies[1, 3, :] = energies of λ₁ snapshots evaluated at λ₃
            Variable name matches your convention (was u_kln in original)
        num_uncorr_samples_per_state : np.ndarray, shape (num_lambda_states,)
            Number of samples from each lambda state
            Variable name matches your convention (was N_k in original)
        temperature : float, default 298.15
            Temperature in Kelvin
        **kwargs : dict
            Additional MBAR parameters (relative_tolerance, verbose, initialize)

        Returns:
        --------
        Dict : MBAR results including free energies and errors
            - 'total_dg': Total free energy change (kJ/mol)
            - 'total_error': Total error estimate (kJ/mol)
            - 'free_energies': Free energies relative to first state (kJ/mol)
            - 'free_energy_errors': Error estimates for each state (kJ/mol)
            - 'Deltaf_ij': Pairwise free energy differences matrix (kJ/mol)
            - 'dDeltaf_ij': Error matrix for pairwise differences (kJ/mol)
            - 'mbar_object': The pymbar MBAR object
            - 'n_states': Number of lambda states
        """
        try:
            # MBAR parameters
            relative_tolerance = kwargs.get('relative_tolerance', 1e-10)
            verbose = kwargs.get('verbose', False)
            initialize = kwargs.get('initialize', 'BAR')

            # Initialize MBAR
            mbar = pymbar.mbar.MBAR(
                uncorr_potential_energies,
                num_uncorr_samples_per_state,
                verbose=verbose,
                relative_tolerance=relative_tolerance,
                initialize=initialize,
            )

            # Get free energy differences
            Deltaf_ij, dDeltaf_ij, theta_ij = mbar.getFreeEnergyDifferences(
                uncertainty_method='svd-ew', return_theta=True
            )

            # Convert to physical units
            kB = 8.314462618e-3  # kJ/(mol·K)
            beta = 1.0 / (kB * temperature)

            # Free energies relative to first state
            f_k = Deltaf_ij[0, :] / beta
            df_k = dDeltaf_ij[0, :] / beta

            # Total free energy change
            total_dg = f_k[-1] - f_k[0]
            total_error = np.sqrt(df_k[0] ** 2 + df_k[-1] ** 2)

            result = {
                'total_dg': total_dg,
                'total_error': total_error,
                'free_energies': f_k,
                'free_energy_errors': df_k,
                'Deltaf_ij': Deltaf_ij / beta,
                'dDeltaf_ij': dDeltaf_ij / beta,
                'mbar_object': mbar,
                'n_states': len(num_uncorr_samples_per_state),
            }

            # Compute overlap matrix if requested
            if kwargs.get('compute_overlap', False):
                overlap_matrix = mbar.computeOverlap()[2]
                result['overlap_matrix'] = overlap_matrix

            return result

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

            # Create heatmap
            im = ax.imshow(overlap_matrix, cmap='Blues', vmin=0, vmax=max_prob)

            # Add text annotations
            for i in range(num_lambda_states):
                for j in range(num_lambda_states):
                    prob = overlap_matrix[i, j]
                    if prob < 0.005:
                        text = ''
                    elif prob > 0.995:
                        text = '1.00'
                    else:
                        text = f"{prob:.2f}"[1:]  # Remove leading 0

                    color = 'white' if prob > max_prob / 2 else 'black'
                    ax.text(
                        j, i, text, ha='center', va='center', color=color, fontsize=8
                    )

            # Set labels
            ax.set_xlabel('Lambda State')
            ax.set_ylabel('Lambda State')
            ax.set_title('MBAR Overlap Matrix')

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Overlap Probability')

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Overlap matrix plot saved to {output_path}")

        except ImportError:
            logger.warning("matplotlib not available for overlap matrix plotting")
        except Exception as e:
            logger.error(f"Failed to plot overlap matrix: {e}")


class ExponentialAveraging:
    """
    Exponential averaging methods adapted from alchemical_analysis.py.
    Implements forward/reverse EXP and Gaussian approximations.
    """

    @staticmethod
    def forward_exp(
        work_values: np.ndarray, temperature: float = 298.15
    ) -> tuple[float, float]:
        """
        Forward exponential averaging (deletion).
        Adapted from alchemical_analysis.py DEXP implementation.

        Parameters:
        -----------
        work_values : np.ndarray
            Work values for forward process
        temperature : float
            Temperature in Kelvin

        Returns:
        --------
        Tuple[float, float] : (free_energy, error)
        """
        try:
            kB = 8.314462618e-3  # kJ/(mol·K)
            beta = 1.0 / (kB * temperature)
            w_reduced = beta * work_values

            dg_reduced, ddg_reduced = pymbar.exp.EXP(w_reduced)
            return dg_reduced / beta, ddg_reduced / beta

        except Exception as e:
            logger.warning(
                f"pymbar EXP failed: {e}, falling back to simple exponential averaging"
            )
            # Fallback implementation
            return ExponentialAveraging._simple_exp(work_values, temperature)

    @staticmethod
    def reverse_exp(
        work_values: np.ndarray, temperature: float = 298.15
    ) -> Tuple[float, float]:
        """
        Reverse exponential averaging (insertion).
        Adapted from alchemical_analysis.py IEXP implementation.

        Parameters:
        -----------
        work_values : np.ndarray
            Work values for reverse process
        temperature : float
            Temperature in Kelvin

        Returns:
        --------
        Tuple[float, float] : (free_energy, error)
        """
        # For reverse direction, negate work values
        dg, ddg = ExponentialAveraging.forward_exp(-work_values, temperature)
        return -dg, ddg

    @staticmethod
    def gaussian_exp(
        work_values: np.ndarray, temperature: float = 298.15, forward: bool = True
    ) -> tuple[float, float]:
        """
        Gaussian approximation to exponential averaging.
        Adapted from alchemical_analysis.py GDEL/GINS implementations.

        Parameters:
        -----------
        work_values : np.ndarray
            Work values
        temperature : float
            Temperature in Kelvin
        forward : bool
            Whether this is forward (True) or reverse (False) direction

        Returns:
        --------
        Tuple[float, float] : (free_energy, error)
        """
        try:
            kB = 8.314462618e-3  # kJ/(mol·K)
            beta = 1.0 / (kB * temperature)

            w_input = beta * work_values if forward else -beta * work_values
            dg_reduced, ddg_reduced = pymbar.exp.EXPGauss(w_input)

            result_dg = dg_reduced / beta if forward else -dg_reduced / beta
            result_ddg = ddg_reduced / beta

            return result_dg, result_ddg

        except Exception as e:
            logger.warning(f"pymbar EXPGauss failed: {e}")
            # Fallback: use simple exponential averaging
            if forward:
                return ExponentialAveraging.forward_exp(work_values, temperature)
            else:
                return ExponentialAveraging.reverse_exp(work_values, temperature)

    @staticmethod
    def _simple_exp(work_values: np.ndarray, temperature: float) -> Tuple[float, float]:
        """
        Simple exponential averaging fallback.

        Parameters:
        -----------
        work_values : np.ndarray
            Work values
        temperature : float
            Temperature in Kelvin

        Returns:
        --------
        Tuple[float, float] : (free_energy, error)
        """
        kB = 8.314462618e-3  # kJ/(mol·K)
        beta = 1.0 / (kB * temperature)

        w_reduced = beta * work_values
        exp_work = np.exp(-w_reduced)

        # Handle numerical issues
        if np.any(np.isnan(exp_work)) or np.any(np.isinf(exp_work)):
            logger.warning("Numerical issues in exponential averaging")
            exp_work = exp_work[np.isfinite(exp_work)]

        if len(exp_work) == 0:
            return float('inf'), float('inf')

        mean_exp = np.mean(exp_work)
        if mean_exp <= 0:
            return float('inf'), float('inf')

        dg = -np.log(mean_exp) / beta

        # Error estimate
        var_exp = np.var(exp_work)
        dg_error = np.sqrt(var_exp) / (mean_exp * np.sqrt(len(exp_work))) / beta

        return dg, dg_error


class BennettAcceptanceRatio:
    """
    Bennett Acceptance Ratio implementation adapted from alchemical_analysis.py.
    Provides optimal free energy estimates between adjacent states.
    """

    @staticmethod
    def compute_bar(
        forward_work: np.ndarray,
        reverse_work: np.ndarray,
        temperature: float = 298.15,
        **kwargs,
    ) -> tuple[float, float]:
        """
        Compute BAR estimate between two adjacent states.
        Adapted from alchemical_analysis.py BAR implementation.

        Parameters:
        -----------
        forward_work : np.ndarray
            Work values from forward simulations
        reverse_work : np.ndarray
            Work values from reverse simulations
        temperature : float
            Temperature in Kelvin
        **kwargs : dict
            Additional parameters for BAR calculation

        Returns:
        --------
        Tuple[float, float] : (free_energy_difference, error)
        """
        try:
            kB = 8.314462618e-3  # kJ/(mol·K)
            beta = 1.0 / (kB * temperature)

            w_F = beta * forward_work
            w_R = beta * reverse_work

            # Use pymbar BAR with optional parameters
            relative_tolerance = kwargs.get('relative_tolerance', 1e-10)
            verbose = kwargs.get('verbose', False)

            dg_reduced, ddg_reduced = pymbar.bar.BAR(
                w_F, w_R, relative_tolerance=relative_tolerance, verbose=verbose
            )

            return dg_reduced / beta, ddg_reduced / beta

        except Exception as e:
            logger.warning(f"pymbar BAR failed: {e}, falling back to simple BAR")
            # Fallback implementation
            return BennettAcceptanceRatio._simple_bar(
                forward_work, reverse_work, temperature
            )

    @staticmethod
    def compute_ubar(
        forward_work: np.ndarray, reverse_work: np.ndarray, temperature: float = 298.15
    ) -> tuple[float, float]:
        """
        Unoptimized BAR (single iteration).
        Adapted from alchemical_analysis.py UBAR implementation.

        Parameters:
        -----------
        forward_work : np.ndarray
            Work values from forward simulations
        reverse_work : np.ndarray
            Work values from reverse simulations
        temperature : float
            Temperature in Kelvin

        Returns:
        --------
        Tuple[float, float] : (free_energy_difference, error)
        """
        try:
            kB = 8.314462618e-3  # kJ/(mol·K)
            beta = 1.0 / (kB * temperature)

            w_F = beta * forward_work
            w_R = beta * reverse_work

            dg_reduced, ddg_reduced = pymbar.bar.BAR(
                w_F, w_R, iterated_solution=False, verbose=False
            )

            return dg_reduced / beta, ddg_reduced / beta

        except Exception as e:
            logger.warning(f"pymbar UBAR failed: {e}")
            # Fallback
            return BennettAcceptanceRatio._simple_bar(
                forward_work, reverse_work, temperature
            )

    @staticmethod
    def compute_rbar(
        forward_work: np.ndarray, reverse_work: np.ndarray, temperature: float = 298.15
    ) -> tuple[float, float]:
        """
        Range-optimized BAR.
        Adapted from alchemical_analysis.py RBAR implementation.

        Parameters:
        -----------
        forward_work : np.ndarray
            Work values from forward simulations
        reverse_work : np.ndarray
            Work values from reverse simulations
        temperature : float
            Temperature in Kelvin

        Returns:
        --------
        Tuple[float, float] : (free_energy_difference, error)
        """
        try:
            kB = 8.314462618e-3  # kJ/(mol·K)
            beta = 1.0 / (kB * temperature)

            w_F = beta * forward_work
            w_R = beta * reverse_work

            # Try different initial guesses
            min_diff = 1e6
            best_dg = 0
            best_ddg = 0

            for trial_dg in range(-10, 11):  # Trial range -10 to +10 kBT
                try:
                    dg_reduced, ddg_reduced = pymbar.bar.BAR(
                        w_F,
                        w_R,
                        DeltaF=trial_dg,
                        iterated_solution=False,
                        verbose=False,
                    )

                    diff = abs(dg_reduced - trial_dg)
                    if diff < min_diff:
                        best_dg = dg_reduced
                        best_ddg = ddg_reduced
                        min_diff = diff

                except Exception:
                    continue

            return best_dg / beta, best_ddg / beta

        except Exception as e:
            logger.warning(f"RBAR failed: {e}")
            return BennettAcceptanceRatio._simple_bar(
                forward_work, reverse_work, temperature
            )

    @staticmethod
    def _simple_bar(
        forward_work: np.ndarray, reverse_work: np.ndarray, temperature: float
    ) -> tuple[float, float]:
        """
        Simplified BAR approximation using exponential averaging.

        Parameters:
        -----------
        forward_work : np.ndarray
            Work values from forward simulations
        reverse_work : np.ndarray
            Work values from reverse simulations
        temperature : float
            Temperature in Kelvin

        Returns:
        --------
        Tuple[float, float] : (free_energy_difference, error)
        """
        dg_forward, err_forward = ExponentialAveraging.forward_exp(
            forward_work, temperature
        )
        dg_reverse, err_reverse = ExponentialAveraging.reverse_exp(
            reverse_work, temperature
        )

        # Simple average (not optimal, but reasonable approximation)
        dg = (dg_forward + dg_reverse) / 2.0
        dg_error = np.sqrt(err_forward**2 + err_reverse**2) / 2.0

        return dg, dg_error


class FreeEnergyEstimator:
    """
    Unified interface for all free energy estimation methods.

    This class provides a simple interface to access all available
    free energy estimation methods with consistent error handling.
    """

    def __init__(self, temperature: float = 298.15):
        """
        Initialize the free energy estimator.

        Parameters:
        -----------
        temperature : float, default 298.15
            Temperature in Kelvin
        """
        self.temperature = temperature
        self.ti = ThermodynamicIntegration()
        self.exp = ExponentialAveraging()
        self.bar = BennettAcceptanceRatio()
        self.mbar = MultistateBAR()

    def estimate_ti(
        self,
        lambda_vectors: np.ndarray,
        dhdl_uncorrelated: np.ndarray,
        num_uncorr_samples_per_state: np.ndarray,
        method: str = 'trapezoidal',
    ) -> dict:
        """
        Estimate total free energy using thermodynamic integration.

        Parameters:
        -----------
        lambda_vectors : np.ndarray, shape (num_lambda_states, num_components)
            Lambda parameter values for each state and component
        dhdl_uncorrelated : np.ndarray, shape (num_lambda_states, num_components, max_uncorr_samples)
            Uncorrelated dH/dλ samples from TI analysis
        num_uncorr_samples_per_state : np.ndarray, shape (num_lambda_states,)
            Number of uncorrelated samples from each lambda state
        method : str, default 'trapezoidal'
            Integration method ('trapezoidal' or 'cubic')

        Returns:
        --------
        Dict : TI results
            - 'method': Method used
            - 'free_energy': Total free energy difference (kJ/mol)
            - 'error': Error estimate (kJ/mol)
            - 'n_points': Number of lambda states used
            - 'success': Whether calculation succeeded
        """
        try:
            # Calculate beta for unit conversion
            kB = 8.314462618e-3  # kJ/(mol·K)
            beta = 1.0 / (kB * self.temperature)

            # Compute TI preliminaries (matches original TIprelim function)
            dlam, ave_dhdl, std_dhdl = self.ti.compute_ti_preliminaries(
                lambda_vectors, dhdl_uncorrelated, num_uncorr_samples_per_state, beta
            )

            # Perform integration
            if method.lower() == 'cubic':
                dg_reduced, ddg_reduced = self.ti.cubic_spline_integration(
                    lambda_vectors, ave_dhdl, std_dhdl
                )
            elif method.lower() == 'trapezoidal':
                dg_reduced, ddg_reduced = self.ti.trapezoidal_integration(
                    lambda_vectors, ave_dhdl, std_dhdl
                )
            else:
                raise ValueError(f"Unknown TI method: {method}")

            # Convert back to physical units (kJ/mol)
            return {
                'method': f'TI_{method}',
                'free_energy': dg_reduced / beta,
                'error': ddg_reduced / beta,
                'n_points': len(lambda_vectors),
                'success': True,
            }
        except Exception as e:
            logger.error(f"TI estimation failed: {e}")
            return {'method': f'TI_{method}', 'success': False, 'error_message': str(e)}

    def estimate_exp(
        self,
        work_values_forward: np.ndarray,
        work_values_reverse: Optional[np.ndarray] = None,
        method: str = 'forward',
    ) -> dict:
        """
        Estimate free energy using exponential averaging.

        Note: This method expects work values in physical units (kJ/mol) and converts to reduced units internally
        to match the original alchemical_analysis.py approach.

        Parameters:
        -----------
        work_values_forward : np.ndarray, shape (num_forward_samples,)
            Forward work values (kJ/mol) from λ=0 → λ=1 transformations
        work_values_reverse : np.ndarray, optional, shape (num_reverse_samples,)
            Reverse work values (kJ/mol) from λ=1 → λ=0 transformations
        method : str, default 'forward'
            EXP method ('forward', 'reverse', 'gaussian_forward', 'gaussian_reverse')

        Returns:
        --------
        Dict : EXP results
            - 'method': Method used
            - 'free_energy': Free energy difference (kJ/mol)
            - 'error': Error estimate (kJ/mol)
            - 'n_samples': Number of samples used
            - 'success': Whether calculation succeeded
        """
        try:
            # Convert to reduced units (matching original workflow)
            kB = 8.314462618e-3  # kJ/(mol·K)
            beta = 1.0 / (kB * self.temperature)

            if method == 'forward':
                w_F = beta * work_values_forward
                dg_reduced, ddg_reduced = self.exp.forward_exp(w_F, self.temperature)
                dg, ddg = dg_reduced / beta, ddg_reduced / beta
            elif method == 'reverse':
                if work_values_reverse is None:
                    raise ValueError("Reverse work values required for reverse EXP")
                w_R = beta * work_values_reverse
                dg_reduced, ddg_reduced = self.exp.reverse_exp(w_R, self.temperature)
                dg, ddg = dg_reduced / beta, ddg_reduced / beta
            elif method == 'gaussian_forward':
                w_F = beta * work_values_forward
                dg_reduced, ddg_reduced = self.exp.gaussian_exp(
                    w_F, self.temperature, forward=True
                )
                dg, ddg = dg_reduced / beta, ddg_reduced / beta
            elif method == 'gaussian_reverse':
                if work_values_reverse is None:
                    raise ValueError(
                        "Reverse work values required for reverse Gaussian EXP"
                    )
                w_R = beta * work_values_reverse
                dg_reduced, ddg_reduced = self.exp.gaussian_exp(
                    w_R, self.temperature, forward=False
                )
                dg, ddg = dg_reduced / beta, ddg_reduced / beta
            else:
                raise ValueError(f"Unknown EXP method: {method}")

            return {
                'method': f'EXP_{method}',
                'free_energy': dg,
                'error': ddg,
                'n_samples': len(work_values_forward),
                'success': True,
            }
        except Exception as e:
            logger.error(f"EXP estimation failed: {e}")
            return {
                'method': f'EXP_{method}',
                'success': False,
                'error_message': str(e),
            }

    def estimate_bar(
        self,
        work_values_forward: np.ndarray,
        work_values_reverse: np.ndarray,
        method: str = 'standard',
    ) -> dict:
        """
        Estimate free energy using Bennett Acceptance Ratio.

        Note: This method expects work values in physical units (kJ/mol) and converts to reduced units internally
        to match the original alchemical_analysis.py approach.

        Parameters:
        -----------
        work_values_forward : np.ndarray, shape (num_forward_samples,)
            Forward work values (kJ/mol) from adjacent lambda states
        work_values_reverse : np.ndarray, shape (num_reverse_samples,)
            Reverse work values (kJ/mol) from adjacent lambda states
        method : str, default 'standard'
            BAR method ('standard', 'unoptimized', 'range_optimized')

        Returns:
        --------
        Dict : BAR results
            - 'method': Method used
            - 'free_energy': Free energy difference (kJ/mol)
            - 'error': Error estimate (kJ/mol)
            - 'n_forward': Number of forward samples
            - 'n_reverse': Number of reverse samples
            - 'success': Whether calculation succeeded
        """
        try:
            # Convert to reduced units (matching original workflow)
            kB = 8.314462618e-3  # kJ/(mol·K)
            beta = 1.0 / (kB * self.temperature)

            w_F = beta * work_values_forward
            w_R = beta * work_values_reverse

            if method == 'standard':
                dg_reduced, ddg_reduced = self.bar.compute_bar(
                    w_F, w_R, self.temperature
                )
            elif method == 'unoptimized':
                dg_reduced, ddg_reduced = self.bar.compute_ubar(
                    w_F, w_R, self.temperature
                )
            elif method == 'range_optimized':
                dg_reduced, ddg_reduced = self.bar.compute_rbar(
                    w_F, w_R, self.temperature
                )
            else:
                raise ValueError(f"Unknown BAR method: {method}")

            # Convert back to physical units
            return {
                'method': f'BAR_{method}',
                'free_energy': dg_reduced / beta,
                'error': ddg_reduced / beta,
                'n_forward': len(work_values_forward),
                'n_reverse': len(work_values_reverse),
                'success': True,
            }
        except Exception as e:
            logger.error(f"BAR estimation failed: {e}")
            return {
                'method': f'BAR_{method}',
                'success': False,
                'error_message': str(e),
            }

    def estimate_mbar(
        self,
        uncorr_potential_energies: np.ndarray,
        num_uncorr_samples_per_state: np.ndarray,
        **kwargs,
    ) -> dict:
        """
        Estimate free energy using MBAR.

        Parameters:
        -----------
        uncorr_potential_energies : np.ndarray, shape (num_lambda_states, num_lambda_states, max_num_snapshots)
            Reduced potential energy matrix (see MultistateBAR.compute_mbar for details)
        num_uncorr_samples_per_state : np.ndarray, shape (num_lambda_states,)
            Number of samples from each lambda state
        **kwargs : dict
            Additional MBAR parameters

        Returns:
        --------
        Dict : MBAR results (see MultistateBAR.compute_mbar for details)
        """
        try:
            result = self.mbar.compute_mbar(
                uncorr_potential_energies,
                num_uncorr_samples_per_state,
                self.temperature,
                **kwargs,
            )
            result['method'] = 'MBAR'
            result['success'] = True
            return result
        except Exception as e:
            logger.error(f"MBAR estimation failed: {e}")
            return {'method': 'MBAR', 'success': False, 'error_message': str(e)}
