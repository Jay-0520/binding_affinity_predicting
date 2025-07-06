"""
Equilibrium detection system for GROMACS FEP simulations.

This module provides comprehensive equilibrium detection methods adapted from a3fe
for use with the binding_affinity_predicting codebase. It integrates with the
existing GROMACS orchestration system.
"""

import logging
import os
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import kpss

from binding_affinity_predicting.components.analysis.autocorrelation import (
    _statistical_inefficiency_chodera,
)
from binding_affinity_predicting.components.analysis.gradient_analyzer import (
    GradientAnalyzer,
)
from binding_affinity_predicting.components.analysis.xvg_data_loader import (
    GromacsXVGParser,
)
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    Calculation,
    LambdaWindow,
    Leg,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EquilibriumDetectionError(Exception):
    """Custom exception for equilibrium detection errors."""

    pass


class EquilibriumBlockGradientDetector:
    """
    Equilibrium detection based on gradient of block-averaged dH/dλ.
    Adapted from check_equil_block_gradient in a3fe.
    (https://github.com/michellab/a3fe)
    """

    def __init__(
        self, block_size: float = 1.0, gradient_threshold: Optional[float] = None
    ):
        """
        Parameters
        ----------
        block_size : float, default 1.0
            Block size for averaging in ns
        gradient_threshold : float, optional
            Threshold for gradient magnitude (kcal/mol/ns). If None,
            equilibration is detected when gradient crosses zero.
        """
        self.block_size = block_size
        self.gradient_threshold = gradient_threshold
        self.gradient_analyzer = GradientAnalyzer()

    def detect_equilibrium(
        self, window: LambdaWindow, run_nos: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[float]]:
        """
        Detect equilibrium using block gradient method.

        Returns
        -------
        Tuple[bool, Optional[float]]
            (equilibrated, equilibration_time_ns)
        """
        try:
            # Read gradient data
            times, gradients = self.gradient_analyzer.read_gradients_from_window(
                window, run_nos
            )

            # Calculate timestep from times array
            if len(times[0]) < 2:
                raise EquilibriumDetectionError("Insufficient data points")

            timestep = times[0][1] - times[0][0]  # ns
            idx_block_size = int(self.block_size / timestep)

            # Calculate gradient of gradient for each run
            gradient_derivatives = []
            for run_gradients in gradients:
                d_dh_dl = self._calculate_gradient_derivative(
                    run_gradients, idx_block_size
                )
                gradient_derivatives.append(d_dh_dl)

            # Calculate mean gradient derivative
            mean_gradient_derivative = np.nanmean(gradient_derivatives, axis=0)

            # Detect equilibration
            equilibrated, equil_time = self._find_equilibration_point(
                times[0], mean_gradient_derivative, idx_block_size
            )

            # Save results
            self._save_results(window, equilibrated, equil_time, run_nos)

            return equilibrated, equil_time

        except Exception as e:
            logger.error(f"Block gradient equilibrium detection failed: {e}")
            return False, None

    def _calculate_gradient_derivative(
        self, gradients: np.ndarray, idx_block_size: int
    ) -> np.ndarray:
        """Calculate derivative of block-averaged gradients."""
        d_dh_dl = np.full(len(gradients), np.nan)

        # Calculate rolling average
        rolling_avg = self._get_rolling_average(gradients, idx_block_size)

        # Calculate derivative
        for i in range(len(gradients)):
            if i < 2 * idx_block_size:
                continue
            d_dh_dl[i] = (
                rolling_avg[i] - rolling_avg[i - idx_block_size]
            ) / self.block_size

        return d_dh_dl

    def _get_rolling_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate rolling average with specified window size."""
        rolling_avg = np.full(len(data), np.nan)

        for i in range(len(data)):
            if i < window_size:
                continue
            rolling_avg[i] = np.mean(data[i - window_size : i])

        return rolling_avg

    def _find_equilibration_point(
        self, times: np.ndarray, gradient_derivative: np.ndarray, idx_block_size: int
    ) -> Tuple[bool, Optional[float]]:
        """Find the equilibration point based on gradient criteria."""
        start_idx = 2 * idx_block_size

        if start_idx >= len(gradient_derivative):
            return False, None

        last_grad = gradient_derivative[start_idx]

        for i in range(start_idx, len(gradient_derivative)):
            grad = gradient_derivative[i]

            if np.isnan(grad):
                continue

            # Check threshold criterion
            if self.gradient_threshold and abs(grad) < self.gradient_threshold:
                return True, times[i]

            # Check zero-crossing criterion
            if not np.isnan(last_grad) and np.sign(last_grad) != np.sign(grad):
                return True, times[i]

            last_grad = grad

        return False, None

    def _save_results(
        self,
        window: LambdaWindow,
        equilibrated: bool,
        equil_time: Optional[float],
        run_nos: Optional[List[int]],
    ):
        """Save equilibration detection results."""
        output_file = Path(window.output_dir) / "equilibration_block_gradient.txt"

        with open(output_file, 'w') as f:
            f.write(f"Equilibrated: {equilibrated}\n")
            f.write(f"Equilibration time: {equil_time} ns\n")
            f.write(f"Block size: {self.block_size} ns\n")
            f.write(f"Gradient threshold: {self.gradient_threshold}\n")
            f.write(f"Run numbers: {run_nos}\n")


class EquilibriumMultiwindowDetector:
    """
    Multi-window equilibrium detection based on cumulative free energy changes.
    Adapted from check_equil_multiwindow_* methods in a3fe.
    """

    def __init__(
        self,
        method: str = "paired_t",
        first_frac: float = 0.1,
        last_frac: float = 0.5,
        intervals: int = 4,
        p_cutoff: float = 0.05,
    ):
        """
        Parameters
        ----------
        method : str, default "paired_t"
            Detection method: "gradient", "kpss", "geweke", or "paired_t"
        first_frac : float, default 0.1
            Fraction of simulation for first part of statistical test
        last_frac : float, default 0.5
            Fraction of simulation for last part of statistical test
        intervals : int, default 4
            Number of intervals to test for equilibration
        p_cutoff : float, default 0.05
            P-value cutoff for statistical tests
        """
        self.method = method
        self.first_frac = first_frac
        self.last_frac = last_frac
        self.intervals = intervals
        self.p_cutoff = p_cutoff
        self.gradient_analyzer = GradientAnalyzer()

        # Validate intervals
        if self.first_frac <= 0 or self.first_frac >= 1:
            raise ValueError("first_frac must be between 0 and 1")
        if self.last_frac <= 0 or self.last_frac >= 1:
            raise ValueError("last_frac must be between 0 and 1")
        if self.first_frac + self.last_frac >= 1:
            raise ValueError("first_frac + last_frac must be < 1")

    def detect_equilibrium(
        self, leg: Leg, run_nos: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[float]]:
        """
        Detect equilibrium across multiple lambda windows.

        Parameters
        ----------
        leg : Leg
            Leg containing multiple lambda windows
        run_nos : List[int], optional
            Run numbers to analyze

        Returns
        -------
        Tuple[bool, Optional[float]]
            (equilibrated, fractional_equilibration_time)
        """
        try:
            if self.method == "paired_t":
                return self._detect_paired_t_based(leg, run_nos)
            elif self.method == "gradient":
                return self._detect_gradient_based(leg, run_nos)
            elif self.method == "kpss":
                return self._detect_kpss_based(leg, run_nos)
            elif self.method == "geweke":
                return self._detect_geweke_based(leg, run_nos)
            else:
                raise ValueError(f"Unsupported method: {self.method}")

        except Exception as e:
            logger.error(f"Multi-window equilibrium detection failed: {e}")
            return False, None

    def _detect_paired_t_based(
        self, leg: Leg, run_nos: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[float]]:
        """
        Paired t-test based detection - exact implementation from a3fe.
        """
        from scipy import stats

        if run_nos is None:
            run_nos = list(range(1, leg.lambda_windows[0].ensemble_size + 1))

        # Initialize results storage
        p_vals_and_times = []
        equilibrated = False
        fractional_equil_time = None
        equil_time = None

        # Calculate test intervals
        start_fracs = np.linspace(0, 1 - self.last_frac, num=self.intervals)

        for start_frac in start_fracs:
            try:
                # Get time series data using MBAR-like approach
                overall_dgs, overall_times = self._get_time_series_multiwindow_mbar(
                    leg.lambda_windows, run_nos, start_frac
                )

                # Calculate slice indices
                first_slice_end_idx = round(self.first_frac * len(overall_dgs[0]))
                last_slice_start_idx = round((1 - self.last_frac) * len(overall_dgs[0]))

                # Extract slices
                first_slice = overall_dgs[:, :first_slice_end_idx]
                last_slice = overall_dgs[:, last_slice_start_idx:]

                # Calculate means for each run
                first_slice_means = np.mean(first_slice, axis=1)
                last_slice_means = np.mean(last_slice, axis=1)

                # Perform paired t-test
                _, p_value = stats.ttest_rel(
                    first_slice_means, last_slice_means, alternative="two-sided"
                )

                # Store results
                p_vals_and_times.append((p_value, overall_times[0][0]))

                # Check if equilibrated
                if p_value > self.p_cutoff and not equilibrated:
                    equilibrated = True
                    fractional_equil_time = start_frac
                    equil_time = overall_times[0][0]

            except Exception as e:
                logger.warning(f"Failed to analyze start_frac {start_frac}: {e}")
                continue

        # Update lambda window attributes if equilibrated
        if equilibrated:
            for lam_win in leg.lambda_windows:
                if hasattr(lam_win, '_equilibrated'):
                    lam_win._equilibrated = True
                if hasattr(lam_win, '_equil_time') and hasattr(
                    lam_win, 'get_tot_simtime'
                ):
                    # Equilibration time per simulation
                    lam_win._equil_time = (
                        fractional_equil_time * lam_win.get_tot_simtime([1])
                    )

        # Save results to file
        self._save_paired_t_results(
            leg,
            equilibrated,
            p_vals_and_times,
            fractional_equil_time,
            equil_time,
            run_nos,
        )

        return equilibrated, fractional_equil_time

    def _get_time_series_multiwindow(
        self,
        lambda_windows: List[LambdaWindow],
        run_nos: List[int],
        start_frac: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get combined time series using MBAR-like approach.
        Simplified version of get_time_series_multiwindow_mbar from a3fe.

        This function is adapted from get_time_series_multiwindow()
        https://github.com/michellab/a3fe/blob/main/a3fe/analyse/process_grads.py
        """
        # Check that weights are defined for all windows
        if not all(
            hasattr(win, 'lam_val_weight') and win.lam_val_weight
            for win in lambda_windows
        ):
            # Set equal weights if not defined
            for win in lambda_windows:
                if not hasattr(win, 'lam_val_weight'):
                    win.lam_val_weight = 1.0 / len(lambda_windows)

        n_runs = len(run_nos)
        n_points = 100  # Block average into 100 points

        # Initialize arrays - one point for each % of the total simulation time
        overall_dgs = np.zeros([n_runs, n_points])
        overall_times = np.zeros([n_runs, n_points])

        # Process each lambda window
        for lam_win in lambda_windows:
            for i, run_no in enumerate(run_nos):
                try:
                    times, gradients = (
                        self.gradient_analyzer.read_gradients_from_window(
                            lam_win, [run_no]
                        )
                    )
                    if len(times) == 0 or len(gradients) == 0:
                        logger.warning(
                            f"No data for window {lam_win.lam_state}, run {run_no}"
                        )
                        continue

                    # read_gradients_from_window() returns a list
                    times = times[0]
                    grads = gradients[0]
                    # Weight the gradients by lambda weight
                    dgs = grads * lam_win.lam_val_weight
                    # Apply start_frac truncation
                    start_idx = 0 if start_frac == 0.0 else round(start_frac * len(dgs))
                    end_idx = len(dgs)  # Use all data to end (end_frac = 1.0)

                    # Make sure we have enough data for block averaging
                    if end_idx - start_idx < n_points:
                        logger.warning(
                            f"Not enough data for window {lam_win.lam_state}, run {run_no}"
                        )
                        continue

                    times = times[start_idx:end_idx]
                    dgs = dgs[start_idx:end_idx]

                    # Resize times using linear interpolation
                    times_resized = np.linspace(times[0], times[-1], n_points)
                    # Block average the gradients into n_points evenly-sized blocks
                    dgs_resized = np.zeros(n_points)
                    indices = np.array(
                        [round(x) for x in np.linspace(0, len(dgs), n_points + 1)]
                    )

                    for j in range(n_points):
                        dgs_resized[j] = np.mean(dgs[indices[j] : indices[j + 1]])

                    overall_dgs[i] += dgs_resized
                    overall_times[i] += times_resized

                except Exception as e:
                    logger.warning(
                        f"Failed to process window {lam_win.lam_state}, run {run_no}: {e}"
                    )
                    continue

        return overall_dgs, overall_times

    def get_time_series_multiwindow_mbar(
        lambda_windows: list[LambdaWindow],  # noqa: F821
        output_dir: str,
        equilibrated: bool = False,
        run_nos: Optional[list[int]] = None,
        start_frac: float = 0.0,
        end_frac: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Check that equilibration stats have been set for all lambda windows if equilibrated is True
        if equilibrated and not all([lam.equilibrated for lam in lambda_windows]):
            raise ValueError(
                "The equilibration times and statistics have not been set for all lambda "
                "windows in the stage. Please set these before running this function."
            )

        # Check the run numbers
        run_nos: list[int] = lambda_windows[0]._get_valid_run_nos(run_nos)

        # Combine all gradients to get the change in free energy with increasing simulation time.
        # Do this so that the total simulation time for each window is spread evenly over the total
        # simulation time for the whole calculation
        n_runs = len(run_nos)
        n_points = 100
        overall_dgs = np.zeros(
            [
                n_runs,
                n_points,
            ]
        )  # One point for each % of the total simulation time
        overall_times = np.zeros([n_runs, n_points])
        start_and_end_fracs = [
            (i, i + (end_frac - start_frac) / n_points)
            for i in np.linspace(start_frac, end_frac, n_points + 1)
        ][
            :-1
        ]  # Throw away the last point as > 1
        # Round the values to avoid floating point errors
        start_and_end_fracs = [
            (round(x[0], 5), round(x[1], 5)) for x in start_and_end_fracs
        ]

        # Run MBAR in parallel
        with get_context("spawn").Pool() as pool:
            results = pool.starmap(
                _compute_dg,
                [
                    (run_no, start_frac, end_frac, output_dir, equilibrated)
                    for run_no in run_nos
                    for start_frac, end_frac in start_and_end_fracs
                ],
            )

            # Reshape the results
            for i, run_no in enumerate(run_nos):
                for j, (start_frac, end_frac) in enumerate(start_and_end_fracs):
                    overall_dgs[i, j] = results[i * len(start_and_end_fracs) + j]

        # Get times per run
        for i, run_no in enumerate(run_nos):
            total_time = sum(
                [
                    lam_win.get_tot_simulation_time([run_no])
                    for lam_win in lambda_windows
                ]
            )
            equil_time = (
                sum([lam_win.equil_time for lam_win in lambda_windows])
                if equilibrated
                else 0
            )
            times = [
                (total_time - equil_time) * fracs[0] + equil_time
                for fracs in start_and_end_fracs
            ]
            overall_times[i] = times

        # Check that we have the same total times for each run
        if not all(
            [
                np.isclose(overall_times[i, -1], overall_times[0, -1])
                for i in range(n_runs)
            ]
        ):
            raise ValueError(
                "Total simulation times are not the same for all runs. Please ensure that "
                "the total simulation times are the same for all runs."
            )

        # Check that we didn't get any NaNs
        if np.isnan(overall_dgs).any():
            raise ValueError(
                "NaNs found in the free energy change. Please check that the simulation "
                "has run correctly."
            )

        return overall_dgs, overall_times

    def _detect_gradient_based(
        self, leg: Leg, run_nos: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[float]]:
        """Gradient-based multi-window detection."""
        # Simplified gradient-based approach
        for discard_fraction in [0.0, 0.1, 0.3, 0.6]:
            try:
                overall_dgs, overall_times = self._get_time_series_multiwindow_mbar(
                    leg.lambda_windows, run_nos or list(range(1, 6)), discard_fraction
                )

                # Simple test: check if gradient variance decreases
                if len(overall_dgs) > 0 and len(overall_dgs[0]) > 10:
                    # Calculate gradient of cumulative free energy
                    mean_dgs = np.mean(overall_dgs, axis=0)
                    gradients = np.gradient(mean_dgs)

                    # Check if latter half has lower variance than first half
                    mid_point = len(gradients) // 2
                    first_half_var = np.var(gradients[:mid_point])
                    second_half_var = np.var(gradients[mid_point:])

                    if second_half_var < 0.5 * first_half_var:  # Arbitrary threshold
                        return True, discard_fraction

            except Exception as e:
                logger.warning(
                    f"Gradient test failed for fraction {discard_fraction}: {e}"
                )
                continue

        return False, None

    def _detect_kpss_based(
        self, leg: Leg, run_nos: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[float]]:
        """KPSS stationarity test based detection."""
        for discard_fraction in [0.0, 0.1, 0.3, 0.5]:
            try:
                overall_dgs, _ = self._get_time_series_multiwindow_mbar(
                    leg.lambda_windows, run_nos or list(range(1, 6)), discard_fraction
                )

                # Use mean across runs
                mean_dgs = np.mean(overall_dgs, axis=0)

                if len(mean_dgs) < 10:
                    continue

                # KPSS test for stationarity
                _, p_value, *_ = kpss(mean_dgs, regression='c', nlags='auto')

                if p_value > 0.05:  # Stationary (null hypothesis not rejected)
                    return True, discard_fraction

            except Exception as e:
                logger.warning(f"KPSS test failed for fraction {discard_fraction}: {e}")
                continue

        return False, None

    def _detect_geweke_based(
        self, leg: Leg, run_nos: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[float]]:
        """Modified Geweke test based detection."""
        from scipy import stats

        for discard_fraction in np.linspace(0, 1 - self.last_frac, self.intervals):
            try:
                overall_dgs, overall_times = self._get_time_series_multiwindow_mbar(
                    leg.lambda_windows, run_nos or list(range(1, 6)), discard_fraction
                )

                # Calculate slice indices for Geweke test
                first_slice_end_idx = int(self.first_frac * len(overall_dgs[0]))
                last_slice_start_idx = int((1 - self.last_frac) * len(overall_dgs[0]))

                # Extract slices for each run
                first_slice_means = []
                last_slice_means = []

                for run_dgs in overall_dgs:
                    first_slice_means.append(np.mean(run_dgs[:first_slice_end_idx]))
                    last_slice_means.append(np.mean(run_dgs[last_slice_start_idx:]))

                # Use independent samples t-test (Geweke-style)
                _, p_value = stats.ttest_ind(
                    first_slice_means,
                    last_slice_means,
                    equal_var=False,  # Welch's t-test
                )

                if p_value > self.p_cutoff:
                    return True, discard_fraction

            except Exception as e:
                logger.warning(
                    f"Geweke test failed for fraction {discard_fraction}: {e}"
                )
                continue

        return False, None

    def _save_paired_t_results(
        self,
        leg: Leg,
        equilibrated: bool,
        p_vals_and_times: List[Tuple[float, float]],
        fractional_equil_time: Optional[float],
        equil_time: Optional[float],
        run_nos: List[int],
    ):
        """Save paired t-test results to file."""
        output_file = Path(leg.output_dir) / "check_equil_multiwindow_paired_t.txt"

        with open(output_file, 'w') as f:
            f.write(f"Equilibrated: {equilibrated}\n")
            f.write(f"p values and times: {p_vals_and_times}\n")
            f.write(f"Fractional equilibration time: {fractional_equil_time}\n")
            f.write(f"Equilibration time: {equil_time} ns\n")
            f.write(f"Run numbers: {run_nos}\n")
            f.write(f"Method: {self.method}\n")
            f.write(
                f"Parameters: first_frac={self.first_frac}, last_frac={self.last_frac}, "
            )
            f.write(f"intervals={self.intervals}, p_cutoff={self.p_cutoff}\n")


class EquilibriumDetectionManager:
    """
    High-level manager for equilibrium detection across different methods.
    """

    def __init__(self, method: str = "block_gradient", **method_kwargs):
        """
        Parameters
        ----------
        method : str, default "block_gradient"
            Detection method: "block_gradient", "chodera", or "multiwindow"
        **method_kwargs
            Method-specific parameters
        """
        self.method = method
        self.method_kwargs = method_kwargs

        # Initialize appropriate detector
        if method == "block_gradient":
            self.detector = BlockGradientDetector(**method_kwargs)
        elif method == "chodera":
            self.detector = ChoderDetector(**method_kwargs)
        elif method == "multiwindow":
            self.detector = MultiwindowDetector(**method_kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def detect_window_equilibrium(
        self, window: LambdaWindow, run_nos: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[float]]:
        """Detect equilibrium for a single lambda window."""
        if self.method == "multiwindow":
            raise ValueError("Use detect_leg_equilibrium for multiwindow method")

        return self.detector.detect_equilibrium(window, run_nos)

    def detect_leg_equilibrium(
        self, leg: Leg, run_nos: Optional[List[int]] = None
    ) -> Dict[str, Tuple[bool, Optional[float]]]:
        """
        Detect equilibrium for all windows in a leg.

        Returns
        -------
        Dict[str, Tuple[bool, Optional[float]]]
            Results for each window, keyed by "lambda_{state}"
        """
        results = {}

        if self.method == "multiwindow":
            # Multi-window methods analyze the entire leg at once
            equilibrated, equil_time = self.detector.detect_equilibrium(leg, run_nos)

            # Apply results to all windows
            for window in leg.lambda_windows:
                results[f"lambda_{window.lam_state}"] = (equilibrated, equil_time)

                # Update window attributes
                if hasattr(window, '_equilibrated'):
                    window._equilibrated = equilibrated
                if hasattr(window, '_equil_time') and equil_time is not None:
                    # Convert fractional time to absolute time for each window
                    total_time = self._estimate_window_total_time(window)
                    window._equil_time = equil_time * total_time
        else:
            # Single-window methods
            for window in leg.lambda_windows:
                try:
                    equilibrated, equil_time = self.detector.detect_equilibrium(
                        window, run_nos
                    )
                    results[f"lambda_{window.lam_state}"] = (equilibrated, equil_time)

                    # Update window attributes
                    if hasattr(window, '_equilibrated'):
                        window._equilibrated = equilibrated
                    if hasattr(window, '_equil_time'):
                        window._equil_time = equil_time

                except Exception as e:
                    logger.error(
                        f"Failed to detect equilibrium for window "
                        f"{window.lam_state}: {e}"
                    )
                    results[f"lambda_{window.lam_state}"] = (False, None)

        return results

    def detect_calculation_equilibrium(
        self, calculation: Calculation, run_nos: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, Tuple[bool, Optional[float]]]]:
        """
        Detect equilibrium for entire calculation.

        Returns
        -------
        Dict[str, Dict[str, Tuple[bool, Optional[float]]]]
            Results nested by leg_type and lambda state
        """
        all_results = {}

        for leg in calculation.legs:
            leg_name = leg.leg_type.name.lower()
            all_results[leg_name] = self.detect_leg_equilibrium(leg, run_nos)

        return all_results

    def _estimate_window_total_time(self, window: LambdaWindow) -> float:
        """Estimate total simulation time for a window (simplified)."""
        # This is a placeholder - in practice you'd read from XVG files
        # or use information from the simulation setup
        return 10.0  # ns


# Convenience functions for easy usage
def detect_equilibrium(
    target: Union[LambdaWindow, Leg, Calculation],
    method: str = "block_gradient",
    run_nos: Optional[List[int]] = None,
    **method_kwargs,
) -> Union[Tuple[bool, Optional[float]], Dict]:
    """
    Convenience function to detect equilibrium.

    Parameters
    ----------
    target : LambdaWindow, Leg, or Calculation
        Target to analyze
    method : str, default "block_gradient"
        Detection method
    run_nos : List[int], optional
        Run numbers to analyze
    **method_kwargs
        Method-specific parameters

    Returns
    -------
    Union[Tuple[bool, Optional[float]], Dict]
        Results depend on target type
    """
    manager = EquilibriumDetectionManager(method, **method_kwargs)

    if isinstance(target, LambdaWindow):
        return manager.detect_window_equilibrium(target, run_nos)
    elif isinstance(target, Leg):
        return manager.detect_leg_equilibrium(target, run_nos)
    elif isinstance(target, Calculation):
        return manager.detect_calculation_equilibrium(target, run_nos)
    else:
        raise ValueError(f"Unsupported target type: {type(target)}")
