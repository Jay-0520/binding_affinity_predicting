"""
Equilibrium detection system for GROMACS FEP simulations.

This module provides comprehensive equilibrium detection methods adapted from a3fe
for use with the binding_affinity_predicting codebase. It integrates with the
existing GROMACS orchestration system.
"""

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from binding_affinity_predicting.components.analysis.gradient_analyzer import (
    GradientAnalyzer,
)
from binding_affinity_predicting.components.analysis.utils import _compute_dg_mbar
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    Calculation,
    LambdaWindow,
    Leg,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EquilibriumBlockGradientDetector:
    """
    Check if the ensemble of simulations at the lambda window is
    equilibrated based on the ensemble gradient between averaged blocks.

    Equilibrium detection based on gradient of block-averaged dH/dλ.
    Adapted from check_equil_block_gradient() in a3fe.
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
        self,
        target: LambdaWindow,
        run_nos: Optional[list[int]] = None,
        save_result: bool = False,
        **kwargs,
    ) -> tuple[bool, Optional[float]]:
        """
        Detect equilibrium per single lambda window using block gradient method.

        Returns
        -------
        Tuple[bool, Optional[float]]
            (equilibrated, equilibration_time_ns)
        """
        if isinstance(target, list):
            raise ValueError(
                "Block gradient method can only analyze single LambdaWindow, not list"
            )

        window = target
        try:
            # Read gradient data
            times, gradients = self.gradient_analyzer.read_gradients_from_window(
                window, run_nos
            )

            # Calculate timestep from times array
            if len(times[0]) < 2:
                raise ValueError("Insufficient data points")

            # calculate number of indices in a block, functionally same as what
            # in A3FE's check_equil_block_gradient()
            timestep = times[0][1] - times[0][0]  # ns
            idx_block_size = int(self.block_size / timestep)

            # Calculate gradient of gradient for each run
            gradient_derivatives = []
            for run_gradients in gradients:
                d_dh_dl = self._calculate_gradient_derivative(
                    gradients=run_gradients, idx_block_size=idx_block_size
                )
                gradient_derivatives.append(d_dh_dl)

            # Calculate mean gradient derivative
            # TODO: should we use np.nanmean here?
            mean_gradient_derivative = np.mean(gradient_derivatives, axis=0)

            # Detect equilibration
            equilibrated, equil_time = self._find_equilibration_point(
                times[0], mean_gradient_derivative, idx_block_size
            )

            # Save results
            if save_result:
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

        # Calculate rolling average of gradients
        rolling_avg = self._get_rolling_average(
            data=gradients, window_size=idx_block_size
        )
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
        if window_size > len(data):
            raise ValueError(
                "Block size cannot be larger than the length of the data array."
            )

        rolling_avg = np.full(len(data), np.nan)

        for i in range(len(data)):
            if i < window_size:
                continue
            rolling_avg[i] = np.mean(data[i - window_size : i])

        return rolling_avg

    def _find_equilibration_point(
        self, times: np.ndarray, gradient_derivative: np.ndarray, idx_block_size: int
    ) -> tuple[bool, Optional[float]]:
        """Find the equilibration point based on gradient criteria.

        Returns True if equilibrated, False otherwise, and the equilibration time if found.
        """
        start_idx = 2 * idx_block_size

        if start_idx >= len(gradient_derivative):
            return False, None

        last_grad = gradient_derivative[start_idx]
        for i, grad in enumerate(gradient_derivative[start_idx:]):
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
        run_nos: Optional[list[int]],
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
        first_frac: float = 0.1,  # only for statistical test slicing
        last_frac: float = 0.5,  # only for statistical test slicing
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
        self,
        target: list[LambdaWindow],
        run_nos: Optional[list[int]] = None,
        **kwargs,
    ) -> tuple[bool, Optional[float]]:
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
        if isinstance(target, LambdaWindow):
            raise ValueError(
                "Multiwindow method requires list of LambdaWindow, not single window"
            )

        lambda_windows = target
        try:
            if self.method == "paired_t":
                return self._detect_paired_t_based(lambda_windows, run_nos)
            else:
                raise ValueError(f"Unsupported method: {self.method}")

        except Exception as e:
            logger.error(f"Multi-window equilibrium detection failed: {e}")
            return False, None

    def _get_time_series_multiwindow(
        self,
        lambda_windows: list[LambdaWindow],
        run_nos: list[int],
        start_frac: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def _get_time_series_multiwindow_mbar(
        self,
        lambda_windows: list[LambdaWindow],
        equilibrated: bool = False,
        run_nos: Optional[list[int]] = None,
        start_frac: float = 0.0,
        end_frac: float = 1.0,
        temperature: float = 298.15,
        units: str = "kcal",
        n_points: int = 100,
        use_multiprocessing: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get time series of free energy changes using MBAR analysis.

        This method performs MBAR calculations on sliding time windows to track
        convergence of the free energy estimate over simulation time.

        Parameters
        ----------
        lambda_windows : List[LambdaWindow]
            List of lambda windows to analyze
        equilibrated : bool, default False
            Whether to account for equilibration times
        run_nos : List[int], optional
            Run numbers to analyze. If None, analyzes all runs.
        start_frac : float, default 0.0
            Starting fraction of simulation time
        end_frac : float, default 1.0
            Ending fraction of simulation time
        temperature : float, default 298.15
            Temperature in Kelvin
        units : str, default "kcal"
            Units for free energy output
        n_points : int, default 100
            Number of time points to evaluate
        use_multiprocessing : bool, default True
            Whether to use multiprocessing for parallel MBAR calculations

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (overall_dgs, overall_times) arrays with shape (n_runs, n_points)
            overall_dgs contains cumulative free energy changes
            overall_times contains corresponding simulation times
        """
        # Validate inputs
        if not lambda_windows:
            raise ValueError("No lambda windows provided")

        # Check equilibration status if required
        if equilibrated and not all(
            getattr(lam, 'equilibrated', False) for lam in lambda_windows
        ):
            raise ValueError(
                "The equilibration times and statistics have not been set for all lambda "
                "windows in the stage. Please set these before running this function."
            )

        # Get valid run numbers
        if run_nos is None:
            run_nos = list(range(1, lambda_windows[0].ensemble_size + 1))

        # Validate run numbers
        max_runs = min(window.ensemble_size for window in lambda_windows)
        run_nos = [r for r in run_nos if 1 <= r <= max_runs]

        if not run_nos:
            raise ValueError("No valid run numbers found")

        n_runs = len(run_nos)
        # Initialize output arrays
        overall_dgs = np.zeros([n_runs, n_points])
        overall_times = np.zeros([n_runs, n_points])
        # Create time window fractions
        start_and_end_fracs = [
            (i, i + (end_frac - start_frac) / n_points)
            for i in np.linspace(start_frac, end_frac, n_points + 1)
        ][
            :-1
        ]  # Remove last point to avoid > 1
        # Round to avoid floating point errors
        start_and_end_fracs = [
            (round(x[0], 5), round(x[1], 5)) for x in start_and_end_fracs
        ]

        logger.info(
            f"Computing MBAR time series for {n_runs} runs, {n_points} time points"
        )

        if use_multiprocessing and len(run_nos) * len(start_and_end_fracs) > 10:
            # Use multiprocessing for large calculations
            try:
                with mp.get_context("spawn").Pool() as pool:
                    # Prepare arguments for parallel processing
                    args_list = [
                        (
                            run_no,
                            start_frac,
                            end_frac,
                            lambda_windows,
                            equilibrated,
                            temperature,
                            units,
                        )
                        for run_no in run_nos
                        for start_frac, end_frac in start_and_end_fracs
                    ]

                    # Compute MBAR for all time windows in parallel
                    results = pool.starmap(_compute_dg_mbar, args_list)

                    # Reshape results into output arrays
                    for i, run_no in enumerate(run_nos):
                        for j, (start_frac_val, end_frac_val) in enumerate(
                            start_and_end_fracs
                        ):
                            idx = i * len(start_and_end_fracs) + j
                            overall_dgs[i, j] = results[idx]

            except Exception as e:
                logger.warning(f"Multiprocessing failed, falling back to serial: {e}")
                use_multiprocessing = False

        if not use_multiprocessing:
            # Serial computation
            logger.info("Using serial computation for MBAR time series")
            for i, run_no in enumerate(run_nos):
                for j, (start_frac_val, end_frac_val) in enumerate(start_and_end_fracs):
                    overall_dgs[i, j] = _compute_dg_mbar(
                        run_no,
                        start_frac_val,
                        end_frac_val,
                        lambda_windows,
                        equilibrated,
                        temperature,
                        units,
                    )

        # Calculate times for each run
        for i, run_no in enumerate(run_nos):
            # Get total simulation time for this run
            total_time = sum(
                getattr(window, 'get_tot_simulation_time', lambda x: 1.0)([run_no])
                for window in lambda_windows
            )

            # Get equilibration time if applicable
            equil_time = 0.0
            if equilibrated:
                equil_time = sum(
                    getattr(window, 'equil_time', 0.0) for window in lambda_windows
                )

            # Calculate time points
            times = [
                (total_time - equil_time) * fracs[0] + equil_time
                for fracs in start_and_end_fracs
            ]
            overall_times[i] = times

        # Validate results
        if np.isnan(overall_dgs).any():
            logger.warning(
                "NaNs found in free energy changes - some MBAR calculations failed"
            )

        # Check time consistency across runs
        for i in range(1, n_runs):
            if not np.allclose(overall_times[i, -1], overall_times[0, -1], rtol=1e-3):
                logger.warning(
                    f"Total simulation times differ between runs: "
                    f"{overall_times[0, -1]:.3f} vs {overall_times[i, -1]:.3f}"
                )

        logger.info("MBAR time series computation completed")
        return overall_dgs, overall_times

    def _detect_paired_t_based(
        self,
        lambda_windows: list[LambdaWindow],
        run_nos: Optional[list[int]] = None,
        output_dir: Optional[str] = None,
    ) -> tuple[bool, Optional[float]]:
        """
        Paired t-test based detection - exact implementation from a3fe.

        Retures
        -------
        Tuple[bool, Optional[float]]
            (equilibrated, fractional_equilibration_time)
        """
        from scipy import stats

        if run_nos is None:
            run_nos = list(range(1, lambda_windows[0].ensemble_size + 1))

        # Initialize results storage
        p_vals_and_times = []
        equilibrated = False
        fractional_equil_time = None
        equil_time = None

        # Calculate test intervals
        start_fracs = np.linspace(0, 1 - self.last_frac, num=self.intervals)
        logger.info(
            f"Running paired t-test for {len(start_fracs)} intervals: {start_fracs}"
        )
        for start_frac in start_fracs:
            # Get time series data using MBAR-like approach
            overall_dgs, overall_times = self._get_time_series_multiwindow_mbar(
                lambda_windows=lambda_windows, run_nos=run_nos, start_frac=start_frac
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

            p_vals_and_times.append((p_value, overall_times[0][0]))

            # Check if equilibrated
            if p_value > self.p_cutoff and not equilibrated:
                equilibrated = True
                fractional_equil_time = start_frac
                equil_time = overall_times[0][0]

        # Update lambda window attributes if equilibrated
        if equilibrated and fractional_equil_time is not None:  # makes mypy happy
            for lam_win in lambda_windows:
                lam_win._equilibrated = True
                lam_win._equil_time = (
                    fractional_equil_time * lam_win.get_tot_simulation_time(run_nos=[1])
                )

        if output_dir is not None:
            self._save_paired_t_results(
                output_dir=output_dir,
                equilibrated=equilibrated,
                p_vals_and_times=p_vals_and_times,
                fractional_equil_time=fractional_equil_time,
                equil_time=equil_time,
                run_nos=run_nos,
            )

        return equilibrated, fractional_equil_time

    def _detect_gradient_based(
        self, lambda_windows: list[LambdaWindow], run_nos: Optional[list[int]] = None
    ) -> tuple[bool, Optional[float]]:
        """Gradient-based multi-window detection."""
        raise NotImplementedError(
            "Gradient-based detection is not implemented yet. "
            "Please use another method like 'paired_t' or 'kpss'."
        )

    def _detect_kpss_based(
        self, lambda_windows: list[LambdaWindow], run_nos: Optional[list[int]] = None
    ) -> tuple[bool, Optional[float]]:
        """KPSS stationarity test based detection."""
        raise NotImplementedError(
            "KPSS-based detection is not implemented yet. "
            "Please use another method like 'paired_t' or 'gradient'."
        )

    def _detect_geweke_based(
        self, lambda_windows: list[LambdaWindow], run_nos: Optional[list[int]] = None
    ) -> tuple[bool, Optional[float]]:
        """Modified Geweke test based detection."""
        raise NotImplementedError(
            "Geweke-based detection is not implemented yet. "
            "Please use another method like 'paired_t' or 'gradient'."
        )

    def _save_paired_t_results(
        self,
        equilibrated: bool,
        output_dir: str,
        p_vals_and_times: list[tuple[float, float]],
        fractional_equil_time: Optional[float],
        equil_time: Optional[float],
        run_nos: list[int],
    ):
        """Save paired t-test results to file."""
        output_file = Path(output_dir) / "check_equil_multiwindow_paired_t.txt"

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


class EquilibriumChoderDetector:
    """
    Chodera-style equilibrium detection based on statistical inefficiency.
    Adapted from check_equil_chodera in a3fe.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "Chodera-style equilibrium detection is not implemented yet. "
            "Please use another method like 'block_gradient' or 'multiwindow'."
        )


class EquilibriumDetectionManager:
    """
    High-level manager for equilibrium detection across different methods.
    """

    detector: Any  # noqa: E704

    def __init__(self, method: str = "multiwindow", **method_kwargs):
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
            self.detector = EquilibriumBlockGradientDetector(**method_kwargs)
        elif method == "multiwindow":
            self.detector = EquilibriumMultiwindowDetector(**method_kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def detect_window_equilibrium(
        self, window: LambdaWindow, run_nos: Optional[list[int]] = None, **kwargs
    ) -> tuple[bool, Optional[float]]:
        """Detect equilibrium for a single lambda window."""
        if self.method == "multiwindow":
            raise ValueError("Use detect_leg_equilibrium for multiwindow method")

        return self.detector.detect_equilibrium(window, run_nos=run_nos, **kwargs)

    def detect_leg_equilibrium(
        self, leg: Leg, run_nos: Optional[list[int]] = None, **kwargs
    ) -> dict[str, tuple[bool, Optional[float]]]:
        """
        Detect equilibrium for all windows in a leg.

        Returns
        -------
        Dict[str, Tuple[bool, Optional[float]]]
            Results for each window, keyed by "lambda_{state}"
        """
        lambda_windows = leg.lambda_windows
        results = {}

        if self.method == "multiwindow":
            # Multi-window methods analyze the entire leg at once
            equilibrated, frac_equil_time = self.detector.detect_equilibrium(
                target=lambda_windows, run_nos=run_nos, **kwargs
            )

            # Apply results to all windows
            for window in leg.lambda_windows:
                results[f"lambda_{window.lam_state}"] = (equilibrated, frac_equil_time)
                window._equilibrated = equilibrated

                # Convert fractional equilibration time to absolute time for this specific window
                # TODO: but get_tot_simulation_time() returns the sum of all runs?
                total_time = window.get_tot_simulation_time(run_nos or [1])
                if frac_equil_time is not None:  # makes mypy happy
                    window._equil_time = frac_equil_time * total_time
                else:
                    window._equil_time = None
        else:
            # Single-window methods
            for window in leg.lambda_windows:
                try:
                    equilibrated, frac_equil_time = self.detector.detect_equilibrium(
                        target=window, run_nos=run_nos, **kwargs
                    )
                    results[f"lambda_{window.lam_state}"] = (
                        equilibrated,
                        frac_equil_time,
                    )

                    window._equilibrated = equilibrated
                    window._equil_time = frac_equil_time

                except Exception as e:
                    logger.error(
                        f"Failed to detect equilibrium for window "
                        f"{window.lam_state}: {e}"
                    )
                    results[f"lambda_{window.lam_state}"] = (False, None)

        return results

    def detect_calculation_equilibrium(
        self, calculation: Calculation, run_nos: Optional[list[int]] = None, **kwargs
    ) -> dict[str, dict[str, tuple[bool, Optional[float]]]]:
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
            all_results[leg_name] = self.detect_leg_equilibrium(leg, run_nos, **kwargs)

        return all_results


# Convenience functions for easy usage
def detect_equilibrium(
    target: Union[LambdaWindow, Leg, Calculation, list[LambdaWindow]],
    method: str = "multiwindow",
    run_nos: Optional[list[int]] = None,
    **method_kwargs,
) -> Union[tuple[bool, Optional[float]], dict]:
    """
    Convenience function to detect equilibrium.

    Parameters
    ----------
    target : LambdaWindow, Leg, or Calculation
        Target to analyze
    method : str, default "multiwindow"
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
    if isinstance(target, LambdaWindow) and method == "multiwindow":
        raise ValueError(
            "Multiwindow method cannot be used with single LambdaWindow. "
            "Use method='block_gradient' for single windows, or pass a Leg/list of windows."
        )

    if isinstance(target, list) and method != "multiwindow":
        raise ValueError(
            f"Method '{method}' cannot be used with list of LambdaWindow objects. "
            "Use method='multiwindow' or pass individual windows."
        )

    manager = EquilibriumDetectionManager(method, **method_kwargs)

    if isinstance(target, LambdaWindow):
        return manager.detect_window_equilibrium(target, run_nos, **method_kwargs)

    elif isinstance(target, Leg):
        return manager.detect_leg_equilibrium(target, run_nos, **method_kwargs)

    elif isinstance(target, Calculation):
        return manager.detect_calculation_equilibrium(target, run_nos, **method_kwargs)

    elif isinstance(target, list):
        # Handle list of LambdaWindow objects for multiwindow analysis
        if not target:
            raise ValueError("Empty list of lambda windows provided")

        if not all(isinstance(window, LambdaWindow) for window in target):
            raise ValueError("All items in target list must be LambdaWindow objects")

        # For multiwindow method, pass the list directly to the detector
        if method == "multiwindow":
            return manager.detector.detect_equilibrium(target, run_nos)
        else:
            raise ValueError(f"Method '{method}' not supported for list of windows")

    else:
        raise ValueError(
            f"Unsupported target type: {type(target)}. "
            f"Expected LambdaWindow, Leg, Calculation, or list[LambdaWindow]"
        )
