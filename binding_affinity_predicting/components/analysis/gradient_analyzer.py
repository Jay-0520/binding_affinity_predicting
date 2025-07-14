"""
Equilibrium detection system for GROMACS FEP simulations.

This module provides comprehensive equilibrium detection methods adapted from a3fe
for use with the binding_affinity_predicting codebase. It integrates with the
existing GROMACS orchestration system.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from binding_affinity_predicting.components.analysis.autocorrelation import (
    _statistical_inefficiency_chodera,
)
from binding_affinity_predicting.components.analysis.xvg_data_loader import (
    GromacsXVGParser,
)
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    LambdaWindow,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GradientAnalyzer:
    """Analyzes dH/dλ gradients from GROMACS XVG files for equilibrium detection."""

    def __init__(self):
        self.xvg_parser = GromacsXVGParser()

    def read_gradients_from_window(
        self, window: LambdaWindow, run_nos: Optional[list[int]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Read dH/dλ gradients from all runs in a lambda window.

        Parameters
        ----------
        window : LambdaWindow
            Lambda window to read gradients from
        run_nos : List[int], optional
            Specific run numbers to read. If None, reads all runs.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            times, gradients arrays where each row corresponds to a run
        """
        if run_nos is None:
            run_nos = list(range(1, window.ensemble_size + 1))

        all_times = []
        all_gradients = []
        for run_no in run_nos:
            try:
                times, gradients = self._read_single_run_gradients(window, run_no)
                all_times.append(times)
                all_gradients.append(gradients)
            except Exception as e:
                logger.warning(f"Failed to read gradients for run {run_no}: {e}")
                continue

        if not all_times:
            raise ValueError(
                f"No valid gradient data found for window {window.lam_state}"
            )

        # Ensure all time series have the same length
        #  all repeats for a given lambda state should have the same number of time points
        #  but different lambda runs may have different lengths due to various simulation time
        min_length = min(len(times) for times in all_times)
        all_times = [times[:min_length] for times in all_times]
        all_gradients = [grads[:min_length] for grads in all_gradients]

        return np.array(all_times), np.array(all_gradients)

    def _read_single_run_gradients(
        self, window: LambdaWindow, run_no: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read gradients from a single run's XVG file."""
        run_dir = Path(window.output_dir) / f"run_{run_no}"
        xvg_file = run_dir / f"lambda_{window.lam_state}_run_{run_no}.xvg"

        if not xvg_file.exists():
            raise FileNotFoundError(f"XVG file not found: {xvg_file}")

        data = self.xvg_parser.parse_xvg_file(xvg_file)

        # Extract times (convert ps to ns); by default GROMACS times are in ps
        times = data['times'] / 1000.0

        # Calculate total dH/dλ (sum of all components)
        total_dhdl = np.zeros(len(times))  # shape (times,)
        for component_data in data['dhdl_components'].values():
            total_dhdl += component_data

        return times, total_dhdl

    def get_time_normalized_sems(
        self,
        lambda_windows: list[LambdaWindow],
        run_nos: Optional[list[int]] = None,
        origin: str = "inter_delta_g",
        smoothen: bool = True,
        # TODO: do we need to implement equilibrated data handling?
        equilibrated: bool = False,
    ) -> np.ndarray:
        """
        Calculate time-normalized standard errors of the mean for gradient data.

        This method implements the same algorithm as A3FE's GradientData.get_time_normalised_sems()
        but works directly with LambdaWindow objects.

        Parameters
        ----------
        lambda_windows : List[LambdaWindow]
            List of lambda windows to analyze
        run_nos : List[int], optional
            Run numbers to include in analysis. If None, uses all available runs.
            NOTE that at this stage, it should already be validated
        origin : str, default "inter_delta_g"
            Type of SEM to calculate:
            - "inter": Inter-run SEM of gradients
            - "intra": Intra-run SEM of gradients
            - "inter_delta_g": Inter-run SEM of free energy changes
        smoothen : bool, default True
            Whether to apply 3-point smoothing to the SEMs
        equilibrated : bool, default False
            Whether to use only equilibrated data

        Returns
        -------
        np.ndarray
            Time-normalized SEMs in units of kcal mol^-1 ns^(1/2)
        """
        if origin not in ["inter", "intra", "inter_delta_g"]:
            raise ValueError("origin must be 'inter', 'intra', or 'inter_delta_g'")

        # Analyze gradients for each window
        sems_by_window = []
        total_times = []

        for window in lambda_windows:
            try:
                # Read gradient data
                _, gradients = self.read_gradients_from_window(window, run_nos)

                # Calculate SEMs based on origin type
                if origin == "inter":
                    # SEM of gradients (dH/dλ)
                    sem = self._calculate_inter_run_sem(gradients)
                elif origin == "intra":
                    # Intra-run SEM of gradients
                    sem = self._calculate_intra_run_sem(gradients)
                elif origin == "inter_delta_g":
                    # SEM of free energy changes (ΔG)
                    sem = self._calculate_inter_run_sem(gradients)
                    # Convert to free energy space using lambda integration weight
                    if (
                        hasattr(window, 'lam_val_weight')
                        and window.lam_val_weight is not None
                    ):
                        sem *= window.lam_val_weight
                    else:
                        logger.warning(
                            f"No lambda weight for window {window.lam_state}, using 1.0"
                        )

                sems_by_window.append(sem)

                # Get total simulation time (sum across all runs)
                total_time = window.get_tot_simulation_time(run_nos)
                total_times.append(total_time)

            except Exception as e:
                logger.error(f"Failed to process window {window.lam_state}: {e}")
                # Use zero SEM for failed windows
                sems_by_window.append(0.0)
                total_times.append(1.0)  # Avoid division by zero

        sems = np.array(sems_by_window)
        total_times = np.array(total_times)

        # Time-normalize the SEMs: multiply by sqrt(total_time)
        # note that A3FE multiplies total time per window by number of runs
        normalized_sems = sems * np.sqrt(total_times)

        if not smoothen:
            return normalized_sems

        # Apply 3-point smoothing (same as A3FE)
        smoothed_sems = []
        max_ind = len(normalized_sems) - 1

        for i, sem in enumerate(normalized_sems):
            if i == 0:
                # First point: average with next point
                sem_smooth = (sem + normalized_sems[i + 1]) / 2
            elif i == max_ind:
                # Last point: average with previous point
                sem_smooth = (sem + normalized_sems[i - 1]) / 2
            else:
                # Middle points: average with neighbors
                sem_smooth = (sem + normalized_sems[i + 1] + normalized_sems[i - 1]) / 3
            smoothed_sems.append(sem_smooth)

        return np.array(smoothed_sems)

    def _calculate_inter_run_sem(self, gradients: np.ndarray) -> float:
        """
        Calculate inter-run (across multiple runs) standard error of the mean for
            a given lambda window.

        Interpretation:
            If I ran this lambda window multiple times, how much would the average gradient vary?

        Parameters
        ----------
        gradients : np.ndarray
            Gradient data with shape (n_runs, n_timepoints)

        Returns
        -------
        float
            Inter-run SEM
        """
        if gradients.ndim != 2:
            raise ValueError("Gradients must be 2D array (n_runs, n_timepoints)")

        # Calculate mean gradient for each run
        run_means = np.mean(gradients, axis=1)

        # Inter-run variance and SEM
        if len(run_means) > 1:
            # We should use sample variance here; different from A3FE which uses population variance
            inter_var = np.var(run_means, ddof=1)
            inter_sem = np.sqrt(inter_var / len(run_means))
        else:
            inter_sem = 0.0

        return inter_sem

    def _calculate_intra_run_sem(self, gradients: np.ndarray) -> float:
        """
        Calculate intra-run (within a single run) standard error of the mean for
            a given lambda window

        Interpretation:
            Given the natural fluctuations and correlations in my MD simulation, how precisely
            do I know the average gradient for this lambda window?


        Parameters
        ----------
        gradients : np.ndarray
            Gradient data with shape (n_runs, n_timepoints)

        Returns
        -------
        float
            Intra-run SEM
        """
        if gradients.ndim != 2:
            raise ValueError("Gradients must be 2D array (n_runs, n_timepoints)")

        # Calculate statistical inefficiency and variance for each run
        intra_vars = []
        squared_sems = []

        for run_grads in gradients:
            # Calculate statistical inefficiency
            stat_ineff = _statistical_inefficiency_chodera(run_grads)
            # Subsample to remove autocorrelation
            subsampled = run_grads[:: int(max(1, stat_ineff))]

            if len(subsampled) > 1:
                var = np.var(subsampled, ddof=1)
                squared_sem = var / len(subsampled)
            else:
                var = 0.0
                squared_sem = 0.0

            intra_vars.append(var)
            squared_sems.append(squared_sem)

        # Average across runs
        mean_squared_sem = np.mean(squared_sems) / len(gradients)
        intra_sem = np.sqrt(mean_squared_sem)

        return intra_sem
