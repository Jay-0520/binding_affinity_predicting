"""
Equilibrium detection system for GROMACS FEP simulations.

This module provides comprehensive equilibrium detection methods adapted from a3fe
for use with the binding_affinity_predicting codebase. It integrates with the
existing GROMACS orchestration system.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import kpss

from binding_affinity_predicting.components.analysis.autocorrelation import (
    _statistical_inefficiency_chodera,
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
