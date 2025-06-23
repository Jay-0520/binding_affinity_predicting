"""
Base simulation analysis framework.

This module provides the abstract base class and common utilities for all types of
molecular simulation analysis, including convergence analysis, statistical processing,
and result management.
"""

import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pymbar

from binding_affinity_predicting.components.simulation_base import SimulationRunner

logger = logging.getLogger(__name__)


class AnalysisResults:
    """Base container for simulation analysis results."""

    def __init__(self):
        self.simulation_results: dict[str, Dict] = {}
        self.statistical_analysis: dict = {}
        self.convergence_analysis: dict = {}
        self.metadata: Dict = {}
        self.analysis_timestamp: float = time.time()

    def add_simulation_result(self, sim_id: str, results: dict):
        """Add results for a specific simulation."""
        self.simulation_results[sim_id] = results

    def get_summary_df(self) -> pd.DataFrame:
        """Return a summary DataFrame of all results."""
        raise NotImplementedError("Subclasses must implement get_summary_df")

    def save_results(self, output_path: Union[str, Path]):
        """Save results to pickle file."""
        output_path = Path(output_path)
        with open(output_path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Results saved to {output_path}")

    @classmethod
    def load_results(cls, input_path: Union[str, Path]):
        """Load results from pickle file."""
        with open(input_path, 'rb') as f:
            return pickle.load(f)


class SimulationAnalyzer(ABC):
    """
    Abstract base class for simulation analysis.

    Provides common functionality for all types of simulation analysis:
    - File management and validation
    - Statistical analysis and decorrelation
    - Convergence analysis
    - Result organization and output

    Subclasses implement specific analysis methods for their simulation type.
    """

    def __init__(
        self,
        sim_runners: List[SimulationRunner],
        output_dir: str,
        temperature: float = 298.15,
        units: str = "kJ/mol",
    ):
        """
        Initialize simulation analyzer.

        Parameters:
        -----------
        sim_runners : List[SimulationRunner]
            List of simulation runners to analyze
        output_dir : str
            Directory for analysis output
        temperature : float
            Temperature in Kelvin
        units : str
            Energy units for output
        """
        self.sim_runners = sim_runners
        self.output_dir = Path(output_dir)
        self.temperature = temperature
        self.units = units

        # Physical constants
        self.kB = 8.314462618e-3  # kJ/(mol·K)
        self.beta = 1.0 / (self.kB * temperature)

        # Analysis utilities
        # self.stats = StatisticalUtils()
        # self.convergence = ConvergenceAnalyzer()
        self.xvg_file_parser = FileParser()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results container
        self.results = self._create_results_container()

        logger.info(
            f"Initialized {self.__class__.__name__} for {len(sim_runners)} simulations"
        )
        logger.info(f"Temperature: {temperature} K, Units: {units}")

    @abstractmethod
    def _create_results_container(self) -> AnalysisResults:
        """Create appropriate results container for this analysis type."""
        pass

    @abstractmethod
    def analyze(self) -> AnalysisResults:
        """
        Perform the complete analysis.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_results_df(self) -> pd.DataFrame:
        """
        Return results as a pandas DataFrame.
        Must be implemented by subclasses.
        """
        pass

    def _check_simulation_status(self) -> Dict[str, List[str]]:
        """
        Check simulation completion status.

        Returns:
        --------
        Dict with 'completed', 'failed', and 'missing' simulation lists
        """
        status = {'completed': [], 'failed': [], 'missing': []}

        for sim_runner in self.sim_runners:
            sim_id = getattr(sim_runner, 'name', str(sim_runner))

            if hasattr(sim_runner, 'finished') and hasattr(sim_runner, 'failed'):
                if sim_runner.failed:
                    status['failed'].append(sim_id)
                elif sim_runner.finished:
                    status['completed'].append(sim_id)
                else:
                    status['missing'].append(sim_id)
            else:
                # Check output directory existence as fallback
                output_dir = getattr(sim_runner, 'output_dir', None)
                if output_dir and Path(output_dir).exists():
                    status['completed'].append(sim_id)
                else:
                    status['missing'].append(sim_id)

        # Log status summary
        logger.info(
            f"Simulation status: {len(status['completed'])} completed, "
            f"{len(status['failed'])} failed, {len(status['missing'])} missing"
        )

        if status['failed']:
            logger.warning(f"Failed simulations: {', '.join(status['failed'][:5])}")
        if status['missing']:
            logger.warning(f"Missing simulations: {', '.join(status['missing'][:5])}")

        return status

    def analyze_convergence(self) -> Dict:
        """
        Analyze convergence for all simulations.
        Returns convergence analysis results.
        """
        convergence_results = {}

        for sim_runner in self.sim_runners:
            sim_id = getattr(sim_runner, 'name', str(sim_runner))
            try:
                conv_result = self._analyze_single_simulation_convergence(sim_runner)
                convergence_results[sim_id] = conv_result
            except Exception as e:
                logger.error(f"Convergence analysis failed for {sim_id}: {e}")
                convergence_results[sim_id] = {'error': str(e)}

        return convergence_results

    @abstractmethod
    def _analyze_single_simulation_convergence(
        self, sim_runner: SimulationRunner
    ) -> Dict:
        """
        Analyze convergence for a single simulation.
        Must be implemented by subclasses.
        """
        pass

    def save_results(self, filename: str = "analysis_results.pkl"):
        """Save analysis results to file."""
        output_path = self.output_dir / filename
        self.results.save_results(output_path)

    def generate_summary_report(self, filename: str = "analysis_summary.txt"):
        """Generate a text summary report."""
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{self.__class__.__name__.upper()} ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Temperature: {self.temperature} K\n")
            f.write(f"Units: {self.units}\n")
            f.write(f"Number of simulations: {len(self.sim_runners)}\n\n")

            # Simulation status
            status = self._check_simulation_status()
            f.write("SIMULATION STATUS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Completed: {len(status['completed'])}\n")
            f.write(f"Failed: {len(status['failed'])}\n")
            f.write(f"Missing: {len(status['missing'])}\n\n")

            # Results summary (to be implemented by subclasses)
            self._write_specific_summary(f)

    @abstractmethod
    def _write_specific_summary(self, file_handle):
        """Write analysis-specific summary content."""
        pass

    def time_statistics(self, start_time: float) -> Dict[str, str]:
        """
        Calculate timing statistics.
        Adapted from alchemical_analysis.py timeStatistics function.
        """
        end_time = time.time()
        elapsed = end_time - start_time

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60

        return {
            'hours': str(hours),
            'minutes': str(minutes),
            'seconds': f"{seconds:.2f}",
            'total_seconds': elapsed,
            'end_time': time.asctime(),
        }

    def get_metadata(self) -> Dict:
        """Get analysis metadata."""
        return {
            'analyzer_class': self.__class__.__name__,
            'temperature': self.temperature,
            'units': self.units,
            'n_simulations': len(self.sim_runners),
            'output_dir': str(self.output_dir),
            'analysis_timestamp': time.time(),
        }


# Utility functions adapted from alchemical_analysis.py


def checkUnitsAndMore(units: str, temperature: float) -> Tuple[str, float, float]:
    """
    Check units and calculate conversion factors.
    Adapted from alchemical_analysis.py checkUnitsAndMore function.
    """
    kB = 8.314462618e-3  # kJ/(mol·K)
    beta = 1.0 / (kB * temperature)

    if units.lower() in ['kj', 'kj/mol']:
        beta_report = beta
        units_str = '(kJ/mol)'
    elif units.lower() in ['kcal', 'kcal/mol']:
        beta_report = beta / 4.184
        units_str = '(kcal/mol)'
    elif units.lower() in ['kbt', 'k_bt']:
        beta_report = 1.0
        units_str = '(k_BT)'
    else:
        raise ValueError(f"Unknown unit type '{units}'. Supported: 'kJ', 'kcal', 'kBT'")

    return units_str, beta, beta_report


def get_methods_list(method_string: str = "") -> List[str]:
    """
    Parse method string and return list of analysis methods.
    Adapted from alchemical_analysis.py getMethods function.
    """
    all_methods = [
        'TI',
        'TI-CUBIC',
        'DEXP',
        'IEXP',
        'GINS',
        'GDEL',
        'BAR',
        'UBAR',
        'RBAR',
        'MBAR',
    ]
    default_methods = ['TI', 'TI-CUBIC', 'DEXP', 'IEXP', 'BAR', 'MBAR']

    if not method_string:
        return default_methods

    if method_string.upper() == 'ALL':
        return all_methods

    # Parse method string (simplified version)
    methods = method_string.replace('+', ' ').replace('_', '-').upper().split()
    valid_methods = [m for m in methods if m in all_methods]

    return valid_methods if valid_methods else default_methods
