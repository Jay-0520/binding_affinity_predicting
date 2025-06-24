"""
Comprehensive Free Energy Perturbation (FEP) Analysis Module.

This module implements specific FEP analysis methods for analyzing GROMACS
FEP simulation results. It inherits common functionality from SimulationAnalyzer
and uses free energy estimation methods from free_energy_estimation_methods.py.

Key Features:
1. Automated dH/dλ file parsing and data collection
2. Statistical analysis and decorrelation
3. Multiple free energy estimation methods (TI, BAR, MBAR, EXP)
4. Convergence analysis and error estimation
5. Binding affinity calculation from leg differences
6. Comprehensive reporting and visualization
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from binding_affinity_predicting.components.analysis.free_energy_estimation_methods import (
    BennettAcceptanceRatio,
    ExponentialAveraging,
    FreeEnergyEstimator,
    MultistateBAR,
    ThermodynamicIntegration,
)
from binding_affinity_predicting.components.analysis.simulation_analyzer_base import (
    AnalysisResults,
    ConvergenceAnalyzer,
    SimulationAnalyzer,
    StatisticalUtils,
)
from binding_affinity_predicting.components.gromacs_orchestration import (
    Calculation,
    LambdaWindow,
    Leg,
)
from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.components.utils import (
    create_method_summary_table,
    format_energy_with_error,
    get_methods_list,
    make_json_serializable,
)
from binding_affinity_predicting.components.utils.xvg_data_loader import (
    GromacsXVGParser,
)
from binding_affinity_predicting.data.enums import LegType

logger = logging.getLogger(__name__)


class FEPAnalysisResults(AnalysisResults):
    """Container for comprehensive FEP analysis results."""

    def __init__(self):
        super().__init__()
        self.leg_results: Dict[str, Dict] = {}
        self.binding_affinity: Optional[float] = None
        self.binding_affinity_error: Optional[float] = None
        self.method_results: Dict[str, Dict] = {}

    def add_leg_result(self, leg_type: str, results: Dict):
        """Add results for a specific leg (BOUND/FREE)."""
        self.leg_results[leg_type] = results
        logger.debug(f"Added results for {leg_type} leg")

    def calculate_binding_affinity(self, preferred_method: str = 'MBAR'):
        """
        Calculate binding affinity from leg results.

        Parameters:
        -----------
        preferred_method : str
            Preferred method for binding affinity calculation
        """
        if 'BOUND' not in self.leg_results or 'FREE' not in self.leg_results:
            logger.warning("Cannot calculate binding affinity: missing leg results")
            return

        bound_results = self.leg_results['BOUND']
        free_results = self.leg_results['FREE']

        # Try preferred method first, then fallback to available methods
        methods_to_try = [preferred_method, 'BAR', 'TI_CUBIC', 'TI', 'DEXP']

        for method in methods_to_try:
            if (
                method in bound_results
                and method in free_results
                and 'total_dg' in bound_results[method]
                and 'total_dg' in free_results[method]
            ):

                bound_dg = bound_results[method]['total_dg']
                bound_error = bound_results[method]['total_error']
                free_dg = free_results[method]['total_dg']
                free_error = free_results[method]['total_error']

                # ΔG_binding = ΔG_bound - ΔG_free
                self.binding_affinity = bound_dg - free_dg
                self.binding_affinity_error = np.sqrt(bound_error**2 + free_error**2)

                self.method_results['binding'] = {
                    'method': method,
                    'bound_dg': bound_dg,
                    'bound_error': bound_error,
                    'free_dg': free_dg,
                    'free_error': free_error,
                    'binding_affinity': self.binding_affinity,
                    'binding_affinity_error': self.binding_affinity_error,
                }

                logger.info(
                    f"Binding affinity calculated using {method}: "
                    f"{self.binding_affinity:.3f} ± {self.binding_affinity_error:.3f} kJ/mol"
                )
                return

        logger.warning(
            "Could not calculate binding affinity: no compatible method results found"
        )

    def get_summary_df(self) -> pd.DataFrame:
        """Return a summary DataFrame of all results."""
        rows = []

        # Add leg results
        for leg_name, leg_data in self.leg_results.items():
            for method, result in leg_data.items():
                if isinstance(result, dict) and 'total_dg' in result:
                    rows.append(
                        {
                            'leg': leg_name,
                            'method': method,
                            'free_energy_kJ_mol': result['total_dg'],
                            'error_kJ_mol': result['total_error'],
                            'description': result.get('method', method),
                        }
                    )

        # Add binding affinity
        if self.binding_affinity is not None:
            binding_data = self.method_results.get('binding', {})
            rows.append(
                {
                    'leg': 'BINDING',
                    'method': binding_data.get('method', 'UNKNOWN'),
                    'free_energy_kJ_mol': self.binding_affinity,
                    'error_kJ_mol': self.binding_affinity_error,
                    'description': 'Binding Affinity (ΔG_bound - ΔG_free)',
                }
            )

        return pd.DataFrame(rows)


class GromacsFEPAnalyzer(SimulationAnalyzer):
    """
    Comprehensive GROMACS FEP analyzer implementing multiple free energy estimation methods.

    This analyzer provides a complete workflow for FEP analysis:
    1. Data collection and parsing from GROMACS output
    2. Statistical analysis and decorrelation
    3. Free energy estimation using multiple methods
    4. Convergence analysis
    5. Binding affinity calculation
    6. Comprehensive reporting
    """

    def __init__(
        self,
        sim_runners: List[SimulationRunner],
        output_dir: str,
        temperature: float = 298.15,
        equilibration_time: float = 1.0,
        units: str = "kJ/mol",
        methods: Optional[List[str]] = None,
    ):
        """
        Initialize comprehensive FEP analyzer.

        Parameters:
        -----------
        sim_runners : List[SimulationRunner]
            List of simulation runners (typically Calculation objects)
        output_dir : str
            Directory for analysis output
        temperature : float
            Temperature in Kelvin
        equilibration_time : float
            Equilibration time to skip (ns)
        units : str
            Energy units for output
        methods : List[str], optional
            Analysis methods to use (default: ['TI', 'TI-CUBIC', 'BAR', 'MBAR'])
        """
        super().__init__(sim_runners, output_dir, temperature, units)

        self.equilibration_time = equilibration_time
        self.methods = (
            get_methods_list(''.join(methods)) if methods else get_methods_list()
        )

        # Initialize analysis tools
        self.fe_estimator = FreeEnergyEstimator(temperature)
        self.xvg_parser = XVGParser()

        logger.info(f"Initialized FEP analyzer with methods: {', '.join(self.methods)}")
        logger.info(
            f"Temperature: {temperature}K, Equilibration: {equilibration_time}ns"
        )

    def _create_results_container(self) -> FEPAnalysisResults:
        """Create FEP-specific results container."""
        return FEPAnalysisResults()

    def analyze(self) -> FEPAnalysisResults:
        """
        Perform comprehensive FEP analysis using multiple methods.
        """
        start_time = time.time()
        logger.info("Starting comprehensive FEP analysis...")

        # Check simulation status
        sim_status = self._check_simulation_status()

        # Analyze each calculation
        for sim_runner in self.sim_runners:
            if isinstance(sim_runner, Calculation):
                try:
                    self._analyze_calculation(sim_runner)
                except Exception as e:
                    logger.error(f"Failed to analyze calculation {sim_runner}: {e}")
                    raise

        # Calculate binding affinity
        self.results.calculate_binding_affinity()

        # Add metadata
        timing = self.time_statistics(start_time)
        self.results.metadata.update(
            {
                'analysis_timing': timing,
                'simulation_status': sim_status,
                'methods_used': self.methods,
                'equilibration_time': self.equilibration_time,
            }
        )

        # Generate comprehensive output
        self._generate_analysis_output()

        logger.info(f"FEP analysis completed in {timing['total_seconds']:.1f} seconds")
        return self.results

    def _analyze_calculation(self, calculation: Calculation):
        """Analyze a complete calculation (all legs)."""
        logger.info(f"Analyzing calculation with {len(calculation.legs)} legs")

        for leg in calculation.legs:
            try:
                leg_results = self._analyze_leg(leg)
                self.results.add_leg_result(leg.leg_type.name, leg_results)
                logger.info(f"Completed analysis for {leg.leg_type.name} leg")
            except Exception as e:
                logger.error(f"Failed to analyze leg {leg.leg_type.name}: {e}")
                raise

    def _analyze_leg(self, leg: Leg) -> Dict:
        """
        Analyze a single leg using all available methods.

        Parameters:
        -----------
        leg : Leg
            Leg object containing lambda windows

        Returns:
        --------
        Dict : Analysis results for the leg
        """
        logger.info(
            f"Analyzing {leg.leg_type.name} leg with {len(leg.lambda_windows)} windows"
        )

        # Create output directory for this leg
        leg_output_dir = self.output_dir / f"{leg.leg_type.name.lower()}_analysis"
        leg_output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Collect and parse all dH/dλ data
        lambda_data = self._collect_lambda_data(leg)

        if not lambda_data:
            raise ValueError(f"No valid data found for {leg.leg_type.name} leg")

        # Step 2: Statistical analysis and decorrelation
        processed_data = self._process_statistical_data(lambda_data)

        # Step 3: Run all analysis methods
        results = {}

        # Thermodynamic Integration
        if any(method.startswith('TI') for method in self.methods):
            ti_results = self._run_ti_analysis(processed_data, leg)
            results.update(ti_results)

        # Exponential averaging methods
        if any(method in ['DEXP', 'IEXP', 'GDEL', 'GINS'] for method in self.methods):
            exp_results = self._run_exp_analysis(processed_data)
            results.update(exp_results)

        # BAR analysis
        if any(method in ['BAR', 'UBAR', 'RBAR'] for method in self.methods):
            bar_results = self._run_bar_analysis(processed_data)
            results.update(bar_results)

        # MBAR analysis
        if 'MBAR' in self.methods:
            mbar_results = self._run_mbar_analysis(processed_data)
            results.update(mbar_results)

        # Step 4: Convergence analysis
        convergence_results = self._analyze_leg_convergence(lambda_data)
        results['convergence_analysis'] = convergence_results

        # Step 5: Save leg-specific results
        self._save_leg_results(results, leg_output_dir, leg.leg_type.name)

        return results

    def _collect_lambda_data(self, leg: Leg) -> Dict:
        """
        Collect dH/dλ data from all lambda windows and runs.

        Parameters:
        -----------
        leg : Leg
            Leg object containing lambda windows

        Returns:
        --------
        Dict : Collected lambda data organized by state
        """
        lambda_data = {}

        for window in leg.lambda_windows:
            lam_state = window.lam_state

            # Get lambda values for this state
            lambda_values = self._get_lambda_values_for_state(leg, lam_state)

            # Collect data from all runs
            window_data = []
            for run_no in range(1, window.ensemble_size + 1):
                run_dir = Path(window.output_dir) / f"run_{run_no}"

                # Look for dH/dλ files (multiple possible names)
                possible_files = [
                    run_dir / f"lambda_{lam_state}_run_{run_no}.xvg",
                    run_dir / f"dhdl.xvg",
                    run_dir / f"dhdl_{lam_state}.xvg",
                ]

                xvg_file = None
                for file_path in possible_files:
                    if file_path.exists():
                        xvg_file = file_path
                        break

                if xvg_file and self.xvg_parser.validate_xvg_file(xvg_file):
                    try:
                        times, dhdl_total, components = self.xvg_parser.parse_dhdl_file(
                            xvg_file,
                            skip_time=self.equilibration_time * 1000,  # ns to ps
                        )

                        if len(times) > 0:
                            run_data = {
                                'times': times,
                                'dhdl_total': dhdl_total,
                                'components': components,
                                'n_samples': len(times),
                                'run_number': run_no,
                                'file_path': str(xvg_file),
                            }
                            window_data.append(run_data)

                    except Exception as e:
                        logger.warning(f"Error parsing {xvg_file}: {e}")
                else:
                    logger.warning(
                        f"No valid dH/dλ file found for window {lam_state}, run {run_no}"
                    )

            if window_data:
                lambda_data[lam_state] = {
                    'lambda_values': lambda_values,
                    'runs': window_data,
                    'n_runs': len(window_data),
                    'total_samples': sum(run['n_samples'] for run in window_data),
                }
                logger.debug(
                    f"Collected {len(window_data)} runs for lambda state {lam_state}"
                )

        logger.info(f"Collected data from {len(lambda_data)} lambda states")
        return lambda_data

    def _get_lambda_values_for_state(
        self, leg: Leg, lam_state: int
    ) -> Dict[str, float]:
        """
        Get the lambda values for a specific state.

        Parameters:
        -----------
        leg : Leg
            Leg object
        lam_state : int
            Lambda state index

        Returns:
        --------
        Dict[str, float] : Lambda values for different coordinates
        """
        leg_type = leg.leg_type

        # Try to get lambda vectors from simulation config
        if hasattr(leg, 'sim_config') and leg.sim_config:
            config = leg.sim_config

            # Get lambda values with bounds checking
            def safe_get_lambda(lambda_list, index, default=0.0):
                if isinstance(lambda_list, dict) and leg_type in lambda_list:
                    lam_vals = lambda_list[leg_type]
                    return lam_vals[index] if index < len(lam_vals) else default
                elif isinstance(lambda_list, list):
                    return lambda_list[index] if index < len(lambda_list) else default
                return default

            bonded = safe_get_lambda(getattr(config, 'bonded_lambdas', {}), lam_state)
            coul = safe_get_lambda(getattr(config, 'coul_lambdas', {}), lam_state)
            vdw = safe_get_lambda(getattr(config, 'vdw_lambdas', {}), lam_state)

            return {'bonded': bonded, 'coulomb': coul, 'vdw': vdw}
        else:
            # Fallback: generate reasonable lambda values
            n_states = len(leg.lambda_windows)
            if n_states > 1:
                lambda_frac = lam_state / (n_states - 1)
            else:
                lambda_frac = 0.0

            return {'bonded': 0.0, 'coulomb': lambda_frac, 'vdw': 0.0}

    def _process_statistical_data(self, lambda_data: Dict) -> Dict:
        """
        Process data with statistical analysis and decorrelation.

        Parameters:
        -----------
        lambda_data : Dict
            Raw lambda data

        Returns:
        --------
        Dict : Processed data with statistical analysis
        """
        processed = {}

        for lam_state, state_data in lambda_data.items():
            runs_data = state_data['runs']

            # Combine all runs for this state
            all_dhdl = []
            run_stats = []

            for run_data in runs_data:
                dhdl = run_data['dhdl_total']

                # Statistical inefficiency analysis
                g = self.stats.statistical_inefficiency(dhdl)

                # Get uncorrelated indices
                uncorr_indices = self.stats.subsample_correlated_data(dhdl, g)
                uncorr_dhdl = dhdl[uncorr_indices]

                all_dhdl.extend(uncorr_dhdl)

                # Per-run statistics
                run_stat = {
                    'run_number': run_data['run_number'],
                    'statistical_inefficiency': g,
                    'n_correlated': len(dhdl),
                    'n_uncorrelated': len(uncorr_dhdl),
                    'mean_dhdl': float(np.mean(uncorr_dhdl)),
                    'std_dhdl': float(np.std(uncorr_dhdl)),
                    'sem_dhdl': float(np.std(uncorr_dhdl) / np.sqrt(len(uncorr_dhdl))),
                }
                run_stats.append(run_stat)

                # Update run data with statistical analysis
                run_data.update(run_stat)

            # Combined statistics across all runs
            all_dhdl = np.array(all_dhdl)
            run_means = [stat['mean_dhdl'] for stat in run_stats]

            processed[lam_state] = {
                'lambda_values': state_data['lambda_values'],
                'runs_data': runs_data,
                'run_statistics': run_stats,
                'combined_dhdl': all_dhdl,
                'mean_dhdl': float(np.mean(all_dhdl)),
                'std_dhdl': float(np.std(all_dhdl)),
                'sem_intra': float(np.std(all_dhdl) / np.sqrt(len(all_dhdl))),
                'sem_inter': (
                    float(np.std(run_means) / np.sqrt(len(run_means)))
                    if len(run_means) > 1
                    else 0.0
                ),
                'n_runs': len(runs_data),
                'n_total_samples': len(all_dhdl),
            }

        return processed

    def _run_ti_analysis(self, processed_data: Dict, leg: Leg) -> Dict:
        """
        Run thermodynamic integration analysis.
        """
        logger.info("Running TI analysis...")

        # Extract lambda values and dH/dλ statistics
        states = sorted(processed_data.keys())

        # Determine which lambda coordinate is varying
        varying_lambda = self._determine_varying_lambda(processed_data, leg)
        logger.debug(f"Detected varying lambda coordinate: {varying_lambda}")

        # Build arrays for integration
        lambda_points = []
        dhdl_means = []
        dhdl_errors = []

        for state in states:
            data = processed_data[state]
            lambda_val = data['lambda_values'][varying_lambda]
            lambda_points.append(lambda_val)
            dhdl_means.append(data['mean_dhdl'])
            # Use inter-run SEM if available, otherwise intra-run SEM
            error = data['sem_inter'] if data['sem_inter'] > 0 else data['sem_intra']
            dhdl_errors.append(error)

        lambda_points = np.array(lambda_points)
        dhdl_means = np.array(dhdl_means)
        dhdl_errors = np.array(dhdl_errors)

        results = {}

        # Trapezoidal integration
        if 'TI' in self.methods:
            try:
                ti_result = self.fe_estimator.estimate_ti(
                    lambda_points, dhdl_means, dhdl_errors, method='trapezoidal'
                )
                if ti_result['success']:
                    results['TI'] = {
                        'total_dg': ti_result['free_energy'],
                        'total_error': ti_result['error'],
                        'method': 'Thermodynamic Integration (Trapezoidal)',
                        'integration_method': 'trapezoidal',
                        'lambda_coordinate': varying_lambda,
                        'lambda_values': lambda_points.tolist(),
                        'dhdl_means': dhdl_means.tolist(),
                        'dhdl_errors': dhdl_errors.tolist(),
                        'n_points': len(lambda_points),
                    }
                    logger.info(
                        f"TI (Trapezoidal): ΔG = {ti_result['free_energy']:.3f} ± {ti_result['error']:.3f} kJ/mol"
                    )
            except Exception as e:
                logger.error(f"TI trapezoidal integration failed: {e}")

        # Cubic spline integration
        if 'TI-CUBIC' in self.methods:
            try:
                ti_result = self.fe_estimator.estimate_ti(
                    lambda_points, dhdl_means, dhdl_errors, method='cubic'
                )
                if ti_result['success']:
                    results['TI_CUBIC'] = {
                        'total_dg': ti_result['free_energy'],
                        'total_error': ti_result['error'],
                        'method': 'Thermodynamic Integration (Cubic Spline)',
                        'integration_method': 'cubic_spline',
                        'lambda_coordinate': varying_lambda,
                        'lambda_values': lambda_points.tolist(),
                        'dhdl_means': dhdl_means.tolist(),
                        'dhdl_errors': dhdl_errors.tolist(),
                        'n_points': len(lambda_points),
                    }
                    logger.info(
                        f"TI (Cubic): ΔG = {ti_result['free_energy']:.3f} ± {ti_result['error']:.3f} kJ/mol"
                    )
            except Exception as e:
                logger.warning(f"TI cubic spline integration failed: {e}")

        return results

    def _run_exp_analysis(self, processed_data: Dict) -> Dict:
        """
        Run exponential averaging analysis.
        """
        logger.info("Running EXP analysis...")

        results = {}
        states = sorted(processed_data.keys())

        # For adjacent state pairs
        for i in range(len(states) - 1):
            state_i = states[i]
            state_j = states[i + 1]

            try:
                # Use dH/dλ data as work approximation
                work_forward = processed_data[state_i]['combined_dhdl']
                work_reverse = processed_data[state_j]['combined_dhdl']

                if len(work_forward) > 0:
                    # Forward exponential averaging (DEXP)
                    if 'DEXP' in self.methods:
                        exp_result = self.fe_estimator.estimate_exp(
                            work_forward, method='forward'
                        )
                        if exp_result['success']:
                            results[f'DEXP_{state_i}_{state_j}'] = {
                                'dg': exp_result['free_energy'],
                                'dg_error': exp_result['error'],
                                'method': f'Gaussian Forward EXP ({state_i}→{state_j})',
                                'n_samples': len(work_forward),
                            }

                if 'GINS' in self.methods and len(work_reverse) > 0:
                    exp_result = self.fe_estimator.estimate_exp(
                        work_forward, work_reverse, method='gaussian_reverse'
                    )
                    if exp_result['success']:
                        results[f'GINS_{state_i}_{state_j}'] = {
                            'dg': exp_result['free_energy'],
                            'dg_error': exp_result['error'],
                            'method': f'Gaussian Reverse EXP ({state_j}→{state_i})',
                            'n_samples': len(work_reverse),
                        }

            except Exception as e:
                logger.warning(
                    f"EXP analysis failed for states {state_i}-{state_j}: {e}"
                )

        return results

    def _run_bar_analysis(self, processed_data: Dict) -> Dict:
        """
        Run BAR analysis between adjacent states.
        """
        logger.info("Running BAR analysis...")

        results = {}
        states = sorted(processed_data.keys())

        total_dg = 0.0
        total_error_sq = 0.0
        n_transitions = 0

        for i in range(len(states) - 1):
            state_i = states[i]
            state_j = states[i + 1]

            try:
                # Use dH/dλ data as work approximation
                work_forward = processed_data[state_i]['combined_dhdl']
                work_reverse = processed_data[state_j]['combined_dhdl']

                if len(work_forward) > 0 and len(work_reverse) > 0:
                    # Standard BAR
                    if 'BAR' in self.methods:
                        bar_result = self.fe_estimator.estimate_bar(
                            work_forward, work_reverse, method='standard'
                        )
                        if bar_result['success']:
                            total_dg += bar_result['free_energy']
                            total_error_sq += bar_result['error'] ** 2
                            n_transitions += 1

                            results[f'BAR_{state_i}_{state_j}'] = {
                                'dg': bar_result['free_energy'],
                                'dg_error': bar_result['error'],
                                'method': f'BAR ({state_i}→{state_j})',
                                'n_forward': len(work_forward),
                                'n_reverse': len(work_reverse),
                            }

                    # Unoptimized BAR
                    if 'UBAR' in self.methods:
                        bar_result = self.fe_estimator.estimate_bar(
                            work_forward, work_reverse, method='unoptimized'
                        )
                        if bar_result['success']:
                            results[f'UBAR_{state_i}_{state_j}'] = {
                                'dg': bar_result['free_energy'],
                                'dg_error': bar_result['error'],
                                'method': f'Unoptimized BAR ({state_i}→{state_j})',
                                'n_forward': len(work_forward),
                                'n_reverse': len(work_reverse),
                            }

                    # Range-optimized BAR
                    if 'RBAR' in self.methods:
                        bar_result = self.fe_estimator.estimate_bar(
                            work_forward, work_reverse, method='range_optimized'
                        )
                        if bar_result['success']:
                            results[f'RBAR_{state_i}_{state_j}'] = {
                                'dg': bar_result['free_energy'],
                                'dg_error': bar_result['error'],
                                'method': f'Range-optimized BAR ({state_i}→{state_j})',
                                'n_forward': len(work_forward),
                                'n_reverse': len(work_reverse),
                            }

            except Exception as e:
                logger.warning(
                    f"BAR analysis failed for states {state_i}-{state_j}: {e}"
                )

        # Total BAR result (sum of adjacent transitions)
        if n_transitions > 0 and 'BAR' in self.methods:
            results['BAR'] = {
                'total_dg': total_dg,
                'total_error': np.sqrt(total_error_sq),
                'method': 'Bennett Acceptance Ratio (summed)',
                'n_transitions': n_transitions,
            }
            logger.info(
                f"BAR: ΔG = {total_dg:.3f} ± {np.sqrt(total_error_sq):.3f} kJ/mol"
            )

        return results

    def _run_mbar_analysis(self, processed_data: Dict) -> Dict:
        """
        Run MBAR analysis using all states simultaneously.
        """
        logger.info("Running MBAR analysis...")

        try:
            states = sorted(processed_data.keys())
            K = len(states)

            # This is a simplified implementation
            # Real MBAR requires potential energies evaluated at all lambda states
            logger.warning(
                "MBAR implementation is simplified - requires full potential energy evaluations"
            )

            # Estimate based on available dH/dλ data (not rigorous!)
            N_k = np.array(
                [processed_data[state]['n_total_samples'] for state in states]
            )

            # Create approximate u_kln matrix
            max_samples = max(N_k)
            u_kln = np.zeros((K, K, max_samples))

            # Very rough approximation - don't use for production!
            for i, state_i in enumerate(states):
                dhdl_i = processed_data[state_i]['combined_dhdl']
                n_samples = len(dhdl_i)

                for j, state_j in enumerate(states):
                    if i == j:
                        u_kln[i, j, :n_samples] = 0.0  # Reference state
                    else:
                        # Rough approximation using cumulative dH/dλ
                        u_kln[i, j, :n_samples] = (
                            dhdl_i[:n_samples] * abs(i - j) * self.beta
                        )

            # Run MBAR
            mbar_result = self.fe_estimator.estimate_mbar(
                u_kln,
                N_k,
                relative_tolerance=1e-10,
                verbose=False,
                compute_overlap=True,
            )

            if mbar_result['success']:
                result = {
                    'MBAR': {
                        'total_dg': mbar_result['total_dg'],
                        'total_error': mbar_result['total_error'],
                        'method': 'Multistate Bennett Acceptance Ratio',
                        'free_energies': mbar_result['free_energies'].tolist(),
                        'free_energy_errors': mbar_result[
                            'free_energy_errors'
                        ].tolist(),
                        'n_states': K,
                        'overlap_matrix': mbar_result.get('overlap_matrix', None),
                    }
                }

                logger.info(
                    f"MBAR: ΔG = {mbar_result['total_dg']:.3f} ± {mbar_result['total_error']:.3f} kJ/mol"
                )
                logger.warning(
                    "MBAR result is approximate - full implementation requires potential energy evaluations"
                )

                return result
            else:
                logger.error("MBAR estimation failed")
                return {}

        except Exception as e:
            logger.error(f"MBAR analysis failed: {e}")
            return {}

    def _analyze_leg_convergence(self, lambda_data: Dict) -> Dict:
        """
        Analyze convergence for a single leg.
        """
        convergence = {}

        for lam_state, state_data in lambda_data.items():
            runs = state_data['runs']

            # Collect all dH/dλ time series from runs
            replica_data = [run['dhdl_total'] for run in runs]

            if replica_data:
                # Analyze convergence within and between replicas
                if len(replica_data) == 1:
                    # Single replica analysis
                    single_conv = self.convergence.analyze_timeseries_convergence(
                        replica_data[0]
                    )
                    convergence[lam_state] = {
                        'single_replica': single_conv,
                        'n_replicas': 1,
                    }
                else:
                    # Multi-replica analysis
                    replica_conv = self.convergence.analyze_replica_convergence(
                        replica_data
                    )

                    # Also analyze first replica individually
                    single_conv = self.convergence.analyze_timeseries_convergence(
                        replica_data[0]
                    )

                    convergence[lam_state] = {
                        'single_replica': single_conv,
                        'replica_convergence': replica_conv,
                        'n_replicas': len(replica_data),
                    }

        return convergence

    def _analyze_single_simulation_convergence(
        self, sim_runner: SimulationRunner
    ) -> Dict:
        """Analyze convergence for a single simulation (required by base class)."""
        if isinstance(sim_runner, Calculation):
            convergence_results = {}
            for leg in sim_runner.legs:
                lambda_data = self._collect_lambda_data(leg)
                leg_convergence = self._analyze_leg_convergence(lambda_data)
                convergence_results[leg.leg_type.name] = leg_convergence
            return convergence_results
        else:
            return {'error': 'Unsupported simulation runner type'}

    def _save_leg_results(self, results: Dict, output_dir: Path, leg_name: str):
        """
        Save comprehensive leg results.
        """
        # Save detailed results as JSON
        json_results = make_json_serializable(results)

        with open(output_dir / "detailed_results.json", 'w') as f:
            json.dump(json_results, f, indent=2)

        # Save summary text report
        self._write_leg_report(results, output_dir / "analysis_report.txt", leg_name)

        # Save convergence analysis separately
        if 'convergence_analysis' in results:
            conv_json = make_json_serializable(results['convergence_analysis'])
            with open(output_dir / "convergence_analysis.json", 'w') as f:
                json.dump(conv_json, f, indent=2)

        # Save MBAR overlap matrix if available
        if 'MBAR' in results and 'overlap_matrix' in results['MBAR']:
            overlap_matrix = results['MBAR']['overlap_matrix']
            if overlap_matrix is not None:
                np.savetxt(output_dir / "mbar_overlap_matrix.txt", overlap_matrix)
                try:
                    MultistateBAR.plot_overlap_matrix(
                        overlap_matrix, output_dir / "mbar_overlap_matrix.png"
                    )
                except Exception as e:
                    logger.warning(f"Failed to plot overlap matrix: {e}")

    def _write_leg_report(self, results: Dict, output_file: Path, leg_name: str):
        """
        Write a comprehensive text report for a leg.
        """
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"FREE ENERGY ANALYSIS REPORT - {leg_name} LEG\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Temperature: {self.temperature} K\n")
            f.write(f"Equilibration time: {self.equilibration_time} ns\n")
            f.write(f"Units: {self.units}\n\n")

            # Main results
            f.write("FREE ENERGY RESULTS:\n")
            f.write("-" * 40 + "\n")

            method_priority = ['MBAR', 'BAR', 'TI_CUBIC', 'TI']
            for method in method_priority:
                if method in results:
                    result = results[method]
                    if 'total_dg' in result:
                        energy_str = format_energy_with_error(
                            result['total_dg'], result['total_error'], self.units
                        )
                        f.write(f"{method:15s}: {energy_str}\n")

            f.write("\n")

            # Method summary table
            method_table = create_method_summary_table(results)
            f.write("METHOD COMPARISON:\n")
            f.write("-" * 40 + "\n")
            f.write(method_table)
            f.write("\n\n")

            # Convergence summary
            if 'convergence_analysis' in results:
                f.write("CONVERGENCE ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                conv = results['convergence_analysis']
                for state, state_conv in conv.items():
                    f.write(f"\nLambda state {state}:\n")
                    if 'replica_convergence' in state_conv:
                        replica = state_conv['replica_convergence']
                        f.write(
                            f"  Inter-replica relative std: {replica.get('relative_std_between', 0):6.3f}\n"
                        )
                        f.write(
                            f"  Number of replicas: {replica.get('n_replicas', 0)}\n"
                        )
                        f.write(f"  Converged: {replica.get('converged', False)}\n")
                    if 'single_replica' in state_conv:
                        single = state_conv['single_replica']
                        f.write(
                            f"  Single replica converged: {single.get('converged', False)}\n"
                        )
                        f.write(
                            f"  Relative fluctuation: {single.get('relative_fluctuation', 0):6.3f}\n"
                        )

    def _generate_analysis_output(self):
        """Generate comprehensive analysis output files."""
        # Save main results
        self.save_results("fep_analysis_results.pkl")

        # Generate summary report
        self.generate_summary_report("fep_analysis_summary.txt")

        # Save results DataFrame as CSV
        df = self.get_results_df()
        df.to_csv(self.output_dir / "fep_results_summary.csv", index=False)

        # Save analysis metadata
        metadata = self.get_metadata()
        metadata.update(
            {
                'legs_analyzed': list(self.results.leg_results.keys()),
                'methods_used': self.methods,
                'binding_affinity': self.results.binding_affinity,
                'binding_affinity_error': self.results.binding_affinity_error,
            }
        )

        with open(self.output_dir / "analysis_metadata.json", 'w') as f:
            json.dump(make_json_serializable(metadata), f, indent=2)

        logger.info(f"Analysis output saved to {self.output_dir}")

    def get_results_df(self) -> pd.DataFrame:
        """Return results as a pandas DataFrame."""
        return self.results.get_summary_df()

    def _write_specific_summary(self, file_handle):
        """Write FEP-specific summary content."""
        file_handle.write("FEP ANALYSIS RESULTS:\n")
        file_handle.write("-" * 40 + "\n")

        # Write leg results
        for leg_name, leg_data in self.results.leg_results.items():
            file_handle.write(f"\n{leg_name} Leg:\n")

            for method in ['MBAR', 'BAR', 'TI_CUBIC', 'TI']:
                if method in leg_data and 'total_dg' in leg_data[method]:
                    result = leg_data[method]
                    energy_str = format_energy_with_error(
                        result['total_dg'], result['total_error'], self.units
                    )
                    file_handle.write(f"  {method}: {energy_str}\n")

        # Write binding affinity
        if self.results.binding_affinity is not None:
            file_handle.write(f"\nBINDING AFFINITY:\n")
            file_handle.write("-" * 40 + "\n")
            binding_data = self.results.method_results.get('binding', {})
            method = binding_data.get('method', 'UNKNOWN')
            file_handle.write(f"Method: {method}\n")

            binding_str = format_energy_with_error(
                self.results.binding_affinity,
                self.results.binding_affinity_error,
                self.units,
            )
            file_handle.write(f"ΔG_binding = {binding_str}\n")

            bound_str = format_energy_with_error(
                binding_data.get('bound_dg', 0),
                binding_data.get('bound_error', 0),
                self.units,
            )
            free_str = format_energy_with_error(
                binding_data.get('free_dg', 0),
                binding_data.get('free_error', 0),
                self.units,
            )
            file_handle.write(f"ΔG_bound = {bound_str}\n")
            file_handle.write(f"ΔG_free = {free_str}\n")


# Convenience functions
def analyze_fep_calculation(
    calculation: Calculation,
    output_dir: str,
    temperature: float = 298.15,
    equilibration_time: float = 1.0,
    units: str = "kJ/mol",
    methods: Optional[List[str]] = None,
) -> FEPAnalysisResults:
    """
    Convenience function for comprehensive FEP analysis.

    Parameters:
    -----------
    calculation : Calculation
        GROMACS Calculation object containing FEP simulation data
    output_dir : str
        Directory for analysis output
    temperature : float
        Temperature in Kelvin
    equilibration_time : float
        Equilibration time to skip (ns)
    units : str
        Energy units for output
    methods : List[str], optional
        Analysis methods to use

    Returns:
    --------
    FEPAnalysisResults : Comprehensive analysis results
    """
    analyzer = GromacsFEPAnalyzer(
        sim_runners=[calculation],
        output_dir=output_dir,
        temperature=temperature,
        equilibration_time=equilibration_time,
        units=units,
        methods=methods,
    )

    return analyzer.analyze()


def compare_fep_methods(
    calculation: Calculation,
    output_dir: str,
    temperature: float = 298.15,
    equilibration_time: float = 1.0,
) -> Dict[str, Dict]:
    """
    Compare all available FEP methods for a calculation.

    Parameters:
    -----------
    calculation : Calculation
        GROMACS Calculation object
    output_dir : str
        Directory for analysis output
    temperature : float
        Temperature in Kelvin
    equilibration_time : float
        Equilibration time to skip (ns)

    Returns:
    --------
    Dict[str, Dict] : Comparison of all methods
    """
    # Run analysis with all methods
    analyzer = GromacsFEPAnalyzer(
        sim_runners=[calculation],
        output_dir=output_dir,
        temperature=temperature,
        equilibration_time=equilibration_time,
        methods=['ALL'],
    )

    results = analyzer.analyze()

    # Extract method comparison
    comparison = {}
    for leg_name, leg_results in results.leg_results.items():
        comparison[leg_name] = {}
        for method, method_results in leg_results.items():
            if isinstance(method_results, dict) and 'total_dg' in method_results:
                comparison[leg_name][method] = {
                    'free_energy': method_results['total_dg'],
                    'error': method_results['total_error'],
                    'method_description': method_results.get('method', method),
                }

    return comparison


def validate_fep_setup(calculation: Calculation) -> Dict[str, List[str]]:
    raise NotImplementedError("FEP setup validation is not implemented yet.")


if __name__ == "__main__":
    # Simple analysis
    results = analyze_fep_calculation(calculation, "/output/path")

    # Custom analysis with specific methods
    analyzer = ComprehensiveFEPAnalyzer(
        sim_runners=[calculation],
        output_dir="/output/path",
        methods=['TI', 'BAR', 'MBAR'],
        temperature=300.0,
        equilibration_time=2.0,
    )
    results = analyzer.analyze()

    # Access results
    binding_affinity = results.binding_affinity
    df = results.get_summary_df()
