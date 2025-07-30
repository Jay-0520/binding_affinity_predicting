import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from binding_affinity_predicting.components.analysis.free_energy_estimators import (
    FreeEnergyEstimator,
)
from binding_affinity_predicting.components.analysis.uncorrelate_subsampler import (
    perform_uncorrelating_subsampling,
)
from binding_affinity_predicting.components.analysis.xvg_data_loader import (
    GromacsXVGParser,
    load_alchemical_data,
)
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    Calculation,
    LambdaWindow,
    Leg,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FepSimulationAnalyzer:
    """
    Dedicated class for analyzing free energy calculations from GROMACS FEP simulations.

    This class is designed to work with the existing gromacs_orchestration.py classes
    (Calculation, Leg, LambdaWindow) without modifying them.
    """

    def __init__(
        self,
        calculation: Calculation,
        temperature: float = 298.15,
        units: str = "kcal",
        software: str = "Gromacs",
    ):
        """
        Initialize the free energy analyzer.

        Parameters
        ----------
        calculation : Calculation
            The GROMACS calculation object to analyze
        temperature : float, default 298.15
            Temperature in Kelvin
        units : str, default "kcal"
            Output units: 'kJ', 'kcal', or 'kBT'
        software : str, default "Gromacs"
            Software package name
        """
        self.calculation = calculation
        self.temperature = temperature
        self.units = units
        self.software = software

        # Initialize the free energy estimator
        self.estimator = FreeEnergyEstimator(
            temperature=temperature, units=units, software=software
        )
        # Storage for analysis results
        self._analysis_results: dict[str, Any] = {}

    def collect_xvg_files_from_window(
        self, window: LambdaWindow, run_nos: Optional[list[int]] = None
    ) -> list[Path]:
        """
        Collect XVG files from a single lambda window.

        Parameters
        ----------
        window : LambdaWindow
            Lambda window to collect files from
        run_nos : list[int], optional
            List of run numbers to include. If None, includes all runs.

        Returns
        -------
        list[Path]
            List of paths to XVG files
        """
        if run_nos is None:
            run_nos = list(range(1, window.ensemble_size + 1))

        xvg_files = []

        for run_no in run_nos:
            run_dir = Path(window.output_dir) / f"run_{run_no}"
            # Look for the standard GROMACS FEP output file
            xvg_file = run_dir / f"lambda_{window.lam_state}_run_{run_no}.xvg"

            if xvg_file.exists():
                xvg_files.append(xvg_file)
            else:
                logger.warning(f"XVG file not found: {xvg_file}")

        logger.debug(
            f"Collected {len(xvg_files)} XVG files from lambda {window.lam_state}"
        )
        return xvg_files

    def collect_xvg_files_from_leg(
        self, leg: Leg, run_nos: Optional[list[int]] = None
    ) -> list[Path]:
        """
        Collect all XVG files from a leg (all lambda windows).

        Parameters
        ----------
        leg : Leg
            Leg to collect files from
        run_nos : list[int], optional
            List of run numbers to include. If None, includes all runs.

        Returns
        -------
        list[Path]
            List of paths to all XVG files in the leg
        """
        all_xvg_files = []

        for window in leg.lambda_windows:
            window_files = self.collect_xvg_files_from_window(window, run_nos)
            all_xvg_files.extend(window_files)

        logger.info(
            f"Collected {len(all_xvg_files)} XVG files from {leg.leg_type.name} leg"
        )
        return all_xvg_files

    def analyze_leg(
        self,
        leg: Leg,
        run_nos: Optional[list[int]] = None,
        skip_time: float = 0.0,
        methods: Optional[list[str]] = None,
        observable: str = "dhdl",
        min_uncorr_samples: int = 50,
        save_results: bool = True,
    ) -> dict:
        """
        Perform complete free energy analysis for a single leg.

        Parameters
        ----------
        leg : Leg
            The leg to analyze
        run_nos : list[int], optional
            List of run numbers to analyze. If None, analyzes all runs.
        skip_time : float, default 0.0
            Time to skip for equilibration (in ps)
        methods : list[str], optional
            List of free energy methods to use. If None, uses all available.
        observable : str, default 'dhdl'
            Observable for correlation analysis ('dhdl', 'dhdl_all', or 'de')
        min_uncorr_samples : int, default 50
            Minimum number of uncorrelated samples required
        save_results : bool, default True
            Whether to save analysis results to file

        Returns
        -------
        dict
            Dictionary containing analysis results for all methods
        """
        logger.info(f"Starting free energy analysis for {leg.leg_type.name} leg...")

        try:
            # Step 1: Collect XVG files
            xvg_files = self.collect_xvg_files_from_leg(leg, run_nos)
            if not xvg_files:
                raise ValueError(f"No XVG files found for {leg.leg_type.name} leg")

            # Step 2: Load alchemical data from XVG files
            logger.info("Loading alchemical data from XVG files...")
            alchemical_data = load_alchemical_data(
                xvg_files=xvg_files,
                skip_time=skip_time,
                temperature=self.temperature,
                reduce_to_dimensionless=True,
            )

            # Step 3: Perform uncorrelated subsampling
            logger.info("Performing autocorrelation analysis and subsampling...")
            dhdl_uncorr, potential_uncorr, num_uncorr_samples_per_state = (
                perform_uncorrelating_subsampling(
                    dhdl_timeseries=alchemical_data['dhdl_timeseries'],
                    lambda_vectors=alchemical_data['lambda_vectors'],
                    start_indices=np.zeros(
                        len(alchemical_data['lambda_vectors']), dtype=int
                    ),
                    end_indices=alchemical_data['nsnapshots'],
                    potential_energies=alchemical_data['potential_energies'],
                    observable=observable,
                    min_uncorr_samples=min_uncorr_samples,
                )
            )

            # Step 4:   Calculate ave_dhdl and std_dhdl from uncorrelated samples
            # already beta-scaled from load_alchemical_data, so they are dimensionless
            logger.info("Calculating dH/dλ statistics for TI methods...")
            ave_dhdl = None
            std_dhdl = None
            if dhdl_uncorr is not None:
                ave_dhdl = np.mean(dhdl_uncorr, axis=2)
                # std_dhdl: standard error of the mean
                # Using ddof=1 for unbiased standard deviation
                std_dhdl = np.std(dhdl_uncorr, axis=2, ddof=1) / np.sqrt(
                    dhdl_uncorr.shape[2]
                )

                logger.info(
                    f"Calculated dH/dλ statistics from uncorrelated samples: "
                    f"ave_dhdl shape={ave_dhdl.shape}, std_dhdl shape={std_dhdl.shape}"
                )

            # Step 5: Estimate free energies using all methods
            logger.info("Estimating free energies using multiple methods...")
            results = self.estimator.estimate_all_methods(
                potential_energies=potential_uncorr,
                sample_counts=num_uncorr_samples_per_state,
                lambda_vectors=alchemical_data['lambda_vectors'],
                ave_dhdl=ave_dhdl,
                std_dhdl=std_dhdl,
                methods=methods,
            )

            # Step 5: Compile analysis results
            analysis_results = {
                'leg_type': leg.leg_type.name,
                'temperature': self.temperature,
                'units': self.units,
                'n_lambda_states': len(alchemical_data['lambda_vectors']),
                'n_xvg_files': len(xvg_files),
                'total_uncorr_samples': int(num_uncorr_samples_per_state.sum()),
                'lambda_vectors': alchemical_data['lambda_vectors'].tolist(),
                'uncorr_samples_per_state': num_uncorr_samples_per_state.tolist(),
                'methods': results,
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_parameters': {
                    'skip_time': skip_time,
                    'observable': observable,
                    'min_uncorr_samples': min_uncorr_samples,
                    'run_nos': run_nos,
                },
            }

            # Step 6: Save results if requested
            if save_results:
                self._save_leg_analysis_results(leg, analysis_results)

            return analysis_results

        except Exception as e:
            logger.error(
                f"Free energy analysis failed for {leg.leg_type.name} leg: {e}"
            )
            raise

    def analyze_calculation(
        self,
        run_nos: Optional[list[int]] = None,
        skip_time: float = 0.0,
        methods: Optional[list[str]] = None,
        observable: str = "dhdl",
        min_uncorr_samples: int = 50,
        save_results: bool = True,
    ) -> dict:
        """
        Perform free energy analysis for the entire calculation (all legs).

        Parameters
        ----------
        run_nos : list[int], optional
            List of run numbers to analyze. If None, analyzes all runs.
        skip_time : float, default 0.0
            Time to skip for equilibration (in ps)
        methods : list[str], optional
            List of free energy methods to use. If None, uses all available.
        observable : str, default 'dhdl'
            Observable for correlation analysis ('dhdl', 'dhdl_all', or 'de')
        min_uncorr_samples : int, default 50
            Minimum number of uncorrelated samples required
        save_results : bool, default True
            Whether to save analysis results to file

        Returns
        -------
        dict
            Dictionary containing analysis results for all legs and binding free energy
        """
        logger.info("Starting comprehensive free energy analysis...")

        all_results: dict[str, Any] = {
            'calculation_info': {
                'input_dir': self.calculation.input_dir,
                'output_dir': self.calculation.output_dir,
                'temperature': self.temperature,
                'units': self.units,
                'ensemble_size': self.calculation.ensemble_size,
                'n_legs': len(self.calculation.legs),
                'analysis_timestamp': datetime.now().isoformat(),
            },
            'leg_results': {},
            'binding_free_energy': None,
        }

        # Analyze each leg
        for leg in self.calculation.legs:
            logger.info(f"Analyzing {leg.leg_type.name} leg...")
            try:
                leg_results = self.analyze_leg(
                    leg=leg,
                    run_nos=run_nos,
                    skip_time=skip_time,
                    methods=methods,
                    observable=observable,
                    min_uncorr_samples=min_uncorr_samples,
                    save_results=save_results,
                )
                all_results['leg_results'][leg.leg_type.name] = leg_results

            except Exception as e:
                logger.error(f"Failed to analyze {leg.leg_type.name} leg: {e}")
                all_results['leg_results'][leg.leg_type.name] = {
                    'error': str(e),
                    'success': False,
                }

        # Calculate binding free energy if we have both bound and free legs
        if (
            'BOUND' in all_results['leg_results']
            and 'FREE' in all_results['leg_results']
        ):
            all_results['binding_free_energy'] = self._calculate_binding_free_energy(
                all_results['leg_results']
            )
        elif 'BOUND' in all_results['leg_results']:
            # For single leg calculations, report the bound leg results
            bound_results = all_results['leg_results']['BOUND']
            if bound_results.get('methods'):
                all_results['binding_free_energy'] = {
                    'note': 'Single leg (BOUND) calculation - not absolute binding free energy',
                    'methods': {
                        method: {
                            'free_energy': result.get('free_energy', 0.0),
                            'error': result.get('error', 0.0),
                            'units': self.units,
                            'success': result.get('success', False),
                        }
                        for method, result in bound_results.get('methods', {}).items()
                        if result.get('success', False)
                    },
                }

        # Save comprehensive results
        if save_results:
            self._save_calculation_analysis(all_results)

        self._analysis_results = all_results
        return all_results

    def _calculate_binding_free_energy(self, leg_results: dict) -> dict:
        """
        Calculate binding free energy from bound and free leg results.

        ΔG_binding = ΔG_bound - ΔG_free
        """
        binding_results: dict[str, Any] = {
            'methods': {},
            'note': 'Binding free energy = ΔG_bound - ΔG_free',
            'units': self.units,
        }

        bound_methods = leg_results.get('BOUND', {}).get('methods', {})
        free_methods = leg_results.get('FREE', {}).get('methods', {})

        # Calculate binding free energy for each method that succeeded in both legs
        for method in bound_methods:
            if (
                method in free_methods
                and bound_methods[method].get('success', False)
                and free_methods[method].get('success', False)
            ):

                dg_bound = bound_methods[method].get('free_energy', 0.0)
                dg_free = free_methods[method].get('free_energy', 0.0)

                error_bound = bound_methods[method].get('error', 0.0)
                error_free = free_methods[method].get('error', 0.0)

                # ΔG_binding = ΔG_bound - ΔG_free
                dg_binding = dg_bound - dg_free
                # Error propagation assuming independent errors
                error_binding = np.sqrt(error_bound**2 + error_free**2)

                binding_results['methods'][method] = {
                    'binding_free_energy': dg_binding,
                    'error': error_binding,
                    'dg_bound': dg_bound,
                    'dg_free': dg_free,
                    'error_bound': error_bound,
                    'error_free': error_free,
                    'units': self.units,
                    'success': True,
                }

        return binding_results

    def _save_leg_analysis_results(self, leg: Leg, results: dict) -> None:
        """Save analysis results for a single leg."""
        import json

        output_file = (
            Path(leg.output_dir) / f"{leg.leg_type.name.lower()}_analysis_results.json"
        )

        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_numpy_for_json(results)

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Leg analysis results saved to {output_file}")

    def _save_calculation_analysis(self, results: dict) -> None:
        """Save comprehensive calculation analysis results."""
        import json

        output_file = (
            Path(self.calculation.output_dir) / "calculation_analysis_results.json"
        )

        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_numpy_for_json(results)

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Comprehensive analysis results saved to {output_file}")

    def _convert_numpy_for_json(self, obj):
        """Recursively convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {
                key: self._convert_numpy_for_json(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_numpy_for_json(item) for item in obj]
        else:
            return obj

    def get_analysis_results(self) -> dict[str, Any]:
        """Get the most recent analysis results."""
        return self._analysis_results

    def export_analysis_summary(
        self, output_file: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Export a summary of analysis results to a file.

        Parameters
        ----------
        output_file : str or Path, optional
            Output file path. If None, uses default naming.
        format : str, default "csv"
            Output format: "csv" or "xlsx"
        """
        if self._analysis_results is None:
            logger.warning("No analysis results available. Run analysis first.")
            return

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "csv" if format == "csv" else "xlsx"
            output_file = (
                Path(self.calculation.output_dir)
                / f"fe_analysis_summary_{timestamp}.{ext}"
            )

        self._export_csv_summary(output_file)

    def _export_csv_summary(self, output_file: Union[str, Path]) -> None:
        """Export results summary to CSV format."""
        import csv

        if isinstance(output_file, str):
            output_file = Path(output_file)

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                ['Leg', 'Method', 'Free_Energy', 'Error', 'Units', 'Success']
            )

            # Leg results
            for leg_name, leg_result in self._analysis_results['leg_results'].items():
                if leg_result.get('methods'):
                    for method, result in leg_result['methods'].items():
                        writer.writerow(
                            [
                                leg_name,
                                method,
                                result.get('free_energy', 0.0),
                                result.get('error', 0.0),
                                leg_result.get('units', self.units),
                                result.get('success', False),
                            ]
                        )

            # Binding free energy results
            binding_results = self._analysis_results.get('binding_free_energy')
            if binding_results and 'methods' in binding_results:
                for method, result in binding_results['methods'].items():
                    if 'binding_free_energy' in result:
                        writer.writerow(
                            [
                                'BINDING',
                                method,
                                result['binding_free_energy'],
                                result['error'],
                                result['units'],
                                result['success'],
                            ]
                        )

        logger.info(f"CSV summary exported to {output_file}")


# Convenience functions for easy usage
def analyze_gromacs_calculation(
    calculation: Calculation,
    temperature: float = 298.15,
    units: str = "kcal",
    **analysis_kwargs,
) -> dict:
    """
    Convenience function to analyze a GROMACS calculation.

    Parameters
    ----------
    calculation : Calculation
        The GROMACS calculation object to analyze
    temperature : float, default 298.15
        Temperature in Kelvin
    units : str, default "kcal"
        Output units
    **analysis_kwargs
        Additional arguments passed to analyze_calculation()

    Returns
    -------
    dict
        Analysis results
    """
    analyzer = FepSimulationAnalyzer(calculation, temperature=temperature, units=units)
    return analyzer.analyze_calculation(**analysis_kwargs)


def analyze_gromacs_leg(
    leg: Leg, temperature: float = 298.15, units: str = "kcal", **analysis_kwargs
) -> dict:
    """
    Convenience function to analyze a single GROMACS leg.

    Parameters
    ----------
    leg : Leg
        The GROMACS leg object to analyze
    temperature : float, default 298.15
        Temperature in Kelvin
    units : str, default "kcal"
        Output units
    **analysis_kwargs
        Additional arguments passed to analyze_leg()

    Returns
    -------
    dict
        Analysis results
    """
    # Create a dummy calculation object for the analyzer
    # need this to make mypy happy
    if not hasattr(leg, 'calculation') or not isinstance(leg.calculation, Calculation):
        raise ValueError("Leg must have a valid .calculation attribute")

    # TODO: need to fix this by having leg.calculation
    analyzer = FepSimulationAnalyzer(
        calculation=leg.calculation,
        temperature=temperature,
        units=units,
    )
    return analyzer.analyze_leg(leg, **analysis_kwargs)
