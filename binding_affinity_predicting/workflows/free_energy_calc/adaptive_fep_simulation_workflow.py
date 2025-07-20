"""
Adaptive FEP workflows integration module.

This module provides high-level workflows that integrate adaptive efficiency,
adaptive equilibration, and lambda optimization for GROMACS FEP calculations.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from binding_affinity_predicting.components.data.schemas import (
    GromacsFepSimulationConfig,
)
from binding_affinity_predicting.components.simulation_fep.adaptive_simulation_runner import (
    AdaptiveSimulationRunner,
)
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    Calculation,
)
from binding_affinity_predicting.components.simulation_fep.lambda_optimizer import (
    LambdaOptimizationManager,
    OptimizationConfig,
)
from binding_affinity_predicting.components.simulation_fep.status_monitor import (
    StatusMonitor,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AdaptiveFepSimulationWorkflow:
    """
    High-level manager for adaptive FEP workflows.

    This class provides a unified interface for running various adaptive
    optimization strategies for GROMACS FEP calculations.
    """

    def __init__(
        self,
        calculation: Calculation,
        enable_monitoring: bool = True,
        use_hpc: bool = True,
        run_sync: bool = True,
        skip_completed_phases: bool = True,
    ) -> None:
        """
        Initialize the adaptive FEP workflow manager.

        Parameters
        ----------
        calculation : Calculation
            GROMACS calculation object to manage
        """
        self.calculation = calculation
        self.workflow_results: dict[str, Any] = {}
        self._last_workflow_results: dict[str, Any] = {}
        self.enable_monitoring = enable_monitoring
        self.use_hpc = use_hpc
        self.run_sync = run_sync
        self.skip_completed_phases = skip_completed_phases

        self.status_monitor: Optional[StatusMonitor]
        # Initialize status monitor
        if self.enable_monitoring:
            self.status_monitor = StatusMonitor(self.calculation)
            logger.info("Status monitoring enabled")
        else:
            self.status_monitor = None
            logger.info("Status monitoring disabled")

        # Resume functionality attributes
        self.workflow_state_file = (
            Path(self.calculation.output_dir) / "workflow_completed_phases.txt"
        )
        self.completed_phases = self._load_completed_phases()

    def _load_completed_phases(self) -> set[str]:
        """
        Load completed phases from previous workflow runs.

        Returns
        -------
        set[str]
            Set of phase names that have been completed
        """
        if not self.skip_completed_phases:
            return set()

        if self.workflow_state_file.exists():
            try:
                with open(self.workflow_state_file, 'r') as f:
                    phases = {line.strip() for line in f.readlines() if line.strip()}
                logger.info(f"Loaded completed phases: {phases}")
                return phases
            except Exception as e:
                logger.warning(f"Could not load workflow state: {e}")
                return set()
        else:
            logger.info("No previous workflow state found")
            return set()

    def _save_completed_phase(self, phase_name: str) -> None:
        """
        Save a completed phase to the workflow state file.

        Parameters
        ----------
        phase_name : str
            Name of the completed phase
        """
        if not self.skip_completed_phases:
            return

        self.completed_phases.add(phase_name)

        try:
            with open(self.workflow_state_file, 'w') as f:
                for phase in sorted(self.completed_phases):
                    f.write(f"{phase}\n")
            logger.info(f"Saved completed phase: {phase_name}")
        except Exception as e:
            logger.warning(f"Could not save workflow state: {e}")

    def _is_phase_completed(self, phase_name: str) -> bool:
        """
        Check if a phase has already been completed.

        Parameters
        ----------
        phase_name : str
            Name of the phase to check

        Returns
        -------
        bool
            True if phase has been completed
        """
        return phase_name in self.completed_phases

    def _print_simple_status(self) -> None:
        """Print a simple status summary using the StatusMonitor."""
        if self.status_monitor:
            try:
                summary = self.status_monitor.get_summary()
                logger.info(f"Status: {summary}")
            except Exception as e:
                logger.warning(f"Could not get status summary: {e}")
        else:
            logger.info("Status monitoring is disabled")

    def _check_simulation_success(self) -> bool:
        """Check if simulations completed successfully using StatusMonitor."""
        if not self.status_monitor:
            # Fallback to calculation.failed if monitoring is disabled
            return not self.calculation.failed

        try:
            status = self.status_monitor.get_status()
            job_counts = status.get("job_counts", {})
            failed_count = job_counts.get("FAILED", 0)
            return failed_count == 0
        except Exception as e:
            logger.warning(f"Could not check simulation success: {e}")
            return not self.calculation.failed

    # TODO: we could implement a adpative feature here
    def _lambda_optimizing(
        self,
        run_nos: Optional[list[int]] = None,
        equilibrated: bool = False,
        apply_optimization: bool = True,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> dict[str, Any]:
        """
        Run lambda spacing optimization workflow.

        Parameters
        ----------
        run_nos : list[int], optional
            Run numbers to analyze
        equilibrated : bool, default False
            Whether simulations are equilibrated
        apply_optimization : bool, default True
            Whether to apply optimized lambda spacing
        optimization_config : OptimizationConfig, optional
            Configuration for optimization

        Returns
        -------
        dict
            Results from lambda optimization
        """
        logger.info("Running Lambda Spacing Optimization...")

        if optimization_config is None:
            optimization_config = OptimizationConfig()

        manager = LambdaOptimizationManager(config=optimization_config)

        optimization_results = manager.optimize_calculation(
            calculation=self.calculation,
            run_nos=run_nos,
            equilibrated=equilibrated,
            apply_results=apply_optimization,
        )

        # Summarize results
        successful_legs = sum(
            1 for result in optimization_results.values() if result.success
        )
        total_legs = len(optimization_results)

        results = {
            'type': 'lambda_optimizing',
            'successful': successful_legs == total_legs,
            'successful_legs': successful_legs,
            'total_legs': total_legs,
            'applied': apply_optimization,
            'optimization_results': optimization_results,
        }

        self.workflow_results['lambda_optimizing'] = results
        return results

    def _run_adaptively(
        self,
        initial_runtime_constant: float = 0.001,
        equilibration_method: str = "multiwindow",
        max_runtime_per_window: float = 30.0,
        run_nos: Optional[list[int]] = None,
        use_hpc: bool = True,
    ) -> dict[str, Any]:
        """
        Run adaptive simulation workflow (equilibration + efficiency optimization).

        Parameters
        ----------
        initial_runtime_constant : float, default 0.001
            Initial runtime constant
        equilibration_method : str, default "multiwindow"
            Equilibration detection method
        max_runtime_per_window : float, default 30.0
            Maximum runtime per window (ns)
        run_nos : list[int], optional
            Run numbers to include

        Returns
        -------
        dict
            Results from simulation workflow
        """
        logger.info("Running Adaptive Simulation Workflow...")

        manager = AdaptiveSimulationRunner(
            calculation=self.calculation,
            initial_runtime_constant=initial_runtime_constant,
            equilibration_method=equilibration_method,
            max_runtime_per_window=max_runtime_per_window,
            use_hpc=use_hpc,
        )
        _ = manager.run_simulation(run_nos=run_nos)

        results = {
            'type': 'adaptive_simulation_workflow',
            'successful': manager.is_equilibrated,
            'iterations': manager.current_iteration,
            'final_runtime_constant': manager.current_runtime_constant,
            'status': manager.get_simulation_status(),
        }

        self.workflow_results['adaptive_simulation'] = results
        return results

    def run_complete_adaptive_workflow(
        self,
        short_run_runtime: float = 2.0,  # 2ns; TODO: could be adaptive as well.
        initial_runtime_constant: float = 0.001,
        equilibration_method: str = "multiwindow",
        max_runtime_per_window: float = 30.0,
        optimize_lambda_spacing: bool = True,
        run_nos: Optional[list[int]] = None,
        monitor_interval: int = 60,  # seconds between status updates
        force_rerun_phases: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Run the complete adaptive FEP workflow with simple monitoring.

        This workflow implements the following sequence:
        1. Short simulations for gradient collection
        2. Lambda spacing optimization based on gradients
        3. Production simulations with optimized lambda values using adaptive equilibration
           and efficiency

        Parameters
        ----------
        short_run_runtime : float, default 2.0
            Runtime for short simulations to collect gradients (ns)
        initial_runtime_constant : float, default 0.001
            Initial runtime constant for production simulations
        equilibration_method : str, default "multiwindow"
            Equilibration detection method for production
        max_runtime_per_window : float, default 30.0
            Maximum runtime per window (ns) for production
        optimize_lambda_spacing : bool, default True
            Whether to optimize lambda spacing after short runs
        run_nos : list[int], optional
            Run numbers to include
        force_rerun_phases : list[str], optional
            list of phase names to force rerun even if completed.
            Valid phases: 'short_simulations', 'lambda_optimizating', 'production_adaptive'

        Returns
        -------
        dict
            Comprehensive results from all workflow phases
        """
        logger.info("🚀 Running Complete Adaptive FEP Workflow (BAB Protocol)...")
        logger.info("=" * 70)

        # Handle force rerun
        if force_rerun_phases:
            for phase in force_rerun_phases:
                if phase in self.completed_phases:
                    logger.info(f"Forcing rerun of phase: {phase}")
                    self.completed_phases.discard(phase)

        # Print resume information
        if self.skip_completed_phases and self.completed_phases:
            logger.info(
                f"Resume mode enabled. Previously completed phases: {self.completed_phases}"
            )
        elif self.skip_completed_phases:
            logger.info("Resume mode enabled. No previously completed phases found.")
        else:
            logger.info("Resume mode disabled. All phases will be executed.")

        workflow_results: dict[str, Any] = {
            'workflow_type': 'complete_adaptive_fep_workflow_BAB',
            'phases_completed': [],
            'phases_skipped': [],
            'overall_successful': False,
            'phase_results': {},
            'resume_enabled': self.skip_completed_phases,
        }

        try:
            # ==========================================================
            # Phase 1: Short simulations for gradient collection
            # ==========================================================
            phase1_name = 'short_simulations'

            if self._is_phase_completed(phase1_name):
                logger.info("PHASE 1: SHORT SIMULATIONS (SKIPPED - ALREADY COMPLETED)")
                logger.info("=" * 50)
                logger.info("Using existing short simulation results")

                short_sim_results = {
                    'type': 'short_simulations',
                    'runtime': short_run_runtime,
                    'successful': True,
                    'skipped': True,
                    'reason': 'Previously completed',
                }

                workflow_results['phases_skipped'].append(phase1_name)
                workflow_results['phase_results'][phase1_name] = short_sim_results

            else:
                logger.info("PHASE 1: SHORT SIMULATIONS FOR GRADIENT COLLECTION")
                logger.info("=" * 50)
                logger.info(
                    f"Running {short_run_runtime} ns simulations for lambda optimization..."
                )

                # Run short simulations
                self.calculation.run(
                    runtime=short_run_runtime,
                    use_hpc=self.use_hpc,
                    run_sync=self.run_sync,
                )

                # Show initial status after starting simulations
                if self.enable_monitoring:
                    logger.info("Phase 1 started, checking initial status...")
                    self._print_simple_status()

                # Wait for completion
                logger.info("Waiting for short simulations to complete...")
                self._wait_for_calculation(check_interval=monitor_interval)

                short_sim_results = {
                    'type': 'short_simulations',
                    'runtime': short_run_runtime,
                    'successful': self._check_simulation_success(),
                    'skipped': False,
                }

                if not short_sim_results['successful']:
                    logger.error("Short simulations failed. Stopping workflow.")
                    if self.enable_monitoring:
                        self._print_simple_status()  # Show final status
                    workflow_results['phase_results'][phase1_name] = short_sim_results
                    return workflow_results

                logger.info(
                    f"Short simulations completed successfully ({short_run_runtime} ns)"
                )

                workflow_results['phases_completed'].append(phase1_name)
                workflow_results['phase_results'][phase1_name] = short_sim_results

                # Mark phase as completed
                self._save_completed_phase(phase1_name)

            # ==========================================================
            # Phase 2: Lambda spacing optimization
            # ==========================================================
            phase2_name = 'lambda_optimizating'

            if optimize_lambda_spacing:
                if self._is_phase_completed(phase2_name):
                    logger.info(
                        "PHASE 2: LAMBDA OPTIMIZATION (SKIPPED - ALREADY COMPLETED)"
                    )
                    logger.info("=" * 50)

                    optimization_results = {
                        'type': 'lambda_optimizing',
                        'successful': True,
                        'skipped': True,
                        'reason': 'Previously completed',
                    }

                    workflow_results['phases_skipped'].append(phase2_name)
                    workflow_results['phase_results'][
                        phase2_name
                    ] = optimization_results
                else:
                    logger.info("PHASE 2: LAMBDA SPACING OPTIMIZATION")
                    logger.info("=" * 50)

                    optimization_results = self._lambda_optimizing(
                        run_nos=run_nos,
                        equilibrated=False,  # use short unequilibrated data
                        apply_optimization=True,  # apply the result directly in prod
                    )

                    workflow_results['phase_results'][
                        phase2_name
                    ] = optimization_results

                    if optimization_results['successful']:
                        logger.info(
                            "Lambda spacing optimization completed successfully"
                        )
                        logger.info(
                            "Calculation has been updated with optimized lambda values"
                        )
                        workflow_results['phases_completed'].append(phase2_name)
                        self._save_completed_phase(phase2_name)
                    else:
                        logger.warning(
                            "Lambda optimization had issues, continuing with original lambda values..."  # noqa: E501
                        )
            else:
                logger.info(
                    "SKIPPING PHASE 2 LAMBDA OPTIMIZATION (optimize_lambda_spacing=False)"
                )

            # ===================================================================================
            # Phase 3: Production simulations with adaptive equilibration and efficiency/runtime
            # ===================================================================================
            phase3_name = 'production_adaptive'

            if self._is_phase_completed(phase3_name):
                logger.info(
                    "PHASE 3: PRODUCTION ADAPTIVE (SKIPPED - ALREADY COMPLETED)"
                )
                logger.info("=" * 50)

                production_results = {
                    'type': 'adaptive_simulation_workflow',
                    'successful': True,
                    'skipped': True,
                    'reason': 'Previously completed',
                }

                workflow_results['phases_skipped'].append(phase3_name)
                workflow_results['phase_results'][phase3_name] = production_results
            else:
                logger.info(
                    "PHASE 3: PRODUCTION WITH ADAPTIVE EQUILIBRATION & EFFICIENCY"
                )
                logger.info("=" * 50)

                # Use optimized lambda values (if optimization was successful) for production runs
                production_results = self._run_adaptively(
                    initial_runtime_constant=initial_runtime_constant,
                    equilibration_method=equilibration_method,
                    max_runtime_per_window=max_runtime_per_window,
                    run_nos=run_nos,
                    use_hpc=self.use_hpc,
                )

                workflow_results['phase_results'][phase3_name] = production_results

                if production_results['successful']:
                    logger.info("Production adaptive simulation completed successfully")
                    workflow_results['phases_completed'].append(phase3_name)
                    self._save_completed_phase(phase3_name)
                else:
                    logger.warning("Production adaptive simulation had issues")

            # ===================================================================================
            # Tracking overall success and summary statistics
            # ===================================================================================
            # Overall success assessment
            required_phases = ['short_simulations', 'production_adaptive']

            workflow_results['overall_successful'] = all(
                phase
                in (
                    workflow_results['phases_completed']
                    + workflow_results['phases_skipped']
                )
                and workflow_results['phase_results'][phase].get('successful', False)
                for phase in required_phases
            )

            # Summary statistics
            total_runtime = 0
            if 'short_simulations' in workflow_results['phase_results']:
                short_results = workflow_results['phase_results']['short_simulations']
                if not short_results.get('skipped', False):
                    total_runtime += short_results.get('runtime', 0)

            if 'production_adaptive' in workflow_results['phase_results']:
                production_status = workflow_results['phase_results'][
                    'production_adaptive'
                ].get('status', {})
                if 'optimalruntime_status' in production_status:
                    runtime_status = production_status['optimalruntime_status']
                    if runtime_status and 'total_simulation_time' in runtime_status:
                        total_runtime += runtime_status['total_simulation_time']

            workflow_results['summary'] = {
                'total_runtime': total_runtime,
                'lambda_optimization_applied': (
                    optimize_lambda_spacing
                    and workflow_results['phase_results']
                    .get('lambda_optimization', {})
                    .get('applied', False)
                ),
                'final_runtime_constant': (
                    workflow_results['phase_results']
                    .get('production_adaptive', {})
                    .get('final_runtime_constant', initial_runtime_constant)
                ),
                'equilibration_iterations': (
                    workflow_results['phase_results']
                    .get('production_adaptive', {})
                    .get('iterations', 0)
                ),
                'phases_skipped': workflow_results['phases_skipped'],
            }

        except Exception as e:
            logger.error(f"Workflow failed with exception: {e}")
            workflow_results['error'] = str(e)

        # Final summary
        logger.info("\n" + "=" * 70)
        if workflow_results['overall_successful']:
            logger.info("🎉 COMPLETE ADAPTIVE WORKFLOW (A3FE) FINISHED SUCCESSFULLY")

        else:
            logger.warning("⚠️ WORKFLOW COMPLETED WITH SOME ISSUES")

        logger.info(
            f"Phases completed: {', '.join(workflow_results['phases_completed'])}"
        )
        if workflow_results['phases_skipped']:
            logger.info(
                f"Phases skipped: {', '.join(workflow_results['phases_skipped'])}"
            )
        logger.info("=" * 70)

        # Show final status
        if self.enable_monitoring:
            self._print_simple_status()

        return workflow_results

    def _wait_for_calculation(self, check_interval: int = 60) -> None:
        """Wait for calculation to complete with status monitoring."""
        if self.calculation.virtual_queue:
            logger.info(
                "calc virtual_queue is not None; waiting for simulations to complete..."
            )

            while self.calculation.running:
                # Print status during waiting if monitoring is enabled
                if self.enable_monitoring:
                    self._print_simple_status()

                time.sleep(check_interval)
                self.calculation.virtual_queue.update()

        else:
            # For local runs, just use a simple wait
            logger.info("Running locally - waiting briefly for completion...")
            while True:
                if self.enable_monitoring:
                    self._print_simple_status()
                if not self.calculation.running:
                    logger.info("Local simulations completed")
                    break
                time.sleep(check_interval)

        if self._check_simulation_success():
            logger.info("✅ All simulations completed successfully")
        else:
            logger.warning("⚠️ Some simulations failed")

    def save_workflow_report(self, output_dir: Optional[str] = None) -> None:
        """
        Save a comprehensive workflow report using the returned workflow results.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save report
        """
        if output_dir is None:
            output_dir = self.calculation.output_dir

        output_path = Path(output_dir) / "adaptive_fep_workflow_report.txt"

        # Get the latest workflow results from the last run
        if (
            not hasattr(self, '_last_workflow_results')
            or not self._last_workflow_results
        ):
            logger.warning("No workflow results available for report generation")
            return

        results = self._last_workflow_results

        with open(output_path, 'w') as f:
            f.write("ADAPTIVE FEP WORKFLOW REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Write workflow metadata
            f.write(f"Workflow Type: {results.get('workflow_type', 'Unknown')}\n")
            f.write(f"Resume Enabled: {results.get('resume_enabled', False)}\n")
            f.write(f"Overall Successful: {results.get('overall_successful', False)}\n")
            f.write(
                f"Phases Completed: {', '.join(results.get('phases_completed', []))}\n"
            )
            f.write(
                f"Phases Skipped: {', '.join(results.get('phases_skipped', []))}\n\n"
            )

            # Write individual phase results
            phase_results = results.get('phase_results', {})
            for phase_name, phase_result in phase_results.items():
                f.write(f"{phase_name.upper().replace('_', ' ')} RESULTS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Type: {phase_result.get('type', 'Unknown')}\n")
                f.write(f"Successful: {phase_result.get('successful', False)}\n")
                f.write(f"Skipped: {phase_result.get('skipped', False)}\n")

                if 'reason' in phase_result:
                    f.write(f"Reason: {phase_result['reason']}\n")
                if 'runtime' in phase_result:
                    f.write(f"Runtime: {phase_result['runtime']} ns\n")
                if 'iterations' in phase_result:
                    f.write(f"Iterations: {phase_result['iterations']}\n")
                if 'final_runtime_constant' in phase_result:
                    f.write(
                        f"Final runtime constant: {phase_result['final_runtime_constant']:.6f}\n"
                    )
                if 'successful_legs' in phase_result and 'total_legs' in phase_result:
                    f.write(
                        f"Successful legs: {phase_result['successful_legs']}/{phase_result['total_legs']}\n"  # noqa: E501
                    )

                f.write("\n")

            # Write summary
            if 'summary' in results:
                summary = results['summary']
                f.write("WORKFLOW SUMMARY:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Runtime: {summary.get('total_runtime', 0)} ns\n")
                f.write(
                    f"Lambda Optimization Applied: {summary.get('lambda_optimization_applied', False)}\n"  # noqa: E501
                )
                f.write(
                    f"Final Runtime Constant: {summary.get('final_runtime_constant', 0):.6f}\n"
                )
                f.write(
                    f"Equilibration Iterations: {summary.get('equilibration_iterations', 0)}\n"
                )
                f.write("\n")

            # Add status summary at the end of the report
            if self.status_monitor:
                f.write("FINAL STATUS SUMMARY:\n")
                f.write("-" * 40 + "\n")
                try:
                    summary = self.status_monitor.get_summary()
                    f.write(f"{summary}\n")
                except Exception as e:
                    f.write(f"Could not get status summary: {e}\n")

        logger.info(f"Workflow report saved to {output_path}")


# Convenience functions for direct usage
def run_adaptive_fep_workflow(
    input_dir: str,
    output_dir: str,
    sim_config: GromacsFepSimulationConfig,
    ensemble_size: int = 5,
    initial_runtime_constant: float = 0.001,
    max_runtime_per_window: float = 30.0,
    short_run_runtime: float = 2.0,  # ns
    use_hpc: bool = True,
    run_sync: bool = False,  # run asynchronously (non-blocking)
    enable_monitoring: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    Run an adaptive FEP calculation with automatic setup.

    This is the main entry point for users who want to run adaptive
    FEP calculations with minimal setup.

    Returns
    -------
    dict
        Comprehensive results from the workflow
    """
    logger.info("Setting up adaptive FEP calculation...")

    # Set up calculation
    calculation = Calculation(
        input_dir=input_dir,
        output_dir=output_dir,
        sim_config=sim_config,
        ensemble_size=ensemble_size,
    )
    calculation.setup()

    # Create workflow manager
    fep_workflow = AdaptiveFepSimulationWorkflow(
        calculation=calculation,
        enable_monitoring=enable_monitoring,
        use_hpc=use_hpc,
        run_sync=run_sync,
    )

    results = fep_workflow.run_complete_adaptive_workflow(
        initial_runtime_constant=initial_runtime_constant,
        max_runtime_per_window=max_runtime_per_window,
        short_run_runtime=short_run_runtime,
        **kwargs,
    )

    # Store results for report generation
    fep_workflow._last_workflow_results = results

    # Save report
    fep_workflow.save_workflow_report()

    return results
