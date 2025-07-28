"""
Adaptive FEP workflows integration module.

This module provides high-level workflows that integrate adaptive efficiency,
adaptive equilibration, and lambda optimization for GROMACS FEP calculations.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional, cast

from binding_affinity_predicting.components.simulation_fep.adaptive_simulation_runner import (
    AdaptiveSimulationRunner,
)
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    Calculation,
    Leg,
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


class LegToCalculationWrapper:
    """
    Minimal wrapper to make a Leg compatible with StatusMonitor.
    This is used internally when creating status monitors for individual legs.
    """

    def __init__(self, leg: Leg):
        self._sub_sim_runners = [leg]
        self.output_dir = leg.output_dir
        self.virtual_queue = leg.virtual_queue

    @property
    def running(self) -> bool:
        return self._sub_sim_runners[0].running

    @property
    def failed(self) -> bool:
        return self._sub_sim_runners[0].failed


class AdaptiveFepSimulationLegLevelWorkflow:
    """
    Leg-level manager for adaptive FEP workflows.

    This class provides a unified interface for running various adaptive
    optimization strategies for a SINGLE GROMACS FEP leg.

    For processing multiple legs, create multiple instances of this class
    or use the CalculationAdaptiveFepOrchestrator.
    """

    def __init__(
        self,
        leg: Leg,
        enable_monitoring: bool = True,
        use_hpc: bool = True,
        run_sync: bool = True,
        skip_completed_phases: bool = True,
    ) -> None:
        """
        Initialize the adaptive FEP workflow manager.

        Parameters
        ----------
        leg : Leg
            GROMACS leg object to manage
        enable_monitoring : bool, default True
            Whether to enable status monitoring
        use_hpc : bool, default True
            Whether to use HPC for simulations
        run_sync : bool, default True
            Whether to run synchronously
        skip_completed_phases : bool, default True
            Whether to skip completed phases (resume functionality)
        """
        self.leg = leg
        self.leg_name = leg.leg_type.name.lower()
        self.workflow_results: dict[str, Any] = {}
        self._last_workflow_results: dict[str, Any] = {}
        self.enable_monitoring = enable_monitoring
        self.use_hpc = use_hpc
        self.run_sync = run_sync
        self.skip_completed_phases = skip_completed_phases

        self.status_monitor: Optional[StatusMonitor]
        # Initialize status monitor
        if self.enable_monitoring:
            # Create a temporary calculation wrapper for the status monitor
            temp_calc = LegToCalculationWrapper(self.leg)
            self.status_monitor = StatusMonitor(cast(Calculation, temp_calc))
            logger.info(f"{self.leg_name} leg - Status monitoring enabled")
        else:
            self.status_monitor = None
            logger.info(f"{self.leg_name} leg - Status monitoring disabled")

        # Resume functionality attributes
        self.workflow_state_file = (
            Path(self.leg.output_dir) / f"{self.leg_name}_workflow_completed_phases.txt"
        )
        self.completed_phases = self._load_completed_phases()

        logger.info(f"Initialized adaptive FEP workflow for {self.leg_name} leg")

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
            return not self.leg.failed

        try:
            status = self.status_monitor.get_status()
            job_counts = status.get("job_counts", {})
            failed_count = job_counts.get("FAILED", 0)
            return failed_count == 0
        except Exception as e:
            logger.warning(f"Could not check simulation success: {e}")
            return not self.leg.failed

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

        optimization_result = manager.optimizer.optimize(
            lambda_windows=self.leg.lambda_windows,
            leg_type=self.leg.leg_type,
            run_nos=run_nos,
            equilibrated=equilibrated,
        )
        success = optimization_result.success
        applied = False

        if success and apply_optimization:
            try:
                logger.info(f"{self.leg_name} leg - applying optimization results...")
                manager._apply_to_leg(
                    self.leg, optimization_result, backup_original=True
                )
                applied = True
                logger.info(f"{self.leg_name} leg - optimization applied successfully")
            except Exception as e:
                logger.error(f"{self.leg_name} leg - failed to apply optimization: {e}")
                success = False

        results = {
            'type': 'lambda_optimizing',
            'leg_name': self.leg_name,
            'successful': success,
            'applied': applied,
            'optimization_result': optimization_result,
            'improvement_factor': (
                optimization_result.overall_improvement if success else 1.0
            ),
        }

        self.workflow_results['lambda_optimizing'] = results
        return results

    def _run_adaptively(
        self,
        initial_runtime_constant: float = 0.001,
        equilibration_method: str = "multiwindow",
        max_runtime_per_window: float = 30.0,
        max_iterations: int = 10,
        run_nos: Optional[list[int]] = None,
        # use_hpc: bool = True,
    ) -> dict[str, Any]:
        """
        Run adaptive simulation workflow (equilibration + efficiency optimization)
          for this leg.

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
        logger.info(f"{self.leg_name} leg - Running Adaptive Simulation Protocol...")

        manager = AdaptiveSimulationRunner(
            target=self.leg,
            initial_runtime_constant=initial_runtime_constant,
            equilibration_method=equilibration_method,
            max_runtime_per_window=max_runtime_per_window,
            max_iterations=max_iterations,
            use_hpc=self.use_hpc,
        )
        success = manager.run_simulation(run_nos=run_nos)

        results = {
            'type': 'adaptive_simulation_protocol',
            'leg_name': self.leg_name,
            'successful': success,
            'iterations': manager.current_iteration,
            'final_runtime_constant': manager.current_runtime_constant,
            'status': manager.get_simulation_status(),
        }

        self.workflow_results['adaptive_simulation_protocol'] = results
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
        Run the complete adaptive FEP workflow for this leg with simple monitoring.

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
        logger.info(
            f"🚀 {self.leg_name.upper()} LEG - Running Complete Adaptive FEP Workflow (BAB Protocol)..."  # noqa: E501
        )
        logger.info("=" * 70)

        # Handle force rerun
        if force_rerun_phases:
            for phase in force_rerun_phases:
                if phase in self.completed_phases:
                    logger.info(
                        f"{self.leg_name} leg - forcing rerun of phase: {phase}"
                    )
                    self.completed_phases.discard(phase)

        # Print resume information
        if self.skip_completed_phases and self.completed_phases:
            logger.info(
                f"{self.leg_name} leg - Resume mode enabled. Previously completed phases: {self.completed_phases}"  # noqa: E501
            )
        elif self.skip_completed_phases:
            logger.info(
                f"{self.leg_name} leg - Resume mode enabled. No previously completed phases found."
            )
        else:
            logger.info(
                f"{self.leg_name} leg - Resume mode disabled. All phases will be executed."
            )

        workflow_results: dict[str, Any] = {
            'workflow_type': f'complete_adaptive_fep_workflow_leg_{self.leg_name}',
            'leg_name': self.leg_name,
            'leg_type': self.leg.leg_type.name,
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
                logger.info(
                    f"{self.leg_name} leg - PHASE 1: SHORT SIMULATIONS (SKIPPED - ALREADY COMPLETED)"  # noqa: E501
                )
                logger.info("=" * 50)
                logger.info(
                    f"{self.leg_name} leg - Using existing short simulation results"
                )

                short_sim_results = {
                    'type': 'short_simulations',
                    'leg_name': self.leg_name,
                    'runtime': short_run_runtime,
                    'successful': True,
                    'skipped': True,
                    'reason': 'Previously completed',
                }
                cast(list[str], workflow_results['phases_skipped']).append(phase1_name)
                workflow_results['phase_results'][phase1_name] = short_sim_results

            else:
                logger.info(
                    f"{self.leg_name} leg - PHASE 1: SHORT SIMULATIONS FOR GRADIENT COLLECTION"
                )
                logger.info("=" * 50)
                logger.info(
                    f"{self.leg_name} leg - Running {short_run_runtime} ns simulations for lambda optimization..."  # noqa: E501
                )

                # Run short simulations on this leg only
                self.leg.run(
                    runtime=short_run_runtime,
                    use_hpc=self.use_hpc,
                )

                # Show initial status after starting simulations
                if self.enable_monitoring:
                    logger.info(
                        f"{self.leg_name} leg - Phase 1 started, checking initial status..."
                    )
                    self._print_simple_status()

                # Wait for completion
                logger.info(
                    f"{self.leg_name} leg - Waiting for short simulations to complete..."
                )
                self._wait_for_leg(check_interval=monitor_interval)

                short_sim_results = {
                    'type': 'short_simulations',
                    'leg_name': self.leg_name,
                    'runtime': short_run_runtime,
                    'successful': self._check_simulation_success(),
                    'skipped': False,
                }

                if not short_sim_results['successful']:
                    logger.error(
                        f"{self.leg_name} leg - Short simulations failed. Stopping workflow."
                    )
                    if self.enable_monitoring:
                        self._print_simple_status()  # Show final status
                    workflow_results['phase_results'][phase1_name] = short_sim_results
                    return workflow_results

                logger.info(
                    f"{self.leg_name} leg - Short simulations completed successfully ({short_run_runtime} ns)"  # noqa: E501
                )

                cast(list[str], workflow_results['phases_completed']).append(
                    phase1_name
                )
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
                        f"{self.leg_name} leg - PHASE 2: LAMBDA OPTIMIZATION (SKIPPED - ALREADY COMPLETED)"  # noqa: E501
                    )
                    logger.info("=" * 50)

                    optimization_results = {
                        'type': 'lambda_optimizing',
                        'leg_name': self.leg_name,
                        'successful': True,
                        'applied': True,
                        'skipped': True,
                        'reason': 'Previously completed',
                    }

                    cast(list[str], workflow_results['phases_skipped']).append(
                        phase2_name
                    )
                    workflow_results['phase_results'][
                        phase2_name
                    ] = optimization_results
                else:
                    logger.info(
                        f"{self.leg_name} leg - PHASE 2: LAMBDA SPACING OPTIMIZATION"
                    )
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
                            f"{self.leg_name} leg - Lambda spacing optimization completed successfully"  # noqa: E501
                        )
                        logger.info(
                            f"{self.leg_name} leg - Leg has been updated with optimized lambda values"  # noqa: E501
                        )
                        cast(list[str], workflow_results['phases_completed']).append(
                            phase2_name
                        )
                        self._save_completed_phase(phase2_name)
                    else:
                        logger.warning(
                            f"{self.leg_name} leg - Lambda optimization had issues, continuing with original lambda values..."  # noqa: E501
                        )
            else:
                logger.info(
                    f"{self.leg_name} leg - SKIPPING PHASE 2 LAMBDA OPTIMIZATION (optimize_lambda_spacing=False)"  # noqa: E501
                )

            # ===================================================================================
            # Phase 3: Production simulations with adaptive equilibration and efficiency/runtime
            # ===================================================================================
            phase3_name = 'production_adaptive'

            if self._is_phase_completed(phase3_name):
                logger.info(
                    f"{self.leg_name} leg - PHASE 3: PRODUCTION ADAPTIVE (SKIPPED - ALREADY COMPLETED)"  # noqa: E501
                )
                logger.info("=" * 50)

                production_results = {
                    'type': 'adaptive_simulation_workflow',
                    'leg_name': self.leg_name,
                    'successful': True,
                    'skipped': True,
                    'reason': 'Previously completed',
                }

                cast(list[str], workflow_results['phases_skipped']).append(phase3_name)
                workflow_results['phase_results'][phase3_name] = production_results
            else:
                logger.info(
                    f"{self.leg_name} leg - PHASE 3: PRODUCTION WITH ADAPTIVE EQUILIBRATION & EFFICIENCY"  # noqa: E501
                )
                logger.info("=" * 50)

                # Use optimized lambda values (if optimization was successful) for production runs
                production_results = self._run_adaptively(
                    initial_runtime_constant=initial_runtime_constant,
                    equilibration_method=equilibration_method,
                    max_runtime_per_window=max_runtime_per_window,
                    run_nos=run_nos,
                )

                workflow_results['phase_results'][phase3_name] = production_results

                if production_results['successful']:
                    logger.info(
                        f"{self.leg_name} leg - Production adaptive simulation completed successfully"  # noqa: E501
                    )
                    cast(list[str], workflow_results['phases_completed']).append(
                        phase3_name
                    )
                    self._save_completed_phase(phase3_name)
                else:
                    logger.warning(
                        f"{self.leg_name} leg - Production adaptive simulation had issues"
                    )

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
                and cast(dict[str, bool], workflow_results['phase_results'][phase]).get(
                    'successful', False
                )
                for phase in required_phases
            )
            # Summary statistics
            total_runtime = 0
            if 'short_simulations' in workflow_results['phase_results']:
                short_results: dict[str, Any] = workflow_results['phase_results'][
                    'short_simulations'
                ]
                if not short_results.get('skipped', False):
                    total_runtime += short_results.get('runtime', 0)

            if 'production_adaptive' in workflow_results['phase_results']:
                production_status: dict[str, Any] = workflow_results['phase_results'][
                    'production_adaptive'
                ].get('status', {})
                if 'optimalruntime_status' in production_status:
                    runtime_status = production_status['optimalruntime_status']
                    if runtime_status and 'total_simulation_time' in runtime_status:
                        total_runtime += runtime_status['total_simulation_time']

            workflow_results['summary'] = {
                'leg_name': self.leg_name,
                'total_runtime': total_runtime,
                'lambda_optimization_applied': (
                    optimize_lambda_spacing
                    and workflow_results['phase_results']
                    .get('lambda_optimizing', {})
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
            logger.error(f"{self.leg_name} leg - Workflow failed with exception: {e}")
            workflow_results['error'] = str(e)

        # Final summary
        logger.info("\n" + "=" * 70)
        if workflow_results['overall_successful']:
            logger.info(
                f"🎉 {self.leg_name.upper()} LEG - COMPLETE ADAPTIVE WORKFLOW FINISHED SUCCESSFULLY"
            )
        else:
            logger.warning(
                f"⚠️ {self.leg_name.upper()} LEG - WORKFLOW COMPLETED WITH SOME ISSUES"
            )

        logger.info(
            f"{self.leg_name} leg - Phases completed: {', '.join(workflow_results['phases_completed'])}"  # noqa: E501
        )
        if workflow_results['phases_skipped']:
            logger.info(
                f"{self.leg_name} leg - Phases skipped: {', '.join(workflow_results['phases_skipped'])}"  # noqa: E501
            )
        logger.info("=" * 70)

        # Show final status
        if self.enable_monitoring:
            self._print_simple_status()

        return workflow_results

    def _wait_for_leg(self, check_interval: int = 60) -> None:
        """Wait for calculation to complete with status monitoring."""
        logger.info(f"{self.leg_name} leg - waiting for simulations to complete...")

        if self.leg.virtual_queue:
            logger.info(
                "calc virtual_queue is not None; waiting for simulations to complete..."
            )
            while self.leg.running:
                # Print status during waiting if monitoring is enabled
                if self.enable_monitoring:
                    self._print_simple_status()

                time.sleep(check_interval)
                self.leg.virtual_queue.update()

        else:
            # For local runs, just use a simple wait
            logger.info("Running locally - waiting briefly for completion...")
            while True:
                if self.enable_monitoring:
                    self._print_simple_status()
                if not self.leg.running:
                    logger.info("Local simulations completed")
                    break
                time.sleep(check_interval)

        if self._check_simulation_success():
            logger.info(
                f"✅ {self.leg_name} leg - All simulations completed successfully"
            )
        else:
            logger.warning(f"⚠️ {self.leg_name} leg - Some simulations failed")

    def save_workflow_report(self, output_dir: Optional[str] = None) -> None:
        """
        Save a comprehensive workflow report using the returned workflow results.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save report
        """
        if output_dir is None:
            output_dir = self.leg.output_dir

        output_path = (
            Path(output_dir) / f"adaptive_fep_workflow_report_{self.leg_name}_leg.txt"
        )

        # Get the latest workflow results from the last run
        if (
            not hasattr(self, '_last_workflow_results')
            or not self._last_workflow_results
        ):
            logger.warning(
                f"{self.leg_name} leg - No workflow results available for report generation"
            )
            return

        results = self._last_workflow_results

        with open(output_path, 'w') as f:
            f.write(f"ADAPTIVE FEP WORKFLOW REPORT - {self.leg_name.upper()} LEG\n")
            f.write("=" * 60 + "\n\n")

            # Write workflow metadata
            f.write(f"Leg Name: {results.get('leg_name', 'Unknown')}\n")
            f.write(f"Leg Type: {results.get('leg_type', 'Unknown')}\n")
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
                if 'improvement_factor' in phase_result:
                    f.write(
                        f"Improvement factor: {phase_result['improvement_factor']:.3f}\n"
                    )
                if 'applied' in phase_result:
                    f.write(f"Applied: {phase_result['applied']}\n")

                f.write("\n")

            # Write summary
            if 'summary' in results:
                summary = results['summary']
                f.write("LEG WORKFLOW SUMMARY:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Leg Name: {summary.get('leg_name', 'Unknown')}\n")
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

        logger.info(f"{self.leg_name} leg - Workflow report saved to {output_path}")

    def get_status(self) -> dict[str, Any]:
        """Get current status of this leg's workflow."""
        return dict(self.workflow_results)

    def cleanup_state(self) -> None:
        """Clean up state files for fresh start."""
        if self.workflow_state_file.exists():
            try:
                self.workflow_state_file.unlink()
                logger.info(f"{self.leg_name} leg - cleaned up workflow state file")
            except Exception as e:
                logger.warning(f"{self.leg_name} leg - could not clean up state: {e}")

        self.completed_phases = set()


class AdaptiveFepSimulationCalculationLevelWorkflow:
    """
    Orchestrator that manages multiple AdaptiveFepSimulationLegLevelWorkflow instances
    for processing an entire calculation.
    """

    def __init__(
        self,
        calculation: Calculation,
        enable_monitoring: bool = True,
        use_hpc: bool = True,
        run_sync: bool = True,
        skip_completed_phases: bool = True,
    ):
        self.calculation = calculation
        self.enable_monitoring = enable_monitoring
        self.use_hpc = use_hpc
        self.run_sync = run_sync
        self.skip_completed_phases = skip_completed_phases

        # Create leg workflows
        self.leg_workflows: dict[str, AdaptiveFepSimulationLegLevelWorkflow] = {}

        for leg in calculation.legs:
            leg_name = leg.leg_type.name.lower()
            self.leg_workflows[leg_name] = AdaptiveFepSimulationLegLevelWorkflow(
                leg=leg,
                enable_monitoring=enable_monitoring,
                use_hpc=use_hpc,
                run_sync=run_sync,
                skip_completed_phases=skip_completed_phases,
            )

        logger.info(
            f"Initialized calculation orchestrator with {len(self.leg_workflows)} leg workflows"
        )

    def run_all_legs(
        self,
        run_sequentially: bool = True,
        **kwargs,
    ) -> dict[str, dict[str, Any]]:
        """
        Run adaptive workflows for all legs.

        Parameters
        ----------
        run_sequentially : bool, default True
            Whether to run legs sequentially (True) or in parallel (False)
        **kwargs
            Parameters passed to each leg workflow

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Results for each leg
        """
        logger.info("🚀 Running adaptive workflows for all legs...")

        results = {}
        if run_sequentially:
            for leg_name, leg_workflow in self.leg_workflows.items():
                logger.info(f"\n{'='*70}")
                logger.info(f"PROCESSING {leg_name.upper()} LEG WITH ADAPTIVE WORKFLOW")
                logger.info(f"{'='*70}")

                leg_result = leg_workflow.run_complete_adaptive_workflow(**kwargs)
                results[leg_name] = leg_result

                # Store results for report generation
                leg_workflow._last_workflow_results = leg_result
        else:
            # TODO: Implement parallel processing using threading
            logger.warning("Parallel processing not implemented, using sequential")
            return self.run_all_legs(run_sequentially=True, **kwargs)

        # Summary
        successful_legs = sum(
            1 for result in results.values() if result['overall_successful']
        )
        total_legs = len(results)

        logger.info(f"\n{'='*70}")
        if successful_legs == total_legs:
            logger.info("🎉 ALL LEGS COMPLETED SUCCESSFULLY")
        else:
            logger.warning(
                f"⚠️ PARTIAL SUCCESS: {successful_legs}/{total_legs} legs successful"
            )

        for leg_name, result in results.items():
            status = "✅" if result['overall_successful'] else "❌"
            logger.info(f"  {leg_name.upper()}: {status}")

        logger.info(f"{'='*70}")

        return results

    def get_leg_workflow(self, leg_name: str) -> AdaptiveFepSimulationLegLevelWorkflow:
        """Get the workflow for a specific leg."""
        if leg_name not in self.leg_workflows:
            raise ValueError(f"Leg not found: {leg_name}")
        return self.leg_workflows[leg_name]

    def retry_leg(self, leg_name: str, **kwargs) -> dict[str, Any]:
        """Retry a specific leg."""
        leg_workflow = self.get_leg_workflow(leg_name)
        return leg_workflow.run_complete_adaptive_workflow(**kwargs)

    def save_all_reports(self) -> None:
        """Save workflow reports for all legs."""
        for leg_workflow in self.leg_workflows.values():
            leg_workflow.save_workflow_report()

    def cleanup_all_states(self) -> None:
        """Clean up state files for all legs."""
        for leg_workflow in self.leg_workflows.values():
            leg_workflow.cleanup_state()

    def get_overall_status(self) -> dict[str, Any]:
        """Get overall status across all legs."""
        all_statuses = {}
        for leg_name, leg_workflow in self.leg_workflows.items():
            all_statuses[leg_name] = leg_workflow.get_status()

        # Calculate overall statistics
        total_legs = len(all_statuses)
        successful_legs = sum(
            1
            for status in all_statuses.values()
            if status.get('overall_successful', False)
        )

        return {
            'total_legs': total_legs,
            'successful_legs': successful_legs,
            'success_rate': successful_legs / total_legs if total_legs > 0 else 0,
            'leg_statuses': all_statuses,
        }


# Convenience functions for direct usage
def run_adaptive_fep_workflow_single_leg(
    leg: Leg,
    short_run_runtime: float = 2.0,
    initial_runtime_constant: float = 0.001,
    max_runtime_per_window: float = 30.0,
    optimize_lambda_spacing: bool = True,
    use_hpc: bool = True,
    run_sync: bool = True,
    enable_monitoring: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    Run an adaptive FEP workflow on a single leg.

    This is a convenience function for users who want to run adaptive
    FEP calculations on individual legs.

    Parameters
    ----------
    leg : Leg
        GROMACS leg object to process
    short_run_runtime : float, default 2.0
        Runtime for gradient collection phase (ns)
    initial_runtime_constant : float, default 0.001
        Initial runtime constant for adaptive phase
    max_runtime_per_window : float, default 30.0
        Maximum runtime per window (ns)
    optimize_lambda_spacing : bool, default True
        Whether to run lambda optimization
    use_hpc : bool, default True
        Whether to use HPC
    run_sync : bool, default True
        Whether to run synchronously
    enable_monitoring : bool, default True
        Whether to enable monitoring
    **kwargs
        Additional arguments for the workflow

    Returns
    -------
    Dict[str, Any]
        Results from the leg workflow
    """
    logger.info(f"Setting up adaptive FEP workflow for {leg.leg_type.name} leg...")

    # Create leg workflow
    leg_workflow = AdaptiveFepSimulationLegLevelWorkflow(
        leg=leg,
        enable_monitoring=enable_monitoring,
        use_hpc=use_hpc,
        run_sync=run_sync,
    )
    results = leg_workflow.run_complete_adaptive_workflow(
        short_run_runtime=short_run_runtime,
        initial_runtime_constant=initial_runtime_constant,
        max_runtime_per_window=max_runtime_per_window,
        optimize_lambda_spacing=optimize_lambda_spacing,
        **kwargs,
    )

    # Store results for report generation
    leg_workflow._last_workflow_results = results

    # Save report
    leg_workflow.save_workflow_report()

    return results


def run_adaptive_fep_workflow_calculation(
    calculation: Calculation,
    short_run_runtime: float = 2.0,
    initial_runtime_constant: float = 0.001,
    max_runtime_per_window: float = 30.0,
    optimize_lambda_spacing: bool = True,
    use_hpc: bool = True,
    run_sync: bool = True,
    enable_monitoring: bool = True,
    run_sequentially: bool = True,
    **kwargs,
) -> dict[str, dict[str, Any]]:
    """
    Run adaptive FEP workflows for an entire calculation using leg-level processing.

    Parameters
    ----------
    calculation : Calculation
        GROMACS calculation object containing multiple legs
    run_sequentially : bool, default True
        Whether to process legs sequentially
    **kwargs
        Additional arguments passed to each leg workflow

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Results for each leg in the calculation
    """
    logger.info("Setting up adaptive FEP workflows for entire calculation...")

    # Create orchestrator
    orchestrator = AdaptiveFepSimulationCalculationLevelWorkflow(
        calculation=calculation,
        enable_monitoring=enable_monitoring,
        use_hpc=use_hpc,
        run_sync=run_sync,
    )

    # Run all legs
    results = orchestrator.run_all_legs(
        run_sequentially=run_sequentially,
        short_run_runtime=short_run_runtime,
        initial_runtime_constant=initial_runtime_constant,
        max_runtime_per_window=max_runtime_per_window,
        optimize_lambda_spacing=optimize_lambda_spacing,
        **kwargs,
    )

    # Save all reports
    orchestrator.save_all_reports()

    return results
