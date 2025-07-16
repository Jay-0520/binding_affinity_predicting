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
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    Calculation,
)
from binding_affinity_predicting.components.simulation_fep.lambda_optimizer import (
    LambdaOptimizationManager,
    OptimizationConfig,
)
from binding_affinity_predicting.components.simulation_fep.run_adaptive_simulation import (
    AdaptiveSimulationRunner,
)
from binding_affinity_predicting.components.simulation_fep.runtime_allocator import (
    OptimalRuntimeAllocator,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AdaptiveFepWorkflowManager:
    """
    High-level manager for adaptive FEP workflows.

    This class provides a unified interface for running various adaptive
    optimization strategies for GROMACS FEP calculations.
    """

    def __init__(self, calculation: Calculation):
        """
        Initialize the adaptive FEP workflow manager.

        Parameters
        ----------
        calculation : Calculation
            GROMACS calculation object to manage
        """
        self.calculation = calculation
        self.workflow_results: dict[str, Any] = {}

    def _run_lambda_optimization(
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
            'type': 'lambda_optimization',
            'successful': successful_legs == total_legs,
            'successful_legs': successful_legs,
            'total_legs': total_legs,
            'applied': apply_optimization,
            'optimization_results': optimization_results,
        }

        self.workflow_results['lambda_optimization'] = results
        return results

    def _run_adaptive_simulation(
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
        _ = manager.run_adaptive_simulation(run_nos=run_nos)

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
        short_run_runtime: float = 2.0,
        initial_runtime_constant: float = 0.001,
        equilibration_method: str = "multiwindow",
        max_runtime_per_window: float = 30.0,
        optimize_lambda_spacing: bool = True,
        run_nos: Optional[list[int]] = None,
        use_hpc: bool = True,
        run_sync: bool = True,
    ) -> dict[str, Any]:
        """
        Run the complete adaptive FEP workflow following A3FE protocol.

        This workflow implements the correct A3FE sequence:
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

        Returns
        -------
        dict
            Comprehensive results from all workflow phases
        """
        logger.info("🚀 Running Complete Adaptive FEP Workflow (BAB Protocol)...")
        logger.info("=" * 70)

        workflow_results: dict[str, Any] = {
            'workflow_type': 'complete_adaptive_fep_workflow_BAB',
            'phases_completed': [],
            'overall_successful': False,
            'phase_results': {},
        }

        try:
            # Phase 1: Short simulations for gradient collection
            logger.info("PHASE 1: SHORT SIMULATIONS FOR GRADIENT COLLECTION")
            logger.info("-" * 50)
            logger.info(
                f"Running {short_run_runtime} ns simulations for lambda optimization..."
            )

            # Run short simulations
            self.calculation.run(
                runtime=short_run_runtime, use_hpc=use_hpc, run_sync=run_sync
            )

            # Wait for completion
            logger.info("Waiting for short simulations to complete...")
            self._wait_for_calculation()

            short_sim_results = {
                'type': 'short_simulations',
                'runtime': short_run_runtime,
                'successful': not self.calculation.failed,
            }

            workflow_results['phases_completed'].append('short_simulations')
            workflow_results['phase_results']['short_simulations'] = short_sim_results

            if not short_sim_results['successful']:
                logger.error("Short simulations failed. Stopping workflow.")
                return workflow_results

            logger.info(
                f"Short simulations completed successfully ({short_run_runtime} ns)"
            )

            # Phase 2: Lambda spacing optimization
            if optimize_lambda_spacing:
                logger.info("\nPHASE 2: LAMBDA SPACING OPTIMIZATION")
                logger.info("-" * 50)

                optimization_results = self._run_lambda_optimization(
                    run_nos=run_nos,
                    equilibrated=False,  # use short unequilibrated data
                    apply_optimization=True,  # apply the result directly in prod
                )

                workflow_results['phases_completed'].append('lambda_optimization')
                workflow_results['phase_results'][
                    'lambda_optimization'
                ] = optimization_results

                if optimization_results['successful']:
                    logger.info("Lambda spacing optimization completed successfully")
                    logger.info(
                        "Calculation has been updated with optimized lambda values"
                    )
                else:
                    logger.warning(
                        "Lambda optimization had issues, continuing with original lambda values..."
                    )
            else:
                logger.info(
                    "\nSKIPPING LAMBDA OPTIMIZATION (optimize_lambda_spacing=False)"
                )

            # Phase 3: Production simulations with adaptive equilibration and efficiency/runtime
            logger.info(
                "\nPHASE 3: PRODUCTION WITH ADAPTIVE EQUILIBRATION & EFFICIENCY"
            )
            logger.info("-" * 50)

            # Use optimized lambda values (if optimization was successful) for production runs
            production_results = self._run_adaptive_simulation(
                initial_runtime_constant=initial_runtime_constant,
                equilibration_method=equilibration_method,
                max_runtime_per_window=max_runtime_per_window,
                run_nos=run_nos,
                use_hpc=use_hpc,
            )

            workflow_results['phases_completed'].append('production_adaptive')
            workflow_results['phase_results'][
                'production_adaptive'
            ] = production_results

            if production_results['successful']:
                logger.info("Production adaptive simulation completed successfully")
            else:
                logger.warning("Production adaptive simulation had issues")

            # Overall success assessment
            required_phases = ['short_simulations', 'production_adaptive']
            if optimize_lambda_spacing:
                # Lambda optimization is not strictly required for success
                pass

            workflow_results['overall_successful'] = all(
                phase in workflow_results['phases_completed']
                and workflow_results['phase_results'][phase].get('successful', False)
                for phase in required_phases
            )

            # Summary statistics
            total_runtime = short_run_runtime
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
                'lambda_optimization_applied': optimize_lambda_spacing
                and workflow_results['phase_results']
                .get('lambda_optimization', {})
                .get('applied', False),
                'final_runtime_constant': production_results.get(
                    'final_runtime_constant', initial_runtime_constant
                ),
                'equilibration_iterations': production_results.get('iterations', 0),
            }

        except Exception as e:
            logger.error(f"Workflow failed with exception: {e}")
            workflow_results['error'] = str(e)

        # Final summary
        logger.info("\n" + "=" * 70)
        if workflow_results['overall_successful']:
            logger.info("🎉 COMPLETE ADAPTIVE WORKFLOW (A3FE) FINISHED SUCCESSFULLY")
            summary = workflow_results['summary']
            logger.info(f"Total simulation time: {summary['total_runtime']} ns")
            logger.info(
                f"Lambda optimization applied: {summary['lambda_optimization_applied']}"
            )
            logger.info(
                f"Equilibration iterations: {summary['equilibration_iterations']}"
            )
            logger.info(
                f"Final runtime constant: {summary['final_runtime_constant']:.6f}"
            )
        else:
            logger.warning("⚠️ WORKFLOW COMPLETED WITH SOME ISSUES")

        logger.info(
            f"Phases completed: {', '.join(workflow_results['phases_completed'])}"
        )
        logger.info("=" * 70)

        return workflow_results

    def _wait_for_calculation(self, check_interval: int = 60) -> None:
        """Wait for calculation to complete."""
        if self.calculation.virtual_queue:
            logger.info("Waiting for simulations to complete...")
            while self.calculation.running:
                time.sleep(check_interval)
                self.calculation.virtual_queue.update()

            if self.calculation.failed:
                logger.warning("Some simulations failed")
            else:
                logger.info("All simulations completed successfully")

    def save_workflow_report(self, output_dir: Optional[str] = None) -> None:
        """
        Save a comprehensive workflow report.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save report
        """
        if output_dir is None:
            output_dir = self.calculation.output_dir

        output_path = Path(output_dir) / "adaptive_fep_workflow_report.txt"

        with open(output_path, 'w') as f:
            f.write("ADAPTIVE FEP WORKFLOW REPORT\n")
            f.write("=" * 60 + "\n\n")

            if not self.workflow_results:
                f.write("No workflow results available.\n")
                return

            for workflow_name, results in self.workflow_results.items():
                f.write(f"{workflow_name.upper()} RESULTS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Type: {results.get('type', 'Unknown')}\n")
                f.write(f"Successful: {results.get('successful', False)}\n")

                if 'phases_completed' in results:
                    f.write(
                        f"Phases completed: {', '.join(results['phases_completed'])}\n"
                    )

                if 'iterations' in results:
                    f.write(f"Iterations: {results['iterations']}\n")

                if 'final_runtime_constant' in results:
                    f.write(
                        f"Final runtime constant: {results['final_runtime_constant']:.6f}\n"
                    )

                f.write("\n")

        logger.info(f"Workflow report saved to {output_path}")


# Convenience functions for direct usage
def run_adaptive_fep_workflow(
    input_dir: str,
    output_dir: str,
    sim_config: GromacsFepSimulationConfig,
    ensemble_size: int = 5,
    initial_runtime_constant: float = 0.001,
    max_runtime_per_window: float = 30.0,
    use_hpc: bool = True,
    use_sync: bool = False,
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
    workflow_manager = AdaptiveFepWorkflowManager(calculation)

    results = workflow_manager.run_complete_adaptive_workflow(
        initial_runtime_constant=initial_runtime_constant,
        max_runtime_per_window=max_runtime_per_window,
        use_hpc=use_hpc,
        use_sync=use_sync,
        **kwargs,
    )

    # Save report
    workflow_manager.save_workflow_report()

    return results
