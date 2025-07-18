"""
Adaptive equilibration manager for GROMACS FEP simulations.

This module implements adaptive equilibration detection that automatically
adjusts runtime constants and uses the adaptive efficiency loop to optimize
simulation time allocation during equilibration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from binding_affinity_predicting.components.analysis.equilibrium_detecter import (
    EquilibriumDetectionManager,
)
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    Calculation,
)
from binding_affinity_predicting.components.simulation_fep.runtime_allocator import (
    OptimalRuntimeAllocator,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AdaptiveSimulationRunner:
    """
    Manager for adaptive equilibration with efficiency optimization.

    This class implements the adaptive equilibration algorithm that:
    1. Runs simulations with efficiency optimization
    2. Checks for equilibration using multiwindow detection
    3. Adjusts runtime constant and re-optimizes if not equilibrated
    4. Continues until equilibration is achieved
    """

    def __init__(
        self,
        calculation: Calculation,
        initial_runtime_constant: float = 0.001,
        equilibration_method: str = "multiwindow",
        max_runtime_per_window: float = 30.0,
        runtime_reduction_factor: float = 0.25,
        max_iterations: int = 10,
        use_hpc: bool = True,
    ):
        """
        Initialize adaptive simulation runner.

        Parameters
        ----------
        calculation : Calculation
            GROMACS calculation object to manage
        initial_runtime_constant : float, default 0.001
            The initial_runtime_constant (kcal**2 mol**-2 ns*-1) only affects behaviour if
            running adaptively. This is used to calculate how long to run each simulation for
            based on the current uncertainty of the per-window free energy estimate.
        equilibration_method : str, default "multiwindow"
            Method for equilibration detection
        max_runtime_per_window : float, default 30.0
            Maximum runtime per simulation window (ns)
        runtime_reduction_factor : float, default 0.25
            Factor to reduce runtime constant if not equilibrated (quarter it)
            This is consistent with default setting in A3FE. see below code snippet
            ```
            if equilibrated:
                break
            else:
                self.runtime_constant /= 4
                self._maximally_efficient = False
            ```
        max_iterations : int, default 10
            Maximum number of equilibration iterations
        """
        self.calculation = calculation
        self.initial_runtime_constant = initial_runtime_constant
        self.current_runtime_constant = initial_runtime_constant
        self.equilibration_method = equilibration_method
        self.max_runtime_per_window = max_runtime_per_window
        self.runtime_reduction_factor = runtime_reduction_factor
        self.max_iterations = max_iterations
        self.use_hpc = use_hpc

        # Initialize equilibrium detection manager
        self.detection_manager = EquilibriumDetectionManager(
            method=equilibration_method
        )

        # Initialize runtime allocator
        self.optimal_runtime_allocator = OptimalRuntimeAllocator(
            calculation=calculation,
            runtime_constant=self.current_runtime_constant,
            max_runtime_per_window=max_runtime_per_window,
            use_hpc=self.use_hpc,
        )

        # State tracking
        self._equilibrated = False
        self._iteration = 0
        self._equilibration_results: dict[str, Any] = {}

        logger.info("Initialized adaptive simulation manager:")
        logger.info(f"  Initial runtime constant: {initial_runtime_constant}")
        logger.info(f"  Equilibration method: {equilibration_method}")
        logger.info(f"  Max iterations: {max_iterations}")

    def run_simulation(self, run_nos: Optional[list[int]] = None) -> bool:
        """
        Run the adaptive simulation workflow.

        This implements the main algorithm:
        1. Run adaptive efficiency optimization
        2. Check for equilibration
        3. If not equilibrated, reduce runtime constant and repeat
        4. Continue until equilibrated or max iterations reached

        Parameters
        ----------
        run_nos : list[int], optional
            Run numbers to include in analysis

        Returns
        -------
        bool
            True if equilibration was achieved, False otherwise
        """
        logger.info("=" * 60)
        logger.info("STARTING ADAPTIVE SIMULATION WORKFLOW")
        logger.info("=" * 60)

        self._equilibrated = False
        self._iteration = 0

        while not self._equilibrated and self._iteration < self.max_iterations:
            self._iteration += 1

            logger.info(f"*** EQUILIBRATION ITERATION {self._iteration} ***")
            logger.info(f"Runtime constant: {self.current_runtime_constant}")

            # Step 1: Run adaptive efficiency optimization
            logger.info("Step 1: Running adaptive efficiency optimization...")
            try:
                self.optimal_runtime_allocator.set_runtime_constant(
                    self.current_runtime_constant
                )
                self.optimal_runtime_allocator.run_adaptive_efficiency_loop(
                    run_nos=run_nos
                )

                if self.optimal_runtime_allocator.kill_thread:
                    logger.info("Efficiency optimization was stopped")
                    break

                logger.info("Adaptive efficiency optimization completed")

            except Exception as e:
                logger.error(f"Efficiency optimization failed: {e}")
                break

            # Step 2: Check for equilibration
            logger.info("Step 2: Checking for equilibration...")
            try:
                equilibration_achieved = self._check_equilibration(run_nos)

                if equilibration_achieved:
                    self._equilibrated = True
                    logger.info("EQUILIBRATION ACHIEVED!")
                    break
                else:
                    logger.info("Equilibration not achieved")

            except Exception as e:
                logger.error(f"Equilibration detection failed: {e}")
                break

            # Step 3: Adjust runtime constant for next iteration
            if self._iteration < self.max_iterations:
                logger.info("Step 3: Adjusting runtime constant for next iteration...")
                self._adjust_runtime_constant()
            else:
                logger.warning(f"Maximum iterations ({self.max_iterations}) reached")

        # Final status
        if self._equilibrated:
            logger.info("=" * 60)
            logger.info("ADAPTIVE SIMULATION COMPLETED SUCCESSFULLY")
            logger.info(f"Final runtime constant: {self.current_runtime_constant}")
            logger.info(f"Iterations required: {self._iteration}")
            logger.info("=" * 60)
        else:
            logger.warning("=" * 60)
            logger.warning("ADAPTIVE SIMULATION DID NOT CONVERGE")
            logger.warning(f"Stopped after {self._iteration} iterations")
            logger.warning("=" * 60)

        return self._equilibrated

    def _check_equilibration(self, run_nos: Optional[list[int]]) -> bool:
        """
        Check if the calculation has reached equilibration.

        Uses the configured equilibration detection method on all legs.
        """
        logger.info(
            f"Checking equilibration using {self.equilibration_method} method..."
        )

        try:
            # Check equilibration for the entire calculation
            all_results = self.detection_manager.detect_calculation_equilibrium(
                calculation=self.calculation, run_nos=run_nos
            )

            self._equilibration_results = all_results

            # Determine if all legs are equilibrated
            all_equilibrated = True
            for leg_name, leg_results in all_results.items():
                leg_equilibrated = all(
                    result[0]
                    for result in leg_results.values()  # result[0] is equilibrated bool
                )

                logger.info(f"{leg_name} leg equilibrated: {leg_equilibrated}")

                if leg_equilibrated:
                    # Log equilibration details
                    for window_name, (
                        equilibrated,
                        frac_equil_time,
                    ) in leg_results.items():
                        if frac_equil_time is not None:
                            logger.info(
                                f" {window_name}: {frac_equil_time:.3f} fractional equilibration time"  # noqa: E501
                            )

                if not leg_equilibrated:
                    all_equilibrated = False

            logger.info(f"Overall equilibration status: {all_equilibrated}")
            return all_equilibrated

        except Exception as e:
            logger.error(f"Equilibration detection failed: {e}")
            return False

    def _adjust_runtime_constant(self) -> None:
        """
        Adjust the runtime constant for the next iteration.

        Reduces the runtime constant by the specified factor, which should
        lead to longer simulations in the next efficiency optimization.
        """
        old_constant = self.current_runtime_constant
        self.current_runtime_constant *= self.runtime_reduction_factor

        logger.info(
            f"Reducing runtime constant: {old_constant:.6f} → {self.current_runtime_constant:.6f}"
        )
        logger.info(
            f"This should increase predicted optimal runtimes by a factor of "
            f"{1/np.sqrt(self.runtime_reduction_factor):.2f}"
        )

        # Reset efficiency status in the manager
        self.optimal_runtime_allocator._maximally_efficient = False

    def stop(self) -> None:
        """Stop the adaptive simulation process."""
        logger.info("Stopping adaptive simulation...")
        self.optimal_runtime_allocator.stop()

    @property
    def is_equilibrated(self) -> bool:
        """Check if equilibration has been achieved."""
        return self._equilibrated

    @property
    def current_iteration(self) -> int:
        """Get the current iteration number."""
        return self._iteration

    def get_simulation_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the adaptive simulation process.

        Returns
        -------
        dict
            Status information including equilibration results and efficiency metrics
        """
        status = {
            'equilibrated': self._equilibrated,
            'current_iteration': self._iteration,
            'max_iterations': self.max_iterations,
            'initial_runtime_constant': self.initial_runtime_constant,
            'current_runtime_constant': self.current_runtime_constant,
            'equilibration_method': self.equilibration_method,
            'equilibration_results': self._equilibration_results,
            # TODO: maybe better to add optimalruntime_status at some point
            'optimalruntime_status': None,
        }

        return status

    def save_adaptive_simulation_run_report(
        self, output_dir: Optional[str] = None
    ) -> None:
        """
        Save a detailed report to file.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save report. If None, uses calculation output directory.
        """
        if output_dir is None:
            output_dir = self.calculation.output_dir

        output_path = Path(output_dir) / "adaptive_equilibration_report.txt"

        with open(output_path, 'w') as f:
            f.write("ADAPTIVE EQUILIBRATION REPORT\n")
            f.write("=" * 40 + "\n\n")

            # Key results
            f.write(f"Equilibration achieved: {self._equilibrated}\n")
            f.write(f"Iterations completed: {self._iteration}\n")
            f.write(f"Initial runtime constant: {self.initial_runtime_constant:.6f}\n")
            f.write(f"Final runtime constant: {self.current_runtime_constant:.6f}\n")
            f.write(f"Method: {self.equilibration_method}\n\n")

            # Equilibration status by leg
            if self._equilibration_results:
                f.write("EQUILIBRATION STATUS:\n")
                f.write("-" * 25 + "\n")
                for leg_name, leg_results in self._equilibration_results.items():
                    all_equilibrated = all(result[0] for result in leg_results.values())
                    f.write(
                        f"{leg_name} leg: {'EQUILIBRATED' if all_equilibrated else 'NOT EQUILIBRATED'}\n"  # noqa: E501
                    )

            # Efficiency status
            efficiency_status = self.optimal_runtime_allocator.get_allocator_status()
            f.write(
                f"\nEfficiency achieved: {efficiency_status['is_maximally_efficient']}\n"
            )
            f.write(
                f"Total simulation time: {efficiency_status['total_simulation_time']:.1f} ns\n"
            )

        logger.info(f"Equilibration report saved to {output_path}")


def run_adaptive_simulations(
    calculation: Calculation,
    initial_runtime_constant: float = 0.001,
    equilibration_method: str = "multiwindow",
    max_runtime_per_window: float = 30.0,
    run_nos: Optional[list[int]] = None,
    save_report: bool = True,
    **kwargs,
) -> AdaptiveSimulationRunner:
    """
    Convenience function to run the complete adaptive simulation workflow.

    Parameters
    ----------
    calculation : Calculation
        GROMACS calculation object
    initial_runtime_constant : float, default 0.001
        Initial runtime constant for efficiency optimization
    equilibration_method : str, default "multiwindow"
        Method for equilibration detection ('multiwindow' or 'block_gradient')
    max_runtime_per_window : float, default 30.0
        Maximum runtime per window (ns)
    run_nos : list[int], optional
        Run numbers to include in analysis
    save_report : bool, default True
        Whether to save a detailed report
    **kwargs
        Additional arguments for AdaptiveSimulationRunner

    Returns
    -------
    AdaptiveSimulationManager
        Manager object with results
    """
    logger.info("Starting adaptive simulation workflow...")

    # Create and run the manager
    manager = AdaptiveSimulationRunner(
        calculation=calculation,
        initial_runtime_constant=initial_runtime_constant,
        equilibration_method=equilibration_method,
        max_runtime_per_window=max_runtime_per_window,
        **kwargs,
    )

    # Run the adaptive simulation
    success = manager.run_simulation(run_nos=run_nos)

    if save_report:
        manager.save_adaptive_simulation_run_report()

    if success:
        logger.info(
            f"Adaptive simulation completed successfully in {manager.current_iteration} iterations"
        )
    else:
        logger.warning(
            f"Adaptive simulation did not converge after {manager.current_iteration} iterations"
        )

    return manager
