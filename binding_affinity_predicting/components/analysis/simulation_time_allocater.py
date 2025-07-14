import logging
import time
from typing import Optional

import numpy as np

from binding_affinity_predicting.components.analysis.gradient_analyzer import (
    GradientAnalyzer,
)
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    Calculation,
    LambdaWindow,
    Leg,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AdaptiveEfficiencyManager:
    """
    Manager for adaptive efficiency allocation in GROMACS FEP calculations.

    This class implements the adaptive efficiency algorithm from A3FE that
    automatically allocates simulation time to achieve maximal estimation
    efficiency of free energy differences.
    """

    def __init__(
        self,
        calculation: Calculation,
        runtime_constant: float = 0.001,
        relative_simulation_cost: float = 1.0,
        max_runtime_per_window: float = 30.0,
        cycle_pause: int = 60,
        use_hpc: bool = True,
    ) -> None:
        """
        Initialize adaptive efficiency manager.

        Parameters
        ----------
        calculation : Calculation
            GROMACS calculation object to manage
        runtime_constant : float, default 0.001
            Runtime constant (kcal²·mol⁻²·ns⁻¹) for efficiency optimization
        relative_simulation_cost : float, default 1.0
            Relative computational cost of simulations
        max_runtime_per_window : float, default 30.0
            Maximum runtime per simulation window (ns)
        cycle_pause : int, default 60
            Time to wait between status checks (seconds)
        """
        self.calculation = calculation
        self.runtime_constant = runtime_constant
        self.relative_simulation_cost = relative_simulation_cost
        self.max_runtime_per_window = max_runtime_per_window
        self.cycle_pause = cycle_pause
        self.use_hpc = use_hpc

        # Initialize gradient analyzer
        self.gradient_analyzer = GradientAnalyzer()

        # State tracking
        self._maximally_efficient = False
        self.kill_thread = False

        logger.info("Initialized adaptive efficiency manager with:")
        logger.info(f"  Runtime constant: {runtime_constant}")
        logger.info(f"  Max runtime per window: {max_runtime_per_window} ns")

    def run_adaptive_efficiency_loop(self, run_nos: Optional[list[int]] = None) -> None:
        """
        Run the main adaptive efficiency loop.

        This method implements the core algorithm from A3FE that:
        1. Waits for all windows to finish current simulations
        2. Analyzes gradient data to get time-normalized SEMs
        3. Calculates optimal runtime for each window
        4. Resubmits windows that haven't reached optimal efficiency
        5. Repeats until all windows reach maximum efficiency

        Parameters
        ----------
        run_nos : list[int], optional
            Run numbers to include in analysis
        """
        logger.info("Starting adaptive efficiency optimization loop...")

        # Reset efficiency flag
        self._maximally_efficient = False

        iteration = 0
        while not self._maximally_efficient and not self.kill_thread:
            iteration += 1
            logger.info(f"=== Adaptive Efficiency Iteration {iteration} ===")
            logger.info(
                "Maximum efficiency not achieved. Allocating simulation time..."
            )

            # Wait for all windows to finish current simulations
            self._wait_for_all_windows()

            if self.kill_thread:
                logger.info("Kill signal received: exiting adaptive efficiency loop")
                return

            # Analyze all legs for efficiency
            all_legs_efficient = True

            for leg in self.calculation.legs:
                leg_efficient = self._process_leg_for_efficiency(leg, run_nos)
                if not leg_efficient:
                    all_legs_efficient = False

            # Check if we've reached maximum efficiency
            if all_legs_efficient:
                self._maximally_efficient = True
                logger.info(
                    "Maximum efficiency achieved with runtime constant "
                    f"{self.runtime_constant} kcal²·mol⁻²·ns⁻¹"
                )
            else:
                logger.info(
                    "Some windows still need optimization. "
                    "Will check again after simulations complete."
                )

    def _wait_for_all_windows(self) -> None:
        """Wait for all lambda windows to finish their current simulations."""
        logger.info("Waiting for all windows to complete current simulations...")

        while True:
            if self.kill_thread:
                return

            # Check if any windows are still running
            any_running = False

            for leg in self.calculation.legs:
                for window in leg.lambda_windows:
                    if window.running:
                        any_running = True
                        break
                if any_running:
                    break

            if not any_running:
                logger.info("All windows have completed their simulations")
                break

            # Update virtual queue and wait
            if self.calculation.virtual_queue:
                self.calculation.virtual_queue.update()

            logger.debug("Some windows still running, waiting...")
            time.sleep(self.cycle_pause)

    def _process_leg_for_efficiency(
        self, leg: Leg, run_nos: Optional[list[int]]
    ) -> bool:
        """
        Process a single leg for efficiency optimization.

        Returns True if all windows in the leg have reached optimal efficiency.
        """
        logger.info(f"Processing {leg.leg_type.name} leg for efficiency...")

        # Get time-normalized SEMs using the gradient analyzer
        try:
            smooth_dg_sems = self.gradient_analyzer.get_time_normalized_sems(
                leg.lambda_windows,
                run_nos=run_nos,
                origin="inter_delta_g",
                smoothen=True,
                equilibrated=False,
            )
        except Exception as e:
            logger.error(
                f"Failed to get gradient data for {leg.leg_type.name} leg: {e}"
            )
            return True  # Skip this leg

        all_windows_efficient = True

        for i, window in enumerate(leg.lambda_windows):
            try:
                normalized_sem_dg = smooth_dg_sems[i]
                window_efficient = self._process_window_for_efficiency(
                    window=window, normalized_sem_dg=normalized_sem_dg, run_nos=run_nos
                )
                if not window_efficient:
                    all_windows_efficient = False
            except Exception as e:
                logger.error(f"Failed to process window {window.lam_state}: {e}")
                continue

        return all_windows_efficient

    def _process_window_for_efficiency(
        self,
        window: LambdaWindow,
        normalized_sem_dg: float,
        run_nos: Optional[list[int]],
    ) -> bool:
        """
        Process a single window for efficiency optimization.

        Returns True if the window has reached optimal efficiency.
        """
        # Calculate predicted optimal runtime
        predicted_runtime = self._calculate_optimal_runtime(normalized_sem_dg)

        # Get actual runtime for this window
        actual_runtime = window.get_tot_simulation_time(run_nos or [1])

        logger.info(f"Window λ {window.lam_state}:")
        logger.info(f"  Predicted optimal runtime: {predicted_runtime:.3f} ns")
        logger.info(f"  Actual runtime: {actual_runtime:.3f} ns")

        # Apply maximum runtime constraint
        max_total_runtime = self.max_runtime_per_window * window.ensemble_size
        if predicted_runtime > max_total_runtime:
            logger.info(
                f"  Predicted runtime ({predicted_runtime:.3f} ns) exceeds "
                f"maximum ({max_total_runtime:.3f} ns). Using maximum."
            )
            predicted_runtime = max_total_runtime

        # Check if we need more simulation time
        if actual_runtime < predicted_runtime:
            resubmit_time = (predicted_runtime - actual_runtime) / window.ensemble_size

            # Limit resubmission to at most the current total simulation time
            max_resubmit = actual_runtime / window.ensemble_size
            if resubmit_time > max_resubmit:
                resubmit_time = max_resubmit
                logger.info(
                    f"  Limiting resubmission time to {resubmit_time:.3f} ns "
                    f"(current total per replica)"
                )

            # Round up to nearest 0.1 ns
            resubmit_time = np.ceil(resubmit_time * 10) / 10

            if resubmit_time > 0:
                logger.info(
                    f"  Window has not reached maximum efficiency. "
                    f"Resubmitting for {resubmit_time:.3f} ns"
                )

                # Resubmit the window
                try:
                    window.run(
                        run_nos=run_nos, runtime=resubmit_time, use_hpc=self.use_hpc
                    )
                    return False  # Window is not yet efficient
                except Exception as e:
                    logger.error(f"Failed to resubmit window {window.lam_state}: {e}")
                    return True  # Assume efficient to avoid infinite loop
            else:
                logger.info(
                    "  Resubmission time too small (<0.1 ns), considering efficient"
                )
                return True
        else:
            logger.info(
                "  Window has reached optimal runtime. No further simulation required"
            )
            return True

    def _calculate_optimal_runtime(self, normalized_sem_dg: float) -> float:
        """
        Calculate the optimal runtime for maximum efficiency.

        Based on the A3FE formula:
        optimal_runtime = (normalized_sem) / sqrt(runtime_constant * relative_cost)
        """
        if normalized_sem_dg <= 0:
            logger.warning(
                f"Non-positive normalized SEM: {normalized_sem_dg}, returning 0 runtime"
            )
            return 0.0

        optimal_runtime = normalized_sem_dg / np.sqrt(
            self.runtime_constant * self.relative_simulation_cost
        )
        # Ensure non-negative result (protect against numerical errors)
        return max(0.0, optimal_runtime)

    def set_runtime_constant(self, new_constant: float) -> None:
        """
        Update the runtime constant and reset efficiency status.

        This is useful for adaptive equilibration workflows where the
        runtime constant may be adjusted during the process.
        """
        logger.info(
            f"Updating runtime constant: {self.runtime_constant} → {new_constant}"
        )
        self.runtime_constant = new_constant
        self._maximally_efficient = False

    def stop(self) -> None:
        """Stop the adaptive efficiency loop."""
        logger.info("Stopping adaptive efficiency optimization...")
        self.kill_thread = True

    @property
    def is_maximally_efficient(self) -> bool:
        """Check if the calculation has reached maximum efficiency."""
        return self._maximally_efficient
