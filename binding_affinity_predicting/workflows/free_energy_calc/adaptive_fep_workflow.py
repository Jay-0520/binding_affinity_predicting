import logging
from typing import Optional

from binding_affinity_predicting.components.data.schemas import (
    GromacsFepSimulationConfig,
)
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    Calculation,
)
from binding_affinity_predicting.components.simulation_fep.lambda_optimizer import (
    LambdaOptimizationManager,
    OptimizationConfig,
    OptimizationResult,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: need to consolidate OptimizationConfig with GromacsFepSimulationConfig
def run_calculation_with_lambda_optimization(
    custom_sim_config: GromacsFepSimulationConfig,
    input_dir: str,
    output_dir: str,
    ensemble_size: int = 2,
    run_nos: Optional[list[int]] = None,
    equilibrated: bool = False,
    apply_optimization: bool = True,
    gradient_collection_runtime: float = 1.0,
    optimized_runtime: float = 5.0,
    use_hpc: bool = False,
    run_sync: bool = False,
) -> tuple[Calculation, dict[str, OptimizationResult]]:
    """
    Workflow with proper runtime management for optimization
    """

    logger.info("=" * 60)
    logger.info("CALCULATION WITH OPTIMIZATION - RUNTIME MANAGEMENT")
    logger.info("=" * 60)

    calc = Calculation(
        input_dir=input_dir,
        output_dir=output_dir,
        ensemble_size=ensemble_size,
        sim_config=custom_sim_config,
    )
    calc.setup()
    logger.info(f"✅ Calculation set up with {len(calc.legs)} legs")

    logger.info(
        f"Step 2: Running SHORT simulation for gradient collection ({gradient_collection_runtime} ns)..."  # noqa: E501
    )
    calc.run(
        runtime=gradient_collection_runtime,
        use_hpc=use_hpc,
        run_sync=run_sync,
    )

    if use_hpc:
        logger.info("Waiting for gradient collection to complete...")
        _wait_for_calculation_completion(calc)

    logger.info("✅ Gradient collection completed")
    logger.info("Step 3: Optimizing lambda spacing...")

    manager = LambdaOptimizationManager(config=OptimizationConfig())

    if run_nos is None:
        run_nos = list(range(1, ensemble_size + 1))

    optimization_results = manager.optimize_calculation(
        calculation=calc,
        run_nos=run_nos,
        equilibrated=equilibrated,
        apply_results=apply_optimization,
    )

    if apply_optimization and any(
        result.success for result in optimization_results.values()
    ):
        logger.info(
            f"Step 4: Running LONGER simulation with optimized spacing ({optimized_runtime} ns)..."
        )

        for leg in calc.legs:
            logger.info(
                f"   {leg.leg_type.name}: {len(leg.lambda_windows)} lambda windows (optimized)"
            )

        calc.run(
            runtime=optimized_runtime,
            use_hpc=use_hpc,
            run_sync=run_sync,
        )

        if use_hpc:
            logger.info("Waiting for optimized simulation to complete...")
            _wait_for_calculation_completion(calc)

        logger.info("✅ Optimized simulation completed")
    else:
        logger.info(
            "Step 4: Skipping optimized simulation (no successful optimization)"
        )

    logger.info("=" * 60)
    logger.info("WORKFLOW COMPLETED")
    logger.info(f"  Gradient collection: {gradient_collection_runtime} ns")
    logger.info(f"  Optimized production: {optimized_runtime} ns")
    logger.info("=" * 60)

    return calc, optimization_results


def _wait_for_calculation_completion(
    calc: Calculation, check_interval: int = 30
) -> None:
    """
    Wait for HPC calculation to complete using existing SimulationRunner status methods
    """
    import time

    logger.info("Monitoring calculation progress...")

    while calc.running:
        logger.info("Calculation still running... checking again in 30s")
        time.sleep(check_interval)

    if calc.failed:
        logger.error("❌ Calculation completed with failures!")
        failed_sims = calc.failed_simulations
        logger.error(f"Failed simulations: {len(failed_sims)}")
        for failed_sim in failed_sims:
            logger.error(f"  - {failed_sim}")
    elif calc.finished:
        logger.info("✅ Calculation completed successfully")
    else:
        logger.warning("⚠️ Calculation stopped but status unclear")
