import logging

from binding_affinity_predicting.components.data.enums import LegType
from binding_affinity_predicting.components.data.schemas import (
    GromacsFepSimulationConfig,
)
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    Calculation,
)
from binding_affinity_predicting.components.simulation_fep.lambda_optimizer import (
    LambdaOptimizationManager,
    OptimizationConfig,
    StageConfig,
)

logger = logging.getLogger(__name__)


custom_optimizing_config = OptimizationConfig(
    restrained=StageConfig(target_error=3.0),
    discharging=StageConfig(target_error=3.0),
    vanishing=StageConfig(target_error=3.0),
)

# fmt: off
bound_bonded = [  # noqa: E127,E128,E131,E122,E501
    0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.35, 0.5, 0.75,  # noqa: E127,E128,E131,E122,E501
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,          # noqa: E127,E128,E131,E122,E501
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,          # noqa: E127,E128,E131,E122,E501
]

bound_coul = [  # noqa: E127,E128,E131,E122,E501
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # noqa: E127,E128,E131,E122,E501
    0.25, 0.5, 0.75, 1.0,                                    # noqa: E127,E128,E131,E122,E501
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,        # noqa: E127,E128,E131,E122,E501
    1.0, 1.0, 1.0, 1.0, 1.0,                                 # noqa: E127,E128,E131,E122,E501
]

bound_vdw = [             # noqa: E127,E128,E131,E122,E501
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      # noqa: E127,E128,E131,E122,E501
    0.0, 0.0, 0.0, 0.0, 0.0,                               # noqa: E127,E128,E131,E122,E501
    0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75,   # noqa: E127,E128,E131,E122,E501
    0.8, 0.85, 0.9, 0.95, 1.0,                             # noqa: E127,E128,E131,E122,E501
]
# fmt: on

free_coul = [0.0, 0.50]
free_vdw = [0.0, 0.50]

# Build dictionaries
custom_bonded = {LegType.BOUND: bound_bonded}
custom_coul = {LegType.BOUND: bound_coul, LegType.FREE: free_coul}
custom_vdw = {LegType.BOUND: bound_vdw, LegType.FREE: free_vdw}


custom_sim_config = GromacsFepSimulationConfig(
    bonded_lambdas=custom_bonded,
    coul_lambdas=custom_coul,
    vdw_lambdas=custom_vdw,
)


gradient_collection_runtime = 0.002  # ns
prod_runtime = 0.005  # ns

calc = Calculation(
    input_dir="/Users/jingjinghuang/Documents/fep_workflow/test_classes/input",
    output_dir="/Users/jingjinghuang/Documents/fep_workflow/test_classes/output",
    ensemble_size=2,
    sim_config=custom_sim_config,
)


calc.setup()

logger.info(f"✅ Calculation set up with {len(calc.legs)} legs")
for leg in calc.legs:
    logger.info(f"   {leg.leg_type.name}: {len(leg.lambda_windows)} lambda windows")


calc.run(
    runtime=gradient_collection_runtime,
    use_hpc=False,
    run_sync=True,
)
manager = LambdaOptimizationManager(config=custom_optimizing_config)

optimization_results = manager.optimize_calculation(
    calculation=calc,
    equilibrated=False,
    apply_results=True,
)

# Now run production with optimized lambdas
for leg in calc.legs:
    logger.info(
        f"   {leg.leg_type.name}: {len(leg.lambda_windows)} lambda windows (optimized)"
    )

calc.run(
    runtime=prod_runtime,
    use_hpc=False,
    run_sync=True,
)
