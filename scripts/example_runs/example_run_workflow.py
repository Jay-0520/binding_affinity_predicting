from binding_affinity_predicting.components.data.enums import LegType
from binding_affinity_predicting.components.data.schemas import (
    GromacsFepSimulationConfig,
)
from binding_affinity_predicting.workflows.free_energy_calc.adaptive_fep_simulation_workflow import (  # noqa: E501
    run_adaptive_fep_workflow,
)

bound_bonded = [0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
bound_coul = [0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
bound_vdw = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0]

free_coul = [0.0, 0.50]
free_vdw = [0.0, 0.50]

custom_bonded = {LegType.BOUND: bound_bonded}
custom_coul = {
    LegType.BOUND: bound_coul,
    LegType.FREE: free_coul,
}
custom_vdw = {
    LegType.BOUND: bound_vdw,
    LegType.FREE: free_vdw,
}

custom_sim_config = GromacsFepSimulationConfig(
    bonded_lambdas=custom_bonded,
    coul_lambdas=custom_coul,
    vdw_lambdas=custom_vdw,
    mdp_overrides={  # we can override default MDP parameters here
        "nsteps": 5_000,
        "couple_intramol": "yes",  # this is only for make gmx happy, but this setting is wrong for FEP  # noqa: E501
    },
    # mdrun_options="-ntmpi 1 -ntomp 1"
)

input_dir = "/Users/jingjinghuang/Documents/fep_workflow/test_classes3/input"
output_dir = "/Users/jingjinghuang/Documents/fep_workflow/test_classes3/output"

results = run_adaptive_fep_workflow(
    input_dir=input_dir,
    output_dir=output_dir,
    ensemble_size=1,
    sim_config=custom_sim_config,
    use_hpc=False,
    run_sync=False,  # so that we can monitor the progress
    short_run_runtime=1,  # 2 ns for short simulations
    monitor_interval=5,  # time in secs between status checks
    enable_monitoring=True,
    initial_runtime_constant=0.1,
    optimize_lambda_spacing=False,
    # force_rerun_phases=["short_simulations"],
)
