from binding_affinity_predicting.components.gromacs_orchestration import Calculation
from binding_affinity_predicting.components.status_monitor import StatusMonitor
from binding_affinity_predicting.data.enums import LegType
from binding_affinity_predicting.data.schemas import GromacsFepSimulationConfig

bound_bonded = [0.0, 0.25]
bound_coul = [0.0, 0.00]
bound_vdw = [0.0, 0.10]

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
)

calc = Calculation(
    input_dir="/Users/jingjinghuang/Documents/fep_workflow/test_classes/input",
    output_dir="/Users/jingjinghuang/Documents/fep_workflow/test_classes/output",
    ensemble_size=2,
    sim_config=custom_sim_config,
)

calc.setup()
calc.run(use_hpc=False)

# monitor and check calculation progress
monitor = StatusMonitor(calc)
print(monitor.get_summary())
# exammple ouput would be:
# "2/4 finished, 0 failed, 1 running 1 queue. No failed jobs."
