import time

from binding_affinity_predicting.components.analysis.simulation_analyzer import (
    analyze_gromacs_calculation,
)
from binding_affinity_predicting.components.data.enums import LegType
from binding_affinity_predicting.components.data.schemas import (
    GromacsFepSimulationConfig,
)
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    Calculation,
)
from binding_affinity_predicting.components.simulation_fep.status_monitor import (
    StatusMonitor,
)

bound_bonded = [0.0, 0.5, 1.0, 1.0, 1.0]
bound_coul = [0.0, 0.0, 0.5, 1.0, 1.0]
bound_vdw = [0.0, 0.0, 0.0, 0.5, 1.0]

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
        "nsteps": 1_000,
    },
)

calc = Calculation(
    input_dir="/Users/jingjinghuang/Documents/fep_workflow/test_classes2/input",
    output_dir="/Users/jingjinghuang/Documents/fep_workflow/test_classes2/output",
    ensemble_size=2,
    sim_config=custom_sim_config,
)

test_runtime = 0.002  # ns

calc.setup()
calc.run(
    runtime=test_runtime,  # we can override MDP runtime parameters here
    use_hpc=False,
    run_sync=False,
)

# monitor and check calculation progress
monitor = StatusMonitor(calc)

# poll until nothing is left in “running” or “queued”
while True:
    summary = monitor.get_summary()
    print(summary)
    # exammple ouput would be:
    # "2/4 finished, 0 failed, 1 running 1 queue. No failed jobs."

    # if there are no more jobs running or queued, break out
    parts = summary.split(',')
    finished, failed = parts[0].strip(), parts[1].strip()
    running = int(parts[2].split()[0])
    queued = int(parts[3].split()[0])
    if running == 0 and queued == 0:
        break

    time.sleep(30)  # wait 30 s before checking again


# run analysis on the first leg of the calculation
# results = analyze_gromacs_leg(
#     leg=calc.legs[0])
results = analyze_gromacs_calculation(
    calculation=calc,
)
print("All done!")
