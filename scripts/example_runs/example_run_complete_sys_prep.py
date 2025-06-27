import logging

from binding_affinity_predicting.components.data.schemas import (
    BaseWorkflowConfig,
    EmpiricalPreEquilibrationConfig,
    EnergyMinimisationConfig,
    EnsembleEquilibrationConfig,
    EnsembleEquilibrationReplicaConfig,
    PreEquilStageConfig,
    SimulationRestraint,
)
from binding_affinity_predicting.workflows.free_energy_calc.system_prep_workflow import (
    run_complete_system_setup_bound_and_free,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


custom_steps = [
    PreEquilStageConfig(
        runtime=1.0,  # ps
        temperature_start=298.15,
        temperature_end=298.15,
        restraint=SimulationRestraint.ALL,
        pressure=None,
    ),
    PreEquilStageConfig(
        runtime=1.0,  # ps
        temperature_start=298.15,
        temperature_end=298.15,
        restraint=SimulationRestraint.BACKBONE,
        pressure=None,
    ),
]

custom_rep = EnsembleEquilibrationReplicaConfig(runtime=0.001)  # ns
custom_replicas = [custom_rep, custom_rep]

custom_ensemble_equil = EnsembleEquilibrationConfig(
    num_replicas=2, replicas=custom_replicas
)

custom_preequil = EmpiricalPreEquilibrationConfig(steps=custom_steps)

custom_min = EnergyMinimisationConfig(steps=100)

cfg = BaseWorkflowConfig(
    slurm=False,  # run everything locally
    param_preequilibration=custom_preequil,
    param_energy_minimisation=custom_min,
    param_ensemble_equilibration=custom_ensemble_equil,
    # to use "-ntmpi 1 -ntomp 8" we need to compile GROMACS with OpenMP support
    mdrun_options="-ntmpi 1 -ntomp 1",
)


system_list = run_complete_system_setup_bound_and_free(
    protein_path="/Users/jingjinghuang/Documents/fep_workflow/test_classes/input/protein.pdb",
    ligand_path="/Users/jingjinghuang/Documents/fep_workflow/test_classes/input/ligand.sdf",
    output_dir="/Users/jingjinghuang/Documents/fep_workflow/test_classes/input/equilibration",
    filename_stem="bound",
    config=cfg,
    use_slurm=False,
)
