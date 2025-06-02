import logging
import os

from binding_affinity_predicting.components.utils import save_workflow_config
from binding_affinity_predicting.data.schemas import (
    BaseWorkflowConfig,
    EmpiricalPreEquilibrationConfig,
    EnergyMinimisationConfig,
    PreEquilStageConfig,
    SimulationRestraint,
)
from binding_affinity_predicting.workflows.free_energy_calc.system_prep_workflow import (
    _prepare_equilibrated_molecular_systems,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

custom_steps = [
    PreEquilStageConfig(
        runtime=2.0,
        temperature_start=298.15,
        temperature_end=298.15,
        restraint=SimulationRestraint.ALL,
        pressure=None,
    ),
    PreEquilStageConfig(
        runtime=2.0,
        temperature_start=298.15,
        temperature_end=298.15,
        restraint=SimulationRestraint.BACKBONE,
        pressure=None,
    ),
]

custom_preequil = EmpiricalPreEquilibrationConfig(steps=custom_steps)

custom_min = EnergyMinimisationConfig(steps=500)

cfg = BaseWorkflowConfig(
    slurm=False,  # run everything locally
    param_preequilibration=custom_preequil,
    param_energy_minimisation=custom_min,
    # to use "-ntmpi 1 -ntomp 8" we need to compile GROMACS with OpenMP support
    mdrun_options="-ntmpi 1 -ntomp 1",
)

input_dir = "/Users/jingjinghuang/Documents/fep_workflow/debug/inputs/in_files"
output_dir = "/Users/jingjinghuang/Documents/fep_workflow/debug/inputs/tmp_output"
os.makedirs(output_dir, exist_ok=True)

system = _prepare_equilibrated_molecular_systems(
    config=cfg,
    filename_stem="bound",
    protein_path=os.path.join(input_dir, "protein.pdb"),
    ligand_path=os.path.join(input_dir, "ligand.sdf"),
    output_dir=output_dir,
    use_slurm=cfg.slurm,
)

# Save the workflow config to a file
save_workflow_config(
    cfg,
    os.path.join(output_dir, "workflow_config.pkl"),
)

logger.info("Done all steps successfully.")
