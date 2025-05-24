import logging
import os
from typing import Optional

import BioSimSpace.Sandpit.Exscientia as BSS  # type: ignore[import]

from binding_affinity_predicting.components.utils import save_workflow_config
from binding_affinity_predicting.data.schemas import (
    BaseWorkflowConfig,
    EmpiricalPreEquilibrationConfig,
    EnergyMinimisationConfig,
    EnsembleEquilibrationConfig,
    EnsembleEquilibrationReplicaConfig,
    PreEquilStageConfig,
    SimulationRestraint,
)
from binding_affinity_predicting.workflows.free_energy_calc.system_prep_workflow import (
    prepare_ensemble_equilibration_replicas,
    prepare_preequil_molecular_system,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# system = prepare_preequil_molecular_system(
#     config=cfg,
#     output_nametag="bound",
#     protein_path=os.path.join(input_dir, "protein.pdb"),
#     ligand_path=os.path.join(input_dir, "ligand.sdf"),
#     output_dir=output_dir,
#     use_slurm=cfg.slurm,
# )


def run_complete_system_preparation(
    *,
    output_nametag: str,
    config: BaseWorkflowConfig = BaseWorkflowConfig(),
    protein_path: Optional[str] = None,
    ligand_path: Optional[str] = None,
    water_path: Optional[str] = None,
    output_dir: str,
    use_slurm: bool = True,
) -> BSS._SireWrappers._system.System:

    system_preequil = prepare_preequil_molecular_system(
        filename_stem=output_nametag,
        config=config,
        protein_path=protein_path,
        ligand_path=ligand_path,
        water_path=water_path,
        output_dir=output_dir,
        use_slurm=use_slurm,
    )

    prepare_ensemble_equilibration_replicas(
        source=system_preequil,
        config=config,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    # Save the workflow config to a file
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
    custom_rep = EnsembleEquilibrationReplicaConfig(runtime=0.002)  # ns
    custom_replicas = [custom_rep, custom_rep]

    custom_ensemble_equil = EnsembleEquilibrationConfig(
        num_replicas=2, replicas=custom_replicas
    )

    custom_min = EnergyMinimisationConfig(steps=500)

    cfg = BaseWorkflowConfig(
        slurm=False,  # run everything locally
        param_preequilibration=custom_preequil,
        param_energy_minimisation=custom_min,
        param_ensemble_equilibration=custom_ensemble_equil,
        # to use "-ntmpi 1 -ntomp 8" we need to compile GROMACS with OpenMP support
        mdrun_options="-ntmpi 1 -ntomp 1",
    )

    run_complete_system_preparation(
        output_nametag="bound",
        config=cfg,
        protein_path="/Users/jingjinghuang/Documents/fep_workflow/test_my_repo/input/protein.pdb",
        ligand_path="/Users/jingjinghuang/Documents/fep_workflow/test_my_repo/input/ligand.sdf",
        output_dir="/Users/jingjinghuang/Documents/fep_workflow/test_my_repo/input/equilibrations",
        use_slurm=False,
    )
    save_workflow_config(
        cfg,
        os.path.join(
            "/Users/jingjinghuang/Documents/fep_workflow/debug/inputs/tmp_output",
            "workflow_config.pkl",
        ),
    )
