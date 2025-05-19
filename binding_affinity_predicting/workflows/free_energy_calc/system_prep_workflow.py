import logging
from pathlib import Path
from typing import Callable, Union

from binding_affinity_predicting.components import system_prep
from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.hpc_cluster.slurm import run_slurm
from binding_affinity_predicting.schemas.enums import LegType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def prepare_fep_simulation_system(
    leg_type: LegType,
    run_dir: str,
    use_slurm: bool = True,
):
    # 1. Parameterise
    logger.info("Step 1: Parameterise")
    system_prep.parameterise_input(
        leg_type=leg_type,
        input_dir=run_dir,
        output_dir=run_dir,
    )

    # 2. Solvate
    logger.info("Step 2: Solvate")
    system_prep.solvate_input(
        leg_type=leg_type,
        input_dir=run_dir,  # solvate_input uses the output of parameterise_input
        output_dir=run_dir,
    )

    # 3. Minimise
    logger.info("Step 3: Minimise")
    if use_slurm:
        run_slurm(
            sys_prep_fn=system_prep.minimise_input,
            wait=True,
            run_dir=run_dir,
            job_name=f"minimise_{leg_type.name.lower()}",
            leg_type=leg_type,
            input_dir=run_dir,
            output_dir=run_dir,
            slurm=False,  # Important: the called function shouldn't spawn another SLURM job
        )
    else:
        system_prep.minimise_input(
            leg_type=leg_type, input_dir=run_dir, output_dir=run_dir, slurm=False
        )

    # 4. Heat & Pre-equil
    logger.info("Step 4: Heat & Pre-equil")
    if use_slurm:
        run_slurm(
            sys_prep_fn=system_prep.heat_and_preequil_input,
            wait=True,
            run_dir=run_dir,
            job_name=f"heat_and_preequil_{leg_type.name.lower()}",
            leg_type=leg_type,
            input_dir=run_dir,
            output_dir=run_dir,
            slurm=False,  # Don't nest SLURM submissions
        )
    else:
        system_prep.heat_and_preequil_input(
            leg_type=leg_type, input_dir=run_dir, output_dir=run_dir, slurm=False
        )

    logger.info("All steps complete!")
