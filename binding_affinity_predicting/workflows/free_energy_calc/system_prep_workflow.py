import logging
import os
from typing import Optional

import BioSimSpace.Sandpit.Exscientia as BSS  # type: ignore[import]

from binding_affinity_predicting.data.schemas import WorkflowConfig
from binding_affinity_predicting.hpc_cluster.slurm import run_slurm
from binding_affinity_predicting.simulation.parameterise import parameterise_system
from binding_affinity_predicting.simulation.preequilibraiton import (
    energy_minimise_system,
    preequilibrate_system,
)
from binding_affinity_predicting.simulation.system_preparation import solvate_system

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def prepare_preequil_molecular_system(
    *,
    output_nametag: str,
    config: WorkflowConfig = WorkflowConfig(),
    protein_path: Optional[str] = None,
    ligand_path: Optional[str] = None,
    water_path: Optional[str] = None,
    output_dir: str,
    use_slurm: bool = True,
) -> BSS._SireWrappers._system.System:
    """
    Prepare the system for free energy calculations.

    1) Parameterise
    2) Solvate
    3) Minimise
    4) Pre-equilibrate

    Parameters
    ----------
    config : WorkflowConfig
        Configuration object containing the parameters for the workflow.
    output_nametag : str
        The name of the output file (without extension).
        e.g. "complex" will create "complex.gro".
    protein_path : str, optional
        Path to the protein structure file (PDB/GRO/etc).
    ligand_path : str, optional
        Path to the ligand structure file (PDB/GRO/etc).
    water_path : str, optional
        Path to the water structure file (PDB/GRO/etc).
    output_dir : str
        Directory where the output files will be saved.
    use_slurm : bool, default True
        If True, run the workflow using SLURM. If False, run locally.
    """
    # ensure base output exists
    os.makedirs(output_dir, exist_ok=True)

    # ── 1) PARAMETERISE ──────────────────────────────────────────────────
    logger.info("Step 1: Parameterise the system...")
    system_parameterised = parameterise_system(
        protein_path=protein_path,
        ligand_path=ligand_path,
        water_path=water_path,
        protein_ff=config.param_system_prep.forcefields["protein"],
        ligand_ff=config.param_system_prep.forcefields["ligand"],
        water_ff=config.param_system_prep.forcefields["water"],
        water_model=config.param_system_prep.water_model,
        output_file_path=os.path.join(output_dir, f"{output_nametag}.gro"),
    )

    # ── 2) SOLVATE ──────────────────────────────────────────────────────
    logger.info("Step 2: Solvate the system...")
    system_solvated = solvate_system(
        source=system_parameterised,
        water_model=config.param_system_prep.water_model,
        ion_conc=config.param_system_prep.ion_conc,
        output_file_path=os.path.join(output_dir, f"{output_nametag}_solvated.gro"),
    )

    # ── 3) ENERGY MINIMISE ─────────────────────────────────────────────────────
    logger.info("Step 3: Energy minimise the system...")
    energy_min_out = os.path.join(output_dir, f"{output_nametag}_energy_min.gro")
    min_kwargs = dict(
        source=system_solvated,
        output_file_path=energy_min_out,
        min_steps=config.param_energy_minimisation.steps,
        mdrun_options=config.mdrun_options,
        process_name="minimise_system",
    )
    if use_slurm:
        run_slurm(
            sys_prep_fn=energy_minimise_system,
            wait=True,
            run_dir=output_dir,
            job_name="minimise_system",
            **min_kwargs,
        )
        # once the SLURM job finishes, reload from file
        system_energy_min = BSS.IO.readMolecules(energy_min_out)
    else:
        system_energy_min = energy_minimise_system(**min_kwargs)

    # ── 4) PRE-EQUILIBRATE ───────────────────────────────────────────────
    logger.info("Step 4: Pre-equilibrate the system...")
    preequil_out = os.path.join(output_dir, f"{output_nametag}_preequiled.gro")
    preequil_kwargs = dict(
        source=system_energy_min,
        steps=config.param_preequilibration.steps,
        work_dir=output_dir,
        mdrun_options=config.mdrun_options,
        output_file_path=preequil_out,
        process_name="preequil_system",
    )

    if use_slurm:
        run_slurm(
            sys_prep_fn=preequilibrate_system,
            wait=True,
            run_dir=output_dir,
            job_name="preequil_system",
            **preequil_kwargs,
        )
        system_preequil = BSS.IO.readMolecules(preequil_out)
    else:
        system_preequil = preequilibrate_system(**preequil_kwargs)

    logger.info("All steps complete!")
    return system_preequil


def prepare_fep_simulation_systems():
    pass
