import glob
import logging
import os
from typing import Optional, Union

import BioSimSpace.Sandpit.Exscientia as BSS  # type: ignore[import]

from binding_affinity_predicting.data.schemas import BaseWorkflowConfig
from binding_affinity_predicting.hpc_cluster.slurm import run_slurm
from binding_affinity_predicting.simulation.parameterise import parameterise_system
from binding_affinity_predicting.simulation.preequilibration import (
    energy_minimise_system,
    preequilibrate_system,
    run_ensemble_equilibration,
)
from binding_affinity_predicting.simulation.system_preparation import (
    extract_restraint_from_traj,
    solvate_system,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def prepare_preequil_molecular_system(
    *,
    filename_stem: str,
    config: BaseWorkflowConfig = BaseWorkflowConfig(),
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
    filename_stem : str
        NOTE: The name of the output file (without extension).
        e.g. "bound" will create "bound.gro" (structure) and "bound.top" (topology).
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
        filename_stem=filename_stem,
        output_dir=output_dir,
    )

    # ── 2) SOLVATE ──────────────────────────────────────────────────────
    logger.info("Step 2: Solvate the system...")
    system_solvated = solvate_system(
        source=system_parameterised,
        water_model=config.param_system_prep.water_model,
        ion_conc=config.param_system_prep.ion_conc,
        filename_stem=f"{filename_stem}_solvated",
        output_dir=output_dir,
    )

    # ── 3) ENERGY MINIMISE ─────────────────────────────────────────────────────
    logger.info("Step 3: Energy minimise the system...")
    min_kwargs = dict(
        source=system_solvated,
        filename_stem=f"{filename_stem}_minimised",
        output_dir=output_dir,
        min_steps=config.param_energy_minimisation.steps,
        mdrun_options=config.mdrun_options,
    )
    if use_slurm:
        run_slurm(
            sys_prep_fn=energy_minimise_system,
            wait=True,
            run_dir=output_dir,
            job_name="minimise_system",
            **min_kwargs,
        )
        # once the SLURM job finishes, reload from file; need to include file extension
        system_energy_min = BSS.IO.readMolecules(
            os.path.join(output_dir, f"{filename_stem}_minimised.gro")
        )
    else:
        system_energy_min = energy_minimise_system(**min_kwargs)

    # ── 4) PRE-EQUILIBRATE ───────────────────────────────────────────────
    logger.info("Step 4: Pre-equilibrate the system...")
    # preequil_out = os.path.join(output_dir, f"{output_nametag}_preequiled.gro")
    preequil_kwargs = dict(
        source=system_energy_min,
        steps=config.param_preequilibration.steps,
        mdrun_options=config.mdrun_options,
        filename_stem=f"{filename_stem}_preequiled",
        output_dir=output_dir,
    )

    if use_slurm:
        run_slurm(
            sys_prep_fn=preequilibrate_system,
            wait=True,
            run_dir=output_dir,
            job_name="preequil_system",
            **preequil_kwargs,
        )
        system_preequil = BSS.IO.readMolecules(
            os.path.join(output_dir, f"{filename_stem}_preequiled.gro")
        )
    else:
        system_preequil = preequilibrate_system(**preequil_kwargs)

    logger.info("All steps complete!")
    return system_preequil


def prepare_ensemble_equilibration_replicas(
    *,
    source: Union[str, BSS._SireWrappers._system.System],
    filename_stem: str = "ensemble_equilibration",
    output_dir: str,
    config: BaseWorkflowConfig = BaseWorkflowConfig(),
    **extra_protocol_kwargs,
) -> tuple[
    list[BSS._SireWrappers._system.System], list[BSS.FreeEnergy._restraint.Restraint]
]:
    """
    Run ensemble equilibration for the given system and
    extract restraints from trajectory for each replica.

    Trajectory is loaded using MDAnalysis by default

    Parameters
    ----------
    source : str or BSS._SireWrappers._system.System
        The source system to equilibrate. If a string, it is treated as a path to the system file.
    output_dir : str
        Directory where the output files will be saved.
    filename_stem : str
        The stem of the filename to use for the output files.
    config : BaseWorkflowConfig
        Configuration object containing the parameters for the workflow.
    extra_protocol_kwargs : dict
        Additional keyword arguments to pass to the equilibration function.
    """
    # ensure the base output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Run the actual BSS equilibration
    replicas = config.param_ensemble_equilibration.replicas
    logger.info(f"Running ensemble equilibration with {len(replicas)} replicas...")

    system_list = run_ensemble_equilibration(
        source=source,
        replicas=replicas,
        filename_stem=filename_stem,
        output_dir=output_dir,
        mdrun_options=config.mdrun_options,
        process_name="ensemble_equilibration",
        **extra_protocol_kwargs,
    )

    # 2. Now loop over each replicate and extract its restraint
    logger.info("Extracting restraints from each replica...")
    restraint_list = []
    for idx in range(1, len(system_list) + 1):
        rep_dir = os.path.join(output_dir, f"ensemble_equilibration_{idx}")

        # grab the exactly one .xtc
        trajs = glob.glob(os.path.join(rep_dir, "*.xtc"))
        if len(trajs) != 1:
            raise RuntimeError(
                f"Expected 1 trajectory in {rep_dir}, but found {len(trajs)}."
            )
        trajectory_file = trajs[0]

        # grab the exactly one .tpr file.
        # NOTE tpr is forced to be the top file in BSS for restraint extraction
        tops = glob.glob(os.path.join(rep_dir, "*.tpr"))
        if len(tops) != 1:
            raise RuntimeError(
                f"Expected 1 topology in {rep_dir}, but found {len(tops)}."
            )
        topology_file = tops[0]

        # use the temperature from the previous replica run
        temperature_equil = config.param_ensemble_equilibration.replicas[
            idx - 1
        ].temperature
        restraint = extract_restraint_from_traj(
            work_dir=rep_dir,
            trajectory_file=trajectory_file,
            topology_file=topology_file,
            system=system_list[idx - 1],
            output_filename=f"restraint_{idx}.itp",  # GROMACS format for restraints
            temperature=temperature_equil,
            append_to_ligand_selection=config.append_to_ligand_selection,
        )
        restraint_list.append(restraint)

    return system_list, restraint_list


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

    prepare_preequil_molecular_system(
        filename_stem=output_nametag,
        config=config,
        protein_path=protein_path,
        ligand_path=ligand_path,
        water_path=water_path,
        output_dir=output_dir,
        use_slurm=use_slurm,
    )
    prepare_ensemble_equilibration_replicas(
        source=output_nametag,
        output_basename=os.path.join(output_dir, f"{output_nametag}_ensemble"),
        config=config,
        filename_stem=output_nametag,
        output_dir=output_dir,
    )


def move_and_create_files():
    """
    Move files to the correct location and create any necessary files.
    """
    pass


def prepare_fep_simulation_systems():
    pass
