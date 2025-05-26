import glob
import logging
import os
from typing import Optional, Union

import BioSimSpace.Sandpit.Exscientia as BSS  # type: ignore[import]

from binding_affinity_predicting.data.enums import LegType, PreparationStage
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
from binding_affinity_predicting.simulation.utils import (
    load_system_from_source,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _prepare_equilibrated_molecular_systems(
    *,
    filename_stem: Union[str, LegType],
    config: BaseWorkflowConfig = BaseWorkflowConfig(),
    protein_path: Optional[str] = None,
    ligand_path: Optional[str] = None,
    water_path: Optional[str] = None,
    output_dir: str,
    use_slurm: bool = True,
) -> BSS._SireWrappers._system.System:
    """
    Prepare equilibrated systems for free energy calculations.

    1) Parameterise
    2) Solvate
    3) Minimise
    4) Pre-equilibrate
    5) Run ensemble equilibrations -> multiple repeated systems

    Parameters
    ----------
    config : WorkflowConfig
        Configuration object containing the parameters for the workflow.
    filename_stem : str or LegType
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
    if isinstance(filename_stem, LegType):
        filename_stem = filename_stem.name  # convert LegType to string

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
        filename_stem=f"{filename_stem}{PreparationStage.PARAMETERISED.file_suffix}",
        output_dir=output_dir,
    )

    # ── 2) SOLVATE ──────────────────────────────────────────────────────
    logger.info("Step 2: Solvate the system...")
    system_solvated = solvate_system(
        source=system_parameterised,
        water_model=config.param_system_prep.water_model,
        ion_conc=config.param_system_prep.ion_conc,
        filename_stem=f"{filename_stem}{PreparationStage.SOLVATED.file_suffix}",
        output_dir=output_dir,
    )

    # ── 3) ENERGY MINIMISE ─────────────────────────────────────────────────────
    logger.info("Step 3: Energy minimise the system...")
    min_kwargs = dict(
        source=system_solvated,
        filename_stem=f"{filename_stem}{PreparationStage.MINIMISED.file_suffix}",
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
            os.path.join(
                output_dir,
                f"{filename_stem}{PreparationStage.MINIMISED.file_suffix}.gro",
            )
        )
    else:
        system_energy_min = energy_minimise_system(**min_kwargs)

    # ── 4) PRE-EQUILIBRATE ───────────────────────────────────────────────
    logger.info("Step 4: Pre-equilibrate the system...")
    preequil_kwargs = dict(
        source=system_energy_min,
        steps=config.param_preequilibration.steps,
        filename_stem=f"{filename_stem}{PreparationStage.PREEQUILIBRATED.file_suffix}",
        output_dir=output_dir,
        mdrun_options=config.mdrun_options,
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
            os.path.join(
                output_dir,
                f"{filename_stem}{PreparationStage.PREEQUILIBRATED.file_suffix}.gro",
            )
        )
    else:
        system_preequil = preequilibrate_system(**preequil_kwargs)

    # ── 5) RUN ENSEMBLE EQUILIBRATION ───────────────────────────────────────
    logger.info("Step 5: Run ensemble equilibration...")
    equil_kwargs = dict(
        source=system_preequil,
        replicas=config.param_ensemble_equilibration.replicas,
        filename_stem=f"{filename_stem}{PreparationStage.EQUILIBRATED.file_suffix}",
        output_dir=output_dir,
        mdrun_options=config.mdrun_options,
    )
    if use_slurm:
        run_slurm(
            sys_prep_fn=run_ensemble_equilibration,
            wait=True,
            run_dir=output_dir,
            job_name="ensemble_equilibration",
            **equil_kwargs,
        )
        system_equil_list = [
            BSS.IO.readMolecules(
                os.path.join(
                    output_dir,
                    f"{filename_stem}{PreparationStage.EQUILIBRATED.file_suffix}_{ndx}.gro",
                )
            )
            for ndx in range(1, config.param_ensemble_equilibration.num_replicas + 1)
        ]
    else:
        system_equil_list = run_ensemble_equilibration(**equil_kwargs)

    logger.info("Great! All steps complete!")
    return system_equil_list


def _prepare_restraints_from_ensemble_equilibration(
    *,
    source_list: Union[list[str], list[BSS._SireWrappers._system.System]],
    work_dir: str,
    config: BaseWorkflowConfig = BaseWorkflowConfig(),
    filename_stem: str,
    **extra_restraint_kwargs,
) -> list[BSS.FreeEnergy._restraint.Restraint]:
    """
    extract restraints from trajectory for each replica.
    Trajectory is loaded using MDAnalysis by default

    Parameters
    ----------
    source_list : str or BSS._SireWrappers._system.System
        List of source files (MUST BE .gro or .top files) or systems to load.
    work_dir : str
        Directory where the source files are located.
    filename_stem : str
        The stem of the filename to use for the output files.
    config : BaseWorkflowConfig
        Configuration object containing the parameters for the workflow.
    extra_protocol_kwargs : dict
        Additional keyword arguments to pass to the equilibration function.
    """
    # load the system from the source
    system_list = [load_system_from_source(source) for source in source_list]

    logger.info("Extracting restraints from each replica...")
    restraint_list = []
    for idx in range(1, len(system_list) + 1):
        # grab the exactly one .xtc for a given run, avoiding get #files
        # run this function multiple times leads to multiple #files
        trajs = glob.glob(
            os.path.join(
                work_dir,
                f"{filename_stem}{PreparationStage.EQUILIBRATED.file_suffix}_{idx}.xtc",
            )
        )
        if len(trajs) != 1:
            raise RuntimeError(
                f"Expected 1 trajectory(XTC) for run_{idx} in {work_dir}, but found {len(trajs)}."
            )
        trajectory_file = trajs[0]

        # grab the exactly one .tpr file for a given run, avoiding get #files
        # NOTE tpr is forced to be the top file in BSS for restraint extraction
        tops = glob.glob(
            os.path.join(
                work_dir,
                f"{filename_stem}{PreparationStage.EQUILIBRATED.file_suffix}_{idx}.tpr",
            )
        )
        if len(tops) != 1:
            raise RuntimeError(
                f"Expected 1 topology(TPR) in for run_{idx} in {work_dir}, but found {len(tops)}."
            )
        topology_file = tops[0]

        # use the temperature from the previous replica run
        temperature_equil = config.param_ensemble_equilibration.replicas[
            idx - 1
        ].temperature
        restraint = extract_restraint_from_traj(
            work_dir=work_dir,
            trajectory_file=trajectory_file,
            topology_file=topology_file,
            system=system_list[idx - 1],
            # GROMACS format for restraints, if output_filename provided, save
            # restraint to {work_dir}/{output_filename}
            output_filename=f"restraint_{idx}.itp",
            temperature=temperature_equil,
            append_to_ligand_selection=config.append_to_ligand_selection,
            restraint_idx=idx,  # Make sure results are named correctly
            **extra_restraint_kwargs,
        )
        restraint_list.append(restraint)

    return restraint_list


# def _move_and_create_files(input_dir: str, output_dir: str, filename_stem: str):
#     """
#     Move files to the correct location and create any necessary files.

#     Parameters
#     ----------
#     input_dir : str
#         Directory where the equilibration files are located.
#     output_dir : str
#         Directory where the files should be moved to.
#     """
#     output_bound_dir = Path(output_dir) / "bound"
#     output_bound_dir.mkdir(parents=True, exist_ok=True)
#     for dir_name in ["discharge", "restrain", "vanish"]:
#         os.makedirs(Path(output_bound_dir) / dir_name, exist_ok=True)

#         move_link_and_create_files()


def run_complete_system_setup_bound_and_free(
    *,
    protein_path: Optional[str] = None,
    ligand_path: Optional[str] = None,
    water_path: Optional[str] = None,
    config: BaseWorkflowConfig = BaseWorkflowConfig(),
    filename_stem: str,
    output_dir: str,
    use_slurm: bool = True,
) -> BSS._SireWrappers._system.System:

    # 1. get the equilibrated systems from source
    equil_system_list = _prepare_equilibrated_molecular_systems(
        filename_stem=filename_stem,
        config=config,
        protein_path=protein_path,
        ligand_path=ligand_path,
        water_path=water_path,
        output_dir=output_dir,
        use_slurm=use_slurm,
    )
    # 2. extract restraints from the equilibrated systems
    _ = _prepare_restraints_from_ensemble_equilibration(
        source_list=equil_system_list,
        config=config,
        filename_stem=filename_stem,
        work_dir=output_dir,
    )


def prepare_fep_simulation_systems():
    pass
