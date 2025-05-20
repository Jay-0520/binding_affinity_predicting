import logging
import os
from typing import Optional, Sequence, Union

import BioSimSpace.Sandpit.Exscientia as BSS  # type: ignore[import]

from binding_affinity_predicting.components.utils import check_has_wat_and_box
from binding_affinity_predicting.data.schemas import (
    EnsembleEquilibrationConfig,
    PreEquilStageConfig,
)
from binding_affinity_predicting.simulation.utils import run_process

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def energy_minimise_system(
    source: Union[str, BSS._SireWrappers._system.System],
    output_file_path: str,
    min_steps: int = 1_000,
    mdrun_options: Optional[str] = None,
    process_name: Optional[str] = None,
    **extra_protocol_kwargs,
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Minimise the input structure with GROMACS.

    Parameters
    ----------
    source : str or System
        Path to the input file (PDB/GRO/etc) or an existing BioSimSpace System
    output_file_path : str
        Path to the output directory where the minimised system will be saved.
    min_steps : int
        Number of minimisation steps to perform.
    mdrun_options : str, optional
        Additional options to pass to the GROMACS mdrun command.
    process_name : Optional[str]
        Name of the process. If None, a default name "gromacs" will be used.
        NOTE that {work_dir}/{process_name}.xtc/gro/edr/log defines the output files
    **extra_protocol_kwargs
        Any additional named arguments to pass into `BSS.Protocol.Minimisation`

    Returns
    -------
    minimised_system : BSS._SireWrappers._system.System
        Minimised system.
    """
    if isinstance(source, str):
        solvated_system = BSS.IO.readMolecules(str(source))  # type: ignore
    else:
        solvated_system = source

    # Check that it is actually solvated in a box of water
    check_has_wat_and_box(solvated_system)

    # Build the minimisation protocol
    logger.info(f"Minimising input structure with {min_steps} steps...")
    protocol = BSS.Protocol.Minimisation(steps=min_steps, **extra_protocol_kwargs)
    # Run gromacs here
    minimised_system = run_process(
        system=solvated_system,
        protocol=protocol,
        mdrun_options=mdrun_options,
        process_name=process_name,
    )

    # Save, renaming the velocity property to foo so avoid saving velocities. Saving the
    # velocities sometimes causes issues with the size of the floats overflowing the RST7
    # format.
    if output_file_path:
        logger.info(f"Writing solvated system to {output_file_path}")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        BSS.IO.saveMolecules(
            str(output_file_path),
            minimised_system,
            fileformat=["gro87", "grotop"],
            property_map={"velocity": "foo"},
        )

    return minimised_system


def preequilibrate_system(
    source: Union[str, BSS._SireWrappers._system.System],
    steps: Sequence[Union[PreEquilStageConfig, dict]],
    output_file_path: Optional[str] = None,
    work_dir: Optional[str] = None,
    mdrun_options: Optional[str] = None,
    process_name: Optional[str] = None,
    **extra_protocol_kwargs,
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Load a solvated system (from a file or already-loaded System), perform a sequence
    of NVT/NPT equilibration steps, and optionally save the final System.

    Parameters
    ----------
    source : str or System
        Path to the input file (PDB/GRO/etc) or an existing BioSimSpace System.
    steps : Sequence[EquilStep]
        A list of equilibration steps. Each step is a dict of parameters:
            (runtime, temperature_start, temperature_end, restraint, pressure)
        see the EquilStep class for details.
        Use None for pressure_atm to perform NVT.
    output_file_path : Optional[str]
        If provided, writes the final System to this path (e.g. "out/preequil.gro").
    work_dir : Optional[str]
        Directory to run GROMACS in. If None, a temp directory is created.
    mdrun_options : Optional[str]
        Extra mdrun flags as a single string (e.g. "-ntmpi 1 -ntomp 1").
    process_name : Optional[str]
        Name of the process. If None, a default name "gromacs" will be used.
        NOTE that {work_dir}/{process_name}.xtc/gro/edr/log defines the output files
    **extra_protocol_kwargs
        Any additional named arguments to pass into `BSS.Protocol.Equilibration`

    Returns
    -------
    System
        The equilibrated BioSimSpace System.
    """
    if isinstance(source, str):
        system = BSS.IO.readMolecules(str(source))  # type: ignore
    else:
        system = source

    # Check that it is solvated and has a box
    check_has_wat_and_box(system)

    # normalize every step into an EquilStep
    normalized: list[PreEquilStageConfig] = []
    for s in steps:
        if isinstance(s, PreEquilStageConfig):
            normalized.append(s)
        elif isinstance(s, dict):
            normalized.append(PreEquilStageConfig.model_validate(s))
        else:
            raise TypeError(f"Each step must be EquilStep or dict, got {type(s)}.")

    # iterate equilibration steps
    preeq_system = system
    for _step_param in normalized:
        runtime = _step_param.runtime
        temperature_start = _step_param.temperature_start
        temperature_end = _step_param.temperature_end
        restraint = _step_param.restraint
        pressure = _step_param.pressure

        logger.info(
            f"Running equilibration step: {runtime} ps, temperature {temperature_start}"
            f"->{temperature_end} k, restraint={restraint}, pressure={pressure} atm"
        )
        preeq_system = _heat_and_preequil_system_bss(
            system=preeq_system,
            runtime_ps=runtime,
            temperature_start_k=temperature_start,
            temperature_end_k=temperature_end,
            restraint=restraint,
            pressure_atm=pressure,
            work_dir=work_dir,
            mdrun_options=mdrun_options,
            process_name=process_name,
            **extra_protocol_kwargs,
        )

    if output_file_path:
        logger.info(f"Writing pre-equilibrated system to {output_file_path}")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        BSS.IO.saveMolecules(
            str(output_file_path), preeq_system, fileformat=["gro87", "grotop"]
        )

    return preeq_system


def _heat_and_preequil_system_bss(
    system: BSS._SireWrappers._system.System,
    runtime_ps: float,
    temperature_start_k: float,
    temperature_end_k: float,
    restraint: Optional[str] = None,
    pressure_atm: Optional[float] = None,
    work_dir: Optional[str] = None,
    mdrun_options: Optional[str] = None,
    process_name: Optional[str] = None,
    **extra_protocol_kwargs,
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Pure equilibration: run through a single NVT/NPT step.

    Parameters
    ----------
    system : BioSimSpace._SireWrappers._system.System
        A BioSimSpace System (e.g. returned by BSS.IO.readMolecules).
    runtime_ps : float
        Runtime in picoseconds.
    temperature_start_k : float
        Starting temperature in Kelvin.
    temperature_end_k : float
        Ending temperature in Kelvin.
    restraint : Optional[str]
        Restraint type (e.g. "backbone", "heavy").
    pressure_atm : Optional[float]
        Pressure in atm. If None, NVT is performed.
    work_dir : Optional[str]
        Directory to run GROMACS in. If None, a temp directory is created.
    mdrun_options : Optional[str]
        Extra mdrun flags as a single string (e.g. "-ntmpi 1 -ntomp 1").
    process_name : Optional[str]
        Name of the process. If None, a default name "gromacs" will be used.
        NOTE that {work_dir}/{process_name}.xtc/gro/edr/log defines the output files
    **extra_protocol_kwargs
        Any additional named arguments to pass into `BSS.Protocol.Equilibration`
    """
    # convert to BSS units
    runtime_ps = runtime_ps * BSS.Units.Time.picosecond
    temperature_start_k = temperature_start_k * BSS.Units.Temperature.kelvin
    temperature_end_k = temperature_end_k * BSS.Units.Temperature.kelvin
    pressure_atm = (
        pressure_atm * BSS.Units.Pressure.atm if pressure_atm is not None else None
    )
    # Build the equilibration protocol
    protocol = BSS.Protocol.Equilibration(
        runtime=runtime_ps,
        temperature_start=temperature_start_k,
        temperature_end=temperature_end_k,
        pressure=pressure_atm,
        restraint=restraint,
        **extra_protocol_kwargs,
    )
    # Run GROMACS here
    heated_system = run_process(
        system=system,
        protocol=protocol,
        work_dir=work_dir,
        process_name=process_name,
        mdrun_options=mdrun_options,
    )

    return heated_system


def _equilibrate_system_bss(
    system: BSS._SireWrappers._system.System,
    runtime_ns: float,  # NOTE unit is changed to ns here in BSS
    temperature_k: float,
    pressure_atm: Optional[float] = None,
    restraint: Optional[str] = None,
    work_dir: Optional[str] = None,
    mdrun_options: Optional[str] = None,
    process_name: Optional[str] = None,
    **extra_protocol_kwargs,
) -> BSS._SireWrappers._system.System:
    """
    Run a single NVT/NPT equilibration step using the Production protocol.
    """
    # convert to BSS units
    runtime_ns = runtime_ns * BSS.Units.Time.nanosecond
    temperature_k = temperature_k * BSS.Units.Temperature.kelvin
    pressure_atm = (
        pressure_atm * BSS.Units.Pressure.atm if pressure_atm is not None else None
    )

    protocol = BSS.Protocol.Production(
        runtime=runtime_ns,
        temperature=temperature_k,
        pressure=pressure_atm,
        restraint=restraint,
        **extra_protocol_kwargs,
    )

    # Run GROMACS here
    equilibrated_system = run_process(
        system=system,
        protocol=protocol,
        work_dir=work_dir,
        mdrun_options=mdrun_options,
        process_name=process_name,
    )

    return equilibrated_system


def run_ensemble_equilibration(
    source: Union[str, BSS._SireWrappers._system.System],
    replicas: Sequence[Union[EnsembleEquilibrationConfig, dict]],
    output_file_path: Optional[str] = None,
    work_dir: Optional[str] = None,
    mdrun_options: Optional[str] = None,
    process_name: Optional[str] = None,
    **extra_protocol_kwargs,
) -> BSS._SireWrappers._system.System:
    """
    Run an ensemble of equilibration simulations by using BSS.Protocol.Production.

    Parameters
    ----------
    source : str or System
        Path to the input file (PDB/GRO/etc) or an existing BioSimSpace System.
    steps : Sequence[EnsembleEquilibrationConfig]
        A list of equilibration steps. Each step is a dict of parameters:
            (runtime, temperature_start, temperature_end, restraint, pressure)
        Use None for pressure_atm to perform NVT.
    output_file_path : Optional[str]
        If provided, writes the final System to this path (e.g. "out/preequil.gro").
    work_dir : Optional[str]
        Directory to run GROMACS in. If None, a temp directory is created.
    mdrun_options : Optional[str]
        Extra mdrun flags as a single string (e.g. "-ntmpi 1 -ntomp 1").
    process_name : Optional[str]
        Name of the process. If None, a default name "gromacs" will be used.
    **extra_protocol_kwargs
        Any additional named arguments to pass into `BSS.Protocol.Equilibration`
    """

    # launch a ensemble of replicates
    for i in range(1, replicas + 1):
        logger.info(f"Launching replicate {i}/{replicas}...")
        _equilibrate_system_bss(
            system=source,
            runtime_ps=5,
            temperature_k=298.15,
            pressure_atm=1,
            restraint=None,
            work_dir=work_dir,
            mdrun_options=mdrun_options,
            process_name=f"eb_{i}",
            **extra_protocol_kwargs,
        )

    # Check if we have already run any ensemble equilibration simulations
    outdirs_to_run = [outdir for outdir in outdirs if not _os.path.isdir(outdir)]
    outdirs_already_run = [outdir for outdir in outdirs if _os.path.isdir(outdir)]
    self._logger.info(
        f"Found {len(outdirs_already_run)} ensemble equilibration directories already run."
    )

    for outdir in outdirs_to_run:
        _subprocess.run(["mkdir", "-p", outdir], check=True)
        for input_file in [
            f"{self.input_dir}/{ifile}"
            for ifile in _PreparationStage.PREEQUILIBRATED.get_simulation_input_files(
                self.leg_type
            )
        ]:
            _subprocess.run(["cp", "-r", input_file, outdir], check=True)

        # Also write a pickle of the config to the output directory
        sysprep_config.save_pickle(outdir, self.leg_type)

    if sysprep_config.slurm:
        if self.leg_type == _LegType.BOUND:
            job_name = "ensemble_equil_bound"
            fn = _system_prep.slurm_ensemble_equilibration_bound
        elif self.leg_type == _LegType.FREE:
            job_name = "ensemble_equil_free"
            fn = _system_prep.slurm_ensemble_equilibration_free
        else:
            raise ValueError("Invalid leg type.")

        # For each ensemble member to be run, run a 5 ns simulation in a seperate directory

        for i, outdir in enumerate(outdirs_to_run):
            self._logger.info(
                f"Running ensemble equilibration {i + 1} of {len(outdirs_to_run)}. Submitting through SLURM..."
            )
            self._run_slurm(fn, wait=False, run_dir=outdir, job_name=job_name)

        self.virtual_queue.wait()  # Wait for all jobs to finish

        # Check that the required input files have been produced, since slurm can fail silently
        for i, outdir in enumerate(outdirs_to_run):
            for file in _PreparationStage.PREEQUILIBRATED.get_simulation_input_files(
                self.leg_type
            ) + ["somd.rst7"]:
                if not _os.path.isfile(f"{outdir}/{file}"):
                    raise RuntimeError(
                        f"SLURM job failed to produce {file}. Please check the output of the "
                        f"last slurm log in {outdir} directory for errors."
                    )

    else:  # Not slurm
        for i, outdir in enumerate(outdirs_to_run):
            self._logger.info(
                f"Running ensemble equilibration {i + 1} of {len(outdirs_to_run)}..."
            )
            _system_prep.run_ensemble_equilibration(
                self.leg_type,
                outdir,
                outdir,
            )

    # Give the output files unique names
    equil_numbers = [int(outdir.split("_")[-1]) for outdir in outdirs_to_run]
    for equil_number, outdir in zip(equil_numbers, outdirs_to_run):
        _subprocess.run(
            ["mv", f"{outdir}/somd.rst7", f"{outdir}/somd_{equil_number}.rst7"],
            check=True,
        )

    # Load the system and mark the ligand to be decoupled
    self._logger.info("Loading pre-equilibrated system...")
    pre_equilibrated_system = _BSS.IO.readMolecules(
        [
            f"{self.input_dir}/{file}"
            for file in _PreparationStage.PREEQUILIBRATED.get_simulation_input_files(
                self.leg_type
            )
        ]
    )

    # Mark the ligand to be decoupled so the restraints searching algorithm works
    lig = _BSS.Align.decouple(pre_equilibrated_system[0], intramol=True)
    pre_equilibrated_system.updateMolecule(0, lig)

    # If this is the bound leg, search for restraints
    if self.leg_type == _LegType.BOUND:
        # For each run, load the trajectory and extract the restraints
        for i, outdir in enumerate(outdirs):
            self._logger.info(f"Loading trajectory for run {i + 1}...")
            top_file = f"{self.input_dir}/{_PreparationStage.PREEQUILIBRATED.get_simulation_input_files(self.leg_type)[0]}"
            traj = _BSS.Trajectory.Trajectory(
                topology=top_file,
                trajectory=f"{outdir}/gromacs.xtc",
                system=pre_equilibrated_system,
            )
            self._logger.info(f"Selecting restraints for run {i + 1}...")
            restraint = _BSS.FreeEnergy.RestraintSearch.analyse(
                method="BSS",
                system=pre_equilibrated_system,
                traj=traj,
                work_dir=outdir,
                temperature=298.15 * _BSS.Units.Temperature.kelvin,
                append_to_ligand_selection=sysprep_config.append_to_ligand_selection,
            )

            # Check that we actually generated a restraint
            if restraint is None:
                raise ValueError(f"No restraints found for run {i + 1}.")

            # Save the restraints to a text file and store within the Leg object
            with open(f"{outdir}/restraint_{i + 1}.txt", "w") as f:
                f.write(restraint.toString(engine="SOMD"))  # type: ignore
            self.restraints.append(restraint)

        return pre_equilibrated_system

    else:  # Free leg
        return pre_equilibrated_system
