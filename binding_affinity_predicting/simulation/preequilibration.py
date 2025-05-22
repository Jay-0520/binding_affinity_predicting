import logging
import os
import tempfile
from typing import Optional, Sequence, Union

import BioSimSpace.Sandpit.Exscientia as BSS  # type: ignore[import]

from binding_affinity_predicting.components.utils import check_has_wat_and_box
from binding_affinity_predicting.data.schemas import (
    EnsembleEquilibrationConfig,
    PreEquilStageConfig,
)
from binding_affinity_predicting.simulation.utils import decouple_ligand_in_system, run_process

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
    process_name: str = "preequilibration",
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


def _equilibrating_system_bss(
    system: BSS._SireWrappers._system.System,
    runtime_ns: float,  # NOTE unit is changed to ns here in BSS
    temperature_k: float = 300.0,
    pressure_atm: float = 1.0,
    timestep_fs: float = 2,
    work_dir: Optional[str] = None,
    mdrun_options: Optional[str] = None,
    process_name: Optional[str] = None,
    **extra_protocol_kwargs,
) -> BSS._SireWrappers._system.System:
    """
    Run a single NVT/NPT equilibration step using the Production protocol.

    NOTE that for biological system, it's mostly NPT (300K and 1 atm)

    Parameters
    ----------
    system : BioSimSpace._SireWrappers._system.System
        A BioSimSpace System (e.g. returned by BSS.IO.readMolecules).
    runtime_ns : float; Runtime in nanoseconds.
    temperature_k : float; Temperature in Kelvin.
    pressure_atm : Optional[float]
        Pressure in atm. If None, NVT is performed.
    timestep_fs : Optional[float]
        Timestep in femtoseconds. If None, the default timestep is used.
    work_dir : Optional[str]
        Directory to run GROMACS in. If None, a temp directory is created.
    mdrun_options : Optional[str]
        Extra mdrun flags as a single string (e.g. "-ntmpi 1 -ntomp 1").
    process_name : Optional[str]
        Name of the process. If None, a default name "gromacs" will be used.
        NOTE that {work_dir}/{process_name}.xtc/gro/edr/log defines the output files
    **extra_protocol_kwargs
        Any additional named arguments to pass into `BSS.Protocol.Production`

    Returns
    -------
    System
        The equilibrated BioSimSpace System.
    """
    # Check that it is solvated in a box of water
    check_has_wat_and_box(system)

    # convert to BSS units
    runtime_ns = runtime_ns * BSS.Units.Time.nanosecond
    temperature_k = temperature_k * BSS.Units.Temperature.kelvin
    timestep_fs = (
        timestep_fs * BSS.Units.Time.femtosecond if timestep_fs is not None else None
    )
    pressure_atm = (
        pressure_atm * BSS.Units.Pressure.atm if pressure_atm is not None else None
    )

    # decouple the ligand in the system
    # TODO: we don't need to decouple the ligand here, move this to a separate step in the workflow
    system = decouple_ligand_in_system(system)

    protocol = BSS.Protocol.Production(
        runtime=runtime_ns,
        temperature=temperature_k,
        pressure=pressure_atm,
        timestep=timestep_fs,
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
    process_name: str = "ensemble_equilibration",
    **extra_protocol_kwargs,
) -> list[BSS._SireWrappers._system.System]:
    """
    Run an ensemble of equilibration simulations by using BSS.Protocol.Production.

    Note that in A3FE protocol, this step is only needed for BOUND leg
    see ref: https://pubs.acs.org/doi/10.1021/acs.jctc.4c00806

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
    if isinstance(source, str):
        base_system = BSS.IO.readMolecules(source)
    else:
        base_system = source

    # normalize every step into an EquilStep
    configs: list[EnsembleEquilibrationConfig] = []
    for s in replicas:
        if isinstance(s, EnsembleEquilibrationConfig):
            configs.append(s)
        elif isinstance(s, dict):
            configs.append(EnsembleEquilibrationConfig.model_validate(s))
        else:
            raise TypeError(f"Each step must be EquilStep or dict, got {type(s)}.")

    # run each replicate
    systems = []
    for ndx, _repeat in enumerate(configs, start=1):
        logger.info(f"Launching replicate equilibration run {ndx}/{len(configs)}...")

        if work_dir:
            rep_dir = os.path.join(work_dir, f"ensemble_equilibration_{ndx}")
            os.makedirs(rep_dir, exist_ok=True)
        else:
            rep_dir = tempfile.mkdtemp(prefix=f"{process_name}_{ndx}_")

        pname = f"{process_name}_{ndx}"
        equilibrated_system_out = _equilibrating_system_bss(
            system=base_system,
            runtime_ps=_repeat.runtime,
            temperature_k=_repeat.temperature,
            pressure_atm=_repeat.pressure,
            restraint=_repeat.restraint,
            timestep_fs=_repeat.timestep,
            work_dir=rep_dir,
            mdrun_options=mdrun_options,
            process_name=pname,
            **extra_protocol_kwargs,
        )
        systems.append(equilibrated_system_out)

        out_gro = os.path.join(rep_dir, f"{pname}.gro")
        logger.info(f"Saving replicate {ndx} system to {out_gro}")
        BSS.IO.saveMolecules(
            str(out_gro), equilibrated_system_out, fileformat=["gro87", "grotop"]
        )

    return systems
