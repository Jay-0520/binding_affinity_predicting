import logging
import os
from typing import Optional, Sequence, Union

import BioSimSpace.Sandpit.Exscientia as BSS  # type: ignore[import]

from binding_affinity_predicting.components.utils import check_has_wat_and_box
from binding_affinity_predicting.data.schemas import EquilStep
from binding_affinity_predicting.simulation.utils import run_process

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def energy_minimise_system(
    source: Union[str, BSS._SireWrappers._system.System],
    output_file_path: str,
    min_steps: int = 1_000,
    mdrun_options: Optional[str] = None,
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

    # Minimise
    logger.info(f"Minimising input structure with {min_steps} steps...")
    protocol = BSS.Protocol.Minimisation(steps=min_steps)
    minimised_system = run_process(
        system=solvated_system, protocol=protocol, mdrun_options=mdrun_options
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
    steps: Optional[Sequence[Union[EquilStep, dict]]] = None,
    output_file_path: Optional[str] = None,
    work_dir: Optional[str] = None,
    mdrun_options: Optional[str] = None,
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Load a solvated system (from a file or already-loaded System), perform a sequence
    of NVT/NPT equilibration steps, and optionally save the final System.

    Parameters
    ----------
    source : str or System
        Path to the input file (PDB/GRO/etc) or an existing BioSimSpace System.
    steps : Sequence[EquilStep]
        A list of equilibration steps. Each step is a dict:
            (runtime, temperature_start, temperature_end, restraint, pressure)
        see the EquilStep class for details.
        Use None for pressure_atm to perform NVT.
    output_path : Optional[str]
        If provided, writes the final System to this path (e.g. "out/preequil.gro").
    work_dir : Optional[str]
        Directory to run GROMACS in. If None, a temp directory is created.
    mdrun_options : Optional[str]
        Extra mdrun flags as a single string (e.g. "-nt 4 -v").

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

    # if no steps, just return the system
    if not steps:
        logger.warning("No equilibration steps provided; returning input system.")
        return system

    # normalize every step into an EquilStep
    normalized: list[EquilStep] = []
    for s in steps:
        if isinstance(s, EquilStep):
            normalized.append(s)
        elif isinstance(s, dict):
            normalized.append(EquilStep.model_validate(s))
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
            "->{temperature_end} k, restraint={restraint}, pressure={pressure} atm"
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
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Pure equilibration: run through a single NVT/NPT step.
    """
    # convert to BSS units
    runtime_ps = runtime_ps * BSS.Units.Time.picosecond
    temperature_start_k = temperature_start_k * BSS.Units.Temperature.kelvin
    temperature_end_k = temperature_end_k * BSS.Units.Temperature.kelvin

    pressure_atm = (
        pressure_atm * BSS.Units.Pressure.atm if pressure_atm is not None else None
    )

    protocol = BSS.Protocol.Equilibration(
        runtime=runtime_ps,
        temperature_start=temperature_start_k,
        temperature_end=temperature_end_k,
        pressure=pressure_atm,
        restraint=restraint,
    )
    heated_system = run_process(
        system=system, protocol=protocol, work_dir=work_dir, mdrun_options=mdrun_options
    )

    return heated_system
