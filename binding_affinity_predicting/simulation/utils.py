import logging
import time
from typing import Optional

import BioSimSpace.Sandpit.Exscientia as BSS  # type: ignore[import]
from BioSimSpace.Sandpit.Exscientia._SireWrappers import Molecule  # type: ignore[import]
from sire.legacy import Mol as SireMol  # type: ignore[import]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_process(
    system: BSS._SireWrappers._system.System,
    protocol: BSS.Protocol._protocol.Protocol,
    work_dir: Optional[str] = None,
    mdrun_options: Optional[str] = None,
) -> BSS._SireWrappers._system.System:
    """
    Run a process with GROMACS, raising informative
    errors in the event of a failure.


    Parameters
    ----------
    system : BSS._SireWrappers._system.System
        System to run the process on.
    protocol : BSS._Protocol._protocol.Protocol
        Protocol to run the process with.
    work_dir : str, optional
        The working directory to run the process in. If none,
        a temporary directory will be created.
    mdrun_options : str, optional
        Additional options to pass to the GROMACS mdrun command.
        If `mdrun_options` is provided (e.g. "-nt 4 -v"), it will be split
        and passed as arguments.

    Returns
    -------
    system : BSS._SireWrappers._system.System
    """
    process = BSS.Process.Gromacs(system, protocol, work_dir=work_dir)

    # Added by JJ-2025-05-05 for local run on Mac
    if mdrun_options:
        args = mdrun_options.split()
        for flag, val in zip(args[::2], args[1::2]):
            process.setArg(flag, val)

    process.start()
    process.wait()

    time.sleep(10)
    if process.isError():
        logger.error(process.stdout())
        logger.error(process.stderr())
        process.getStdout()
        process.getStderr()
        raise BSS._Exceptions.ThirdPartyError("The process failed.")

    system = process.getSystem(block=True)
    if system is None:
        logger.error(process.stdout())
        logger.error(process.stderr())
        process.getStdout()
        process.getStderr()
        raise BSS._Exceptions.ThirdPartyError("The final system is None.")

    return system


def rename_lig(
    bss_system: BSS._SireWrappers._system.System, new_name: str = "LIG"
) -> None:  # type: ignore
    """Rename the ligand in a BSS system.

    Parameters
    ----------
    bss_system : BioSimSpace.Sandpit.Exscientia._SireWrappers._system.System
        The BSS system.
    new_name : str
        The new name for the ligand.
    Returns
    -------
    None
    """
    # Ensure that we only have one molecule
    if len(bss_system) != 1:
        raise ValueError("BSS system must only contain one molecule.")

    # Extract the sire object for the single molecule
    mol = Molecule(bss_system[0])
    mol_sire = mol._sire_object

    # Create an editable version of the sire object
    mol_edit = mol_sire.edit()

    # Rename the molecule and the residue to the supplied name
    resname = SireMol.ResName(new_name)  # type: ignore
    mol_edit = mol_edit.residue(SireMol.ResIdx(0)).rename(resname).molecule()  # type: ignore
    mol_edit = mol_edit.edit().rename(new_name).molecule()

    # Commit the changes and update the system
    mol._sire_object = mol_edit.commit()
    bss_system.updateMolecule(0, mol)
