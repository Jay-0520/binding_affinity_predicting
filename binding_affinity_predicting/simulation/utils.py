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
    process_name: Optional[str] = None,
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
    process_name : str, optional
        The name of the process. If none, a default name "gromacs" will be used.
        NOTE that {work_dir}/{process_name}.xtc/gro/edr/log defines the output files
    mdrun_options : str, optional
        Additional options to pass to the GROMACS mdrun command.
        If `mdrun_options` is provided (e.g. "-nt 4 -v"), it will be split
        and passed as arguments.

    Returns
    -------
    system : BSS._SireWrappers._system.System
    """
    process = BSS.Process.Gromacs(
        system, protocol, work_dir=work_dir, name=process_name
    )

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
) -> None:
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
    resname = SireMol.ResName(new_name)
    mol_edit = mol_edit.residue(SireMol.ResIdx(0)).rename(resname).molecule()
    mol_edit = mol_edit.edit().rename(new_name).molecule()

    # Commit the changes and update the system
    mol._sire_object = mol_edit.commit()
    bss_system.updateMolecule(0, mol)


def decouple_ligand_in_system(
    system: BSS._SireWrappers._system.System,
    ligand_resname: str = "LIG",
    min_atoms: int = 5,
    max_atoms: int = 100,
) -> BSS._SireWrappers._system.System:
    """
    Locate the ligand molecule in a BioSimSpace System by residue name
    decouple it, replace it in the system, and return the modified system.

    Parameters
    ----------
    system
        A BSS System containing protein, ligand, waters, etc.
    ligand_resname
        The residue name to look for first (e.g. "LIG").
    min_atoms, max_atoms
        Fallback atom‐count heuristic if no molecule is named ligand_resname.

    Returns
    -------
    system
        The same System object, with the ligand molecule decoupled via BSS.Align.decouple.
    """
    n_mols = system.nMolecules()
    ligand_idx = None

    # try explicit residue‐name match
    for i in range(n_mols):
        mol = system[i]
        try:
            name = mol._sire_object.residue(0).name().value()
        except Exception:
            continue
        if name == ligand_resname:
            ligand_idx = i
            break

    if ligand_idx is None:
        raise ValueError(
            f"Could not identify ligand: no residue named '{ligand_resname}'."
        )

    # ) Decouple & replace
    lig = BSS.Align.decouple(system[ligand_idx], intramol=True)

    # Check if the decoupled molecule is a ligand based on molecule size
    if not (min_atoms < lig.nAtoms() < max_atoms):
        raise ValueError(
            f"Decoupled molecule at index {ligand_idx} has {lig.nAtoms()} atoms; "
            "does not look like a ligand."
        )
    # TODO: not sure why we need this - by JJH-2025-05-22; it seems align.decouple() 
    # may mess up the indexing?
    system.updateMolecule(ligand_idx, lig)
    return system
