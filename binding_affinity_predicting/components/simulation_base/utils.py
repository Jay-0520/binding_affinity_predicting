import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

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
    decouple it if not already decoupled, replace/update it in the system,
    and return the modified system.

    Parameters
    ----------
    system
        A BSS System containing protein, ligand, waters, etc.
    ligand_resname
        The residue name to look for first (e.g. "LIG").
    min_atoms, max_atoms
        Fallback atom-count heuristic if no molecule is named ligand_resname.

    Returns
    -------
    system
        The same System object, with the ligand molecule decoupled via BSS.Align.decouple.
    """
    n_mols = system.nMolecules()
    ligand_idx = None

    # try explicit residue-name match
    for i in range(n_mols):
        mol = system[i]
        try:
            name = mol._sire_object.residue(0).name().value()
        except Exception:
            continue
        if name == ligand_resname:
            ligand_idx = i
            break

    # try fallback based on size
    if ligand_idx is None:
        for i in range(n_mols):
            na = system[i].nAtoms()
            if min_atoms < na < max_atoms:
                if ligand_idx is not None:
                    raise ValueError(f"Multiple ligand candidates ({ligand_idx}, {i})")
                ligand_idx = i

    if ligand_idx is None:
        raise ValueError(
            f"Could not identify ligand: no residue named '{ligand_resname}' "
            f"and no molecule in {min_atoms}-{max_atoms} atoms."
        )

    # try decoupling, but skip if already decoupled
    try:
        lig = BSS.Align.decouple(system[ligand_idx], intramol=True)
    except Exception as e:
        msg = str(e).lower()
        if "already been decoupled" in msg or "isdecoupled" in msg:
            logger.warning("Ligand already decoupled, skipping.")
            return system
        raise e

    # sanity check based on molecule size
    if not (min_atoms < lig.nAtoms() < max_atoms):
        raise ValueError(
            f"Decoupled molecule at index {ligand_idx} has {lig.nAtoms()} atoms; "
            "does not look like a ligand."
        )
    # TODO: not sure why we need this - by JJH-2025-05-22; it seems align.decouple()
    # may mess up the indexing?
    system.updateMolecule(ligand_idx, lig)
    return system


def save_system_to_local(
    system: BSS._SireWrappers._system.System,
    filename_stem: Optional[str] = None,
    output_dir: Optional[str] = None,
    fileformat: list[str] = ["gro87", "grotop"],
    **save_kwargs,
) -> None:
    """
    If output_dir is provided, write out `<output_dir>/<filename_stem>.<ext>` for
    each fmt in fileformat, e.g. ["gro87","grotop"] -> .gro and .top, and log accordingly.

    Parameters
    ----------
    system : BioSimSpace._SireWrappers._system.System
        The system to save.
    filename_stem : Optional[str]
        The stem of the filename to use for the output files. Must be set if output_dir is set.
    output_dir : Optional[str]
        Directory to write the output files to. If None, not written to disk.
    fileformat : list[str]
        List of file formats to save the system in.

    Any additional keyword arguments (e.g. `property_map`) will be passed
    directly through to `BSS.IO.saveMolecules`.
    """
    if not output_dir:
        return

    # enforce that filename_stem comes with output_dir
    if output_dir is not None and not filename_stem:
        raise ValueError("`filename_stem` must be provided when `output_dir` is set.")

    # ensure directory exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # note that gro87 -> gro and grotop -> top files in GROMACS
    files_str = ", ".join(f"{filename_stem}.{ext}" for ext in fileformat)
    logger.info(f"Writing {files_str} to {out_dir}")

    # Build the *basename* including the stem
    prefix = out_dir / filename_stem

    # BSS.IO.saveMolecules() takes in a basename, meaning that:
    # 1. "a/b/c/tmp_name" or "a/b/c/tmp_name.gro" -> "tmp_name.gro" and "tmp_name.top"
    #     in the directory "a/b/c".
    # 2. "tmp_name.gro" or "tmp_name" -> "tmp_name.gro" and "tmp_name.top" in the
    #     current directory.
    BSS.IO.saveMolecules(str(prefix), system, fileformat=fileformat, **save_kwargs)


def load_system_from_source(
    source: Union[str, BSS._SireWrappers._system.System],
) -> BSS._SireWrappers._system.System:
    """
    Load a BioSimSpace System from disk.

    If `source` is already a System, return it unchanged.
    If it's a path ending in .gro or .top, assume its partner file
    (same stem, other extension) lives alongside it and load both.

    Parameters
    ----------
    source : str or BSS._SireWrappers._system.System
        Path to the input file or an existing BioSimSpace System.
        input file MUST BE a GROMACS .gro or .top file.
    """
    if not isinstance(source, str):
        return source

    stem, ext = os.path.splitext(source)
    ext = ext.lower()

    if ext not in (".gro", ".top"):
        raise ValueError(f"Expected a GROMACS .gro or .top file, got '{source}'.")

    gro_path = stem + ".gro"
    top_path = stem + ".top"

    missing = [p for p in (gro_path, top_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Both GRO/TOP files must exist for loading into a BSS system."
            f"Missing: {', '.join(missing)}"
        )

    return BSS.IO.readMolecules([gro_path, top_path])


def load_fresh_system(
    source: Union[str, BSS._SireWrappers._system.System],
) -> BSS._SireWrappers._system.System:
    """
    Return a freshly-loaded System for each call.

    - If `source` is a str, delegate to load_system_from_source(().
    - If `source` is already a System, dump it to a temp GRO/TOP pair
      and re-load so that further mutations don't affect the original.
      and always return a fresh copy
    """
    if isinstance(source, str):
        return load_system_from_source(source)

    # source is a System: snapshot to disk then re-load
    with tempfile.TemporaryDirectory(prefix="bss_sys_") as td:
        base_path = os.path.join(td, "temp_system")
        BSS.IO.saveMolecules(base_path, source, fileformat=["gro87", "grotop"])
        gro_path = base_path + ".gro"
        fresh = load_system_from_source(source=gro_path)

    return fresh


def check_has_wat_and_box(system: BSS._SireWrappers._system.System) -> None:
    """Check that the system has water and a box."""
    if system.getBox() == (None, None):
        raise ValueError("System does not have a box.")
    if system.nWaterMolecules() == 0:
        raise ValueError("System does not have water.")
