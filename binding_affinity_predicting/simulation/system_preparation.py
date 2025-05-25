import logging
from typing import Callable, Optional, Union

import BioSimSpace.Sandpit.Exscientia as BSS  # type: ignore[import]

from binding_affinity_predicting.simulation.utils import (
    decouple_ligand_in_system,
    load_system_from_source,
    save_system_to_local,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def solvate_system(
    source: Union[str, BSS._SireWrappers._system.System],
    water_model: str = "tip3p",
    filename_stem: Optional[str] = None,
    output_dir: Optional[str] = None,
    **solvate_kwargs,
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Determine an appropriate (rhombic dodecahedron)
    box size, then solvate the input structure using
    TIP3P water, adding 150 mM NaCl to the system.
    The resulting system is saved to the input directory.

    Parameters
    ----------
    source : str or System
        Path to the input file (PDB/GRO/etc) or an existing BioSimSpace System
    water_model : str
        Water model to use for solvation (e.g. "tip3p")
    filename_stem : Optional[str]
        The stem of the filename to use for the output files. Must be set if output_dir is set.
    output_dir : Optional[str]
        Directory to write the output files to. If None, not written to disk.
    **solvate_kwargs : dict
        Additional keyword arguments to pass to the solvation function.

    Returns
    -------
    solvated_system : BSS._SireWrappers._system.System
        Solvated system.
    """
    base_system = load_system_from_source(source)

    solvated_system = _solvate_molecular_system_bss(
        system=base_system, water_model=water_model, **solvate_kwargs
    )

    save_system_to_local(
        system=solvated_system,
        filename_stem=filename_stem,
        output_dir=output_dir,
    )

    return solvated_system


def _solvate_molecular_system_bss(
    system: BSS._SireWrappers._system.System,
    water_model: str,
    ion_conc: float = 0.15,
    md_box_factory: Callable[
        [float], tuple[list[float], list[float]]
    ] = BSS.Box.rhombicDodecahedronHexagon,
    padding_distance: float = 15.0,
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Solvate (and ionize) an existing BioSimSpace System.

    Note that currently ion must be NaCl

    Parameters
    ----------
    system: BioSimSpace._SireWrappers._system.System
        A BioSimSpace System (e.g. returned by BSS.IO.readMolecules).
    water_model: str
        Name of the water model (e.g. "TIP3P", "SPC/E").
    ion_conc: float
        Salt concentration in mol/L (e.g. 0.150).
    box_factory
        Function that takes a box length (in BSS units) and returns (Box, angles).
    padding_distance: float
        Padding in Å around the molecule.
    exclude_distance: Optional[float]
        If not None, exclude existing waters > this distance (Å) from non-water atoms.

    Returns
    -------
    System
        The solvated & ionized system.
    """
    if not isinstance(system, BSS._SireWrappers._system.System):
        raise TypeError("system must be a BioSimSpace System!")

    # Determine the box size
    # Taken from https://github.com/michellab/BioSimSpaceTutorials/blob/main/01_introduction/\
    # 02_molecular_setup.ipynb
    # Get the minimium and maximum coordinates of the bounding box that
    # minimally encloses the protein.
    logger.info("Determining optimal box dimension...")
    # Want to get box size based on complex/ ligand, exlcuding any crystallographic waters
    non_waters = [mol for mol in system if mol.nAtoms() != 3]  # type: ignore
    dry_system = BSS._SireWrappers._system.System(non_waters)  # type: ignore
    box_min, box_max = dry_system.getAxisAlignedBoundingBox()

    # Work out the box size from the difference in the coordinates.
    box_size = [y - x for x, y in zip(box_min, box_max)]

    # Add 15 A padding to the box size in each dimension.
    padding = padding_distance * BSS.Units.Length.angstrom

    # Work out an appropriate box. This will used in each dimension to ensure
    # that the cutoff constraints are satisfied if the molecule rotates.
    box_length = max(box_size) + 2 * padding
    box, angles = md_box_factory(box_length)

    # Exclude waters if they are too far from the protein. These are unlikely
    # to be important for the simulation and including them would require a larger
    # box. Exclude if further than 10 A from the protein.
    try:
        waters_to_exclude = [
            wat
            for wat in system.search(
                "water and not (water within 10 of protein)"
            ).molecules()
        ]
        # If we have failed to convert to molecules (old BSS bug), then do this for each molecule.
        if hasattr(waters_to_exclude[0], "toMolecule"):
            waters_to_exclude = [wat.toMolecule() for wat in waters_to_exclude]
        logger.info(
            f"Excluding {len(waters_to_exclude)} waters that are over 10 A from the protein"
        )
    except ValueError:
        waters_to_exclude = []

    system.removeMolecules(waters_to_exclude)

    solvated_system = BSS.Solvent.solvate(
        model=water_model,
        molecule=system,
        box=box,
        angles=angles,
        ion_conc=ion_conc,
    )

    return solvated_system


def extract_restraint_from_traj(
    work_dir: str,
    trajectory_file: str,
    topology_file: str,
    system: BSS._SireWrappers._system.System,
    output_filename: Optional[str] = None,
    temperature: float = 300.0,
    append_to_ligand_selection: str = "",
    **extra_restraint_kwargs,
) -> BSS.FreeEnergy._restraint.Restraint:  # type: ignore
    """
    For a BOUND leg, loop over trajectories in each outdir,
    run BSS.FreeEnergy.RestraintSearch.analyse(), save to txt,
    and accumulate into restraint_list. Return the final system.

    Parameters
    ----------
    work_dir : str
        The simulation directory containing the trajectory files.
    trajectory_file : str
        The trajectory file to analyze.
    topology_file : str
        The topology file to analyze.
        NOTE is forced to be GROMACS TPR file to be compatible with BSS
    system : BSS._SireWrappers._system.System
        The system to analyze.
    output_filename : Optional[str]
        The output filename for the restraint file.
    temperature : float
        The temperature of the system in Kelvin.
    append_to_ligand_selection : str
        Appends the supplied string to the default atom selection which chooses
        the atoms in the ligand to consider as potential anchor points. The default
        atom selection is f'resname {ligand_resname} and not name H*'. Uses the
        mdanalysis atom selection language. For example, 'not name O*' will result
        in an atom selection of f'resname {ligand_resname} and not name H* and not
        name O*'.

    Returns
    -------
    """
    # convert to BSS units
    temperature = temperature * BSS.Units.Temperature.kelvin

    # Mark the ligand to be decoupled so the restraints searching algorithm works
    system = decouple_ligand_in_system(system=system)

    traj = BSS.Trajectory.Trajectory(
        topology=topology_file,
        trajectory=trajectory_file,
        system=system,
    )
    restraint = BSS.FreeEnergy.RestraintSearch.analyse(
        method="BSS",
        system=system,  # must contain a single decoupled molecule
        traj=traj,
        work_dir=work_dir,
        temperature=temperature,
        append_to_ligand_selection=append_to_ligand_selection,
        **extra_restraint_kwargs,
    )
    if restraint is None:
        raise ValueError(f"No restraint found in {work_dir} for {trajectory_file}")

    if output_filename:
        with open(f"{work_dir}/{output_filename}", "w") as f:
            f.write(restraint.toString(engine="GROMACS"))

    return restraint
