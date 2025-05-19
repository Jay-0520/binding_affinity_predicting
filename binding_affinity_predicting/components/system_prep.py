from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.schemas.enums import LegType
import logging
from BioSimSpace.Sandpit.Exscientia._SireWrappers import Molecule as Molecule
from sire.legacy import Mol as SireMol
from binding_affinity_predicting.schemas.enums import (
    LegType,
    StageType,
    PreparationStage,
)
import pathlib as _pathlib
import pickle as _pkl
from typing import Optional, Callable
import BioSimSpace.Sandpit.Exscientia as BSS
from pydantic import BaseModel
from pydantic import Field
import os
from binding_affinity_predicting.components.utils import check_has_wat_and_box
import logging
import pathlib
import BioSimSpace.Sandpit.Exscientia as BSS
import time


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _parameterise_water(
    file_path: str,
    forcefield: str = "ff14SB",
    water_model: str = "tip3p",
    output_file_path: Optional[str] = None,
) -> BSS._SireWrappers._system.System:
    """
    Parameterise all (crystal) water molecules in a given file using the specified forcefield and water model.

    Parameters
    ----------
    filename : str
        Path to the water coordinate file (e.g. water.pdb) containing water molecules.
    forcefield : str
        Force field name to apply (e.g. "ff14SB").
    water_model : str
        Identifier for the water model (e.g. "tip3p", "spce").
    output_file_path : Optional[str]
        If provided, the assembled System will be written to this path (e.g. "out/waters.gro").
        The file format will be GROMACS (gro87 or grotop).

    Returns
    -------
    System
        A BioSimSpace System containing the assembled, parameterised water molecules.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Water file not found: {file_path}")

    # empty or malformatted pdb file results in a BSS error
    waters_sys = BSS.IO.readMolecules(str(file_path))
    logger.info(
        f"Parameterising {len(waters_sys)} water molecule(s) with forcefield={forcefield}, water_model={water_model}",
    )

    # Parameterise each water and extract the molecule
    parameterised = [
        BSS.Parameters.parameterise(
            molecule=wat,
            forcefield=forcefield,
            water_model=water_model,
        ).getMolecule()
        for wat in waters_sys
    ]

    # Assemble into a single System
    system = parameterised[0].toSystem()
    for mol in parameterised[1:]:
        system += mol  # only works for different molecules

    if output_file_path:
        logger.info(f"Writing parameterised water system to {output_file_path}")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        BSS.IO.saveMolecules(
            str(output_file_path), system, fileformat=["gro87", "grotop"]
        )

    return system


def _parameterise_ligand(
    file_path: str,
    forcefield: str = "openff_unconstrained-2.0.0",
    output_file_path: Optional[str] = None,
) -> BSS._SireWrappers._system.System:
    """
    Parameterise the ligand. In a classic FEP calculation, we should only have one ligand

    Parameters
    ----------
    file_path : str
        Path to the ligand coordinate file (e.g. ligand.sdf) containing the ligand molecule.
    forcefield : str
        Force field name to apply (e.g. "openff_unconstrained-2.0.0").
    output_file_path : Optional[str]
        If provided, the assembled System will be written to this path (e.g. "out/ligand.gro").
        The file format will be GROMACS (gro87 or grotop).

    Returns
    -------
    System
        A BioSimSpace System containing the assembled, parameterised ligand molecule.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ligand file not found: {file_path}")

    lig_sys = BSS.IO.readMolecules(file_path)
    logger.info(
        f"Parameterising {len(lig_sys)} ligand molecule(s) with forcefield={forcefield}",
    )

    # Parameterise each water and extract the molecule
    rename_lig(lig_sys, "LIG")  # Ensure that the ligand is named "LIG"
    parameterised = []
    for lig in lig_sys:
        lig_charge = round(lig.charge().value())
        if lig_charge != 0:
            logger.warning(
                f"Ligand {lig.__str__} has a charge of {lig_charge}. Co-alchemical ion approach will be used."
                " Ensure that your box is large enough to avoid artefacts."
            )
        param_args = {"molecule": lig, "forcefield": forcefield}
        # Only include ligand charge if we're using gaff (OpenFF doesn't need it)
        if "gaff" in forcefield:
            param_args["net_charge"] = lig_charge

        param_lig = BSS.Parameters.parameterise(**param_args).getMolecule()
        parameterised.append(param_lig)

    # Assemble into a single System
    system = parameterised[0].toSystem()
    for mol in parameterised[1:]:
        system += mol  # only works for different molecules

    if output_file_path:
        logger.info(f"Writing parameterised ligand system to {output_file_path}")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        BSS.IO.saveMolecules(
            str(output_file_path), system, fileformat=["gro87", "grotop"]
        )

    return system


def _parameterise_protein(
    file_path: str, forcefield: str = "ff14SB", output_file_path: Optional[str] = None
) -> BSS._SireWrappers._system.System:
    """
    Parameterise the protein. In a classic FEP calculation, we should only have one protein

    Parameters
    ----------
    file_path : str
        Path to the protein coordinate file (e.g. "protein.pdb") containing the protein molecule.
    forcefield : str
        Force field name to apply (e.g. "ff14SB").
    output_file_path : Optional[str]
        If provided, the assembled System will be written to this path (e.g. "out/protein.gro").

    Returns
    -------
    System
        A BioSimSpace System containing the assembled, parameterised protein molecule.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Protein file not found: {file_path}")

    protein_sys = BSS.IO.readMolecules(file_path)
    logger.info(
        f"Parameterising {len(protein_sys)} protein molecule(s) with forcefield={forcefield}",
    )

    # Parameterise each water and extract the molecule
    parameterised = [
        BSS.Parameters.parameterise(
            molecule=protein,
            forcefield=forcefield,
        ).getMolecule()
        for protein in protein_sys
    ]

    # Assemble into a single System
    system = parameterised[0].toSystem()
    for mol in parameterised[1:]:
        system += mol  # only works for different molecules

    if output_file_path:
        logger.info(f"Writing parameterised protein system to {output_file_path}")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        BSS.IO.saveMolecules(
            str(output_file_path), system, fileformat=["gro87", "grotop"]
        )

    return system


def parameterise_system(
    *,
    protein_path: Optional[str] = None,
    ligand_path: Optional[str] = None,
    water_path: Optional[str] = None,
    protein_ff: str = "ff14SB",
    ligand_ff: str = "openff_unconstrained-2.0.0",
    water_ff: str = "ff14SB",
    water_model: str = "tip3p",
    output_file_path: Optional[str] = None,
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Parameterise and assemble a combined System of any subset of protein, ligand and water.
    must pass at least one of `protein_path`, `ligand_path` or `water_path`.

    Parameters
    ----------
    protein_path: str
        PDB file for the protein (e.g. "protein.pdb").
    ligand_path: str
        SDF (or other) file for the ligand (e.g. "ligand.sdf").
    water_path: str
        PDB file for crystal waters (e.g. "water.pdb").
    protein_ff: str
        Force field for the protein (default "ff14SB").
    ligand_ff: str
        Force field for the ligand (default "openff_unconstrained-2.0.0").
    water_ff: str
        Force field for waters (default "ff14SB").
    water_model: str
        Water model identifier (default "tip3p").
    output_file_path: str
        If provided, writes the combined system out in GROMACS formats.

    Returns
    -------
    System
        The assembled, parameterised BioSimSpace System.
    """
    components = []

    if protein_path:
        prot_sys = _parameterise_protein(
            file_path=protein_path,
            forcefield=protein_ff,
        )
        components.append(prot_sys)

    if ligand_path:
        lig_sys = _parameterise_ligand(
            file_path=ligand_path,
            forcefield=ligand_ff,
        )
        components.append(lig_sys)

    if water_path:
        wat_sys = _parameterise_water(
            file_path=water_path,
            forcefield=water_ff,
            water_model=water_model,
        )
        components.append(wat_sys)

    if not components:
        raise ValueError(
            "Must supply at least one of protein_path, ligand_path or water_path!"
        )

    # Assemble into one System
    full_system = components[0]
    for sys_part in components[1:]:
        full_system += sys_part  # only works for different molecules

    if output_file_path:
        logger.info(f"Writing assembled system to {output_file_path}")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        BSS.IO.saveMolecules(
            str(output_file_path), full_system, fileformat=["gro87", "grotop"]
        )

    return full_system


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
    # Taken from https://github.com/michellab/BioSimSpaceTutorials/blob/main/01_introduction/02_molecular_setup.ipynb
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


def solvate_system(
    file_path: str,
    water_model: str = "tip3p",
    output_file_path: Optional[str] = None,
    **solvate_kwargs,
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Determine an appropriate (rhombic dodecahedron)
    box size, then solvate the input structure using
    TIP3P water, adding 150 mM NaCl to the system.
    The resulting system is saved to the input directory.

    Parameters
    ----------
    file_path : str
        Path to the input structure file.
    water_model : str
        Water model to use for solvation (e.g. "tip3p")
    output_file_path : str
        file to the output directory where the solvated system will be saved.
    **solvate_kwargs : dict
        Additional keyword arguments to pass to the solvation function.

    Returns
    -------
    solvated_system : BSS._SireWrappers._system.System
        Solvated system.
    """
    base_system = BSS.IO.readMolecules(file_path)  # type: ignore

    solvated_system = _solvate_molecular_system_bss(
        system=base_system, water_model=water_model, **solvate_kwargs
    )

    if output_file_path:
        logger.info(f"Writing solvated system to {output_file_path}")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        BSS.IO.saveMolecules(
            str(output_file_path), solvated_system, fileformat=["gro87", "grotop"]
        )

    return solvated_system


def minimise_system(
    file_path: str,
    output_file_path: str,
    min_steps: int=1_000,
    slurm: bool = True,
    mdrun_options: Optional[str] = None,
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Minimise the input structure with GROMACS.

    Parameters
    ----------
    file_path : str
        Path to the input structure file.
    output_file_path : str
        Path to the output directory where the minimised system will be saved.
    min_steps : int
        Number of minimisation steps to perform.
    slurm : bool
        Whether to use SLURM for the minimisation.
    mdrun_options : str, optional
        Additional options to pass to the GROMACS mdrun command.

    Returns
    -------
    minimised_system : BSS._SireWrappers._system.System
        Minimised system.
    """
    solvated_system = BSS.IO.readMolecules(list(file_path))  # returns a System
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
            str(output_file_path), minimised_system, fileformat=["gro87", "grotop"],
            property_map={"velocity": "foo"},
        )

    return minimised_system


def heat_and_preequil_input(
    leg_type: LegType,
    input_dir: str,
    output_dir: str,
    slurm: bool = True,
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Heat the input structure from 0 to 298.15 K with GROMACS.

    Parameters
    ----------
    leg_type : LegType
        The type of the leg.
    input_dir : str
        The path to the input directory, where the required files are located.
    output_dir : str
        The path to the output directory, where the files will be saved.

    Returns
    -------
    preequilibrated_system : BSS._SireWrappers._system.System
        Pre-Equilibrated system.
    """
    cfg = SystemPreparationConfig.from_pickle(input_dir, leg_type)

    # Load the minimised system
    logger.info("Loading minimised system...")
    minimised_system = BSS.IO.readMolecules(
        [
            f"{input_dir}/{file}"
            for file in PreparationStage.MINIMISED.get_simulation_input_files(leg_type)
        ]
    )

    # Check that it is solvated and has a box
    check_has_wat_and_box(minimised_system)

    logger.info(
        f"NVT equilibration for {cfg.runtime_short_nvt} ps while restraining all non-solvent atoms"
    )
    protocol = BSS.Protocol.Equilibration(
        runtime=cfg.runtime_short_nvt * BSS.Units.Time.picosecond,
        temperature_start=0 * BSS.Units.Temperature.kelvin,
        temperature_end=cfg.end_temp * BSS.Units.Temperature.kelvin,
        restraint="all",
    )
    equil1 = run_process(system=minimised_system, protocol=protocol, leg_type=leg_type)

    # If this is the bound leg, carry out step with backbone restraints
    if leg_type == LegType.BOUND:
        logger.info(
            f"NVT equilibration for {cfg.runtime_nvt} ps while restraining all backbone atoms"
        )
        protocol = BSS.Protocol.Equilibration(
            runtime=cfg.runtime_nvt * BSS.Units.Time.picosecond,
            temperature=cfg.end_temp * BSS.Units.Temperature.kelvin,
            restraint="backbone",
        )
        equil2 = run_process(system=equil1, protocol=protocol, leg_type=leg_type)

    else:  # Free leg - skip the backbone restraint step
        equil2 = equil1

    logger.info(f"NVT equilibration for {cfg.runtime_nvt} ps without restraints")
    protocol = BSS.Protocol.Equilibration(
        runtime=cfg.runtime_nvt * BSS.Units.Time.picosecond,
        temperature=cfg.end_temp * BSS.Units.Temperature.kelvin,
    )
    equil3 = run_process(system=equil2, protocol=protocol, leg_type=leg_type)

    logger.info(
        f"NPT equilibration for {cfg.runtime_npt} ps while restraining non-solvent heavy atoms"
    )
    protocol = BSS.Protocol.Equilibration(
        runtime=cfg.runtime_npt * BSS.Units.Time.picosecond,
        pressure=1 * BSS.Units.Pressure.atm,
        temperature=cfg.end_temp * BSS.Units.Temperature.kelvin,
        restraint="heavy",
    )
    equil4 = run_process(system=equil3, protocol=protocol, leg_type=leg_type)

    logger.info(
        f"NPT equilibration for {cfg.runtime_npt_unrestrained} ps without restraints"
    )
    protocol = BSS.Protocol.Equilibration(
        runtime=cfg.runtime_npt_unrestrained * BSS.Units.Time.picosecond,
        pressure=1 * BSS.Units.Pressure.atm,
        temperature=cfg.end_temp * BSS.Units.Temperature.kelvin,
    )
    preequilibrated_system = run_process(
        system=equil4, protocol=protocol, leg_type=leg_type
    )

    # Save, renaming the velocity property to foo so avoid saving velocities. Saving the
    # velocities sometimes causes issues with the size of the floats overflowing the RST7
    # format.
    BSS.IO.saveMolecules(
        f"{output_dir}/{leg_type.name.lower()}{PreparationStage.PREEQUILIBRATED.file_suffix}",
        preequilibrated_system,
        fileformat=["gro87", "grotop"],
        property_map={"velocity": "foo"},
    )

    return preequilibrated_system


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


class SystemPreparationConfig(BaseModel):
    """
    Pydantic model for holding system preparation configuration.

    Attributes
    ----------
    slurm: bool
        Whether to use SLURM for the preparation.
    forcefields : dict
        Forcefields to use for the ligand, protein, and water.
    water_model : str
        Water model to use.
    ion_conc : float
        Ion concentration in M.
    steps : int
        Number of steps for the minimisation.
    runtime_short_nvt : int
        Runtime for the short NVT equilibration in ps.
    runtime_nvt : int
        Runtime for the NVT equilibration in ps.
    end_temp : float
        End temperature for the NVT equilibration in K.
    runtime_npt : int
        Runtime for the NPT equilibration in ps.
    runtime_npt_unrestrained : int
        Runtime for the unrestrained NPT equilibration in ps.
    ensemble_equilibration_time : int
        Ensemble equilibration time in ps.
    append_to_ligand_selection: str
        If this is a bound leg, this appends the supplied string to the default atom
        selection which chooses the atoms in the ligand to consider as potential anchor
        points. The default atom selection is f'resname {ligand_resname} and not name H*'.
        Uses the mdanalysis atom selection language. For example, 'not name O*' will result
        in an atom selection of f'resname {ligand_resname} and not name H* and not name O*'.
    use_same_restraints: bool
        If True, the same restraints will be used for all of the bound leg repeats - by default
        , the restraints generated for the first repeat are used. This allows meaningful
        comparison between repeats for the bound leg. If False, the unique restraints are
        generated for each repeat.
    """

    slurm: bool = Field(True)
    forcefields: dict = {
        "ligand": "openff_unconstrained-2.0.0",
        "protein": "ff14SB",
        "water": "tip3p",
    }
    water_model: str = "tip3p"
    ion_conc: float = Field(0.15, ge=0, lt=1)  # M
    steps: int = Field(1000, gt=0, lt=100_000)  # This is the default for BSS
    runtime_short_nvt: int = Field(5, gt=0, lt=500)  # ps
    runtime_nvt: int = Field(50, gt=0, lt=5_000)  # ps
    end_temp: float = Field(298.15, gt=0, lt=350)  # K
    runtime_npt: int = Field(400, gt=0, lt=40_000)  # ps
    runtime_npt_unrestrained: int = Field(1000, gt=0, lt=100_000)  # ps
    ensemble_equilibration_time: int = Field(5000, gt=0, lt=50_000)  # ps
    append_to_ligand_selection: str = Field(
        "",
        description="Atom selection to append to the ligand selection during restraint searching.",
    )
    use_same_restraints: bool = Field(
        True,
        description="Whether to use the same restraints for all repeats of the bound leg. Note "
        "that this should be used if you plan to run adaptively.",
    )
    # Added by JJ-2025-05-05
    mdrun_options: Optional[str] = Field(
        None, description="Extra flags for 'gmx mdrun' (e.g., '-ntmpi 1 -ntomp 8')."
    )
    lambda_values: dict = {
        LegType.BOUND: {
            StageType.RESTRAIN: [0.0, 0.125, 0.25, 0.375, 0.5, 1.0],
            StageType.DISCHARGE: [0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0],
            StageType.VANISH: [
                0.0,
                0.025,
                0.05,
                0.075,
                0.1,
                0.125,
                0.15,
                0.175,
                0.2,
                0.225,
                0.25,
                0.275,
                0.3,
                0.325,
                0.35,
                0.375,
                0.4,
                0.425,
                0.45,
                0.475,
                0.5,
                0.525,
                0.55,
                0.575,
                0.6,
                0.625,
                0.65,
                0.675,
                0.7,
                0.725,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
            ],
        },
        LegType.FREE: {
            StageType.DISCHARGE: [0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0],
            StageType.VANISH: [
                0.0,
                0.028,
                0.056,
                0.111,
                0.167,
                0.222,
                0.278,
                0.333,
                0.389,
                0.444,
                0.5,
                0.556,
                0.611,
                0.667,
                0.722,
                0.778,
                0.889,
                1.0,
            ],
        },
    }

    class Config:
        """
        Pydantic model configuration.
        """

        extra = "forbid"
        validate_assignment = True

    def get_tot_simtime(self, n_runs: int, leg_type: LegType) -> int:
        """
        Get the total simulation time for the ensemble equilibration.

        Parameters
        ----------
        n_runs : int
            Number of ensemble equilibration runs.
        leg_type : LegType
            The type of the leg.

        Returns
        -------
        int
            Total simulation time in ps.
        """

        # See functions below for where these times are used.
        tot_simtime = 0
        tot_simtime += self.runtime_short_nvt
        tot_simtime += (
            self.runtime_nvt * 2 if leg_type == LegType.BOUND else self.runtime_nvt
        )
        tot_simtime += self.runtime_npt * 2
        tot_simtime += self.runtime_npt_unrestrained
        tot_simtime += self.ensemble_equilibration_time * n_runs
        return tot_simtime

    def save_pickle(self, save_dir: str, leg_type: LegType) -> None:
        """
        Save the configuration to a pickle file.

        Parameters
        ----------
        save_dir : str
            Directory to save the pickle file to.

        leg_type : LegType
            The type of the leg. Used to name the pickle file.
        """
        # First, convert to dict
        model_dict = self.model_dump()

        # Save the dict to a pickle file
        save_path = save_dir + "/" + self.get_file_name(leg_type)
        with open(save_path, "wb") as f:
            _pkl.dump(model_dict, f)

    @classmethod
    def from_pickle(cls, save_dir: str, leg_type: LegType) -> "SystemPreparationConfig":
        """
        Load the configuration from a pickle file.

        Parameters
        ----------
        save_dir : str
            Directory to load the pickle file from.

        leg_type : LegType
            The type of the leg. Used to decide the name of the pickle
            file to load.

        Returns
        -------
        SystemPreparationConfig
            Loaded configuration.
        """

        # Load the dict from the pickle file
        load_path = save_dir + "/" + cls.get_file_name(leg_type)
        with open(load_path, "rb") as f:
            model_dict = _pkl.load(f)

        # Create the model from the dict
        return cls.parse_obj(model_dict)

    @staticmethod
    def get_file_name(leg_type: LegType) -> str:
        """Get the name of the pickle file for the configuration."""
        return f"system_preparation_config_{leg_type.name.lower()}.pkl"
