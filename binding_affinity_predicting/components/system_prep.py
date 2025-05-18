from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.schemas.enums import LegType
import logging
# from binding_affinity_predicting.components.system_prep import SystemPreparationConfig
from BioSimSpace.Sandpit.Exscientia._SireWrappers import Molecule as Molecule
from sire.legacy import Mol as SireMol
from binding_affinity_predicting.schemas.enums import (
    LegType,
    StageType,
    PreparationStage,
)
import pathlib as _pathlib
import pickle as _pkl
import warnings as _warnings
from typing import Optional

import BioSimSpace.Sandpit.Exscientia as BSS
from pydantic import BaseModel
from pydantic import Field
import os
from binding_affinity_predicting.components.utils import check_has_wat_and_box


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def parameterise_input(leg_type: LegType, input_dir: str, output_dir: str,
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Paramaterise the input structure, using Open Force Field v.2.0 'Sage'
    for the ligand, AMBER ff14SB for the protein, and TIP3P for the water.
    The resulting system is saved to the input directory.

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
    parameterised_system : BSS._SireWrappers._system.System
        Parameterised system.
    """
    cfg = SystemPreparationConfig.from_pickle(input_dir, leg_type)

    logger.info("Parameterising input...")
    # Parameterise the ligand
    logger.info("Parameterising ligand...")
    lig_sys = BSS.IO.readMolecules(f"{input_dir}/ligand.sdf")
    # Ensure that the ligand is named "LIG"
    rename_lig(lig_sys, "LIG")
    # Check charge of the ligand
    lig = lig_sys[0]
    lig_charge = round(lig.charge().value())
    if lig_charge != 0:
        _warnings.warn(
            f"Ligand has a charge of {lig_charge}. Co-alchemical ion approach will be used."
            " Ensure that your box is large enough to avoid artefacts."
        )

    # Only include ligand charge if we're using gaff (OpenFF doesn't need it)
    param_args = {"molecule": lig, "forcefield": cfg.forcefields["ligand"]}
    if "gaff" in cfg.forcefields["ligand"]:
        param_args["net_charge"] = lig_charge

    param_lig = BSS.Parameters.parameterise(**param_args).getMolecule()

    # If bound, then parameterise the protein and waters and add to the system
    if leg_type == LegType.BOUND:
        # Parameterise the protein
        logger.info("Parameterising protein...")
        protein = BSS.IO.readMolecules(f"{input_dir}/protein.pdb")[0]
        param_protein = BSS.Parameters.parameterise(
            molecule=protein, forcefield=cfg.forcefields["protein"]
        ).getMolecule()

        # Parameterise the waters, if they are supplied
        # Check that waters are supplied
        param_waters = []
        if _pathlib.Path(f"{input_dir}/waters.pdb").exists():
            logger.info("Crystallographic waters detected. Parameterising...")
            waters = BSS.IO.readMolecules(f"{input_dir}/waters.pdb")
            for water in waters:
                param_waters.append(
                    BSS.Parameters.parameterise(
                        molecule=water,
                        water_model=cfg.forcefields["water"],
                        forcefield=cfg.forcefields["protein"],
                    ).getMolecule()
                )

        # Create the system
        logger.info("Assembling parameterised system...")
        parameterised_system = param_lig + param_protein
        for water in param_waters:
            parameterised_system += water

    # This is the free leg, so just turn the ligand into a system
    else:
        parameterised_system = param_lig.toSystem()

    # Save the system
    logger.info("Saving parameterised system...")
    BSS.IO.saveMolecules(
        f"{output_dir}/{leg_type.name.lower()}{PreparationStage.PARAMETERISED.file_suffix}",
        parameterised_system,
        fileformat=["gro87", "grotop"],   # GROMACS fileformat must be something like gro87 for saving 
    )

    return parameterised_system


def solvate_input(
    leg_type: LegType, input_dir: str, output_dir: str
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Determine an appropriate (rhombic dodecahedron)
    box size, then solvate the input structure using
    TIP3P water, adding 150 mM NaCl to the system.
    The resulting system is saved to the input directory.

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
    solvated_system : BSS._SireWrappers._system.System
        Solvated system.
    """
    cfg = SystemPreparationConfig.from_pickle(input_dir, leg_type)

    # Load the parameterised system
    logger.info("Loading parameterised system...")
    parameterised_system = BSS.IO.readMolecules(
        [
            f"{input_dir}/{file}"
            for file in PreparationStage.PARAMETERISED.get_simulation_input_files(
                leg_type
            )
        ]
    )

    # Determine the box size
    # Taken from https://github.com/michellab/BioSimSpaceTutorials/blob/main/01_introduction/02_molecular_setup.ipynb
    # Get the minimium and maximum coordinates of the bounding box that
    # minimally encloses the protein.
    logger.info("Determining optimal rhombic dodecahedral box...")
    # Want to get box size based on complex/ ligand, exlcuding any crystallographic waters
    non_waters = [mol for mol in parameterised_system if mol.nAtoms() != 3]  # type: ignore
    dry_system = BSS._SireWrappers._system.System(non_waters)  # type: ignore
    box_min, box_max = dry_system.getAxisAlignedBoundingBox()

    # Work out the box size from the difference in the coordinates.
    box_size = [y - x for x, y in zip(box_min, box_max)]

    # Add 15 A padding to the box size in each dimension.
    padding = 15 * BSS.Units.Length.angstrom

    # Work out an appropriate box. This will used in each dimension to ensure
    # that the cutoff constraints are satisfied if the molecule rotates.
    box_length = max(box_size) + 2 * padding
    box, angles = BSS.Box.rhombicDodecahedronHexagon(box_length)

    # Exclude waters if they are too far from the protein. These are unlikely
    # to be important for the simulation and including them would require a larger
    # box. Exclude if further than 10 A from the protein.
    try:
        waters_to_exclude = [
            wat
            for wat in parameterised_system.search(
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
    parameterised_system.removeMolecules(waters_to_exclude)

    logger.info(f"Solvating system with {cfg.water_model} water and {cfg.ion_conc} M NaCl...")
    solvated_system = BSS.Solvent.solvate(
        model=cfg.water_model,
        molecule=parameterised_system,
        box=box,
        angles=angles,
        ion_conc=cfg.ion_conc,
    )

    # Save the system
    logger.info("Saving solvated system")
    BSS.IO.saveMolecules(
        f"{output_dir}/{leg_type.name.lower()}{PreparationStage.SOLVATED.file_suffix}",
        solvated_system,
        fileformat=["gro87", "grotop"],
    )

    return solvated_system


def minimise_input(
    leg_type: LegType, input_dir: str, output_dir: str, slurm: bool = True,
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Minimise the input structure with GROMACS.

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
    minimised_system : BSS._SireWrappers._system.System
        Minimised system.
    """
    cfg = SystemPreparationConfig.from_pickle(input_dir, leg_type)

    # Load the solvated system
    logger.info("Loading solvated system...")
    solvated_system = BSS.IO.readMolecules(
        [
            f"{input_dir}/{file}"
            for file in PreparationStage.SOLVATED.get_simulation_input_files(leg_type)
        ]
    )

    # Check that it is actually solvated in a box of water
    check_has_wat_and_box(solvated_system)

    # Minimise
    logger.info(f"Minimising input structure with {cfg.steps} steps...")
    protocol = BSS.Protocol.Minimisation(steps=cfg.steps)
    minimised_system = run_process(system=solvated_system, protocol=protocol, leg_type=leg_type)

    # Save, renaming the velocity property to foo so avoid saving velocities. Saving the
    # velocities sometimes causes issues with the size of the floats overflowing the RST7
    # format.
    BSS.IO.saveMolecules(
        f"{output_dir}/{leg_type.name.lower()}{PreparationStage.MINIMISED.file_suffix}",
        minimised_system,
        fileformat=["gro87", "grotop"],
        property_map={"velocity": "foo"},
    )

    return minimised_system



def heat_and_preequil_input(
    leg_type: LegType, input_dir: str, output_dir: str,  slurm: bool = True,
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

    logger.info(f"NPT equilibration for {cfg.runtime_npt_unrestrained} ps without restraints")
    protocol = BSS.Protocol.Equilibration(
        runtime=cfg.runtime_npt_unrestrained * BSS.Units.Time.picosecond,
        pressure=1 * BSS.Units.Pressure.atm,
        temperature=cfg.end_temp * BSS.Units.Temperature.kelvin,
    )
    preequilibrated_system = run_process(system=equil4, protocol=protocol, leg_type=leg_type)

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
    leg_type: LegType,
    work_dir: Optional[str] = None,
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

    Returns
    -------
    system : BSS._SireWrappers._system.System
        System after the process has been run.
    """
    process = BSS.Process.Gromacs(system, protocol, work_dir=work_dir)

    # Added by JJ-2025-05-05 for local run on Mac
    cfg = SystemPreparationConfig.from_pickle(work_dir or "./input", leg_type)
    mdrun_extra_args = cfg.mdrun_options.split()
    for flag, value in zip(mdrun_extra_args[::2], mdrun_extra_args[1::2]):
        process.setArg(flag, value)
    
    process.start()
    process.wait()
    import time

    time.sleep(10)
    if process.isError():
        logger.info(process.stdout())
        logger.info(process.stderr())
        process.getStdout()
        process.getStderr()
        raise BSS._Exceptions.ThirdPartyError("The process failed.")
    system = process.getSystem(block=True)
    if system is None:
        logger.info(process.stdout())
        logger.info(process.stderr())
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
    lambda_values: dict = {LegType.BOUND: {StageType.RESTRAIN: [0.0, 0.125, 0.25, 0.375, 0.5, 1.0], 
                                           StageType.DISCHARGE: [0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0], 
                                           StageType.VANISH: [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]}, 
                            LegType.FREE: {StageType.DISCHARGE: [0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0], 
                                           StageType.VANISH: [0.0, 0.028, 0.056, 0.111, 0.167, 0.222, 0.278, 0.333, 0.389, 0.444, 0.5, 0.556, 0.611, 0.667, 0.722, 0.778, 0.889, 1.0]}}


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
    def from_pickle(
        cls, save_dir: str, leg_type: LegType
    ) -> "SystemPreparationConfig":
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
