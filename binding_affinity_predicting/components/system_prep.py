import pathlib as _pathlib
import pickle as _pkl
import warnings as _warnings
from typing import Optional as _Optional

import BioSimSpace.Sandpit.Exscientia as _BSS
from pydantic import BaseModel as _BaseModel
from pydantic import Field as _Field


import BioSimSpace.Sandpit.Exscientia as _BSS
from BioSimSpace.Sandpit.Exscientia._SireWrappers import Molecule as _Molecule
from sire.legacy import Mol as _SireMol
from binding_affinity_predicting.schemas.enums import (
    LegType,
    StageType,
    PreparationStage,
)


def rename_lig(
    bss_system: _BSS._SireWrappers._system.System, new_name: str = "LIG"
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
    mol = _Molecule(bss_system[0])
    mol_sire = mol._sire_object

    # Create an editable version of the sire object
    mol_edit = mol_sire.edit()

    # Rename the molecule and the residue to the supplied name
    resname = _SireMol.ResName(new_name)  # type: ignore
    mol_edit = mol_edit.residue(_SireMol.ResIdx(0)).rename(resname).molecule()  # type: ignore
    mol_edit = mol_edit.edit().rename(new_name).molecule()

    # Commit the changes and update the system
    mol._sire_object = mol_edit.commit()
    bss_system.updateMolecule(0, mol)


class SystemPreparationConfig(_BaseModel):
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

    slurm: bool = _Field(True)
    forcefields: dict = {
        "ligand": "openff_unconstrained-2.0.0",
        "protein": "ff14SB",
        "water": "tip3p",
    }
    water_model: str = "tip3p"
    ion_conc: float = _Field(0.15, ge=0, lt=1)  # M
    steps: int = _Field(1000, gt=0, lt=100_000)  # This is the default for _BSS
    runtime_short_nvt: int = _Field(5, gt=0, lt=500)  # ps
    runtime_nvt: int = _Field(50, gt=0, lt=5_000)  # ps
    end_temp: float = _Field(298.15, gt=0, lt=350)  # K
    runtime_npt: int = _Field(400, gt=0, lt=40_000)  # ps
    runtime_npt_unrestrained: int = _Field(1000, gt=0, lt=100_000)  # ps
    ensemble_equilibration_time: int = _Field(5000, gt=0, lt=50_000)  # ps
    append_to_ligand_selection: str = _Field(
        "",
        description="Atom selection to append to the ligand selection during restraint searching.",
    )
    use_same_restraints: bool = _Field(
        True,
        description="Whether to use the same restraints for all repeats of the bound leg. Note "
        "that this should be used if you plan to run adaptively.",
    )
    # Added by JJ-2025-05-05
    mdrun_options: _Optional[str] = _Field(
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
    

def parameterise_input(leg_type: LegType, input_dir: str, output_dir: str,
) -> _BSS._SireWrappers._system.System:  # type: ignore
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
    parameterised_system : _BSS._SireWrappers._system.System
        Parameterised system.
    """
    cfg = SystemPreparationConfig.from_pickle(input_dir, leg_type)

    print("Parameterising input...")
    # Parameterise the ligand
    print("Parameterising ligand...")
    lig_sys = _BSS.IO.readMolecules(f"{input_dir}/ligand.sdf")
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

    param_lig = _BSS.Parameters.parameterise(**param_args).getMolecule()

    # If bound, then parameterise the protein and waters and add to the system
    if leg_type == LegType.BOUND:
        # Parameterise the protein
        print("Parameterising protein...")
        protein = _BSS.IO.readMolecules(f"{input_dir}/protein.pdb")[0]
        param_protein = _BSS.Parameters.parameterise(
            molecule=protein, forcefield=cfg.forcefields["protein"]
        ).getMolecule()

        # Parameterise the waters, if they are supplied
        # Check that waters are supplied
        param_waters = []
        if _pathlib.Path(f"{input_dir}/waters.pdb").exists():
            print("Crystallographic waters detected. Parameterising...")
            waters = _BSS.IO.readMolecules(f"{input_dir}/waters.pdb")
            for water in waters:
                param_waters.append(
                    _BSS.Parameters.parameterise(
                        molecule=water,
                        water_model=cfg.forcefields["water"],
                        forcefield=cfg.forcefields["protein"],
                    ).getMolecule()
                )

        # Create the system
        print("Assembling parameterised system...")
        parameterised_system = param_lig + param_protein
        for water in param_waters:
            parameterised_system += water

    # This is the free leg, so just turn the ligand into a system
    else:
        parameterised_system = param_lig.toSystem()

    # Save the system
    print("Saving parameterised system...")
    _BSS.IO.saveMolecules(
        f"{output_dir}/{leg_type.name.lower()}{PreparationStage.PARAMETERISED.file_suffix}",
        parameterised_system,
        fileformat=["gro87", "grotop"],
    )

    return parameterised_system