import logging
import os
import pickle as _pkl
import time
from typing import Any, Callable, Optional, Union

import BioSimSpace.Sandpit.Exscientia as BSS  # type: ignore[import]
from BioSimSpace.Sandpit.Exscientia._SireWrappers import Molecule  # type: ignore[import]
from pydantic import BaseModel, Field
from sire.legacy import Mol as SireMol  # type: ignore[import]

from binding_affinity_predicting.components.utils import check_has_wat_and_box
from binding_affinity_predicting.schemas.enums import LegType, StageType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    ion_conc: float = Field(0.15, ge=0, lt=1)  # unit: M or mol/L
    energy_min_steps: int = Field(1000, gt=0, lt=100_000)  # This is the default for BSS
    preequilibration_steps: list[EquilStep] = Field(
        default_factory=lambda: [
            EquilStep(5, 0.0, 298.15, "all", None),
            EquilStep(10, 298.15, 298.15, "backbone", None),
            EquilStep(10, 298.15, 298.15, None, None),
            EquilStep(10, 298.15, 298.15, "heavy", 1.0),
            EquilStep(10, 298.15, 298.15, None, 1.0),
        ]
    )

    # runtime_short_nvt: int = Field(5, gt=0, lt=500)  # ps
    # runtime_nvt: int = Field(50, gt=0, lt=5_000)  # ps
    # end_temp: float = Field(298.15, gt=0, lt=350)  # K
    # runtime_npt: int = Field(400, gt=0, lt=40_000)  # ps
    # runtime_npt_unrestrained: int = Field(1000, gt=0, lt=100_000)  # ps
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
