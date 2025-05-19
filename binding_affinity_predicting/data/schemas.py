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
from binding_affinity_predicting.data.enums import LegType, StageType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RunnerConfig:
    dg_multiplier: int = 1
    ensemble_size: int = 5
    dump: bool = True


class SystemPreparationConfig(BaseModel):
    """
    Pydantic model for holding system preparation configuration.
    """

    forcefields: dict = Field(
        default_factory=lambda: {
            "ligand": "openff_unconstrained-2.0.0",
            "protein": "ff14SB",
            "water": "tip3p",
        }
    )
    water_model: str = "tip3p"
    ion_conc: float = Field(0.15, ge=0, lt=1)  # Mnit: M or mol/L

    class Config:
        extra = "forbid"
        validate_assignment = True


class EquilStep(BaseModel):
    """
    One NVT/NPT equilibration step.
    Units: runtime (ps), temperature (K), restraint (all|backbone|heavy), pressure (atm).
    """

    runtime: float
    temperature_start: float
    temperature_end: float
    restraint: Optional[str]
    pressure: Optional[float]

    class Config:
        extra = "forbid"
        validate_assignment = True


class PreEquilibrationConfig(BaseModel):
    """Config block for all pre-equilibration steps."""

    steps: list[EquilStep] = Field(
        default_factory=lambda: [
            EquilStep(
                runtime=5,
                temperature_start=0.0,
                temperature_end=298.15,
                restraint="all",
                pressure=None,
            ),
            EquilStep(
                runtime=10,
                temperature_start=298.15,
                temperature_end=298.15,
                restraint="backbone",
                pressure=None,
            ),
            EquilStep(
                runtime=10,
                temperature_start=298.15,
                temperature_end=298.15,
                restraint=None,
                pressure=None,
            ),
            EquilStep(
                runtime=10,
                temperature_start=298.15,
                temperature_end=298.15,
                restraint="heavy",
                pressure=1.0,
            ),
            EquilStep(
                runtime=10,
                temperature_start=298.15,
                temperature_end=298.15,
                restraint=None,
                pressure=1.0,
            ),
        ]
    )

    class Config:
        extra = "forbid"
        validate_assignment = True


class EnergyMinimisationConfig(BaseModel):
    """Config block for energy minimisation."""

    steps: int = Field(1000, gt=0, lt=100_000)

    class Config:
        extra = "forbid"
        validate_assignment = True


class FepSimulationConfig(BaseModel):
    """
    Configuration for the λ‐schedule used in bound and free legs.
    """

    lambda_values: dict[LegType, dict[StageType, list[float]]] = Field(
        default_factory=lambda: {
            LegType.BOUND: {
                StageType.RESTRAIN: [0.0, 0.125, 0.25, 0.375, 0.5, 1.0],
                StageType.DISCHARGE: [
                    0.0,
                    0.143,
                    0.286,
                    0.429,
                    0.571,
                    0.714,
                    0.857,
                    1.0,
                ],
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
                StageType.DISCHARGE: [
                    0.0,
                    0.143,
                    0.286,
                    0.429,
                    0.571,
                    0.714,
                    0.857,
                    1.0,
                ],
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
    )
    append_to_ligand_selection: str = Field(
        "",
        description="Atom selection to append to the ligand selection during restraint searching.",
    )
    use_same_restraints: bool = Field(
        True,
        description="Whether to use the same restraints for all repeats of the bound leg. Note "
        "that this should be used if you plan to run adaptively.",
    )
    ensemble_equilibration_time: int = Field(5000, gt=0, lt=50_000)  # ps

    class Config:
        extra = "forbid"
        validate_assignment = True


class WorkflowConfig(BaseModel):
    """Top-level configuration for entire system prep workflow."""

    slurm: bool = True
    param_system_prep: SystemPreparationConfig = Field(
        default_factory=SystemPreparationConfig
    )
    param_preequilibration: PreEquilibrationConfig = Field(
        default_factory=PreEquilibrationConfig
    )
    param_energy_minimisation: EnergyMinimisationConfig = Field(
        default_factory=EnergyMinimisationConfig
    )
    param_fep_params: FepSimulationConfig = Field(default_factory=FepSimulationConfig)

    # Added by JJ-2025-05-05
    mdrun_options: Optional[str] = Field(
        None, description="Extra flags for 'gmx mdrun' (e.g., '-ntmpi 1 -ntomp 8')."
    )

    class Config:
        extra = "forbid"
        validate_assignment = True

    def get_tot_simtime(self, n_runs: int, leg_type: LegType) -> float:
        raise NotImplementedError("...")
