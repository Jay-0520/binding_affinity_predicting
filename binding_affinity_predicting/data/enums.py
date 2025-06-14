"""Enums required for Classes in the run package."""

from enum import Enum
from typing import List as _List

__all__ = [
    "JobStatus",
    "StageType",
    "LegType",
    "PreparationStage",
]


class JobStatus(int, Enum):
    """An enumeration of the possible job statuses"""

    NONE = 0
    QUEUED = 1
    FINISHED = 2
    FAILED = 3
    KILLED = 4
    RUNNING = 5


class GromacsLambdaMdpTemplate(str, Enum):
    """Enumeration of the GROMACS mdp templates for different lambda states."""

    TEMPLATE = "lambda.template.mdp"


class SimulationRestraint(str, Enum):
    ALL = "all"
    BACKBONE = "backbone"
    HEAVY = "heavy"


class StageType(int, Enum):
    """Enumeration of the types of stage."""

    RESTRAIN = 1
    DISCHARGE = 2
    VANISH = 3

    @property
    def bss_perturbation_type(self) -> str:
        """Return the corresponding BSS perturbation type."""
        if self == StageType.RESTRAIN:
            return "restraint"
        elif self == StageType.DISCHARGE:
            return "discharge_soft"
        elif self == StageType.VANISH:
            return "vanish_soft"
        else:
            raise ValueError("Unknown stage type.")


class LegType(int, Enum):
    """The type of leg in the calculation."""

    BOUND = 1
    FREE = 2

    @property
    def name(self) -> str:
        """Return the name of the leg type."""
        if self == LegType.BOUND:
            return "bound"
        elif self == LegType.FREE:
            return "free"
        else:
            raise ValueError(f"Unknown leg type: {self}")


class PreparationStage(int, Enum):
    """The stage of preparation of the input files."""

    STRUCTURES_ONLY = 1
    PARAMETERISED = 2
    SOLVATED = 3
    MINIMISED = 4
    PREEQUILIBRATED = 5
    EQUILIBRATED = 6

    @property
    def file_suffix(self) -> str:
        """Return the suffix to use for the files in this stage."""
        if self == PreparationStage.STRUCTURES_ONLY:
            return ""
        elif self == PreparationStage.PARAMETERISED:
            return ""
        elif self == PreparationStage.SOLVATED:
            return "_solvated"
        elif self == PreparationStage.MINIMISED:
            return "_minimised"
        elif self == PreparationStage.PREEQUILIBRATED:
            return "_preequiled"
        elif self == PreparationStage.EQUILIBRATED:
            return "_equilibrated"
        else:
            raise ValueError(f"Unknown preparation stage: {self}")

    def get_simulation_input_files(self, leg_type: LegType) -> _List[str]:
        """Return the input files required for the simulation in this stage."""
        if self == PreparationStage.STRUCTURES_ONLY:
            if leg_type == LegType.BOUND:
                return [
                    "protein.pdb",
                    "ligand.sdf",
                ]  # Need sdf for parameterisation of lig
            elif leg_type == LegType.FREE:
                return ["ligand.sdf"]
        else:
            return [
                f"{leg_type.name.lower()}{self.file_suffix}.{file_type}"
                for file_type in ["gro", "top"]  # force to use GROMACS
            ]
