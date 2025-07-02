import logging
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, model_validator

from binding_affinity_predicting.components.data.enums import (
    LegType,
    SimulationRestraint,
    StageType,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RunnerConfig(BaseModel):
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
    water_model: str = Field("tip3p", description="Water model to use.")
    ion_conc: float = Field(0.15, ge=0, lt=1)

    class Config:
        extra = "forbid"
        validate_assignment = True


class PreEquilStageConfig(BaseModel):
    """
    One NVT/NPT equilibration step.
    Units: runtime (ps), temperature (K), restraint (all|backbone|heavy), pressure (atm).
    """

    runtime: float  # ps
    temperature_start: float
    temperature_end: float
    restraint: Optional[SimulationRestraint]  # restraint can be None or a string
    pressure: Optional[float]  # pressure can be float or a string

    class Config:
        extra = "forbid"
        validate_assignment = True


# TODO: maybe move this to somewhere else?
class EmpiricalPreEquilibrationConfig(BaseModel):
    """
    Config block for all pre-equilibration steps.

    This procedure is based on workflow from this paper:
    https://pubs.acs.org/doi/10.1021/acs.jctc.4c00806
    """

    steps: list[PreEquilStageConfig] = Field(
        default_factory=lambda: [
            PreEquilStageConfig(
                runtime=5,
                temperature_start=0.0,
                temperature_end=298.15,
                restraint=SimulationRestraint.ALL,
                pressure=None,
            ),
            PreEquilStageConfig(
                runtime=10,
                temperature_start=298.15,
                temperature_end=298.15,
                restraint=SimulationRestraint.BACKBONE,
                pressure=None,
            ),
            PreEquilStageConfig(
                runtime=10,
                temperature_start=298.15,
                temperature_end=298.15,
                restraint=None,
                pressure=None,
            ),
            PreEquilStageConfig(
                runtime=10,
                temperature_start=298.15,
                temperature_end=298.15,
                restraint=SimulationRestraint.HEAVY,
                pressure=1.0,
            ),
            PreEquilStageConfig(
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


# TODO: ensure all parameters in gromacs_orchestration.py are present here
class GromacsFepSimulationConfig(BaseModel):
    """
    For each leg (BOUND or FREE), specify three parallel lambda schedules:
      - bonded-lambdas
      - coul-lambdas
      - vdw-lambdas

    All three lists must be of identical length for any given LegType.
    If your “FREE” (ligand) leg only needs Coulomb and vdw lambdas, you can still
    supply a “dummy” bonded vector of the same length, for consistency.
    """

    # fmt: off
    bonded_lambdas: dict[LegType, list[float]] = Field(
        default_factory=lambda: {
           LegType.BOUND: [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.35, 0.5,  # noqa: E127,E128,E131,E122,E501
                             0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,      # noqa: E127,E128,E131,E122,E501
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,       # noqa: E127,E128,E131,E122,E501
                             1.0, 1.0, 1.0]  # noqa: E127,E128,E131,E122, E501
        },
        description="Only LegType.BOUND is allowed here; FREE is omitted entirely.",
    )

    coul_lambdas: dict[LegType, list[float]] = Field(
        default_factory=lambda: {
                        LegType.BOUND: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # noqa: E127,E128,E131,E122, E501
                             0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # noqa: E127,E128,E131,E122, E501
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],   # noqa: E127,E128,E131,E122, E501
            LegType.FREE:  [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # noqa: E127,E128,E131,E122, E501
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],    # noqa: E127,E128,E131,E122, E501
        }
    )
    vdw_lambdas: dict[LegType, list[float]] = (
        Field(
            default_factory=lambda: {
                           LegType.BOUND: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # noqa: E127,E128,E131,E122, E501
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4,   # noqa: E127,E128,E131,E122, E501
                             0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],  # noqa: E127,E128,E131,E122, E501
            LegType.FREE:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4,   # noqa: E127,E128,E131,E122, E501
                            0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],   # noqa: E127,E128,E131,E122, E501
            }
        )
    )
    # fmt: on
    mdp_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of MDP parameters to override (e.g., "
        "{'nsteps': 50000, 'dt': 0.002})",
    )
    slurm_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of SLURM parameters to override (e.g., "
        "{'ntasks': 8, 'time': '24:00:00'})",
    )
    modules_to_load: list[str] = Field(
        default_factory=list,
        description="List of environment modules to load before running GROMACS",
    )
    pre_commands: list[str] = Field(
        default_factory=list,
        description="List of shell commands to execute before running GROMACS",
    )
    post_commands: list[str] = Field(
        default_factory=list,
        description="List of shell commands to execute after running GROMACS",
    )

    @model_validator(mode="after")
    def ensure_same_length(cls, model):
        """
        Enforce:
         - BOUND must have bonded, coul, vdw all present & same length.
         - FREE    must have coul, vdw present & same length; bonded may be missing.

        TODO: double check if this is always true for GROMACS
        """
        bonded = model.bonded_lambdas
        coul = model.coul_lambdas
        vdw = model.vdw_lambdas

        # 1) Ensure exactly one key in bonded_lambdas and it is BOUND
        if set(bonded.keys()) != {LegType.BOUND}:
            raise ValueError("`bonded_lambdas` must contain exactly LegType.BOUND.")

        # 2) Make sure that coul_lambdas and vdw_lambdas both have exactly two keys: BOUND and FREE
        missing = {LegType.BOUND, LegType.FREE} - set(coul.keys())
        if missing:
            raise ValueError(f"`coul_lambdas` is missing: {missing}")
        missing = {LegType.BOUND, LegType.FREE} - set(vdw.keys())
        if missing:
            raise ValueError(f"`vdw_lambdas` is missing: {missing}")

        # 3) Validate that lengths match among the three lists for BOUND
        bound_bonded = bonded[LegType.BOUND]
        bound_coul = coul[LegType.BOUND]
        bound_vdw = vdw[LegType.BOUND]
        if not (len(bound_bonded) == len(bound_coul) == len(bound_vdw)):
            raise ValueError(
                f"BOUND: `bonded`, `coul`, and `vdw` lists must be the same length; "
                f"got {len(bound_bonded)}/{len(bound_coul)}/{len(bound_vdw)}."
            )

        # 4) Validate that FREE has only coul & vdw, and their lengths match
        free_coul = coul[LegType.FREE]
        free_vdw = vdw[LegType.FREE]
        if len(free_coul) != len(free_vdw):
            raise ValueError(
                f"FREE: `coul_lambdas` and `vdw_lambdas` must be same length; "
                f"got {len(free_coul)}/{len(free_vdw)}."
            )

        return model


class SomdFepSimulationConfig(BaseModel):
    """
    Configuration for the λ-schedule used in bound and free legs.
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
    use_same_restraints: bool = Field(
        True,
        description="Whether to use the same restraints for all repeats of the bound leg. Note "
        "that this should be used if you plan to run adaptively.",
    )

    class Config:
        extra = "forbid"
        validate_assignment = True


class EnsembleEquilibrationReplicaConfig(BaseModel):
    """
    Configuration for one replica of ensemble-based equilibration.
    """

    runtime: float = Field(5.0, gt=0, lt=50_000)  # NOTE: it is nanoseconds here
    timestep: float = Field(2.0, gt=0, lt=10_000)  # fs
    temperature: float = Field(300)
    pressure: float = Field(1.0)  # atm
    restraint: Optional[SimulationRestraint] = (
        None  # only "all", "backbone", "heavy" are allowed
    )

    class Config:
        extra = "forbid"
        validate_assignment = True


class EnsembleEquilibrationConfig(BaseModel):
    num_replicas: int = Field(5)  # number of replicas for ensemble equilibration
    # Optional[list[EnsembleEquilibrationReplicaConfig]] = None -> mypy always treats it as None
    # at runtime, even though @model_validator fills it in so got a mypy error (len(None))
    replicas: list[EnsembleEquilibrationReplicaConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    def _fill_in_replicas(cls, values):
        n = values.get("num_replicas", 5)
        # only populate if they didn`t explicitly supply any replicas
        if values.get("replicas") is None:
            values["replicas"] = [
                EnsembleEquilibrationReplicaConfig() for _ in range(n)
            ]
        return values


# TODO: create a CustomWorkflowConfig ?
class BaseWorkflowConfig(BaseModel):
    """
    Top-level configuration for entire system prep workflow.

    This procedure is based on workflow from this paper:
    https://pubs.acs.org/doi/10.1021/acs.jctc.4c00806

    Try using default_factory callable that runs on every instantiation ->
    brand-new default value each time, but maybe not needed
    """

    slurm: bool = True
    # Need to add # type: ignore[call-arg] because mypy does not recognize __init__() properly
    param_system_prep: SystemPreparationConfig = Field(
        default_factory=lambda: SystemPreparationConfig()
    )  # type: ignore[call-arg]
    param_preequilibration: EmpiricalPreEquilibrationConfig = Field(
        default_factory=lambda: EmpiricalPreEquilibrationConfig()  # type: ignore[call-arg]
    )
    param_energy_minimisation: EnergyMinimisationConfig = Field(
        default_factory=lambda: EnergyMinimisationConfig()
    )  # type: ignore[call-arg]
    param_ensemble_equilibration: EnsembleEquilibrationConfig = Field(
        default_factory=lambda: EnsembleEquilibrationConfig()  # type: ignore[call-arg]
    )
    param_fep_params: Union[
        SomdFepSimulationConfig, GromacsFepSimulationConfig
    ] = Field(
        default_factory=lambda: GromacsFepSimulationConfig()
    )  # type: ignore[call-arg]

    append_to_ligand_selection: str = Field(
        "",
        description="Atom selection to append to the ligand selection during restraint searching.",
    )

    # Added by JJ-2025-05-05
    mdrun_options: Optional[str] = Field(
        None, description="Extra flags for 'gmx mdrun' (e.g., '-ntmpi 1 -ntomp 1')."
    )

    class Config:
        extra = "forbid"
        validate_assignment = True

    def get_tot_simtime(self, n_runs: int, leg_type: LegType) -> float:
        raise NotImplementedError("...")
