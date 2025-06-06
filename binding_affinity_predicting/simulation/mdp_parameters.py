"""
Script generators for GROMACS ABFE calculations.

This module provides Python classes to generate MDP files and SLURM submit scripts
programmatically, replacing the need for template files.
"""

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class MDPParameters(BaseModel):
    """Pydantic BaseModel for MDP file parameters with validation
    and sensible defaults for ABFE calculations."""

    # Run control
    integrator: str = Field(default="sd", description="Stochastic leap-frog integrator")
    nsteps: int = Field(
        default=250,
        ge=1,
        description="Number of simulation steps (will be overridden by runtime_ns)",
    )
    dt: float = Field(default=0.002, gt=0, description="2 fs timestep")
    comm_mode: str = Field(
        default="Linear", description="Remove center of mass translation"
    )
    nstcomm: int = Field(
        default=100, ge=1, description="Frequency for center of mass motion removal"
    )

    # Output control
    nstxout: int = Field(default=0, ge=0, description="Don't save coordinates to .trr")
    nstvout: int = Field(default=0, ge=0, description="Don't save velocities to .trr")
    nstfout: int = Field(default=0, ge=0, description="Don't save forces to .trr")
    nstxout_compressed: int = Field(
        default=1000,
        ge=0,
        description="xtc compressed trajectory output every 1000 steps",
    )
    compressed_x_precision: int = Field(
        default=1000, ge=1, description="Precision for compressed trajectory"
    )
    nstlog: int = Field(default=1000, ge=1, description="Update log file frequency")
    nstenergy: int = Field(default=1000, ge=1, description="Save energies frequency")
    nstcalcenergy: int = Field(
        default=100, ge=1, description="Calculate energies frequency"
    )

    # Bonds/Constraints
    constraint_algorithm: str = Field(
        default="lincs", description="Holonomic constraints"
    )
    constraints: str = Field(
        default="all-bonds", description="Constrain all bonds (not just hydrogens)"
    )
    lincs_iter: int = Field(default=1, ge=1, description="Accuracy of LINCS")
    lincs_order: int = Field(default=4, ge=1, description="Accuracy of LINCS")
    lincs_warnangle: int = Field(
        default=30, ge=0, le=180, description="Maximum angle before LINCS complains"
    )
    continuation: str = Field(
        default="yes",
        pattern="^(yes|no)$",
        description="Useful for exact continuations and reruns",
    )

    # Neighbor searching
    cutoff_scheme: str = Field(default="Verlet", description="Cutoff scheme")
    ns_type: str = Field(default="grid", description="Search neighboring grid cells")
    nstlist: int = Field(default=10, ge=1, description="Neighborlist update frequency")
    rlist: float = Field(
        default=1.0, gt=0, description="Short-range neighborlist cutoff (nm)"
    )
    pbc: str = Field(default="xyz", description="3D periodic boundary conditions")

    # Electrostatics
    coulombtype: str = Field(default="PME", description="Particle Mesh Ewald")
    rcoulomb: float = Field(
        default=1.0, gt=0, description="Short-range electrostatic cutoff (nm)"
    )
    ewald_geometry: str = Field(
        default="3d", description="Ewald sum in all three dimensions"
    )
    pme_order: int = Field(default=6, ge=4, description="Interpolation order for PME")
    fourierspacing: float = Field(
        default=0.10, gt=0, description="Grid spacing for FFT"
    )
    ewald_rtol: float = Field(
        default=1e-6,
        gt=0,
        description="Relative strength of Ewald-shifted direct potential",
    )

    # van der Waals
    vdw_type: str = Field(default="PME", description="vdW interaction type")
    rvdw: float = Field(default=1.0, gt=0, description="vdW cutoff")
    vdw_modifier: str = Field(
        default="Potential-Shift", description="vdW potential modifier"
    )
    ewald_rtol_lj: float = Field(
        default=1e-3, gt=0, description="LJ Ewald relative tolerance"
    )
    lj_pme_comb_rule: str = Field(
        default="Geometric", description="LJ PME combination rule"
    )
    dispcorr: str = Field(
        default="EnerPres", description="Long-range dispersion correction"
    )

    # Temperature coupling
    tc_grps: str = Field(default="System", description="Temperature coupling groups")
    tau_t: float = Field(
        default=1.0, gt=0, description="Time constant for temperature coupling"
    )
    ref_t: float = Field(default=300, gt=0, description="Reference temperature")

    # Pressure coupling
    pcoupl: str = Field(
        default="Parrinello-Rahman", description="Pressure coupling algorithm"
    )
    pcoupltype: str = Field(
        default="isotropic", description="Uniform scaling of box vectors"
    )
    tau_p: float = Field(
        default=2.0, gt=0, description="Time constant for pressure coupling"
    )
    ref_p: float = Field(default=1.0, description="Reference pressure (bar)")
    compressibility: float = Field(
        default=4.5e-5, gt=0, description="Isothermal compressibility (bar^-1)"
    )

    # Velocity generation
    gen_vel: str = Field(
        default="no", pattern="^(yes|no)$", description="Velocity generation is off"
    )

    # Free energy calculations
    free_energy: str = Field(
        default="yes", pattern="^(yes|no)$", description="Free energy calculation"
    )
    couple_moltype: str = Field(default="LIG", description="Molecule type to couple")
    couple_lambda0: str = Field(default="vdw-q", description="State at lambda=0")
    couple_lambda1: str = Field(default="none", description="State at lambda=1")
    couple_intramol: str = Field(
        default="no",
        pattern="^(yes|no)$",
        description="Couple intramolecular interactions",
    )
    separate_dhdl_file: str = Field(
        default="yes", pattern="^(yes|no)$", description="Separate dH/dl file"
    )
    init_lambda_state: Optional[int] = Field(
        default=None, ge=0, description="Initial lambda state"
    )
    nstdhdl: int = Field(default=100, ge=1, description="Frequency to write dH/dl")
    calc_lambda_neighbors: int = Field(
        default=-1, description="Calculate neighbors (-1 = all)"
    )

    # Lambda vectors (will be populated based on simulation config)
    bonded_lambdas: list[float] = Field(
        default_factory=list, description="Bonded lambda values"
    )
    coul_lambdas: list[float] = Field(
        default_factory=list, description="Coulomb lambda values"
    )
    vdw_lambdas: list[float] = Field(
        default_factory=list, description="vdW lambda values"
    )

    # Soft-core parameters
    sc_alpha: float = Field(default=0.5, ge=0, description="Soft-core alpha parameter")
    sc_power: int = Field(default=1, ge=1, description="Soft-core power")
    sc_sigma: float = Field(default=0.3, ge=0, description="Soft-core sigma parameter")

    # Additional custom parameters
    custom_params: dict[str, Union[str, int, float]] = Field(
        default_factory=dict, description="Additional custom parameters"
    )

    @field_validator("bonded_lambdas", "coul_lambdas", "vdw_lambdas")
    def validate_lambda_values(cls, v: list[float]) -> list[float]:
        """Validate that lambda values are between 0 and 1."""
        for val in v:
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"Lambda values must be between 0.0 and 1.0, got {val}"
                )
        return v

    @model_validator(mode="after")
    def validate_lambda_consistency(cls, model: "MDPParameters") -> "MDPParameters":
        bonded = model.bonded_lambdas
        coul = model.coul_lambdas
        vdw = model.vdw_lambdas

        # All three lists must have the same length (if they`re nonempty)
        if bonded and coul and vdw:
            lengths = {len(bonded), len(coul), len(vdw)}
            if len(lengths) != 1:
                raise ValueError(
                    f"All lambda vectors must have the same length. "
                    f"Got bonded={len(bonded)}, coul={len(coul)}, vdw={len(vdw)}"
                )

        init_state = model.init_lambda_state
        if init_state is not None and bonded:
            if init_state >= len(bonded):
                raise ValueError(
                    f"init_lambda_state ({init_state}) "
                    f"must be < number of lambda states ({len(bonded)})"
                )

        return model

    @field_validator("integrator")
    def validate_integrator(cls, v: str) -> str:
        """Validate integrator choice."""
        valid_integrators = ["md", "sd", "steep", "cg", "l-bfgs"]
        if v not in valid_integrators:
            raise ValueError(f"Integrator must be one of {valid_integrators}, got {v}")
        return v

    class Config:
        extra = "forbid"  # Don't allow extra fields
        validate_assignment = True  # Validate on assignment
        json_schema_extra = {
            "example": {
                "integrator": "sd",
                "nsteps": 2500000,
                "dt": 0.002,
                "ref_t": 300,
                "bonded_lambdas": [0.0, 0.5, 1.0],
                "coul_lambdas": [0.0, 0.5, 1.0],
                "vdw_lambdas": [0.0, 0.5, 1.0],
                "init_lambda_state": 0,
            }
        }


class MDPGenerator:
    """Generates GROMACS MDP files for ABFE calculations"""

    def __init__(self, base_params: Optional[MDPParameters] = None) -> None:
        """
        Initialize MDP generator.

        Parameters
        ----------
        base_params : MDPParameters, optional
            Base parameters to use. If None, uses default parameters.
        """
        self.base_params = base_params or MDPParameters()

    def generate_mdp_content(
        self,
        lambda_state: int,
        bonded_lambdas: list[float],
        coul_lambdas: list[float],
        vdw_lambdas: list[float],
        runtime_ns: Optional[float] = None,
        custom_overrides: Optional[dict[str, Union[str, int, float]]] = None,
    ) -> str:
        """
        Generate MDP file content for a specific lambda state.

        Parameters
        ----------
        lambda_state : int
            Lambda state index (0-based)
        bonded_lambdas : List[float]
            Bonded lambda values for all states
        coul_lambdas : List[float]
            Coulomb lambda values for all states
        vdw_lambdas : List[float]
            van der Waals lambda values for all states
        runtime_ns : float, optional
            Runtime in nanoseconds. If provided, calculates nsteps.
        custom_overrides : Dict, optional
            Custom parameter overrides

        Returns
        -------
        str
            Complete MDP file content
        """
        # Create a copy of base parameters using Pydantic's copy method
        params = self.base_params.model_copy(deep=True)

        # Set lambda-specific parameters
        params.init_lambda_state = lambda_state
        params.bonded_lambdas = bonded_lambdas
        params.coul_lambdas = coul_lambdas
        params.vdw_lambdas = vdw_lambdas

        # Calculate nsteps if runtime is provided
        if runtime_ns is not None:
            params.nsteps = int(runtime_ns * 1000 / params.dt)

        # Apply custom overrides
        if custom_overrides:
            for key, value in custom_overrides.items():
                if hasattr(params, key):
                    setattr(params, key, value)
                else:
                    params.custom_params[key] = value

        # Validate the final parameters (Pydantic will validate automatically on assignment)
        return self._format_mdp_content(params)

    def _format_mdp_content(self, params: MDPParameters) -> str:
        """Format parameters into MDP file content."""
        lines = [
            ";====================================================",
            "; Production simulation for ABFE calculation",
            "; Generated automatically with Pydantic validation",
            ";====================================================",
            "",
            ";----------------------------------------------------",
            "; RUN CONTROL",
            ";----------------------------------------------------",
            f"integrator   = {params.integrator}            ; stochastic leap-frog integrator",
            f"nsteps       = {params.nsteps}        ; number of simulation steps",
            f"dt           = {params.dt}         ; time step (ps)",
            f"comm-mode    = {params.comm_mode}        ; remove center of mass translation",
            f"nstcomm      = {params.nstcomm}  ; frequency for center of mass motion removal",
            "",
            ";----------------------------------------------------",
            "; OUTPUT CONTROL",
            ";----------------------------------------------------",
            f"nstxout                = {params.nstxout}          ; don't save coordinates to .trr",
            f"nstvout                = {params.nstvout}          ; don't save velocities to .trr",
            f"nstfout                = {params.nstfout}          ; don't save forces to .trr",
            f"nstxout-compressed     = {params.nstxout_compressed}  ; xtc compressed trajectory output",  # noqa
            f"compressed-x-precision = {params.compressed_x_precision} ; precision for compressed trajectory",  # noqa
            f"nstlog                 = {params.nstlog}       ; update log file frequency",
            f"nstenergy              = {params.nstenergy}       ; save energies frequency",
            f"nstcalcenergy          = {params.nstcalcenergy}        ; calculate energies frequency",  # noqa
            "",
            ";----------------------------------------------------",
            "; BONDS",
            ";----------------------------------------------------",
            f"constraint_algorithm   = {params.constraint_algorithm}      ; holonomic constraints",
            f"constraints            = {params.constraints}  ; constrain all bonds",
            f"lincs_iter             = {params.lincs_iter}          ; accuracy of LINCS (1 is default)",  # noqa
            f"lincs_order            = {params.lincs_order}          ; also related to accuracy (4 is default)",  # noqa
            f"lincs-warnangle        = {params.lincs_warnangle}         ; maximum angle before LINCS complains",  # noqa
            f"continuation           = {params.continuation}        ; useful for exact continuations and reruns",  # noqa
            "",
            ";----------------------------------------------------",
            "; NEIGHBOR SEARCHING",
            ";----------------------------------------------------",
            f"cutoff-scheme   = {params.cutoff_scheme}",
            f"ns-type         = {params.ns_type}   ; search neighboring grid cells",
            f"nstlist         = {params.nstlist}     ; neighborlist update frequency",
            f"rlist           = {params.rlist}    ; short-range neighborlist cutoff (in nm)",
            f"pbc             = {params.pbc}    ; 3D PBC",
            "",
            ";----------------------------------------------------",
            "; ELECTROSTATICS",
            ";----------------------------------------------------",
            f"coulombtype      = {params.coulombtype}      ; Particle Mesh Ewald for long-range electrostatics",  # noqa
            f"rcoulomb         = {params.rcoulomb}      ; short-range electrostatic cutoff (in nm)",
            f"ewald_geometry   = {params.ewald_geometry}       ; Ewald sum is performed in all three dimensions",  # noqa
            f"pme-order        = {params.pme_order}        ; interpolation order for PME",
            f"fourierspacing   = {params.fourierspacing}     ; grid spacing for FFT",
            f"ewald-rtol       = {params.ewald_rtol}     ; relative strength of the Ewald-shifted direct potential",  # noqa
            "",
            ";----------------------------------------------------",
            "; VDW",
            ";----------------------------------------------------",
            f"vdw-type                = {params.vdw_type}",
            f"rvdw                    = {params.rvdw}",
            f"vdw-modifier            = {params.vdw_modifier}",
            f"ewald-rtol-lj           = {params.ewald_rtol_lj}",
            f"lj-pme-comb-rule        = {params.lj_pme_comb_rule}",
            f"DispCorr                = {params.dispcorr}",
            "",
            ";----------------------------------------------------",
            "; TEMPERATURE & PRESSURE COUPL",
            ";----------------------------------------------------",
            f"tc_grps          = {params.tc_grps}",
            f"tau_t            = {params.tau_t}",
            f"ref_t            = {params.ref_t}",
            f"pcoupl           = {params.pcoupl}",
            f"pcoupltype       = {params.pcoupltype}            ; uniform scaling of box vectors",
            f"tau_p            = {params.tau_p}                 ; time constant (ps)",
            f"ref_p            = {params.ref_p}                 ; reference pressure (bar)",
            f"compressibility  = {params.compressibility}       ; isothermal compressibility (bar^-1)",  # noqa
            "",
            ";----------------------------------------------------",
            "; VELOCITY GENERATION",
            ";----------------------------------------------------",
            f"gen_vel      = {params.gen_vel}       ; Velocity generation",
            "",
            ";----------------------------------------------------",
            "; FREE ENERGY CALCULATIONS",
            ";----------------------------------------------------",
            f"free-energy              = {params.free_energy}",
            f"couple-moltype           = {params.couple_moltype}",
            f"couple-lambda0           = {params.couple_lambda0}",
            f"couple-lambda1           = {params.couple_lambda1}",
            f"couple-intramol          = {params.couple_intramol}",
            f"separate-dhdl-file       = {params.separate_dhdl_file}",
            f"sc-alpha                 = {params.sc_alpha}",
            f"sc-power                 = {params.sc_power}",
            f"sc-sigma		 = {params.sc_sigma}",
            f"init-lambda-state        = {params.init_lambda_state}",
            f"bonded-lambdas           = {' '.join(map(str, params.bonded_lambdas))}",
            f"coul-lambdas             = {' '.join(map(str, params.coul_lambdas))}",
            f"vdw-lambdas              = {' '.join(map(str, params.vdw_lambdas))}",
            f"nstdhdl                  = {params.nstdhdl}",
            f"calc-lambda-neighbors    = {params.calc_lambda_neighbors}",
        ]

        # Add custom parameters
        if params.custom_params:
            lines.extend(["", ";----------------------------------------------------"])
            lines.extend(["; CUSTOM PARAMETERS"])
            lines.extend([";----------------------------------------------------"])
            for key, value in params.custom_params.items():
                lines.append(f"{key:<23} = {value}")

        return "\n".join(lines) + "\n"

    def write_mdp_file(
        self,
        output_path: Union[str, Path],
        lambda_state: int,
        bonded_lambdas: list[float],
        coul_lambdas: list[float],
        vdw_lambdas: list[float],
        runtime_ns: Optional[float] = None,
        custom_overrides: Optional[dict[str, Union[str, int, float]]] = None,
    ) -> None:
        """
        Write MDP file to disk

        Parameters
        ----------
        output_path : str or Path
            Output file path
        lambda_state : int
            Lambda state index
        bonded_lambdas : List[float]
            Bonded lambda values
        coul_lambdas : List[float]
            Coulomb lambda values
        vdw_lambdas : List[float]
            van der Waals lambda values
        runtime_ns : float, optional
            Runtime in nanoseconds
        custom_overrides : Dict, optional
            Custom parameter overrides
        """
        content = self.generate_mdp_content(
            lambda_state=lambda_state,
            bonded_lambdas=bonded_lambdas,
            coul_lambdas=coul_lambdas,
            vdw_lambdas=vdw_lambdas,
            runtime_ns=runtime_ns,
            custom_overrides=custom_overrides,
        )

        Path(output_path).write_text(content)

    def export_parameters_json(self, output_path: Union[str, Path]) -> None:
        """Export current parameters to JSON file for reproducibility."""
        Path(output_path).write_text(self.base_params.model_dump_json(indent=2))


# Example usage and factory functions
def create_default_mdp_generator() -> MDPGenerator:
    """Create MDP generator with sensible defaults for ABFE calculations."""
    return MDPGenerator()


def create_custom_mdp_generator(**kwargs) -> MDPGenerator:
    """Create MDP generator with custom Pydantic-validated parameters."""
    params = MDPParameters(**kwargs)
    return MDPGenerator(params)
