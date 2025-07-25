"""
Script generators for GROMACS ABFE calculations.

This module provides Python classes to generate MDP files and SLURM submit scripts
programmatically, replacing the need for template files.
"""

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator


class SlurmParameters(BaseModel):
    """Pydantic BaseModel for SLURM submit script parameters with validation."""

    # Basic SLURM parameters
    job_name: str = Field(default="gromacs_abfe", description="Job name")
    nodes: int = Field(default=1, ge=1, description="Number of nodes")
    ntasks_per_node: int = Field(default=1, ge=1, description="Tasks per node")
    cpus_per_task: int = Field(default=8, ge=1, description="CPUs per task")
    gres: str = Field(default="gpu:1", description="Generic resources (e.g., GPUs)")
    time: str = Field(default="24:00:00", description="Wall time limit")
    mem: str = Field(default="16G", description="Memory requirement")

    # Output files
    output_file: str = Field(default="slurm-%j.out", description="Standard output file")
    error_file: str = Field(default="slurm-%j.err", description="Standard error file")

    # Additional SLURM directives
    custom_directives: dict[str, str] = Field(
        default_factory=dict, description="Additional SLURM directives"
    )

    @field_validator("time")
    def validate_time_format(cls, v):
        """Validate SLURM time format."""
        import re

        # Accept formats like: 10:00:00, 1-10:00:00, 60, 60:00
        time_patterns = [
            r"^\d+$",  # minutes only
            r"^\d+:\d{2}$",  # minutes:seconds
            r"^\d+:\d{2}:\d{2}$",  # hours:minutes:seconds
            r"^\d+-\d+:\d{2}:\d{2}$",  # days-hours:minutes:seconds
        ]
        if not any(re.match(pattern, v) for pattern in time_patterns):
            raise ValueError(
                f"Invalid time format: {v}. Use formats like '10:00:00', '1-10:00:00', etc."
            )
        return v

    @field_validator("mem")
    def validate_memory_format(cls, v):
        """Validate memory format."""
        import re

        if not re.match(r"^\d+[KMGT]?B?$", v, re.IGNORECASE):
            raise ValueError(
                f"Invalid memory format: {v}. Use formats like '16G', '32GB', '1024M', etc."
            )
        return v

    class Config:
        extra = "forbid"
        validate_assignment = True
        schema_extra = {
            "example": {
                "job_name": "abfe_lambda_0_run_1",
                "partition": "gpu",
                "time": "24:00:00",
                "mem": "32G",
                "gres": "gpu:a100:1",
            }
        }


class SlurmSubmitGenerator:
    """Generates SLURM submit scripts for GROMACS ABFE calculations using Pydantic BaseModel."""

    def __init__(self, base_params: Optional[SlurmParameters] = None) -> None:
        """
        Initialize SLURM submit script generator.

        Parameters
        ----------
        base_params : SlurmParameters, optional
            Base SLURM parameters. If None, uses default parameters.
        """
        self.base_params = base_params or SlurmParameters()

    def generate_submit_script(
        self,
        lambda_state: int,
        run_number: int,
        gmx_exe: str = "gmx",
        mdp_file: str = "lambda.mdp",
        gro_file: str = "system.gro",
        top_file: str = "system.top",
        tpr_file: str = "system.tpr",
        output_prefix: Optional[str] = None,
        modules_to_load: Optional[list[str]] = None,
        custom_overrides: Optional[dict[str, str]] = None,
        pre_commands: Optional[list[str]] = None,
        post_commands: Optional[list[str]] = None,
    ) -> str:
        """Generate SLURM submit script content with Pydantic validation."""
        # Create a copy of base parameters
        params = self.base_params.copy(deep=True)

        # Set job-specific parameters
        if output_prefix is None:
            output_prefix = f"lambda_{lambda_state}_run_{run_number}"

        params.job_name = f"abfe_{output_prefix}"
        params.output_file = f"{output_prefix}.out"
        params.error_file = f"{output_prefix}.err"

        # Apply custom overrides
        if custom_overrides:
            for key, value in custom_overrides.items():
                if hasattr(params, key):
                    setattr(params, key, value)
                else:
                    params.custom_directives[key] = value

        # Pydantic will validate automatically
        return self._format_submit_script(
            params=params,
            lambda_state=lambda_state,
            run_number=run_number,
            gmx_exe=gmx_exe,
            mdp_file=mdp_file,
            gro_file=gro_file,
            top_file=top_file,
            tpr_file=tpr_file,
            output_prefix=output_prefix,
            modules_to_load=modules_to_load,
            pre_commands=pre_commands,
            post_commands=post_commands,
        )

    def _format_submit_script(
        self,
        params: SlurmParameters,
        lambda_state: int,
        run_number: int,
        gmx_exe: str,
        mdp_file: str,
        gro_file: str,
        top_file: str,
        tpr_file: str,
        output_prefix: str,
        modules_to_load: Optional[list[str]] = None,
        pre_commands: Optional[list[str]] = None,
        post_commands: Optional[list[str]] = None,
    ) -> str:
        """Format parameters into submit script content."""
        lines = [
            "#!/bin/bash",
            "",
            f"#SBATCH --job-name={params.job_name}",
            f"#SBATCH --nodes={params.nodes}",
            f"#SBATCH --ntasks-per-node={params.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={params.cpus_per_task}",
            f"#SBATCH --gres={params.gres}",
            f"#SBATCH --time={params.time}",
            f"#SBATCH --mem={params.mem}",
            f"#SBATCH --output={params.output_file}",
            f"#SBATCH --error={params.error_file}",
        ]

        # Add custom SLURM directives
        for key, value in params.custom_directives.items():
            lines.append(f"#SBATCH --{key}={value}")

        lines.extend(
            [
                "",
                "# Job information",
                'echo "Job started at: $(date)"',
                f'echo "Lambda state: {lambda_state}"',
                f'echo "Run number: {run_number}"',
                'echo "Working directory: $(pwd)"',
                'echo "SLURM_JOB_ID: $SLURM_JOB_ID"',
                'echo "Node: $SLURM_NODELIST"',
                "",
            ]
        )

        # Load modules
        if modules_to_load:
            lines.extend(
                [
                    "# Load required modules",
                    *[f"module load {module}" for module in modules_to_load],
                    "",
                ]
            )

        # Pre-commands
        if pre_commands:
            lines.extend(
                [
                    "# Pre-simulation commands",
                    *pre_commands,
                    "",
                ]
            )

        # Main GROMACS commands
        lines.extend(
            [
                "# GROMACS simulation",
                'echo "Starting GROMACS preparation..."',
                f"{gmx_exe} grompp -f {mdp_file} -c {gro_file} -p {top_file} -o {tpr_file} -maxwarn 10",  # noqa
                "",
                'echo "Starting GROMACS simulation..."',
                f"{gmx_exe} mdrun -deffnm {output_prefix} -s {tpr_file} -cpi {output_prefix}.cpt",
                "",
            ]
        )

        # Post-commands
        if post_commands:
            lines.extend(
                [
                    "# Post-simulation commands",
                    *post_commands,
                    "",
                ]
            )

        lines.extend(['echo "Job finished at: $(date)"', "exit 0"])

        return "\n".join(lines) + "\n"

    def write_submit_script(
        self,
        output_path: Union[str, Path],
        lambda_state: int,
        run_number: int,
        gmx_exe: str = "gmx",
        mdp_file: str = "lambda.mdp",
        gro_file: str = "system.gro",
        top_file: str = "system.top",
        tpr_file: str = "system.tpr",
        output_prefix: Optional[str] = None,
        modules_to_load: Optional[list[str]] = None,
        custom_overrides: Optional[dict[str, str]] = None,
        pre_commands: Optional[list[str]] = None,
        post_commands: Optional[list[str]] = None,
    ) -> None:
        """
        Write submit script to disk and make it executable with Pydantic validation.
        """
        content = self.generate_submit_script(
            lambda_state=lambda_state,
            run_number=run_number,
            gmx_exe=gmx_exe,
            mdp_file=mdp_file,
            gro_file=gro_file,
            top_file=top_file,
            tpr_file=tpr_file,
            output_prefix=output_prefix,
            modules_to_load=modules_to_load,
            custom_overrides=custom_overrides,
            pre_commands=pre_commands,
            post_commands=post_commands,
        )

        output_file = Path(output_path)
        output_file.write_text(content)
        output_file.chmod(0o755)  # Make executable


def create_default_slurm_generator(
    time: str = "24:00:00", mem: str = "16G"
) -> SlurmSubmitGenerator:
    """Create SLURM generator with common defaults and Pydantic validation."""
    params = SlurmParameters(time=time, mem=mem)
    return SlurmSubmitGenerator(params)


def create_custom_slurm_generator(**kwargs) -> SlurmSubmitGenerator:
    """Create SLURM generator with custom Pydantic-validated parameters."""
    params = SlurmParameters(**kwargs)
    return SlurmSubmitGenerator(params)
