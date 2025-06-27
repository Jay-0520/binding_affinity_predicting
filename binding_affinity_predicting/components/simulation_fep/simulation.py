import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from binding_affinity_predicting.components.data.enums import JobStatus
from binding_affinity_predicting.components.simulation_base.mdp_parameters import (
    MDPGenerator,
    create_custom_mdp_generator,
)
from binding_affinity_predicting.components.simulation_base.slurm_parameters import (
    SlurmSubmitGenerator,
    create_custom_slurm_generator,
)

logger = logging.getLogger(__name__)


class Simulation:
    """
    Speicific simulation runner for a single lambda window simulation in GROMACS
    """

    def __init__(
        self,
        lam_state: int,
        gro_file: str,
        top_file: str,
        work_dir: Union[str, Path],
        gmx_exe: str = "gmx",
        extra_files: Optional[list[str]] = None,
        run_index: int = 1,
        bonded_list: Sequence[float] = (),
        coul_list: Sequence[float] = (),
        vdw_list: Sequence[float] = (),
        extra_params: Optional[dict[str, Any]] = None,
        mdp_generator: Optional[MDPGenerator] = None,
        slurm_generator: Optional[SlurmSubmitGenerator] = None,
        runtime_ns: Optional[float] = None,
        mdp_overrides: Optional[dict[str, Union[str, int, float]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        lam_state : int
            Lambda state in GROMACS mdp file for the parameter init-lambda-state
        work_dir : str or Path
            Working directory for this simulation
        gmx_exe : str
            GROMACS executable command
        gro_file : str, optional
            Path to structure file
        top_file : str, optional
            Path to topology file
        extra_files : List[str], optional
            Additional files needed for simulation
        run_index : int
            Run/replica index
        bonded_list : Sequence[float]
            Bonded lambda values for all states
        coul_list : Sequence[float]
            Coulomb lambda values for all states
        vdw_list : Sequence[float]
            van der Waals lambda values for all states
        extra_params : Dict, optional
            Additional parameters
        mdp_generator : MDPGenerator, optional
            MDP file generator. If None, uses default.
        slurm_generator : SlurmSubmitGenerator, optional
            SLURM script generator. If None, uses default.
        runtime_ns : float, optional
            Simulation runtime in nanoseconds
        mdp_overrides : Dict, optional
            Custom MDP parameter overrides
        mdp_template : str, optional
            DEPRECATED: Path to MDP template (for backward compatibility)
        """
        self.lam_state = lam_state
        self.work_dir = Path(work_dir)
        self.gmx_exe = gmx_exe
        self.gro_file = gro_file
        self.top_file = top_file
        self.extra_files = extra_files or []
        self.run_index = run_index
        self.bonded_list = list(bonded_list)
        self.coul_list = list(coul_list)
        self.vdw_list = list(vdw_list)
        self.extra_params = extra_params or {}
        self.runtime_ns = runtime_ns
        self.mdp_overrides = mdp_overrides or {}

        # Initialize Pydantic-based generators
        self.mdp_generator: MDPGenerator = (
            mdp_generator or self._create_default_mdp_generator()
        )
        self.slurm_generator: SlurmSubmitGenerator = (
            slurm_generator or self._create_default_slurm_generator()
        )

        # Set up file paths
        self.tpr_file = str(self.work_dir / f"lambda_{lam_state}_run_{run_index}.tpr")
        self.output_file = str(
            self.work_dir / f"lambda_{lam_state}_run_{run_index}.out"
        )
        self.mdp_file = str(self.work_dir / f"lambda_{lam_state}.mdp")

        # Status tracking
        self._finished = False
        self._failed = False
        self._running = False

        # Enhanced status tracking using JobStatus
        self.job_status = JobStatus.NONE
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.setup_time: Optional[datetime] = None
        # Simple flag to track if this was submitted via HPC
        self._submitted_via_hpc = False

        self._validate_inputs()

    def _create_default_mdp_generator(self) -> MDPGenerator:
        """Create a default MDP generator with any custom overrides."""
        # Use any mdp_overrides provided to customize the generator
        return create_custom_mdp_generator(**self.mdp_overrides)

    def _create_default_slurm_generator(self) -> SlurmSubmitGenerator:
        """Create a default SLURM generator."""
        return create_custom_slurm_generator()

    @property
    def running(self) -> bool:
        """Required property for compatibility with SimulationRunner"""
        return self._running

    @property
    def failed(self) -> bool:
        """Whether simulation has failed."""
        return self._failed

    @property
    def finished(self) -> bool:
        """Whether simulation has finished successfully."""
        return self._finished

    def _validate_inputs(self) -> None:
        """Validate input parameters and create work directory."""
        if not self.work_dir.exists():
            logger.info(f"Creating work directory: {self.work_dir}")
            self.work_dir.mkdir(parents=True, exist_ok=True)

        # Check lam_state is a valid index
        if not isinstance(self.lam_state, int):
            raise ValueError("Lambda state must be an integer.")

        # Only validate index bounds if we have lambda lists
        if self.bonded_list and (
            self.lam_state < 0 or self.lam_state >= len(self.bonded_list)
        ):
            raise ValueError(
                f"Lambda index {self.lam_state} is out of range for "
                f"the provided lists (length {len(self.bonded_list)})."
            )

        # Validate file paths if provided
        if self.gro_file and not Path(self.gro_file).exists():
            raise ValueError(
                f"gro_file must be a valid coordinate file: {self.gro_file}"
            )

        if self.top_file and not Path(self.top_file).exists():
            raise ValueError(f"top_file must be a valid topology file: {self.top_file}")

    def setup(self) -> None:
        """
        Prepare the input files for the simulation using Pydantic generators.

        This method replaces the old template-based approach with validated
        parameter generation using Pydantic models.
        """
        try:
            # Generate MDP file using the Pydantic generator with validation
            self.mdp_generator.write_mdp_file(
                output_path=self.mdp_file,
                lambda_state=self.lam_state,
                bonded_lambdas=self.bonded_list,
                coul_lambdas=self.coul_list,
                vdw_lambdas=self.vdw_list,
                runtime_ns=self.runtime_ns,
                custom_overrides=self.mdp_overrides,
            )
            self.setup_time = datetime.now()

            # Update status to QUEUED after setup
            if self.job_status == JobStatus.NONE:
                self.job_status = JobStatus.QUEUED

            logger.info(
                f"Generated validated MDP file for lambda {self.lam_state}: {self.mdp_file}"
            )

        except Exception as e:
            logger.error(
                f"Failed to generate MDP file for lambda {self.lam_state}: {e}"
            )
            self.job_status = JobStatus.FAILED
            self._failed = True
            raise

    def generate_submit_script(
        self,
        output_path: Optional[Union[str, Path]] = None,
        modules_to_load: Optional[list[str]] = None,
        slurm_overrides: Optional[dict[str, str]] = None,
        pre_commands: Optional[list[str]] = None,
        post_commands: Optional[list[str]] = None,
    ) -> str:
        """
        Generate SLURM submit script for this simulation using Pydantic validation.

        Parameters
        ----------
        output_path : str or Path, optional
            Path for submit script. If None, uses work_dir/submit_gmx.sh
        modules_to_load : List[str], optional
            Environment modules to load
        slurm_overrides : Dict[str, str], optional
            Custom SLURM parameter overrides
        pre_commands : List[str], optional
            Commands to run before simulation
        post_commands : List[str], optional
            Commands to run after simulation

        Returns
        -------
        str
            Path to generated submit script
        """
        if output_path is None:
            output_path = self.work_dir / "submit_gmx.sh"

        try:
            self.slurm_generator.write_submit_script(
                output_path=output_path,
                lambda_state=self.lam_state,
                run_number=self.run_index,
                gmx_exe=self.gmx_exe,
                mdp_file=os.path.basename(self.mdp_file),
                gro_file=os.path.basename(self.gro_file),
                top_file=os.path.basename(self.top_file),
                tpr_file=os.path.basename(self.tpr_file),
                modules_to_load=modules_to_load,
                custom_overrides=slurm_overrides,
                pre_commands=pre_commands,
                post_commands=post_commands,
            )

            logger.info(f"Generated validated submit script: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to generate submit script: {e}")
            raise

    def run(self) -> None:
        """
        Run the simulation locally (blocking).
        """
        self.job_status = JobStatus.RUNNING
        self.start_time = datetime.now()
        self._running = True  # Set running to True at startz

        # 1. Setup if not already done
        if not hasattr(self, "mdp_file"):
            logger.warning("Must setup system first before running simulation...")
            self.setup()

        # 2. gmx grompp
        logger.info(f"Preparing tpr for lambda {self.lam_state} in {self.work_dir}")
        grompp_cmd = [
            self.gmx_exe,
            "grompp",
            "-f",
            os.path.basename(self.mdp_file),
            "-c",
            os.path.basename(self.gro_file) if self.gro_file else "system.gro",
            "-p",
            os.path.basename(self.top_file) if self.top_file else "system.top",
            "-o",
            os.path.basename(self.tpr_file),
            "-maxwarn",
            "10",  # Allow some warnings
        ]
        try:
            result = subprocess.run(
                grompp_cmd,
                cwd=self.work_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.debug(f"grompp output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"grompp failed for λ={self.lam_state} in {self.work_dir}: {e}\n"
                f"stdout: {e.stdout}\nstderr: {e.stderr}"
            )
            self.job_status = JobStatus.FAILED
            self._failed = True
            return

        # 3. gmx mdrun
        logger.info(f"Running mdrun for lambda {self.lam_state} in {self.work_dir}")
        mdrun_cmd = [
            self.gmx_exe,
            "mdrun",
            "-deffnm",
            f"lambda_{self.lam_state}_run_{self.run_index}",
            "-cpi",
            # this always works even if the cpt file does not exist
            f"lambda_{self.lam_state}_run_{self.run_index}.cpt",
        ]
        try:
            result = subprocess.run(
                mdrun_cmd, cwd=self.work_dir, check=True, capture_output=True, text=True
            )
            logger.info(f"mdrun completed for λ={self.lam_state}")
            logger.debug(f"mdrun output: {result.stdout}")
            self._finished = True
            self.job_status = JobStatus.FINISHED

        except subprocess.CalledProcessError as e:
            logger.error(
                f"mdrun failed for λ={self.lam_state} in {self.work_dir}: {e}\n"
                f"stdout: {e.stdout}\nstderr: {e.stderr}"
            )
            self.job_status = JobStatus.FAILED
            self._failed = True
            return

        self._running = False
        self.end_time = datetime.now()

    @property
    def failed_simulations(self) -> list["Simulation"]:
        return [self] if self._failed else []

    @property
    def runtime_seconds(self) -> float:
        """Get runtime in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def status_info(self) -> dict[str, Any]:
        """Get detailed status information using JobStatus."""
        return {
            "job_status": self.job_status.name,
            "job_status_value": self.job_status.value,
            "lambda_state": self.lam_state,
            "run_index": self.run_index,
            "finished": self._finished,
            "failed": self._failed,
            "running": self._running,
            "setup_time": self.setup_time.isoformat() if self.setup_time else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "runtime_seconds": self.runtime_seconds,
        }

    def export_configuration(self, output_dir: Union[str, Path]) -> None:
        """
        Export MDP and SLURM configurations for this simulation.

        Parameters
        ----------
        output_dir : str or Path
            Directory to export configurations to
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export MDP configuration
        self.mdp_generator.export_parameters_json(
            output_path / f"mdp_lambda_{self.lam_state}_run_{self.run_index}.json"
        )

        logger.info(f"Exported simulation configurations to {output_path}")

    def get_configuration_summary(self) -> dict[str, Any]:
        """Get a summary of current simulation configuration."""
        return {
            "simulation_info": {
                "lambda_state": self.lam_state,
                "run_index": self.run_index,
                "work_dir": str(self.work_dir),
                "finished": self._finished,
                "failed": self._failed,
            },
            "lambda_vectors": {
                "bonded": self.bonded_list,
                "coulomb": self.coul_list,
                "vdw": self.vdw_list,
            },
            "files": {
                "gro_file": self.gro_file,
                "top_file": self.top_file,
                "mdp_file": self.mdp_file,
                "tpr_file": self.tpr_file,
            },
            "mdp_parameters": self.mdp_generator.base_params.dict(),
            "slurm_parameters": self.slurm_generator.base_params.dict(),
        }

    def __str__(self) -> str:
        """String representation of the simulation."""
        return (
            f"Simulation[λ={self.lam_state}, run={self.run_index}, "
            f"dir={self.work_dir.name}, finished={self._finished}]"
        )
