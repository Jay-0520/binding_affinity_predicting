import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class Simulation:
    """
    Speicific simulation runner for a single lambda window simulation in GROMACS
    """

    def __init__(
        self,
        lam_state: int,
        work_dir: Path,
        gmx_exe="gmx",
        mdp_template=None,
        gro_file=None,
        top_file=None,
        extra_files=None,
        run_index=1,
        extra_params=None,
    ):
        """
        Parameters
        ----------
        lam_state: int
            Lambda state in gromacs mdp file for the parameter init-lambda-state
        """
        self.lam_state = lam_state
        self.work_dir = Path(work_dir)
        self.gmx_exe = gmx_exe
        self.mdp_template = mdp_template
        self.gro_file = gro_file
        self.top_file = top_file
        self.extra_files = extra_files or []
        self.run_index = run_index
        self.extra_params = extra_params or {}
        self.tpr_file = os.path.join(
            self.work_dir, f"lambda_{lam_state}_run{run_index}.tpr"
        )
        self.output_file = os.path.join(
            self.work_dir, f"lambda_{lam_state}_run{run_index}.out"
        )
        self.finished = False
        self.failed = False

        self._validate_inputs()

    def _validate_inputs(self):
        # Ensure work_dir exists
        if not self.work_dir.exists():
            logger.info(f"Creating work directory: {self.work_dir}")
            self.work_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(self.lam_state, int):
            raise ValueError("Lambda state must be a integer.")
        elif self.lam_state < 0:
            raise ValueError("Lambda state must be equal to or larger than 0.")
        else:
            logger.info(
                f"Running simulation {self.__class__.__name__} with Lambda state:"
                f" {self.lam_state}..."
            )

    def setup(self):
        # Prepare .mdp for this lambda
        mdp_file = os.path.join(self.work_dir, f"lambda_{self.lam_state}.mdp")
        if self.mdp_template:
            with open(self.mdp_template) as f:
                lines = f.readlines()
            with open(mdp_file, "w") as fout:
                for line in lines:
                    # Simple replace for lambda (customize as needed)
                    if line.strip().startswith("init-lambda-state"):
                        fout.write(f"init-lambda-state        = {self.lam_state}\n")
                    else:
                        fout.write(line)
        else:
            raise ValueError("mdp_template is required to prepare inputs.")

        self.mdp_file = mdp_file

    def run(self):
        # 1. gmx grompp
        logger.info(f"Preparing tpr for lambda {self.lam} in {self.work_dir}")
        grompp_cmd = [
            self.gmx_exe,
            "grompp",
            "-f",
            self.mdp_file,
            "-c",
            self.gro_file,
            "-p",
            self.top_file,
            "-o",
            self.tpr_file,
        ]
        try:
            subprocess.run(grompp_cmd, cwd=self.work_dir, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"grompp failed for λ={self.lam_state} in {self.work_dir}: {e}"
            )
            self.failed = True
            return

        # 2. gmx mdrun
        logger.info(f"Running mdrun for lambda {self.lam_state} in {self.work_dir}")
        mdrun_cmd = [
            self.gmx_exe,
            "mdrun",
            "-deffnm",
            f"lambda_{self.lam_state:.2f}_run{self.run_index}",
        ]
        try:
            subprocess.run(mdrun_cmd, cwd=self.work_dir, check=True)
            self.finished = True
        except subprocess.CalledProcessError as e:
            logger.error(f"mdrun failed for λ={self.lam_state} in {self.work_dir}: {e}")
            self.failed = True

    @property
    def failed_simulations(self):
        return [self] if self.failed else []
