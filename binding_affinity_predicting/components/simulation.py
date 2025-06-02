import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Sequence

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
        mdp_template: Optional[str] = None,
        gro_file: Optional[str] = None,
        top_file: Optional[str] = None,
        extra_files: Optional[list[str]] = None,
        run_index: int = 1,
        bonded_list: Sequence[float] = (),
        coul_list: Sequence[float] = (),
        vdw_list: Sequence[float] = (),
        extra_params=None,
    ):
        """
        Parameters
        ----------
        lam_state: int
            Lambda state in gromacs mdp file for the parameter init-lambda-state
        mdp_template : str
            Path to the original MDP template that contains placeholders
        """
        self.lam_state = lam_state
        self.work_dir = Path(work_dir)
        self.gmx_exe = gmx_exe
        self.mdp_template = mdp_template
        self.gro_file = gro_file
        self.top_file = top_file
        self.extra_files = extra_files or []
        self.run_index = run_index
        self.bonded_list = list(bonded_list)
        self.coul_list = list(coul_list)
        self.vdw_list = list(vdw_list)
        self.extra_params = extra_params or {}
        self.tpr_file = os.path.join(
            self.work_dir, f"lambda_{lam_state}_run_{run_index}.tpr"
        )
        self.output_file = os.path.join(
            self.work_dir, f"lambda_{lam_state}_run_{run_index}.out"
        )
        self.finished = False
        self.failed = False

        self._validate_inputs()

    def _validate_inputs(self):
        if not self.work_dir.exists():
            logger.info(f"Creating work directory: {self.work_dir}")
            self.work_dir.mkdir(parents=True, exist_ok=True)

        # Check lam_state is a valid index
        if not isinstance(self.lam_state, int):
            raise ValueError("Lambda state must be an integer.")
        if self.lam_state < 0 or self.lam_state >= len(self.bonded_list):
            raise ValueError(
                f"Lambda index {self.lam_state} is out of range for "
                f"the provided lists (length {len(self.bonded_list)})."
            )

        if self.mdp_template is None or not Path(self.mdp_template).exists():
            raise ValueError(
                "mdp_template must be a valid path to an existing MDP template."
            )

        if self.gro_file is None or not Path(self.gro_file).exists():
            raise ValueError("gro_file must be a valid coordinate file.")

        if self.top_file is None or not Path(self.top_file).exists():
            raise ValueError("top_file must be a valid topology file.")

    def setup(self):
        """
        Prepare the input files for the simulation.

        will rewrite the MDP template so that the four lines:

        init-lambda-state        = LAMBDA_STATE_INPUT
        bonded-lambdas           = BONDED_LAMBDAS_INIT
        coul-lambdas             = COUL_LAMBDAS_INIT
        vdw-lambdas              = VDW_LAMBDAS_INIT

        become e.g.:

        init-lambda-state        = 5
        bonded-lambdas           = 0.0 0.01 0.025 0.05 … (etc)
        coul-lambdas             = 0.0 0.00 0.000 0.00 … (etc)
        vdw-lambdas              = 0.0 0.00 0.000 0.00 … (etc)

        """
        mdp_out = self.work_dir / f"lambda_{self.lam_state}.mdp"
        lines = Path(self.mdp_template).read_text().splitlines()

        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("init-lambda-state"):
                # Replace with the integer index
                new_lines.append(f"init-lambda-state        = {self.lam_state}")
            elif stripped.startswith("bonded-lambdas"):
                # Join the full bonded list into one single space-separated string
                bonded_str = " ".join(str(x) for x in self.bonded_list)
                new_lines.append(f"bonded-lambdas           = {bonded_str}")
            elif stripped.startswith("coul-lambdas"):
                coul_str = " ".join(str(x) for x in self.coul_list)
                new_lines.append(f"coul-lambdas             = {coul_str}")
            elif stripped.startswith("vdw-lambdas"):
                vdw_str = " ".join(str(x) for x in self.vdw_list)
                new_lines.append(f"vdw-lambdas              = {vdw_str}")
            else:
                # Leave every other line untouched
                new_lines.append(line)

        # Write the modified MDP out
        mdp_out.write_text("\n".join(new_lines) + "\n")
        self.mdp_file = str(mdp_out)

    def run(self):
        # 1. gmx grompp
        logger.info(f"Preparing tpr for lambda {self.lam_state} in {self.work_dir}")
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
            f"lambda_{self.lam_state}_run_{self.run_index}",
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
