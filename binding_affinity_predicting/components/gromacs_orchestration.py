"""
gromacs_orchestration.py

Functionality for setting up and running an entire ABFE calculation using GROMACS,
consisting of two legs (bound and unbound) and multiple lambda‐windows per leg.
In GROMACS, all three alchemical stages (RESTRAIN → DISCHARGE → VANISH) are encoded
in a single MDP file via bonded‐, coulomb‐, and vdw‐lambda vectors, so we do not
instantly split into separate Stage objects. Instead, each lambda window is its own
sub‐runner and drives GROMACS with a different `init‐lambda‐state`.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Sequence

from binding_affinity_predicting.components.simulation import Simulation
from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.data.enums import LegType, PreparationStage
from binding_affinity_predicting.data.schemas import (
    GromacsFepSimulationConfig,
)
from binding_affinity_predicting.hpc_cluster.virtual_queue import VirtualQueue

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LambdaWindow(SimulationRunner):
    runtime_attributes = {"_finished": False, "_failed": False, "_dg": None}

    def __init__(
        self,
        lam_state: int,
        input_dir: str,
        output_dir: str,
        sim_params: dict,
        ensemble_size: int = 5,
        virtual_queue: Optional[VirtualQueue] = None,
    ):
        super().__init__(
            input_dir=input_dir, output_dir=output_dir, ensemble_size=ensemble_size
        )
        self.lam_state = lam_state
        self.virtual_queue = virtual_queue
        self.sim_params = sim_params

        # Initialize _sub_sim_runners as list of Simulation objects
        self._sub_sim_runners: list[Simulation] = []
        for run_no in range(1, ensemble_size + 1):
            run_name = f"run_{str(run_no).zfill(2)}"
            sim_base_dir = f"{self.output_dir}/{run_name}"

            sim = Simulation(
                lam_state=lam_state,
                run_index=run_no,
                work_dir=sim_base_dir,
                **sim_params,
            )
            self._sub_sim_runners.append(sim)

    def setup(self) -> None:
        super().setup()

    def run(self, run_nos: Optional[list[int]] = None, *args, **kwargs) -> None:
        """Run simulations for specified run numbers"""
        run_nos = self._get_valid_run_nos(run_nos)

        # Run only the specified simulations
        for run_no in run_nos:
            if run_no < 1 or run_no > self.ensemble_size:
                raise ValueError(
                    f"Invalid run number {run_no}. Must be in [1..{self.ensemble_size}]."
                )
            sim_index = run_no - 1
            self._sub_sim_runners[sim_index].run()

        # Update status based on simulation results
        self._update_status()

    def _update_status(self):
        """Update _failed and _finished status based on sub-simulations"""
        if any(getattr(sim, "failed", False) for sim in self._sub_sim_runners):
            self._failed = True
            self._finished = False
        else:
            self._failed = False
            self._finished = all(
                getattr(sim, "finished", False) for sim in self._sub_sim_runners
            )

    @property
    def failed_simulations(self) -> list[SimulationRunner]:
        """
        Return a list of all child Simulation(...) runners that have failed.
        """
        return [sim for sim in self.sims if getattr(sim, "failed", False)]

    @property
    def running(self) -> bool:
        """
        True if any child Simulation(...) is still running.
        """
        return any(getattr(sim, "running", False) for sim in self.sims)


class Leg(SimulationRunner):
    """
    A Leg (BOUND or FREE) that groups multiple lambda-window replicas.
    Builds the requested directory tree and symlinks inputs.
    """

    runtime_attributes = {"_finished": False, "_failed": False}

    def __init__(
        self,
        leg_type: LegType,
        lam_indices: list[int],
        ensemble_size: int,
        input_dir: str,
        output_dir: str,
        sim_config: GromacsFepSimulationConfig,
    ):
        super().__init__(
            input_dir=input_dir, output_dir=output_dir, ensemble_size=ensemble_size
        )
        self.leg_type = leg_type
        self.lam_indices = lam_indices
        self.ensemble_size = ensemble_size
        self.sim_config = sim_config
        self._sub_sim_runners: Sequence[LambdaWindow] = []

    @property
    def failed_simulations(self):
        return [win for win in self._sub_sim_runners if win._failed]

    def setup(self) -> None:
        """
        Break down the large setup into discrete steps:
          1) prepare_leg_base()
          2) copy_equilibrated_files()
          3) copy_common_templates()
          4) make_lambda_run_dirs_and_link()
          5) instantiate_lambda_windows()
        """
        leg_base = Path(self.output_dir)
        leg_base.mkdir(parents=True, exist_ok=True)

        # 1) Prepare the “input” subfolder under leg_base
        leg_input = leg_base / "input"
        leg_input.mkdir(exist_ok=True)

        # 2) Copy pre‐equilibrated .gro/.top/.itp into leg_input
        self._copy_equilibrated_files(leg_input)

        # 3) Copy the MDP template + run script into leg_input
        self._copy_common_templates(leg_input)

        # 4) For each λ index and each replicate, make run dirs & symlink inputs
        self._make_lambda_run_dirs_and_link(leg_input)

        # 5) Finally, build each LambdaWindow and call its setup()
        self._instantiate_lambda_windows()

    def _copy_equilibrated_files(self, leg_input: Path) -> None:
        """
        Copy all pre‐equilibrated .gro/.top (plus .itp for BOUND) from
        `input_dir/equilibration/` into `leg_input/`.
        """
        suffix = PreparationStage.EQUILIBRATED.file_suffix
        src_equil = Path(self.input_dir) / "equilibration"

        for run_idx in range(1, self.ensemble_size + 1):
            # copy <leg>_equilibrated_{run_idx}_final.gro/.top
            for ext in (".gro", ".top"):
                fname = f"{self.leg_type.name.lower()}{suffix}_{run_idx}_final{ext}"
                src_file = src_equil / fname
                if not src_file.exists():
                    raise FileNotFoundError(f"Cannot find {src_file}")
                shutil.copy(src_file, leg_input / fname)

            # if BOUND, also copy restraint_{run_idx}.itp
            if self.leg_type is LegType.BOUND:
                restr_name = f"restraint_{run_idx}.itp"
                src_rest = src_equil / restr_name
                if not src_rest.exists():
                    raise FileNotFoundError(f"Cannot find {src_rest}")
                shutil.copy(src_rest, leg_input / restr_name)

    def _copy_common_templates(self, leg_input: Path) -> None:
        """
        Copy the λ‐template.mdp and submit_gmx.sh script into leg_input.
        """
        tmpl_mdp = Path(self.input_dir) / "lambda.template.mdp"
        if not tmpl_mdp.exists():
            raise FileNotFoundError(f"MDP template not found at {tmpl_mdp}")
        shutil.copy(tmpl_mdp, leg_input / "lambda.template.mdp")

        run_scr = Path(self.input_dir) / "submit_gmx.sh"
        if not run_scr.exists():
            raise FileNotFoundError(f"Run script not found at {run_scr}")
        shutil.copy(run_scr, leg_input / "submit_gmx.sh")

    def _make_lambda_run_dirs_and_link(self, leg_input: Path) -> None:
        """
        Under `leg_base/λ_{k}/run_{r}/`, create folders and symlink or copy:
          - {leg}_equilibrated_{r}_final.gro
          - {leg}_equilibrated_{r}_final.top
          - restraint_{r}.itp    (only for BOUND)
          - lambda.template.mdp
          - submit_gmx.sh
        """
        suffix = PreparationStage.EQUILIBRATED.file_suffix
        leg_base = Path(self.output_dir)

        for lam_idx in self.lam_indices:
            lam_dir = leg_base / f"lambda_{lam_idx}"
            lam_dir.mkdir(exist_ok=True)

            for run_idx in range(1, self.ensemble_size + 1):
                run_dir = lam_dir / f"run_{run_idx}"
                run_dir.mkdir(exist_ok=True)

                # a) gather filenames to symlink from leg_input
                to_link = []
                for ext in (".gro", ".top"):
                    fname = f"{self.leg_type.name.lower()}{suffix}_{run_idx}_final{ext}"
                    to_link.append(fname)
                if self.leg_type is LegType.BOUND:
                    to_link.append(f"restraint_{run_idx}.itp")

                # b) create or overwrite each symlink
                for filename in to_link:
                    srcfile = leg_input / filename
                    destfile = run_dir / filename

                    if destfile.exists() or destfile.is_symlink():
                        logger.info(f"Removing existing file/symlink: {destfile}")
                        destfile.unlink()

                    os.symlink(os.path.relpath(srcfile, start=run_dir), destfile)

                # c) copy (not symlink) the MDP template and run script into run_dir
                for fname in ("lambda.template.mdp", "submit_gmx.sh"):
                    srcfile = leg_input / fname
                    destfile = run_dir / fname
                    if not srcfile.exists():
                        raise FileNotFoundError(f"Missing template/script at {srcfile}")
                    shutil.copy(srcfile, destfile)

    def _instantiate_lambda_windows(self) -> None:
        """
        Now that each run_{r} folder exists under lambda_{k}, create one LambdaWindow
        per λ, passing it the correct `sim_params` for that λ.
        """
        suffix = PreparationStage.EQUILIBRATED.file_suffix
        leg_base = Path(self.output_dir)

        for lam_idx in self.lam_indices:
            for run_idx in range(1, self.ensemble_size + 1):
                run_dir = leg_base / f"lambda_{lam_idx}" / f"run_{run_idx}"
                # TODO: temporary value for testing the code
                λ_float = self.sim_config.coul_lambdas[self.leg_type][lam_idx]

                sim_params = {
                    "gmx_exe": "gmx",
                    "mdp_template": str(run_dir / "lambda.template.mdp"),
                    "gro_file": str(
                        run_dir
                        / f"{self.leg_type.name.lower()}{suffix}_{run_idx}_final.gro"
                    ),
                    "top_file": str(
                        run_dir
                        / f"{self.leg_type.name.lower()}{suffix}_{run_idx}_final.top"
                    ),
                    "extra_params": {"lambda_float": λ_float},
                }

                window = LambdaWindow(
                    lam_state=lam_idx,
                    input_dir=self.input_dir,
                    output_dir=str(run_dir),
                    sim_params=sim_params,
                    ensemble_size=self.ensemble_size,
                )
                self._sub_sim_runners.append(window)
                window.setup()

    def run(
        self,
        run_nos: Optional[list[int]] = None,
        adaptive: bool = False,
        runtime: Optional[float] = None,
    ) -> None:
        """
        Run all λ-windows in this Leg.

        Parameters
        ----------
        run_nos : List[int] or None
            If provided, only run those replicate indices per λ.
        adaptive : bool
            Ignored here (passed down to LambdaWindow if needed).
        runtime : float or None
            If adaptive=False, must provide `runtime` (ns).
        """
        run_nos = self._get_valid_run_nos(run_nos)

        for window in self._sub_sim_runners:
            # Pass run_nos & runtime into each LambdaWindow
            window.run(run_nos=run_nos, runtime=runtime or 0.0)

        self._running = True
        if any(win._failed for win in self._sub_sim_runners):
            self._failed = True
            self._finished = False
        else:
            self._failed = False
            self._finished = True

    def run(
        self,
        run_nos: Optional[list[int]] = None,
        adaptive: bool = False,
        runtime: Optional[float] = None,
    ) -> None:
        """
        Run all lambda windows in this leg.

        Parameters
        ----------
        run_nos : List[int] or None
            If provided, only run the specified replica indices (1-based) per λ.
        hpc : bool
            If True, submit each run_{run_idx} folder’s `submit_gmx.sh` to SLURM
            (via `sbatch`). Otherwise run locally via window.run().
        """
        run_nos = self._get_valid_run_nos(run_nos)

        for window in self._sub_sim_runners:
            window.run(run_nos=run_nos, adaptive=adaptive, runtime=runtime)


class Calculation(SimulationRunner):
    """
    Class to set up and run an entire ABFE calculation in GROMACS,
    consisting of two legs (BOUND and FREE), each with a list of lambda states
    and an ensemble of replicas per lambda.
    """

    # Only these two legs for GROMACS
    required_legs = [LegType.BOUND, LegType.FREE]
    _sub_sim_runners: Sequence["Leg"]

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        sim_config: GromacsFepSimulationConfig,
        equil_detection: str = "multiwindow",
        runtime_constant: Optional[float] = 0.001,
        ensemble_size: int = 5,
    ) -> None:
        super().__init__(
            input_dir=input_dir, output_dir=output_dir, ensemble_size=ensemble_size
        )
        self.sim_config = sim_config

    def setup(self) -> None:
        if getattr(self, "setup_complete", False):
            logger.info("Setup already complete. Skipping...")
            return

        output_root = Path(self.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        self.legs = []
        for leg_type in self.required_legs:
            leg_base = output_root / leg_type.name.lower()
            # TODO: temporary value for testing the code
            float_list = self.sim_config.coul_lambdas[leg_type]
            lam_indices = list(range(len(float_list)))
            leg = Leg(
                leg_type=leg_type,
                lam_indices=lam_indices,
                input_dir=self.input_dir,
                output_dir=str(leg_base),
                ensemble_size=self.ensemble_size,
                sim_config=self.sim_config,
            )
            leg.setup()
            self.legs.append(leg)

        self.setup_complete = True
        self._dump()

    def run(
        self,
        run_nos: Optional[list[int]] = None,
        adaptive: bool = False,
        runtime: Optional[float] = None,
    ) -> None:
        """
        Run the entire ABFE calculation, either locally or via SLURM.

        Parameters
        ----------
        run_nos : List[int] or None
            If provided, only run those replica indices (1-based) for each lambda.
        runtime : float, Optional, default: None
            If adaptive is False, runtime must be supplied and stage will run for this
            number of nanoseconds.
        hpc : bool
            If True, submit all windows’ `submit_gmx.sh` to SLURM (nonblocking).
            If False, run everything locally (blocking).
        """
        if not self.setup_complete:
            raise ValueError("Calculation has not been set up yet. Call setup() first.")

        # If SLURM‐mode, build exactly one VirtualQueue for the entire calc:
        # if use_slurm:
        #     vq = VirtualQueue(log_dir=self.output_dir, stream_log_level=logging.INFO)
        # else:
        #     vq = None

        super().run(run_nos=run_nos, adaptive=adaptive, runtime=runtime)
