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
import threading
from pathlib import Path
from typing import Optional, Sequence
from time import sleep

from binding_affinity_predicting.components.simulation import Simulation
from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.data.enums import LegType, PreparationStage, JobStatus
from binding_affinity_predicting.data.schemas import (
    GromacsFepSimulationConfig,
)
from binding_affinity_predicting.hpc_cluster.virtual_queue import VirtualQueue, Job

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LambdaWindow(SimulationRunner):
    runtime_attributes = {"_finished": False, "_failed": False, "_dg": None}

    def __init__(
        self,
        lam_state: int,
        input_dir: str,
        work_dir: str,
        sim_params: dict,
        ensemble_size: int = 5,
        virtual_queue: Optional[VirtualQueue] = None,
    ):
        super().__init__(
            input_dir=input_dir, output_dir=work_dir, ensemble_size=ensemble_size
        )
        self.lam_state = lam_state
        self.virtual_queue = virtual_queue
        self.sim_params = sim_params

        # Initialize _sub_sim_runners as list of Simulation objects
        self._sub_sim_runners: list[Simulation] = []  # type: ignore
        for run_no in range(1, ensemble_size + 1):

            sim = Simulation(
                lam_state=lam_state,
                run_index=run_no,
                work_dir=Path(work_dir),
                **sim_params,
            )
            self._sub_sim_runners.append(sim)

    def setup(self) -> None:
        super().setup()

    def run(self, run_nos: Optional[list[int]] = None, runtime: Optional[float] = None, 
            use_hpc: bool = True, *args, **kwargs) -> None:
        """Run simulations for specified run numbers"""
        run_nos = self._get_valid_run_nos(run_nos)

        if use_hpc and self.virtual_queue is not None:
            self._run_hpc(run_nos, runtime)
        else:
            self._run_local(run_nos, runtime)

    def _run_local(self, run_nos: list[int], runtime: Optional[float] = None) -> None:
        """Run simulations locally (blocking)"""
        # Run only the specified simulations
        for run_no in run_nos:
            if run_no < 1 or run_no > self.ensemble_size:
                raise ValueError(
                    f"Invalid run number {run_no}. Must be in [1..{self.ensemble_size}]."
                )
            sim_index = run_no - 1
            if runtime is not None:
                # Pass runtime to simulation if provided
                self._sub_sim_runners[sim_index].run(runtime=runtime)
            else:
                self._sub_sim_runners[sim_index].run()

        # Update status based on simulation results
        self._update_status()

    def _run_hpc(self, run_nos: list[int], runtime: Optional[float] = None) -> None:
        """Submit simulations to SLURM via VirtualQueue (non-blocking)"""
        self._running = True
        self._submitted_jobs: list[Job] = []

        for run_no in run_nos:
            if run_no < 1 or run_no > self.ensemble_size:
                raise ValueError(
                    f"Invalid run number {run_no}. Must be in [1..{self.ensemble_size}]."
                )
            
            # Build SLURM submission command
            run_dir = Path(self.output_dir) / f"run_{run_no}"
            submit_script = run_dir / "submit_gmx.sh"
            
            # Prepare the command list for sbatch
            command_list = [str(submit_script)]
            if runtime is not None:
                command_list.extend(["--runtime", str(runtime)])
            
            # Set up SLURM output file base
            slurm_file_base = str(run_dir / f"slurm_lambda_{self.lam_state}_run_{run_no}")
            
            # Submit to virtual queue
            job = self.virtual_queue.submit(
                command_list=command_list,
                slurm_file_base=slurm_file_base
            )
            self._submitted_jobs.append(job)
            
            logger.info(f"Submitted lambda {self.lam_state}, run {run_no} to SLURM (job ID: {job.virtual_job_id})")

    def kill(self) -> None:
        """Kill all running simulations"""
        if self.virtual_queue and self._submitted_jobs:
            for job in self._submitted_jobs:
                if job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
                    self.virtual_queue.kill(job)
                    logger.info(f"Killed job {job.virtual_job_id}")
        
        # Also kill local simulations if any
        for sim in self._sub_sim_runners:
            if hasattr(sim, 'kill'):
                sim.kill()
        
        self._running = False

    def _update_status(self):
        """
        Update _failed, _finished and _running status based on sub-simulations or SLURM jobs

        This is to check if for a given LambdaWindow (which contains multiple replicas), 
        its replicas have all finished or not.
        """
        if self.virtual_queue and self._submitted_jobs:
            # --- HPC/SLURM branch ---
            finished_jobs = 0
            failed_jobs = 0
            
            for job in self._submitted_jobs:
                if job.status == JobStatus.FINISHED:
                    finished_jobs += 1
                elif job.status == JobStatus.FAILED:
                    failed_jobs += 1
            # if any SLURM job has failed, this window is “failed”
            if failed_jobs > 0:
                self._failed = True
                self._finished = False
                self._running = False
            # If every single SLURM job is marked FINISHED, window is done
            elif finished_jobs == len(self._submitted_jobs):
                self._failed = False
                self._finished = True
                self._running = False
            else:
                # Otherwise, at least one job is still pending/running
                self._running = True
        else:
            # --- Local branch: look at each SimulationRunner’s flags ---
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
        return [sim for sim in self._sub_sim_runners if getattr(sim, "failed", False)]

    @property
    def running(self) -> bool:
        """
        True if any child Simulation(...) is still running.
        """
        if self.virtual_queue and self._submitted_jobs:
            # Update status first
            self._update_status()
            return self._running
        else:
            return any(getattr(sim, "running", False) for sim in self._sub_sim_runners)


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
        virtual_queue: Optional[VirtualQueue] = None,
    ):
        super().__init__(
            input_dir=input_dir, output_dir=output_dir, ensemble_size=ensemble_size
        )
        self.leg_type = leg_type
        self.lam_indices = lam_indices
        self.ensemble_size = ensemble_size
        self.sim_config = sim_config
        self.virtual_queue = virtual_queue
        self._sub_sim_runners: list[LambdaWindow] = []
        self._running = False
        self.running_wins = []  # Track currently running windows

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

        bonded_full = self.sim_config.bonded_lambdas[self.leg_type]  # list of floats
        coul_full = self.sim_config.coul_lambdas[self.leg_type]
        vdw_full = self.sim_config.vdw_lambdas[self.leg_type]

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
                    "bonded_list": bonded_full,
                    "coul_list": coul_full,
                    "vdw_list": vdw_full,
                }

                window = LambdaWindow(
                    lam_state=lam_idx,
                    input_dir=self.input_dir,
                    work_dir=str(run_dir),
                    sim_params=sim_params,
                    ensemble_size=self.ensemble_size,
                    virtual_queue=self.virtual_queue,
                )
                self._sub_sim_runners.append(window)
                window.setup()

    def run(
        self,
        run_nos: Optional[list[int]] = None,
        adaptive: bool = False,
        runtime: Optional[float] = None,
        use_hpc: bool = True,
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
        hpc : bool
            If True, submit to SLURM via VirtualQueue (non-blocking).
            If False, run locally (blocking).
        """
        run_nos = self._get_valid_run_nos(run_nos)

        if use_hpc:
            self._run_hpc(run_nos, runtime)
        else:
            self._run_local(run_nos, runtime)

    def _run_local(self, run_nos: list[int], runtime: Optional[float] = None) -> None:
        """Run all lambda windows locally (blocking)"""
        for window in self._sub_sim_runners:
            window.run(run_nos=run_nos, runtime=runtime, use_hpc=False)

        self._running = True
        if any(win._failed for win in self._sub_sim_runners):
            self._failed = True
            self._finished = False
        else:
            self._failed = False
            self._finished = True

    def _run_hpc(self, run_nos: list[int], runtime: Optional[float] = None) -> None:
        """Submit all lambda windows to SLURM (non-blocking)"""
        self._running = True
        self.running_wins = []

        for window in self._sub_sim_runners:
            window.run(run_nos=run_nos, runtime=runtime, use_hpc=True)
            self.running_wins.append(window)

        logger.info(f"Submitted {len(self.running_wins)} lambda windows to SLURM for leg {self.leg_type.name}")

    def kill(self) -> None:
        """Kill all running lambda windows"""
        for window in self._sub_sim_runners:
            window.kill()
        self._running = False
        self.running_wins = []

    @property
    def running(self) -> bool:
        """True if any lambda window is still running"""
        if self.virtual_queue and self.running_wins:
            # Update running_wins list by removing finished windows
            still_running = []
            for win in self.running_wins:
                if win.running:
                    still_running.append(win)
            self.running_wins = still_running
            return len(self.running_wins) > 0
        else:
            return any(win.running for win in self._sub_sim_runners)

    def wait(self, cycle_pause: int = 60) -> None:
        """Wait for all lambda windows to finish (for HPC runs)"""
        if not self.virtual_queue:
            return  # Nothing to wait for in local mode

        while self.running:
            sleep(cycle_pause)
            # Update virtual queue
            self.virtual_queue.update()
            
            # Update status
            finished_wins = []
            for win in self.running_wins:
                win._update_status()
                if not win.running:
                    finished_wins.append(win)
                    logger.info(f"Lambda window {win.lam_state} finished")

            # Remove finished windows
            for win in finished_wins:
                self.running_wins.remove(win)

        # Update final status
        if any(win._failed for win in self._sub_sim_runners):
            self._failed = True
            self._finished = False
        else:
            self._failed = False
            self._finished = True
        
        self._running = False
        logger.info(f"All lambda windows in leg {self.leg_type.name} completed")


class Calculation(SimulationRunner):
    """
    Class to set up and run an entire ABFE calculation in GROMACS,
    consisting of two legs (BOUND and FREE), each with a list of lambda states
    and an ensemble of replicas per lambda.
    """

    # Only these two legs for GROMACS
    required_legs = [LegType.BOUND]  # , LegType.FREE]
    _sub_sim_runners: Sequence["Leg"]

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        sim_config: GromacsFepSimulationConfig,
        equil_detection: str = "multiwindow",
        runtime_constant: Optional[float] = 0.001,
        ensemble_size: int = 5,
        virtual_queue: Optional[VirtualQueue] = None,
    ) -> None:
        super().__init__(
            input_dir=input_dir, output_dir=output_dir, ensemble_size=ensemble_size
        )
        self.sim_config = sim_config
        self.equil_detection = equil_detection
        self.runtime_constant = runtime_constant
        self._running = False
        self.run_thread = None
        self.kill_thread = False

        # Set up virtual queue if not provided
        if virtual_queue is None:
            self.virtual_queue = VirtualQueue(
                log_dir=output_dir, 
            )
        else:
            self.virtual_queue = virtual_queue

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
                virtual_queue=self.virtual_queue,
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
        use_hpc: bool = True,
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

        run_nos = self._get_valid_run_nos(run_nos)


        if use_hpc:
            # Run in background thread for HPC mode
            self.run_thread = threading.Thread(
                target=self._run_hpc_threaded,
                args=(run_nos, adaptive, runtime),
                name="ABFE_Calculation"
            )
            self.run_thread.start()
        else:
            # Run locally (blocking)
            self._run_local(run_nos, adaptive, runtime)
        

    def _run_local(self, run_nos: list[int], adaptive: bool, runtime: Optional[float]) -> None:
        """Run the calculation locally (blocking)"""
        logger.info(f"Starting ABFE calculation locally with {len(self.legs)} legs")

        # Run each leg
        for leg in self.legs:
            logger.info(f"Running leg: {leg.leg_type.name}")
            leg.run(run_nos=run_nos, adaptive=adaptive, runtime=runtime, use_hpc=False)

        # Update overall status
        if any(leg._failed for leg in self.legs):
            self._failed = True
            self._finished = False
        else:
            self._failed = False
            self._finished = all(leg._finished for leg in self.legs)

        logger.info(
            f"ABFE calculation completed. Failed: {self._failed}, Finished: {self._finished}"
        )

    def _run_hpc_threaded(self, run_nos: list[int], adaptive: bool, runtime: Optional[float]) -> None:
        """Run the calculation via SLURM in a background thread"""
        try:
            self.kill_thread = False
            self._running = True
            
            logger.info(f"Starting ABFE calculation via SLURM with {len(self.legs)} legs")

            # Submit all legs to SLURM
            for leg in self.legs:
                if self.kill_thread:
                    logger.info("Kill thread requested: stopping calculation")
                    return
                    
                logger.info(f"Submitting leg: {leg.leg_type.name}")
                leg.run(run_nos=run_nos, adaptive=adaptive, runtime=runtime, use_hpc=True)

            # Wait for all legs to complete
            self._wait_for_completion()

        except Exception as e:
            logger.exception(f"Error in ABFE calculation: {e}")
            self._failed = True
        finally:
            self._running = False

    def _wait_for_completion(self, cycle_pause: int = 60) -> None:
        """Wait for all legs to complete"""
        while any(leg.running for leg in self.legs):
            if self.kill_thread:
                logger.info("Kill thread requested: stopping wait")
                return
                
            sleep(cycle_pause)
            
            # Update virtual queue
            self.virtual_queue.update()
            
            # Check leg status
            for leg in self.legs:
                if not leg.running and leg in self.legs:
                    status = "FAILED" if leg._failed else "FINISHED"
                    logger.info(f"Leg {leg.leg_type.name} {status}")

        # Update overall status
        if any(leg._failed for leg in self.legs):
            self._failed = True
            self._finished = False
        else:
            self._failed = False
            self._finished = all(leg._finished for leg in self.legs)

        logger.info(
            f"ABFE calculation completed. Failed: {self._failed}, Finished: {self._finished}"
        )

    def kill(self) -> None:
        """Kill all running simulations"""
        logger.info("Killing ABFE calculation...")
        self.kill_thread = True
        
        for leg in self.legs:
            leg.kill()
        
        if self.virtual_queue:
            # Kill all remaining jobs in the virtual queue
            for job in self.virtual_queue.queue:
                if job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
                    self.virtual_queue.kill(job)
        
        self._running = False
        logger.info("ABFE calculation killed")

    def wait(self) -> None:
        """Wait for the calculation to complete (for HPC runs)"""
        if self.run_thread and self.run_thread.is_alive():
            self.run_thread.join()
        elif self.virtual_queue:
            # If running without threading, wait for virtual queue
            self.virtual_queue.wait()

    @property
    def running(self) -> bool:
        """True if the calculation is currently running"""
        if self.run_thread and self.run_thread.is_alive():
            return True
        return self._running or any(leg.running for leg in getattr(self, 'legs', []))