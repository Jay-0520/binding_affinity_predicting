"""
gromacs_orchestration.py

Functionality for setting up and running an entire ABFE calculation using GROMACS,
consisting of two legs (bound and unbound) and multiple lambda-windows per leg.
In GROMACS, all three alchemical stages (RESTRAIN → DISCHARGE → VANISH) are encoded
in a single MDP file via bonded-, coulomb-, and vdw-lambda vectors, so we do not
instantly split into separate Stage objects. Instead, each lambda window is its own
sub-runner and drives GROMACS with a different `init-lambda-state`.
"""

import logging
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from binding_affinity_predicting.components.simulation import Simulation
from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.data.enums import JobStatus, LegType, PreparationStage
from binding_affinity_predicting.data.schemas import (
    GromacsFepSimulationConfig,
)
from binding_affinity_predicting.hpc_cluster.virtual_queue import Job, VirtualQueue
from binding_affinity_predicting.simulation.mdp_parameters import (
    MDPGenerator,
    create_custom_mdp_generator,
)
from binding_affinity_predicting.simulation.slurm_parameters import (
    SlurmSubmitGenerator,
    create_custom_slurm_generator,
)

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
        mdp_generator: Optional[MDPGenerator] = None,
        slurm_generator: Optional[SlurmSubmitGenerator] = None,
    ) -> None:
        super().__init__(
            input_dir=input_dir, output_dir=work_dir, ensemble_size=ensemble_size
        )
        self.lam_state = lam_state
        self.virtual_queue = virtual_queue
        self.sim_params = sim_params

        # Initialize Pydantic-based generators
        self.mdp_generator = mdp_generator or self._create_default_mdp_generator()
        self.slurm_generator = slurm_generator or self._create_default_slurm_generator()

        # Initialize _sub_sim_runners as list of Simulation objects
        self._sub_sim_runners: list[Simulation] = []  # type: ignore

    def _create_default_mdp_generator(self) -> MDPGenerator:
        """Create default MDP generator with parameters from sim_params."""
        # Extract MDP-specific parameters from sim_params
        mdp_overrides = self.sim_params.get("mdp_overrides", {})

        # Create generator with any custom MDP parameters
        return create_custom_mdp_generator(**mdp_overrides)

    def _create_default_slurm_generator(self) -> SlurmSubmitGenerator:
        """Create default SLURM generator with parameters from sim_params."""
        # Extract SLURM-specific parameters from sim_params
        slurm_overrides = self.sim_params.get("slurm_overrides", {})

        # Create generator with any custom SLURM parameters
        return create_custom_slurm_generator(**slurm_overrides)

    def setup(self) -> None:
        """Set up simulation objects for each run in this lambda window"""
        # Initialize _sub_sim_runners as list of Simulation objects
        self._sub_sim_runners.clear()  # Clear any existing simulations

        for run_no in range(1, self.ensemble_size + 1):
            # Each simulation should run in its own run_X directory
            individual_run_dir = Path(self.output_dir) / f"run_{run_no}"

            # Create run-specific sim_params by formatting templates
            gro_file, top_file = self._prepare_run_specific_files(
                run_no, individual_run_dir
            )

            # Generate MDP file for this run using Pydantic generator
            # self._generate_mdp_file(individual_run_dir, run_no)

            # Create simulation object with Pydantic generators
            sim = Simulation(
                lam_state=self.lam_state,
                gro_file=gro_file,
                top_file=top_file,
                work_dir=individual_run_dir,
                run_index=run_no,
                gmx_exe=self.sim_params.get("gmx_exe", "gmx"),
                bonded_list=self.sim_params.get("bonded_list", []),
                coul_list=self.sim_params.get("coul_list", []),
                vdw_list=self.sim_params.get("vdw_list", []),
                extra_params=self.sim_params.get("extra_params", {}),
                mdp_generator=self.mdp_generator,
                slurm_generator=self.slurm_generator,
                runtime_ns=self.sim_params.get("runtime_ns"),
                mdp_overrides=self.sim_params.get("mdp_overrides", {}),
            )
            # Set up the simulation's MDP file immediately
            sim.setup()

            self._sub_sim_runners.append(sim)

        # Generate SLURM submit scripts for each simulation
        self._generate_submit_scripts()

        # set up the list of "Simulation" object in this case
        super().setup()

    def _prepare_run_specific_files(
        self, run_no: int, run_dir: Path
    ) -> tuple[str, str]:
        """Prepare run-specific file paths by resolving templates."""
        if "gro_file_template" in self.sim_params:
            gro_file = str(
                run_dir / self.sim_params["gro_file_template"].format(run_idx=run_no)
            )
        else:
            gro_file = self.sim_params.get("gro_file", str(run_dir / "system.gro"))

        if "top_file_template" in self.sim_params:
            top_file = str(
                run_dir / self.sim_params["top_file_template"].format(run_idx=run_no)
            )
        else:
            top_file = self.sim_params.get("top_file", str(run_dir / "system.top"))

        return gro_file, top_file

    def _generate_submit_scripts(self) -> None:
        """
        Generate SLURM submit scripts for each simulation.
        This is done in LambdaWindow because it coordinates the ensemble.
        """
        for run_no in range(1, self.ensemble_size + 1):
            sim = self._sub_sim_runners[run_no - 1]

            # Get SLURM parameters from sim_params
            modules_to_load = self.sim_params.get("modules_to_load", [])
            slurm_overrides = self.sim_params.get("slurm_overrides", {})
            pre_commands = self.sim_params.get("pre_commands", [])
            post_commands = self.sim_params.get("post_commands", [])

            # Generate submit script using the simulation's method
            sim.generate_submit_script(
                modules_to_load=modules_to_load,
                slurm_overrides=slurm_overrides,  # params are overrided here
                pre_commands=pre_commands,
                post_commands=post_commands,
            )

            logger.info(
                f"Generated submit script for lambda {self.lam_state}, run {run_no}"
            )

    def run(
        self,
        run_nos: Optional[list[int]] = None,
        runtime: Optional[float] = None,
        use_hpc: bool = True,
    ) -> None:
        """Run simulations for specified run numbers"""
        run_nos = self._get_valid_run_nos(run_nos)

        if use_hpc and self.virtual_queue is not None:
            self._run_hpc(run_nos, runtime)
        else:
            self._run_local(run_nos)  # we don't need runtime here

    def _run_local(self, run_nos: list[int]) -> None:
        """Run simulations locally (blocking)"""
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

    def _run_hpc(self, run_nos: list[int], runtime: Optional[float] = None) -> None:
        """Submit simulations to SLURM via VirtualQueue (non-blocking)"""
        if self.virtual_queue is None:  # need this check to make mypy happy
            raise RuntimeError("`_run_hpc` called without a VirtualQueue")

        self._running = True
        self._submitted_jobs: list[Job] = []

        for run_no in run_nos:
            if run_no < 1 or run_no > self.ensemble_size:
                raise ValueError(
                    f"Invalid run number {run_no}. Must be in [1..{self.ensemble_size}]."
                )
            sim_index = run_no - 1
            sim = self._sub_sim_runners[sim_index]
            sim._submitted_via_hpc = True
            sim.job_status = JobStatus.QUEUED  # It's queued in SLURM

            # Build SLURM submission command
            run_dir = Path(self.output_dir) / f"run_{run_no}"
            submit_script = run_dir / "submit_gmx.sh"
            # Prepare the command list for sbatch
            command_list = [str(submit_script)]
            if runtime is not None:
                command_list.extend(["--runtime", str(runtime)])
            # Set up SLURM output file base
            slurm_file_base = str(
                run_dir / f"slurm_lambda_{self.lam_state}_run_{run_no}"
            )
            # Submit to virtual queue
            job: Job = self.virtual_queue.submit(
                command_list=command_list, slurm_file_base=slurm_file_base
            )
            self._submitted_jobs.append(job)

            logger.info(
                f"Submitted lambda {self.lam_state}, run {run_no} to "
                f"SLURM (job ID: {job.virtual_job_id})"
            )

    def update_simulation_statuses(self) -> None:
        """
        Update simulation statuses based on VirtualQueue job statuses.
        Call this periodically to sync HPC job status with simulation status.
        """
        if not self._submitted_jobs:
            return

        for i, job in enumerate(self._submitted_jobs):
            if i < len(self._sub_sim_runners):
                sim = self._sub_sim_runners[i]

                # Update simulation status based on job status
                if hasattr(job, "status"):
                    if job.status == "FINISHED":
                        sim._finished = True
                        sim.job_status = JobStatus.FINISHED
                        if sim.end_time is None:
                            sim.end_time = datetime.now()
                    elif job.status == "FAILED":
                        sim._failed = True
                        sim.job_status = JobStatus.FAILED
                        if sim.end_time is None:
                            sim.end_time = datetime.now()
                    elif job.status == "RUNNING":
                        sim._running = True
                        sim.job_status = JobStatus.RUNNING
                        if sim.start_time is None:
                            sim.start_time = datetime.now()
                    elif job.status == "QUEUED":
                        sim.job_status = JobStatus.QUEUED

    def __str__(self) -> str:
        """
        Return a string representation of the LambdaWindow class.
        """
        return (
            f"{self.__class__.__name__}"
            f"[lam_state = {self.lam_state},"
            f" ensemble={self.ensemble_size},"
            f" input={Path(self.input_dir).name},"
            f" output={Path(self.output_dir).name}]"
        )


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
        mdp_generator: Optional[MDPGenerator] = None,
        slurm_generator: Optional[SlurmSubmitGenerator] = None,
    ) -> None:
        super().__init__(
            input_dir=input_dir,
            output_dir=output_dir,
            ensemble_size=ensemble_size,
            virtual_queue=virtual_queue,
        )
        self.leg_type = leg_type
        self.lam_indices = lam_indices
        self.sim_config = sim_config

        # Initialize Pydantic-based generators
        self.mdp_generator = mdp_generator or self._create_default_mdp_generator()
        self.slurm_generator = slurm_generator or self._create_default_slurm_generator()
        self._sub_sim_runners: list["LambdaWindow"] = []  # type: ignore[assignment]

    def _create_default_mdp_generator(self) -> MDPGenerator:
        """Create default MDP generator with parameters from sim_config."""
        mdp_overrides = getattr(self.sim_config, "mdp_overrides", {})
        return create_custom_mdp_generator(**mdp_overrides)

    def _create_default_slurm_generator(self) -> SlurmSubmitGenerator:
        """Create default SLURM generator with parameters from sim_config."""
        slurm_overrides = getattr(self.sim_config, "slurm_overrides", {})
        return create_custom_slurm_generator(**slurm_overrides)

    def setup(self) -> None:
        """
        Break down the large setup into discrete steps:
          1) prepare_leg_base()
          2) copy_equilibrated_files()
          3) make_lambda_run_dirs_and_link()
          4) instantiate_lambda_windows()
        """
        leg_base = Path(self.output_dir)
        leg_base.mkdir(parents=True, exist_ok=True)

        # 1) Prepare the “input” subfolder under leg_base
        leg_input = leg_base / "input"
        leg_input.mkdir(exist_ok=True)

        # 2) Copy pre-equilibrated .gro/.top/.itp into leg_input
        self._copy_equilibrated_files(leg_input)

        # 3) For each λ index and each replicate, make run dirs & symlink inputs
        self._make_lambda_run_dirs_and_link(leg_input)

        # 4) Finally, build each LambdaWindow and call its setup()
        self._instantiate_lambda_windows()

    def _copy_equilibrated_files(self, leg_input: Path) -> None:
        """
        Copy all pre-equilibrated .gro/.top (plus .itp for BOUND) from
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

    def _make_lambda_run_dirs_and_link(self, leg_input: Path) -> None:
        """
        Under `leg_base/λ_{k}/run_{r}/`, create folders and symlink or copy:
          - {leg}_equilibrated_{r}_final.gro
          - {leg}_equilibrated_{r}_final.top
          - restraint_{r}.itp    (only for BOUND)
          - lambda.template.mdp
          - submit_gmx.template.sh
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
                    srcfile: Path = leg_input / filename
                    destfile: Path = run_dir / filename

                    if destfile.exists() or destfile.is_symlink():
                        logger.info(f"Removing existing file/symlink: {destfile}")
                        destfile.unlink()

                    os.symlink(os.path.relpath(srcfile, start=run_dir), destfile)

    def _instantiate_lambda_windows(self) -> None:
        """
        Now that each run_{r} folder exists under lambda_{k}, create one LambdaWindow
        per λ, passing it the correct `sim_params` for that λ.
        """
        suffix = PreparationStage.EQUILIBRATED.file_suffix
        leg_base = Path(self.output_dir)

        bonded_full: list[float] = self.sim_config.bonded_lambdas[
            self.leg_type
        ]  # list of floats
        coul_full: list[float] = self.sim_config.coul_lambdas[self.leg_type]
        vdw_full: list[float] = self.sim_config.vdw_lambdas[self.leg_type]

        # Create ONE LambdaWindow per lambda index (not per run)
        for lam_idx in self.lam_indices:
            lam_dir = leg_base / f"lambda_{lam_idx}"

            # TODO: temporary value for testing the code
            λ_float = self.sim_config.coul_lambdas[self.leg_type][lam_idx]

            # Prepare sim_params for this lambda window
            sim_params = {
                "gmx_exe": getattr(self.sim_config, "gmx_exe", "gmx"),
                "gro_file_template": f"{self.leg_type.name.lower()}{suffix}_{{run_idx}}_final.gro",
                "top_file_template": f"{self.leg_type.name.lower()}{suffix}_{{run_idx}}_final.top",
                "extra_params": {"lambda_float": λ_float},
                "bonded_list": bonded_full,
                "coul_list": coul_full,
                "vdw_list": vdw_full,
                # Add parameters from sim_config
                "runtime_ns": getattr(self.sim_config, "runtime_ns", None),
                "mdp_overrides": getattr(self.sim_config, "mdp_overrides", {}),
                "slurm_overrides": getattr(self.sim_config, "slurm_overrides", {}),
                "modules_to_load": getattr(self.sim_config, "modules_to_load", []),
                "pre_commands": getattr(self.sim_config, "pre_commands", []),
                "post_commands": getattr(self.sim_config, "post_commands", []),
            }

            window = LambdaWindow(
                lam_state=lam_idx,
                input_dir=self.input_dir,
                work_dir=str(lam_dir),
                sim_params=sim_params,
                ensemble_size=self.ensemble_size,
                virtual_queue=self.virtual_queue,
                mdp_generator=self.mdp_generator,
                slurm_generator=self.slurm_generator,
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

        if use_hpc and self.virtual_queue:
            self._run_hpc(run_nos, runtime)
        else:
            self._run_local(run_nos, runtime)

    def _run_local(self, run_nos: list[int], runtime: Optional[float] = None) -> None:
        """Run all lambda windows locally (blocking)"""
        logger.info(
            f"Running leg {self.leg_type.name} locally with {len(self._sub_sim_runners)} windows"
        )
        super()._run_local(run_nos)

    def _run_hpc(self, run_nos: list[int], runtime: Optional[float] = None) -> None:
        """Submit all lambda windows to SLURM (non-blocking)"""
        logger.info(
            f"Submitting leg {self.leg_type.name} to SLURM "
            f"with {len(self._sub_sim_runners)} windows"
        )
        super()._run_hpc(run_nos, runtime)

    def __str__(self) -> str:
        """
        Return a string representation of the Leg class.
        """
        return (
            f"{self.__class__.__name__}"
            f"[Leg_type = {self.leg_type},"
            f" ensemble={self.ensemble_size},"
            f" input={Path(self.input_dir).name},"
            f" output={Path(self.output_dir).name}]"
        )


class Calculation(SimulationRunner):
    """
    Class to set up and run an entire ABFE calculation in GROMACS,
    consisting of two legs (BOUND and FREE), each with a list of lambda states
    and an ensemble of replicas per lambda.
    """

    # Only these two legs for GROMACS
    required_legs = [LegType.BOUND]  # , LegType.FREE]

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        sim_config: GromacsFepSimulationConfig,
        equil_detection: str = "multiwindow",
        runtime_constant: Optional[float] = 0.001,
        ensemble_size: int = 5,
        virtual_queue: Optional[VirtualQueue] = None,
        mdp_generator: Optional[MDPGenerator] = None,
        slurm_generator: Optional[SlurmSubmitGenerator] = None,
    ) -> None:
        # Set up virtual queue if not provided
        if virtual_queue is None:
            virtual_queue = VirtualQueue(log_dir=output_dir)

        super().__init__(
            input_dir=input_dir,
            output_dir=output_dir,
            ensemble_size=ensemble_size,
            virtual_queue=virtual_queue,
        )
        self.sim_config = sim_config
        self.equil_detection = equil_detection
        self.runtime_constant = runtime_constant

        # Initialize Pydantic-based generators at the calculation level
        self.mdp_generator = mdp_generator or self._create_default_mdp_generator()
        self.slurm_generator = slurm_generator or self._create_default_slurm_generator()
        self._sub_sim_runners: list[Leg] = []  # type: ignore[assignment]

    def _create_default_mdp_generator(self) -> MDPGenerator:
        """Create default MDP generator from sim_config."""
        mdp_overrides = getattr(self.sim_config, "mdp_overrides", {})
        return create_custom_mdp_generator(**mdp_overrides)

    def _create_default_slurm_generator(self) -> SlurmSubmitGenerator:
        """Create default SLURM generator from sim_config."""
        slurm_overrides = getattr(self.sim_config, "slurm_overrides", {})
        return create_custom_slurm_generator(**slurm_overrides)

    def setup(self) -> None:
        if getattr(self, "setup_complete", False):
            logger.info("Setup already complete. Skipping...")
            return

        logger.info("Setting up ABFE calculation...")

        output_root = Path(self.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

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
                mdp_generator=self.mdp_generator,
                slurm_generator=self.slurm_generator,
            )
            self._sub_sim_runners.append(leg)

        # Set up all legs
        super().setup()
        self.setup_complete = True
        # self._dump()
        logger.info("ABFE calculation setup completed successfully")

    def run(
        self,
        run_nos: Optional[list[int]] = None,
        adaptive: bool = False,
        runtime: Optional[float] = None,
        use_hpc: bool = True,
        run_sync: bool = True,
    ) -> Optional[threading.Thread]:
        """
        Run the entire ABFE calculation, either locally or via SLURM.

        if runs locally, run asynchronously

        Parameters
        ----------
        run_nos : List[int] or None
            If provided, only run those replica indices (1-based) for each lambda.
        runtime : float, Optional, default: None
            If adaptive is False, runtime must be supplied and stage will run for this
            number of nanoseconds.
        hpc : bool
            If True, submit all windows` `submit_gmx.sh` to SLURM (nonblocking).
            If False, run everything locally (blocking).
        """
        if not getattr(self, "setup_complete", False):
            raise ValueError("Calculation has not been set up yet. Call setup() first.")

        run_nos = self._get_valid_run_nos(run_nos)

        # Track execution mode for status checking
        self._last_run_used_hpc = use_hpc

        logger.info(f"Starting ABFE calculation with {len(self._sub_sim_runners)} legs")

        if use_hpc:
            self._run_hpc(run_nos=run_nos, runtime=runtime)
            return None
        else:
            # run locally (blocking)
            if run_sync:
                self._run_local(run_nos=run_nos)
                return
            # now run locally AND asynchronously (non-blocking)
            else:

                def run_in_thread():
                    self.run(
                        run_nos=run_nos,
                        adaptive=adaptive,
                        runtime=runtime,
                        use_hpc=False,
                    )

                thread = threading.Thread(target=run_in_thread, daemon=True)
                thread.start()
                return thread

    def _run_local(
        self,
        run_nos: list[int],
    ) -> None:
        """Run the calculation locally (blocking)"""
        logger.info("Running ABFE calculation locally")
        super()._run_local(run_nos=run_nos)

    def _run_hpc(self, run_nos: list[int], runtime: Optional[float] = None) -> None:
        """Submit the calculation to SLURM (non-blocking)"""
        logger.info("Submitting ABFE calculation to SLURM")
        super()._run_hpc(run_nos=run_nos, runtime=runtime)

    @property
    def legs(self) -> list[Leg]:
        """Convenience property to access legs (for backward compatibility)"""
        return self._sub_sim_runners

    def clean_simulations(self, clean_logs: bool = False) -> None:
        """Clean simulation files from all legs"""
        for leg in self._sub_sim_runners:
            leg.clean_simulations(clean_logs=clean_logs)
