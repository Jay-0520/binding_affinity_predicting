import logging
import pathlib
import threading
from abc import ABC
from itertools import count
from pathlib import Path
from time import sleep
from typing import Optional

from binding_affinity_predicting.components.data.enums import JobStatus
from binding_affinity_predicting.components.simulation_fep.utils import (
    ensure_dir_exist,
)
from binding_affinity_predicting.components.supercluster.virtual_queue import (
    Job,
    VirtualQueue,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimulationRunner(ABC):
    """An abstract base class for simulation runners. Note that
    self._sub_sim_runners (a list of SimulationRunner objects controlled
    by the current SimulationRunner) must be set in order to use methods
    such as run()

    handles only simulation workflow (orchestration, file management, running jobs, etc).

    """

    # Count the number of instances so we can name things uniquely
    # for each instance
    class_count = count()

    # Create list of files to be deleted by self.clean()
    run_files = ["*.png", "overall_stats.dat", "results.csv"]

    # Create a dict of attributes which can be modified by algorithms when
    # running the simulation, but which should be reset if the user wants to
    # re-run. This takes the form {attribute_name: reset_value}
    runtime_attributes: dict = {"_finished": False, "_failed": False, "_running": False}

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        ensemble_size: int = 5,
        save_state_on_init: bool = True,
        virtual_queue: Optional[VirtualQueue] = None,
    ) -> None:
        # Set up the directories (which may be overwritten if the
        # simulation runner is subsequently loaded from a pickle file)
        # Make sure that we always use absolute paths

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ensemble_size = ensemble_size
        self.virtual_queue = virtual_queue

        # Initialize sub-runners and runtime flags
        self._sub_sim_runners: list[SimulationRunner] = []
        self._submitted_jobs: list[Job] = []  # For HPC jobs
        self._init_runtime_attributes()

        ensure_dir_exist(Path(self.input_dir))
        ensure_dir_exist(Path(self.output_dir))

        # (Optional) save initial state if needed
        if save_state_on_init:
            logger.info(
                f"Saving {self.__class__.__name__} state to "
                f"{self.input_dir}/{self.__class__.__name__}.pkl"
            )
            # self._dump()  # implement pickling if needed

    def _init_runtime_attributes(self) -> None:
        # Reset any runtime flags defined in subclasses
        for attribute, value in self.runtime_attributes.items():
            setattr(self, attribute, value)

    def setup(self) -> None:
        """Recursively set up all sub-runners."""
        logger.info(f"Setting up {self.__class__.__name__}...")
        for sub_sim_runner in self._sub_sim_runners:
            sub_sim_runner.setup()

    def run(
        self, run_nos: Optional[list[int]] = None, *args, **kwargs
    ) -> Optional[threading.Thread]:
        """
        Recursively run all sub-runners. Subclasses can override to insert custom logic
        before/after sub-runners are launched.
        """
        run_nos = self._get_valid_run_nos(run_nos)

        logger.info(f"Running run numbers {run_nos} for {self.__class__.__name__}...")
        for sub_sim_runner in self._sub_sim_runners:
            sub_sim_runner.run(run_nos, *args, **kwargs)
        return None

    def kill(self) -> None:
        """Kill all running simulations and jobs."""
        logger.info(f"Killing {self.__class__.__name__}...")

        # Kill HPC jobs if any
        if self.virtual_queue and self._submitted_jobs:
            for job in self._submitted_jobs:
                if job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
                    self.virtual_queue.kill(job)
                    logger.info(f"Killed job {job.virtual_job_id}")

        # Recursively kill all sub-runners
        for sub_sim_runner in self._sub_sim_runners:
            sub_sim_runner.kill()

        self._running = False

    def wait(self, cycle_pause: int = 60) -> None:
        """
        Wait for all simulations to complete. For HPC runs, periodically check status.
        """
        if not self.virtual_queue:
            # Local mode - simple wait
            sleep(30)
            while self.running:
                sleep(30)
        else:
            # HPC mode - update virtual queue and check status
            while self.running:
                sleep(cycle_pause)
                self.virtual_queue.update()
                self._update_status()

    def _update_status(self):
        """
        Update _failed, _finished and _running status based on sub-simulations or jobs.
        This method should be overridden by subclasses for specific logic.
        """
        if self.virtual_queue and self._submitted_jobs:
            # --- HPC/SLURM branch ---
            finished_jobs = sum(
                1 for job in self._submitted_jobs if job.status == JobStatus.FINISHED
            )
            failed_jobs = sum(
                1 for job in self._submitted_jobs if job.status == JobStatus.FAILED
            )

            # If any SLURM job has failed, this window is “failed”
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
            # --- Local branch: look at each SimulationRunner`s flags ---
            if self._sub_sim_runners:
                if any(getattr(sub, "_failed", False) for sub in self._sub_sim_runners):
                    self._failed = True
                    self._finished = False
                    self._running = False
                elif all(
                    getattr(sub, "_finished", False) for sub in self._sub_sim_runners
                ):
                    self._failed = False
                    self._finished = True
                    self._running = False
                else:
                    self._running = any(sub.running for sub in self._sub_sim_runners)

    def _run_local(self, run_nos: list[int], runtime: Optional[float] = None) -> None:
        """Run simulations locally (blocking). Override in subclasses."""
        for sub_runner in self._sub_sim_runners:
            sub_runner.run(run_nos=run_nos, runtime=runtime, use_hpc=False)
        self._update_status()

    def _run_hpc(self, run_nos: list[int], runtime: Optional[float] = None) -> None:
        """Submit simulations to HPC (non-blocking). Override in subclasses."""
        self._running = True
        for sub_runner in self._sub_sim_runners:
            sub_runner.run(run_nos=run_nos, runtime=runtime, use_hpc=True)

    @property
    def running(self) -> bool:
        """True if this runner or any sub-runner is currently running."""
        self._update_status()

        # first check our own status
        if hasattr(self, "_running") and self._running:
            return True

        # check sub-runners; if there are child runners
        if self._sub_sim_runners:
            return any(sub.running for sub in self._sub_sim_runners)

        # if no child runners -> return False
        return False

    @property
    def finished(self) -> bool:
        """True if all sub-runners are finished and none failed."""
        self._update_status()

        # if there are no children, simply look at this runner`s own _finished attribute
        if not self._sub_sim_runners:
            return getattr(self, "_finished", False)

        return (
            all(getattr(sub, "finished", False) for sub in self._sub_sim_runners)
            and not self.failed  # the runner and any of its descendants cannot be failed
        )

    @property
    def failed(self) -> bool:
        """True if this runner or any sub-runner has failed."""
        self._update_status()

        # Check our own status first
        if hasattr(self, "_failed") and self._failed:
            return True

        # Then check sub-runners
        if self._sub_sim_runners:
            return any(sub.failed for sub in self._sub_sim_runners)

        return False

    @property
    def failed_simulations(self) -> list["SimulationRunner"]:
        """
        Return a flat list of any descendant sub-runners whose `.failed == True`.
        """
        result = []

        # Add ourselves if we failed
        if hasattr(self, "_failed") and self._failed:
            result.append(self)

        # Recursively check sub-runners
        for sub in self._sub_sim_runners:
            if sub.failed:
                result.append(sub)
            # Get failed simulations from sub-runners recursively
            result.extend(sub.failed_simulations)

        return result

    def get_tot_simulation_time(self, run_nos: Optional[list[int]] = None) -> float:
        """
        Get the total simulation time in ns for the {self.__class__.__name__}.
        and any sub-simulation runners.

        Parameters
        ----------
        run_nos : List[int], Optional, default=None
            A list of the run numbers to analyse. If None, all runs are analysed.

        Returns
        -------
        tot_simtime : float
            The total simulation time in ns.
        """
        run_nos = self._get_valid_run_nos(run_nos)
        return sum(
            [
                sub_sim_runner.get_tot_simulation_time(run_nos=run_nos)
                for sub_sim_runner in self._sub_sim_runners
            ]
        )  # ns

    def get_tot_gpu_time(self, run_nos: Optional[list[int]] = None) -> float:
        """
        Get the total simulation time in GPU hours for the {self.__class__.__name__}.
        and any sub-simulation runners.

        Parameters
        ----------
        run_nos : List[int], Optional, default=None
            A list of the run numbers to analyse. If None, all runs are analysed.

        Returns
        -------
        tot_gpu_time : float
            The total simulation time in GPU hours.
        """
        run_nos = self._get_valid_run_nos(run_nos)
        return sum(
            [
                sub_sim_runner.get_tot_gpu_time(run_nos=run_nos)
                for sub_sim_runner in self._sub_sim_runners
            ]
        )

    @property
    def tot_simtime(self) -> float:
        """
        The total simulation time in ns for the {self.__class__.__name__} and
        any sub-simulation runners.
        """
        return sum(
            [
                sub_sim_runner.get_tot_simulation_time()
                for sub_sim_runner in self._sub_sim_runners
            ]
        )  # ns

    @property
    def tot_gpu_time(self) -> float:
        """
        The total simulation time in GPU hours for the {self.__class__.__name__}
        and any sub-simulation runners.
        """
        return sum(
            [
                sub_sim_runner.get_tot_gpu_time()
                for sub_sim_runner in self._sub_sim_runners
            ]
        )  # GPU hours

    def is_equilibrated(self, run_nos: Optional[list[int]] = None) -> bool:
        """
        Whether the {self.__class__.__name__} is equilibrated. This updates
        the _equilibrated and _equil_time attributes of the lambda windows,
        which are accessed by the equilibrated and equil_time properties.

        Parameters
        ----------
        run_nos : List[int], Optional, default=None
            A list of the run numbers to check for equilibration. If None, all runs are analysed.

        Returns
        -------
        equilibrated : bool
            Whether the {self.__class__.__name__} is equilibrated.
        """
        run_nos = self._get_valid_run_nos(run_nos)
        return all(
            [
                sub_sim_runner.is_equilibrated(run_nos=run_nos)
                for sub_sim_runner in self._sub_sim_runners
            ]
        )

    @property
    def equilibrated(self) -> float:
        f"""Whether the {self.__class__.__name__} is equilibrated."""
        return all(
            [sub_sim_runner.equilibrated for sub_sim_runner in self._sub_sim_runners]
        )

    @property
    def equil_time(self) -> float:
        """
        The equilibration time, per member of the ensemble, in ns, for the and
        any sub-simulation runners.
        """
        return sum(
            [sub_sim_runner.equil_time for sub_sim_runner in self._sub_sim_runners]
        )  # ns

    def _get_valid_run_nos(self, run_nos: Optional[list[int]] = None) -> list[int]:
        """
        Check the requested run numbers are valid, and return
        a list of all run numbers if None was passed.

        Parameters
        ----------
        run_nos : List[int], Optional, default=None
            A list of the run numbers to run. If None, all runs are returned.

        Returns
        -------
        run_nos : List[int]
            A list of valid run numbers.
        """
        if run_nos is not None:
            # Check that the run numbers correspond to valid runs
            if any([run_no > self.ensemble_size for run_no in run_nos]):
                raise ValueError(
                    f"Invalid run numbers {run_nos}. All run numbers must be less than "
                    "or equal to {self.ensemble_size}"
                )
            # Check that no run numbers are repeated
            if len(run_nos) != len(set(run_nos)):
                raise ValueError(
                    f"Invalid run numbers {run_nos}. All run numbers must be unique"
                )
            # Check that the run numbers are greater than 0
            if any([run_no < 1 for run_no in run_nos]):
                raise ValueError(
                    f"Invalid run numbers {run_nos}. All run numbers must be greater than 0"
                )
        else:
            run_nos = list(range(1, self.ensemble_size + 1))

        return run_nos

    def __str__(self) -> str:
        """
        Return a string representation of the simulation runner.
        """
        return (
            f"{self.__class__.__name__}"
            f"[ensemble={self.ensemble_size},"
            f" input={pathlib.Path(self.input_dir).name},"
            f" output={pathlib.Path(self.output_dir).name}]"
        )

    def clean_simulations(self, clean_logs=False) -> None:
        raise NotImplementedError()

    def reset(self, reset_sub_sims: bool = True) -> None:
        """
        Reset all attributes changed by the runtime
        algorithms to their default values.

        Parameters
        ----------
        reset_sub_sims : bool, default=True
            If True, also reset any sub-simulation runners.
        """
        for attr, value in self.__class__.runtime_attributes.items():
            logger.info(f"Resetting the attribute {attr} to {value}.")
            setattr(self, attr, value)

        if reset_sub_sims:
            if hasattr(self, "_sub_sim_runners"):
                for sub_sim_runner in self._sub_sim_runners:
                    sub_sim_runner.reset()

    # TDDO: maybe not needed
    def add_subrunner(self, runner: "SimulationRunner") -> None:
        raise NotImplementedError()

    def save(self) -> None:
        """Save the current state of the simulation object to a pickle file."""
        self._dump()

    def _dump(self) -> None:
        """Dump the current state of the simulation object to a pickle file, and do
        the same for any sub-simulations."""
        # with open(f"{self.output_dir}/{self.__class__.__name__}.pkl", "wb") as ofile:
        #     pickle.dump(self._picklable_copy.__dict__, ofile)
        # for sub_sim_runner in self._sub_sim_runners:
        #     sub_sim_runner._dump()
        raise NotImplementedError()
