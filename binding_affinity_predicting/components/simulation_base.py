import logging
import os
import pathlib
from abc import ABC
from itertools import count
from pathlib import Path
from time import sleep
from typing import Optional, Sequence

from binding_affinity_predicting.components.utils import (
    ensure_dir_exist,
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
    runtime_attributes: dict = {}

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        ensemble_size: int = 5,
        save_state_on_init: bool = True,
    ) -> None:
        # Set up the directories (which may be overwritten if the
        # simulation runner is subsequently loaded from a pickle file)
        # Make sure that we always use absolute paths

        self.input_dir = input_dir
        self.output_dir = output_dir
        ensure_dir_exist(Path(self.input_dir))
        ensure_dir_exist(Path(self.output_dir))

        # Add the dg_multiplier
        self.ensemble_size = ensemble_size

        # Check if we are starting from a previous simulation runner
        if os.path.exists(f"{self.input_dir}/{self.__class__.__name__}.pkl"):
            logger.info(
                f"Loading previous {self.__class__.__name__}. Any arguments will be overwritten..."
            )
        else:
            self._sub_sim_runners: Sequence["SimulationRunner"] = []
            self._init_runtime_attributes()
            # Save state or not
            if save_state_on_init:
                logger.info(
                    f"Saving {self.__class__.__name__} state "
                    f"to {self.input_dir}/{self.__class__.__name__}.pkl"
                )

    def _init_runtime_attributes(self) -> None:
        for attribute, value in self.runtime_attributes.items():
            setattr(self, attribute, value)

    def run(self, run_nos: Optional[list[int]] = None, *args, **kwargs) -> None:
        """
        Run the simulation runner and any sub-simulation runners.
        Parameters
        ----------
        run_nos : List[int], Optional, default=None
            A list of the run numbers to run. If None, all runs are run.
        """
        run_nos = self._get_valid_run_nos(run_nos)

        logger.info(f"Running run numbers {run_nos} for {self.__class__.__name__}...")
        for sub_sim_runner in self._sub_sim_runners:
            sub_sim_runner.run(run_nos=run_nos, *args, **kwargs)

    def setup(self) -> None:
        """Set up the {self.__class__.__name__} and all sub-simulation runners."""
        logger.info(f"Setting up {self.__class__.__name__}...")
        for sub_sim_runner in self._sub_sim_runners:
            sub_sim_runner.setup()

    def kill(self) -> None:
        """Kill all sub-simulation runners."""
        logger.info(f"Killing {self.__class__.__name__}...")
        for sub_sim_runner in self._sub_sim_runners:
            sub_sim_runner.kill()

    def wait(self) -> None:
        """Wait for the {self.__class__.__name__} to finish running."""
        # Give the simulation runner a chance to start
        sleep(30)
        while self.running:
            sleep(30)  # Check every 30 seconds

    @property
    def running(self) -> bool:
        return any(sub.running for sub in self._sub_sim_runners)

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

    @property
    def failed_simulations(self) -> list["SimulationRunner"]:
        """The failed sub-simulation runners"""
        return [
            failure
            for sub_sim_runner in self._sub_sim_runners
            for failure in sub_sim_runner.failed_simulations
        ]

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
        pass
