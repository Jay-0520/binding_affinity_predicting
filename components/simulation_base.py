"""Abstract base class for simulation runners."""

from __future__ import annotations

import copy 
import logging 
import os 
import pathlib 
import pickle 
import subprocess 
from abc import ABC
from itertools import count 
from time import sleep 
from typing import Any 
from typing import Optional
from typing import Tuple
from typing import Union
from binding_affinity_predicting.components.utils import load_simulation_state, ensure_dir_exist, dump_simulation_state, load_simulation_state

import numpy as np
import pandas as pd
import scipy.stats as stats


logger = logging.getLogger(__name__)
# from ..analyse.exceptions import AnalysisError as _AnalysisError
# from ..analyse.plot import plot_convergence as _plot_convergence
# from ..analyse.plot import plot_sq_sem_convergence as _plot_sq_sem_convergence
# from ._logging_formatters import _A3feFileFormatter, _A3feStreamFormatter


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
    runtime_attributes = {}

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        dg_multiplier: int = 1,
        ensemble_size: int = 5,
        update_paths: bool = True,
        save_state_on_init: bool = True,
    ) -> None:
        # Set up the directories (which may be overwritten if the
        # simulation runner is subsequently loaded from a pickle file)
        # Make sure that we always use absolute paths

        self.input_dir = input_dir
        self.output_dir = output_dir
        ensure_dir_exist(self.input_dir)
        ensure_dir_exist(self.output_dir)


        # Add the dg_multiplier
        if dg_multiplier not in [-1, 1]:
            raise ValueError(
                f"dg_multiplier must be either +1 or -1, not {dg_multiplier}."
            )
        self.dg_multiplier = dg_multiplier
        self.ensemble_size = ensemble_size

        # Check if we are starting from a previous simulation runner
        if pathlib.Path(f"{self.input_dir}/{self.__class__.__name__}.pkl").is_file():
            load_simulation_state(
                update_paths=update_paths
            )  # May overwrite the above attributes and options

        else:
            # Initialise sub-simulation runners with an empty list
            self._sub_sim_runners: list[SimulationRunner] = []

            # Initialise runtime attributes with default values
            for attribute, value in self.runtime_attributes.items():
                setattr(self, attribute, value)

            # Save state
            if save_state_on_init:
                dump_simulation_state()


    def _init_runtime(self) -> None:
        # Initialize attributes for runtime
        self.runtime_state: dict[str, Any] = {}

    def add_subrunner(self, runner: SimulationRunner) -> None:
        self._sub_sim_runners.append(runner)

    def run(self, run_nos: Optional[list[int]] = None) -> None:
        run_nos = run_nos or list(range(1, self.config.ensemble_size + 1))
        logger.info(f"Running runs: {run_nos}")
        for sub in self._sub_sim_runners:
            sub.run(run_nos)

    def kill(self) -> None:
        logger.info("Killing runs")
        for sub in self._sub_sim_runners:
            sub.kill()

    def wait(self) -> None:
        logger.info("Waiting for completion...")
        while any(sub.running for sub in self._sub_sim_runners):
            sleep(30)

    @property
    def running(self) -> bool:
        return any(sub.running for sub in self._sub_sim_runners)


    def get_tot_simtime(self, run_nos: Optional[list[int]] = None) -> float:
        f"""
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
                sub_sim_runner.get_tot_simtime(run_nos=run_nos)
                for sub_sim_runner in self._sub_sim_runners
            ]
        )  # ns

    def get_tot_gpu_time(self, run_nos: Optional[list[int]] = None) -> float:
        f"""
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
        f"""The total simulation time in ns for the {self.__class__.__name__} and any sub-simulation runners."""
        return sum(
            [
                sub_sim_runner.get_tot_simtime()
                for sub_sim_runner in self._sub_sim_runners
            ]
        )  # ns

    @property
    def tot_gpu_time(self) -> float:
        f"""The total simulation time in GPU hours for the {self.__class__.__name__} and any sub-simulation runners."""
        return sum(
            [
                sub_sim_runner.get_tot_gpu_time()
                for sub_sim_runner in self._sub_sim_runners
            ]
        )  # GPU hours
    
    @property
    def failed_simulations(self) -> list[SimulationRunner]:
        """The failed sub-simulation runners"""
        return [
            failure
            for sub_sim_runner in self._sub_sim_runners
            for failure in sub_sim_runner.failed_simulations
        ]

    def is_equilibrated(self, run_nos: Optional[list[int]] = None) -> bool:
        f"""
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
                    f"Invalid run numbers {run_nos}. All run numbers must be less than or equal to {self.ensemble_size}"
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
    
    
    def get_results(self, run_nos: Optional[list[int]] = None) -> dict[any]:
        """
        Get the results of the simulation.

        Parameters
        ----------
        run_nos : List[int], Optional, default=None
            A list of the run numbers to get results for. If None, all runs are returned.

        Returns
        -------
        results : dict
            A dictionary of the results of the simulation.
        """
        logger.info(f"Getting results for runs {run_nos} for {self.__class__.__name__}......")
        run_nos = self._get_valid_run_nos(run_nos)
        return {
            sub_sim_runner.run: sub_sim_runner.get_results(run_nos=run_nos)
            for sub_sim_runner in self._sub_sim_runners
            if sub_sim_runner.run in run_nos
        }


    def clean(self, clean_logs: bool = False) -> None:
        for pattern in ["*.png", "results.csv"]:
            for p in self.base_dir.glob(pattern):
                p.unlink()
        if clean_logs:
            log = self.base_dir / f"{self.__class__.__name__}.log"
            log.write_text("")
