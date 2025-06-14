"""
NOTE This whole script is a work in progress. by JJH 2025-06-13

"""

import logging

import pandas as pd

from binding_affinity_predicting.components.simulation_base import SimulationRunner

logger = logging.getLogger(__name__)


def get_simulation_results() -> None:
    """
    I suppose all results are written to the output directory somewhere
    """
    pass


class SimulationAnalyzer:
    def __init__(self, sim_runners: list[SimulationRunner], output_dir: str) -> None:
        """
        Abstract base class for simulation analysis.
        Handles only simulation analysis (orchestration, file management, running jobs, etc).
        Parameters
        ----------
        sim_runners : list[SimulationRunner]
            A list of SimulationRunner objects to be analysed.
        output_dir : str
            The directory where the analysis results will be saved.
        """
        self.sim_runners = sim_runners
        self.output_dir = output_dir

    def _check_simulation_status(self) -> None:
        raise NotImplementedError("Not implemented yet")

    def analyse(
        self,
    ) -> None:
        raise NotImplementedError("Not implemented yet")

    def get_results_df(
        self,
    ) -> pd.DataFrame:
        raise NotImplementedError("Not implemented yet")

    def analyse_convergence(self) -> None:
        raise NotImplementedError("Not implemented yet")

    def aggregate_results(self) -> None:
        raise NotImplementedError("Not implemented yet")
