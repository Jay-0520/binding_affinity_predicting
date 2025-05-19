import copy
import logging
import os
import pathlib
import pickle
import subprocess
from abc import ABC
from itertools import count
from time import sleep
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats

from binding_affinity_predicting.components.simulation_base import \
    SimulationRunner

logger = logging.getLogger(__name__)

# def get_results(self, run_nos: Optional[list[int]] = None) -> dict[any]:
#     """
#     Get the raw results from each sub-simulation runner.

#     Parameters
#     ----------
#     run_nos : List[int], Optional, default=None
#         A list of the run numbers to get results for. If None, all runs are returned.

#     Returns
#     -------
#     results : dict
#         A dict mapping each sub-runner’s name (str(runner)) to the dict
#         that runner.get_results() returns.
#     """
#     run_nos = self._get_valid_run_nos(run_nos)
#     logger.info(
#         f"Gathering raw results for runs {run_nos} from {self.__class__.__name__}..."
#     )
#     results: dict[str, dict[str, np.ndarray]] = {}
#     for runner in self.sim_runners:
#         # TODO: what is a good key?
#         key = str(runner)
#         results[key] = runner.get_results(run_nos=run_nos)
#     return results


def get_simulation_results() -> None:
    """
    I suppose all results are written to the output directory somewhere
    """
    pass


class SimulationAnalyzer:
    def __init__(self, sim_runners: list[SimulationRunner], output_dir: str):
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

    def _check_simulation_status():
        """
        check if all simulations being analysis actually completed running
        and none of them have failed
        """
        raise NotImplementedError("Not implemented yet")

    def analyse(
        self,
        slurm: bool = True,
        run_nos: Optional[list[int]] = None,
        subsampling=False,
        fraction: float = 1,
        plot_rmsds: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Analyse the simulation runner and any
        sub-simulations, and return the overall free energy
        change.

        Parameters
        ----------
        slurm : bool, optional, default=True
            Whether to use slurm for the analysis.
        run_nos : List[int], Optional, default=None
            A list of the run numbers to analyse. If None, all runs are analysed.
        subsampling: bool, optional, default=False
            If True, the free energy will be calculated by subsampling using
            the methods contained within pymbar.
        fraction: float, optional, default=1
            The fraction of the data to use for analysis. For example, if
            fraction=0.5, only the first half of the data will be used for
            analysis. If fraction=1, all data will be used. Note that unequilibrated
            data is discarded from the beginning of simulations in all cases.
        plot_rmsds: bool, optional, default=False
            Whether to plot RMSDS. This is slow and so defaults to False.

        Returns
        -------
        dg_overall : np.ndarray
            The overall free energy change for each of the
            ensemble size repeats.
        er_overall : np.ndarray
            The overall error for each of the ensemble size
            repeats.
        """
        run_nos = self._get_valid_run_nos(run_nos)

        logger.info(f"Analysing runs {run_nos} for {self.__class__.__name__}...")
        dg_overall = np.zeros(len(run_nos))
        er_overall = np.zeros(len(run_nos))

        # Make sure that simulations are all complete and check that none of the simulations have failed
        self._check_simulation_status()

        # Analyse the sub-simulation runners
        for sub_sim_runner in self.sim_runners:
            dg, er = sub_sim_runner.analyse(
                slurm=slurm,
                run_nos=run_nos,
                subsampling=subsampling,
                fraction=fraction,
                plot_rmsds=plot_rmsds,
            )
            # Decide if the component should be added or subtracted
            # according to the dg_multiplier attribute
            dg_overall += dg * sub_sim_runner.dg_multiplier
            er_overall = np.sqrt(er_overall**2 + er**2)

        # Log the overall free energy changes
        logger.info(f"Overall free energy changes: {dg_overall} kcal mol-1")
        logger.info(f"Overall errors: {er_overall} kcal mol-1")

        # Calculate the 95 % confidence interval assuming Gaussian errors
        mean_free_energy = np.mean(dg_overall)
        # Gaussian 95 % C.I.
        conf_int = (
            stats.t.interval(
                0.95,
                len(dg_overall) - 1,
                mean_free_energy,
                scale=stats.sem(dg_overall),
            )[1]
            - mean_free_energy
        )  # 95 % C.I.

        # Write overall MBAR stats to file
        with open(f"{self.output_dir}/overall_stats.dat", "a") as ofile:
            ofile.write(
                "###################################### Free Energies ########################################\n"
            )
            ofile.write(
                f"Mean free energy: {mean_free_energy: .3f} + /- {conf_int:.3f} kcal/mol\n"
            )
            for i in range(self.ensemble_size):
                ofile.write(
                    f"Free energy from run {i + 1}: {dg_overall[i]: .3f} +/- {er_overall[i]:.3f} kcal/mol\n"
                )
            ofile.write(
                "Errors are 95 % C.I.s based on the assumption of a Gaussian distribution of free energies\n"
            )

        # Update internal state with result
        self._delta_g = dg_overall
        self._delta_g_er = er_overall

        return dg_overall, er_overall

    def get_results_df(
        self,
        save_csv: bool = True,
        add_sub_sim_runners: bool = True,
    ) -> pd.DataFrame:
        """
        Return the results in dataframe format

        Parameters
        ----------
        save_csv : bool, optional, default=True
            Whether to save the results as a csv file

        add_sub_sim_runners : bool, optional, default=True
            Whether to show the results from the sub-simulation runners.

        Returns
        -------
        results_df : pd.DataFrame
            A dataframe containing the results
        """
        # Create a dataframe to store the results
        headers = [
            "dg / kcal mol-1",
            "dg_95_ci / kcal mol-1",
            "tot_simtime / ns",
            "tot_gpu_time / GPU hours",
        ]
        results_df = pd.DataFrame(columns=headers)

        if add_sub_sim_runners:
            # Add the results for each of the sub-simulation runners
            for sub_sim_runner in self._sub_sim_runners:
                sub_results_df = sub_sim_runner.get_results_df(save_csv=save_csv)
                results_df = pd.concat([results_df, sub_results_df])

        try:  # To extract the overall free energy changes from a previous call of analyse()
            dgs = self._delta_g
            ers = self._delta_g_er
        except AttributeError:
            raise _AnalysisError(
                f"Analysis has not been performed for {self.__class__.__name__}. Please call analyse() first."
            )
        if dgs is None or ers is None:
            raise _AnalysisError(
                f"Analysis has not been performed for {self.__class__.__name__}. Please call analyse() first."
            )

        # Calculate the 95 % confidence interval assuming Gaussian errors
        mean_free_energy = np.mean(dgs)
        conf_int = (
            stats.t.interval(
                0.95,
                len(dgs) - 1,  # type: ignore
                mean_free_energy,
                scale=stats.sem(dgs),
            )[1]
            - mean_free_energy
        )  # 95 % C.I.

        new_row = {
            "dg / kcal mol-1": round(mean_free_energy, 2),
            "dg_95_ci / kcal mol-1": round(conf_int, 2),
            "tot_simtime / ns": round(self.tot_simtime),
            "tot_gpu_time / GPU hours": round(self.tot_gpu_time),
        }

        # Get the index name
        if hasattr(self, "stage_type"):
            index_prefix = f"{self.stage_type.name.lower()}_"
        elif hasattr(self, "leg_type"):
            index_prefix = f"{self.leg_type.name.lower()}_"
        else:
            index_prefix = ""
        results_df.loc[index_prefix + self.__class__.__name__.lower()] = new_row

        # Get the normalised GPU time
        results_df["normalised_gpu_time"] = (
            results_df["tot_gpu_time / GPU hours"] / self.tot_gpu_time
        )
        # Round to 3 s.f.
        results_df["normalised_gpu_time"] = results_df["normalised_gpu_time"].apply(
            lambda x: round(x, 3)
        )

        if save_csv:
            results_df.to_csv(f"{self.output_dir}/results.csv")

        return results_df

    def analyse_convergence():
        pass

    def aggregate_results():
        pass
