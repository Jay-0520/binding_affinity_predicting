"""Functionality for setting up and running an entire ABFE calculation,
consisting of two legs (bound and unbound) and multiple stages."""

__all__ = ["Calculation"]

import logging
import os as _os
import shutil
from pathlib import Path
from typing import Optional

# from binding_affinity_predicting.data.enums import PreparationStage
from binding_affinity_predicting.components.leg import Leg
from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.components.stage import Stage
from binding_affinity_predicting.data.enums import LegType, PreparationStage
from binding_affinity_predicting.data.schemas import SystemPreparationConfig

# Notes from the paper - by JJ-2025-05-06
# The simulations with the ligand in solvent collectively make up the free leg, while those with the receptor–ligand
# complex make up the bound leg. Sets of calculations where interactions of a given type are introduced or removed
# are termed stages: receptor–ligand restraints were introduced, charges were scaled, and Lennard-Jones (LJ) terms
# were scaled in the restrain, discharge, and vanish stages, respectively

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Calculation(SimulationRunner):
    """
    Class to set up and run an entire ABFE calculation, consisting of two legs
    (bound and unbound) and multiple stages.


    """

    # commented out by JJ 2025-05-03
    # required_input_files = [
    #     "run_somd.sh",
    #     "protein.pdb",
    #     "ligand.sdf",
    #     "template_config.cfg",
    # ]  # Waters.pdb is optional

    required_legs = [LegType.FREE, LegType.BOUND]

    def __init__(
        self,
        equil_detection: str = "multiwindow",
        runtime_constant: Optional[float] = 0.001,
        ensemble_size: int = 5,
        input_dir: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        """
        Instantiate a calculation based on files in the input dir. If calculation.pkl exists in the
        base directory, the calculation will be loaded from this file and any arguments
        supplied will be overwritten.

        Parameters
        ----------
        equil_detection : str, Optional, default: "multiwindow"
            Method to use for equilibration detection. Options are:
            - "multiwindow": Use the multiwindow paired t-test method to detect equilibration.
                             This is applied on a per-stage basis.
            - "chodera": Use Chodera's method to detect equilibration.
        runtime_constant: float, Optional, default: 0.001
            The runtime_constant (kcal**2 mol**-2 ns*-1) only affects behaviour if running adaptively, and must
            be supplied if running adaptively. This is used to calculate how long to run each simulation for based on
            the current uncertainty of the per-window free energy estimate, as discussed in the docstring of the run() method.
        runtime_constant : float, Optional, default: 0.001
            The runtime constant to use for the calculation, in kcal^2 mol^-2 ns^-1.
            This must be supplied if running adaptively. Each window is run until the
            SEM**2 / runtime >= runtime_constant.
        ensemble_size : int, Optional, default: 5
            Number of simulations to run in the ensemble.
        base_dir : str, Optional, default: None
            Path to the base directory in which to set up the legs and stages. If None,
            this is set to the current working directory.
        input_dir : str, Optional, default: None
            Path to directory containing input files for the simulations. If None, this
            is set to `current_working_directory/input`.

        Returns
        -------
        None
        """
        super().__init__(
            base_dir=base_dir,
            input_dir=input_dir,
            output_dir=None,
            ensemble_size=ensemble_size,
            dump=False,
        )

        if not self.loaded_from_pickle:
            self.equil_detection = equil_detection
            self.runtime_constant = runtime_constant
            self.setup_complete: bool = False

            # Validate the input
            self._validate_input()

            # Save the state
            self._dump()

    @property
    def legs(self) -> list[Leg]:
        return self._sub_sim_runners

    @legs.setter
    def legs(self, value) -> None:
        logger.info("Modifying/ creating legs")
        self._sub_sim_runners = value

    def _validate_input(self) -> None:
        """Check that the required input files are present in the input directory."""
        # Check backwards, as we care about the most advanced preparation stage
        for prep_stage in reversed(PreparationStage):
            files_absent = False
            for leg_type in Calculation.required_legs:
                for file in Leg.required_input_files[leg_type][prep_stage]:
                    if not _os.path.isfile(f"{self.input_dir}/{file}"):
                        files_absent = True
            # We have the required files for this prep stage for both legs, and this is the most
            # advanced prep stage that files are present for
            if not files_absent:
                self._prep_stage = prep_stage
                self._logger.info(
                    f"Found all required input files for preparation stage {prep_stage.name.lower()}"
                )
                return
        # We didn't find all required files for any of the prep stages
        raise ValueError(
            f"Could not find all required input files for "
            f"any preparation stage. Required files are: {Leg.required_input_files[LegType.BOUND]}"
            f"and {Leg.required_input_files[LegType.FREE]}"
        )

    @property
    def prep_stage(self) -> PreparationStage:
        if self.legs:
            min_prep_stage = PreparationStage.PREEQUILIBRATED
            for leg in self.legs:
                min_prep_stage = min(
                    [min_prep_stage, leg.prep_stage], key=lambda x: x.value
                )
            self._prep_stage = min_prep_stage

        return self._prep_stage

    @property
    def is_complete(self) -> bool:
        f"""Whether the {self.__class__.__name__} has completed."""
        # Check if the overall_stats.dat file exists
        if Path(f"{self.output_dir}/overall_stats.dat").is_file():
            return True

        return False

    def setup(
        self,
        bound_leg_sysprep_config: Optional[SystemPreparationConfig] = None,
        free_leg_sysprep_config: Optional[SystemPreparationConfig] = None,
    ) -> None:
        """
        Set up the calculation. This involves parametrising, equilibrating, and
        deriving restraints for the bound leg. Most of the work is done by the
        Leg class.

        Parameters
        ----------
        bound_leg_sysprep_config: SystemPreparationConfig, opttional, default = None
            The system preparation configuration to use for the bound leg. If None, the default
            configuration is used.
        free_leg_sysprep_config: SystemPreparationConfig, opttional, default = None
            The system preparation configuration to use for the free leg. If None, the default
            configuration is used.
        """
        if getattr(self, 'setup_complete', False):
            logger.info("Setup already complete. Skipping...")
            return

        configs = {
            LegType.BOUND: bound_leg_sysprep_config,
            LegType.FREE: free_leg_sysprep_config,
        }

        # prepare output root
        output_root = Path(self.base_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        # instantiate and setup legs
        self.legs = []
        for leg_type in self.required_legs:
            leg_base = output_root / leg_type.name.lower()
            leg = Leg(
                leg_type=leg_type,
                stages=[
                    Stage(stage, self.input_dir, str(leg_base / stage.name.lower()))
                    for stage in leg.required_stages[leg_type]
                ],
                input_dir=self.input_dir,
                output_dir=str(leg_base),
            )
            leg.setup(configs[leg_type])
            self.legs.append(leg)

        # Save the state
        self.setup_complete = True
        self._dump()

    def run(
        self,
        run_nos: Optional[list[int]] = None,
        adaptive: bool = True,
        runtime: Optional[float] = None,
        runtime_constant: Optional[float] = None,
        parallel: bool = True,
    ) -> None:
        """
        Run all stages and perform analysis once finished. If running adaptively,
        cycles of short runs then optimal runtime estimation are performed, where the optimal
        runtime is estimated according to

        Returns
        -------
        None
        """
        if not self.setup_complete:
            raise ValueError(
                "The calculation has not been set up yet. Please call setup() first."
            )

        if runtime_constant:
            self.recursively_set_attr("runtime_constant", runtime_constant)

        super().run(
            run_nos=run_nos, adaptive=adaptive, runtime=runtime, parallel=parallel
        )
