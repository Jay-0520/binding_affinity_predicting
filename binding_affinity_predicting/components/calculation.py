"""Functionality for setting up and running an entire ABFE calculation,
consisting of two legs (bound and unbound) and multiple stages."""

__all__ = ["Calculation"]

import logging
import os
from pathlib import Path
from typing import Optional

from binding_affinity_predicting.components.leg import Leg
from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.components.stage import Stage
from binding_affinity_predicting.data.enums import LegType, PreparationStage

# from binding_affinity_predicting.data.enums import PreparationStage
from binding_affinity_predicting.data.schemas import (
    FepSimulationConfig,
    SystemPreparationConfig,
)

# Notes from the paper - by JJ-2025-05-06
# The simulations with the ligand in solvent collectively make up the free leg,
# while those with the receptor–ligand complex make up the bound leg. Sets of
# calculations where interactions of a given type are introduced or removed
# are termed stages: receptor–ligand restraints were introduced, charges were
# scaled, and Lennard-Jones (LJ) terms were scaled in the restrain, discharge,
#  and vanish stages, respectively

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
        input_dir: str,
        output_dir: str,
        sim_config: FepSimulationConfig,
        equil_detection: str = "multiwindow",
        runtime_constant: Optional[float] = 0.001,
        ensemble_size: int = 5,
    ) -> None:
        """
        Returns
        -------
        None
        """
        super().__init__(
            input_dir=input_dir,
            output_dir=output_dir,
            ensemble_size=ensemble_size,
        )
        self.sim_config = sim_config

        # if not self.loaded_from_pickle:
        #     self.equil_detection = equil_detection
        #     self.runtime_constant = runtime_constant
        #     self.setup_complete: bool = False

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
        pass

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
        if getattr(self, "setup_complete", False):
            logger.info("Setup already complete. Skipping...")
            return

        output_root = Path(self.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        self.legs = []
        for leg_type in self.required_legs:
            leg_base = output_root / leg_type.name.lower()
            # instantiate leg with its Stage objects
            stages = []
            for st in Leg.required_stages[leg_type]:
                lam_list = self.sim_config.lambda_values[leg_type][st]
                stages.append(
                    Stage(
                        stage_type=st,
                        num_lambdas=len(lam_list),
                        ensemble_size=self.ensemble_size,
                        leg_type=leg_type,
                        mdp_template=os.path.join(self.input_dir, "lambda.gmx.mdp"),
                        run_script_template=os.path.join(self.input_dir, "run_gmx.sh"),
                    )
                )
            leg = Leg(
                leg_type=leg_type,
                stages=stages,
                input_dir=self.input_dir,
                output_dir=str(leg_base),
                ensemble_size=self.ensemble_size,
            )
            leg.setup(
                {
                    LegType.BOUND: bound_leg_sysprep_config,
                    LegType.FREE: free_leg_sysprep_config,
                }[leg_type]
            )
            self.legs.append(leg)

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
