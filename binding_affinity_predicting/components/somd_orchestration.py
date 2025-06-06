"""Functionality for setting up and running an entire ABFE calculation,
consisting of two legs (bound and unbound) and multiple stages.
This is a simplifed version from this repo: https://github.com/michellab/a3fe
"""

__all__ = ["Calculation"]

import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Sequence

from binding_affinity_predicting.components.simulation import Simulation
from binding_affinity_predicting.components.simulation_base import SimulationRunner

# from binding_affinity_predicting.data.enums import PreparationStage
from binding_affinity_predicting.data.enums import LegType, PreparationStage, StageType
from binding_affinity_predicting.data.schemas import (
    SomdFepSimulationConfig,
)

# Notes from the paper - by JJ-2025-05-06
# The simulations with the ligand in solvent collectively make up the free leg, while those
# with the receptor-ligand complex make up the bound leg. Sets of calculations where
# interactions of a given type are introduced or removed are termed stages: receptor-ligand
# restraints were introduced, charges were scaled, and Lennard-Jones (LJ) terms
# were scaled in the restrain, discharge, and vanish stages, respectively

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: to update and refactor the code below - JJH 2025-05-31
class Stage(SimulationRunner):
    def __init__(
        self,
        stage_type: StageType,
        num_lambdas: int,
        ensemble_size: int,
        leg_type: LegType,
        mdp_template: str,
        run_script_template: str,
    ) -> None:

        # we`ll override dirs later in Leg.setup()
        super().__init__(input_dir="", output_dir="", ensemble_size=ensemble_size)
        self.stage_type = stage_type
        # e.g. lam_list = [0.0, 0.1, ...]  - here just indices
        self.lam_list = list(range(num_lambdas))
        self.mdp_template = Path(mdp_template)
        self.run_script_template = Path(run_script_template)
        self.leg_type = leg_type

    def _make_window(self, lam: float):
        pass

    def setup(self):
        out_root = Path(self.output_dir)

        for lam in self.lam_list:
            lam_dir = out_root / f"lambda_{lam}"
            lam_dir.mkdir(parents=True, exist_ok=True)

            for run in range(1, self.ensemble_size + 1):
                run_dir = lam_dir / f"run_{run}"
                run_dir.mkdir(parents=True, exist_ok=True)

                # symlink exactly the per-run equilibration & restraint files
                # assume self.input_dir still points at the stage-level "input/" folder
                stage_in = Path(self.input_dir)
                suffix = PreparationStage.EQUILIBRATED.file_suffix
                gro_src = stage_in / f"{self.leg_type.name}{suffix}_{run}_final.gro"
                top_src = stage_in / f"{self.leg_type.name}{suffix}_{run}_final.top"
                rest_src = stage_in / f"restraint_{run}.itp"

                for src in (gro_src, top_src, rest_src):
                    if not src.exists():
                        raise FileNotFoundError(
                            f"Expected {src} for λ={lam}, run={run}"
                        )

                    # If there's already something here, remove it so we can re-link:
                    dest = run_dir / src.name
                    if dest.exists() or dest.is_symlink():
                        dest.unlink()

                    (dest).symlink_to(src.resolve())

                # write a fresh MDP in run_dir
                self._write_mdp(lam_state=lam, dest_dir=run_dir)

                # write a fresh run script in run_dir
                self._write_run_script(lam_state=lam, run_index=run, dest_dir=run_dir)

    def _write_mdp(self, lam_state: int, dest_dir: Path):
        """Copy the MDP template into dest_dir and set init-lambda-state."""
        target = dest_dir / f"lambda_{lam_state}.mdp"
        lines = self.mdp_template.read_text().splitlines()
        out = []
        for L in lines:
            if L.strip().startswith("init-lambda-state"):
                out.append(f"init-lambda-state        = {lam_state}")
            else:
                out.append(L)
        target.write_text("\n".join(out))

    def _write_run_script(self, lam_state: int, run_index: int, dest_dir: Path):
        """
        Copy a template run script into dest_dir/run_{i}.sh, replacing
        placeholders like {LAM} and {RUN} if present.
        """
        tpl = self.run_script_template.read_text()
        content = tpl.replace("{LAM}", str(lam_state)).replace("{RUN}", str(run_index))
        target = dest_dir / "run_gmx.sh"
        target.write_text(content)
        # make executable?
        # target.chmod(0o755)

    def run(self, *args, **kwargs):
        for w in self._sub_sim_runners:
            w.run(*args, **kwargs)
        self._finished = all(w._finished for w in self._sub_sim_runners)
        self._failed = any(w._failed for w in self._sub_sim_runners)


class Leg(SimulationRunner):

    runtime_attributes = {"_finished": False, "_failed": False, "_dg": None}

    # Required input files for each leg type and preparation stage.
    required_input_files: dict[LegType, dict[PreparationStage, list[str]]] = {}
    for leg_type in LegType:
        required_input_files[leg_type] = {}
        for prep_stage in PreparationStage:
            required_input_files[leg_type][prep_stage] = [
                "run_gmx.sh",
                *prep_stage.get_simulation_input_files(leg_type),
            ]

    required_stages: dict[LegType, list[StageType]] = {
        LegType.BOUND: [StageType.RESTRAIN, StageType.DISCHARGE, StageType.VANISH],
        LegType.FREE: [StageType.DISCHARGE, StageType.VANISH],
    }

    def __init__(
        self,
        leg_type: LegType,
        stages: list[Stage],
        input_dir: str,
        output_dir: str,
        ensemble_size: int,
    ) -> None:
        super().__init__(input_dir, output_dir, ensemble_size=5)
        self.leg_type = leg_type
        self.stages = stages
        self.ensemble_size = ensemble_size

    def setup(self):
        src_root = Path(self.input_dir)
        equiv_dir = src_root / "equilibration"
        dest_root = Path(self.output_dir)
        dest_root.mkdir(parents=True, exist_ok=True)

        for stage in self.stages:
            # 1) make the stage directories
            stage_dir = dest_root / stage.stage_type.name.lower()
            stage_input = stage_dir / "input"
            stage_output = stage_dir / "output"
            stage_input.mkdir(parents=True, exist_ok=True)
            stage_output.mkdir(parents=True, exist_ok=True)

            # 2) copy all required static SOMD inputs
            for fn in self.required_input_files[self.leg_type][stage.stage_type]:
                src = src_root / fn
                if not src.exists():
                    # fallback into the equilibration subfolder
                    src = equiv_dir / fn
                if not src.exists():
                    raise FileNotFoundError(f"Cannot find required input file {fn}")
                shutil.copy(src, stage_input)

            # 3) copy in the N ensemble-equilibrated gro/top, plus restraints if bound
            suffix = PreparationStage.EQUILIBRATED.file_suffix
            for i in range(1, self.ensemble_size + 1):
                for ext in (".gro", ".top"):
                    fn = f"{self.leg_type.name.lower()}{suffix}_{i}_final{ext}"
                    shutil.copy(equiv_dir / fn, stage_input)
                if self.leg_type is LegType.BOUND:
                    shutil.copy(equiv_dir / f"restraint_{i}.itp", stage_input)

            # 4) copy run script & mdp
            shutil.copy(src_root / "lambda.gmx.mdp", stage_input)
            shutil.copy(src_root / "run_gmx.sh", stage_input)

            # 5) now delegate into the Stage.setup() to build per-lambda/run dirs
            stage.input_dir = str(stage_input)
            stage.output_dir = str(stage_output)
            stage.leg_type = self.leg_type
            stage.setup()

    def run(self, run_nos=None, *args, **kwargs):
        for stage in self._sub_sim_runners:
            stage.run(run_nos=run_nos, *args, **kwargs)
        self._finished = all(s._finished for s in self._sub_sim_runners)
        self._failed = any(s._failed for s in self._sub_sim_runners)

    @property
    def failed_simulations(self):
        return [fail for s in self._sub_sim_runners for fail in s.failed_simulations]


class LambdaWindow(SimulationRunner):
    runtime_attributes = {"_finished": False, "_failed": False, "_dg": None}

    def __init__(
        self,
        lam_state: int,
        input_dir: str,
        output_dir: str,
        sim_params: dict,
        run_index=1,
    ) -> None:
        super().__init__(input_dir, output_dir, ensemble_size=1)
        self.lam_state = lam_state
        self.run_index = run_index
        self.simulation = Simulation(
            lam_state=lam_state,
            work_dir=Path(output_dir),
            run_index=run_index,
            **sim_params,
        )
        self._sub_sim_runners = []

    def setup(self):
        self.simulation.setup()

    def run(self, run_nos=None, *args, **kwargs):
        self.simulation.run()
        self._finished = self.simulation.finished
        self._failed = self.simulation.failed

    @property
    def failed_simulations(self):
        return self.simulation.failed_simulations


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
        sim_config: SomdFepSimulationConfig,
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
    def legs(self) -> Sequence[SimulationRunner]:
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
    def is_complete(self) -> bool:
        f"""Whether the {self.__class__.__name__} has completed."""
        # Check if the overall_stats.dat file exists
        if Path(f"{self.output_dir}/overall_stats.dat").is_file():
            return True

        return False

    def setup(
        self,
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
                input_dir=str(self.input_dir),
                output_dir=str(leg_base),
                ensemble_size=self.ensemble_size,
            )
            leg.setup()
            self.legs.append(leg)

        self.setup_complete = True
        self._dump()

    def run(self, *args, run_nos: Optional[list[int]] = None, **kwargs) -> None:
        raise NotImplementedError(
            "The run method is not implemented yet. Please implement it according to your needs."
        )
