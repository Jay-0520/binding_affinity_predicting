import logging
from pathlib import Path
from typing import Optional

from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.components.stage import Stage
from binding_affinity_predicting.components.utils import move_link_or_copy_files
from binding_affinity_predicting.data.enums import LegType, PreparationStage, StageType
from binding_affinity_predicting.data.schemas import SystemPreparationConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Leg(SimulationRunner):

    runtime_attributes = {"_finished": False, "_failed": False, "_dg": None}

    required_input_files = {}

    required_stages = {
        LegType.BOUND: [StageType.RESTRAIN, StageType.DISCHARGE, StageType.VANISH],
        LegType.FREE: [StageType.DISCHARGE, StageType.VANISH],
    }

    def __init__(
        self,
        leg_type: LegType,
        stages: list[Stage],
        input_dir: str,
        output_dir: str,
    ):
        super().__init__(input_dir, output_dir, ensemble_size=5)
        self.leg_type = leg_type
        self._sub_sim_runners = stages

    def setup(self, sysprep_config: Optional[SystemPreparationConfig] = None):
        """
        Prepare each stage's input for execution (e.g., generate inputs, write scripts).
        """
        src_root = Path(self.input_dir)
        dest_root = Path(self.output_dir) / self.leg_type.name.lower()
        dest_root.mkdir(parents=True, exist_ok=True)

        # ensure all required static inputs are in src
        # copy final structures and restraints into src before run
        for stage in self._sub_sim_runners:
            stage_name = stage.stage_type.name.lower()
            stage_dir = dest_root / stage_name
            stage_input_dir = stage_dir / "input"
            stage_output_dir = stage_dir / "output"

            stage_input_dir.mkdir(parents=True, exist_ok=True)
            stage_output_dir.mkdir(parents=True, exist_ok=True)

            # copy the required static files into input/
            req_files = self.required_input_files[self.leg_type][stage.stage_type]
            move_link_or_copy_files(
                src_dir=str(src_root),
                filenames=req_files,
                dest_dir=str(stage_input_dir),
            )

            # final gro/top from ensemble equilibration
            suffix = PreparationStage.EQUILIBRATED.file_suffix
            for i in range(1, self.ensemble_size + 1):
                for ext in ('.gro', '.top'):
                    fname = f"{self.leg_type.name.lower()}{suffix}_{i}_final{ext}"
                    move_link_or_copy_files(
                        str(src_root), [fname], str(stage_input_dir)
                    )

            # for bound legs, also bring in the restraint itp
            if self.leg_type is LegType.BOUND:
                # NOTE: we use the same restraints for all repeats
                restraint_files = [
                    f"restraint_{i+1}.itp" for i in range(self.ensemble_size)
                ]
                move_link_or_copy_files(
                    src_dir=str(src_root),
                    filenames=restraint_files,
                    dest_dir=str(stage_input_dir),
                )

            # copy general run scripts and mdp files
            move_link_or_copy_files(
                src_dir=src_root,
                filenames=["lambda.gmx.mdp", "run_gmx.sh"],
                dest_dir=stage_input_dir,
            )

            # assign paths back to stage
            stage.input_dir = str(stage_input_dir)
            stage.output_dir = str(stage_output_dir)

            logger.info(f"Leg {self.leg_type.name} setup complete.")

    def run(self, run_nos=None, *args, **kwargs):
        for stage in self._sub_sim_runners:
            stage.run(run_nos=run_nos, *args, **kwargs)
        self._finished = all(s._finished for s in self._sub_sim_runners)
        self._failed = any(s._failed for s in self._sub_sim_runners)

    @property
    def failed_simulations(self):
        return [fail for s in self._sub_sim_runners for fail in s.failed_simulations]
