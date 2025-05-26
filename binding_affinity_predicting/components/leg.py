import logging
import shutil
from pathlib import Path
from typing import Optional

from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.components.stage import Stage
from binding_affinity_predicting.data.enums import LegType, PreparationStage, StageType
from binding_affinity_predicting.data.schemas import (
    SystemPreparationConfig,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Leg(SimulationRunner):

    runtime_attributes = {"_finished": False, "_failed": False, "_dg": None}

    # Required input files for each leg type and preparation stage.
    required_input_files = {}
    for leg_type in LegType:
        required_input_files[leg_type] = {}
        for prep_stage in PreparationStage:
            required_input_files[leg_type][prep_stage] = [
                "run_gmx.sh",
            ] + prep_stage.get_simulation_input_files(leg_type)

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
        ensemble_size: int,
    ):
        super().__init__(input_dir, output_dir, ensemble_size=5)
        self.leg_type = leg_type
        self._sub_sim_runners = stages
        self.ensemble_size = ensemble_size

    def setup(self, sysprep_config: Optional[SystemPreparationConfig] = None):
        src_root = Path(self.input_dir)
        equiv_dir = src_root / "equilibration"
        dest_root = Path(self.output_dir)
        dest_root.mkdir(parents=True, exist_ok=True)

        for stage in self._sub_sim_runners:
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

            # 3) copy in the N ensemble‐equilibrated gro/top, plus restraints if bound
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

            # 5) now delegate into the Stage.setup() to build per‐lambda/run dirs
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
