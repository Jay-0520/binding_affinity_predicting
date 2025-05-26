from pathlib import Path

from binding_affinity_predicting.components.lambda_window import LambdaWindow
from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.data.enums import StageType


class Stage(SimulationRunner):
    def __init__(
        self,
        stage_type: StageType,
        input_dir: str,
        output_dir: str,
    ):
        super().__init__(input_dir, output_dir, ensemble_size=1)
        self.stage_type = stage_type
        # create lam windows
        # e.g., lam_list from config, here stub with [0.0,1.0]
        lam_list = [0.0, 1.0]
        self._sub_sim_runners = [self._make_window(lam) for lam in lam_list]

    def _make_window(self, lam: float):
        from binding_affinity_predicting.components.lambda_window import LambdaWindow

        return LambdaWindow(
            lam_state=lam,
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            sim_params={},
            run_index=1,
        )

    def setup(self):
        for w in self._sub_sim_runners:
            w.setup()

    def run(self, *args, **kwargs):
        for w in self._sub_sim_runners:
            w.run(*args, **kwargs)
        self._finished = all(w._finished for w in self._sub_sim_runners)
        self._failed = any(w._failed for w in self._sub_sim_runners)
