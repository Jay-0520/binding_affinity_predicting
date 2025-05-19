from pathlib import Path

from binding_affinity_predicting.components.lambda_window import LambdaWindow
from binding_affinity_predicting.components.simulation_base import SimulationRunner


class Stage(SimulationRunner):
    runtime_attributes = {"_finished": False, "_failed": False, "_dg": None}

    def __init__(
        self,
        stage_type: str,
        lam_list: list[float],
        input_dir: str,
        output_dir: str,
        sim_params: dict,
        run_index=1,
    ):
        super().__init__(input_dir, output_dir, ensemble_size=1)
        self.stage_type = stage_type
        self.lam_list = lam_list
        self.run_index = run_index
        self._sub_sim_runners = [
            LambdaWindow(
                lam=lam,
                input_dir=input_dir,
                output_dir=output_dir,
                sim_params=sim_params,
                run_index=run_index,
            )
            for lam in lam_list
        ]

    def setup(self):
        for sim_runner in self._sub_sim_runners:
            sim_runner.setup()

    def run(self, run_nos=None, *args, **kwargs):
        for sim_runner in self._sub_sim_runners:
            sim_runner.run(run_nos=run_nos, *args, **kwargs)
        self._finished = all(sr._finished for sr in self._sub_sim_runners)
        self._failed = any(sr._failed for sr in self._sub_sim_runners)

    @property
    def failed_simulations(self):
        return [fail for w in self._sub_sim_runners for fail in w.failed_simulations]
