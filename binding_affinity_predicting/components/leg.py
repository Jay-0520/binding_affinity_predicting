from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.components.stage import Stage
from pathlib import Path


class Leg(SimulationRunner):
    runtime_attributes = {"_finished": False, "_failed": False, "_dg": None}

    def __init__(self, leg_type, stages: list, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir, ensemble_size=1)
        self.leg_type = leg_type
        self._sub_sim_runners = stages

    def setup(self):
        for stage in self._sub_sim_runners:
            stage.setup()

    def run(self, run_nos=None, *args, **kwargs):
        for stage in self._sub_sim_runners:
            stage.run(run_nos=run_nos, *args, **kwargs)
        self._finished = all(s._finished for s in self._sub_sim_runners)
        self._failed = any(s._failed for s in self._sub_sim_runners)

    @property
    def failed_simulations(self):
        return [fail for s in self._sub_sim_runners for fail in s.failed_simulations]
