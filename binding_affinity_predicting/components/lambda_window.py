from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.components.simulation import Simulation
from pathlib import Path


class LambdaWindow(SimulationRunner):
    runtime_attributes = {"_finished": False, "_failed": False, "_dg": None}

    def __init__(
        self,
        lam_state: int,
        input_dir: str,
        output_dir: str,
        sim_params: dict,
        run_index=1,
    ):
        super().__init__(input_dir, output_dir, ensemble_size=1)
        self.lam_state = lam_state
        self.run_index = run_index
        self.simulation = Simulation(
            lam_state=lam_state, work_dir=output_dir, run_index=run_index, **sim_params
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
