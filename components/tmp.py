class SimulationRunner(ABC):
    ...
    self._sub_sim_runners: list[SimulationRunner] = []
    ...

    def run(self, run_nos=None):
        for sub in self._sub_sim_runners:
            sub.run(run_nos)

    @property
    def running(self) -> bool:
        return any(sub.running for sub in self._sub_sim_runners)

    def get_tot_simtime(self, run_nos=None):
        return sum(sub.get_tot_simtime(run_nos) for sub in self._sub_sim_runners)
    



class DummyLeafRunner(SimulationRunner):
    """
    A leaf SimulationRunner that simulates “work” by sleeping,
    then returns dummy free‐energy data.
    """

    def __init__(self, input_dir, output_dir, *, sleep_time=1.0, **kwargs):
        super().__init__(input_dir=input_dir,
                         output_dir=output_dir,
                         **kwargs)
        self.sleep_time = sleep_time

    def _run_once(self, run_no: int, **kwargs) -> None:
        # e.g. create a dummy output file for each run
        out = Path(self.output_dir) / f"run_{run_no}.txt"
        time.sleep(self.sleep_time)           # pretend to do work
        out.write_text(f"Completed run {run_no}\n")

    def _analyse_once(self, run_no: int, *, fraction: float) -> Tuple[float, float]:
        # read whatever your run produced (here: ignore it)
        # compute dummy ΔG and error
        dg = np.random.normal(loc=0.0, scale=1.0) * fraction
        er = np.abs(np.random.normal(scale=0.1))
        return dg, er
    


leaf = DummyLeafRunner("in/leaf", "out/leaf", ensemble_size=3)
leaf.run()                    # runs 3 sleeps
dgs, ers = leaf.analyse()     # returns 3 random ΔG’s + errors



ensemble = SimulationRunner(...)  # your composite class
ensemble.add_subrunner(leaf)
ensemble.add_subrunner(AnotherLeafRunner(...))
ensemble.run()
ensemble.analyse()