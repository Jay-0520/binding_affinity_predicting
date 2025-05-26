from pathlib import Path

from binding_affinity_predicting.components.lambda_window import LambdaWindow
from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.data.enums import LegType, PreparationStage, StageType


class Stage(SimulationRunner):
    def __init__(
        self,
        stage_type: StageType,
        num_lambdas: int,
        ensemble_size: int,
        leg_type: LegType,
        mdp_template: str,
        run_script_template: str,
    ):

        # we’ll override dirs later in Leg.setup()
        super().__init__(input_dir="", output_dir="", ensemble_size=ensemble_size)
        self.stage_type = stage_type
        # e.g. lam_list = [0.0, 0.1, ...]  – here just indices
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

                # symlink exactly the per‐run equilibration & restraint files
                # assume self.input_dir still points at the stage‐level "input/" folder
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

                    # If there's already something here, remove it so we can re‐link:
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
