"""
gromacs_orchestration.py

Functionality for setting up and running an entire ABFE calculation using GROMACS,
consisting of two legs (bound and unbound) and multiple lambda‐windows per leg.
In GROMACS, all three alchemical stages (RESTRAIN → DISCHARGE → VANISH) are encoded
in a single MDP file via bonded‐, coulomb‐, and vdw‐lambda vectors, so we do not
instantly split into separate Stage objects. Instead, each lambda window is its own
sub‐runner and drives GROMACS with a different `init‐lambda‐state`.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Sequence

from binding_affinity_predicting.components.simulation import Simulation
from binding_affinity_predicting.components.simulation_base import SimulationRunner
from binding_affinity_predicting.data.enums import LegType, PreparationStage
from binding_affinity_predicting.data.schemas import (
    GromacsFepSimulationConfig,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LambdaWindow(SimulationRunner):
    runtime_attributes = {"_finished": False, "_failed": False, "_dg": None}

    def __init__(
        self,
        lam_state: int,
        input_dir: str,
        work_dir: str,
        sim_params: dict,
        run_index=1,
    ):
        """ """
        super().__init__(input_dir, work_dir, ensemble_size=1)
        self.lam_state = lam_state
        self.run_index = run_index
        self.simulation = Simulation(
            lam_state=lam_state,
            work_dir=Path(work_dir),
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


class Leg(SimulationRunner):
    """
    A Leg (BOUND or FREE) that groups multiple lambda-window replicas.
    Builds the requested directory tree and symlinks inputs.
    """

    runtime_attributes = {"_finished": False, "_failed": False}

    def __init__(
        self,
        leg_type: LegType,
        lam_indices: list[int],
        ensemble_size: int,
        input_dir: str,
        output_dir: str,
        sim_config: GromacsFepSimulationConfig,
    ):
        super().__init__(
            input_dir=input_dir, output_dir=output_dir, ensemble_size=ensemble_size
        )
        self.leg_type = leg_type
        self.lam_indices = lam_indices
        self.ensemble_size = ensemble_size
        self.sim_config = sim_config
        self._sub_sim_runners: Sequence[LambdaWindow] = []

    @property
    def failed_simulations(self):
        return [win for win in self._sub_sim_runners if win._failed]

    def setup(self):
        """
        1) Create leg_base/
        2) Under leg_base, create leg_base/input/ and copy all .gro/.top/restraints + templates
        3) Under leg_base, create one folder per λ index (lambda_{lam_idx}/)
           and inside each, one folder per replica (run_{run_idx}/)
        4) In each run_{run_idx}/, create symlinks back to leg_base/input/
        5) Instantiate LambdaWindow for each (lam_idx, run_idx)
        """
        src_equil = Path(self.input_dir) / "equilibration"
        leg_base = Path(self.output_dir)
        leg_base.mkdir(parents=True, exist_ok=True)

        leg_input = leg_base / "input"
        leg_input.mkdir(exist_ok=True)

        # Copy all pre‐equilibrated .gro/.top and restraint files into leg_input
        # Example filenames: bound_equilibrated_1_final.gro, bound_equilibrated_1_final.top,
        # restraint_1.itp
        suffix = PreparationStage.EQUILIBRATED.file_suffix
        for run_idx in range(1, self.ensemble_size + 1):
            for ext in (".gro", ".top"):
                fname = f"{self.leg_type.name.lower()}{suffix}_{run_idx}_final{ext}"
                src_file = src_equil / fname
                if not src_file.exists():
                    raise FileNotFoundError(f"Cannot find {src_file}")
                shutil.copy(src_file, leg_input / fname)

            if self.leg_type is LegType.BOUND:
                restr_name = f"restraint_{run_idx}.itp"
                src_rest = src_equil / restr_name
                if not src_rest.exists():
                    raise FileNotFoundError(f"Cannot find {src_rest}")
                shutil.copy(src_rest, leg_input / restr_name)

        # Copy MDP template and run script into leg_input
        tmpl_mdp = Path(self.input_dir) / "lambda.template.mdp"
        if not tmpl_mdp.exists():
            raise FileNotFoundError(f"MDP template not found at {tmpl_mdp}")
        shutil.copy(tmpl_mdp, leg_input / "lambda.template.mdp")

        run_scr = Path(self.input_dir) / "submit_gmx.sh"
        if not run_scr.exists():
            raise FileNotFoundError(f"Run script not found at {run_scr}")
        shutil.copy(run_scr, leg_input / "submit_gmx.sh")

        # For each lambda index, create lambda_{lam_idx}/ and inside each, run_{run_idx}/
        for lam_idx in self.lam_indices:
            lam_dir = leg_base / f"lambda_{lam_idx}"
            lam_dir.mkdir(exist_ok=True)

            for run_idx in range(1, self.ensemble_size + 1):
                run_dir = lam_dir / f"run_{run_idx}"
                run_dir.mkdir(exist_ok=True)

                # Symlink equilibrated GRO/TOP:
                to_link = []
                for ext in (".gro", ".top"):
                    fname = f"{self.leg_type.name.lower()}{suffix}_{run_idx}_final{ext}"
                    to_link.append(fname)

                # If bound, symlink restraint:
                if self.leg_type is LegType.BOUND:
                    to_link.append(f"restraint_{run_idx}.itp")

                # now create (or overwrite) every symlink in one loop
                for filename in to_link:
                    srcfile = (
                        leg_input / filename
                    )  # e.g. “…/bound/input/bound_equilibrated_1_final.gro”
                    destfile = (
                        run_dir / filename
                    )  # e.g. “…/bound/lambda_0/run_1/bound_equilibrated_1_final.gro”

                    if destfile.exists() or destfile.is_symlink():
                        logger.info(f"Removing existing file/symlink: {destfile}")
                        destfile.unlink()

                    os.symlink(os.path.relpath(srcfile, start=run_dir), destfile)

                for filename in ("lambda.template.mdp", "submit_gmx.sh"):
                    srcfile = leg_input / filename
                    destfile = run_dir / filename
                    if not srcfile.exists():
                        raise FileNotFoundError(f"Run script not found at {srcfile}")
                    shutil.copy(srcfile, destfile)

                # 5) Build sim_params for Simulation inside LambdaWindow.
                #    We need to tell `Simulation` how to find the MDP, GRO, and TOP inside run_dir.
                λ_float = self.sim_config.lambda_values[self.leg_type][lam_idx]
                sim_params = {
                    "gmx_exe": "gmx",
                    # MDP template path (inside this run_dir, linked as "lambda.template.mdp"):
                    "mdp_template": str(run_dir / "lambda.template.mdp"),
                    # GRO & TOP filenames (they are symlinks in run_dir):
                    "gro_file": str(
                        run_dir
                        / f"{self.leg_type.name.lower()}{suffix}_{run_idx}_final.gro"
                    ),
                    "top_file": str(
                        run_dir
                        / f"{self.leg_type.name.lower()}{suffix}_{run_idx}_final.top"
                    ),
                    # If Simulation ever needs the float λ-value, pass it here:
                    "extra_params": {"lambda_float": λ_float},
                }

                # 5) Instantiate LambdaWindow
                window = LambdaWindow(
                    lam_state=lam_idx,
                    run_index=run_idx,
                    input_dir=str(run_dir),
                    work_dir=str(run_dir),
                    sim_params=sim_params,
                )
                # Set as sub‐runner, then call its .setup()
                self._sub_sim_runners.append(window)
                window.setup()

    def run(self, run_nos=None, *args, **kwargs):
        """
        Run all lambda windows in this leg.
        """
        for win in self._sub_sim_runners:
            win.run()
        self._finished = all(win._finished for win in self._sub_sim_runners)
        self._failed = any(win._failed for win in self._sub_sim_runners)


class Calculation(SimulationRunner):
    """
    Class to set up and run an entire ABFE calculation in GROMACS,
    consisting of two legs (BOUND and FREE), each with a list of lambda states
    and an ensemble of replicas per lambda.
    """

    # Only these two legs for GROMACS
    required_legs = [LegType.BOUND, LegType.FREE]
    _sub_sim_runners: Sequence["Leg"]

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        sim_config: GromacsFepSimulationConfig,
        equil_detection: str = "multiwindow",
        runtime_constant: Optional[float] = 0.001,
        ensemble_size: int = 5,
    ) -> None:
        super().__init__(
            input_dir=input_dir, output_dir=output_dir, ensemble_size=ensemble_size
        )
        self.sim_config = sim_config

    @property
    def legs(self) -> Sequence["Leg"]:
        return self._sub_sim_runners

    @legs.setter
    def legs(self, value: Sequence["Leg"]) -> None:
        logger.info("Creating/modifying Leg objects")
        self._sub_sim_runners = value

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
            float_list = self.sim_config.lambda_values[leg_type]
            lam_indices = list(range(len(float_list)))
            leg = Leg(
                leg_type=leg_type,
                lam_indices=lam_indices,
                input_dir=self.input_dir,
                output_dir=str(leg_base),
                ensemble_size=self.ensemble_size,
                sim_config=self.sim_config,
            )
            leg.setup()
            self.legs.append(leg)

        self.setup_complete = True
        self._dump()

    def run(self, *args, run_nos: Optional[list[int]] = None, **kwargs) -> None:
        raise NotImplementedError()
