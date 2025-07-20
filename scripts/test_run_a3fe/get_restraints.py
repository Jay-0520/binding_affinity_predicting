import logging

import BioSimSpace.Sandpit.Exscientia as _BSS
from a3fe.run.enums import LegType as LegType
from a3fe.run.enums import PreparationStage as _PreparationStage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


leg_type = LegType.BOUND  # or LegType.FREE, depending on the leg you want to run

input_dir = "/Users/jingjinghuang/Documents/fep_workflow/dataset/input"
pre_equilibrated_system = _BSS.IO.readMolecules(
    [
        f"{input_dir}/{file}"
        for file in _PreparationStage.PREEQUILIBRATED.get_simulation_input_files(
            leg_type
        )
    ]
)

top_file = f"{input_dir}/{_PreparationStage.PREEQUILIBRATED.get_simulation_input_files(leg_type)[0]}"  # noqa: E501

lig = _BSS.Align.decouple(pre_equilibrated_system[0], intramol=True)
pre_equilibrated_system.updateMolecule(0, lig)


i = 3
# for i in range(1, 4):  # Change the range to the number of runs you want
outdir = f"/Users/jingjinghuang/Documents/fep_workflow/dataset/output/ensemble_equilibration_{i}"
traj = _BSS.Trajectory.Trajectory(
    topology=top_file,
    trajectory=f"{outdir}/gromacs.xtc",
    system=pre_equilibrated_system,
)
logger.info(f"Selecting restraints for run {i}...")
restraint = _BSS.FreeEnergy.RestraintSearch.analyse(
    method="BSS",
    system=pre_equilibrated_system,
    traj=traj,
    work_dir=outdir,
    temperature=298.15 * _BSS.Units.Temperature.kelvin,
    append_to_ligand_selection="",
)

# Check that we actually generated a restraint
if restraint is None:
    raise ValueError(f"No restraints found for run {i}.")

# Save the restraints to a text file and store within the Leg object
with open(f"{outdir}/restraint_{i}.txt", "w") as f:
    # NOTE we can use "gromacs" as the engine here by JJH-2025-05-17
    f.write(restraint.toString(engine="SOMD"))  # type: ignore
