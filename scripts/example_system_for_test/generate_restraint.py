from binding_affinity_predicting.components.simulation_base.system_preparation import (
    extract_restraint_from_traj,
)
from binding_affinity_predicting.components.simulation_base.utils import (
    load_system_from_source,
)

sys_load = load_system_from_source(  # noqa: E501
    source='/Users/jingjinghuang/Documents/fep_workflow/a3fe/a3fe/data/example_run_dir/bound/ensemble_equilibration_1/gromacs.gro'  # noqa: E501
)

work_dir = "/Users/jingjinghuang/Documents/fep_workflow/a3fe/a3fe/data/example_run_dir/bound/ensemble_equilibration_1"  # noqa: E501

restraint = extract_restraint_from_traj(
    work_dir=work_dir,
    trajectory_file=f"{work_dir}/gromacs.xtc",
    topology_file=f"{work_dir}/gromacs.tpr",
    system=sys_load,
    output_filename="restraint.itp",
)
