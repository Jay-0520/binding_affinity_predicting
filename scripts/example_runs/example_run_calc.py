from binding_affinity_predicting.components.gromacs_orchestration import Calculation
from binding_affinity_predicting.data.schemas import GromacsFepSimulationConfig

calc = Calculation(
    input_dir="/Users/jingjinghuang/Documents/fep_workflow/test_classes/input",
    output_dir="/Users/jingjinghuang/Documents/fep_workflow/test_classes/output",
    ensemble_size=2,
    sim_config=GromacsFepSimulationConfig(),
)


calc.setup()
calc.run()
