import a3fe as a3
from a3fe.run.system_prep import SystemPreparationConfig

# 1) Turn off Slurm everywhere
sysprep_cfg = SystemPreparationConfig(slurm=False,
                                      mdrun_options="-ntmpi 1 -ntomp 1")  # added for local run on mac

print('step-1...')
calc = a3.Calculation(ensemble_size=1, 
                      base_dir="/Users/jingjinghuang/Documents/fep_workflow/test_run",
                      input_dir="/Users/jingjinghuang/Documents/fep_workflow/test_run/input")
print("step-2: setup (GROMACS prep will run locally)")
calc.setup(
    bound_leg_sysprep_config=sysprep_cfg,
    free_leg_sysprep_config=sysprep_cfg,
)
print('step-3...')
calc.get_optimal_lam_vals()
print('step-4...')
calc.run(adaptive=False, runtime = 5) # Run non-adaptively for 5 ns per replicate
print('step-5...')
calc.wait()
print('step-6...')
calc.set_equilibration_time(1) # Discard the first ns of simulation time
print('step-7...')
calc.analyse()
calc.save()
