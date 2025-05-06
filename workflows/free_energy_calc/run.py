import a3fe as a3
from a3fe.run.system_prep import SystemPreparationConfig

# turn off slurm everywhere
# NOTE: we might not need mdrun_options in prod
sysprep_cfg = SystemPreparationConfig(slurm=False,
                                      mdrun_options="-ntmpi 1 -ntomp 1")  # added for local run on mac

print('step-1...')
calc = a3.Calculation(ensemble_size=1, 
                      base_dir="/Users/jingjinghuang/Documents/fep_workflow/test_run",
                      # need to have protein.pdb, ligand.sdf, run_somd.sh and template_config.cfg
                      input_dir="/Users/jingjinghuang/Documents/fep_workflow/test_run/input")
print("step-2: setup gromacs local run")
calc.setup(
    bound_leg_sysprep_config=sysprep_cfg,
    free_leg_sysprep_config=sysprep_cfg,
)
print('step-3...')
calc.get_optimal_lam_vals()
print('step-4...')
calc.run(adaptive=False, runtime = 5) # run non-adaptively for 5 ns per replicate
print('step-5...')
calc.wait()
print('step-6...')
calc.set_equilibration_time(1) # discard the first ns of simulation time
print('step-7...')
calc.analyse()
calc.save()
