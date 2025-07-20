import a3fe as a3

# sysprep_cfg = SystemPreparationConfig(
#     mdrun_options="-ntmpi 1 -ntomp 1",
#     runtime_short_nvt=5,
#     runtime_nvt=10,
#     runtime_npt=10,  # added for local test run on mac; unit - ps
#     runtime_npt_unrestrained=10,  # added for local test run on mac; unit - ps
#     ensemble_equilibration_time=10,
# )  # added for local test run on mac; unit - ps

print('step-1...')
calc = a3.Calculation(
    ensemble_size=1,
    base_dir="/Users/jingjinghuang/Documents/fep_workflow/test_run",
    input_dir="/Users/jingjinghuang/Documents/fep_workflow/test_run/input",
)
print("step-2: setup (GROMACS prep will run locally)")
calc.setup()
print('step-3...')
calc.get_optimal_lam_vals()
print('step-4...')
calc.run(
    adaptive=False,
    runtime=5,  # run non-adaptively for 5 ns per replicate
    parallel=False,
)  # run things sequentially
print('step-5...')
calc.wait()
print('step-6...')
calc.set_equilibration_time(1)  # Discard the first ns of simulation time
print('step-7...')
calc.analyse()
calc.save()
