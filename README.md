# binding_affinity_predicting


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Workflows to automate protein–ligand docking and free‐energy calculations

---

## 🚀 Features

- **Parameterisation** of proteins, ligands, and crystal waters  
- **Solvation** in boxes with configurable salt concentration  
- **Energy minimisation** and **pre-equilibration** (NVT/NPT) protocols  
- **Ensemble equilibration** for bound-leg restraint selection  
- Fully **configurable** via a Pydantic `WorkflowConfig` model  
- Optional **SLURM** integration for HPC clusters  

---

## 📦 Installation

Requires **Python ≥ 3.9, <3.12**.

```bash
# Clone repo
git clone https://github.com/YourUsername/binding_affinity_predicting.git
cd binding_affinity_predicting

# Install the package via Conda
conda env create -f environment.yaml
conda activate binding_env
pip install --no-deps . 


```
## 🛠️ Quickstart
```Python

from binding_affinity_predicting.workflows.free_energy_calc.system_prep_workflow import run_complete_system_setup_bound_and_free


cfg = BaseWorkflowConfig(
    slurm=False,  # run everything locally
    param_preequilibration=custom_preequil,
    param_energy_minimisation=custom_min,
    param_ensemble_equilibration=custom_ensemble_equil,
    # to use "-ntmpi 1 -ntomp 8" we need to compile GROMACS with OpenMP support
    mdrun_options="-ntmpi 1 -ntomp 1",
)

system_list = run_complete_system_setup_bound_and_free(
    config=cfg,
    protein_path="input/protein.pdb",
    ligand_path="input/ligand.sdf",
    filename_stem="bound",
    output_dir="output/prep",
    use_slurm=False,
)
```