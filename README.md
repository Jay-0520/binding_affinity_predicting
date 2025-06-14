# binding_affinity_predicting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Automated workflows for protein–ligand binding affinity prediction using GROMACS free energy perturbation (FEP) calculations**

---

## 🚀 Features

### System Preparation
- **Automated parameterisation** of proteins, ligands, and crystal waters  
- **Solvation** in periodic boxes with configurable salt concentration  
- **Energy minimisation** and **pre-equilibration** (NVT/NPT) protocols  
- **Ensemble equilibration** for optimal bound-leg restraint selection  

### FEP Calculations
- **Multi-stage lambda optimization** for GROMACS FEP simulations
  - Restrained stage (bonded-lambdas: 0.0 → 1.0)
  - Discharging stage (coul-lambdas: 0.0 → 1.0) 
  - Vanishing stage (vdw-lambdas: 0.0 → 1.0)
- **Intelligent lambda spacing** based on gradient analysis and statistical inefficiency
- **Automated error estimation** with autocorrelation correction

### Infrastructure
- Fully **configurable** via Pydantic `WorkflowConfig` models  
- **SLURM integration** for HPC clusters with virtual queue management
- **Comprehensive logging** and error handling

---

## 📦 Installation

**Requirements:** Python ≥ 3.9, <3.12

```bash
# Clone the repository
git clone https://github.com/YourUsername/binding_affinity_predicting.git
cd binding_affinity_predicting

# Create conda environment and install dependencies
conda env create -f environment.yaml
conda activate binding_env

# Install the package
pip install --no-deps .
```

### Additional Requirements
- **GROMACS** (≥2021) compiled with OpenMP support for optimal performance
- **SLURM** (optional, for HPC cluster execution)

---

## 🛠️ Quick Start

### System Preparation

Prepare protein-ligand complex systems for FEP calculations:

```python
from binding_affinity_predicting.data.schemas import BaseWorkflowConfig
from binding_affinity_predicting.workflows.free_energy_calc.system_prep_workflow import (
    run_complete_system_setup_bound_and_free
)

# Configure system preparation workflow
config = BaseWorkflowConfig(
    slurm=False,  # Run locally (set True for SLURM clusters)
    mdrun_options="-ntmpi 1 -ntomp 8",  # Adjust based on your hardware
)

# Prepare bound and free leg systems
system_list = run_complete_system_setup_bound_and_free(
    config=config,
    protein_path="input/protein.pdb",
    ligand_path="input/ligand.sdf", 
    filename_stem="system1",
    output_dir="output/preparation",
    use_slurm=False,
)
```

### FEP Calculation with Lambda Optimization

Run optimized free energy perturbation calculations:

```python
from binding_affinity_predicting.components.lambda_optimizer import (
    OptimizationConfig,
    GromacsFepSimulationConfig,
)
from binding_affinity_predicting.components.gromacs_orchestration import Calculation


# Initiate the calculation
calc = Calculation(input_dir="~/input",
                    output_dir="~/output",
                    ensemble_size = 2,
                    sim_config=GromacsFepSimulationConfig())
                
calc.setup()
# Run calculation with short runtime to generate data for optimizing
calc.run(runtime=short_runtime, use_hpc=False, run_sync=True)

# Initialize optimizer and run calculation
manager = LambdaOptimizationManager(config=OptimizationConfig())

# Optimize lambda spacing based on initial gradients
results = optimizer.optimize_calculation(
    calculation=calc,
    equilibrated=True,  # Only use equilibrated data
    apply_results=True  # Apply optimized spacing automatically
)

# Run the optimized FEP calculation
calc.run(runtime=long_runtime, use_hpc=False, run_sync=True) 
```

### Results Analysis

TBD

---

## 📁 Project Structure

```
binding_affinity_predicting/
├── components/           # Core calculation components
│   ├── gromacs_orchestration.py  # GROMACS workflow orchestration
│   └── lambda_optimizer.py       # Multi-stage lambda optimization
├── data/                # Data models and schemas
│   ├── schemas.py       # Pydantic configuration models
│   └── enums.py         # Status and type enumerations
├── hpc_cluster/         # HPC integration utilities
│   └── virtual_queue.py # SLURM virtual queue management
├── simulation/          # Simulation analysis tools
│   └── autocorrelation.py  # Statistical inefficiency analysis
└── workflows/           # High-level workflow orchestration
    └── free_energy_calc/   # FEP-specific workflows
```

---

## 🔬 Algorithm Details

### Lambda Optimization Strategy

The optimizer recognizes that GROMACS FEP calculations consist of three distinct thermodynamic stages:

1. **Restrained Stage**: Bonded interactions are turned on (bonded λ: 0→1, coul=0, vdw=0)
2. **Discharging Stage**: Electrostatic interactions are turned off (coul λ: 0→1, bonded=1, vdw=0)  
3. **Vanishing Stage**: van der Waals interactions are turned off (vdw λ: 0→1, bonded=1, coul=1)

Each stage is optimized independently using:
- **Gradient variance analysis** to identify "difficult" regions
- **Statistical inefficiency correction** for accurate error estimates
- **Thermodynamic length minimization** for optimal lambda spacing

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support
- **Email**: jjhuang0520@outlook.com (Jay Huang)