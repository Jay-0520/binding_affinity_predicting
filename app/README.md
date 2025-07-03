# Free Energy Analysis Streamlit App ⚛️

A user-friendly web interface for analyzing GROMACS free energy perturbation (FEP) simulations. This app provides correlation analysis and free energy estimation using multiple statistical methods.

## Features

- **Data Upload & Processing**: Load GROMACS XVG files from FEP simulations
- **Correlation Analysis**: Automatic detection of correlated samples and subsampling
- **Free Energy Estimation**: Multiple methods including TI, BAR, MBAR, and exponential averaging
- **Interactive Visualizations**: Real-time plots and statistical summaries
- **Export Results**: Download analysis results in JSON format

## Requirements

### Python Dependencies
```bash
pip install streamlit
```

### Analysis Package
This app requires your `binding_affinity_predicting` package to be installed and importable. Make sure the following modules are available:
- `binding_affinity_predicting.components.analysis.xvg_data_loader`
- `binding_affinity_predicting.components.analysis.uncorrelate_subsampler`
- `binding_affinity_predicting.components.analysis.free_energy_estimators`

To install `binding_affinity_predicting`, following this [README.md](https://github.com/Jay-0520/binding_affinity_predicting/blob/main/README.md) file

## Usage

### 1. Start the App
```bash
streamlit run streamlit_app.py
```

This will open your default web browser and navigate to `http://localhost:8501`

### 2. Prepare Your Data
Ensure you have GROMACS XVG files from your FEP simulations. Each file should contain:
- dH/dλ time series for each component (coulomb, vdw, bonded)
- Cross-evaluation energies between lambda states
- Proper lambda state information in the header

**Important**: Set `calc-lambda-neighbors = -1` in your GROMACS MDP files to generate cross-evaluation data.

### 3. Analysis Workflow

#### Step 1: Upload Data & Correlation Analysis
1. Navigate to the **"Data Upload & Correlation Analysis"** tab
2. Set global parameters in the sidebar:
   - Temperature (K)
   - Output units (kcal, kJ, or kBT)
   - Skip time for equilibration (ps)
3. Upload your XVG files (one per lambda state)
4. Configure correlation analysis parameters:
   - **Observable**: `dhdl` (recommended), `dhdl_all`, or `de`
   - **Min uncorrelated samples**: Typically 50-100
   - **Fast analysis**: Enable for large datasets
5. Click **"Perform Correlation Analysis"**

#### Step 2: Free Energy Estimation
1. Navigate to the **"Free Energy Estimation"** tab
2. Select methods to run:
   - **TI Methods**: Trapezoidal, Cubic Spline
   - **BAR Methods**: BAR, MBAR, UBAR, RBAR
   - **EXP Methods**: DEXP, IEXP, GDEL, GINS
3. Adjust advanced parameters if needed
4. Click **"Calculate Free Energies"**
5. View results and download JSON export
