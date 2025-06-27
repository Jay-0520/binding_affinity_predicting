# README

## Legacy Reference Scripts

We include the following legacy scripts from the [MobleyLab/alchemical-analysis](https://github.com/MobleyLab/alchemical-analysis) project:

- `alchemical_analysis.py`
- `corruptxvg.py`
- `parser_gromacs.py`
- `unixlike.py`

**Purpose:**\
These scripts are used solely to generate reference free energy (`ΔG`) values from a set of test GROMACS FEP `.xvg` files. We use those reference values to validate our refactored estimator implementations via pytest.

---

## Generating Reference Data

1. First generate a set of `*.xvg` files by performing a short FEP calculation
2. Use `parser_gromacs.py` to load `*.xvg` files and save lambda/energy matrices into a data file (e.g., `lambda_data.pkl`).
   - This data file will be used as direct input for testingthe new free energy estimators
   
3. Run:

   ```bash
   python alchemical_analysis.py -d . -p lambda -t 300 -s 0 -u kcal -w -g -m 'all'
   ```
   Adding this flag `-m 'all'` to return results from all free energy estimators

4. This will produce:

   - `results.txt` (human-readable summary)
   - `results.pickle` (full-precision data)
   - `results.txt` contains `ΔG` values from various estimators which are the `reference data`

---

## Running the Test Suite

With the `reference data` committed, run the corresponding pytests (e.g., `test_free_energy_estimators.py`)

Tests will compare our `ThermodynamicIntegration`, `ExponentialAveraging`, `BennettAcceptanceRatio`, `MultistateBAR`, etc., against the legacy script’s outputs (a.k.a, the `reference data` ).

---

> **Note:**\
> The scripts in this folder are *not* used in production; they exist only to generate reference data, and verify the new implementation 
of free energy estimators

