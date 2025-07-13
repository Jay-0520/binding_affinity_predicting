"""
we need to run the following python code to generate the SOMD .dat files:

overall_dgs, overall_times = get_time_series_multiwindow_mbar(
    output_dir="/Users/jingjinghuang/Documents/fep_workflow/test_analysis/example_restraint_stage/output",
    lambda_windows=restrain_stage.lam_windows,
    start_frac=0.0,
    end_frac=0.75,
    equilibrated=True,
    run_nos=[1],
)

SOMD .dat files will be stored in ~/example_restraint_stage/output

then we can run this script to load the SOMD .dat files and convert them into a format (pickle file)
compatible with MBAR calculation in:
binding_affinity_predicting.components.analysis.equilibrium_detecter._compute_dg_mbar
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from binding_affinity_predicting.components.analysis.equilibrium_detecter import (
    _compute_dg_mbar,
)

logger = logging.getLogger(__name__)


def read_somd_simfile(simfile_path: str) -> pd.DataFrame:
    """
    Read a SOMD simulation file (.dat format) into a pandas DataFrame.

    Parameters
    ----------
    simfile_path : str
        Path to the SOMD simulation file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for step, potential, gradient, forward/backward Metropolis, u_kl values
    """
    with open(simfile_path, 'r') as f:
        lines = f.readlines()

    # Find the header line with column definitions
    header_line = None
    data_start_idx = 0

    for i, line in enumerate(lines):
        if line.startswith('#') and '[step]' in line:
            # This is the header line with column definitions
            header_line = line.strip('#').strip()
            data_start_idx = i + 1
            break

    # Skip any additional header lines
    while data_start_idx < len(lines) and lines[data_start_idx].startswith('#'):
        data_start_idx += 1

    # Read data lines
    data_lines = []
    for line in lines[data_start_idx:]:
        if not line.startswith('#') and line.strip():
            row_data = line.strip().split()
            data_lines.append(row_data)

    if not data_lines:
        raise ValueError(f"No data found in {simfile_path}")

    # Check data consistency
    col_counts = [len(row) for row in data_lines]
    most_common_cols = max(set(col_counts), key=col_counts.count)

    # Filter rows with consistent column count
    consistent_data = [row for row in data_lines if len(row) == most_common_cols]

    if not consistent_data:
        raise ValueError(f"No consistent data rows found in {simfile_path}")

    # Convert to DataFrame
    df = pd.DataFrame(consistent_data, dtype=float)

    # Set appropriate column names based on SOMD format
    if most_common_cols == 11:
        # Standard SOMD format with u_kl values for each lambda state
        df.columns = [
            'step',
            'potential',
            'gradient',
            'forward_metro',
            'backward_metro',
            'u_kl_0',
            'u_kl_1',
            'u_kl_2',
            'u_kl_3',
            'u_kl_4',
            'u_kl_5',
        ]
    elif most_common_cols >= 5:
        # Basic format with at least the main energy columns
        base_cols = ['step', 'potential', 'gradient', 'forward_metro', 'backward_metro']
        extra_cols = [f'u_kl_{i}' for i in range(most_common_cols - 5)]
        df.columns = base_cols + extra_cols
    else:
        # Fallback to generic column names
        df.columns = [f'col_{i}' for i in range(most_common_cols)]

    return df


def extract_lambda_from_path(file_path: str) -> float:
    """
    Extract lambda value from file path or directory structure.

    Parameters
    ----------
    file_path : str
        Path containing lambda information

    Returns
    -------
    float
        Lambda value
    """
    path_str = str(file_path)

    # Look for lambda in directory name (e.g., lambda_0.000, lambda0.5)
    lambda_match = re.search(r'lambda[_\-]?(\d+\.?\d*)', path_str)
    if lambda_match:
        return float(lambda_match.group(1))

    # Look for lambda in filename
    filename = Path(path_str).name
    lambda_match = re.search(r'(\d+\.?\d*)', filename)
    if lambda_match:
        return float(lambda_match.group(1))

    raise ValueError(f"Could not extract lambda value from path: {file_path}")


def load_somd_alchemical_data_trunc(
    dat_files: List[str],
    temperature: float = 298.15,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
) -> Dict:
    """
    Load alchemical data from SOMD .dat files in a format compatible with MBAR.

    Parameters
    ----------
    dat_files : List[str]
        List of paths to SOMD .dat files
    temperature : float
        Temperature in Kelvin
    start_frac : float
        Fraction of data to start from (0.0 = beginning)
    end_frac : float
        Fraction of data to end at (1.0 = end)

    Returns
    -------
    Dict
        Dictionary with potential_energies array and metadata
    """
    # Sort files by lambda value
    lambda_data = []
    for dat_file in dat_files:
        try:
            lambda_val = extract_lambda_from_path(dat_file)
            lambda_data.append((lambda_val, dat_file))
        except ValueError:
            logger.warning(f"Could not extract lambda from {dat_file}, skipping")
            continue

    lambda_data.sort(key=lambda x: x[0])

    if not lambda_data:
        raise ValueError("No valid lambda files found")

    # Read all data files
    all_dataframes = []
    lambda_values = []

    for lambda_val, dat_file in lambda_data:
        try:
            df = read_somd_simfile(dat_file)
            all_dataframes.append(df)
            lambda_values.append(lambda_val)
            logger.info(
                f"Loaded λ={lambda_val}: {len(df)} snapshots from {Path(dat_file).name}"
            )
        except Exception as e:
            logger.warning(f"Error reading {dat_file}: {e}")
            continue

    if not all_dataframes:
        raise ValueError("No data could be loaded from .dat files")

    # Check if we have u_kl data (SOMD format with energies at all lambda states)
    first_df = all_dataframes[0]
    u_kl_columns = [col for col in first_df.columns if col.startswith('u_kl_')]

    logger.info(f"Found {len(u_kl_columns)} u_kl columns: {u_kl_columns}")

    # Determine the number of snapshots to use
    min_snapshots = min(len(df) for df in all_dataframes)

    # Apply time window
    start_idx = int(start_frac * min_snapshots)
    end_idx = int(end_frac * min_snapshots)

    if end_idx <= start_idx:
        raise ValueError(f"Invalid time window: start={start_idx}, end={end_idx}")

    n_snapshots = end_idx - start_idx
    n_states = len(all_dataframes)

    # Extract potential energy data
    potential_energies = np.zeros((n_states, n_states, n_snapshots))

    # Convert temperature to kT units
    kT = temperature * 8.314462618e-3  # R in kJ/mol/K

    for i, df in enumerate(all_dataframes):
        # Truncate to time window
        windowed_df = df.iloc[start_idx:end_idx].copy()

        if len(u_kl_columns) >= n_states:
            # We have u_kl data for all states - this is the ideal case!
            for j in range(n_states):
                u_kl_col = f'u_kl_{j}'
                if u_kl_col in windowed_df.columns:
                    # u_kl values are already in reduced units in SOMD
                    potential_energies[i, j, :] = windowed_df[u_kl_col].values
                else:
                    logger.warning(
                        f"Missing {u_kl_col} in lambda {lambda_values[i]} data"
                    )
                    # Use potential energy as fallback
                    potential_energies[i, j, :] = windowed_df['potential'].values / kT
        else:
            # Fallback: only have potential energies at simulated state
            logger.warning(
                "Limited u_kl data - using potential energies with approximations"
            )

            # Use potential energy for diagonal elements
            potential_energies[i, i, :] = windowed_df['potential'].values / kT

            # Estimate off-diagonal elements (crude approximation)
            for j in range(n_states):
                if i != j:
                    # Linear interpolation based on lambda difference
                    alpha = abs(lambda_values[i] - lambda_values[j])
                    potential_energies[i, j, :] = potential_energies[i, i, :] * (
                        1 + 0.1 * alpha
                    )

    return {
        'potential_energies': potential_energies,
        'nsnapshots': np.full(n_states, n_snapshots),
        'lambda_values': np.array(lambda_values),
        'temperature': temperature,
        'n_states': n_states,
        'n_snapshots': n_snapshots,
        'u_kl_available': len(u_kl_columns) >= n_states,
    }


def load_a3fe_mbar_data(
    output_dir: str,
    run_no: int,
    percentage_start: float,
    percentage_end: float,
    temperature: float = 298.15,
) -> Dict:
    # pattern = f"simfile_truncated_{round(percentage_end, 3)}_end_{round(percentage_start, 3)}_start.dat"
    pattern = (
        f"simfile_truncated_{percentage_end:.3f}_end_{percentage_start:.3f}_start.dat"
    )
    lambda_dirs = sorted(Path(output_dir).glob("lambda*"))

    if not lambda_dirs:
        raise FileNotFoundError(f"No lambda* directories found in {output_dir}")

    dat_files_with_lambda = []
    for lambda_dir in lambda_dirs:
        # Extract lambda value from directory name
        try:
            lambda_val = extract_lambda_from_path(str(lambda_dir))
        except ValueError:
            logger.warning(f"Could not extract lambda from directory: {lambda_dir}")
            continue

        run_dir = lambda_dir / f"run_{run_no:02d}"
        dat_file = run_dir / pattern

        if dat_file.exists():
            dat_files_with_lambda.append((lambda_val, str(dat_file)))
            logger.info(f"Found file for λ={lambda_val}: {dat_file}")
        else:
            logger.warning(f"Missing file: {dat_file}")

    if not dat_files_with_lambda:
        raise FileNotFoundError(
            f"No .dat files found for run {run_no} with pattern {pattern}"
        )

    # Sort by lambda value (same as analyse_freenrg would do)
    dat_files_with_lambda.sort(key=lambda x: x[0])

    logger.info(f"Loading {len(dat_files_with_lambda)} lambda windows for run {run_no}")
    lambda_values = [item[0] for item in dat_files_with_lambda]
    dat_files = [item[1] for item in dat_files_with_lambda]

    # Load the data using the existing helper function
    alchemical_data = load_somd_alchemical_data_trunc(
        dat_files=dat_files,
        temperature=temperature,
        start_frac=0.0,  # Files are already truncated to the desired window
        end_frac=1.0,
    )

    return alchemical_data


def load_a3fe_mbar_data_complete(
    output_dir: str,
    run_no: int,
    temperature: float = 298.15,
) -> Dict:
    # pattern = f"simfile_truncated_{round(percentage_end, 3)}_end_{round(percentage_start, 3)}_start.dat"
    pattern = f"simfile_equilibrated.dat"
    lambda_dirs = sorted(Path(output_dir).glob("lambda*"))

    if not lambda_dirs:
        raise FileNotFoundError(f"No lambda* directories found in {output_dir}")

    dat_files_with_lambda = []
    for lambda_dir in lambda_dirs:
        try:
            lambda_val = extract_lambda_from_path(str(lambda_dir))
        except ValueError:
            logger.warning(f"Could not extract lambda from directory: {lambda_dir}")
            continue

        run_dir = lambda_dir / f"run_{run_no:02d}"
        dat_file = run_dir / pattern

        if dat_file.exists():
            dat_files_with_lambda.append((lambda_val, str(dat_file)))
            logger.info(f"Found file for λ={lambda_val}: {dat_file}")
        else:
            logger.warning(f"Missing file: {dat_file}")

    if not dat_files_with_lambda:
        raise FileNotFoundError(
            f"No .dat files found for run {run_no} with pattern {pattern}"
        )

    # Sort by lambda value (same as analyse_freenrg would do)
    dat_files_with_lambda.sort(key=lambda x: x[0])

    logger.info(f"Loading {len(dat_files_with_lambda)} lambda windows for run {run_no}")
    lambda_values = [item[0] for item in dat_files_with_lambda]
    dat_files = [item[1] for item in dat_files_with_lambda]

    # Load the data using the existing helper function
    alchemical_data = load_somd_alchemical_data_trunc(
        dat_files=dat_files,
        temperature=temperature,
        start_frac=0.0,
        end_frac=1.0,
    )

    return alchemical_data


if __name__ == "__main__":
    # Test reading a single file
    # dat_file = "path/to/simfile_truncated_47.0_end_46.0_start.dat"
    # df = read_somd_simfile(dat_file)
    # print(f"Loaded data shape: {df.shape}")
    # print(f"Columns: {df.columns.tolist()}")
    # compare_with_a3fe_mbar(
    #     output_dir="/Users/jingjinghuang/Documents/fep_workflow/test_analysis/example_restraint_stage/output",
    #     run_no=1,
    #     percentage_start=99.0,
    #     percentage_end=100.0,
    #     temperature=298.15
    # )

    results = load_a3fe_mbar_data_complete(
        output_dir="/Users/jingjinghuang/Documents/fep_workflow/test_analysis/example_restraint_stage/output",
        run_no=1,
        temperature=298.15,
    )

    # dump results to a pickle file
    import pickle

    with open("mbar_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print('-----results-----', results)
    # Example of how to use the comparison function
    # Test full comparison
    # output_dir = "/path/to/your/output"
    # result = test_mbar_comparison(output_dir, run_no=1)
    # print(f"MBAR result: {result}")
