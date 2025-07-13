import logging
from pathlib import Path
from typing import Optional

import numpy as np

from binding_affinity_predicting.components.analysis.free_energy_estimators import (
    FreeEnergyEstimator,
)
from binding_affinity_predicting.components.analysis.xvg_data_loader import (
    load_alchemical_data,
)
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    LambdaWindow,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_lambda_components_changing(lambda_vectors: np.ndarray) -> np.ndarray:
    """
    Determine which lambda components are actively changing at each state.

    A component is considered "changing" at a lambda state if it differs from
    either the previous or next lambda state. This identifies which energy
    components are actively participating in the transformation at each point.

    Parameters:
    -----------
    lambda_vectors : np.ndarray, shape (num_lambda_states, num_components)
        Lambda parameter values for each state and component

    Returns:
    --------
    components_changing : np.ndarray, shape (num_lambda_states, num_components), dtype=bool
        Boolean mask where True indicates the component is changing at that lambda state

    Examples:
    ---------
    >>> lambda_vecs = np.array([[1.0, 1.0], [0.5, 1.0], [0.0, 1.0], [0.0, 0.5], [0.0, 0.0]])
    >>> changing = get_lambda_components_changing(lambda_vecs)
    >>> changing
    array([[ True, False],   # State 0: coulomb changing (1.0→0.5)
           [ True, False],   # State 1: coulomb changing (0.5→0.0)
           [ True,  True],   # State 2: coulomb changing (0.0), vdw starting (1.0→0.5)
           [False,  True],   # State 3: vdw changing (0.5→0.0)
           [False, False]])  # State 4: nothing changing (final state)
    """
    num_lambda_states, num_components = lambda_vectors.shape
    components_changing = np.zeros([num_lambda_states, num_components], dtype=bool)

    for component_idx in range(num_components):
        for lambda_idx in range(num_lambda_states - 1):
            # Check if component value changes between consecutive lambda states
            current_value = lambda_vectors[lambda_idx, component_idx]
            next_value = lambda_vectors[lambda_idx + 1, component_idx]

            if (
                abs(next_value - current_value) > 1e-10
            ):  # Use small tolerance for float comparison
                # Mark both current and next states as having this component changing
                components_changing[lambda_idx, component_idx] = True
                components_changing[lambda_idx + 1, component_idx] = True

    return components_changing


def calculate_beta_parameter(
    temperature: float = 298.15, units: str = 'kJ', software: str = 'Gromacs'
) -> float:
    """
    Calculate beta and beta_report parameters exactly as in alchemical_analysis.py:
    https://github.com/MobleyLab/alchemical-analysis/blob/master/alchemical_analysis/alchemical_analysis.py

    This replicates the checkUnitsAndMore() function logic.

    Parameters:
    -----------
    temperature : float, default 298.15
        Temperature in Kelvin
    units : str, default 'kJ'
        Output units: 'kJ', 'kcal', or 'kBT'
    software : str, default 'Gromacs'
        Software package name (affects kcal handling)

    Returns:
    --------
    tuple[float, float] : (beta, beta_report)
        beta: 1/(kB*T) in simulation units
        beta_report: conversion factor for output units
    """
    # Boltzmann constant from original (kJ/mol/K)
    kB = 1.3806488 * 6.02214129 / 1000.0  # Exact value from alchemical_analysis.py
    beta = 1.0 / (kB * temperature)

    # Check if software uses kcal (Sire, Amber vs Gromacs)
    b_kcal = software.upper() in ['SIRE', 'AMBER']

    if units.lower() == 'kj':
        beta_report = beta / (4.184**b_kcal)
    elif units.lower() == 'kcal':
        beta_report = (4.184 ** (not b_kcal)) * beta
    elif units.lower() == 'kbt':
        beta_report = 1.0
    else:
        raise ValueError(
            f"Unknown units '{units}': only 'kJ', 'kcal', and 'kBT' supported"
        )

    return beta_report


# This function is only used by equilibrium_detecter.py
def _load_alchemical_data_for_equil_detect(
    lambda_windows: list[LambdaWindow],
    run_no: int,
    temperature: float = 298.15,
    skip_time: float = 0.0,
    reduce_to_dimensionless: bool = True,
    use_equilibrated: bool = False,
) -> Optional[dict]:
    """Load alchemical data for a specific run across all lambda windows."""
    xvg_files = []
    for window in lambda_windows:
        run_dir = Path(window.output_dir) / f"run_{run_no}"
        if use_equilibrated:
            # Look for equilibrated simulation files
            xvg_file = (
                run_dir / f"lambda_{window.lam_state}_run_{run_no}_equilibrated.xvg"
            )
            # Or however your equilibrated files are named
        else:
            xvg_file = run_dir / f"lambda_{window.lam_state}_run_{run_no}.xvg"

        if xvg_file.exists():
            xvg_files.append(xvg_file)
        else:
            logger.warning(f"XVG file not found: {xvg_file}")
            return None

    if not xvg_files:
        logger.warning(f"No XVG files found for run {run_no}")
        return None

    # Load alchemical data
    alchemical_data = load_alchemical_data(
        xvg_files=xvg_files,
        skip_time=skip_time,
        temperature=temperature,
        reduce_to_dimensionless=reduce_to_dimensionless,
    )

    return alchemical_data


# This function is only used by equilibrium_detecter.py
def _compute_dg_mbar(
    run_no: int,
    start_frac: float,
    end_frac: float,
    lambda_windows: list[LambdaWindow],
    equilibrated: bool = False,
    temperature: float = 298.15,
    units: str = "kcal",
) -> float:
    """
    Helper function to compute free energy change using MBAR for a list of time windows.

    This function is designed to be used with multiprocessing.

    Parameters
    ----------
    run_no : int
        Run number to analyze
    start_frac : float
        Start fraction of simulation time
    end_frac : float
        End fraction of simulation time
    lambda_windows : List[LambdaWindow]
        List of lambda windows
    equilibrated : bool
        Whether to use equilibration times
    temperature : float
        Temperature in Kelvin
    units : str
        Units for output

    Returns
    -------
    float
        Free energy change from MBAR
    """
    try:
        # Load data for this run
        alchemical_data = _load_alchemical_data_for_equil_detect(
            lambda_windows=lambda_windows,
            run_no=run_no,
            temperature=temperature,
            skip_time=0.0,
            use_equilibrated=equilibrated,
        )

        if alchemical_data is None:
            logger.warning(
                f"No alchemical data found for run {run_no}. Skipping MBAR computation."
            )
            return np.nan

        potential_energies = alchemical_data['potential_energies']
        nsnapshots = alchemical_data['nsnapshots']

        # Determine time window indices
        total_snapshots = min(nsnapshots)  # Use minimum to ensure all windows have data

        equil_offset = 0
        if equilibrated:
            raise NotImplementedError(
                "Equilibration handling is not implemented in this function _compute_dg_mbar()."
            )

        # Calculate time window boundaries
        start_idx = equil_offset + int(start_frac * (total_snapshots - equil_offset))
        end_idx = equil_offset + int(end_frac * (total_snapshots - equil_offset))

        if end_idx <= start_idx:
            logger.warning(
                f"Invalid time window for run {run_no}: {start_idx}-{end_idx}"
            )
            return np.nan

        # Extract time window data
        window_potential = potential_energies[:, :, start_idx:end_idx]
        window_samples = np.full(len(lambda_windows), end_idx - start_idx, dtype=int)

        # TODO: do we really need this check? if so, what is the cutoff?
        if window_samples[0] < 2:
            logger.warning(f"Too few samples in time window for run {run_no}")
            return np.nan

        # Run MBAR on this time window
        estimator = FreeEnergyEstimator(
            temperature=temperature, units=units, software="Gromacs"
        )

        result = estimator.estimate_mbar(
            potential_energies=window_potential,
            num_samples_per_state=window_samples,
            regular_estimate=False,  # Just return endpoint free energy
        )

        if result['success']:
            return result['free_energy']
        else:
            logger.warning(
                f"MBAR failed for run {run_no}, time window {start_frac}-{end_frac}"
            )
            return np.nan

    except Exception as e:
        logger.warning(f"Error computing MBAR for run {run_no}: {e}")
        return np.nan
