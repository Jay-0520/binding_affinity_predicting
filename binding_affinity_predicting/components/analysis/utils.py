import numpy as np


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
