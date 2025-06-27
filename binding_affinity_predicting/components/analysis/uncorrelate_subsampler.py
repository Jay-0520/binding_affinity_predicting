import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pymbar

from binding_affinity_predicting.components.analysis.utils import (
    get_lambda_components_changing,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def statistical_inefficiency(data: np.ndarray, fast: bool = False) -> float:
    """
    Calculate statistical inefficiency (correlation time) using pymbar or fallback.
    """
    if len(data) < 10:
        return 1.0

    clean = data[np.isfinite(data)]
    if len(clean) < 10:
        logger.warning("Too few finite data points for statistical inefficiency")
        return 1.0

    try:
        return pymbar.timeseries.statisticalInefficiency(clean, fast=fast)
    except Exception as e:
        logger.warning(f"pymbar failed: {e}")
        raise RuntimeError(
            "Failed to calculate statistical inefficiency using pymbar. "
            "Ensure pymbar is installed and data is valid."
        ) from e


def subsample_correlated_data(data: np.ndarray, g: float | None = None) -> np.ndarray:
    """
    Subsample correlated data to get (approximately) independent samples.
    """
    if g is None:
        g = statistical_inefficiency(data)

    try:
        return pymbar.timeseries.subsampleCorrelatedData(data, g=g)
    except Exception:
        step = max(1, int(g))
        return np.arange(0, len(data), step)


def uncorrelate_for_ti(
    dhdl_timeseries: np.ndarray,
    lambda_vectors: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    autocorr_observable: str = 'dhdl',
    min_uncorr_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform autocorrelation analysis for Thermodynamic Integration (TI) methods.

    TI methods only need dH/dλ time series data. This function focuses exclusively
    on extracting uncorrelated dH/dλ samples for reliable integration.

    Adapted from:
    https://github.com/MobleyLab/alchemical-analysis/blob/master/alchemical_analysis/alchemical_analysis.py

    Parameters:
    -----------
    dhdl_timeseries : np.ndarray, shape (num_lambda_states, num_components, max_total_snapshots)
        Time series of dH/dλ values for each lambda state and component.
        Note that this can be loaded from GROMACS XVG files
        - num_lambda_states: Number of lambda windows in the transformation
        - num_components: Number of lambda components (e.g., coulomb, vdw, bonded)
        - max_total_snapshots: Maximum number of snapshots across all lambda states
    lambda_vectors : np.ndarray, shape (num_lambda_states, num_components)
        Lambda parameter values for each state and component.
        Example: [[1.0, 1.0], [0.8, 1.0], [0.6, 1.0], ...] for coulomb+vdw transformation
    start_indices : np.ndarray, shape (num_lambda_states,)
        Starting snapshot indices for analysis
    end_indices : np.ndarray, shape (num_lambda_states,)
        Ending snapshot indices for analysis
    autocorr_observable : str, default 'dhdl'
        Observable for autocorrelation analysis:
        - 'dhdl': Sum over changing components only (recommended)
        - 'dhdl_all': Sum over all components
    min_uncorr_samples : int, default 50
        Minimum number of uncorrelated samples required

    Returns:
    --------
    dhdl_uncorrelated : np.ndarray, shape (num_lambda_states, num_components, max_uncorr_samples)
        Uncorrelated dH/dλ samples for TI integration
    num_uncorr_samples_per_state : np.ndarray, shape (num_lambda_states,)
        Number of independent samples from each lambda state

    Notes:
    ------
    - Designed specifically for TI methods (trapezoidal, cubic spline integration)
    - Only analyzes dH/dλ data - no cross-evaluation energies needed
    - Focuses on physically relevant components (those actively changing)
    """
    logger.info("=== Thermodynamic Integration (TI) Autocorrelation Analysis ===")

    # Validate autocorr_observable for TI
    if autocorr_observable.lower() == 'de':
        raise ValueError(
            "'dE' observable not applicable for TI methods. Use 'dhdl' or 'dhdl_all'"
        )

    # Determine which components are changing
    lambda_components_changing = get_lambda_components_changing(lambda_vectors)

    num_lambda_states, num_components = lambda_vectors.shape
    max_analysis_window = max(end_indices - start_indices)

    # Initialize outputs
    num_uncorr_samples_per_state = np.zeros(num_lambda_states, dtype=int)
    correlation_times = np.zeros(num_lambda_states, dtype=float)
    dhdl_uncorrelated = np.zeros(
        [num_lambda_states, num_components, max_analysis_window], dtype=float
    )

    logger.info(
        f"{'Lambda':>6} {'Window_Size':>12} {'Uncorr_N':>12} {'Corr_Time_g':>15} {'Components':>12}"
    )

    # Process each lambda state
    for lambda_idx in range(num_lambda_states):
        # Define which portion of the time series to analyze for this lambda state.
        window_start = start_indices[lambda_idx]
        window_end = end_indices[lambda_idx]
        window_size = window_end - window_start

        # Construct observable for autocorrelation analysis
        if autocorr_observable.lower() == 'dhdl':
            # Find last repeated lambda state (handles duplicate lambda vectors)
            # Use sum over changing components only (physics-informed choice)
            # Find last repeated lambda state for correct component indexing
            # TODO: why? Sometimes the same lambda vector appears multiple times in
            # different positions (not so common?) - by JJH 2025-06-21
            last_repeated_lambda = lambda_idx
            for other_lambda in range(num_lambda_states):
                if np.array_equal(
                    lambda_vectors[lambda_idx], lambda_vectors[other_lambda]
                ):
                    last_repeated_lambda = other_lambda

            # changing_components -> [False, True, False]; there will only be one "True" value
            # at each lambda state based on the mdp file settings - by JJH 2025-06-21
            changing_components = lambda_components_changing[last_repeated_lambda]
            # dhdl_timeseries[lambda_idx, :, :]: All data for current lambda state
            # [:, changing_components, :]: Only changing components
            # [:, :, window_start:window_end]: Only analysis time window
            # np.sum(..., axis=0): Sum over components (must be one component?) → single time series
            # e.g., dhdl_timeseries[2, [False, True, False], 1000:9000] -> Shape: (8000,)
            # only "vdW" component (is this a fixed position in GROMACS?), 8000 time points
            observable_series = np.sum(
                dhdl_timeseries[
                    lambda_idx, changing_components, window_start:window_end
                ],
                axis=0,
            )

        elif autocorr_observable.lower() == 'dhdl_all':
            # Use sum over all components (more conservative)
            # NOTE: this is less physically meaningful but can be useful - by JJH 2025-06-21
            observable_series = np.sum(
                dhdl_timeseries[lambda_idx, :, window_start:window_end], axis=0
            )
        else:
            raise ValueError(f"Unknown autocorr_observable: '{autocorr_observable}'")

        # Calculate statistical inefficiency (correlation time)
        if not np.any(observable_series):
            # Handle all-zeros case (no fluctuations)
            correlation_times[lambda_idx] = 1.0
            logger.info(
                f"WARNING: No fluctuations detected for lambda state {lambda_idx}, setting g=1"
            )
        else:
            correlation_times[lambda_idx] = statistical_inefficiency(observable_series)

        # Extract indices of statistically independent samples
        # Note: subsample_correlated_data returns indices relative to observable_series
        relative_uncorr_indices = np.array(
            subsample_correlated_data(
                observable_series, g=correlation_times[lambda_idx]
            )
        )
        global_uncorr_indices = window_start + relative_uncorr_indices
        num_uncorr_found = len(global_uncorr_indices)  # number of uncorrelated samples

        # Handle insufficient uncorrelated samples
        if num_uncorr_found < min_uncorr_samples:
            logger.info(
                f"WARNING: Only {num_uncorr_found} uncorrelated samples found at "
                f"lambda state {lambda_idx}; using all {window_size} samples from analysis window"
            )
            # Fall back to using all samples in the analysis window
            # basically equal to len(observable_series)
            global_uncorr_indices = window_start + np.arange(window_size)
            num_samples_to_store = window_size
        else:
            num_samples_to_store = num_uncorr_found

        num_uncorr_samples_per_state[lambda_idx] = num_samples_to_store

        # Store uncorrelated dH/dλ samples for all components
        for component_idx in range(num_components):
            dhdl_uncorrelated[lambda_idx, component_idx, :num_samples_to_store] = (
                dhdl_timeseries[lambda_idx, component_idx, global_uncorr_indices]
            )

        logger.info(
            f"{lambda_idx:>6} {window_size:>12} {num_samples_to_store:>12} "
            f"{correlation_times[lambda_idx]:>15.2f}"
        )

    total_samples = num_uncorr_samples_per_state.sum()
    logger.info(f"\nTI Analysis Summary:")
    logger.info(f"  Total uncorrelated samples: {total_samples}")
    logger.info(f"  Ready for TI integration methods")

    return dhdl_uncorrelated, num_uncorr_samples_per_state


# TODO: consider remove duplicated code with uncorrelate_for_ti() - by JJH 2025-06-22
def uncorrelate_for_mbar_bar_exp(
    potential_energies: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    autocorr_observable: str = 'dE',
    min_uncorr_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform autocorrelation analysis for MBAR/BAR methods, as well as
    for exponential‐averaging methods

    MBAR/BAR methods need cross-evaluation potential energies. This function focuses
    on extracting uncorrelated potential energy samples for overlap analysis.

    Adapted from:
    https://github.com/MobleyLab/alchemical-analysis/blob/master/alchemical_analysis/alchemical_analysis.py

    Parameters:
    -----------
    potential_energies : np.ndarray or None, shape (num_lambda_states, num_lambda_states, max_total_snapshots)
        Note that this can be loaded from GROMACS EDR files
        Reduced potential energies U_klt where:
        - First index: lambda state where snapshot was generated
        - Second index: lambda state where energy is evaluated
        - Third index: snapshot/time index
        Set to None if not available (e.g., for some software packages)

        We need this because MBAR needs overlap information between ALL pairs of lambda states
          for example, for 4 lambda states (λ₀, λ₁, λ₂, λ₃): lambda_states = [0.0, 0.3, 0.7, 1.0]
            For lambda_idx = 1 (λ = 0.3), we need:
            potential_energies[1, 0, :] = U₀(snapshots_from_λ₁)  # Evaluate λ₁ snapshots at λ₀
            potential_energies[1, 1, :] = U₁(snapshots_from_λ₁)  # Evaluate λ₁ snapshots at λ₁
            potential_energies[1, 2, :] = U₂(snapshots_from_λ₁)  # Evaluate λ₁ snapshots at λ₂
            potential_energies[1, 3, :] = U₃(snapshots_from_λ₁)  # Evaluate λ₁ snapshots at λ₃

          -> physical meaning: how much energy would these λ₁ configurations have at other λ states
            (full interactions, native state, weaker interactions, and non-interacting particles)?

        Note that BAR only needs adjacent pairs of lambda states, e.g.,
            W_forward = U₁(snapshots_from_λ₀) - U₀(snapshots_from_λ₀)
            W_reverse = U₀(snapshots_from_λ₁) - U₁(snapshots_from_λ₁)

    start_indices : np.ndarray, shape (num_lambda_states,)
        Starting snapshot indices for analysis
    end_indices : np.ndarray, shape (num_lambda_states,)
        Ending snapshot indices for analysis
    autocorr_observable : str, default 'dE'
        Observable for autocorrelation analysis:
        - 'dE': Energy differences between adjacent states (recommended for MBAR/BAR)
    min_uncorr_samples : int, default 50
        Minimum number of uncorrelated samples required
    verbose : bool, default False
        Print detailed analysis information

    Returns:
    --------
    uncorr_potential_energies : np.ndarray, shape (num_lambda_states, num_lambda_states, max_uncorr_samples)
        Uncorrelated potential energy cross-evaluations for MBAR/BAR
    num_uncorr_samples_per_state : np.ndarray, shape (num_lambda_states,)
        Number of independent samples from each lambda state

    Notes:
    ------
    - Designed specifically for MBAR/BAR methods
    - Analyzes cross-evaluation energies for phase space overlap
    - Uses energy differences (dE) as default observable for better correlation analysis
    """
    logger.info("=== MBAR/BAR Autocorrelation Analysis ===")

    # Validate inputs
    if autocorr_observable.lower() not in ['de']:
        logger.warning(
            f"Observable '{autocorr_observable}' not typical for MBAR/BAR. Consider using 'dE'"
        )

    num_lambda_states = potential_energies.shape[0]
    max_analysis_window = max(end_indices - start_indices)

    # Initialize outputs
    num_uncorr_samples_per_state = np.zeros(num_lambda_states, dtype=int)
    correlation_times = np.zeros(num_lambda_states, dtype=float)
    uncorr_potential_energies = np.zeros(
        [num_lambda_states, num_lambda_states, max_analysis_window], dtype=np.float64
    )

    logger.info(
        f"{'Lambda':>6} {'Window_Size':>12} {'Uncorr_N':>12} {'Corr_Time_g':>15} {'Observable':>12}"
    )

    # Process each lambda state
    for lambda_idx in range(num_lambda_states):
        window_start = start_indices[lambda_idx]
        window_end = end_indices[lambda_idx]
        window_size = window_end - window_start

        if autocorr_observable.lower() == 'de':
            # Use energy differences between adjacent states
            if lambda_idx < num_lambda_states - 1:
                # Forward energy difference: U(λ_{i+1}) - U(λ_i)
                target_lambda = lambda_idx + 1
                observable_series = potential_energies[
                    lambda_idx, target_lambda, window_start:window_end
                ]
            else:
                # Backward energy difference for last lambda state
                target_lambda = lambda_idx - 1
                observable_series = potential_energies[
                    lambda_idx, target_lambda, window_start:window_end
                ]
        else:
            raise ValueError(
                f"Observable '{autocorr_observable}' not supported for MBAR/BAR analysis"
            )

        # Calculate correlation time
        if not np.any(observable_series):
            correlation_times[lambda_idx] = 1.0
            logger.warning(
                f"No fluctuations in energy differences for lambda state {lambda_idx}"
            )
        else:
            correlation_times[lambda_idx] = statistical_inefficiency(observable_series)

        # Extract uncorrelated samples
        relative_uncorr_indices = np.array(
            subsample_correlated_data(
                observable_series, g=correlation_times[lambda_idx]
            )
        )
        global_uncorr_indices = window_start + relative_uncorr_indices
        num_uncorr_found = len(global_uncorr_indices)

        # Handle insufficient uncorrelated samples
        if num_uncorr_found < min_uncorr_samples:
            logger.info(
                f"WARNING: Only {num_uncorr_found} uncorrelated samples found at "
                f"lambda state {lambda_idx}; using all {window_size} samples from analysis window"
            )
            # Fall back to using all samples in the analysis window
            # basically equal to len(observable_series)
            global_uncorr_indices = window_start + np.arange(window_size)
            num_samples_to_store = window_size
        else:
            num_samples_to_store = num_uncorr_found

        num_uncorr_samples_per_state[lambda_idx] = num_samples_to_store

        # Store uncorrelated potential energy samples for all lambda state pairs
        for target_lambda in range(num_lambda_states):
            uncorr_potential_energies[
                lambda_idx, target_lambda, :num_samples_to_store
            ] = potential_energies[lambda_idx, target_lambda, global_uncorr_indices]

        logger.info(
            f"{lambda_idx:>6} {window_size:>12} {num_samples_to_store:>12} "
            f"{correlation_times[lambda_idx]:>15.2f}"
        )

    total_samples = num_uncorr_samples_per_state.sum()
    logger.info(f"\nMBAR/BAR Analysis Summary:")
    logger.info(f"  Total uncorrelated samples: {total_samples}")
    logger.info(f"  Cross-evaluation matrix ready for MBAR/BAR")

    return uncorr_potential_energies, num_uncorr_samples_per_state


def uncorrelate_for_all_methods(
    dhdl_timeseries: np.ndarray,
    lambda_vectors: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    potential_energies: Optional[np.ndarray] = None,
    ti_observable: str = 'dhdl',
    mbar_observable: str = 'dE',
    min_uncorr_samples: int = 50,
) -> dict:
    """
    Convenience function to perform autocorrelation analysis for all free energy estimation methods.

    Parameters:
    -----------
    dhdl_timeseries : np.ndarray
        dH/dλ time series data
    lambda_vectors : np.ndarray
        Lambda parameter values
    start_indices, end_indices : np.ndarray
        Analysis window boundaries
    potential_energies : np.ndarray, optional
        Cross-evaluation energies (required for MBAR/BAR)
    ti_observable : str, default 'dhdl'
        Observable for TI analysis
    mbar_observable : str, default 'dE'
        Observable for MBAR/BAR analysis
    min_uncorr_samples : int, default 50
        Minimum uncorrelated samples threshold

    Returns:
    --------
    dict with keys:
        - 'ti': Results for TI methods (dhdl_uncorrelated, num_uncorr_samples)
        - 'mbar': Results for MBAR/BAR methods (if potential_energies provided)
        - 'summary': Analysis summary statistics
    """
    logger.info("=== Complete Free Energy Autocorrelation Analysis ===")

    results = {}
    # TI Analysis (always possible with dH/dλ data)
    logger.info("\n1. Analyzing for Thermodynamic Integration (TI)...")

    dhdl_uncorr, ti_num_samples = uncorrelate_for_ti(
        dhdl_timeseries=dhdl_timeseries,
        lambda_vectors=lambda_vectors,
        start_indices=start_indices,
        end_indices=end_indices,
        autocorr_observable=ti_observable,
        min_uncorr_samples=min_uncorr_samples,
    )

    results['ti'] = {
        'dhdl_uncorrelated': dhdl_uncorr,
        'num_uncorr_samples_per_state': ti_num_samples,
        'method': 'TI',
        'observable': ti_observable,
    }

    # MBAR/BAR Analysis (only if potential energies available)
    if potential_energies is not None:
        logger.info("\n2. Analyzing for MBAR/BAR methods...")

        potential_uncorr, mbar_num_samples = uncorrelate_for_mbar_bar(
            potential_energies=potential_energies,
            lambda_vectors=lambda_vectors,
            start_indices=start_indices,
            end_indices=end_indices,
            autocorr_observable=mbar_observable,
            min_uncorr_samples=min_uncorr_samples,
        )

        results['mbar'] = {
            'uncorr_potential_energies': potential_uncorr,
            'num_uncorr_samples_per_state': mbar_num_samples,
            'method': 'MBAR/BAR',
            'observable': mbar_observable,
        }
    else:
        logger.info("\n2. Skipping MBAR/BAR analysis (no potential_energies provided)")
        results['mbar'] = None

    # Summary statistics
    results['summary'] = {
        'num_lambda_states': len(lambda_vectors),
        'ti_total_samples': ti_num_samples.sum(),
        'mbar_total_samples': (
            mbar_num_samples.sum() if potential_energies is not None else 0
        ),
        'methods_available': ['TI']
        + (['MBAR', 'BAR'] if potential_energies is not None else []),
    }

    logger.info(f"\n=== Analysis Complete ===")
    logger.info(f"Available methods: {results['summary']['methods_available']}")
    logger.info(f"TI effective samples: {results['summary']['ti_total_samples']}")
    if potential_energies is not None:
        logger.info(
            f"MBAR effective samples: {results['summary']['mbar_total_samples']}"
        )

    return results
