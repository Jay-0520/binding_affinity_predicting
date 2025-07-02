import logging
from typing import Optional, Tuple

import numpy as np
import pymbar

from binding_affinity_predicting.components.analysis.autocorrelation import (
    _statistical_inefficiency_chodera,
)
from binding_affinity_predicting.components.analysis.utils import (
    get_lambda_components_changing,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calc_statistical_inefficiency(
    data: np.ndarray,
    fast: bool = False,
    method: str = "pymbar",
    preprocessing: Optional[str] = None,
) -> float:
    """
    Calculate statistical inefficiency (correlation time) using pymbar.

    g = 1 -> PERFECT independence - each data point is completely uncorrelated
    g > 1 -> data points are correlated, g is the correlation time
        - g = 5 -> Only 1 in 5 data points is truly independent
        - g = 20 -> Only 1 in 20 data points is truly independent

    Parameters:
    -----------
    data : np.ndarray
        Time series data for correlation analysis
    fast : bool, default False
        Use faster algorithm for large datasets

    Returns:
    --------
    float
        Statistical inefficiency (correlation time)
    """
    if len(data) == 0:
        raise ValueError("Cannot calculate statistical inefficiency for empty data")

    if len(data) < 10:
        return 1.0

    clean_data = data[np.isfinite(data)]
    if len(clean_data) < 10:
        logger.warning("Too few finite data points for statistical inefficiency")
        return 1.0

    # Apply preprocessing
    if preprocessing is not None:
        clean_data = _preprocess_timeseries(
            data=clean_data, preprocessing=preprocessing
        )

    # Zero variance → no correlation
    if np.var(clean_data) == 0.0:
        return 1.0

    try:
        if method == "pymbar":
            return float(
                pymbar.timeseries.statistical_inefficiency(clean_data, fast=fast)
            )
        elif method == "chodera":
            return _statistical_inefficiency_chodera(clean_data, fast=fast)
        else:
            raise ValueError(f"Unknown method: {method}")

    except Exception as e:
        logger.warning(
            f"Statistical inefficiency calculation failed with {method}: {e}"
        )
        raise


# TODO: need to implement this function properly
def _preprocess_timeseries(data: np.ndarray, preprocessing: str) -> np.ndarray:
    """
    Most autocorrelation routines (e.g. in pymbar or algorithms from Chodera’s lab)
    assume time series is stationary: its mean and variance stay roughly constant over time

    Parameters
    ----------
    data : np.ndarray
        Input time series data
    preprocessing : str
        Preprocessing method to apply

    Returns
    -------
    np.ndarray
        Preprocessed data
    """
    if preprocessing == "none":
        return data.copy()

    elif preprocessing == "center":
        # Center data (subtract mean)
        return data - np.mean(data)

    elif preprocessing == "standardize":
        # Standardize data (zero mean, unit variance)
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        if std_val == 0:
            logger.warning("Cannot standardize data with zero variance")
            return data - mean_val
        return (data - mean_val) / std_val

    elif preprocessing == "detrend":
        # Remove linear trend
        x = np.arange(len(data))
        # Simple linear regression
        slope = np.cov(x, data)[0, 1] / np.var(x)
        intercept = np.mean(data) - slope * np.mean(x)
        trend = slope * x + intercept
        return data - trend

    elif preprocessing == "diff":
        # Use first differences
        if len(data) < 2:
            logger.warning("Cannot compute differences for data with < 2 points")
            return data
        return np.diff(data)

    else:
        raise ValueError(
            f"Unknown preprocessing method: {preprocessing}. "
            f"Use 'none', 'center', 'standardize', 'detrend', or 'diff'"
        )


def subsample_correlated_data(
    data: np.ndarray, g: Optional[float] = None
) -> np.ndarray:
    """
    Subsample correlated data to get approximately independent samples.

    Parameters:
    -----------
    data : np.ndarray
        Correlated time series data
    g : float, optional
        Statistical inefficiency. If None, will be calculated automatically

    Returns:
    --------
    np.ndarray
        Indices of uncorrelated samples
    """
    if g is None:
        g = calc_statistical_inefficiency(data)

    try:
        return pymbar.timeseries.subsample_correlated_data(data, g=g)
    except Exception:
        step = max(1, int(g))
        return np.arange(0, len(data), step)


def _find_last_repeated_lambda_state(state_idx: int, lambda_vectors: np.ndarray) -> int:
    """
    Find the last occurrence of a repeated lambda state.

    Sometimes the same lambda vector appears multiple times in different positions.
    This function finds the last occurrence for consistent component indexing.

    Parameters:
    -----------
    state_idx : int
        Current lambda state index
    lambda_vectors : np.ndarray
        Lambda parameter vectors for all states

    Returns:
    --------
    int
        Index of the last repeated lambda state
    """
    current_lambda = lambda_vectors[state_idx]
    last_repeated = state_idx

    for other_idx in range(len(lambda_vectors)):
        if np.array_equal(current_lambda, lambda_vectors[other_idx]):
            last_repeated = other_idx

    return last_repeated


def _construct_observable_series(
    observable: str,
    lambda_state_idx: int,
    dhdl_timeseries: np.ndarray,
    potential_energies: Optional[np.ndarray],
    lambda_vectors: np.ndarray,
    lambda_components_changing: np.ndarray,
    window_start: int,
    window_end: int,
) -> np.ndarray:
    """
    Construct observable time series for correlation analysis.


    Adapted from:
    https://github.com/MobleyLab/alchemical-analysis/blob/master/alchemical_analysis/alchemical_analysis.py

    Parameters:
    -----------
    observable : str
        Type of observable ('dhdl', 'dhdl_all', or 'de')
    lambda_state_idx : int
        Current lambda state index
    dhdl_timeseries : np.ndarray
        dH/dλ time series data

    potential_energies : np.ndarray or None, shape (num_lambda_states, num_lambda_states,
       max_total_snapshots)
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


    lambda_vectors : np.ndarray
        Lambda parameter vectors
    lambda_components_changing : np.ndarray
        Boolean mask of changing components per state
    window_start, window_end : int
        Analysis window boundaries

    Returns:
    --------
    np.ndarray
        Observable time series for correlation analysis
    """
    num_lambda_states = len(lambda_vectors)

    if observable == 'dhdl':
        # Use only changing components (physics-based choice)
        reference_state = _find_last_repeated_lambda_state(
            lambda_state_idx, lambda_vectors
        )
        # changing_mask -> [False, True, False]; there will only be one "True" value
        # at each lambda state based on the mdp file settings - by JJH 2025-06-21
        changing_mask = lambda_components_changing[reference_state]

        if not np.any(changing_mask):
            logger.warning(f"No changing components for state {lambda_state_idx}")
            return np.zeros(window_end - window_start)

        # dhdl_timeseries[lambda_idx, :, :]: All data for current lambda state
        # [:, changing_components, :]: Only changing components
        # [:, :, window_start:window_end]: Only analysis time window
        # np.sum(..., axis=0): Sum over components (must be one component?) → single time series
        # e.g., dhdl_timeseries[2, [False, True, False], 1000:9000] -> Shape: (8000,)
        # only "vdW" component (is this a fixed position in GROMACS?), 8000 time points
        return np.sum(
            dhdl_timeseries[lambda_state_idx, changing_mask, window_start:window_end],
            axis=0,
        )

    elif observable == 'dhdl_all':
        # Use sum over all components (more conservative)
        # NOTE: this is less physically meaningful but can be useful - by JJH 2025-06-21
        return np.sum(
            dhdl_timeseries[lambda_state_idx, :, window_start:window_end], axis=0
        )

    elif observable == 'de':
        # Use energy differences between adjacent states
        if potential_energies is None:
            raise ValueError("potential_energies required for 'de' observable")

        if lambda_state_idx < num_lambda_states - 1:
            target_state = lambda_state_idx + 1
        else:
            target_state = lambda_state_idx - 1

        return potential_energies[
            lambda_state_idx, target_state, window_start:window_end
        ]

    else:
        raise ValueError(
            f"Unknown observable: '{observable}'. Use 'dhdl', 'dhdl_all', or 'de'"
        )


def perform_uncorrelating_subsampling(
    dhdl_timeseries: np.ndarray,
    lambda_vectors: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    potential_energies: Optional[np.ndarray] = None,
    observable: str = 'dhdl',
    min_uncorr_samples: int = 50,
    fast_analysis: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Perform autocorrelation analysis and extract uncorrelated samples.

    All free energy methods can use the same uncorrelated samples for consistency.
    TI methods only need dH/dλ time series data, whereas MBAR/BAR methods need cross-evaluation
    potential energies.

    Adapted from:
    https://github.com/MobleyLab/alchemical-analysis/blob/master/alchemical_analysis/alchemical_analysis.py

    Parameters:
    -----------
    dhdl_timeseries : np.ndarray, shape (num_lambda_states, num_components, max_snapshots)
        Time series of dH/dλ values
    lambda_vectors : np.ndarray, shape (num_lambda_states, num_components)
        Lambda parameter values for each state and component
    start_indices, end_indices : np.ndarray, shape (num_lambda_states,)
        Analysis window boundaries for each lambda state
    end_indices : np.ndarray, shape (num_lambda_states,)

    potential_energies : np.ndarray, optional, shape (num_lambda_states, num_lambda_states,
      max_snapshots)
        Cross-evaluation potential energies (required for 'de' observable)
    observable : str, default 'dhdl'
        Observable for correlation analysis:
        - 'dhdl': Sum over changing components (recommended, physics-based)
        - 'dhdl_all': Sum over all components (conservative)
        - 'de': Energy differences between adjacent states (good for overlap analysis)
    min_uncorr_samples : int, default 50
        Minimum uncorrelated samples threshold
    fast_analysis : bool, default False
        Use faster algorithms for large datasets

    Returns:
    --------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        - dhdl_uncorrelated : shape (num_lambda_states, num_components, max_uncorr_samples)
          Uncorrelated dH/dλ data for TI methods (None if not available)
        - potential_energies_uncorrelated : shape (num_lambda_states, num_lambda_states,
           max_uncorr_samples)
          Uncorrelated potential energies for MBAR/BAR/EXP (None if not available)
        - num_samples_per_state : shape (num_lambda_states,)
          Number of uncorrelated samples per lambda state

    Notes:
    ------
    - Implements global correlation analysis: all methods use same uncorrelated samples
    - dH/dλ data always extracted (needed for TI methods)
    - Potential energy data extracted if available (needed for MBAR/BAR/EXP methods)
    - Choice of observable affects correlation analysis but not final data extraction
    """
    logger.info(f"=== Autocorrelation Analysis (observable: {observable}) ===")

    # Input validation
    if dhdl_timeseries.ndim != 3:
        raise ValueError("dhdl_timeseries must be 3D: (states, components, snapshots)")

    if observable == 'de' and potential_energies is None:
        raise ValueError("potential_energies required for 'de' observable")

    if observable not in ['dhdl', 'dhdl_all', 'de']:
        raise ValueError(
            f"Unknown observable: '{observable}'. Use 'dhdl', 'dhdl_all', or 'de'"
        )

    num_lambda_states, num_components, _ = dhdl_timeseries.shape
    max_window_size = max(end_indices - start_indices)

    # Pre-calculate changing components for dH/dλ observables
    lambda_components_changing = None
    if observable in ['dhdl', 'dhdl_all']:
        lambda_components_changing = get_lambda_components_changing(lambda_vectors)

    # Initialize results
    num_uncorr_samples_per_state = np.zeros(num_lambda_states, dtype=int)
    correlation_times = np.zeros(num_lambda_states, dtype=float)
    uncorr_indices_per_state = []

    logger.info(
        f"{'State':>6} {'Window':>8} {'Uncorr':>8} {'g':>10} {'Observable':>12}"
    )

    # Process each lambda state to determine uncorrelated indices
    for lambda_state_idx in range(num_lambda_states):
        window_start = start_indices[lambda_state_idx]
        window_end = end_indices[lambda_state_idx]
        window_size = window_end - window_start

        try:
            # Construct observable time series
            observable_series = _construct_observable_series(
                observable=observable,
                lambda_state_idx=lambda_state_idx,
                dhdl_timeseries=dhdl_timeseries,
                potential_energies=potential_energies,
                lambda_vectors=lambda_vectors,
                lambda_components_changing=lambda_components_changing,
                window_start=window_start,
                window_end=window_end,
            )

            # Calculate correlation time
            if np.all(observable_series == 0) or np.var(observable_series) == 0:
                correlation_times[lambda_state_idx] = 1.0
                logger.warning(
                    f"WARNING: No fluctuations detected for lambda state {lambda_state_idx}, setting g=1"  # noqa: E501
                )
            else:
                correlation_times[lambda_state_idx] = calc_statistical_inefficiency(
                    observable_series, fast=fast_analysis
                )

            # Extract indices of statistically independent samples
            # Note: subsample_correlated_data returns indices relative to observable_series
            uncorr_indices_relative = np.array(
                subsample_correlated_data(
                    observable_series, g=correlation_times[lambda_state_idx]
                )
            )
            global_uncorr_indices = window_start + uncorr_indices_relative
            num_uncorr_found = len(global_uncorr_indices)

            # Handle insufficient samples
            if num_uncorr_found < min_uncorr_samples:
                logger.warning(
                    f"State {lambda_state_idx}: only {num_uncorr_found} uncorr samples, "
                    f"using all {window_size} samples from analysis window"
                )
                # Fall back to using all samples in the analysis window
                # basically equal to len(observable_series)
                global_uncorr_indices = window_start + np.arange(window_size)
                num_samples_final = window_size
            else:
                num_samples_final = num_uncorr_found

            num_uncorr_samples_per_state[lambda_state_idx] = num_samples_final
            uncorr_indices_per_state.append(global_uncorr_indices)

            logger.info(
                f"{lambda_state_idx:>6} {window_size:>8} {num_samples_final:>8} "
                f"{correlation_times[lambda_state_idx]:>10.2f} {observable:>12}"
            )

        except Exception as e:
            logger.error(f"Failed to process state {lambda_state_idx}: {e}")
            # Fallback: use all samples
            num_uncorr_samples_per_state[lambda_state_idx] = window_size
            uncorr_indices_per_state.append(window_start + np.arange(window_size))

    # Extract uncorrelated data arrays
    dhdl_uncorr = None
    potential_uncorr = None

    # Always extract dH/dλ data (needed for TI methods)
    dhdl_uncorr = np.zeros(
        (num_lambda_states, num_components, max_window_size), dtype=float
    )
    for lambda_idx in range(num_lambda_states):
        num_samples_to_store = num_uncorr_samples_per_state[lambda_idx]
        global_uncorr_indices = uncorr_indices_per_state[lambda_idx]

        # Store uncorrelated dH/dλ samples for all components
        for component_idx in range(num_components):
            dhdl_uncorr[lambda_idx, component_idx, :num_samples_to_store] = (
                dhdl_timeseries[lambda_idx, component_idx, global_uncorr_indices]
            )

    # Extract potential energy data if available (needed for MBAR/BAR/EXP)
    if potential_energies is not None:
        potential_uncorr = np.zeros(
            (num_lambda_states, num_lambda_states, max_window_size), dtype=np.float64
        )
        for lambda_idx in range(num_lambda_states):
            num_samples_to_store = num_uncorr_samples_per_state[lambda_idx]
            global_uncorr_indices = uncorr_indices_per_state[lambda_idx]
            # Store uncorrelated potential energy samples for all lambda state pairs
            for target_lambda in range(num_lambda_states):
                potential_uncorr[lambda_idx, target_lambda, :num_samples_to_store] = (
                    potential_energies[lambda_idx, target_lambda, global_uncorr_indices]
                )

    # Determine method compatibility
    method_compatibility = []
    if dhdl_uncorr is not None:
        method_compatibility.extend(['TI', 'TI_CUBIC'])
    if potential_uncorr is not None:
        method_compatibility.extend(['MBAR', 'BAR', 'EXP'])

    total_samples = int(num_uncorr_samples_per_state.sum())

    logger.info("\nAutocorrelation Analysis Summary:")
    logger.info(f"  Observable: {observable}")
    logger.info(f"  Total uncorrelated samples: {total_samples}")
    logger.info(f"  Compatible methods: {', '.join(method_compatibility)}")

    return dhdl_uncorr, potential_uncorr, num_uncorr_samples_per_state


def perform_uncorrelating_subsampling_multi_observable(
    dhdl_timeseries: np.ndarray,
    lambda_vectors: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    potential_energies: Optional[np.ndarray] = None,
    observables: list[str] = ['dhdl', 'dhdl_all', 'de'],
    min_uncorr_samples: int = 50,
    fast_analysis: bool = False,
) -> dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]]:
    """
    Convenience function to perform autocorrelation analysis with multiple observables.

    This allows comparing different correlation analysis approaches and choosing
    the most appropriate one for a specific system.

    Parameters:
    -----------
    dhdl_timeseries : np.ndarray
        dH/dλ time series data
    lambda_vectors : np.ndarray
        Lambda parameter values
    start_indices, end_indices : np.ndarray
        Analysis window boundaries
    potential_energies : np.ndarray, optional
        Cross-evaluation energies
    observables : list[str], default ['dhdl', 'dhdl_all', 'de']
        List of observables to analyze
    min_uncorr_samples : int, default 50
        Minimum uncorrelated samples threshold
    fast_analysis : bool, default False
        Use faster algorithms for large datasets

    Returns:
    --------
    dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]]
        Results for each observable type, where each tuple contains:
        (dhdl_uncorrelated, potential_energies_uncorrelated, num_samples_per_state)
    """
    logger.info("=== Multi-Observable Autocorrelation Analysis ===")

    results = {}

    for obs in observables:
        if obs == 'de' and potential_energies is None:
            logger.warning(
                f"Skipping '{obs}' observable: potential_energies not provided"
            )
            continue

        logger.info(f"\nAnalyzing with observable: {obs}")

        try:
            results[obs] = perform_uncorrelating_subsampling(
                dhdl_timeseries=dhdl_timeseries,
                lambda_vectors=lambda_vectors,
                start_indices=start_indices,
                end_indices=end_indices,
                potential_energies=potential_energies,
                observable=obs,
                min_uncorr_samples=min_uncorr_samples,
                fast_analysis=fast_analysis,
            )
        except Exception as e:
            logger.error(f"Failed to analyze observable '{obs}': {e}")
            continue

    # Summary comparison
    logger.info("\n=== Multi-Observable Analysis Summary ===")
    for obs, (dhdl_uncorr, potential_uncorr, num_samples) in results.items():
        total_samples = int(num_samples.sum())
        methods = []
        if dhdl_uncorr is not None:
            methods.extend(['TI', 'TI_CUBIC'])
        if potential_uncorr is not None:
            methods.extend(['MBAR', 'BAR', 'EXP'])
        logger.info(
            f"  {obs}: {total_samples} total samples, " f"methods: {', '.join(methods)}"
        )

    return results
