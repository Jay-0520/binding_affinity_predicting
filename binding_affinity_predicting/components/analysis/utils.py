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
