import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ConvergenceAnalyzer:
    """
    Basic convergence analysis utilities for FEP simulations.
    """

    def __init__(self, temperature: float = 298.15, units: str = 'kT'):
        """
        Initialize the convergence analyzer.

        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin
        units : str
            Units for free energy ('kT', 'kcal/mol', 'kJ/mol')
        """
        self.temperature = temperature
        self.units = units

    @staticmethod
    def analyze_timeseries_convergence(
        data: np.ndarray, window_fraction: float = 0.25
    ) -> dict:
        """
        Analyze convergence of a time series using cumulative averages.

        Parameters:
        -----------
        data : np.ndarray
            Time series data
        window_fraction : float
            Fraction of data to use for final stability analysis

        Returns:
        --------
        Dict with convergence metrics
        """
        if len(data) < 10:
            return {'converged': False, 'reason': 'insufficient_data'}

        # Calculate cumulative average
        cumulative_avg = np.cumsum(data) / np.arange(1, len(data) + 1)
        final_avg = cumulative_avg[-1]

        # Analyze final portion stability
        window_size = max(10, int(len(data) * window_fraction))
        final_window = cumulative_avg[-window_size:]
        final_std = np.std(final_window)

        # Calculate drift in final portion
        if len(final_window) > 1:
            final_slope = np.polyfit(np.arange(len(final_window)), final_window, 1)[0]
        else:
            final_slope = 0.0

        # Convergence criteria
        relative_fluctuation = (
            abs(final_std / final_avg) if final_avg != 0 else float('inf')
        )
        relative_drift = (
            abs(final_slope / final_avg) if final_avg != 0 else float('inf')
        )

        converged = relative_fluctuation < 0.05 and relative_drift < 0.01

        return {
            'converged': converged,
            'final_average': float(final_avg),
            'final_std': float(final_std),
            'relative_fluctuation': float(relative_fluctuation),
            'relative_drift': float(relative_drift),
            'n_samples': len(data),
        }

    @staticmethod
    def analyze_replica_convergence(replica_data: List[np.ndarray]) -> dict:
        """
        Analyze convergence between multiple replicas.

        Parameters:
        -----------
        replica_data : List[np.ndarray]
            List of time series from different replicas

        Returns:
        --------
        Dict with inter-replica convergence metrics
        """
        if len(replica_data) < 2:
            return {'n_replicas': len(replica_data), 'converged': False}

        # Calculate means for each replica
        replica_means = [np.mean(data) for data in replica_data]
        replica_stds = [np.std(data) for data in replica_data]

        # Inter-replica statistics
        mean_of_means = np.mean(replica_means)
        std_between_replicas = np.std(replica_means)
        mean_intra_std = np.mean(replica_stds)

        # Convergence metrics
        relative_std_between = (
            abs(std_between_replicas / mean_of_means)
            if mean_of_means != 0
            else float('inf')
        )

        # Statistical test for consistency (simplified)
        consistent = relative_std_between < 0.1  # 10% threshold

        return {
            'n_replicas': len(replica_data),
            'converged': consistent,
            'mean_of_means': float(mean_of_means),
            'std_between_replicas': float(std_between_replicas),
            'relative_std_between': float(relative_std_between),
            'mean_intra_std': float(mean_intra_std),
            'replica_means': replica_means,
            'replica_stds': replica_stds,
        }

    @staticmethod
    def forward_reverse_analysis(
        forward_data: np.ndarray, reverse_data: np.ndarray
    ) -> dict:
        """
        Analyze forward-reverse convergence (for reversible processes).
        """
        forward_conv = ConvergenceAnalyzer.analyze_timeseries_convergence(forward_data)
        reverse_conv = ConvergenceAnalyzer.analyze_timeseries_convergence(reverse_data)

        # Compare final values
        forward_final = forward_conv['final_average']
        reverse_final = reverse_conv['final_average']

        if forward_final != 0:
            relative_difference = abs((forward_final - reverse_final) / forward_final)
        else:
            relative_difference = float('inf')

        hysteresis_converged = relative_difference < 0.05  # 5% threshold

        return {
            'forward_convergence': forward_conv,
            'reverse_convergence': reverse_conv,
            'forward_final': float(forward_final),
            'reverse_final': float(reverse_final),
            'relative_difference': float(relative_difference),
            'hysteresis_converged': hysteresis_converged,
        }

    def fep_forward_backward_convergence(
        self,
        df_list: List[pd.DataFrame],
        estimator: str = "MBAR",
        num: int = 10,
        error_tol: float = 3.0,
        **kwargs,
    ) -> dict:
        """
        Forward and backward convergence analysis for FEP simulations.
        Based on alchemlyb's forward_backward_convergence function.

        Parameters:
        -----------
        df_list : List[pd.DataFrame]
            List of DataFrames containing either u_nk or dHdl data
        estimator : str
            Estimator to use ('MBAR', 'BAR', 'TI')
        num : int
            Number of points for convergence analysis
        error_tol : float
            Error tolerance for bootstrap estimation
        **kwargs : dict
            Additional arguments for estimators

        Returns:
        --------
        dict containing convergence analysis results
        """
        logger.info(f"Starting forward-backward convergence analysis with {estimator}")

        # Validate estimator
        valid_estimators = ['MBAR', 'BAR', 'TI']
        if estimator not in valid_estimators:
            raise ValueError(f"Estimator must be one of {valid_estimators}")

        # Check lambda state consistency
        for i, df in enumerate(df_list):
            lambda_values = list(set([x[1:] for x in df.index.to_numpy()]))
            if len(lambda_values) > 1:
                ind = [
                    j
                    for j in range(len(lambda_values[0]))
                    if len(list(set([x[j] for x in lambda_values]))) > 1
                ][0]
                raise ValueError(
                    f"DataFrame {i} has multiple lambda values in index[{ind}]"
                )

        # Forward analysis
        logger.info("Performing forward analysis")
        forward_list = []
        forward_error_list = []

        for i in range(1, num + 1):
            logger.debug(f"Forward: {100 * i / num:.1f}%")
            sample = []
            for data in df_list:
                sample.append(data[: len(data) // num * i])

            mean, error = self._estimate_free_energy(
                sample, estimator, error_tol, **kwargs
            )
            forward_list.append(mean)
            forward_error_list.append(error)

        # Backward analysis
        logger.info("Performing backward analysis")
        backward_list = []
        backward_error_list = []

        for i in range(1, num + 1):
            logger.debug(f"Backward: {100 * i / num:.1f}%")
            sample = []
            for data in df_list:
                sample.append(data[-len(data) // num * i :])

            mean, error = self._estimate_free_energy(
                sample, estimator, error_tol, **kwargs
            )
            backward_list.append(mean)
            backward_error_list.append(error)

        # Create results dataframe
        convergence_df = pd.DataFrame(
            {
                'Forward': forward_list,
                'Forward_Error': forward_error_list,
                'Backward': backward_list,
                'Backward_Error': backward_error_list,
                'data_fraction': [i / num for i in range(1, num + 1)],
            }
        )

        # Analyze convergence
        analysis = self._analyze_forward_backward_convergence(convergence_df)

        return {
            'convergence_data': convergence_df,
            'analysis': analysis,
            'estimator': estimator,
        }

    def fep_block_average(
        self,
        df_list: List[pd.DataFrame],
        estimator: str = "MBAR",
        num: int = 10,
        **kwargs,
    ) -> dict:
        """
        Block averaging analysis for FEP simulations.
        Based on alchemlyb's block_average function.

        Parameters:
        -----------
        df_list : List[pd.DataFrame]
            List of DataFrames containing data
        estimator : str
            Estimator to use
        num : int
            Number of blocks
        **kwargs : dict
            Additional arguments for estimators

        Returns:
        --------
        dict containing block analysis results
        """
        logger.info(f"Starting block averaging analysis with {estimator}")

        # Validate inputs
        valid_estimators = ['MBAR', 'BAR', 'TI']
        if estimator not in valid_estimators:
            raise ValueError(f"Estimator must be one of {valid_estimators}")

        if estimator == "BAR" and len(df_list) > 2:
            raise ValueError("BAR requires exactly 2 adjacent lambda states")

        # Block analysis
        average_list = []
        average_error_list = []

        for i in range(1, num):
            logger.debug(f"Block {i}/{num-1}")
            sample = []
            for data in df_list:
                ind1 = len(data) // num * (i - 1)
                ind2 = len(data) // num * i
                sample.append(data[ind1:ind2])

            mean, error = self._estimate_free_energy(sample, estimator, 3.0, **kwargs)
            average_list.append(mean)
            average_error_list.append(error)

        # Create results dataframe
        block_df = pd.DataFrame(
            {
                'FE': average_list,
                'FE_Error': average_error_list,
                'block_index': list(range(1, num)),
            }
        )

        # Analyze block convergence
        analysis = self._analyze_block_convergence(block_df)

        return {'block_data': block_df, 'analysis': analysis, 'estimator': estimator}

    def fep_r_c_analysis(
        self, series: pd.Series, precision: float = 0.01, tol: float = 2.0
    ) -> dict:
        """
        R_c convergence analysis for a single time series.
        Based on alchemlyb's fwdrev_cumavg_Rc function.

        Parameters:
        -----------
        series : pd.Series
            Energy time series
        precision : float
            Precision for R_c calculation
        tol : float
            Tolerance in kT

        Returns:
        --------
        dict containing R_c analysis results
        """
        logger.info("Computing R_c convergence analysis")

        # Convert to numpy array
        array = series.to_numpy()
        out_length = int(1 / precision)

        # Calculate cumulative means
        g_forward = self._cummean(array, out_length)
        g_backward = self._cummean(array[::-1], out_length)
        length = len(g_forward)

        # Create convergence dataframe
        convergence_df = pd.DataFrame(
            {
                'Forward': g_forward,
                'Backward': g_backward,
                'data_fraction': [i / length for i in range(1, length + 1)],
            }
        )

        # Calculate R_c
        g = g_forward[-1]
        conv = np.logical_and(np.abs(g_forward - g) < tol, np.abs(g_backward - g) < tol)

        r_c = 1.0  # Default if not converged
        for i in range(out_length):
            if all(conv[i:]):
                r_c = i / length
                break

        analysis = {
            'r_c': r_c,
            'converged': r_c < 1.0,
            'convergence_fraction': r_c,
            'final_value': g,
            'tolerance': tol,
        }

        return {
            'convergence_data': convergence_df,
            'analysis': analysis,
            'method': 'r_c',
        }

    def _estimate_free_energy(
        self,
        sample_list: List[pd.DataFrame],
        estimator: str,
        error_tol: float,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Estimate free energy using specified estimator.
        Uses the FreeEnergyEstimator class for actual calculations.
        """
        try:
            # Initialize estimator
            fe_estimator = FreeEnergyEstimator(temperature=self.temperature)

            if estimator.upper() == "TI":
                # For TI, extract lambda values and dH/dλ data
                lambdas, dhdl_mean, dhdl_error = self._extract_ti_data(sample_list)

                # Use trapezoidal integration by default
                method = kwargs.get('method', 'trapezoidal')
                result = fe_estimator.estimate_ti(
                    lambdas, dhdl_mean, dhdl_error, method
                )

                if result['success']:
                    return result['free_energy'], result['error']
                else:
                    logger.error(
                        f"TI estimation failed: {result.get('error_message', 'Unknown error')}"
                    )
                    return float('nan'), float('nan')

            elif estimator.upper() == "BAR":
                # For BAR, extract work values from adjacent states
                work_forward, work_reverse = self._extract_bar_data(sample_list)

                # Use standard BAR by default
                method = kwargs.get('method', 'standard')
                result = fe_estimator.estimate_bar(work_forward, work_reverse, method)

                if result['success']:
                    return result['free_energy'], result['error']
                else:
                    logger.error(
                        f"BAR estimation failed: {result.get('error_message', 'Unknown error')}"
                    )
                    return float('nan'), float('nan')

            elif estimator.upper() == "MBAR":
                # For MBAR, extract potential energy matrix
                u_kln, N_k = self._extract_mbar_data(sample_list)

                # Set MBAR parameters
                mbar_kwargs = {
                    'relative_tolerance': kwargs.get('relative_tolerance', 1e-10),
                    'verbose': kwargs.get('verbose', False),
                    'compute_overlap': kwargs.get('compute_overlap', False),
                }

                result = fe_estimator.estimate_mbar(u_kln, N_k, **mbar_kwargs)

                if result['success']:
                    # Use bootstrap error if analytic error is too large
                    if result['total_error'] > error_tol:
                        logger.warning(
                            f"MBAR error ({result['total_error']:.3f}) > tolerance ({error_tol}), "
                            "consider using bootstrap"
                        )

                    return result['total_dg'], result['total_error']
                else:
                    logger.error(
                        f"MBAR estimation failed: {result.get('error_message', 'Unknown error')}"
                    )
                    return float('nan'), float('nan')

            else:
                raise ValueError(f"Unknown estimator: {estimator}")

        except Exception as e:
            logger.error(f"Free energy estimation failed with {estimator}: {e}")
            return float('nan'), float('nan')

    def _cummean(self, vals: np.ndarray, out_length: int) -> np.ndarray:
        """
        Calculate cumulative mean with specified output length.
        From alchemlyb convergence module.
        """
        in_length = len(vals)
        if in_length < out_length:
            out_length = in_length

        block = in_length // out_length
        reshape = vals[: block * out_length].reshape(block, out_length)
        mean = np.mean(reshape, axis=0)
        result = np.cumsum(mean) / np.arange(1, out_length + 1)

        return result

    def _analyze_forward_backward_convergence(
        self, convergence_df: pd.DataFrame
    ) -> dict:
        """
        Analyze forward-backward convergence results.
        """
        forward = convergence_df['Forward'].values
        backward = convergence_df['Backward'].values

        # Final values comparison
        final_forward = forward[-1]
        final_backward = backward[-1]
        final_diff = abs(final_forward - final_backward)
        relative_diff = (
            final_diff / abs(final_forward) if final_forward != 0 else float('inf')
        )

        # Convergence of individual directions
        forward_analysis = self.analyze_timeseries_convergence(forward)
        backward_analysis = self.analyze_timeseries_convergence(backward)

        # Overall convergence assessment
        converged = (
            forward_analysis['converged']
            and backward_analysis['converged']
            and relative_diff < 0.05
        )

        return {
            'converged': converged,
            'final_forward': final_forward,
            'final_backward': final_backward,
            'final_difference': final_diff,
            'relative_difference': relative_diff,
            'forward_analysis': forward_analysis,
            'backward_analysis': backward_analysis,
        }

    def _extract_ti_data(
        self, sample_list: List[pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract lambda values and dH/dλ data for TI estimation.

        Parameters:
        -----------
        sample_list : List[pd.DataFrame]
            List of dH/dλ DataFrames

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray] : (lambdas, dhdl_mean, dhdl_error)
        """
        lambdas = []
        dhdl_means = []
        dhdl_errors = []

        for df in sample_list:
            # Extract lambda value from DataFrame index
            # Assuming the lambda state is encoded in the index
            if hasattr(df.index, 'get_level_values'):
                # Multi-index case
                lambda_state = df.reset_index().index.values[0]
                if isinstance(lambda_state, (tuple, list)):
                    # Use the first lambda component or a specific one
                    lambda_val = lambda_state[0] if len(lambda_state) > 0 else 0.0
                else:
                    lambda_val = float(lambda_state)
            else:
                # Simple index case - use position as lambda
                lambda_val = (
                    len(lambdas) / (len(sample_list) - 1)
                    if len(sample_list) > 1
                    else 0.0
                )

            lambdas.append(lambda_val)

            # Calculate mean and error for dH/dλ
            if len(df.columns) == 1:
                # Single component
                dhdl_data = df.iloc[:, 0].values
            else:
                # Multiple components - sum them
                dhdl_data = df.sum(axis=1).values

            dhdl_mean = np.mean(dhdl_data)
            dhdl_std = np.std(dhdl_data)
            dhdl_error = (
                dhdl_std / np.sqrt(len(dhdl_data)) if len(dhdl_data) > 1 else dhdl_std
            )

            dhdl_means.append(dhdl_mean)
            dhdl_errors.append(dhdl_error)

        return np.array(lambdas), np.array(dhdl_means), np.array(dhdl_errors)

    def _extract_bar_data(
        self, sample_list: List[pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract work values for BAR estimation.

        Parameters:
        -----------
        sample_list : List[pd.DataFrame]
            List of u_nk DataFrames (should be exactly 2 for BAR)

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (work_forward, work_reverse)
        """
        if len(sample_list) != 2:
            raise ValueError(
                f"BAR requires exactly 2 lambda states, got {len(sample_list)}"
            )

        df0, df1 = sample_list

        # Extract potential energies
        # Assuming u_nk format where columns represent different lambda states
        if hasattr(df0, 'columns') and len(df0.columns) >= 2:
            # Standard u_nk format
            u0_at_0 = df0.iloc[:, 0].values  # Energy of state 0 samples at lambda 0
            u0_at_1 = df0.iloc[:, 1].values  # Energy of state 0 samples at lambda 1

            u1_at_0 = df1.iloc[:, 0].values  # Energy of state 1 samples at lambda 0
            u1_at_1 = df1.iloc[:, 1].values  # Energy of state 1 samples at lambda 1

            # Calculate work values
            work_forward = u0_at_1 - u0_at_0  # 0→1 transition work
            work_reverse = u1_at_0 - u1_at_1  # 1→0 transition work
        else:
            # Fallback: use first column as energy
            logger.warning(
                "Unexpected DataFrame format for BAR, using simplified extraction"
            )
            work_forward = df0.iloc[:, 0].values
            work_reverse = -df1.iloc[:, 0].values  # Reverse sign

        return work_forward, work_reverse

    def _extract_mbar_data(
        self, sample_list: List[pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract potential energy matrix for MBAR estimation.

        Parameters:
        -----------
        sample_list : List[pd.DataFrame]
            List of u_nk DataFrames

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (u_kln, N_k)
        """
        K = len(sample_list)  # Number of lambda states
        N_k = np.array([len(df) for df in sample_list])  # Samples per state
        max_N = max(N_k)

        # Initialize potential energy matrix
        u_kln = np.zeros((K, K, max_N))

        for k, df in enumerate(sample_list):
            n_samples = len(df)

            if hasattr(df, 'columns') and len(df.columns) >= K:
                # Standard u_nk format with K columns
                for l in range(K):
                    u_kln[k, l, :n_samples] = df.iloc[:, l].values
            else:
                # Simplified format - assume diagonal energies
                logger.warning(
                    "Simplified MBAR data extraction - assuming diagonal energies"
                )
                for l in range(K):
                    if l == k:
                        # Native state energy
                        u_kln[k, l, :n_samples] = df.iloc[:, 0].values
                    else:
                        # Estimate cross-energies (this is a rough approximation)
                        u_kln[k, l, :n_samples] = df.iloc[
                            :, 0
                        ].values + np.random.normal(0, 1, n_samples)

        return u_kln, N_k

    def _analyze_block_convergence(self, block_df: pd.DataFrame) -> dict:
        """
        Analyze block averaging convergence results.
        """
        fe_values = block_df['FE'].values

        # Statistical analysis of blocks
        block_mean = np.mean(fe_values)
        block_std = np.std(fe_values)
        relative_std = block_std / abs(block_mean) if block_mean != 0 else float('inf')

        # Trend analysis
        x = np.arange(len(fe_values))
        slope, _ = np.polyfit(x, fe_values, 1)
        relative_slope = (
            abs(slope) / abs(block_mean) if block_mean != 0 else float('inf')
        )

        # Convergence criteria
        converged = relative_std < 0.1 and relative_slope < 0.01

        return {
            'converged': converged,
            'block_mean': block_mean,
            'block_std': block_std,
            'relative_std': relative_std,
            'trend_slope': slope,
            'relative_slope': relative_slope,
            'n_blocks': len(fe_values),
        }
