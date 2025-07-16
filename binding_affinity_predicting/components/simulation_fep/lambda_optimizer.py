"""
Clean Multi-Stage Lambda Optimizer for GROMACS FEP simulations.

This optimizer recognizes that GROMACS FEP consists of three distinct stages:
1. Restrained stage: bonded-lambdas vary (0.0 → 1.0), coul=0.0, vdw=0.0
2. Discharging stage: coul-lambdas vary (0.0 → 1.0), bonded=1.0, vdw=0.0
3. Vanishing stage: vdw-lambdas vary (0.0 → 1.0), bonded=1.0, coul=1.0

Each stage is optimized independently for optimal lambda spacing.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import Field, BaseModel

from binding_affinity_predicting.components.analysis.autocorrelation import (
    _statistical_inefficiency_chodera as _get_statistical_inefficiency,
)
from binding_affinity_predicting.components.data.enums import LegType
from binding_affinity_predicting.components.simulation_fep.gromacs_orchestration import (
    Calculation,
    LambdaWindow,
    Leg,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO need to consolidate these Enum and configs
# ============================================================================
# Enums and Configuration Classes for Lambda Optimization
# ============================================================================
class FepStage(Enum):
    """GROMACS FEP simulation stages"""

    RESTRAINED = "restrained"  # bonded-lambdas vary
    DISCHARGING = "discharging"  # coul-lambdas vary
    VANISHING = "vanishing"  # vdw-lambdas vary


class SpacingMethod(Enum):
    """Method for lambda spacing optimization"""

    TARGET_ERROR = "target_error"
    FIXED_WINDOWS = "fixed_windows"
    USER_PROVIDED = "user_provided"


class StageConfig(BaseModel):
    """Configuration for optimizing a specific FEP stage"""

    method: SpacingMethod = SpacingMethod.TARGET_ERROR
    target_error: float = 1.0
    n_windows: Optional[int] = None
    user_lambdas: Optional[list[float]] = None
    error_type: str = "root_var"  # "root_var" or "sem"
    sem_origin: str = "inter"  # "inter" or "intra"
    smoothen_errors: bool = True
    round_lambdas: bool = True


class OptimizationConfig(BaseModel):
    """Overall configuration for multi-stage optimization"""

    restrained: StageConfig = Field(default_factory=lambda: StageConfig())
    discharging: StageConfig = Field(default_factory=lambda: StageConfig())
    vanishing: StageConfig = Field(default_factory=lambda: StageConfig())

    def get_stage_config(self, stage: FepStage) -> StageConfig:
        """Get configuration for a specific stage"""
        return getattr(self, stage.value)


# ============================================================================
# Data Classes for Results
# ============================================================================
@dataclass
class WindowAnalysis:
    """Analysis results for a single lambda window"""

    lambda_state: int
    coul_lambda: float
    vdw_lambda: float
    bonded_lambda: float
    stage: FepStage
    stage_lambda: float  # The varying lambda for this stage

    # Gradient statistics
    mean_gradient: float
    root_variance: float  # This is related to "difficulty"
    sem_inter: float
    sem_intra: float
    n_data_points: int
    simulation_time: float
    statistical_inefficiency: float


@dataclass
class StageResult:
    """Optimization results for a single FEP stage"""

    stage: FepStage
    original_lambdas: list[float]
    optimal_lambdas: list[float]
    window_analyses: list[WindowAnalysis]
    improvement_factor: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class OptimizationResult:
    """Complete multi-stage optimization results"""

    leg_type: str
    stage_results: dict[FepStage, StageResult]

    # Combined lambda vectors for GROMACS
    original_bonded: list[float]
    original_coul: list[float]
    original_vdw: list[float]

    optimal_bonded: list[float]
    optimal_coul: list[float]
    optimal_vdw: list[float]

    overall_improvement: float
    success: bool


# ============================================================================
# Core Optimizer Classes
# ============================================================================
class GradientAnalyzer:
    """Handles reading and analyzing gradient data from GROMACS windows"""

    def analyze_window(
        self,
        window: LambdaWindow,
        run_nos: Optional[list[int]] = None,
        equilibrated: bool = False,
    ) -> WindowAnalysis:
        """Analyze gradients from a single lambda window"""

        try:
            lambda_state = window.lam_state
        except AttributeError:
            logger.warning(f"Lambda state not found in window {LambdaWindow.__str__}")
            raise ValueError(f"Lambda state not found in window {LambdaWindow.__str__}")

        # Extract original lambda values for a given LambdaWindow
        # lambdas = (bonded, coul, vdw) and cannot be None
        lambdas = self._extract_lambda_values(window)

        # Determine FEP stage based on lambda values
        # and return the original varying lambda
        coul_lambda, vdw_lambda, bonded_lambda = lambdas
        stage, stage_lambda = self._determine_stage(
            coul_lambda, vdw_lambda, bonded_lambda
        )

        # Collect gradient data from all runs
        gradient_data = self._collect_gradients(window, run_nos, equilibrated)

        # Calculate statistics
        stats = self._calculate_statistics(gradient_data)

        return WindowAnalysis(
            lambda_state=lambda_state,
            coul_lambda=coul_lambda,
            vdw_lambda=vdw_lambda,
            bonded_lambda=bonded_lambda,
            stage=stage,
            stage_lambda=stage_lambda,
            **stats,
        )

    def _extract_lambda_values(
        self, window: LambdaWindow
    ) -> tuple[float, float, float]:
        """Extract (coul, vdw, bonded) lambda values from window"""
        try:
            sim_params = window.sim_params
            try:
                lam_state = window.lam_state
            except AttributeError:
                logger.warning("Lambda state not found in window")
                raise ValueError("Lambda state not found in window")

            coul = (
                float(sim_params.get('coul_list', [0.0])[lam_state])
                if lam_state < len(sim_params.get('coul_list', []))
                else 0.0
            )
            vdw = (
                float(sim_params.get('vdw_list', [0.0])[lam_state])
                if lam_state < len(sim_params.get('vdw_list', []))
                else 0.0
            )
            bonded = (
                float(sim_params.get('bonded_list', [0.0])[lam_state])
                if lam_state < len(sim_params.get('bonded_list', []))
                else 0.0
            )

            return coul, vdw, bonded
        except Exception as e:
            logger.warning(f"Failed to extract lambda values: {e}")
            raise ValueError(
                f"Failed to extract lambda values from window {LambdaWindow}"
            ) from e

    def _determine_stage(
        self, coul: float, vdw: float, bonded: float
    ) -> tuple[FepStage, float]:
        """
        TODO: may need to be more robust in the future.
        Determine FEP stage and return the varying lambda

        This is due to specific mdp settings in GROMACS FEP simulations.
        so stage is defined by the varying lambda:
        - RESTRAINED: bonded-lambdas vary (0.0 → 1.0)
        - DISCHARGING: coul-lambdas vary (0.0 → 1.0)
        - VANISHING: vdw-lambdas vary (0.0 → 1.0)

        Returns:
            FepStage: The determined FEP stage
            float: The varying lambda for this stage
        """
        tol = 1e-6  # 0.00001
        if abs(coul) < tol and abs(vdw) < tol:
            return FepStage.RESTRAINED, bonded
        elif abs(bonded - 1.0) < tol and abs(vdw) < tol:
            return FepStage.DISCHARGING, coul
        elif abs(bonded - 1.0) < tol and abs(coul - 1.0) < tol:
            return FepStage.VANISHING, vdw
        else:
            # Fallback heuristic
            if bonded < 0.99:
                return FepStage.RESTRAINED, bonded
            elif coul < 0.99:
                return FepStage.DISCHARGING, coul
            else:
                return FepStage.VANISHING, vdw

    def _collect_gradients(
        self, window: LambdaWindow, run_nos: Optional[list[int]], equilibrated: bool
    ) -> dict:
        """Collect gradient data from all runs"""
        if run_nos is None:
            run_nos = list(range(1, window.ensemble_size + 1))

        all_gradients = []
        run_means = []
        total_time = 0.0

        for run_no in run_nos:
            gradients, sim_time = self._read_run_gradients(window, run_no, equilibrated)
            # gradients cannot be None
            if len(gradients) > 0:
                all_gradients.append(gradients)
                run_means.append(np.mean(gradients))
                total_time += sim_time

        # In theory, all_gradients should not be empty
        if not all_gradients:
            raise ValueError(
                f"No valid gradients found for window {window} with runs {run_nos}"
            )

        return {
            'all_gradients': all_gradients,
            'run_means': run_means,
            'total_time': total_time,
        }

    def _read_run_gradients(
        self, window: LambdaWindow, run_no: int, equilibrated: bool
    ) -> tuple[np.ndarray, float]:
        """Read gradients from a single run"""
        try:
            run_dir = Path(window.output_dir) / f"run_{run_no}"
            xvg_file = run_dir / f"lambda_{window.lam_state}_run_{run_no}.xvg"

            if not xvg_file.exists():
                raise FileNotFoundError(
                    f"Gradient file not found for run {run_no}: {xvg_file}"
                )

            times, gradients = self._parse_xvg_file(xvg_file)

            # only consider gradients after equilibration
            if equilibrated and hasattr(window, '_equil_time'):
                equil_idx = np.searchsorted(times, window._equil_time)
                times = times[equil_idx:]
                gradients = gradients[equil_idx:]

            return gradients, times[-1] if len(times) > 0 else 0.0

        except Exception as e:
            logger.warning(f"Failed to read run {run_no}: {e}")
            raise ValueError(
                f"Failed to read gradients for run {run_no} in window {window}"
            ) from e

    def _parse_xvg_file(self, xvg_file: Path) -> tuple[np.ndarray, np.ndarray]:
        """Parse GROMACS .xvg file to extract total gradients"""
        times, gradients = [], []

        with open(xvg_file, 'r') as f:
            for line in f:
                if line.startswith(('#', '@', '&')):
                    continue

                parts = line.strip().split()
                if len(parts) >= 4:
                    time_ps = float(parts[0])
                    dhdl_total = (
                        float(parts[1]) + float(parts[2]) + float(parts[3])
                    ) / 4.184  # kJ/mol to kcal/mol

                    times.append(time_ps / 1000.0)  # ps to ns
                    gradients.append(dhdl_total)

        return np.array(times), np.array(gradients)

    def _calculate_statistics(self, gradient_data: dict) -> dict:
        """Calculate gradient statistics with autocorrelation correction"""
        all_gradients = gradient_data['all_gradients']
        run_means = gradient_data['run_means']

        # Calculate statistical inefficiencies per run
        stat_ineffs = []
        subsampled_gradients = []

        for grads in all_gradients:
            stat_ineff = _get_statistical_inefficiency(grads)
            stat_ineffs.append(stat_ineff)

            # Subsample to get independent data
            subsampled = grads[:: max(1, int(stat_ineff))]
            subsampled_gradients.append(subsampled)

        # Use subsampled data for statistics
        combined_subsampled = np.concatenate(subsampled_gradients)
        mean_gradient = np.mean(combined_subsampled)
        root_variance = np.sqrt(np.var(combined_subsampled))

        # Corrected SEMs
        mean_stat_ineff = np.mean(stat_ineffs)
        n_effective = len(combined_subsampled)  # Already subsampled

        # Inter-run SEM (corrected)
        sem_inter = np.std(run_means) / np.sqrt(len(run_means))

        # Intra-run SEM (corrected)
        sem_intra = root_variance / np.sqrt(n_effective)

        return {
            'mean_gradient': mean_gradient,
            'root_variance': root_variance,
            'sem_inter': sem_inter,
            'sem_intra': sem_intra,
            'n_data_points': n_effective,  # Effective sample size
            'simulation_time': gradient_data['total_time'],
            'statistical_inefficiency': mean_stat_ineff,
        }


class StageOptimizer:
    """Optimizes lambda spacing for a single FEP stage"""

    def optimize_stage(
        self, stage: FepStage, analyses: list[WindowAnalysis], config: StageConfig
    ) -> StageResult:
        """Optimize lambda spacing for a single stage"""

        if not analyses:
            return StageResult(
                stage=stage,
                original_lambdas=[],
                optimal_lambdas=[],
                window_analyses=[],
                improvement_factor=1.0,
                success=False,
                error_message="No analysis data",
            )

        # Sort by stage lambda
        analyses.sort(key=lambda x: x.stage_lambda)
        original_lambdas = [a.stage_lambda for a in analyses]

        try:
            # Calculate optimal lambdas
            optimal_lambdas = self._calculate_optimal_lambdas(analyses, config)

            # Calculate improvement
            improvement = self._calculate_improvement(
                original_lambdas, analyses, config
            )

            return StageResult(
                stage=stage,
                original_lambdas=original_lambdas,
                optimal_lambdas=optimal_lambdas,
                window_analyses=analyses,
                improvement_factor=improvement,
                success=True,
            )

        except Exception as e:
            return StageResult(
                stage=stage,
                original_lambdas=original_lambdas,
                optimal_lambdas=original_lambdas,
                window_analyses=analyses,
                improvement_factor=1.0,
                success=False,
                error_message=str(e),
            )

    def _calculate_optimal_lambdas(
        self, analyses: list[WindowAnalysis], config: StageConfig
    ) -> list[float]:
        """Calculate optimal lambda spacing"""

        if config.method == SpacingMethod.USER_PROVIDED:
            if config.user_lambdas is None:
                raise ValueError("user_lambdas required for USER_PROVIDED method")
            return config.user_lambdas

        # Extract errors based on config
        errors = self._get_errors(analyses, config)
        lambdas = [a.stage_lambda for a in analyses]

        if config.smoothen_errors:
            errors = self._smooth_errors(errors)

        if config.method == SpacingMethod.TARGET_ERROR:
            return self._optimize_for_target_error(
                lambdas, errors, config.target_error, config.round_lambdas
            )
        elif config.method == SpacingMethod.FIXED_WINDOWS:
            if config.n_windows is None:
                raise ValueError("n_windows required for FIXED_WINDOWS method")
            return self._optimize_for_fixed_windows(
                lambdas, errors, config.n_windows, config.round_lambdas
            )
        else:
            raise ValueError(f"Unknown method: {config.method}")

    def _get_errors(
        self, analyses: list[WindowAnalysis], config: StageConfig
    ) -> list[float]:
        """Extract error values based on configuration"""
        if config.error_type == "root_var":
            return [a.root_variance for a in analyses]
        elif config.error_type == "sem":
            if config.sem_origin == "inter":
                return [a.sem_inter for a in analyses]
            else:
                return [a.sem_intra for a in analyses]
        else:
            raise ValueError(f"Unknown error type: {config.error_type}")

    def _smooth_errors(self, errors: list[float]) -> list[float]:
        """Simple error smoothing"""
        if len(errors) <= 2:
            return errors

        smoothed = []
        for i in range(len(errors)):
            if i == 0:
                smoothed.append((errors[i] + errors[i + 1]) / 2)
            elif i == len(errors) - 1:
                smoothed.append((errors[i - 1] + errors[i]) / 2)
            else:
                smoothed.append((errors[i - 1] + errors[i] + errors[i + 1]) / 3)

        return smoothed

    def _optimize_for_target_error(
        self,
        lambdas: list[float],
        errors: list[float],
        target_error: float,
        round_lambdas: bool,
    ) -> list[float]:
        """Optimize for target error per interval using trapezoidal rule

        This is exactly the same as the original implementation here:
        "a3fe/a3fe/analyse/process_grads.py
        GradientData.calculate_optimal_lam_vals(delta_er=target_error)"
        """

        # Step 1: Calculate integrated errors using trapezoidal rule
        integrated_errors = []
        for i in range(len(lambdas)):
            # Trapezoidal integration from start to point i
            integrated_errors.append(np.trapz(errors[: i + 1], lambdas[: i + 1]))

        # Step 2: Calculate number of lambda values (not intervals)
        total_error = integrated_errors[-1]
        n_lam_vals = round(total_error / target_error) + 1
        n_lam_vals = max(n_lam_vals, 2)  # Ensure at least 2 points

        # Step 3: Create equal error intervals
        requested_sem_vals = np.linspace(0, total_error, n_lam_vals)

        # Step 4: Map back to lambda values using interpolation
        optimal = []
        for requested_sem in requested_sem_vals:
            optimal_lam_val = np.interp(requested_sem, integrated_errors, lambdas)
            if round_lambdas:
                optimal_lam_val = round(optimal_lam_val, 3)
            optimal.append(optimal_lam_val)

        return optimal

    # TODO: need to consolidate this with _optimize_for_target_error()
    def _optimize_for_fixed_windows(
        self,
        lambdas: list[float],
        errors: list[float],
        n_windows: int,
        round_lambdas: bool,
    ) -> list[float]:
        """
        Optimize for fixed number of windows

        This is exactly the same as the original implementation here:
        "a3fe/a3fe/analyse/process_grads.py
        GradientData.calculate_optimal_lam_vals(n_lam_vals=n_windows)"
        """

        integrated_errors = [
            np.trapz(errors[: i + 1], lambdas[: i + 1]) for i in range(len(lambdas))
        ]
        total_error = integrated_errors[-1]
        n_lam_vals = max(2, n_windows)  # Use provided number directly

        requested_sem_vals = np.linspace(0, total_error, n_lam_vals)
        optimal = []
        for requested_sem in requested_sem_vals:
            optimal_lam_val = np.interp(requested_sem, integrated_errors, lambdas)
            if round_lambdas:
                optimal_lam_val = round(optimal_lam_val, 3)
            optimal.append(optimal_lam_val)
        return optimal

    # TODO: something is wrong here?
    def _calculate_improvement(
        self,
        original: list[float],
        analyses: list[WindowAnalysis],
        config: StageConfig,
        optimal: Optional[list[float]] = None,
    ) -> float:
        """
        Calculate the theoretical improvement factor in the standard deviation (error
        type = "root_var") or standard error of the mean (error type = "sem") of the
        free energy change between the first and last lambda windows, if the windows
        were to be spaced optimally. The improvement factor is defined as the ratio
        of the standard deviation or standard error of the mean of the free energy
        change with the optimal spacing to that with the initial spacing (with equal
        number of lambda windows).

        See: Lundborg, Magnus, Jack Lidmar, and Berk Hess. "On the Path to Optimal Alchemistry."
        The Protein Journal (2023): 1-13.

        This is exactly the same as the original implementation here:
        "a3fe/a3fe/analyse/process_grads.py
        GradientData.get_predicted_improvement_factor()"
        """
        errors = self._get_errors(analyses, config)
        lambdas = [a.stage_lambda for a in analyses]

        # Calculate optimal variance (thermodynamic length squared)
        v_opt = (np.trapz(errors, lambdas) ** 2) / len(original)

        # Assign boundaries for each original lambda value
        lam_boundaries = {}
        for i, lam_val in enumerate(original):
            if i == 0:
                lam_boundaries[lam_val] = [
                    0,
                    lam_val + 0.5 * (original[i + 1] - lam_val),
                ]
            elif i == len(original) - 1:
                lam_boundaries[lam_val] = [
                    lam_val - 0.5 * (lam_val - original[i - 1]),
                    1,
                ]
            else:
                lam_boundaries[lam_val] = [
                    lam_val - 0.5 * (lam_val - original[i - 1]),
                    lam_val + 0.5 * (original[i + 1] - lam_val),
                ]

        # Calculate integrated errors
        integrated_errors = []
        for i in range(len(analyses)):
            integrated_errors.append(np.trapz(errors[: i + 1], lambdas[: i + 1]))

        # Calculate weighted errors for each original lambda
        weighted_initial_errors: list[float] = []
        for lam_val in original:
            boundaries = lam_boundaries[lam_val]
            initial_error = np.interp(boundaries[0], lambdas, integrated_errors)
            final_error = np.interp(boundaries[1], lambdas, integrated_errors)
            weighted_initial_errors.append(final_error - initial_error)

        weighted_initial_errors_array: np.ndarray = np.array(
            weighted_initial_errors, dtype=float
        )
        v_initial: float = float(np.sum(weighted_initial_errors_array**2))

        # Return improvement factor
        return np.sqrt(v_opt / v_initial) if v_initial > 0 else 1.0


class MultiStageOptimizer:
    """Main optimizer that coordinates all stages"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.gradient_analyzer = GradientAnalyzer()
        self.stage_optimizer = StageOptimizer()

    def optimize(
        self,
        lambda_windows: list[LambdaWindow],
        leg_type: LegType,
        run_nos: Optional[list[int]] = None,
        equilibrated: bool = False,
    ) -> OptimizationResult:
        """Optimize lambda spacing for all stages"""

        logger.info(f"Starting multi-stage optimization for {leg_type.name} leg...")

        try:
            # Analyze all windows
            analyses = self._analyze_windows(lambda_windows, run_nos, equilibrated)
            if not analyses:
                return self._create_failed_result(leg_type, "No analysis data")

            # Group by stage
            stage_groups = self._group_by_stage(analyses)

            # Optimize each stage
            stage_results = {}
            for stage, stage_analyses in stage_groups.items():
                if stage_analyses:
                    config = self.config.get_stage_config(stage)
                    result = self.stage_optimizer.optimize_stage(
                        stage, stage_analyses, config
                    )
                    stage_results[stage] = result

            # Combine results
            return self._combine_results(stage_results, leg_type)

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self._create_failed_result(leg_type, str(e))

    def _analyze_windows(
        self,
        windows: list[LambdaWindow],
        run_nos: Optional[list[int]],
        equilibrated: bool,
    ) -> list[WindowAnalysis]:
        """Analyze all lambda windows"""
        analyses = []

        for window in windows:
            analysis = self.gradient_analyzer.analyze_window(
                window, run_nos, equilibrated
            )
            if analysis:
                analyses.append(analysis)
                logger.info(
                    f"Analyzed λ{analysis.lambda_state}: {analysis.stage.value} "
                    f"stage_λ={analysis.stage_lambda:.3f} σ={analysis.root_variance:.2f}"
                )

        return analyses

    def _group_by_stage(
        self, analyses: list[WindowAnalysis]
    ) -> dict[FepStage, list[WindowAnalysis]]:
        """Group analyses by FEP stage"""
        groups: dict[FepStage, list[WindowAnalysis]] = {stage: [] for stage in FepStage}

        for analysis in analyses:
            groups[analysis.stage].append(analysis)

        for stage, group in groups.items():
            if group:
                logger.info(f"{stage.value} stage: {len(group)} windows")

        return groups

    def _combine_results(
        self, stage_results: dict[FepStage, StageResult], leg_type: LegType
    ) -> OptimizationResult:
        """Combine stage results into final result"""

        # Initialize lambda vectors
        orig_bonded, orig_coul, orig_vdw = [], [], []
        opt_bonded, opt_coul, opt_vdw = [], [], []

        # Process stages in order
        for stage in [FepStage.RESTRAINED, FepStage.DISCHARGING, FepStage.VANISHING]:
            if stage not in stage_results:
                continue

            result = stage_results[stage]
            n_orig = len(result.original_lambdas)
            n_opt = len(result.optimal_lambdas)

            if stage == FepStage.RESTRAINED:
                orig_bonded.extend(result.original_lambdas)
                orig_coul.extend([0.0] * n_orig)
                orig_vdw.extend([0.0] * n_orig)

                opt_bonded.extend(result.optimal_lambdas)
                opt_coul.extend([0.0] * n_opt)
                opt_vdw.extend([0.0] * n_opt)

            elif stage == FepStage.DISCHARGING:
                orig_bonded.extend([1.0] * n_orig)
                orig_coul.extend(result.original_lambdas)
                orig_vdw.extend([0.0] * n_orig)

                opt_bonded.extend([1.0] * n_opt)
                opt_coul.extend(result.optimal_lambdas)
                opt_vdw.extend([0.0] * n_opt)

            elif stage == FepStage.VANISHING:
                orig_bonded.extend([1.0] * n_orig)
                orig_coul.extend([1.0] * n_orig)
                orig_vdw.extend(result.original_lambdas)

                opt_bonded.extend([1.0] * n_opt)
                opt_coul.extend([1.0] * n_opt)
                opt_vdw.extend(result.optimal_lambdas)

        # Calculate overall improvement
        improvements = [
            r.improvement_factor for r in stage_results.values() if r.success
        ]
        overall_improvement = np.mean(improvements) if improvements else 1.0
        success = all(r.success for r in stage_results.values())

        optimization_result = OptimizationResult(
            leg_type=leg_type.name,
            stage_results=stage_results,
            original_bonded=orig_bonded,
            original_coul=orig_coul,
            original_vdw=orig_vdw,
            optimal_bonded=opt_bonded,
            optimal_coul=opt_coul,
            optimal_vdw=opt_vdw,
            overall_improvement=overall_improvement,
            success=success,
        )

        self._log_results(optimization_result)
        return optimization_result

    def _log_results(self, result: OptimizationResult) -> None:
        """Log optimization results"""
        logger.info("=" * 60)
        logger.info("MULTI-STAGE OPTIMIZATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Leg: {result.leg_type}")
        logger.info(f"Success: {result.success}")
        logger.info(f"Overall improvement: {result.overall_improvement:.3f}")

        for stage, stage_result in result.stage_results.items():
            logger.info(f"\n{stage.value.upper()}:")
            if stage_result.success:
                logger.info(
                    f"  {len(stage_result.original_lambdas)} → {len(stage_result.optimal_lambdas)} windows"  # noqa: E501
                )
                logger.info(f"  Improvement: {stage_result.improvement_factor:.3f}")
                logger.info(f"  Lambdas: {stage_result.optimal_lambdas}")
            else:
                logger.info(f"  ❌ FAILED: {stage_result.error_message}")

        logger.info("\nCOMBINED VECTORS:")
        logger.info(f"bonded-lambdas: {result.optimal_bonded}")
        logger.info(f"coul-lambdas:   {result.optimal_coul}")
        logger.info(f"vdw-lambdas:    {result.optimal_vdw}")
        logger.info("=" * 60)

    def _create_failed_result(
        self, leg_type: LegType, error_msg: str
    ) -> OptimizationResult:
        """Create failed result"""
        return OptimizationResult(
            leg_type=leg_type.name,
            stage_results={},
            original_bonded=[],
            original_coul=[],
            original_vdw=[],
            optimal_bonded=[],
            optimal_coul=[],
            optimal_vdw=[],
            overall_improvement=1.0,
            success=False,
        )


class LambdaOptimizationManager:
    """High-level manager for lambda optimization"""

    def __init__(self, config: OptimizationConfig):
        self.optimizer = MultiStageOptimizer(config)

    def optimize_calculation(
        self,
        calculation: Calculation,
        run_nos: Optional[list[int]] = None,
        equilibrated: bool = False,
        apply_results: bool = False,
    ) -> dict[str, OptimizationResult]:
        """Optimize entire calculation"""

        results = {}

        for leg in calculation.legs:
            logger.info(f"Optimizing {leg.leg_type.name} leg...")

            result = self.optimizer.optimize(
                lambda_windows=leg._sub_sim_runners,
                leg_type=leg.leg_type,
                run_nos=run_nos,
                equilibrated=equilibrated,
            )

            results[leg.leg_type.name] = result

            if apply_results and result.success:
                self._apply_to_leg(leg, result)

        return results

    def _apply_to_leg(
        self, leg: Leg, result: OptimizationResult, backup_original: bool = True
    ) -> None:
        """Apply optimization results to a leg with proper directory setup"""

        logger.info(f"Applying optimization results to {leg.leg_type.name} leg...")
        logger.info(f"Original windows: {len(leg.lambda_windows)}")
        logger.info(f"Optimized windows: {len(result.optimal_bonded)}")

        try:
            # Step 1: Backup original directories if requested
            if backup_original:
                leg.create_backup()

            # Step 2: Update lambda vectors in sim_config
            leg.sim_config.bonded_lambdas[leg.leg_type] = result.optimal_bonded
            leg.sim_config.coul_lambdas[leg.leg_type] = result.optimal_coul
            leg.sim_config.vdw_lambdas[leg.leg_type] = result.optimal_vdw

            # Step 3: Update lambda indices to match new window count
            leg.lam_indices = list(range(len(result.optimal_bonded)))

            # Step 4: Clear existing lambda windows
            leg._sub_sim_runners.clear()

            # Step 5: Remove old lambda directories and recreate using existing Leg methods
            self._clean_and_recreate_directories(leg)

            # Step 6: Instantiate new lambda windows (reusing existing method)
            leg._instantiate_lambda_windows()

            logger.info(
                f"✅ Successfully applied optimization to {leg.leg_type.name} leg"
            )
            logger.info(f"Created {len(leg.lambda_windows)} new lambda windows")

        except Exception as e:
            logger.error(f"❌ Failed to apply optimization: {e}")
            # Try to restore from backup if it exists
            if backup_original and leg.has_backup:
                leg.restore_from_backup()
            raise

    def _clean_and_recreate_directories(self, leg: Leg) -> None:
        """Clean old lambda directories and recreate using existing Leg methods"""
        import shutil

        leg_base = Path(leg.output_dir)
        leg_input = leg_base / "input"

        # Step 1: Remove existing lambda_* directories (but keep input/ and other files)
        logger.info("Removing old lambda directories...")
        for item in leg_base.iterdir():
            if item.is_dir() and item.name.startswith("lambda_"):
                shutil.rmtree(item)

        # Step 2: Verify leg_input exists (should already exist from original setup)
        if not leg_input.exists():
            raise RuntimeError(f"Input directory not found: {leg_input}")

        # Step 3: Reuse existing Leg method to create new directory structure
        logger.info(f"Creating {len(leg.lam_indices)} new lambda directories...")
        leg._make_lambda_run_dirs_and_link(leg_input)

        logger.info("✅ Directory structure recreated successfully")
