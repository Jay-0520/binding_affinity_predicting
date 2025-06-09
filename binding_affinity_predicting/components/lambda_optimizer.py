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
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, model_validator

from binding_affinity_predicting.components.gromacs_orchestration import (
    Calculation,
    LambdaWindow,
    Leg,
)
from binding_affinity_predicting.data.enums import LegType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# Enums and Configuration Classes
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


@dataclass
class StageConfig:
    """Configuration for optimizing a specific FEP stage"""

    method: SpacingMethod = SpacingMethod.TARGET_ERROR
    target_error: float = 1.0
    n_windows: Optional[int] = None
    user_lambdas: Optional[List[float]] = None
    error_type: str = "root_var"  # "root_var" or "sem"
    sem_origin: str = "inter"  # "inter" or "intra"
    smoothen_errors: bool = True
    round_lambdas: bool = True


@dataclass
class OptimizationConfig:
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
    root_variance: float
    sem_inter: float
    sem_intra: float
    n_data_points: int
    simulation_time: float


@dataclass
class StageResult:
    """Optimization results for a single FEP stage"""

    stage: FepStage
    original_lambdas: List[float]
    optimal_lambdas: List[float]
    window_analyses: List[WindowAnalysis]
    improvement_factor: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class OptimizationResult:
    """Complete multi-stage optimization results"""

    leg_type: str
    stage_results: Dict[FepStage, StageResult]

    # Combined lambda vectors for GROMACS
    original_bonded: List[float]
    original_coul: List[float]
    original_vdw: List[float]

    optimal_bonded: List[float]
    optimal_coul: List[float]
    optimal_vdw: List[float]

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
    ) -> Optional[WindowAnalysis]:
        """Analyze gradients from a single lambda window"""

        # Extract lambda values and determine stage
        lambdas = self._extract_lambda_values(window)
        if lambdas is None:
            return None

        coul_lambda, vdw_lambda, bonded_lambda = lambdas
        stage, stage_lambda = self._determine_stage(
            coul_lambda, vdw_lambda, bonded_lambda
        )

        # Collect gradient data from all runs
        gradient_data = self._collect_gradients(window, run_nos, equilibrated)
        if gradient_data is None:
            return None

        # Calculate statistics
        stats = self._calculate_statistics(gradient_data)

        return WindowAnalysis(
            lambda_state=getattr(window, 'lam_state', 0),
            coul_lambda=coul_lambda,
            vdw_lambda=vdw_lambda,
            bonded_lambda=bonded_lambda,
            stage=stage,
            stage_lambda=stage_lambda,
            **stats,
        )

    def _extract_lambda_values(
        self, window: LambdaWindow
    ) -> Optional[tuple[float, float, float]]:
        """Extract (coul, vdw, bonded) lambda values from window"""
        try:
            sim_params = window.sim_params
            lam_state = getattr(window, 'lam_state', 0)

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
            return None

    def _determine_stage(
        self, coul: float, vdw: float, bonded: float
    ) -> tuple[FepStage, float]:
        """Determine FEP stage and return the varying lambda"""
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
    ) -> Optional[dict]:
        """Collect gradient data from all runs"""
        if run_nos is None:
            run_nos = list(range(1, window.ensemble_size + 1))

        all_gradients = []
        run_means = []
        total_time = 0.0

        for run_no in run_nos:
            gradients, sim_time = self._read_run_gradients(window, run_no, equilibrated)
            if gradients is not None and len(gradients) > 0:
                all_gradients.append(gradients)
                run_means.append(np.mean(gradients))
                total_time += sim_time

        if not all_gradients:
            return None

        return {
            'all_gradients': all_gradients,
            'run_means': run_means,
            'total_time': total_time,
        }

    def _read_run_gradients(
        self, window: LambdaWindow, run_no: int, equilibrated: bool
    ) -> Tuple[Optional[np.ndarray], float]:
        """Read gradients from a single run"""
        try:
            run_dir = Path(window.output_dir) / f"run_{run_no}"
            xvg_file = run_dir / f"lambda_{window.lam_state}_run_{run_no}.xvg"

            if not xvg_file.exists():
                return None, 0.0

            times, gradients = self._parse_xvg_file(xvg_file)

            if equilibrated and hasattr(window, '_equil_time'):
                equil_idx = np.searchsorted(times, window._equil_time)
                times = times[equil_idx:]
                gradients = gradients[equil_idx:]

            return gradients, times[-1] if len(times) > 0 else 0.0

        except Exception as e:
            logger.warning(f"Failed to read run {run_no}: {e}")
            return None, 0.0

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
        """Calculate gradient statistics"""
        all_gradients = gradient_data['all_gradients']
        run_means = gradient_data['run_means']

        # Combine all gradients and calculate basic stats
        combined = np.concatenate(all_gradients)
        mean_gradient = np.mean(combined)
        root_variance = np.sqrt(np.var(combined))

        # Calculate SEMs
        n_runs = len(run_means)
        sem_inter = np.std(run_means) / np.sqrt(n_runs) if n_runs > 1 else 0.0

        # Simplified intra-run SEM
        intra_vars = [np.var(grads) for grads in all_gradients]
        mean_n_points = np.mean([len(grads) for grads in all_gradients])
        sem_intra = np.sqrt(np.mean(intra_vars) / mean_n_points)

        return {
            'mean_gradient': mean_gradient,
            'root_variance': root_variance,
            'sem_inter': sem_inter,
            'sem_intra': sem_intra,
            'n_data_points': len(combined),
            'simulation_time': gradient_data['total_time'],
        }


class StageOptimizer:
    """Optimizes lambda spacing for a single FEP stage"""

    def optimize_stage(
        self, stage: FepStage, analyses: List[WindowAnalysis], config: StageConfig
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
                original_lambdas, optimal_lambdas, analyses, config
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
        self, analyses: List[WindowAnalysis], config: StageConfig
    ) -> List[float]:
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
        self, analyses: List[WindowAnalysis], config: StageConfig
    ) -> List[float]:
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

    def _smooth_errors(self, errors: List[float]) -> List[float]:
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
        lambdas: List[float],
        errors: List[float],
        target_error: float,
        round_lambdas: bool,
    ) -> List[float]:
        """Optimize for target error per interval"""

        # Interpolate and integrate
        lam_interp = np.linspace(0, 1, 1000)
        errors_interp = np.interp(lam_interp, lambdas, errors)
        cumulative = np.cumsum(errors_interp) * (lam_interp[1] - lam_interp[0])

        # Place lambdas at target intervals
        max_error = cumulative[-1]
        n_intervals = max(2, int(np.ceil(max_error / target_error)))

        optimal = [0.0]
        for i in range(1, n_intervals):
            target = (i / n_intervals) * max_error
            idx = np.argmin(np.abs(cumulative - target))
            optimal.append(lam_interp[idx])
        optimal.append(1.0)

        if round_lambdas:
            optimal = [round(x, 3) for x in optimal]

        return optimal

    def _optimize_for_fixed_windows(
        self,
        lambdas: List[float],
        errors: List[float],
        n_windows: int,
        round_lambdas: bool,
    ) -> List[float]:
        """Optimize for fixed number of windows"""

        # High-resolution interpolation
        lam_interp = np.linspace(0, 1, 10000)
        errors_interp = np.interp(lam_interp, lambdas, errors)
        cumulative = np.cumsum(errors_interp)
        total_difficulty = cumulative[-1]

        # Equal difficulty intervals
        optimal = [0.0]
        for i in range(1, n_windows - 1):
            target = (i / (n_windows - 1)) * total_difficulty
            idx = np.argmin(np.abs(cumulative - target))
            optimal.append(lam_interp[idx])
        optimal.append(1.0)

        if round_lambdas:
            optimal = [round(x, 3) for x in optimal]

        return optimal

    def _calculate_improvement(
        self,
        original: List[float],
        optimal: List[float],
        analyses: List[WindowAnalysis],
        config: StageConfig,
    ) -> float:
        """Calculate improvement factor (simplified)"""

        errors = self._get_errors(analyses, config)
        lambdas = [a.stage_lambda for a in analyses]

        # Simple thermodynamic length calculation
        total_error = np.trapz(errors, lambdas)

        # Theoretical optimal variance
        v_opt = (total_error**2) / len(optimal)

        # Current variance (simplified)
        v_current = (total_error**2) / len(original)

        return np.sqrt(v_opt / v_current) if v_current > 0 else 1.0


class MultiStageOptimizer:
    """Main optimizer that coordinates all stages"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.gradient_analyzer = GradientAnalyzer()
        self.stage_optimizer = StageOptimizer()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def optimize(
        self,
        lambda_windows: List[LambdaWindow],
        leg_type: LegType,
        run_nos: Optional[List[int]] = None,
        equilibrated: bool = False,
    ) -> OptimizationResult:
        """Optimize lambda spacing for all stages"""

        self.logger.info(
            f"Starting multi-stage optimization for {leg_type.name} leg..."
        )

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
            self.logger.error(f"Optimization failed: {e}")
            return self._create_failed_result(leg_type, str(e))

    def _analyze_windows(
        self,
        windows: List[LambdaWindow],
        run_nos: Optional[List[int]],
        equilibrated: bool,
    ) -> List[WindowAnalysis]:
        """Analyze all lambda windows"""
        analyses = []

        for window in windows:
            analysis = self.gradient_analyzer.analyze_window(
                window, run_nos, equilibrated
            )
            if analysis:
                analyses.append(analysis)
                self.logger.info(
                    f"Analyzed λ{analysis.lambda_state}: {analysis.stage.value} "
                    f"stage_λ={analysis.stage_lambda:.3f} σ={analysis.root_variance:.2f}"
                )

        return analyses

    def _group_by_stage(
        self, analyses: List[WindowAnalysis]
    ) -> Dict[FepStage, List[WindowAnalysis]]:
        """Group analyses by FEP stage"""
        groups = {stage: [] for stage in FepStage}

        for analysis in analyses:
            groups[analysis.stage].append(analysis)

        for stage, group in groups.items():
            if group:
                self.logger.info(f"{stage.value} stage: {len(group)} windows")

        return groups

    def _combine_results(
        self, stage_results: Dict[FepStage, StageResult], leg_type: LegType
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

        result = OptimizationResult(
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

        self._log_results(result)
        return result

    def _log_results(self, result: OptimizationResult) -> None:
        """Log optimization results"""
        self.logger.info("=" * 60)
        self.logger.info("MULTI-STAGE OPTIMIZATION RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Leg: {result.leg_type}")
        self.logger.info(f"Success: {result.success}")
        self.logger.info(f"Overall improvement: {result.overall_improvement:.3f}")

        for stage, stage_result in result.stage_results.items():
            self.logger.info(f"\n{stage.value.upper()}:")
            if stage_result.success:
                self.logger.info(
                    f"  {len(stage_result.original_lambdas)} → {len(stage_result.optimal_lambdas)} windows"
                )
                self.logger.info(
                    f"  Improvement: {stage_result.improvement_factor:.3f}"
                )
                self.logger.info(f"  Lambdas: {stage_result.optimal_lambdas}")
            else:
                self.logger.info(f"  ❌ FAILED: {stage_result.error_message}")

        self.logger.info(f"\nCOMBINED VECTORS:")
        self.logger.info(f"bonded-lambdas: {result.optimal_bonded}")
        self.logger.info(f"coul-lambdas:   {result.optimal_coul}")
        self.logger.info(f"vdw-lambdas:    {result.optimal_vdw}")
        self.logger.info("=" * 60)

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


# ============================================================================
# Manager Class for Integration
# ============================================================================


class LambdaOptimizationManager:
    """High-level manager for lambda optimization"""

    def __init__(self, config: OptimizationConfig):
        self.optimizer = MultiStageOptimizer(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def optimize_calculation(
        self,
        calculation: Calculation,
        run_nos: Optional[List[int]] = None,
        equilibrated: bool = False,
        apply_results: bool = False,
    ) -> Dict[str, OptimizationResult]:
        """Optimize entire calculation"""

        results = {}

        for leg in calculation.legs:
            self.logger.info(f"Optimizing {leg.leg_type.name} leg...")

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

    def _apply_to_leg(self, leg: Leg, result: OptimizationResult) -> None:
        """Apply optimization results to a leg"""
        try:
            # Update lambda vectors
            leg.sim_config.bonded_lambdas[leg.leg_type] = result.optimal_bonded
            leg.sim_config.coul_lambdas[leg.leg_type] = result.optimal_coul
            leg.sim_config.vdw_lambdas[leg.leg_type] = result.optimal_vdw

            # Recreate windows
            leg.lam_indices = list(range(len(result.optimal_bonded)))
            leg._sub_sim_runners.clear()
            leg._instantiate_lambda_windows()

            self.logger.info(f"✅ Applied optimization to {leg.leg_type.name} leg")

        except Exception as e:
            self.logger.error(f"❌ Failed to apply optimization: {e}")
            raise


# ============================================================================
# Simple Usage Examples
# ============================================================================


def create_simple_config(target_error: float = 1.0) -> OptimizationConfig:
    """Create simple uniform configuration"""
    return OptimizationConfig(
        restrained=StageConfig(target_error=target_error),
        discharging=StageConfig(target_error=target_error),
        vanishing=StageConfig(target_error=target_error),
    )


def example_usage():
    """Example of how to use the optimizer"""

    # Simple configuration - same settings for all stages
    config = create_simple_config(target_error=1.0)

    # Create manager
    manager = LambdaOptimizationManager(config)

    # Optimize calculation
    # results = manager.optimize_calculation(
    #     calculation=your_gromacs_calculation,
    #     run_nos=[1, 2],
    #     apply_results=False  # Review results first
    # )

    # Check results
    # for leg_name, result in results.items():
    #     if result.success:
    #         print(f"✅ {leg_name}: Optimized successfully")
    #         print(f"   Improvement: {result.overall_improvement:.3f}")
    #         print(f"   bonded-lambdas: {result.optimal_bonded}")
    #         print(f"   coul-lambdas:   {result.optimal_coul}")
    #         print(f"   vdw-lambdas:    {result.optimal_vdw}")
    #     else:
    #         print(f"❌ {leg_name}: Optimization failed")

    return config


def example_stage_specific():
    """Example with different settings for each stage"""

    config = OptimizationConfig(
        restrained=StageConfig(
            method=SpacingMethod.USER_PROVIDED,
            user_lambdas=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        ),
        discharging=StageConfig(method=SpacingMethod.FIXED_WINDOWS, n_windows=6),
        vanishing=StageConfig(
            method=SpacingMethod.TARGET_ERROR,
            target_error=0.8,
            error_type="sem",
            sem_origin="inter",
        ),
    )

    manager = LambdaOptimizationManager(config)
    return config


if __name__ == "__main__":
    print("Clean Multi-Stage Lambda Optimizer for GROMACS")
    print("=" * 50)
    print("🔗 RESTRAINED stage:  bonded-lambdas vary")
    print("⚡ DISCHARGING stage: coul-lambdas vary")
    print("💨 VANISHING stage:   vdw-lambdas vary")
    print("✅ Clean, modular design")
    print("✅ Easy configuration")
    print("✅ Comprehensive logging")
    print("✅ GROMACS integration")
