import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from binding_affinity_predicting.components.gromacs_orchestration import (
    Calculation,
    LambdaWindow,
    Leg,
)
from binding_affinity_predicting.data.enums import LegType

logger = logging.getLogger(__name__)


class FepStage(Enum):
    """GROMACS FEP simulation stages"""

    RESTRAINED = "restrained"  # bonded-lambdas vary
    DISCHARGING = "discharging"  # coul-lambdas vary
    VANISHING = "vanishing"  # vdw-lambdas vary


@dataclass
class ValidationResult:
    """Result of lambda vector validation"""

    is_valid: bool
    error_message: Optional[str] = None
    warnings: list[str] = None
    stage_boundaries: Optional[dict[FepStage, tuple[int, int]]] = (
        None  # (start_idx, end_idx)
    )

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class LambdaVectorValidator:
    """Validates that lambda vectors follow proper GROMACS FEP stage format"""

    def __init__(self, tolerance: float = 1e-6):
        self.tol = tolerance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate_lambda_vectors(
        self,
        bonded_lambdas: list[float],
        coul_lambdas: list[float],
        vdw_lambdas: list[float],
    ) -> ValidationResult:
        """
        Validate that lambda vectors follow proper GROMACS FEP format.

        Expected format:
        1. RESTRAINED stage: bonded varies (0→1), coul=0, vdw=0
        2. DISCHARGING stage: coul varies (0→1), bonded=1, vdw=0
        3. VANISHING stage: vdw varies (0→1), bonded=1, coul=1

        Parameters
        ----------
        bonded_lambdas : list[float]
            Bonded lambda values
        coul_lambdas : list[float]
            Coulomb lambda values
        vdw_lambdas : list[float]
            Van der Waals lambda values

        Returns
        -------
        ValidationResult
            Validation results with error messages and stage boundaries
        """

        # Basic length check
        n_states = len(bonded_lambdas)
        if not (len(coul_lambdas) == n_states and len(vdw_lambdas) == n_states):
            return ValidationResult(
                is_valid=False,
                error_message="Lambda vectors have different lengths: "
                f"bonded={len(bonded_lambdas)}, coul={len(coul_lambdas)}, vdw={len(vdw_lambdas)}",
            )

        if n_states < 3:
            return ValidationResult(
                is_valid=False,
                error_message=f"Need at least 3 lambda states, got {n_states}",
            )

        # Identify stages
        try:
            stage_boundaries = self._identify_stages(
                bonded_lambdas, coul_lambdas, vdw_lambdas
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False, error_message=f"Failed to identify FEP stages: {str(e)}"
            )

        # Validate each stage
        validation_errors = []
        warnings = []

        for stage, (start_idx, end_idx) in stage_boundaries.items():
            stage_errors, stage_warnings = self._validate_stage(
                stage, start_idx, end_idx, bonded_lambdas, coul_lambdas, vdw_lambdas
            )
            validation_errors.extend(stage_errors)
            warnings.extend(stage_warnings)

        # Check stage continuity
        continuity_errors = self._validate_stage_continuity(
            stage_boundaries, bonded_lambdas, coul_lambdas, vdw_lambdas
        )
        validation_errors.extend(continuity_errors)

        # Final result
        is_valid = len(validation_errors) == 0
        error_message = "; ".join(validation_errors) if validation_errors else None

        if is_valid:
            self.logger.info("✅ Lambda vectors validation passed")
            for stage, (start, end) in stage_boundaries.items():
                self.logger.info(
                    f"   {stage.value}: states {start}-{end} ({end-start+1} windows)"
                )
        else:
            self.logger.error(f"❌ Lambda vectors validation failed: {error_message}")

        return ValidationResult(
            is_valid=is_valid,
            error_message=error_message,
            warnings=warnings,
            stage_boundaries=stage_boundaries if is_valid else None,
        )

    def _identify_stages(
        self, bonded: list[float], coul: list[float], vdw: list[float]
    ) -> dict[FepStage, tuple[int, int]]:
        """Identify the boundaries of each FEP stage"""

        n_states = len(bonded)
        stage_boundaries: dict[FepStage, tuple[int, int]] = {}

        # Find RESTRAINED stage: bonded varies, coul=0, vdw=0
        restrained_states = []
        for i in range(n_states):
            if abs(coul[i]) < self.tol and abs(vdw[i]) < self.tol:
                restrained_states.append(i)

        if restrained_states:
            stage_boundaries[FepStage.RESTRAINED] = (
                restrained_states[0],
                restrained_states[-1],
            )

        # Find DISCHARGING stage: bonded=1, coul varies, vdw=0
        discharging_states = []
        for i in range(n_states):
            if (
                abs(bonded[i] - 1.0) < self.tol
                and abs(vdw[i]) < self.tol
                and coul[i] > self.tol
            ):
                discharging_states.append(i)

        if discharging_states:
            stage_boundaries[FepStage.DISCHARGING] = (
                discharging_states[0],
                discharging_states[-1],
            )

        # Find VANISHING stage: bonded=1, coul=1, vdw varies
        vanishing_states = []
        for i in range(n_states):
            if (
                abs(bonded[i] - 1.0) < self.tol
                and abs(coul[i] - 1.0) < self.tol
                and vdw[i] > self.tol
            ):
                vanishing_states.append(i)

        if vanishing_states:
            stage_boundaries[FepStage.VANISHING] = (
                vanishing_states[0],
                vanishing_states[-1],
            )

        # Check that we found all expected stages
        expected_stages = {
            FepStage.RESTRAINED,
            FepStage.DISCHARGING,
            FepStage.VANISHING,
        }
        found_stages = set(stage_boundaries.keys())

        if not expected_stages.issubset(found_stages):
            missing = expected_stages - found_stages
            raise ValueError(f"Missing FEP stages: {[s.value for s in missing]}")

        return stage_boundaries

    def _validate_stage(
        self,
        stage: FepStage,
        start_idx: int,
        end_idx: int,
        bonded: list[float],
        coul: list[float],
        vdw: list[float],
    ) -> tuple[list[str], list[str]]:
        """Validate a specific FEP stage"""

        errors: list[str] = []
        warnings: list[str] = []

        # Extract stage data
        stage_bonded = bonded[start_idx : end_idx + 1]
        stage_coul = coul[start_idx : end_idx + 1]
        stage_vdw = vdw[start_idx : end_idx + 1]

        if stage == FepStage.RESTRAINED:
            errors.extend(
                self._validate_varying_lambda(stage_bonded, "bonded", stage.value)
            )
            errors.extend(
                self._validate_fixed_lambda(stage_coul, 0.0, "coul", stage.value)
            )
            errors.extend(
                self._validate_fixed_lambda(stage_vdw, 0.0, "vdw", stage.value)
            )

        elif stage == FepStage.DISCHARGING:
            errors.extend(
                self._validate_fixed_lambda(stage_bonded, 1.0, "bonded", stage.value)
            )
            errors.extend(
                self._validate_varying_lambda(stage_coul, "coul", stage.value)
            )
            errors.extend(
                self._validate_fixed_lambda(stage_vdw, 0.0, "vdw", stage.value)
            )

        elif stage == FepStage.VANISHING:
            errors.extend(
                self._validate_fixed_lambda(stage_bonded, 1.0, "bonded", stage.value)
            )
            errors.extend(
                self._validate_fixed_lambda(stage_coul, 1.0, "coul", stage.value)
            )
            errors.extend(self._validate_varying_lambda(stage_vdw, "vdw", stage.value))

        # Check for reasonable number of windows
        n_windows = len(stage_bonded)
        if n_windows < 2:
            errors.append(
                f"{stage.value} stage has only {n_windows} window(s), need at least 2"
            )
        elif n_windows < 4:
            warnings.append(
                f"{stage.value} stage has only {n_windows} windows, consider more for better convergence"
            )

        return errors, warnings

    def _validate_varying_lambda(
        self, lambdas: list[float], lambda_type: str, stage_name: str
    ) -> list[str]:
        """Validate that lambda values properly vary from 0 to 1"""
        errors: list[str] = []

        if len(lambdas) < 2:
            return [f"{stage_name} {lambda_type}-lambda has insufficient data points"]

        # Check start and end values
        if abs(lambdas[0]) > self.tol:
            errors.append(
                f"{stage_name} {lambda_type}-lambda should start at 0.0, got {lambdas[0]:.6f}"
            )
        if abs(lambdas[-1] - 1.0) > self.tol:
            errors.append(
                f"{stage_name} {lambda_type}-lambda should end at 1.0, got {lambdas[-1]:.6f}"
            )

        # Check monotonicity
        for i in range(1, len(lambdas)):
            if lambdas[i] < lambdas[i - 1] - self.tol:
                errors.append(
                    f"{stage_name} {lambda_type}-lambda is not monotonic: "
                    f"λ[{i-1}]={lambdas[i-1]:.6f} > λ[{i}]={lambdas[i]:.6f}"
                )

        # Check range
        min_val, max_val = min(lambdas), max(lambdas)
        if min_val < -self.tol:
            errors.append(
                f"{stage_name} {lambda_type}-lambda has negative values: min={min_val:.6f}"
            )
        if max_val > 1.0 + self.tol:
            errors.append(
                f"{stage_name} {lambda_type}-lambda exceeds 1.0: max={max_val:.6f}"
            )

        return errors

    def _validate_fixed_lambda(
        self,
        lambdas: list[float],
        expected_value: float,
        lambda_type: str,
        stage_name: str,
    ) -> list[str]:
        """Validate that lambda values are fixed at expected value"""
        errors: list[str] = []

        for i, val in enumerate(lambdas):
            if abs(val - expected_value) > self.tol:
                errors.append(
                    f"{stage_name} {lambda_type}-lambda[{i}] should be {expected_value:.1f}, "
                    f"got {val:.6f}"
                )

        return errors

    def _validate_stage_continuity(
        self,
        stage_boundaries: dict[FepStage, tuple[int, int]],
        bonded: list[float],
        coul: list[float],
        vdw: list[float],
    ) -> list[str]:
        """Validate continuity between stages"""
        errors: list[str] = []

        # Check RESTRAINED → DISCHARGING transition
        if (
            FepStage.RESTRAINED in stage_boundaries
            and FepStage.DISCHARGING in stage_boundaries
        ):
            rest_end = stage_boundaries[FepStage.RESTRAINED][1]
            disch_start = stage_boundaries[FepStage.DISCHARGING][0]

            if disch_start > rest_end + 1:
                errors.append(
                    f"Gap between RESTRAINED and DISCHARGING stages: "
                    f"indices {rest_end} → {disch_start}"
                )
            if abs(bonded[rest_end] - 1.0) > self.tol:
                errors.append(
                    f"RESTRAINED stage should end with bonded=1.0, got {bonded[rest_end]:.6f}"
                )

        # Check DISCHARGING → VANISHING transition
        if (
            FepStage.DISCHARGING in stage_boundaries
            and FepStage.VANISHING in stage_boundaries
        ):
            disch_end = stage_boundaries[FepStage.DISCHARGING][1]
            van_start = stage_boundaries[FepStage.VANISHING][0]

            if van_start > disch_end + 1:
                errors.append(
                    f"Gap between DISCHARGING and VANISHING stages: "
                    f"indices {disch_end} → {van_start}"
                )
            if abs(coul[disch_end] - 1.0) > self.tol:
                errors.append(
                    f"DISCHARGING stage should end with coul=1.0, got {coul[disch_end]:.6f}"
                )

        return errors

    def suggest_corrections(
        self,
        bonded_lambdas: list[float],
        coul_lambdas: list[float],
        vdw_lambdas: list[float],
    ) -> Optional[dict[str, list[float]]]:
        """Suggest corrected lambda vectors if validation fails"""

        validation = self.validate_lambda_vectors(
            bonded_lambdas, coul_lambdas, vdw_lambdas
        )

        if validation.is_valid:
            return None  # No correction needed

        self.logger.info("Generating suggested lambda vector corrections...")

        suggested = {
            "bonded": [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "coul": [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0],
            "vdw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }

        vanishing_lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        suggested = {
            "bonded": [0.0, 0.25, 0.5, 0.75, 1.0]
            + [1.0] * 5
            + [1.0] * len(vanishing_lambdas),
            "coul": [0.0] * 5
            + [0.0, 0.25, 0.5, 0.75, 1.0]
            + [1.0] * len(vanishing_lambdas),
            "vdw": [0.0] * 10 + vanishing_lambdas,
        }

        self.logger.info("Suggested lambda vectors:")
        self.logger.info(f"  bonded-lambdas = {suggested['bonded']}")
        self.logger.info(f"  coul-lambdas   = {suggested['coul']}")
        self.logger.info(f"  vdw-lambdas    = {suggested['vdw']}")

        return suggested
