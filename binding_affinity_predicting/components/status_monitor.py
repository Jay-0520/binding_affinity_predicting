"""
Utilities for monitoring GROMACS calculation status.
Works with Calculation, Leg, LambdaWindow, and Simulation objects.
"""

from pathlib import Path
from typing import Any

from binding_affinity_predicting.components.gromacs_orchestration import Calculation
from binding_affinity_predicting.components.simulation import Simulation
from binding_affinity_predicting.data.enums import JobStatus


class StatusMonitor:
    """Monitor status of GROMACS calculations.
    We only consider status defined in Simulation class
     (FINISHED, FAILED and RUNNING)
    """

    def __init__(self, calculation: "Calculation") -> None:
        self.calculation = calculation

    def get_status(self, update_hpc: bool = True) -> dict[str, Any]:
        """
        Get simple job status counts and locations for all simulations.
        Works for both local and HPC execution modes.
        """
        if update_hpc:
            self._update_hpc_statuses()

        status: dict[str, Any] = {
            "job_counts": {
                "RUNNING": 0,
                "FINISHED": 0,
                "FAILED": 0,
                "QUEUED": 0,
            },
            "job_locations": {
                "RUNNING": [],
                "FINISHED": [],
                "FAILED": [],
                "QUEUED": [],
                "NONE": [],
            },
            "total_jobs": 0,
            "execution_mode": (
                "hpc"
                if getattr(self.calculation, "_last_run_used_hpc", False)
                else "local"
            ),
        }

        # Go through all simulations
        for leg in self.calculation._sub_sim_runners:
            for window in leg._sub_sim_runners:
                for sim in window._sub_sim_runners:
                    status["total_jobs"] += 1

                    # Determine job status
                    job_status = self._get_job_status(sim)
                    job_status_name = job_status.name

                    # Update counts
                    status["job_counts"][job_status_name] += 1

                    # Add location info
                    work_dir = str(getattr(sim, "work_dir", "unknown"))
                    location_info = {
                        "work_dir": work_dir,
                        "lambda_state": getattr(sim, "lam_state", "unknown"),
                        "run_index": getattr(sim, "run_index", "unknown"),
                    }
                    status["job_locations"][job_status_name].append(location_info)

        return status

    def _update_hpc_statuses(self) -> None:
        """
        Update all HPC simulation statuses by checking VirtualQueue jobs.
        Call this before getting status to ensure up-to-date information.
        """
        for leg in self.calculation._sub_sim_runners:
            for window in leg._sub_sim_runners:
                if hasattr(window, "update_simulation_statuses"):
                    window.update_simulation_statuses()

    def _get_job_status(self, sim: "Simulation") -> JobStatus:
        """Determine job status for a simulation."""
        # Check if simulation has explicit job_status
        if hasattr(sim, "job_status") and isinstance(sim.job_status, JobStatus):
            return sim.job_status

        # Simple fallback logic when sim already has status
        if getattr(sim, "_failed", False):
            return JobStatus.FAILED

        if getattr(sim, "_finished", False):
            return JobStatus.FINISHED

        if getattr(sim, "_running", False) or getattr(sim, "running", False):
            return JobStatus.RUNNING

        if getattr(sim, "_submitted_via_hpc", False):
            return JobStatus.QUEUED

        # For local runs, check if output files exist to detect completion
        if not getattr(self.calculation, "_last_run_used_hpc", False):
            work_dir = getattr(sim, "work_dir", None)
            if work_dir and Path(work_dir).exists():
                # Check for GROMACS output files
                run_index = getattr(sim, "run_index", 1)
                lam_state = getattr(sim, "lam_state", 0)

                # Look for typical GROMACS output files
                # TODO: now we hard-coded the output file format file
                # need to centralize this somewhere
                output_patterns = [
                    f"lambda_{lam_state}_run_{run_index}.log",
                    f"lambda_{lam_state}_run_{run_index}.edr",
                ]

                work_path = Path(work_dir)
                output_exists = any(
                    (work_path / pattern).exists() for pattern in output_patterns
                )

                if output_exists:
                    # Check if it completed successfully
                    log_file = work_path / f"lambda_{lam_state}_run_{run_index}.log"
                    if log_file.exists() and self._check_completion(str(log_file)):
                        # Update simulation status
                        if hasattr(sim, "finished"):
                            sim._finished = True
                        if hasattr(sim, "job_status"):
                            sim.job_status = JobStatus.FINISHED
                        return JobStatus.FINISHED
                    else:
                        # Output files exist but no completion indicator - might have failed
                        if self._check_for_errors(work_path):
                            if hasattr(sim, "failed"):
                                sim._failed = True
                            if hasattr(sim, "job_status"):
                                sim.job_status = JobStatus.FAILED
                            return JobStatus.FAILED

        # If it`s not running, not finished, not failed, and no files exist
        # → maybe it`s just QUEUED
        mdp_file = getattr(sim, "mdp_file", None)
        if mdp_file and Path(mdp_file).exists():
            return JobStatus.QUEUED

        # Otherwise, no evidence of queue, no files, so it`s “NONE” (never set up or never started)
        return JobStatus.NONE

    def _check_completion(self, log_file: str) -> bool:
        """Check if GROMACS completed successfully by looking at log file."""
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                if len(lines) < 5:
                    return False

                last_lines = "".join(lines[-20:]).lower()
                return any(
                    indicator in last_lines
                    for indicator in [
                        "Finished mdrun",
                    ]
                )
        except (IOError, OSError):
            return False

    def _check_for_errors(self, work_path: Path) -> bool:
        """Check for error indicators in work directory."""
        try:
            if list(work_path.glob("*.err")) or list(work_path.glob("core.*")):
                return True

            for log_file in work_path.glob("*.log"):
                try:
                    with open(log_file, "r") as f:
                        content = f.read().lower()
                        if "fatal error" in content or "segmentation fault" in content:
                            return True
                except (IOError, OSError):
                    continue

            return False
        except Exception:
            return False

    def get_summary(self) -> str:
        """
        Get a one-line status summary string of the form:
        “X/Y finished, F failed, R running. Failed job locations: L1, L2, …”
        If there are no failed jobs, it will say “No failed jobs.”
        """
        status = self.get_status()

        # Pull out the job_counts sub-dict (or use empty defaults)
        job_counts = status.get("job_counts", {})
        finished = job_counts.get("FINISHED", 0)
        failed = job_counts.get("FAILED", 0)
        running = job_counts.get("RUNNING", 0)
        queue = job_counts.get("QUEUED", 0)

        total = status.get("total_jobs", finished + failed + running + queue)

        failed_job_locations = status.get("job_locations", {}).get("FAILED", []) or []

        summary = f"{finished}/{total} finished, {failed} failed, {running} running, {queue} queue."  # noqa

        if failed_job_locations:
            locs_str = ", ".join(failed_job_locations)
            summary += f" Failed job locations: {locs_str}"
        else:
            summary += " No failed jobs."

        return summary


# Convenience functions for easy access
def get_calculation_status(calculation: "Calculation") -> dict[str, Any]:
    """Get status for a calculation."""
    monitor = StatusMonitor(calculation)
    return monitor.get_status()


def get_calculation_summary(calculation: "Calculation") -> str:
    """Get one-line summary for a calculation."""
    monitor = StatusMonitor(calculation)
    return monitor.get_summary()
