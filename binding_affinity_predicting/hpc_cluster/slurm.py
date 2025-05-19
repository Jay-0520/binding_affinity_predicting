import logging
import os
from pathlib import Path
from time import sleep
from typing import Callable, Dict, List, Optional, Tuple

from binding_affinity_predicting.hpc_cluster.utils import get_slurm_file_base
from binding_affinity_predicting.hpc_cluster.virtual_queue import VirtualQueue

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_slurm(
    sys_prep_fn: Callable[..., None],
    wait: bool,
    run_dir: str,
    job_name: str,
    stream_log_level: int = logging.INFO,
    *fn_args,
    **fn_kwargs,
) -> None:
    """
    Submit one of system_prep functions as a SLURM job via the virtual queue.

    Parameters
    ----------
    sys_prep_fn : Callable[..., None]
        The system_prep function, e.g. minimise_input or heat_and_preequil_input.
    wait : bool
        If True, block until *all* virtual‐queued jobs (not just this one) have finished.
    run_dir : str
        Directory where 'run_somd.sh' (SLURM header) lives and where we write *.sh.
    job_name : str
        Base name for the SLURM script and job.
    *fn_args, **fn_kwargs:
        Extra positional and keyword args passed through to sys_prep_fn when Python is invoked.

    """
    # Create a virtual queue for the prep jobs
    virtual_queue = VirtualQueue(log_dir=run_dir, stream_log_level=stream_log_level)

    header_lines = []
    # 1) Pull only the SBATCH/#!/bin/bash lines from run_somd.sh
    file_fullpath = os.path.join(run_dir, "run_gmx.sh")
    try:
        with open(file_fullpath, "r") as f:
            for line in f:
                if line.startswith("#SBATCH") or line.startswith("#!/bin/bash"):
                    header_lines.append(line)
                else:
                    break
    except FileNotFoundError:
        logger.error(f"Could not find SLURM header template at {file_fullpath}")
        raise

    # 2) Append Python call
    arglist = ", ".join([repr(arg) for arg in fn_args])
    kwarglist = ", ".join([f"{k}={repr(v)}" for k, v in fn_kwargs.items()])
    all_args = ", ".join([x for x in (arglist, kwarglist) if x])
    header_lines.append(
        f"\npython -c 'from {sys_prep_fn.__module__} import {sys_prep_fn.__name__}; {sys_prep_fn.__name__}({all_args})'"
    )
    slurm_script = f"{run_dir}/{job_name}.sh"
    with open(slurm_script, "w") as file:
        file.writelines(header_lines)

    logger.debug(f"Wrote SLURM script to {slurm_script}")

    # 4) Submit it via virtual queue
    submit_cmd = ["--chdir", str(run_dir), str(slurm_script)]
    slurm_base = get_slurm_file_base(str(slurm_script))

    try:
        job = virtual_queue.submit(submit_cmd, slurm_file_base=slurm_base)
        logger.info(f"Submitted {job_name!r} as virtual job {job.virtual_job_id}")
    except Exception as e:
        logger.error(f"Failed to submit {job_name!r} to virtual queue: {e}")
        raise

    # 5) Update the virtual queue to submit the job
    virtual_queue.update()

    # 6) Always wait untit the job is submitted to the real slurm queue
    while virtual_queue._pre_queue:
        logger.debug("Waiting for all pre‐queued jobs to be pushed to SLURM…")
        sleep(5 * 60)
        virtual_queue.update()

    # 7) If the caller requested a blocking wait, wait for *all* jobs to finish
    if wait:
        while job in virtual_queue.queue:
            logger.info(f"Waiting for job {job} to complete")
            sleep(30)
            virtual_queue.update()
