"""Functionality for processing slurm files."""

import logging
import os as _os
import re as _re
from time import sleep
from typing import Callable, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_slurm_file_base(slurm_file: str) -> str:
    """
    Find out what the slurm output file will be called.

    Parameters
    ----------
    slurm_file : str
        The absolute path to the slurm job file.

    Returns
    -------
    slurm_file_base : str
        The file base for for any output written by the slurm job.
    """
    # Get the path to the base dir
    base_dir = _os.path.dirname(slurm_file)
    # Find the slurm output file
    with open(slurm_file, "r") as f:
        for line in f:
            split_line = _re.split(" |=", line)
            if len(split_line) > 0 and split_line[0] == "#SBATCH":
                if split_line[1] == "--output" or split_line[1] == "-o":
                    slurm_pattern = split_line[2]
                    if "%" in slurm_pattern:
                        file_base = slurm_pattern.split("%")[0]
                        return _os.path.join(base_dir, file_base)
                    else:
                        return _os.path.join(base_dir, slurm_pattern)

    # We haven't returned - raise an error
    raise RuntimeError(f"Could not find slurm output file name in {slurm_file}")


#### Adapted from https://stackoverflow.com/questions/50246304/using-python-decorators-to-retry-request ####
def retry(
    times: int,
    exceptions: tuple[Exception],
    wait_time: int,
    logger: Optional[logging.Logger] = None,
) -> Callable:
    """
    Retry a function a given number of times if the specified exceptions are raised.

    Parameters
    ----------
    times : int
        The number of retries to attempt before raising the error.
    exceptions : Tuple[Exception]
        A list of exceptions for which the function will be retried. The
        function will not be retried if an exception is raised which is not
        supplied.
    wait_time : int
        How long to wait between retries, in seconds.

    Returns
    -------
    decorator: Callable
        The retry decorator
    """

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.error(
                        f"Exception thrown when attempting to run {func}, attempt "
                        f"{attempt} of {times}"
                    )
                    logger.error(f"Exception thrown: {e}")
                    attempt += 1
                    # Wait for specified time before trying again
                    sleep(wait_time)

            return func(*args, **kwargs)

        return newfn

    return decorator
