import logging
from pathlib import Path

def _set_up_logging(self, null: bool = False) -> None:
    """
    Set up the logging for the simulation runner.

    Parameters
    ----------
    null : bool, optional, default=False
        Whether to silence all logging by writing to the null logger.
    """
    # If logger exists, remove it and start again
    if hasattr(self, "_logger"):
        handlers = self._logger.handlers[:]
        for handler in handlers:
            self._logger.removeHandler(handler)
            handler.close()
        del self._logger
    # Name each logger individually to avoid clashes
    self._logger = _logging.getLogger(
        f"{str(self)}_{next(self.__class__.class_count)}"
    )
    self._logger.propagate = False
    self._logger.setLevel(_logging.DEBUG)
    # For the file handler, we want to log everything
    file_handler = _logging.FileHandler(
        f"{self.base_dir}/{self.__class__.__name__}.log"
    )
    file_handler.setFormatter(_A3feFileFormatter())
    file_handler.setLevel(_logging.DEBUG)
    # For the stream handler, we want to log at the user-specified level
    stream_handler = _logging.StreamHandler()
    stream_handler.setFormatter(_A3feStreamFormatter())
    stream_handler.setLevel(self._stream_log_level)
    # Add the handlers to the logger
    self._logger.addHandler(file_handler)
    self._logger.addHandler(stream_handler)