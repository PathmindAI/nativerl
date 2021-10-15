import os

from ray.tune.logger import CSVLogger, DEFAULT_LOGGERS
from ray.tune.result import EXPR_PROGRESS_FILE


class PathmindCSVLogger(CSVLogger):
    def _init(self):
        """CSV outputted with Headers as first set of results."""
        progress_file = os.path.join(self.logdir, EXPR_PROGRESS_FILE)
        self._continuing = (
            os.path.exists(progress_file) and os.path.getsize(progress_file) > 0
        )
        self._file = open(progress_file, "a")
        self._csv_out = None


def get_loggers():
    # (ray.tune.logger.JsonLogger,
    #  ray.tune.logger.CSVLogger,
    #  ray.tune.logger.TBXLogger)
    loggers = list(DEFAULT_LOGGERS)
    loggers[1] = PathmindCSVLogger
    return loggers
