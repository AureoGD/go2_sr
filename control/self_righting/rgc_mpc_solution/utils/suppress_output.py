import os
import warnings
from contextlib import contextmanager


@contextmanager
def suppress_output():

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        null_fd = None
        save_stdout = None
        save_stderr = None

        try:

            null_fd = os.open(os.devnull, os.O_RDWR)

            save_stdout = os.dup(1)
            save_stderr = os.dup(2)

            os.dup2(null_fd, 1)
            os.dup2(null_fd, 2)

            yield

        finally:

            if save_stdout is not None:
                os.dup2(save_stdout, 1)

            if save_stderr is not None:
                os.dup2(save_stderr, 2)

            if null_fd is not None:
                os.close(null_fd)

            if save_stdout is not None:
                os.close(save_stdout)

            if save_stderr is not None:
                os.close(save_stderr)
