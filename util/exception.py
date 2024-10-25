# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import TypeVar, Literal

F = TypeVar("F")

class Expect:
    def __init__(self, *exceptions: Exception, traceback: bool = False):
        self.exceptions = exceptions
        self.traceback = traceback

    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, e_tb):
        managed = len(self.exceptions) == 0 or e_type in self.exceptions
        if managed and self.traceback and e_tb:
            import traceback
            if e_type is KeyboardInterrupt:
                print()
            print(e_type.__name__, e_val)
            traceback.print_tb(e_tb)
        return managed

    def __call__(self, fn: F) -> F:
        def decorator(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return decorator

def no_throw(fn: F) -> F:
    return Expect()(fn)

import signal
class MaskSignal:
    from signal import SIGINT, SIGTERM, SIGQUIT, SIGABRT, SIGKILL, SIGSTOP

    def __init__(self, *signals: int):
        self.signals = signals

    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, e_tb):
        return e_type is KeyboardInterrupt

    def __call__(self, fn: F) -> F:
        def decorator(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return decorator
