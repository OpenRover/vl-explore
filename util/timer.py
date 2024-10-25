# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import sys
from typing import Callable
from time import perf_counter as perf, time as now


def log(*args):
    print(*args, file=sys.stderr)


UNIT = "s"
UNIT_UP = [("min", 60), ("hr", 60), ("day", 24)]
UNIT_DN = [("ms", 1000), ("us", 1000), ("ns", 1000)]

class Duration:
    def __init__(self, duration: float):
        self.duration = duration

    @staticmethod
    def format(t: float, width: int = 6):
        unit = UNIT
        t, sign = abs(t), "-" if t < 0 else " "
        if t < 1.0:
            for u, d in UNIT_DN:
                if t < 1.0:
                    t, unit = t * d, u
                else:
                    break
        else:
            for u, d in UNIT_UP:
                if t >= d:
                    t, unit = t / d, u
                else:
                    break
        t = f"{t:.02f}".rjust(width, " ")
        u = unit.ljust(2, " ")
        return f"{sign}{t} {u}"

    def __str__(self):
        return self.format(self.duration)


class Timer:

    def __init__(self, name: str = None, print: Callable = log, origin: float = None):
        self.name = name
        self.print = print
        self.origin = origin

    def __enter__(self):
        self.start = perf()

    def __exit__(self, *_, **__):
        msg = []
        if self.name:
            msg.append(self.name)
        msg.append(f"took {Duration(perf() - self.start)}")
        if self.origin is not None:
            msg.append(",")
            msg.append(f"delay {Duration(now() - self.origin)}")
        self.print(" ".join(msg))

    @classmethod
    def time(cls, name, print: Callable = print, origin: float = None):
        def decorator(func):
            def wrapper(*args, **kwargs):
                with cls(name, print=print, origin=origin):
                    return func(*args, **kwargs)

            return wrapper

        return decorator
