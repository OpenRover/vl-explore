# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from io import TextIOWrapper

from ..threads import protocol

class Matcher:
    t0: float = None
    d0: str = None
    t1: float = None
    d1: str = None

    decay: float = 0.5
    intervals = list[float]()

    class Outdated(Exception):
        pass

    def forward(self, init: bool = False):
        if init:
            self.t0, self.d0 = next(self.src)
            self.t1, self.d1 = next(self.src)
        else:
            self.t0, self.d0 = self.t1, self.d1
            self.t1, self.d1 = next(self.src)
        assert self.t1 >= self.t0, (self.t0, self.t1)
        self.interval = self.t1 - self.t0

    def __init__(self, src: TextIOWrapper, protocol: type[protocol.JsonProtocol]):

        def gen():
            for line in src:
                for ts, *_ in self.protocol.decode(line):
                    self.intervals.append(ts)
                    yield ts, line.strip()

        self.src = gen()
        self.protocol = protocol
        self.forward(init=True)

    def __call__(self, ts: float) -> tuple[float, str]:
        while ts >= self.t1:
            self.forward()
        if ts < self.t0:
            raise self.Outdated()
        return self.t0, self.d0, self.d1

    def freq(self, N: int = 5):
        self.intervals = self.intervals[-N:]
        if len(self.intervals) < 2:
            return 0
        dt = self.intervals[-1] - self.intervals[0]
        if dt <= 0:
            return 0
        n = len(self.intervals) - 1
        return n / dt
