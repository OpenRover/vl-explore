# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import TypeVar

from util.queue import Queue, deque
from util.iter import first_item

from . import transports as TP


T = TypeVar("T")


class Outdated(Exception):
    pass


def seek(arr: deque[T], ts: float):
    assert len(arr) > 0
    t: tuple[float] = first_item(zip(*arr))
    for t0, t1 in zip(t, t[1:]):
        if t0 <= ts < t1:
            return arr[0]
        elif t1 < ts:
            arr.popleft()
        elif t0 > ts:
            raise Outdated()
        else:
            return None

@Queue.Loop()
def mux(
    q0: Queue[TP.ImageMsg],
    q1: Queue[TP.CorrelationMsg],
    q2: Queue[TP.TextMsg],
    q3: Queue[TP.StatsMsg],
    out: Queue[TP.RenderMsg],
):
    l0: deque[TP.ImageMsg] = deque()
    l1: deque[TP.CorrelationMsg] = deque()
    l2: deque[TP.TextMsg] = deque()
    l3: deque[TP.StatsMsg] = deque()

    queues = (q0, q1, q2, q3)
    lists = (l0, l1, l2, l3)

    while True:
        for q, l in zip(queues, lists):
            q(l.append)
        while all(lists):
            t = l0[0][0]
            try:
                u1 = seek(l1, t)
                u2 = seek(l2, t)
                u3 = seek(l3, t)
            except Outdated:
                l0.popleft()
                continue
            if u1 and u2 and u3:
                _, frame = l0.popleft()
                _, _, correlation = u1
                _, action = u2
                _, stats = u3
                out.put((frame, correlation, action, stats))
