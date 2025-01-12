# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
# This node performs correlation between visual perception and nav prompts.
# In addition, it maintains a "known space" database using all previous data,
# and use it to evaluate the familiarity of incoming embeddings.
# ==============================================================================
from numpy import ndarray, array, zeros
from torch import from_numpy
from time import time as now

from . import protocol
from ..utils.ros import Node, ros_entry, Count, log_to

import models.clip as clip

from util import types
from util.env import CWD, select_device, to_device
from util.sockets import SocketTransport, Server, Client
from util.timer import Timer
from util.exception import Expect
from util.queue import Queue
from util.params import STDDEV_THRESHOLD
from prompts import Prompt

class DataBase:
    threshold: float = 0.9  # Threshold for merge operation
    decay: float = 0.8  # Decay factor for rolling average merge

    def block(self) -> ndarray:
        return zeros((self.block_size, self.width), dtype=self.dtype)

    def __init__(self, width: int, block_size: int = 512, dtype=float):
        self.block_size = block_size
        self.dtype = dtype
        self.count = 0
        self.width = width
        self.db = self.block()
        # Pointers to the points last seen in the database
        # Length = tiles, each element is the index of the last seen point
        self.heads: list[int] = None

    def __len__(self):
        return self.count

    def __call__(self, x: ndarray | list, stddev: list[float]) -> list[float]:
        # x: (N, 512)
        if not isinstance(x, ndarray):
            x = array(x, dtype=self.dtype)
        N, _ = x.shape
        familiarity: list[float] = []
        stddev = [min(max((d - 0.1) * 10, 0.0), 1.0) for d in stddev]
        if len(self):
            assert self.heads is not None
            assert len(self.heads) == len(x), [len(self.heads), len(x)]
            assert self.counts is not None
            assert len(self.counts) == len(self), [len(self.counts), len(self)]
            # Calculate maximum correlation with existing points
            # (L, 512) @ (N, 512).T -> (L, N)
            corr: ndarray = self.database() @ x.T
            corr = corr.T  # (N, L)
            # (N, L) index array
            indexes = corr.argsort(axis=1)
            for n, (i, c, v, d) in enumerate(zip(indexes, corr, x, stddev)):
                if d <= STDDEV_THRESHOLD:
                    familiarity.append(0.0)
                    continue
                # Find the maximum correlation
                idx = i[-1]
                # Check if max correlation is above threshold
                if c[idx] > self.threshold:
                    # Check if the point is the last seen point
                    if self.heads[n] == idx:
                        # Exclude last seen point from familiarity
                        if len(i) > 1:
                            familiarity.append(float(c[i[-2]]))
                        else:
                            familiarity.append(0.0)
                    else:
                        familiarity.append(float(c[idx]))
                    # Merge into existing history
                    self.heads[n] = self.merge(idx, v)
                else:
                    self.heads[n] = self.append(v)
                    familiarity.append(float(c[idx]))
        else:
            self.heads = [0] * len(x)
            self.counts = []
            for n, (v, d) in enumerate(zip(x, stddev)):
                if d < 0.0:
                    pass
                elif len(self) == 0:
                    self.heads[n] = self.append(v)
                else:
                    c: ndarray = self.database() @ v
                    idx = c.argmax()
                    if c[idx] > self.threshold:
                        self.heads[n] = self.merge(idx, v)
                    else:
                        self.heads[n] = self.append(v)
                familiarity.append(0.0)
        # Return familiarity
        assert len(familiarity) == N, f"{len(familiarity)} != {N}"
        return familiarity

    def database(self):
        return self.db[: self.count]

    def append(self, x: ndarray):
        assert x.shape == (self.width,), x.shape
        self.counts.append(1)
        n = self.db.shape[0]
        i = self.count
        self.count += 1
        should_resize = False
        while i >= n:
            should_resize = True
            n += self.block_size
        if should_resize:
            self.db.resize((n, self.width))
        self.db[i] = x
        return i

    def merge(self, index: int, x: ndarray):
        assert x.shape == (self.width,), x.shape
        assert index < len(self)
        w1, w2 = self.decay, 1 - self.decay
        self.db[index] = self.db[index] * w1 + x * w2
        return index


def correlate(
    stddev: list[float],
    embeddings: ndarray | int,
    prompts: list[Prompt],
    database: DataBase,
):
    """
    Perform correlation between embeddings, text prompts and history db.
    """
    if embeddings is None:
        return [[("Invalid", -1.0, 1.0, s)] * len(prompts) for s in stddev]
    else:
        t = to_device(from_numpy(embeddings.copy()))
        scores = list(p(t) for p in prompts)
        correlation: list[list[types.Correlation]] = [[] for _ in scores]
        for tile_id, familiarity in enumerate(database(embeddings, stddev)):
            for prompt_id, c in enumerate(correlation):
                assert len(c) == tile_id
                label, score = scores[prompt_id][tile_id]
                std = stddev[tile_id]
                # Apply standard deviation threshold
                if std <= STDDEV_THRESHOLD:
                    score = -abs(score)
                    familiarity = 1.0
                c.append([label, score, familiarity, std])
        return correlation


class Correlator(Node):

    input: SocketTransport[types.PerceptionStamped]
    output: SocketTransport[types.CorrelationStamped]

    def __init__(self):
        super().__init__("correlator")
        self.mixer = self.strategy.MotionMixer()
        # I/O Sockets
        kw = dict(logger=self.get_logger())
        self.input = Client(CWD / "perception.socket", protocol.Perception, **kw)
        self.input.start()
        self.output = Server(CWD / "correlation.socket", protocol.Correlation, **kw)
        self.output.start()


@ros_entry
def main():

    node = Correlator()
    logger = protocol.logger = node.get_logger()

    prompts = [p.to("cpu") for p in node.strategy.prompts()]
    prompts.append(Prompt("target").to("cpu"))
    clip.deinit()
    select_device("cpu", reselect=True)

    input = node.input()
    ts, _, embeddings = input.get()
    _, width = embeddings.shape
    dtype = embeddings.dtype
    database = DataBase(width, dtype=dtype)

    count = Count()
    last_report = now()
    with Expect(KeyboardInterrupt, Queue.Closed):
        for timestamp, stddev, embeddings in input:
            log = []
            with Timer(
                *count("Correlate".ljust(10)),
                print=log_to(log),
                origin=timestamp,
            ):
                correlation = correlate(stddev, embeddings, prompts, database)
            node.output.send((timestamp, correlation))

            with open(CWD / "perf.log", "at") as perf:
                for l in log:
                    perf.write(l + "\n")
            if now() - last_report > 10.0:
                last_report = now()
                logger.info(f"Database has {len(database)} samples.")
    return node
