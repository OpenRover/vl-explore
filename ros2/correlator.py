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

from . import Node, ros_entry, ok, spin_once
from . import socket_perception, socket_correlator
from . import protocol

from util import types
from util.sockets import Server, Client
from util.timer import Timer
from prompts import Prompt
import models.clip as clip
import util.env as env


class DataBase:
    threshold: float = 0.9

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
        # Count of points merged to each database entry
        # Length equals to the number of points in the database
        self.counts: list[int] = None

    def __len__(self):
        return self.count

    def __call__(self, x: ndarray | list):
        # x: (N, 512)
        if not isinstance(x, ndarray):
            x = array(x, dtype=self.dtype)
        familiarity: list[float] = []
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
            for n, (i, c, v) in enumerate(zip(indexes, corr, x)):
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
        else:
            self.heads = [0] * len(x)
            self.counts = []
            for n, v in enumerate(x):
                if len(self) == 0:
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
        self.counts[index] += 1
        w = 1.0 / self.counts[index]
        self.db[index] = self.db[index] * (1 - w) + x * w
        return index


def correlate(embeddings: ndarray, prompts: list[Prompt], database: DataBase):
    """
    Perform correlation between embeddings, text prompts and history db.
    """
    t = from_numpy(embeddings.copy())
    scores = list(p(t) for p in prompts)
    correlation: list[list[types.Correlation]] = [[] * len(scores)]
    for tile_id, familiarity in enumerate(database(embeddings)):
        for prompt_id, c in enumerate(correlation):
            assert len(c) == tile_id
            label, score = scores[prompt_id][tile_id]
            c.append([label, score, familiarity])
    return correlation


@ros_entry
def main():
    node = Node("correlator")
    logger = protocol.logger = node.get_logger()
    input = Client(socket_perception, logger=logger)
    output = Server(socket_correlator, logger=logger)
    database: list[DataBase] = None

    prompts = node.strategy.prompts()
    clip.text_model = None
    for prompt in prompts:
        prompt.to("cpu")

    while ok():
        spin_once(node, timeout_sec=0)
        for timestamp, embeddings in protocol.Perception.decode(input):
            if database is None:
                _, width = embeddings.shape
                dtype = embeddings.dtype
                database = DataBase(width, dtype=dtype)
            with Timer(print=logger.info, origin=timestamp):
                results = correlate(embeddings, prompts, database)
            output(*protocol.Correlation.encode(timestamp, results))
    return node