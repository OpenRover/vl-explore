# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import TypeVar
from torch import Tensor
from numpy import float32, frombuffer
from base64 import b64encode, b64decode

from util import types
from util.logger import Logger
from util.sockets import JsonProtocol

logger = Logger(__name__)

T = TypeVar("T")

def type_check(item, t: type):
    if not isinstance(item, t):
        raise TypeError(f"Type check failed: {item} is {type(item)} (expected {t})")

class Perception(JsonProtocol[types.PerceptionStamped]):

    @classmethod
    def to_items(cls, item):
        ts, stddev, perception = item
        yield ts
        yield stddev
        if perception is not None:
            # Frame is valid, send embeddings
            if isinstance(perception, Tensor):
                perception = perception.cpu().numpy()
            data = perception.astype(float32)
            yield list(data.shape)
            buffer = data.tobytes(order="C")
            yield b64encode(buffer).decode()

    @classmethod
    def from_items(cls, items):
        ts, stddev, *array = items
        type_check(ts, float)
        if len(array) == 0:
            perception = None
        if len(array) == 2:
            assert len(array) == 2, array
            shape, b64str = array
            type_check(shape, list)
            type_check(b64str, str)
            shape = tuple(shape)
            buffer = b64decode(b64str)
            perception = frombuffer(buffer, dtype=float32).reshape(shape)
        result: types.PerceptionStamped = ts, stddev, perception
        yield result

class Correlation(JsonProtocol[types.CorrelationStamped]):
    @classmethod
    def to_items(cls, item):
        ts, correlation = item
        yield ts
        yield from correlation

    @classmethod
    def from_items(cls, items):
        assert len(items) > 1, items
        ts, *correlation = items
        type_check(ts, float)
        l = None
        for c in correlation:
            if l is None:
                l = len(c)
            else:
                assert len(c) == l, (l, c)
            for s, *p in c:
                type_check(s, str)
                assert len(p) == 3, p
                for v in p:
                    type_check(v, float)
        result: types.CorrelationStamped = ts, correlation
        yield result

class Motion(JsonProtocol[types.MotionStamped]):
    @classmethod
    def to_items(cls, item):
        ts, delay, motion, msg = item
        yield ts
        yield delay
        yield motion
        if msg is not None:
            yield msg

    @classmethod
    def from_items(cls, items):
        if len(items) == 2: # Backward compatibility
            items.insert(1, 0.0)
        if len(items) == 3:
            if isinstance(items[2], list):
                items.append(None)
            elif isinstance(items[2], str | None):
                items.insert(1, 0.0)
            else:
                raise TypeError(f"Invalid motion frame: {items}")
        assert len(items) == 4, items
        yield items


class Odometry(JsonProtocol[types.OdometryStamped]):
    @classmethod
    def to_items(cls, item):
        yield from item

    @classmethod
    def from_items(cls, items):
        if len(items) != 4:
            raise TypeError(f"Invalid odometry frame: {items}")
        yield items
