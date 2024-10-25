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
        ts, perception = item
        yield ts
        if isinstance(perception, Tensor):
            perception = perception.cpu().numpy()
        data = perception.astype(float32)
        yield list(data.shape)
        buffer = data.tobytes(order="C")
        yield b64encode(buffer).decode()

    @classmethod
    def from_items(cls, items):
        if len(items) != 3:
            return False
        ts, shape, b64str = items
        type_check(ts, float)
        type_check(shape, list)
        type_check(b64str, str)
        shape = tuple(shape)
        buffer = b64decode(b64str)
        perception = frombuffer(buffer, dtype=float32).reshape(shape)
        result: types.PerceptionStamped = ts, perception
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
        result: types.CorrelationStamped = ts, correlation
        yield result

class Motion(JsonProtocol[types.MotionStamped]):
    @classmethod
    def to_items(cls, item):
        ts, delay, motion, msg = item
        yield ts
        yield motion
        if msg is not None:
            yield msg

    @classmethod
    def from_items(cls, items):
        if len(items) == 2:
            items.append(None)
        assert len(items) == 3, items
        yield items
