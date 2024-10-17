# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Generator, Callable, Iterable
from numpy import ndarray, array, float32, frombuffer
from base64 import b64encode, b64decode

from util import types
from util.env import to_device
import util.JSON as JSON
from util.env import Logger

logger = Logger(__name__)


class Protocol:
    @classmethod
    def json(
        cls, lines: Iterable[str], check: Callable = None
    ) -> Generator[list, None, None]:
        for line in lines:
            try:
                line = f"[{line}]"
                result = JSON.parse(line)
                if check is not None:
                    assert check(result), "Check failed"
                yield result
            except Exception as e:
                with open("/tmp/socket-error-content", "w") as f:
                    f.write(line)
                with open("/tmp/socket-error-exception", "w") as f:
                    f.write(str(e))
                logger.error(f"[Protocol {cls.__name__}] Failed to parse line: {e}")
                L = 128
                if len(line) > L:
                    line = line[:L] + f" ({len(line) - L} chars omitted) ..."
                logger.info(f"Line content: {line}")

    @classmethod
    def encode(cls, *args) -> Generator[str, None, None]:
        raise NotImplementedError

    @classmethod
    def decode(cls, lines: Iterable[str]) -> Generator[list, None, None]:
        raise NotImplementedError


class Perception(Protocol):
    @classmethod
    def encode(cls, ts: float, perception: ndarray):
        yield f"{ts:.04f}"
        data = perception.astype(float32)
        yield JSON.stringify(list(data.shape))
        buffer = data.tobytes(order="C")
        yield JSON.stringify(b64encode(buffer).decode())

    @classmethod
    def decode(cls, lines: Iterable[str]):
        def check(items: list):
            if len(items) != 3:
                return False
            ts, shape, b64str = items
            return (
                isinstance(ts, float)
                and isinstance(shape, list)
                and isinstance(b64str, str)
            )

        for ts, shape, b64str in cls.json(lines, check=check):
            shape = tuple(shape)
            buffer = b64decode(b64str)
            perception = frombuffer(buffer, dtype=float32).reshape(shape)
            result: types.PerceptionStamped = ts, perception
            yield result


class Correlation(Protocol):
    @classmethod
    def encode(cls, ts: float, correlation: list[list[types.Correlation]]):
        yield JSON.stringify(ts)
        yield from map(JSON.stringify, correlation)

    @classmethod
    def decode(cls, lines: Iterable[str]):
        def check(items: list):
            return len(items) > 1 and type(items[0]) is float

        for ts, *correlation in cls.json(lines, check=check):
            result: types.CorrelationStamped = ts, correlation
            yield result


class Motion(Protocol):
    @classmethod
    def encode(cls, ts: float, motion: types.Motion, msg: str = None):
        yield JSON.stringify(ts)
        yield JSON.stringify(motion)
        if msg is not None:
            yield JSON.stringify(msg)

    @classmethod
    def decode(cls, lines: Iterable[str]):
        def check(items: list):
            if len(items) == 2:
                items.append(None)
            return len(items) == 3

        for items in cls.json(lines, check=check):
            result: types.MotionStamped = items
            yield result
