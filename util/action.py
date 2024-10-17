# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Generator, Callable, Any


class Action:
    def __init__(
        self,
        name: str,
        generator: Generator[list[float] | None, float, Generator | None],
    ):
        self.name = name
        self.generator = generator
        self.attrs = next(self.generator) or {}
        assert isinstance(self.attrs, dict)

    def __call__(self, confidence: float):
        try:
            return self.generator.send(confidence), False
        except StopIteration as e:
            return e.value, True
    
    def __getitem__(self, key: str):
        return self.attrs[key]

    def __contains__(self, key: str):
        return key in self.attrs

    @classmethod
    def action(
        cls,
        fn: Callable[[Any], Generator[float, float, Generator | None]],
    ):
        def decorator(*args, **kwargs):
            return cls(fn.__name__, fn(*args, **kwargs))

        return decorator
