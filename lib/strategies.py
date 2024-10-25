# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Type, Generator
from prompts import Prompt
from lib.slicer import Slicer, Slicer2x3, Slicer1x1
from lib.renderer import Renderer, Renderer6T1P
from lib.motion_mixer import MotionMixer, MotionMixer2x3, MotionMixer1x3


class Strategy:
    Slicer: Type[Slicer]
    MotionMixer: Type[MotionMixer]
    Renderer: Type[Renderer]

    @staticmethod
    def prompts() -> Generator[Prompt, None, None]:
        raise NotImplementedError


class Strategy6T1P(Strategy):
    Slicer = Slicer2x3
    MotionMixer = MotionMixer2x3
    Renderer = Renderer6T1P

    @staticmethod
    def prompts():
        yield Prompt("navigation")


class Strategy1T3P(Strategy):
    Slicer = Slicer1x1
    MotionMixer = MotionMixer1x3
    Renderer = Renderer

    @staticmethod
    def prompts():
        yield from map(Prompt, "left center right".split())


class Strategy1T6P(Strategy):
    Slicer = Slicer1x1
    MotionMixer = MotionMixer2x3
    Renderer = Renderer

    @staticmethod
    def prompts():
        yield from map(Prompt, "tl tc tr bl bc br".split())


def use(strategy: str) -> Type[Strategy]:
    g = globals()
    attr = f"Strategy{strategy}"
    assert attr in g, f"Invalid strategy: {strategy}"
    return g[attr]
