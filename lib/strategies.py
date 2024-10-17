# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Type
from prompts import Prompt
from lib.slicer import Slicer, Slicer2x3, Slicer1x1
from lib.motion_mixer import MotionMixer, MotionMixer2x3, MotionMixer1x3


class Strategy:
    Slicer: Type[Slicer]
    MotionMixer: Type[MotionMixer]

    @classmethod
    def prompts(cls) -> list[Prompt]:
        raise NotImplementedError


class Strategy6T1P(Strategy):
    Slicer = Slicer2x3
    MotionMixer = MotionMixer2x3

    @classmethod
    def prompts(cls) -> list[Prompt]:
        return [Prompt("navigation")]


class Strategy1T3P(Strategy):
    Slicer = Slicer1x1
    MotionMixer = MotionMixer1x3

    @classmethod
    def prompts(cls) -> list[Prompt]:
        return [Prompt("left"), Prompt("center"), Prompt("right")]


class Strategy1T6P(Strategy):
    Slicer = Slicer1x1
    MotionMixer = MotionMixer2x3

    @classmethod
    def prompts(cls) -> list[Prompt]:
        return [
            Prompt("tl"),
            Prompt("tc"),
            Prompt("tr"),
            Prompt("bl"),
            Prompt("bc"),
            Prompt("br"),
        ]


def use(strategy: str) -> Type[Strategy]:
    g = globals()
    attr = f"Strategy{strategy}"
    assert attr in g, f"Invalid strategy: {strategy}"
    return g[attr]
