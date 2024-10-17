# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Type
import models.clip as clip, cv2, numpy as np
from prompts import Prompt
from lib.slicer import Slicer, Slicer2x3, Slicer1x1
from util.geometry import Region, Point2i as Point
from util.graphics import TextBox, draw_corners
from util.env import Logger
from util.iter import flatten

log = Logger(__file__)


BLUE = np.array([1.0, 0.0, 0.0], dtype=np.float64)
GREEN = np.array([0.0, 1.0, 0.0], dtype=np.float64)
RED = np.array([0.0, 0.0, 1.0], dtype=np.float64)
GRAY = np.array([0.5, 0.5, 0.5], dtype=np.float64)
WHITE = np.ones(3, dtype=np.float64)
BLACK = np.zeros(3, dtype=np.float64)


def colorize(score: float):
    # Check for NaN
    if score != score:
        score = 0
    sat = min(3 * abs(score), 1)
    color = GREEN if score >= 0 else RED
    fg = color * sat + WHITE * (1 - sat)
    mg = color * sat + GRAY * (1 - sat)
    sat /= 2
    bg = color * sat + BLACK * (1 - sat)

    def _(c):
        return list(map(int, c * 255))

    return _(bg), _(mg), _(fg)


def use(name: str):
    attr = f"Nav{name}"
    nav = globals()[attr]
    assert issubclass(nav, Navigation), f"Invalid use case: {attr}"
    return nav


class Navigation:

    SlicerType: Type[Slicer]

    def __init__(
        self,
        frame_size: Point,
        size_limit: Point | int | None,
        prompts: list[Prompt],
        decay: float = 0.5,
    ):
        self.slicer = self.SlicerType(frame_size, size_limit)
        self.prompts = prompts
        self.decay = decay

    def font_scale(self):
        # Base font scale for UI elements
        return min(*(self.slicer.size())) / 800

    def thickness(self):
        # Base thickness for UI elements
        return int(round(2.0 * min(*(self.slicer.size())) / 800))

    # confidence[prompt_id][tile_id]
    confidence: list[list[float]] = None

    # returns list result[prompt_id][tile_id] = (text, score, confidence)
    def __call__(self, frame: np.ndarray):
        embeddings = clip.encode_image(*self.slicer(frame))
        scores = list(p(embeddings) for p in self.prompts)
        # Initialize confidence to zero
        if self.confidence is None:
            self.confidence = list([0.0] * len(s) for s in scores)
        # update confidence - running average
        confidence: list[list[float]] = []
        results: list[list[tuple[str, float, float]]] = []
        for _s, _c in zip(scores, self.confidence):
            result: list[tuple[str, float, float]] = []
            for i, ((_, s), c) in enumerate(zip(_s, _c)):
                _c[i] = c * self.decay + s * (1 - self.decay)
                result.append(_, s, _c[i])
            confidence.append(_c.copy())
            results.append(result)
        return results

    def render_region(
        self,
        frame: np.ndarray,
        pred: tuple[str, float],
        conf: float,
        t_box: TextBox = None,
        background: Region = None,
        corners: Region = None,
    ):
        text, score = pred
        bg, mg, fg = colorize(score)
        if background is not None:
            rect = np.ones((*background.shape, 3))
            rect = background(frame) * 0.6 + rect * bg * 0.2
            frame[background.slice_y, background.slice_x] = rect.astype(np.uint8)
        if corners is not None:
            draw_corners(
                frame,
                corners,
                length=corners.shape[0] // 8,
                color=mg,
                thickness=self.thickness() * 2,
            )
        if t_box is not None:
            for t, va in (
                (text, "middle"),
                (f"{score:.2f} | {conf:.2f}", "bottom"),
            ):
                t_box(
                    frame,
                    t,
                    scale=self.font_scale(),
                    color=fg,
                    thickness=self.thickness(),
                    font=cv2.FONT_HERSHEY_DUPLEX,
                    align="center",
                    vertical_align=va,
                    line_height=1.5,
                    dpi_scale=2.0,
                )

    def render(
        self,
        frame: np.ndarray,
        pred: list[list[tuple[str, float]]],
        confidence: list[list[float]],
    ):
        """Render the navigation UI on the frame"""
        pred = flatten(pred, 2)
        confidence = flatten(confidence, 2)
        for p, c, r in zip(pred, confidence, self.slicer.regions):
            offset = self.thickness() * 8
            c_box = r.offset(Point(-offset, -offset))
            t_box = TextBox(c_box.offset(Point(-offset, -offset)))
            self.render_region(
                frame, p, c, t_box=t_box, background=c_box, corners=c_box
            )


class Nav6T1P(Navigation):

    SlicerType = Slicer2x3

    def __init__(
        self,
        prompts: list[Prompt],
        frame_size: tuple[int, int],
        size_limit: tuple[int, int] | int | None = None,
        **kwargs,
    ):
        super().__init__(prompts, frame_size, size_limit, **kwargs)


class Nav1T3P(Navigation):

    SlicerType = Slicer1x1

    def __init__(
        self,
        prompts: list[Prompt],
        frame_size: tuple[int, int],
        size_limit: tuple[int, int] | int | None = None,
        **kwargs,
    ):
        super().__init__(prompts, frame_size, size_limit, **kwargs)

    def render(
        self,
        frame: np.ndarray,
        pred: list[list[tuple[str, float]]],
        confidence: list[list[float]],
    ):
        region = self.regions[0]
        vec = Point(region.w // 3, 0)
        offset = self.thickness() * 8
        offset = Point(-offset, -offset)
        pl, pc, pr = flatten(pred, 2)
        cl, cc, cr = flatten(confidence, 2)
        rl, rc, rr = [
            Region(region.tl + vec * i, region.bl + vec * j)
            for i, j in zip(range(3), range(1, 4))
        ]
        self.render_region(frame, pl, cl, TextBox(rl.offset(offset)), rl)
        self.render_region(frame, pc, cc, TextBox(rc.offset(offset)), rc, region)
        self.render_region(frame, pr, cr, TextBox(rr.offset(offset)), rr)
