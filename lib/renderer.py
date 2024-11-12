import cv2
from numpy import ndarray, stack

from .slicer import Slicer

from util.iter import flatten
from util.geometry import Region, Point2i as Point
from util.graphics import TextBox, draw_corners
from util.math import project
from util.colors import *

proj_nav = project((-0.33, +0.33), (-1.0, 1.0), clamp=True)
proj_fam = project((1.0, 0.5), (-1.0, 1.0), clamp=True)


def colorize(score: float, c1=WHITE, c2=BLACK, light: bool = True):
    # Check for NaN
    if score != score:
        score = 0.0
    sat = min(abs(score), 1)
    color = c1 if score >= 0 else c2
    A, B = (BLACK, WHITE) if light else (WHITE, BLACK)
    fg = color * sat + A * (1 - sat)
    mg = color * sat + GRAY * (1 - sat)
    sat /= 2
    bg = color * sat + B * (1 - sat)

    return stack([bg, mg, fg], axis=0)


class Renderer:
    def __init__(self, slicer: Slicer):
        self.slicer = slicer
        w, h = slicer.size()
        h1 = int(round(h / 16))
        self.banner_region = Region(0, h - h1, w, h1)
        self.banner_t_box = TextBox(
            self.banner_region,
            ha="center",
            va="middle",
            size=self.font_scale(0.8),
            thickness=self.thickness(0.8),
        )
        self.stats_region = Region(0, 0, w, h1)
        self.stats_t_box = TextBox(
            self.stats_region,
            ha="center",
            va="middle",
            size=self.font_scale(0.8),
            thickness=self.thickness(0.8),
        )

    def __call__(
        self,
        frame: np.ndarray,
        pred: list[list[tuple[str, float, float]]],
    ):
        raise NotImplementedError

    def regions(self):
        return self.slicer.regions

    def font_scale(self, factor: float = 1.0):
        # Base font scale for UI elements
        return factor * min(*(self.slicer.size())) / 1000

    def thickness(self, factor: float = 1.0):
        # Base thickness for UI elements
        return int(round(factor * min(*(self.slicer.size())) / 600))

    def region(
        self,
        frame: ndarray,
        pred: tuple[str, float, float],
        t_box: TextBox = None,
        background: Region = None,
        corners: Region = None,
    ):
        text, navigability, familiarity, stddev = pred

        if background is not None:
            light = background(frame).mean() > 64
        else:
            light = True

        if abs(navigability) <= 1e-4 or stddev < 0.1:
            # Region is invalid, render in grayscale
            bg, mg, fg = u8(colorize(0, GRAY, GRAY, light=light))
        else:
            bg, mg, fg = u8(
                mix(
                    colorize(proj_nav(navigability), GREEN, RED, light=light),
                    colorize(proj_fam(familiarity), BLUE, RED, light=light),
                )
            )

        if background is not None:
            rect = np.ones((*background.shape, 3))
            rect: ndarray = background(frame) * 0.6 + rect * bg * 0.2
            rect = np.rint(rect).astype(np.uint8)
            frame[background.slice_y, background.slice_x] = rect
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
                (f"STDDEV {stddev:.2f}", "top"),
                (text, "middle"),
                (f"N {navigability:.2f} | F {familiarity:.2f}", "bottom"),
            ):
                t_box(
                    frame,
                    t,
                    size=self.font_scale(),
                    color=fg,
                    thickness=self.thickness(),
                    font=cv2.FONT_HERSHEY_DUPLEX,
                    ha="center",
                    va=va,
                    line_height=1.5,
                    dpi_scale=2.0,
                )

    def banner(
        self,
        frame: ndarray,
        text: str,
        color=(255, 128, 0),
        blurred: ndarray | None = None,
    ):
        t, r = self.banner_t_box, self.banner_region
        bg = frame if blurred is None else blurred
        frame[r.slice_y, r.slice_x] = r(bg) * 0.6
        t(frame, text, color=color)

    def stats(
        self,
        frame: ndarray,
        text: str,
        color=(255, 255, 255),
        blurred: ndarray | None = None,
    ):
        t, r = self.stats_t_box, self.stats_region
        bg = frame if blurred is None else blurred
        frame[r.slice_y, r.slice_x] = r(bg) * 0.6
        t(frame, text, color=color)

    def blur(self, frame: np.ndarray):
        sigma = min(*(frame.shape)) / 100
        return cv2.GaussianBlur(frame, (0, 0), sigma)


class Renderer6T1P(Renderer):

    def __call__(
        self,
        frame: np.ndarray,
        pred: list[list[tuple[str, float, float]]],
    ):
        pred = flatten(pred, 2)
        offset = self.thickness() * 8
        for p, r in zip(pred, self.slicer.regions):
            c_box = r.offset(Point(-offset, -offset))
            t_box = TextBox(c_box.offset(Point(-offset, -offset)))
            self.region(frame, p, t_box=t_box, background=c_box, corners=c_box)
