# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from numpy import ndarray


class Region:
    """
    Abstraction of a rectangular region in an image.
    """

    w: int
    h: int
    # Region slices
    slice_x: slice
    slice_y: slice
    # Convenience attributes for OpenCV functions
    tl: tuple[int, int]
    tc: tuple[int, int]
    tr: tuple[int, int]
    ml: tuple[int, int]
    mc: tuple[int, int]
    mr: tuple[int, int]
    bl: tuple[int, int]
    bc: tuple[int, int]
    br: tuple[int, int]

    def __init__(self, x, y, w, h):
        self.w = abs(w)
        self.h = abs(h)
        x1, xm, x2 = sorted([x, x + w // 2, x + w])
        y1, ym, y2 = sorted([y, y + h // 2, y + h])
        # Initialize slice regions
        self.tl = (x1, y1)
        self.tc = (xm, y1)
        self.tr = (x2, y1)
        self.ml = (x1, ym)
        self.mc = (xm, ym)
        self.mr = (x2, ym)
        self.bl = (x1, y2)
        self.bc = (xm, y2)
        self.br = (x2, y2)
        self.slice_x = slice(x1, x2)
        self.slice_y = slice(y1, y2)

    def __call__(self, frame: ndarray) -> ndarray:
        return frame[self.slice_y, self.slice_x]

    def corners(self):
        yield self.tl, 1, 1
        yield self.tr, -1, 1
        yield self.bl, 1, -1
        yield self.br, -1, -1
