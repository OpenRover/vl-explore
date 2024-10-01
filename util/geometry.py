# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import overload
from numpy import ndarray


def i(x: int | float):
    return int(round(x))


class Point2i(tuple[int, int]):
    def __new__(cls, x, y):
        return tuple.__new__(cls, (i(x), i(y)))

    def __add__(self, other):
        if type(other) is int or type(other) is float:
            return Point2i(*[a + other for a in self])
        return Point2i(*[a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        return Point2i(*[a - b for a, b in zip(self, other)])

    def __mul__(self, scale: float):
        return Point2i(*[i(v * scale) for v in self])

    def __truediv__(self, scale: float):
        return Point2i(*[i(v / scale) for v in self])

    def __floordiv__(self, scale: float):
        return Point2i(*[i(v // scale) for v in self])

    def __mod__(self, scale: float):
        return Point2i(*[i(v % scale) for v in self])


class Vector2i(tuple[Point2i, Point2i]):
    def __new__(cls, p1, p2):
        return tuple.__new__(cls, (p1, p2))

    def __add__(self, other):
        return Vector2i(*[a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        return Vector2i(*[a - b for a, b in zip(self, other)])

    def __mul__(self, scale: float):
        return Vector2i(*[v * scale for v in self])


class Region:
    """
    Abstraction of a rectangular region in an image.
    """

    @overload
    def __init__(self, x: int, y: int, w: int, h: int, anchor: str = "corner"): ...

    @overload
    def __init__(self, p1: tuple[int, int], p2: tuple[int, int]): ...

    def __init__(self, *args, anchor: str = "corner"):
        if len(args) == 2:
            p1 = Point2i(*args[0])
            p2 = Point2i(*args[1])
            (x, y), (w, h) = p1, p2 - p1
        elif len(args) == 4:
            x, y, w, h = map(i, args)
        else:
            raise ValueError(f"invalid arguments: {args}")
        self.h, self.w = self.shape = Point2i(abs(h), abs(w))
        if anchor == "corner":
            x1, xm, x2 = sorted([x, x + w // 2, x + w])
            y1, ym, y2 = sorted([y, y + h // 2, y + h])
        elif anchor == "center":
            x1, xm, x2 = sorted([x, x + w // 2, x - w // 2])
            y1, ym, y2 = sorted([y, y + h // 2, y - h // 2])
        else:
            raise ValueError(f"invalid anchor type: {anchor}")
        # Initialize slice regions
        self.tl = Point2i(x1, y1)
        self.tc = Point2i(xm, y1)
        self.tr = Point2i(x2, y1)
        self.ml = Point2i(x1, ym)
        self.mc = Point2i(xm, ym)
        self.mr = Point2i(x2, ym)
        self.bl = Point2i(x1, y2)
        self.bc = Point2i(xm, y2)
        self.br = Point2i(x2, y2)
        self.slice_x = slice(x1, x2)
        self.slice_y = slice(y1, y2)

    def __call__(self, frame: ndarray) -> ndarray:
        return frame[self.slice_y, self.slice_x]

    def corners(self):
        yield self.tl, 1, 1
        yield self.tr, -1, 1
        yield self.bl, 1, -1
        yield self.br, -1, -1

    def scale(self, ratio: float, anchor: str = "mc"):
        """Scale the region by a ratio, only center anchor is currently supported"""
        if anchor != "mc":
            raise ValueError(f"invalid anchor: {anchor}")
        shape = self.shape * ratio
        return Region(*self.mc, *shape, anchor="center")

    def offset(self, v: Point2i):
        """
        Offset the region by a vector.
        Positive means outward (expand), negative means inward (shrink).
        """
        return Region(self.tl - v, self.br + v)

    def __mul__(self, scale: float):
        """DPI Scale, all numbers are multiplied"""
        x, y = self.tl
        w, h = self.w, self.h
        return Region(*[i * scale for i in (x, y, w, h)])

    def __str__(self):
        return f"Region: {self.tl}:{self.br}"
