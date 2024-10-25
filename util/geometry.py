# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import overload, Callable, Iterable, TypeVar
from math import sqrt
from numpy import ndarray


def i(x: int | float):
    return int(round(x))


def f(x: int | float):
    return float(x)


class Point(tuple):
    def zip(self, other: int | float | tuple[int | float]):
        if isinstance(other, Iterable):
            assert len(self) == len(other), "dimension mismatch"
            other = self.__class__(*other)
        else:
            other = self.__class__(other)
        return zip(self, other)

    def __new__(cls, san: Callable, *args):
        if len(args) == 1:
            args = args * 2
        elif len(args) > 2:
            raise ValueError(f"invalid arguments: {args}")
        return tuple.__new__(cls, tuple(map(san, args)))

    def __add__(self, other):
        return self.__class__(*[s + o for s, o in self.zip(other)])

    def __sub__(self, other):
        return self.__class__(*[s - o for s, o in self.zip(other)])

    def __mul__(self, other: float):
        return self.__class__(*[(s * o) for s, o in self.zip(other)])

    def __truediv__(self, other: float):
        return self.__class__(*[(s / o) for s, o in self.zip(other)])

    def __floordiv__(self, other: float):
        return self.__class__(*[(s // o) for s, o in self.zip(other)])

    def __mod__(self, other: float):
        return self.__class__(*[(s % o) for s, o in self.zip(other)])

    def __ge__(self, other):
        return all(s >= o for s, o in self.zip(other))

    def __gt__(self, other):
        return all(s > o for s, o in self.zip(other))

    def __le__(self, other):
        return all(s <= o for s, o in self.zip(other))

    def __lt__(self, other):
        return all(s < o for s, o in self.zip(other))

    def __pow__(self, other):
        return self.__class__(*[(s**o) for s, o in self.zip(other)])

    def norm(self):
        return sqrt(sum(v**2 for v in self))


class Point2i(Point):
    def __new__(cls, *args):
        return Point.__new__(cls, i, *args)


class Point2f(Point):
    def __new__(cls, *args):
        return Point.__new__(cls, f, *args)


class Vector2i(tuple[Point2i, Point2i]):
    def __new__(cls, p1, p2):
        return tuple.__new__(cls, (p1, p2))

    def __add__(self, other):
        return Vector2i(*[a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        return Vector2i(*[a - b for a, b in zip(self, other)])

    def __mul__(self, scale: float):
        return Vector2i(*[v * scale for v in self])


ArrayLike = TypeVar("ArrayLike")


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

    def __call__(self, frame: ArrayLike) -> ArrayLike:
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
