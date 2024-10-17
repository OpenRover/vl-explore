from numpy import ndarray
from cv2 import resize, INTER_LINEAR

from util.graphics import Region
from util.geometry import Point2i as Point


def shape(frame: ndarray):
    return Point(*frame.shape[:2][::-1])


class Slicer:
    @classmethod
    def use(cls, name: str):
        attr = f"Slicer{name}"
        slicer = globals()[attr]
        assert issubclass(slicer, cls), f"Invalid slicer: {attr}"
        return slicer

    regions: list[Region] = []

    def __init__(self, frame_size: Point, max_size: Point | int | None = None):
        self.frame_size = frame_size
        if type(max_size) is int:
            self.max_size = Point(max_size, max_size)
        elif isinstance(max_size, tuple) and len(max_size) == 2:
            self.max_size = Point(*max_size)
        else:
            self.max_size = None
        self.update_size_limit()

    def __call__(self, frame: ndarray):
        frame_size = shape(frame)
        if self.frame_size != frame_size:
            self.regions.clear()
            self.frame_size = frame_size
            self.update_size_limit()
        if self.scaled_size is not None:
            resize(frame, self.scaled_size, dst=frame, interpolation=INTER_LINEAR)
            frame_size = shape(frame)
            assert frame_size <= self.frame_size, "Frame size exceeds limit"
        if not len(self.regions):
            self.update_slice_regions()
            assert len(self.regions), "No regions to slice"
        return (r(frame) for r in self.regions)

    def update_size_limit(self):
        if self.max_size is not None:
            raw_scale = min(*(a / b for a, b in zip(self.frame_size, self.max_size)))
            if raw_scale > 1:
                self.scaled_size = self.frame_size / raw_scale
            else:
                self.scaled_size = None
        else:
            self.scaled_size = None

    def size(self):
        return self.scaled_size or self.frame_size

    # Virtual method
    def update_slice_regions(self):
        raise NotImplementedError


class Slicer2x3(Slicer):
    def update_slice_regions(self):
        size = self.size()
        w, h = size
        # Fit a 3:2 region in the frame for square boxes
        s = min(w // 3, h // 2)
        tl = (self.frame_size - Point(s * 3, s * 2)) / 2
        # Cropping regions
        self.regions = [
            # Far
            *(Region(*(tl + (x, 0)), s, s) for x in range(0, w - s + 1, s)),
            # Near
            *(Region(*(tl + (x, s)), s, s) for x in range(0, w - s + 1, s)),
        ]


class Slicer1x1(Slicer):
    def update_slice_regions(self):
        size = self.size()
        w, h = size
        # Fit a 3:2 region in the frame for square boxes
        s = min(w, h)
        tl = (size - Point(s, s)) / 2
        # Cropping regions
        self.regions = [Region(*tl, s, s)]
