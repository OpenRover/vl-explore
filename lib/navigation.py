# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import models.clip as clip, cv2, numpy as np
from prompts import Prompt
from util.geometry import Region, Point2i as Point
from util.graphics import TextBox, draw_corners
from util.env import Logger

log = Logger(__file__)


RED = np.array([0.0, 0.0, 0.8], dtype=np.float64)
GREEN = np.array([0.0, 0.8, 0.0], dtype=np.float64)
GRAY = np.array([0.5, 0.5, 0.5], dtype=np.float64)


class Navigation:
    def __init__(
        self,
        prompt: Prompt,
        frame_size: tuple[int, int],
        size_limit: tuple[int, int] | int | None = None,
        padding: float = 0.05,
        decay: float = 0.5,
    ):
        self.prompt = prompt
        # Running average decay factor for confidence
        self.decay = decay
        # Size of raw camera frame
        w, h = frame_size
        # Apply optional size limit
        size_limit = size_limit or frame_size
        if isinstance(size_limit, int):
            size_limit = (size_limit, size_limit)
        max_w, max_h = size_limit
        raw_scale = min(w / max_w, h / max_h)
        if raw_scale > 1:
            w = int(w / raw_scale)
            h = int(h / raw_scale)
        self.frame_size = Point(w, h)
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

    confidence: tuple[float, ...] = None

    def __call__(self, frame: np.ndarray):
        if frame.shape[:2] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        embeddings = clip.encode_image(*[r(frame) for r in self.regions])
        scores = self.prompt(embeddings)
        # update confidence - running average
        if self.confidence is None:
            self.confidence = [s for _, s in scores]
        else:
            self.confidence = [
                self.decay * c + (1.0 - self.decay) * s
                for c, (_, s) in zip(self.confidence, scores)
            ]
        return scores, self.confidence, frame

    dpi_scale = 2.0

    def render(self, frame: np.ndarray, pred: list[tuple[str, float]]):
        """Render navigation UI on the frame"""
        scale = self.dpi_scale
        size = self.frame_size * scale
        thickness = int(round(2.0 * min(*size) / 1000))
        font_scale = 1.0 * min(*size) / 1000
        canvas = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        regions = list(r * scale for r in self.regions)
        t_boxes = list(
            TextBox(
                r.scale(0.8),
                scale=font_scale,
                thickness=thickness,
                align="center",
                vertical_align="middle",
                line_height=1.2,
            )
            for r in regions
        )

        for (text, score), region, t_box, confidence in zip(
            pred, regions, t_boxes, self.confidence
        ):
            sat = min(abs(score) / 0.5, 1)
            color = GREEN if score >= 0 else RED
            color = color * sat + GRAY * (1 - sat)
            color = list(map(int, color * 255))
            draw_corners(
                canvas,
                region.offset(Point(-thickness, -thickness) * 4),
                length=region.shape[0] // 8,
                color=color,
                thickness=thickness * 2,
            )
            t_box(canvas, f"{text}\n{score:.2f} | {confidence:.2f}", color=color)
        # Scale down to original DPI
        return cv2.resize(canvas, self.frame_size, interpolation=cv2.INTER_LINEAR)
