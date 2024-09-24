# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import models.clip as clip, cv2, numpy as np
from prompts import Prompt
from util.region import Region
from util.graphics import TextBox, draw_corners
from util.env import Logger

log = Logger(__file__)


RED = np.array([0.0, 0.0, 0.8], dtype=np.float64)
GREEN = np.array([0.0, 0.8, 0.0], dtype=np.float64)
GRAY = np.array([0.5, 0.5, 0.5], dtype=np.float64)


class Navigation:
    regions: list[Region]
    arrows: list[tuple[tuple[int, int], tuple[int, int]]]
    t_boxes: list[TextBox]

    def __init__(
        self,
        prompt: Prompt,
        frame_size: tuple[int, int],
        size_limit: tuple[int, int] | int | None = None,
        padding: float = 0.05,
        decay: float = 0.9,
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
        self.frame_size = w, h
        # Box size
        s = self.s = min(h, w // 3)
        t = (h - s) // 2
        # Cropping regions
        self.regions = L, C, R = [Region(x, t, s, s) for x in range(0, w - s + 1, s)]
        # UI elements - arrows
        delta = [(1, 0), (0, 1), (-1, 0)]
        pad = int(s * padding)
        p12 = int(s * 0.2) + pad
        self.arrows = [
            [
                (x + dx * p12, y + dy * p12),  # arrow tail
                (x + dx * pad, y + dy * pad),  # arrow head
            ]
            for (x, y), (dx, dy) in zip([L.ml, (w // 2, 0), R.mr], delta)
        ]
        # UI elements - text boxes
        tb_size = int(s * 0.5)
        self.thickness = font_size = max(1.0, tb_size / 100)
        # Offset from p2 to center of text box
        p2tc = tb_size // 2 + pad
        self.t_boxes = [
            TextBox(
                Region(
                    x + dx * p2tc,
                    y + dy * p2tc,
                    tb_size,
                    tb_size,
                    anchor="center",
                ),
                vertical_align=vertical_align,
                font_size=int(round(font_size)),
                thickness=self.thickness,
            )
            for ((x, y), _), (dx, dy), vertical_align in zip(
                self.arrows, delta, ("middle", "top", "middle")
            )
        ]

    decay: float = 0.9
    confidence: tuple[float, float, float] = [0, 0, 0]

    def __call__(self, frame: np.ndarray):
        if frame.shape[:2] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        embeddings = clip.encode_image(*[r(frame) for r in self.regions])
        scores = self.prompt(embeddings)
        # update confidence - running average
        self.confidence = [
            self.decay * c + (1.0 - self.decay) * s
            for c, (_, s) in zip(self.confidence, scores)
        ]
        return scores, self.confidence, frame

    def render(self, frame: np.ndarray, pred: list[tuple[str, float]]):
        L, C, R = self.regions
        for (text, score), region, arrow, t_box in zip(
            pred, self.regions, self.arrows, self.t_boxes
        ):
            sat = min(abs(score) / 0.5, 1)
            color = GREEN if score >= 0 else RED
            color = color * sat + GRAY * (1 - sat)
            color = list(map(int, color * 255))
            if region is C:
                draw_corners(
                    frame, region, length=self.s // 8, color=color, thickness=1
                )
            cv2.arrowedLine(
                frame,
                arrow[0],
                arrow[1],
                color=color,
                thickness=1,
                line_type=cv2.LINE_AA,
            )
            t_box(frame, text, color=color)
        return frame
