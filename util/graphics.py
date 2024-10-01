# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import sys, cv2, numpy as np
from .geometry import Region
from enum import Enum


def draw_corners(
    frame,
    r: Region,
    length: int = 10,
    color=(0, 0, 0),
    thickness=1,
    line_type=cv2.LINE_AA,
):
    for (x, y), dx, dy in r.corners():
        for p2 in [(x + dx * length, y), (x, y + dy * length)]:
            cv2.line(
                frame,
                (x, y),
                p2,
                color=color,
                thickness=thickness,
                lineType=line_type,
            )


class TextBox:
    box: Region
    align: str = "left"
    vertical_align: str = "middle"
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 1
    color = (255, 255, 255)
    # Typesetting properties
    line_height = 1.0  # relative to font size
    # Fitting
    shrink = True  # shrink text to fit

    def __init__(self, box: Region, **kwargs):
        self.box = box
        for key, value in kwargs.items():
            setattr(self, key, value)

    def line_size(self, text, scale=None):
        if scale is None:
            scale = self.scale
        return cv2.getTextSize(text, self.font, scale, self.thickness)

    def tokens(self, text: str):
        tokens = []
        for line in text.split("\n"):
            tokens.extend(line.split())
            tokens.append("\n")
        if len(tokens) > 0 and tokens[-1] == "\n":
            tokens.pop()
        return tokens

    def fit(self, text: str, scale: float = None):
        if scale is None:
            scale = self.scale
        if scale <= 0:
            raise ValueError(f"invalid scale ({scale})")
        tokens = self.tokens(text)
        lines: list[tuple[int, list[str]]] = []
        y = 0
        flag_abort = False
        while len(tokens) > 0:
            line = []
            line_width = 0
            while len(tokens) > 0:
                next = tokens[0]
                if next == "\n":
                    # New line requested
                    tokens.pop(0)
                    break
                (w, h), b = self.line_size(" ".join(line + [next]), scale)
                org_y = y + h
                h += b
                if w > self.box.w:
                    if len(line) > 0:
                        # line filled, go to next line
                        break
                    if self.shrink:
                        # single word is too long
                        return self.fit(next, scale * self.box.w / w)
                    print("unable to fit all text", file=sys.stderr)
                    flag_abort = True
                    break
                elif y + h * self.line_height > self.box.h:
                    # vertical overflow
                    return self.fit(text, scale * 0.9)
                else:
                    # add word to line
                    line.append(tokens.pop(0))
                    line_width = w
            if flag_abort:
                break
            match self.align.lower():
                case "left":
                    org_x = 0
                case "center":
                    org_x = (self.box.w - line_width) // 2
                case "right":
                    org_x = self.box.w - line_width
                case _:
                    raise ValueError(f'invalid align option "{self.align}"')
            lines.append((org_x, org_y, line))
            y = int(y + h * self.line_height)

        match self.vertical_align.lower():
            case "top":
                offset_y = 0
            case "middle":
                offset_y = (self.box.h - y) // 2
            case "bottom":
                offset_y = self.box.h - y
            case _:
                raise ValueError(f'invalid vertical_align "{self.vertical_align}"')

        def render(mat):
            x, y = self.box.tl
            for dx, dy, line in lines:
                text = " ".join(line)
                cv2.putText(
                    mat,
                    text,
                    (x + dx, y + dy + offset_y),
                    self.font,
                    scale,
                    self.color,
                    self.thickness,
                )
            return mat

        return render

    def __call__(self, mat: np.ndarray, text: str, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self.fit(text)(mat)

    def __repr__(self):
        return f"TextBox({self.box}, {self.vertical_align})"
