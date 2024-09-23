# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import cv2, sys
from .region import Region
from enum import Enum

FONT = cv2.FONT_HERSHEY_SIMPLEX


class TextBox:
    box: Region
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
        return text.split()

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
            while len(tokens) > 0:
                next = tokens[0]
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
                    return self.fit(text, scale * 0.98)
                line.append(tokens.pop(0))
            if flag_abort:
                break
            lines.append((org_y, line))
            y = int(y + h * self.line_height)

        match self.vertical_align.lower():
            case "top":
                offset_y = 0
            case "middle":
                offset_y = (self.box.h - y) // 2
            case "bottom":
                offset_y = self.box.h - y
            case _:
                raise ValueError(f"invalid vertical_align ({self.vertical_align})")

        def render(mat):
            x, y = self.box.tl
            for dy, line in lines:
                text = " ".join(line)
                cv2.putText(
                    mat,
                    text,
                    (x, y + dy + offset_y),
                    self.font,
                    scale,
                    self.color,
                    self.thickness,
                )
            return mat

        return render

    def __call__(self, mat: cv2.Mat, text: str, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self.fit(text)(mat)
