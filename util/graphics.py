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
    dpi_scale: float = 2.0
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 1
    color = (255, 255, 255)
    opacity: float = 1.0
    # Typesetting properties
    line_height = 1.0  # relative to font size
    # Fitting
    shrink = True  # shrink text to fit

    def __init__(self, box: Region, **kwargs):
        self.box = box
        for key, value in kwargs.items():
            setattr(self, key, value)

    def line_size(self, text, thickness: int, scale=None):
        if scale is None:
            scale = self.scale
        return cv2.getTextSize(text, self.font, scale, thickness)

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
        shape = self.box.shape * self.dpi_scale
        h_lim, w_lim = shape
        fs = scale * self.dpi_scale
        thickness = int(round(self.thickness * self.dpi_scale))
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
                (w, h), b = self.line_size(" ".join(line + [next]), thickness, fs)
                org_y = y + h
                h += b
                if w > w_lim:
                    if len(line) > 0:
                        # line filled, go to next line
                        break
                    if self.shrink:
                        # single word is too long
                        return self.fit(next, scale * w_lim / w)
                    print("unable to fit all text", file=sys.stderr)
                    flag_abort = True
                    break
                elif y + h * self.line_height > h_lim:
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
                    org_x = (w_lim - line_width) // 2
                case "right":
                    org_x = w_lim - line_width
                case _:
                    raise ValueError(f'invalid align option "{self.align}"')
            lines.append((org_x, org_y, line))
            y = int(y + h * self.line_height)

        match self.vertical_align.lower():
            case "top":
                offset_y = 0
            case "middle":
                offset_y = (h_lim - y) // 2
            case "bottom":
                offset_y = h_lim - y
            case _:
                raise ValueError(f'invalid vertical_align "{self.vertical_align}"')

        def render(mat):
            color = np.array(self.color)
            mask = np.zeros(shape, dtype=np.uint8)
            opacity = self.opacity
            val = 255 if opacity >= 1.0 else int(round(255 * opacity))
            for x, y, line in lines:
                text = " ".join(line)
                anchor = (x, y + offset_y)
                cv2.putText(mask, text, anchor, self.font, fs, [val], thickness)
            if shape != self.box.shape:
                mask = cv2.resize(mask, self.box.shape[::-1], interpolation=cv2.INTER_LINEAR)
            mask = mask.astype(np.float32) / 255.0
            mask = np.stack([mask] * 3, axis=-1)
            # apply mask
            mat[self.box.slice_y, self.box.slice_x] = np.rint(
                self.box(mat) * (np.ones_like(mask) - mask) + mask * color
            ).astype(np.uint8)
            return mat

        return render

    def __call__(self, mat: np.ndarray, text: str, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self.fit(text)(mat)

    def __repr__(self):
        return f"TextBox({self.box}, {self.vertical_align})"
