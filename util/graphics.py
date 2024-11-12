# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import sys, cv2, numpy as np
from dataclasses import dataclass, asdict
from typing import TypedDict, Literal
from typing_extensions import Unpack

from .geometry import Region


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

    @dataclass
    class Style:
        # fmt: off
        class StyleArgs(TypedDict):
            # Layout
            ha: Literal["left", "center", "right"] # Horizontal align
            va: Literal["top", "middle", "bottom"] # Vertical align
            # Styling
            font: int                              # cv2.FONT_*# Text properties
            size: float                            # Initial scale factor
            thickness: float                       # Thickness of the lines used to draw a text
            color: tuple[int, int, int]            # Font face color (BGR, 0-255)
            opacity: float                         # Font face opacity (0.0-1.0)
            # Typesetting
            line_height: float                     # Line height relative to font size
            shrink: bool                           # shrink text to fit
            # Rendering
            dpi_scale: float                       # Simulates Hi-DPI scaling - improves text quality
        # Default values
        ha: Literal["left", "center", "right"] = "left"
        va: Literal["top", "middle", "bottom"] = "middle"
        font: int                              = cv2.FONT_HERSHEY_SIMPLEX
        size: float                            = 1.0
        thickness: float                       = 1.0
        color: tuple[int, int, int]            = (0, 0, 0)
        opacity: float                         = 1.0
        line_height: float                     = 1.0
        shrink: bool                           = True
        dpi_scale: float                       = 2.0
        # Computed properties
        @property
        def dpi_scaled_size(self) -> float:
            return self.size * self.dpi_scale
        @property
        def dpi_scaled_thickness(self) -> int:
            return int(round(self.thickness * self.dpi_scale))
        # CSS-like layered hierarchy
        def __call__(self, **kwargs: Unpack["TextBox.Style.StyleArgs"]) -> "TextBox.Style":
            if len(kwargs) > 0:
                attrs = asdict(self)
                attrs.update(kwargs)
                return TextBox.Style(**attrs)
            else:
                return self

    def __init__(self, box: Region, **kwargs: Unpack[Style.StyleArgs]):
        self.box = box
        self.style = TextBox.Style(**kwargs)

    @staticmethod
    def line_size(text, sty: Style):
        return cv2.getTextSize(
            text, sty.font, sty.dpi_scaled_size, sty.dpi_scaled_thickness
        )

    @staticmethod
    def tokens(text: str):
        tokens = []
        for line in text.split("\n"):
            tokens.extend(line.split())
            tokens.append("\n")
        if len(tokens) > 0 and tokens[-1] == "\n":
            tokens.pop()
        return tokens

    @staticmethod
    def fit(
        box: Region, text: str, style: Style = Style(), **attrs: Unpack[Style.StyleArgs]
    ):
        sty = style(**attrs)
        if sty.size <= 0:
            raise ValueError(f"invalid scale ({sty.size})")
        shape = box.shape * sty.dpi_scale
        h_lim, w_lim = shape
        tokens = TextBox.tokens(text)
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
                (w, h), b = TextBox.line_size(" ".join(line + [next]), sty)
                org_y = y + h
                h += b
                if w > w_lim:
                    if len(line) > 0:
                        # line filled, go to next line
                        break
                    if sty.shrink:
                        # single word is too long
                        return TextBox.fit(box, next, sty, size=sty.size * w_lim / w)
                    print("unable to fit all text", file=sys.stderr)
                    flag_abort = True
                    break
                elif y + h * sty.line_height > h_lim:
                    # vertical overflow
                    return TextBox.fit(box, text, sty, size=sty.size * 0.9)
                else:
                    # add word to line
                    line.append(tokens.pop(0))
                    line_width = w
            if flag_abort:
                break
            match sty.ha:
                case "left":
                    org_x = 0
                case "center":
                    org_x = (w_lim - line_width) // 2
                case "right":
                    org_x = w_lim - line_width
                case _:
                    raise ValueError(f'invalid horizontal align "{sty.ha}"')
            lines.append((org_x, org_y, line))
            y = int(y + h * sty.line_height)

        match sty.va:
            case "top":
                offset_y = 0
            case "middle":
                offset_y = (h_lim - y) // 2
            case "bottom":
                offset_y = h_lim - y
            case _:
                raise ValueError(f'invalid vertical align "{sty.va}"')

        def render(mat):
            color = np.array(sty.color)
            mask = np.zeros(shape, dtype=np.uint8)
            opacity = sty.opacity
            val = 255 if opacity >= 1.0 else int(round(255 * opacity))
            for x, y, line in lines:
                text = " ".join(line)
                anchor = (x, y + offset_y)
                cv2.putText(
                    img=mask,
                    text=text,
                    org=anchor,
                    fontFace=sty.font,
                    fontScale=sty.dpi_scaled_size,
                    color=[val],
                    thickness=sty.dpi_scaled_thickness,
                    lineType=cv2.LINE_AA,
                )
            if shape != box.shape:
                mask = cv2.resize(mask, box.shape[::-1], interpolation=cv2.INTER_LINEAR)
            mask = mask.astype(np.float32) / 255.0
            mask = np.stack([mask] * 3, axis=-1)
            # apply mask
            mat[box.slice_y, box.slice_x] = np.rint(
                box(mat) * (np.ones_like(mask) - mask) + mask * color
            ).astype(np.uint8)
            return mat

        return render

    def __call__(self, mat: np.ndarray, text: str, **attrs: Unpack[Style.StyleArgs]):
        return self.fit(self.box, text, self.style, **attrs)(mat)

    def __repr__(self):
        return f"TextBox({self.box}, {self.style.ha}, {self.style.va})"
