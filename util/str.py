# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================


def center(content: str, width: int, fill: str = " ", pad: str | None = None) -> str:
    lim = width - (len(pad) * 2 if pad is not None else 0)
    if len(content) >= lim:
        return content
    if pad is not None:
        content = pad + content + pad
    s = width - len(content)
    l = s // 2
    r = s - l
    return fill * l + content + fill * r
