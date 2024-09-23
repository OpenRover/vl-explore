# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from pathlib import Path
import cv2


def VideoWriter(file: str | Path, fps: float, size: tuple[int, int], *cc: str):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    # Test codecs according to preferences
    cc += ("avc1", "mp4v", "MJPG", "XVID", "DIVX", "hvc1")
    for _cc in cc:
        codec = cv2.VideoWriter_fourcc(*_cc)
        video = cv2.VideoWriter(str(file), codec, fps, size, isColor=True)
        if video.isOpened():
            return video, _cc
        else:
            video.release()
    raise RuntimeError(f"Unable to open video writer for {file}, codecs tried: {cc}")
