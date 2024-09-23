# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from pathlib import Path
import cv2


def VideoWriter(file: str | Path, fps: float, size: tuple[int, int], *cc: str):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    # Test codecs according to preferences
    for cc in cc + ("hvc1", "avc1", "mp4v", "MJPG", "XVID", "DIVX"):
        codec = cv2.VideoWriter_fourcc(*cc)
        video = cv2.VideoWriter(str(file), codec, fps, size, isColor=True)
        if video.isOpened():
            return video, cc
        else:
            video.release()
    raise RuntimeError(f"Unable to open video writer for {file}, codecs tried: {cc}")
