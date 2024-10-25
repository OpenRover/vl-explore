# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from pathlib import Path
import cv2
from numpy import ndarray


class VideoCapture(cv2.VideoCapture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.isOpened():
            raise RuntimeError(f"Unable to open video capture for {args[0]}")
        self.size = (
            int(self.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.fps = self.get(cv2.CAP_PROP_FPS)

    def __iter__(self):
        while self.isOpened():
            ret, frame = self.read()
            if not ret:
                break
            assert type(frame) is ndarray
            yield frame
        self.release()

    def __len__(self):
        return int(self.get(cv2.CAP_PROP_FRAME_COUNT))


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
