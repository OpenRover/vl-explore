# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from util.env import to_device, Logger
from .__dir__ import DIR

log = Logger(__file__)

name = "yolov8m"
model: YOLO | None = None


def init(name=name):
    log.info(f"Loading YOLO model ({name}) ...")
    return to_device(YOLO(DIR / name + ".pt", verbose=False))


@torch.no_grad()
def detect(frame) -> list[Results]:
    global model
    if model is None:
        model = init()
    return model(frame)
