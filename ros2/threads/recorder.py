# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import cv2
from pathlib import Path

from util.env import CWD
from util.sockets import Client
from util.transport import Pool
import util.JSON as JSON

from . import protocol
from ..utils import ros_entry, ImageSubscriber


@ros_entry
def main():
    node = ImageSubscriber("recorder", "image")
    logger = protocol.logger = node.get_logger()

    node.declare_parameter("dst", "/tmp/recording")
    dst = str(node.get_parameter("dst").value)
    path = Path(CWD / dst)
    path.mkdir(parents=False, exist_ok=False)

    per = Client(CWD / "perception.socket", logger=logger)
    per.pipe(open(CWD / "perception.list", "w"))

    cor = Client(CWD / "correlation.socket", logger=logger)
    cor.pipe(open(CWD / "correlation.list", "w"))

    nav = Client(CWD / "navigation.socket", logger=logger)
    nav.pipe(open(CWD / "navigation.list", "w"))

    img = open(CWD / "images.list", "w")
    count = 0


    with Pool.Thread(per, cor, nav):
        for topic, frame, ts in node:
            loc = path / f"{count:06d}.png"
            count += 1
            cv2.imwrite(str(loc), frame)
            img.write(",".join(map(JSON.stringify, [ts.count, loc.name])) + "\n")
