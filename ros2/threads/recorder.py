# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import cv2
from pathlib import Path
from time import sleep, time as now

from rclpy import spin
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from util.env import CWD
from util.sockets import Client
from util.transport import Transport, Pool, Queue
import util.JSON as JSON

from . import protocol
from ..utils import ros_entry, Node, TimeStamp


class Recorder(Node):
    def __init__(self):
        super().__init__("recorder")

class SaveImageTP(Transport[tuple[Path, Image], None]):
    def init(self):
        self.bridge = CvBridge()

    def transform(self, arg):
        path, msg = arg
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite(str(path), img)

@ros_entry
def main():
    node = Recorder()
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

    img_list = open(CWD / "images.list", "w")
    img_queue = Queue[tuple[str, Image]]()
    count = 0
    next_report = now()

    def record_image(msg: Image):
        nonlocal count, next_report
        ts = TimeStamp(msg.header.stamp)
        loc = path / f"{count:06d}.jpg"
        count += 1
        img_list.write(",".join(map(JSON.stringify, [ts.count, loc.name])) + "\n")
        img_queue.put((loc, msg))
        t1 = now()
        if t1 > next_report:
            next_report = t1 + 2
            logger.info(f"{len(img_queue)} images in the queue")


    _ = node.create_subscription(Image, "image", record_image, 10)

    img_tps = [SaveImageTP(img_queue) for _ in range(4)]

    with Pool.Thread(per, cor, nav, *img_tps):
        spin(node)
        while True:
            remaining = len(img_queue)
            if remaining == 0:
                break
            logger.info(f"Waiting for image queue to be cleared ({remaining})")
            sleep(0.25)

    return node
