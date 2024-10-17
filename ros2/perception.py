# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from rclpy import init
from numpy import ndarray
from torch import Tensor

from lib.slicer import Slicer, shape
import models.clip as clip
from util.sockets import Server
from util.timer import Timer

from .utils import ImageSubscriber
from . import ros_entry
from . import protocol, socket_perception


class Perception(ImageSubscriber):
    slicer: Slicer = None

    def __init__(self):
        super().__init__("perception", "img_in")

    def process(self, image: ndarray):
        if self.slicer is None:
            self.slicer = self.strategy.Slicer(shape(image), (960, 640))
        pred = clip.encode_image(*self.slicer(image))
        assert isinstance(pred, Tensor)
        return pred


@ros_entry
def main():
    # Load CLIP vision model
    clip.encode_image()
    node = Perception()
    logger = protocol.logger = node.get_logger()
    output = Server(socket_perception, logger=logger)
    for topic, image, timestamp in node():
        ts = timestamp.count
        with Timer(print=logger.info, origin=ts):
            perception = node.process(image).cpu().numpy()
        output(*protocol.Perception.encode(ts, perception))
        for line in output:
            logger.info("Unhandled inbound message: " + line)
    return [node]
