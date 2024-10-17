# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Iterator
from math import atan2, asin, degrees
from numpy import ndarray

from cv_bridge import CvBridge
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

from . import Node, spin_once, ok

from util.math import ang_diff
from util.geometry import Point2f as Point


class TimeStamp:
    def __init__(self, s: int | float | Time, ns: int = None):
        if type(s) is Time:
            # Directly construct from Time message
            assert ns is None, f"Invalid arguments {(s, ns)}"
            self.stamp = s
            self.count = float(s.sec) + float(s.nanosec) * 1e-9
        else:
            if type(s) is float:
                # Construct from floating point seconds
                assert ns is None, f"Invalid arguments {(s, ns)}"
                ns = int((s - int(s)) * 1e9)
                s = int(s)
            # Construct from integer seconds and nanoseconds
            self.stamp = Time(sec=s, nanosec=ns)
            self.count = float(s) + float(ns) * 1e-9


class ImageSubscriber(Node):

    images: dict[str, tuple[ndarray, TimeStamp]] = {}

    def __init__(self, name, *topics, qos: int = 10, encoding: str = "bgr8"):
        super().__init__(name)
        self.bridge = CvBridge()

        def subscribe_image(topic: str):
            self.images[topic] = None

            def handler(msg: Image):
                frame = self.bridge.imgmsg_to_cv2(msg, encoding)
                self.images[topic] = (frame, TimeStamp(msg.header.stamp))

            return self.create_subscription(Image, topic, handler, qos)

        self.image_subs = [subscribe_image(topic) for topic in topics]

    def grab_next_image(
        self, topic: str, wait: bool = True, keep: bool = False
    ) -> tuple[ndarray, TimeStamp]:
        """
        Grab the next image from the specified topic
        :param topic: (str) the topic to grab the image from
        :param wait: (bool) whether to wait for the next image if it is not currently available
        :param keep: (bool) whether to keep the image inside the internal buffer
        """
        assert topic in self.images, f"Topic '{topic}' not registered"
        while self.images[topic] is None and wait:
            spin_once(self)
        image = self.images[topic]
        if not keep:
            self.images[topic] = None
        return image

    def __getitem__(self, topic: str) -> tuple[ndarray, float]:
        return self.grab_next_image(topic)

    def __call__(self, *topics: str):
        topics = list(topics)
        if len(topics) == 0:
            topics = list(self.images.keys())
        while ok():
            for t in topics:
                ret = self.grab_next_image(t, False, False)
                if ret is not None:
                    yield t, *ret
            spin_once(self)


def attitude_from_quaternion(q: Quaternion):
    # Convert attitude quaternion to euler angles
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = asin(2.0 * (w * y - z * x))
    yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    # Convert to degrees
    return list(ang_diff(0.0, degrees(x)) for x in [roll, pitch, yaw])


class TravelAccumulator:
    def __init__(self, duration_seconds: float = 5.0):
        assert duration_seconds > 0.0, "Duration must be positive"
        self.duration = duration_seconds
        self.queue: list[tuple[float, Point]] = []

    prev_ts: float | None = None
    prev_loc: Point | None = None

    def update(self, odom: Odometry):
        t = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9
        p = Point(
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
        )
        self.queue.append((t, p))
        # Remove outdated data
        deadline = t - self.duration
        while len(self.queue) and self.queue[0][0] < deadline:
            self.prev_loc, _ = self.queue.pop(0)
        if self.prev_loc is None:
            return None
        # Return accumulated offset vector
        return p - self.prev_loc

    def clear(self):
        self.queue.clear()
        self.prev_ts = None
        self.prev_loc = None


def last_item(iterator: Iterator):
    item = None
    for item in iterator:
        pass
    return item
