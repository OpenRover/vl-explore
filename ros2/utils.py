# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import TypeVar, Callable, Literal, overload
from os import getcwd
from time import time as now
from math import atan2, asin, degrees
from numpy import ndarray
from rclpy import init, spin_once, ok
from rclpy.node import Node as ROS2Node
from rclpy.subscription import Subscription

from cv_bridge import CvBridge
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

from lib.strategies import use

from util.math import ang_diff
from util.geometry import Point2f as Point
from util.transport import Transport


class Node(ROS2Node):
    def __init__(self, name: str):
        super().__init__(name)
        self.get_logger().info(f"Node {name} starting at {getcwd()}")
        self.declare_parameter("strategy", "6T1P")
        self.strategy = use(str(self.get_parameter("strategy").value))
        self.mixer = self.strategy.MotionMixer()


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


T = TypeVar("T")
I = TypeVar("I")
O = TypeVar("O")


class Subscriber(Node):
    def __init__(self, name: str):
        super().__init__(name)
        self.subs = set[Subscription]()

    @overload
    def subscribe(
        self, topic: str, typ: type[T], tp: Literal[None], qos: int
    ) -> Transport[T, T]: ...

    @overload
    def subscribe(
        self, topic: str, typ: type[T], tp: Transport[I, O], qos: int
    ) -> Transport[I, O]: ...

    def subscribe(
        self,
        topic: str,
        typ: type[T],
        tp: Transport[I, O] | None = None,
        qos: int = 10,
    ) -> Transport[I, O] | Transport[T, T]:
        if tp is None:
            tp = Transport[T, T]()
        self.subs.add(self.create_subscription(typ, topic, tp.send, qos))
        return tp


class ImageSubscriber(Node):

    flag_image_recv = False
    images: dict[str, tuple[ndarray, TimeStamp]] = {}

    def __init__(self, name, *topics, qos: int = 10, encoding: str = "bgr8"):
        super().__init__(name)
        self.bridge = CvBridge()

        def subscribe_image(topic: str):
            self.images[topic] = None

            def handler(msg: Image):
                frame = self.bridge.imgmsg_to_cv2(msg, encoding)
                self.images[topic] = (frame, TimeStamp(msg.header.stamp))
                self.flag_image_recv = True

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

    def __iter__(self):
        return self(*list(self.images.keys()))


def ros_entry(main: Callable[..., Node | list[Node] | None]):
    def wrapper(*args, **kwargs):
        init()
        ret = None
        try:
            ret = main(*args, **kwargs)
        except KeyboardInterrupt:
            pass
        finally:
            if not isinstance(ret, list):
                ret = [ret]
            for node in ret:
                if isinstance(node, Node):
                    node.destroy_node()

    return wrapper


def attitude_from_quaternion(q: Quaternion):
    # Convert attitude quaternion to euler angles
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = asin(2.0 * (w * y - z * x))
    yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    # Convert to degrees
    return [ang_diff(0.0, degrees(x)) for x in [roll, pitch, yaw]]


from collections import deque


class TravelAccumulator:
    def __init__(self, duration_seconds: float = 5.0):
        assert duration_seconds > 0.0, "Duration must be positive"
        self.duration = duration_seconds
        self.queue: deque[tuple[float, Point]] = deque()

    prev_ts: float | None = None
    prev_loc: Point | None = None

    def shift(self, deadline: float):
        # Shift out all outdated item to prev
        if len(self.queue) == 0:
            return False
        while len(self.queue) > 0:
            t, p = self.queue[0]
            if t < deadline:
                self.prev_loc = p
                self.queue.popleft()
            else:
                break

    def update(self, t: float, p: Point):
        self.queue.append((t, p))
        # Remove outdated data
        deadline = t - self.duration
        self.shift(deadline)
        if self.prev_loc is None:
            return None
        p0 = self.prev_loc
        # Return accumulated offset vector
        return p - p0

    def reset(self):
        self.queue.clear()
        self.prev_ts = None
        self.prev_loc = None


class TrapDetector:

    def __init__(self, *_, **__):
        super().__init__(*_, **__)
        self.trapped_since: float | None = None
        self.trapped_by: dict[str, bool] = {}

    def is_trapped(self):
        return any(self.trapped_by.values())

    def trapped_for(self):
        if self.trapped_since is None:
            return 0.0
        return now() - self.trapped_since

    def trapped_reason(self):
        return [k for k, v in self.trapped_by.items() if v]

    def declare_trapped(self, reason: str, is_trapped: bool, ts: float = None):
        prev = self.trapped_by.get(reason, False)
        self.trapped_by[reason] = is_trapped
        if self.is_trapped():
            assert ts is not None, "Timestamp must be provided when trapped"
            if self.trapped_since is None or ts < self.trapped_since:
                self.trapped_since = ts
                self.get_logger().info(
                    f"TRAPPED BECAUSE {reason} ({self.trapped_since})"
                )
            else:
                # Already been trapped
                pass
        else:
            if prev:
                self.get_logger().info(f"TRAPPED RESOLVED FROM {reason}")
            self.trapped_since = None


class Count:
    def __init__(self, digits: int = 4):
        self.n = 0
        self.digits = digits

    def __call__(self, *args):
        n = self.n = self.n + 1
        prefix = f"[{n}] " if self.digits is None else f"[{n:0{self.digits}}] "
        if len(args) == 1 and callable(args[0]):
            fn = args[0]

            def decorator(*a, **kw):
                return prefix + fn(*a, **kw)

            return decorator
        if len(args) > 0:
            return [prefix + arg for arg in args]
        return n

    def __str__(self):
        return str(self.n)


class Ramp:
    src: float = 0.0
    dst: float = 0.0  # Target velocity
    rate: float = 0.0  # Acceleration limitation
    ts: float = 0.0  # Last update timestamp

    def __init__(self, rate: float = 0.5):
        self.rate = abs(rate)
        self.ts = now()
        if self.rate <= 0:
            raise ValueError(f"Bad rate: {rate}")

    def set(self, dst: float) -> "Ramp":
        self.dst = float(dst)
        return self

    def get(self) -> float:
        t1 = now()
        dt = t1 - self.ts
        self.ts = t1
        limit = self.rate * dt
        # Update and return src value
        delta = min(abs(self.dst - self.src), limit)
        if self.dst > self.src:
            self.src += delta
        else:
            self.src -= delta
        return self.src
    
    def __float__(self) -> float:
        return self.get()

    def __call__(self, value: float) -> float:
        return float(self.set(value))
