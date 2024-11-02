# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from numpy import ndarray
from torch import Tensor
from util.logger import Logger
from util.transport import Transport

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from .ros import TimeStamp

log = Logger(__name__)

TextMsg = tuple[float, str | None]

# (ts, image)
ImageMsg = tuple[float, ndarray]


class ImageTP(Transport[Image, ImageMsg]):
    bridge = CvBridge()
    encoding: str = "bgr8"

    def transform(self, arg: Image):
        ts = TimeStamp(arg.header.stamp).count
        image = self.bridge.imgmsg_to_cv2(arg, desired_encoding=self.encoding)
        assert isinstance(image, ndarray), "Failed to convert image message to ndarray"
        yield ts, image


from nav_msgs.msg import Odometry
from .ros import TimeStamp, attitude_from_quaternion
from util.geometry import Point2f

# (tx, ty)
Location = Point2f
# (rx, ry, rz)
Attitude = tuple[float, float, float]
# (ts, attitude, travel)
OdometryMsg = tuple[float, Location, Attitude]


class OdometryTP(Transport[Odometry, OdometryMsg]):

    def transform(self, msg: Odometry):
        ts = TimeStamp(msg.header.stamp).count
        location = Point2f(msg.pose.pose.position.x, msg.pose.pose.position.y)
        attitude = attitude_from_quaternion(msg.pose.pose.orientation)
        yield tuple((ts, location, attitude))


from lib.strategies import Strategy
from lib.slicer import Slicer, shape
from prompts import Prompt
from util.iter import first_item
from util.timer import Timer


class LogAccumulator(list[str]):
    def __call__(self, *msgs: str, sep: str = " "):
        self.append(sep.join(msgs))


PerceptionMsg = tuple[float, LogAccumulator, Tensor]


class PerceptionTP(Transport[ImageMsg, PerceptionMsg]):
    import models.clip as clip
    strategy: type[Strategy] = None
    size_limit: tuple[int, int] = (960, 640)
    slicer: Slicer = None

    def init(self):
        assert self.strategy is not None, "Strategy not set"

    def transform(self, arg: ImageMsg):
        ts, image = arg
        if self.slicer is None:
            self.slicer = self.strategy.Slicer(shape(image), self.size_limit)
        log = LogAccumulator()
        with Timer("Preprocess".ljust(10), print=log, origin=ts):
            data = self.clip.prepare(self.slicer, image, threads=len(self.slicer))
            del image
        with Timer("Inference".ljust(10), print=log, origin=ts):
            pred = self.clip.encode_image(data).cpu()
            del data
        yield ts, log, pred


from ..threads.correlator import DataBase

# (label, navigability, familiarity)
Correlation = tuple[str, float, float]
# (log, ts, correlation)
CorrelationMsg = tuple[float, LogAccumulator, list[list[Correlation]]]


class CorrelationTP(Transport[PerceptionMsg, CorrelationMsg]):
    prompts: list[Prompt] = None
    database: DataBase = None

    def init(self):
        assert self.prompts is not None, "Prompts not set"

    def transform(self, arg: PerceptionMsg):
        ts, log, perception = arg
        if self.database is None:
            p = perception.cpu().numpy()
            _, width = p.shape
            dtype = p.dtype
            self.database = DataBase(width, dtype=dtype)
        with Timer("Correlate".ljust(10), print=log, origin=ts):
            scores = list(p(perception) for p in self.prompts)
            correlation: list[list[Correlation]] = [[] * len(scores)]
            perception = perception.cpu().numpy()
            for tile_id, familiarity in enumerate(self.database(perception)):
                for prompt_id, c in enumerate(correlation):
                    assert len(c) == tile_id
                    label, score = scores[prompt_id][tile_id]
                    c.append([label, score, familiarity])
        yield ts, log, correlation


from lib.motion_mixer import Motion, MotionMixer


class MotionMsg(tuple[float, Motion]):
    pass


class MotionTP(Transport[CorrelationMsg, MotionMsg]):
    strategy: type[Strategy] = None
    mixer: MotionMixer = None

    def init(self):
        assert self.strategy is not None, "Strategy not set"
        self.mixer = self.strategy.MotionMixer()

    def transform(self, arg: CorrelationMsg):
        ts, _, correlations = arg
        yield MotionMsg((ts, self.mixer(correlations)))


from time import time as now

Stats = list[tuple[str, str]]
StatsMsg = tuple[float, Stats]


class StatsTP(Transport[MotionMsg, StatsMsg]):
    decay: float = 0.5
    freq: float = None
    last_ts: float = None

    def transform(self, arg: MotionMsg):
        ts, motion = arg
        delay = (now() - ts) * 1000
        if self.last_ts is not None:
            dt = ts - self.last_ts
            freq = 1.0 / dt
            if self.freq is None:
                self.freq = freq
            else:
                d1, d2 = self.decay, 1 - self.decay
                freq = self.freq = d1 * freq + d2 * self.freq
        self.last_ts = ts
        ret = [("Latency", f"{delay:.2f} ms")]
        if self.freq is not None:
            ret.append(("Frequency", f"{self.freq:.2f} Hz"))
        yield [("delay", f"ts"), ("motion", motion)]


from lib.renderer import Renderer
from pathlib import Path
from cv2 import imwrite

# (frame, correlations, banner, stats)
RenderMsg = tuple[ndarray, list[list[Correlation]], str, Stats | None]


class RenderTP(Transport[RenderMsg, None]):
    strategy: type[Strategy] = None
    slicer: Slicer = None
    renderer: Renderer = None
    count: int = 0
    path: Path = Path("/tmp/recording")

    def init(self):
        assert self.strategy is not None, "Strategy not set"
        self.path.mkdir(parents=True, exist_ok=True)
        assert self.path.is_dir(), f"Invalid recording destination: {self.path}"

    def transform(self, arg: RenderMsg):
        frame, correlations, banner, stats = arg
        if self.slicer is None:
            self.slicer = self.strategy.Slicer(shape(frame))
        if self.renderer is None:
            self.renderer = self.strategy.Renderer(self.slicer)
        self.renderer(frame, correlations, banner=banner, stats=stats)
        imwrite(str(self.path / f"{self.count:06d}.png"), frame)
        self.count += 1
