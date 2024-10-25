# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from numpy import ndarray
from torch import Tensor
from util.queue import Queue
from threading import Thread

from lib.slicer import Slicer, shape

import models.clip as clip

from util import types
from util.exception import Expect
from util.sockets import SocketTransport, Server
from util.timer import Timer, now

from . import protocol
from ..utils import ros_entry, Count, ImageSubscriber

from util.transport import Transport, Pool
from util.env import CWD

Input = Queue[tuple[float, ndarray, list[str]]]
Output = SocketTransport[types.PerceptionStamped]


def log_to(arr: list[str]):
    def f(s: str):
        arr.append(s)

    return f


@Queue.Loop()
def pred(s: Slicer, i: Input, o: Output, print=print):
    t0, count = now(), Count()
    for ts, img, msg in i:
        with Timer("Preprocess".ljust(10), print=log_to(msg), origin=ts):
            data = clip.prepare(s, img, threads=len(s))
            del img
        with Timer("Inference".ljust(10), print=log_to(msg), origin=ts):
            pred = clip.encode_image(data).cpu()
            del data
        o.send((ts, pred))
        # Process log
        msg = list(count(*msg))
        t1 = now()
        t0, dt = t1, t1 - t0
        freq = 1 / dt
        banner = f" ========== Frame {count.n} | Freq: {freq:.2f} Hz =========="
        msg = [banner, *msg]
        for m in msg:
            print(m)


class Perception(ImageSubscriber):

    output: SocketTransport[types.PerceptionStamped]

    def __init__(self):
        super().__init__("perception", "image")
        self.output = Server(
            path=CWD / "perception.socket",
            protocol=protocol.Perception,
            logger=self.get_logger(),
        ).start()


@ros_entry
def main():
    from util.env import select_device

    select_device()
    clip.init(visual=True)
    node = Perception()
    logger = protocol.logger = node.get_logger()
    for topic, image, timestamp in node():
        slicer = node.strategy.Slicer(shape(image), (960, 640))
        assert len(slicer) > 0, len(slicer)
        break
    input: Input = Queue(drop=True)
    worker = Thread(
        target=pred, args=(slicer, input, node.output, logger.debug), daemon=True
    )
    worker.start()
    with Expect(KeyboardInterrupt), Queue.Loop():
        for topic, image, timestamp in node():
            ts = timestamp.count
            input.put((ts, image, []))
    input.close()
    worker.join()
    return node
