# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from numpy import ndarray
from torch import from_numpy
from util.queue import Queue
from threading import Thread

from lib.slicer import Slicer, shape

import models.clip as clip

from util import types
from util.exception import Expect
from util.sockets import SocketTransport, Server, Client
from util.timer import Timer, now

from . import protocol
from ..utils.ros import ros_entry, Count, ImageSubscriber, ok, center, log_to

from util.env import CWD

Input = Queue[tuple[float, ndarray, list[str]]]
Output = SocketTransport[types.PerceptionStamped]


@Queue.Loop()
def pred(s: Slicer, i: Input, o: Output):
    t0, count = now(), Count()
    for ts, img, msg in i:
        if not ok():
            break
        with Timer("ImgCapture".ljust(10), print=log_to(msg), origin=ts):
            pass
        with Timer("Preprocess".ljust(10), print=log_to(msg), origin=ts):
            data = clip.prepare(s, img, threads=len(s))
            del img
        with Timer("Deviation".ljust(10), print=log_to(msg), origin=ts):
            N, C, *_ = data.shape
            std = list(
                map(
                    float,
                    data.view(N, C, -1)
                    .std(dim=2, keepdim=False)
                    .max(dim=1, keepdim=False)
                    .values.cpu()
                    .numpy()
                    .reshape(-1),
                )
            )
        with Timer("Inference".ljust(10), print=log_to(msg), origin=ts):
            pred = (clip.encode_image(data)).cpu()
            del data
        o.send((ts, std, pred))
        # Process log
        msg = list(count(*msg))
        t1 = now()
        t0, dt = t1, t1 - t0
        freq = 1 / dt
        with open(CWD / "perf.log", "at") as perf:
            perf.write(center(f" Frame {count.n} | Freq: {freq:.2f} Hz ", 52, "=") + "\n")
            for l in msg:
                perf.write(l + "\n")


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

    worker = Thread(target=pred, args=(slicer, input, node.output), daemon=True)
    worker.start()
    with Expect(KeyboardInterrupt), Queue.Loop():
        for topic, image, timestamp in node():
            ts = timestamp.count
            input.put((ts, image, []))
    input.close()
    worker.join()
    return node
