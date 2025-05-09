# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from time import time as now
from numpy import ndarray
from util.queue import Queue
from threading import Thread
from cv2 import cvtColor, COLOR_BGR2RGB

from std_msgs.msg import Empty

from lib.slicer import Slicer, shape

import models.clip as clip

from util import types
from util.exception import Expect
from util.sockets import SocketTransport, Server, Client
from util.timer import Timer, now
from util.str import center

from . import protocol
from ..utils.ros import ros_entry, Count, ImageSubscriber, ok, log_to

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
            img = cvtColor(img, COLOR_BGR2RGB)
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
            pred = (clip.encode_image(data)).cpu().numpy()
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
        # Tick handling - for simulating synchronous operation
        self.tick: bool = False
        self.last_tick: float = now()
        self.tick_sub = self.create_subscription(
            Empty, "tick", self.onTick, 10
        )

    def onTick(self, _: Empty):
        self.tick = True
        self.last_tick = now()


@ros_entry
def main():
    from util.env import select_device

    select_device()
    clip.init(visual=True)
    node = Perception()
    protocol.logger = node.get_logger()
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
            if True: # Parallel mode
                input.put((ts, image, []))
                continue
            if node.tick:
                node.tick = False
                input.put((ts, image, []))
            elif now() - node.last_tick > 10.0:
                node.get_logger().warn("No tick received for 10 seconds.")
                input.put((ts, image, []))
                node.last_tick = now()
    input.close()
    worker.join()
    return node
