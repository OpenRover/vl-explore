# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import TypeVar
from threading import Thread
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from pathlib import Path
from time import time as now
from math import sqrt
from rclpy import init, ok, spin_once

from .utils.ros import transports as TP
from .utils.ros import Subscriber, TravelAccumulator, TrapDetector
from .utils.mux import mux

import models.clip as clip

from util.math import ang_diff, sign, interpolate, clamp
from util.iter import first_item
from util.queue import Queue
from util.action import Action
from util.transport import Pool
from util.exception import Expect

vt_clamp = clamp(0, 1.0)
vr_clamp = clamp(-1.0, 1.0)


class Node(Subscriber, TrapDetector, Action[TP.Motion].Thread):

    transports: list[TP.Transport] = []

    def __init__(self):
        super().__init__("navigation")
        # Timeout (seconds) before trapping signal is asserted
        self.declare_parameter("trap_duration", 5.0)
        self.trap_duration = float(self.get_parameter("trap_duration").value)
        self.accumulator = TravelAccumulator(self.trap_duration)
        # Distance (meters) within which the robot is considered being trapped
        self.declare_parameter("trap_distance", 0.2)
        self.trap_distance = float(self.get_parameter("trap_distance").value)
        # Destination directory for image dumping
        self.declare_parameter("dst", "/tmp/recording")
        path = Path(self.get_parameter("dst").value)
        # CLIP components
        prompts = [p.to("cpu") for p in self.strategy.prompts()]
        clip.init(visual=True, text=False)
        # Motion publisher
        self.motion_pub = self.create_publisher(Twist, "motion", 10)
        # Data queues
        self.hlt_pipe = self.subscribe("halt", Bool, attach_time=True)
        self.msg_pipe: Queue[TP.TextMsg] = Queue()
        self.rec_pipe: Queue[TP.RenderMsg] = Queue()
        # Transports
        T = TypeVar("T")

        def tp(tp: T) -> T:
            self.transports.append(tp)
            return tp

        self.img_tp = TP.ImageTP().start(mode="coroutine")
        self.subscribe("image", TP.Image, self.img_tp)

        kw = dict(strategy=self.strategy)
        self.odm_tp = tp(TP.OdometryTP(self.subscribe("odometry", TP.Odometry)))

        self.prc_tp = tp(TP.PerceptionTP(self.img_tp(drop=True), **kw))
        self.cor_tp = tp(TP.CorrelationTP(self.prc_tp(), prompts=prompts, **kw))
        self.mot_tp = tp(TP.MotionTP(self.cor_tp(), **kw))
        self.stt_tp = tp(TP.StatsTP(self.mot_tp(), decay=0.5))
        self.rec_tp = tp(TP.RenderTP(self.rec_pipe, path=path, **kw))

    vel: list[float] = [0.0, 0.0, 0.0]
    def motion(self, msg: TP.Motion | TP.MotionMsg, info: str = None):
        if isinstance(msg, TP.MotionMsg):
            _, vel = msg
        else:
            vel = msg
        self.get_logger().info(f"{msg} => {vel}")
        vel = list(vel)
        while len(vel) < 3:
            vel.append(0.0)
        assert len(vel) == 3, f"Bad command {msg}"
        vx, vy, vr = self.vel = list(map(float, vel))
        motion = Twist()
        motion.linear.x = vx
        motion.linear.y = vy
        motion.angular.z = vr
        self.motion_pub.publish(motion)
        with Queue.Loop():
            self.msg_pipe.put((now(), info))

    def handle_halt_msg(self, msg: tuple[float, Bool]):
        ts, halt = msg
        self.declare_trapped("halt", halt.data, ts)

    def handle_odom_msg(self, msg: TP.OdometryMsg):
        ts, location, attitude = msg
        travel = self.accumulator.update(ts, location)
        if travel is not None:
            distance = travel.norm()
            if distance < self.trap_distance:
                self.declare_trapped("odometry", True)
            else:
                self.declare_trapped("odometry", False)
        else:
            # Odometry not initialized, assume free to move
            self.declare_trapped("odometry", False)
    
    msg_count: int = 0
    def handle_log_msg(self, msg: tuple[float, TP.LogAccumulator, object]):
        ts, log, _ = msg
        p = self.get_logger().info
        p("========== Frame %d ==========" % self.msg_count)
        for line in log:
            self.get_logger().info(line)
        self.msg_count = (self.msg_count + 1) % 1000

    @Action.action
    def turn_to(self, heading: float, kv: float = 0.2, tolerance: float = 1.0):
        """Turn to a specific heading, stops when angular error is within tolerance"""
        for _, _, (_, _, rz) in self.odm_tp():
            msg = f"Turning from {rz:.2f} deg => {heading:.1f} deg"
            dr = ang_diff(rz, heading)
            if abs(dr) < tolerance:
                yield self.motion([0.0, 0.0, 0.0], msg=msg)
                break
            else:
                vr = vr_clamp(sign(dr) * sqrt(abs(dr / 30.0))) * kv
                if abs(vr) < 0.2:
                    vr = sign(vr) * 0.2
                yield self.motion([0.0, 0.0, vr], msg=msg)

        self.declare_trapped("odometry", False)
        self.accumulator.reset()

    @Action.action
    def look_around(self, direction: float):
        """Perform a 360 degree turn around, find best heading to go next"""
        # (timestamp, heading)
        odm_pipe = self.odm_tp()
        mot_pipe = self.mot_tp()
        trj: list[tuple[float, float]] = []
        _, (_, _, prev_rz) = first_item(odm_pipe)
        accumulated_angle: float = 0.0
        # Listen for motion messages
        direction = vr_clamp(direction)

        for ts, _, (_, _, rz) in odm_pipe:
            trj.append((ts, rz))
            dr, prev_rz = ang_diff(prev_rz, rz), rz
            accumulated_angle += dr
            if abs(accumulated_angle) >= 360.0 and len(trj) > 10:
                break
            p = abs(accumulated_angle) / 360.0
            progress = "{:.1f}".format(100.0 * p).rjust(5, '0')
            msg = f"Looking around - {progress}%"
            self.msg_pipe.put((ts, msg))
            yield self.motion([0.0, 0.0, direction])

        self.msg_pipe.put((now(), "Looking around - [DONE]"))
        yield self.motion([0.0, 0.0, 0.0])
        # Create a mapping between timestamp and heading
        t2r = interpolate(*trj)
        database = [(t2r(t), vx) for t, (vx, _, _) in mot_pipe.close().dump()]
        heading, confidence = max(database, key=lambda x: x[1], default=(None, None))
        if heading is not None and confidence > 0:
            return self.turn_to(heading)
        else:
            # Try again
            self.get_logger().warn("No plausible way found in look around database")
            return self.look_around(direction)

def main():
    init()
    node = Node()
    logger = node.get_logger()
    logger.info("Navigation node initialized")
    mux_thread = Thread(
        target=mux,
        args=(
            node.img_tp(),
            node.cor_tp(),
            node.msg_pipe,
            node.stt_tp(),
            node.rec_pipe,
        ),
        daemon=True,
    )
    hlt_pipe = node.hlt_pipe
    odm_pipe = node.odm_tp()
    log_pipe = node.cor_tp()
    mot_pipe = node.mot_tp()
    pool = Pool.Thread(*node.transports, rate_limit=200)
    with Expect(KeyboardInterrupt, Queue.Closed), pool:
        while ok():
            spin_once(node, timeout_sec=0)
            hlt_pipe(node.handle_halt_msg)
            log_pipe(node.handle_log_msg)
            odm_pipe(node.handle_odom_msg)
            if node.wait_action(node.motion):
                continue
            if mot_pipe(node.motion):
                node.msg_pipe.put((now(), None))
            if node.trapped_for() > node.trap_duration:
                _, _, vr = node.vel
                vr = 0.2 if vr >= 0 else -0.2
                node.look_around(vr)
                node.msg_pipe.put((now(), "Looking around - [INIT]"))
    logger.info("Navigation node terminating")
    mux_thread.join()
    logger.info("Navigation node terminated")
    node.destroy_node()
