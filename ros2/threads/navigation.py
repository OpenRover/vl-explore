# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from math import sqrt
from time import time as now
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from . import protocol
from ..utils import (
    ok,
    spin_once,
    ros_entry,
    attitude_from_quaternion,
    Node,
    TrapDetector,
    TravelAccumulator,
    TimeStamp,
    Count,
    Ramp,
)

from util import types
from util.env import CWD
from util.sockets import SocketTransport, Server, Client
from util.action import Action
from util.math import ang_diff, clamp, sign, interpolate
from util.timer import Timer
from util.geometry import Point2f


vt_clamp = clamp(0, 1.0)
vr_clamp = clamp(-1.0, 1.0)


class Navigation(TrapDetector, Action.Hub, Node):

    input: SocketTransport[types.CorrelationStamped]
    output: SocketTransport[types.MotionStamped]

    def __init__(self):
        super().__init__("navigation")
        self.mixer = self.strategy.MotionMixer()
        # I/O Sockets
        kw = dict(logger=self.get_logger())
        self.input = Client(CWD / "correlation.socket", protocol.Correlation, **kw)
        self.input.start()
        self.output = Server(CWD / "navigation.socket", protocol.Motion, RX=False, **kw)
        self.output.start()
        # Timeout (seconds) before trapping signal is asserted
        self.declare_parameter("trap_duration", 5.0)
        self.trap_duration = float(self.get_parameter("trap_duration").value)
        # Distance (meters) within which the robot is considered being trapped
        self.declare_parameter("trap_distance", 0.5)
        self.trap_distance = float(self.get_parameter("trap_distance").value)
        # Accumulator for travel distance in given duration
        self.accumulator = TravelAccumulator(self.trap_duration)
        # ROS Messages
        self.motion_pub = self.create_publisher(Twist, "motion", 10)
        self.halt_sub = self.create_subscription(Bool, "halt", self.handle_halt_msg, 10)
        self.odom_sub = self.create_subscription(
            Odometry, "odometry", self.handle_odom_msg, 10
        )
        while self.attitude is None:
            spin_once(self)

    vel: types.Motion = (0.0, 0.0, 0.0)
    rmp: tuple[Ramp] = Ramp(0.5), Ramp(0.5), Ramp(0.5)
    msg: str | None = None
    decision_delay: float = 0.0

    def motion(self, vel: types.Motion=None, msg=None):
        if vel is not None:
            self.vel = tuple(map(float, vel)) # Terminal speed demanded
            vx, vy, vz = [r(v) for r, v in zip(self.rmp, vel)]
        else:
            vx, vy, vz = map(float, self.vel)
        # Apply ramping
        motion = Twist()
        motion.linear.x = vx
        motion.linear.y = vy
        motion.angular.z = vz
        self.motion_pub.publish(motion)
        # Assemble navigation message for socket communication
        self.output.send((now(), self.decision_delay, self.vel, msg))
        self.msg = msg

    def handle_halt_msg(self, halt: Bool):
        self.declare_trapped("halt", halt.data, now())

    attitude: tuple[float, tuple[float, float, float]] = None

    def handle_odom_msg(self, msg: Odometry):
        ts = TimeStamp(msg.header.stamp).count
        att = attitude_from_quaternion(msg.pose.pose.orientation)
        self.attitude = (ts, att)
        loc = Point2f(msg.pose.pose.position.x, msg.pose.pose.position.y)
        travel = self.accumulator.update(ts, loc)
        if travel is not None:
            distance = travel.norm()
            if distance < self.trap_distance:
                self.declare_trapped("odometry", True, ts)
            else:
                self.declare_trapped("odometry", False, ts)
        else:
            # Odometry not initialized, assume free to move
            self.declare_trapped("odometry", False, ts)

    correlations: list[types.CorrelationStamped] = []

    @Action.action
    def turn_to(self, heading: float, kv: float = 0.2, tolerance: float = 1.0):
        """Turn to a specific heading, stops when angular error is within tolerance"""
        while True:
            ts, (rx, ry, rz) = self.attitude
            dr = ang_diff(rz, heading)
            if abs(dr) < tolerance:
                yield self.motion((0.0, 0.0, 0.0), f"Turn to {heading:.2f} deg [DONE]")
                break
            else:
                vr = vr_clamp(sign(dr) * sqrt(abs(dr / 30.0))) * kv
                if abs(vr) < 0.2:
                    vr = sign(vr) * 0.2
                yield self.motion(
                    (0.0, 0.0, vr), f"Turning from {rz:.2f} deg to {heading:.2f} deg"
                )

        node.declare_trapped("odometry", False)
        node.accumulator.reset()

    @Action.action
    def look_around(self, direction: float):
        """Perform a 360 degree turn around, find best heading to go next"""
        # (timestamp, heading)
        trj: list[tuple[float, float]] = []
        _, (_, _, prev_rz) = self.attitude
        accumulated_angle: float = 0.0
        # Clear existing correlation database
        self.correlations.clear()

        def info(s=None):
            if s is None:
                p = abs(accumulated_angle) / 360.0
                progress = "{:.1f}".format(100.0 * p)
                s = progress.rjust(5, "0") + "%"
            return f"Looking around - {s}"

        direction = vr_clamp(direction)
        while abs(accumulated_angle) < 360.0 or len(trj) < 10:
            # Continue turning around
            yield self.motion((0.0, 0.0, direction), info())
            ts, (_, _, rz) = self.attitude
            trj.append((ts, rz))
            dr = ang_diff(prev_rz, rz)
            accumulated_angle += dr
            prev_rz = rz
        yield self.motion((0.0, 0.0, 0.0), info("[DONE]"))
        # Create a mapping between timestamp and heading
        t2r = interpolate(*trj)
        database = list[tuple[float, float]]()
        for t, c in self.correlations:
            heading = t2r(t)
            pred = self.mixer.to_numpy(c).reshape(-1, 2)
            n, f = pred.mean(axis=0, keepdims=False)
            # Calculate confidence from navigability and familiarity
            confidence = n * sqrt(abs(1.0 - f))
            database.append((heading, confidence))
        self.correlations.clear()
        # Find the best heading to go next
        best_heading, best_confidence = None, 0.0
        for heading, confidence in database:
            if confidence > best_confidence:
                best_heading, best_confidence = heading, confidence
        self.trapped_since = None
        if best_heading is not None:
            self.turn_to(best_heading)
        else:
            self.get_logger().warn("No plausible way found in look around database")
            # Try again
            self.look_around(direction)


@ros_entry
def main():
    global node
    node = Navigation()
    input = node.input()
    logger = protocol.logger = node.get_logger()
    # Control Rate Throttling
    interval: float = 1.0 / 25.0
    next_loop: float = now() + interval
    count = Count()
    first_correlation = True
    # Wait for input to be ready
    logger.info(f"Waiting for upstream ...")
    while not len(input):
        spin_once(node, timeout_sec=1e-3)
    logger.info(f"Starting initial look around")
    node.look_around(0.2)
    while ok():
        while now() < next_loop:
            spin_once(node, timeout_sec=next_loop - now())
        next_loop += interval
        for ts, correlation in input.dump():
            with Timer(*count("Decision".ljust(10)), print=logger.debug, origin=ts):
                node.correlations.append((ts, correlation))
            if node.msg is not None:
                logger.info(node.msg)
                node.msg = None
        if node.wait_action():
            pass
        elif node.trapped_for() > node.trap_duration and not first_correlation:
            duration = node.trapped_for()
            reason = ", ".join(node.trapped_reason())
            logger.info(str(node.trapped_by))
            logger.info(f"Trapped for {duration:.2f}s ({reason})")
            node.look_around(0.2 if node.vel[2] >= 0.0 else -0.2)
        elif len(node.correlations):
            if first_correlation:
                node.declare_trapped("odometry", False)
                node.accumulator.reset()
                first_correlation = False
            msg = None
            if node.is_trapped():
                duration = node.trapped_for()
                reason = ", ".join(node.trapped_reason())
                msg = f"Trapped for {node.trapped_for():.2f}s ({reason})"
            # Normal operation
            *_, (ts, correlation) = node.correlations
            node.correlations.clear()
            node.decision_delay = now() - ts
            node.motion(node.mixer(correlation), msg=msg)
        else:
            # Keep ramping previous demanded velocities
            node.motion(None)
    return node
