# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import numpy as np
from time import time as now
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from . import protocol
from ..utils.look_around import LookAroundDatabase
from ..utils.ros import (
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
from util.math import ang_diff, clamp, sign, interpolate, project
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
        self.declare_parameter("trap_distance", 0.3)
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
    rmp: tuple[Ramp] = Ramp(0.2), Ramp(0.2), Ramp(0.2)
    decision_delay: float = 0.0

    msg: str | None = None  # Used to log message to console
    last_msg: str = None  # Used to repeat msg to socket

    def motion(self, vel: types.Motion = None, msg=None):
        if vel is not None:
            # Velocity updated
            self.vel = tuple(map(float, vel))
        # Apply ramping
        vx, vy, vr = [r(v) for r, v in zip(self.rmp, self.vel)]
        motion = Twist()
        motion.linear.x = vx
        motion.linear.y = vy
        motion.angular.z = vr
        self.motion_pub.publish(motion)
        # Assemble navigation message for socket communication
        self.output.send(
            (now(), self.decision_delay, (vx, vy, vr), msg or self.last_msg)
        )
        if msg is not None:
            self.last_msg = self.msg = msg

    def handle_halt_msg(self, halt: Bool):
        self.declare_trapped("halt", halt.data, now())

    attitude: tuple[float, tuple[float, float, float]] = None
    location: Point2f = Point2f(0.0, 0.0)

    def handle_odom_msg(self, msg: Odometry):
        ts = TimeStamp(msg.header.stamp).count
        att = attitude_from_quaternion(msg.pose.pose.orientation)
        self.attitude = (ts, att)
        self.location = Point2f(msg.pose.pose.position.x, msg.pose.pose.position.y)
        travel = self.accumulator.update(ts, self.location)
        if travel is not None:
            trapped = travel.norm() < self.trap_distance
        else:
            # Odometry not initialized, assume free to move
            trapped = False
        self.declare_trapped("odometry", trapped, ts)
        tx, ty = self.location
        _, _, rz = att
        enc = protocol.Odometry.encode([ts, tx, ty, rz])
        with open(CWD / "odometry.list", "at") as lst:
            lst.write(",".join(map(str, enc)) + "\n")

    correlations: list[types.CorrelationStamped] = []

    @Action.action
    def move_for(self, duration: float, vel: tuple[float] = (-0.2, 0.0, 0.0)):
        """Turn to a specific heading, stops when angular error is within tolerance"""
        t1: float = now() + duration

        def vel_match(vel, EPS=0.01):
            return all(abs(r.get() - v) < EPS for r, v in zip(self.rmp, vel))

        while not vel_match(vel):
            yield self.motion(vel)
        while now() < t1:
            yield self.motion(vel)
        while any(r.get() > 0.1 for r in self.rmp):
            yield self.motion((0.0, 0.0, 0.0))

    look_around_id: int = 0
    last_look_around_location: Point2f = Point2f(0.0, 0.0)
    # heading, confidence
    last_look_around: LookAroundDatabase = None

    @Action.action
    def turn_to(self, heading: float, kv: float = 0.4, tolerance: float = 5.0):
        """Turn to a specific heading, stops when angular error is within tolerance"""
        ramp = self.rmp[2]
        prev_ramp_rate = ramp.rate
        ramp.rate = 10.0
        heading = ang_diff(0.0, heading)
        t0: float = None
        while True:
            ts, (rx, ry, rz) = self.attitude
            dr = ang_diff(rz, heading)
            if abs(dr) < tolerance:
                if t0 is None:
                    t0 = ts
                elif ts - t0 > 0.5:
                    break  # Stabilized for 0.5 second
            else:
                t0 = None
            # kp-only "PID" controller
            vr = sign(dr) * max(0.2, abs(dr / 90.0) * kv)
            vr = vr_clamp(vr)
            yield self.motion(
                (0.0, 0.0, vr), f"Turning from {rz:.2f} deg to {heading:.2f} deg"
            )

        yield self.motion((0.0, 0.0, 0.0), f"Turn to {heading:.2f} deg [DONE]")

        self.declare_trapped("odometry", False)
        self.accumulator.reset()
        ramp.rate = prev_ramp_rate

    look_around_id: int = 0
    last_look_around_location: Point2f = Point2f(0.0, 0.0)
    # heading, confidence
    last_look_around: LookAroundDatabase = None

    @Action.action
    def look_around(
        self, direction: float, initial: bool = False, neg_window: float = 90.0
    ):
        """Perform a 360 degree turn around, find best heading to go next"""
        _, (_, _, initial_rz) = self.attitude
        # Check if look around has been done in the same location
        if self.last_look_around is not None and self.last_look_around.has_candidates():
            travel = self.location - self.last_look_around_location
            if travel.norm() < self.trap_distance * 2:
                # Reuse previous look around result
                heading, _ = self.last_look_around.next_candidate()
                self.get_logger().info(
                    f"Look around reusing previous result: {heading}"
                )
                return self.turn_to(heading)
        # Initiate new look around procedure
        time_start = now()
        id = self.look_around_id
        self.last_look_around = db = LookAroundDatabase(f"Look Around {id}")
        self.last_look_around_location = self.location
        self.look_around_id += 1
        db.add({"direction": "L" if direction > 0 else "R"})
        db.add({"initial_rz": initial_rz})
        if not initial:
            db.add({"neg_window": neg_window})
        # (timestamp, heading)
        trj: list[tuple[float, float]] = []
        prev_rz = initial_rz
        accumulated_angle: float = 0.0
        # Clear existing correlation database
        self.correlations.clear()

        def info(s=None):
            if s is None:
                p = abs(accumulated_angle) / 360.0
                progress = "{:.1f}".format(100.0 * p)
                s = progress.rjust(5, "0") + "%"
            return f"Looking around {id} - {s}"

        direction = vr_clamp(direction)
        while abs(accumulated_angle) < 360.0 or len(trj) < 10:
            # Continue turning around
            yield self.motion((0.0, 0.0, direction), info())
            ts, (_, _, rz) = self.attitude
            dr = ang_diff(prev_rz, rz)
            accumulated_angle += dr
            prev_rz = rz
            trj.append((ts, accumulated_angle))
        yield self.motion((0.0, 0.0, 0.0), info("[DONE]"))
        db.add({"time": [time_start, now()]})
        self.get_logger().info(
            f"Look around {id} captured {len(self.correlations)} data points"
        )
        # Create a mapping between timestamp and heading
        t2r = interpolate(*trj)
        for t, (*c, target) in self.correlations:
            heading = ang_diff(0.0, t2r(t) + initial_rz)
            pred = self.mixer.to_numpy(c).reshape(2, -1, 3)
            nav, fam, std = pred[:, :, 0], pred[:, :, 1], pred[:, :, 2]
            trg = self.mixer.to_numpy([target]).reshape(2, -1, 3)[:, :, 0]
            FAR, NEAR = 0, 1
            nav: np.ndarray = nav[FAR] * 0.2 + nav[NEAR] * 0.8
            fam: np.ndarray = fam[FAR] * 0.8 + fam[NEAR] * 0.2
            trg: np.ndarray = np.max(trg, axis=0)
            # Give higher weight to the center region
            nav[1] *= 2.0
            fam[1] *= 2.0
            trg[1] *= 2.0
            # Blend navigability and familiarity only when not initial
            score: np.ndarray  # Range [-1, 1]
            if initial:
                score = nav
            else:
                KN = 0.6  # Weight of navigability score
                try:
                    fs = project((fam.max(), fam.min()), (-1.0, 1.0))(fam)
                except:
                    fs = np.zeros_like(fam)
                score = nav * KN + fs * (1.0 - KN)
            db.add([heading, nav.mean(), fam.mean(), trg.mean(), score.mean()])
        self.correlations.clear()
        db.process()
        with open(CWD / "look_around.list", "at") as lst:
            lst.writelines(db)
        if db.has_candidates():
            heading, _ = db.next_candidate()
            self.turn_to(heading)
        else:
            self.get_logger().warn("Look around failed, retrying ...")
            self.look_around(direction, initial=True)


@ros_entry
def main():
    robot = Navigation()
    input = robot.input()
    logger = protocol.logger = robot.get_logger()
    # Control Rate Throttling
    interval: float = 1.0 / 25.0
    next_loop: float = now() + interval
    count = Count()
    first_correlation = True
    # Wait for input to be ready
    logger.info(f"Waiting for upstream ...")
    while not len(input):
        spin_once(robot, timeout_sec=1e-3)
    logger.info(f"Starting initial look around")
    robot.look_around(0.2, initial=True)
    while ok():
        while now() < next_loop:
            spin_once(robot, timeout_sec=next_loop - now())
        next_loop += interval
        for ts, correlation in input.dump():
            robot.decision_delay = now() - ts
            with Timer(*count("Decision".ljust(10)), print=logger.debug, origin=ts):
                robot.correlations.append((ts, correlation))
            if robot.msg is not None:
                logger.info(robot.msg)
                robot.msg = None
        if robot.wait_action():
            pass
        elif (
            robot.trapped_for() > robot.trap_duration
            and not first_correlation
            # Uncomment this to stop in front of the first target
            and robot.mixer.state is None
        ):
            duration = robot.trapped_for()
            reason = ", ".join(robot.trapped_reason())
            logger.info(str(robot.trapped_by))
            logger.info(f"Trapped for {duration:.2f}s ({reason})")
            robot.mixer.reset()
            robot.look_around(0.2 if robot.vel[2] >= 0.0 else -0.2)
            if robot.trapped_by["halt"]:
                # Back off before starting look around
                robot.move_for(1.0, (-0.4, 0.0, 0.0))
        elif len(robot.correlations):
            if first_correlation:
                robot.declare_trapped("odometry", False)
                robot.accumulator.reset()
                first_correlation = False
            msg = None
            if robot.is_trapped():
                duration = robot.trapped_for()
                reason = ", ".join(robot.trapped_reason())
                msg = f"Trapped for {robot.trapped_for():.2f}s ({reason})"
            # Normal operation
            ts, correlation = robot.correlations[-1]
            robot.correlations.clear()
            robot.last_msg = None
            motion = robot.mixer(correlation)
            if robot.mixer.state is not None:
                # Target identified
                msg = robot.mixer.state
                for r, v in zip(robot.rmp, motion):
                    # Skip ramping
                    r.reset(v)
            robot.motion(motion, msg=msg)
        else:
            # Keep ramping previously demanded velocities
            robot.motion(None, msg=None)
    return robot
