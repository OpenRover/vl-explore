# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from math import sqrt
from time import time as now
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from . import Node, ros_entry, spin_once, ok
from .utils import attitude_from_quaternion, TravelAccumulator

from util import types
from util.sockets import Server, Client
from util.action import Action
from util.math import ang_diff, clamp, sign, interpolate
from util.timer import Timer

from . import protocol, socket_correlator, socket_navigation

vt_clamp = clamp(0, 1.0)
vr_clamp = clamp(-1.0, 1.0)


class Navigation(Node):

    actions: list[Action] = []

    def __init__(self):
        super().__init__("navigation")
        self.mixer = self.strategy.MotionMixer()
        # I/O Sockets
        self.input = Client(socket_correlator, logger=self.get_logger())
        self.output = Server(socket_navigation, logger=self.get_logger())
        # Timeout (seconds) before trapping signal is asserted
        self.declare_parameter("trap_duration", 5.0)
        self.trap_duration = float(self.get_parameter("trap_duration").value)
        # Distance (meters) within which the robot is considered being trapped
        self.declare_parameter("trap_distance", 0.2)
        self.trap_distance = float(self.get_parameter("trap_distance").value)
        # Motion publisher
        self.motion_pub = self.create_publisher(Twist, "motion", 10)
        # Roll, Pitch, Yaw
        self.attitude = [now()] + [0.0] * 3
        # Travel Accumulator
        self.accumulator = TravelAccumulator(self.trap_duration)
        # Create subscription
        self.halt_sub = self.create_subscription(Bool, "halt", self.handle_halt_msg, 10)
        self.odom_sub = self.create_subscription(
            Odometry, "odometry", self.handle_odom_msg, 10
        )

    motion: types.Motion = [0.0, 0.0, 0.0]

    def __call__(self, vx: float | int = 0, vy: float | int = 0, vz: float | int = 0):
        motion = Twist()
        vx, vy, vz = map(float, (vx, vy, vz))
        motion.linear.x = vx
        motion.linear.y = vy
        motion.angular.z = vz
        self.motion_pub.publish(motion)
        # Assemble navigation message for socket communication
        self.motion = [vx, vy, vz]
        msg = None
        if len(self.actions):
            action = self.actions[-1]
            if "info" in action and callable(action["info"]):
                msg = action["info"]()
            elif hasattr(action, "name"):
                msg = f"Executing action: {action.name}"
            else:
                msg = "Executing unknown action"
        self.output(*protocol.Motion.encode(now(), self.motion, msg))

    trapped_since: float | None = None
    trapped_by: dict[str, bool] = {}

    def is_trapped(self):
        return any(self.trapped_by.values())

    def trapped_for(self):
        if self.trapped_since is None:
            return 0.0
        return now() - self.trapped_since

    def declare_trapped(self, reason: str, trapped: bool):
        self.trapped_by[reason] = trapped
        if self.is_trapped():
            if self.trapped_since is None:
                self.trapped_since = now()
        else:
            self.trapped_since = None

    def handle_halt_msg(self, halt: Bool):
        self.declare_trapped("halt", halt.data)

    def handle_odom_msg(self, msg: Odometry):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.attitude = [ts] + attitude_from_quaternion(msg.pose.pose.orientation)
        travel = self.accumulator.update(msg)
        if travel is not None:
            distance = travel.norm()
            if distance < self.trap_distance:
                self.declare_trapped("odometry", True)
            else:
                self.declare_trapped("odometry", False)
        else:
            # Odometry not initialized, assume free to move
            self.declare_trapped("odometry", False)

    correlations: list[types.CorrelationStamped] = []

    @Action.action
    def turn_to(self, heading: float, kv: float = 0.2, tolerance: float = 1.0):
        """Turn to a specific heading, stops when angular error is within tolerance"""
        ts, rx, ry, rz = self.attitude

        yield {"info": lambda: f"Turning from {rz:.2f} deg => {heading:.1f} deg"}

        while True:
            ts, rx, ry, rz = self.attitude
            dr = ang_diff(rz, heading)
            if abs(dr) < tolerance:
                yield 0.0, 0.0, 0.0
                break
            else:
                vr = vr_clamp(sign(dr) * sqrt(abs(dr / 30.0))) * kv
                if abs(vr) < 0.2:
                    vr = sign(vr) * 0.2
                yield 0.0, 0.0, vr

        self.trapped_since = None

    @Action.action
    def look_around(self, direction: float):
        """Perform a 360 degree turn around, find best heading to go next"""
        # (timestamp, heading)
        trj: list[tuple[float, float]] = []
        prev_rz: float = self.attitude[3]
        accumulated_angle: float = 0.0
        # Clear existing correlation database
        self.correlations.clear()

        def info():
            progress = "{:.1f}".format(100.0 * abs(accumulated_angle) / 360.0)
            return f"Looking around - {progress.rjust(5, '0')}%"

        yield {"info": info}

        direction = vr_clamp(direction)
        while abs(accumulated_angle) < 360.0 or len(trj) < 10:
            # Continue turning around
            yield 0.0, 0.0, direction
            ts, _, _, rz = self.attitude
            trj.append((ts, rz))
            dr = ang_diff(prev_rz, rz)
            accumulated_angle += dr
            prev_rz = rz
        # Create a mapping between timestamp and heading
        t2r = interpolate(*trj)
        database = list((t2r(t), self.mixer(c)[0]) for t, c in self.correlations)
        self.correlations.clear()
        # Find the best heading to go next
        best_heading, best_confidence = None, 0.0
        for heading, confidence in database:
            if confidence > best_confidence:
                best_heading, best_confidence = heading, confidence
        self.trapped_since = None
        if best_heading is not None:
            return self.turn_to(best_heading)
        else:
            self.get_logger().warn("No plausible way found in look around database")
            # Try again
            return self.look_around(direction)


@ros_entry
def main():
    node = Navigation()
    logger = protocol.logger = node.get_logger()
    # Control Rate Throttling
    interval: float = 1.0 / 20.0
    next_loop: float = now()
    while ok():
        while True:
            spin_once(node)
            if now() >= next_loop:
                next_loop += interval
                break
        for ts, correlation in protocol.Correlation.decode(node.input):
            with Timer(print=logger.info, origin=ts):
                node.correlations.append((ts, correlation))
        if len(node.actions):
            # One or more action is currently in progress
            action = node.actions[-1]
            val, done = action()
            if done:
                node.actions.pop()
            if isinstance(val, Action):
                node.actions.append(val)
            elif isinstance(val, (list, tuple)):
                node(*val)
            elif val is not None:
                raise ValueError(f"Invalid action return value {val}")
        elif node.trapped_for() > node.trap_duration:
            vr = 0.2 if node.motion[2] >= 0.0 else -0.2
            node.actions.append(node.look_around(vr))
        elif len(node.correlations):
            # Normal operation
            *_, (ts, correlation) = node.correlations
            node.correlations.clear()
            node(*node.mixer(correlation))
    return node
