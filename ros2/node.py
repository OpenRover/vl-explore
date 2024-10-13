# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from math import atan2, asin, degrees, sqrt
from time import time as now
from rclpy import ok, spin_once
from rclpy.node import Node as ROS2Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from numpy import ndarray
from util.iter import flatten
from util.math import ang_diff, clamp, sign
from util.action import Action
from util.graphics import Region, TextBox


def bgr8(image: ndarray) -> ndarray:
    if image.shape[2] == 1:
        image = image.repeat(3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.shape[2] != 3:
        raise ValueError("Invalid image format")
    return image


vt_clamp = clamp(0, 1.0)
vr_clamp = clamp(-1.0, 1.0)

t_box: TextBox = None
region: Region = None


def banner(frame: ndarray, text: str):
    global t_box, region
    if t_box is None or region is None:
        h, w = frame.shape[:2]
        region = Region(0, h - 50, w, 50)
        t_box = TextBox(region, align="center", color=(255, 128, 64))
    frame[region.slice_y, region.slice_x] = region(frame) * 0.6
    t_box(frame, text)


class Node(ROS2Node):

    def __init__(self):
        super().__init__("perception")
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, "image_in", self.handle_image_msg, 10
        )
        self.imu_sub = self.create_subscription(Imu, "imu", self.handle_imu_msg, 10)
        self.halt_sub = self.create_subscription(Bool, "halt", self.handle_halt_msg, 10)
        self.image_pub = self.create_publisher(Image, "image_out", 10)
        self.motion_pub = self.create_publisher(Twist, "motion", 10)
        self.get_logger().info("Waiting for initial messages")
        while ok():
            spin_once(self)
            if self.image is None:
                continue
            if self.stamp is None:
                continue
            if self.attitude is None:
                continue
            break
        self.get_logger().info("Perception node initialized")

    def __call__(
        self,
        vx: float | int = 0,
        vy: float | int = 0,
        vz: float | int = 0,
        detect_trap: bool = True,
    ):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(vz)
        self.motion_pub.publish(msg)
        if detect_trap:
            self.forward_motion = vx >= 0.0
            self.check_trapped()
        else:
            # Lack of motion does not indicate being trapped
            self.forward_motion = True

    image: ndarray = None
    stamp: Time = None

    def handle_image_msg(self, msg: Image):
        self.image = bgr8(self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))
        self.stamp = msg.header.stamp

    attitude: list[float] = None

    def handle_imu_msg(self, msg: Imu):
        # Convert attitude quaternion to euler angles
        x, y, z, w = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = asin(2.0 * (w * y - z * x))
        yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # Convert to degrees
        self.attitude = list(ang_diff(0.0, degrees(x)) for x in [roll, pitch, yaw])

    # Signals stopping the robot from exploring further
    forward_motion: bool = False
    halt_signal: bool = False
    # Robot is considered trapped if halted for too long.
    trapped_since: float | None = None
    # Robot has to be free to move for a certain duration without halt before it can be considered free.
    free_to_move_since: float | None = now()

    def check_trapped(self):
        if self.halt_signal or not self.forward_motion:
            self.free_to_move_since = None
            if self.trapped_since is None:
                self.trapped_since = now()
        elif self.free_to_move_since is None:
            self.free_to_move_since = now()
        return self.trapped_since

    def handle_halt_msg(self, msg: Bool):
        self.halt_signal = bool(msg.data)
        self.check_trapped()

    def trapped_for(self, free_threshold: float = 2.0) -> float:
        trapped_since = self.check_trapped()
        if trapped_since is None:
            return 0.0
        t = now()
        if self.free_to_move_since is not None:
            free_duration = t - self.free_to_move_since
            if free_duration > free_threshold:
                self.trapped_since = None
                return 0.0
        trap_duration = float(t - self.trapped_since)
        return trap_duration

    def grab(self):
        image, stamp = self.image, self.stamp
        self.image = None
        self.stamp = None
        valid = image is not None and stamp is not None
        return image, stamp, valid

    def publish_image(self, image: ndarray, stamp: Time):
        msg = self.bridge.cv2_to_imgmsg(bgr8(image), encoding="bgr8")
        msg.header.stamp = stamp
        self.image_pub.publish(msg)

    actions: list[Action] = []

    @Action.action
    def turn_to(self, heading: float, kv: float = 0.2, tolerance: float = 1.0):
        """Turn to a specific heading, stops when angular error is within tolerance"""
        _, _, rz = self.attitude

        yield {
            "render": lambda frame: banner(
                frame, f"Turning from {rz:.2f} deg => {heading:.1f} deg"
            )
        }

        while True:
            _, _, rz = self.attitude
            dr = ang_diff(rz, heading)
            if abs(dr) < tolerance:
                yield 0.0, 0.0, 0.0, False
                break
            else:
                vr = vr_clamp(sign(dr) * sqrt(abs(dr / 30.0))) * kv
                if abs(vr) < 0.2:
                    vr = sign(vr) * 0.2
                yield 0.0, 0.0, vr, False

    @Action.action
    def look_around(self, direction: float):
        """Perform a 360 degree turn around, find best heading to go next"""
        # (heading, confidence)
        database: list[tuple[float, float]] = []
        prev_rz: float = self.attitude[2]
        accumulated_angle: float = 0.0

        def render(frame: ndarray):
            progress = "{:.1f}".format(100.0 * abs(accumulated_angle) / 360.0)
            banner(frame, f"Looking around - {progress.rjust(5, '0')}%")

        yield {"render": render}

        while abs(accumulated_angle) < 360.0 or len(database) < 10:
            # Continue turning around
            confidence = yield 0.0, 0.0, direction, False
            _, _, rz = self.attitude
            database.append((rz, confidence))
            dr = ang_diff(prev_rz, rz, direction=direction)
            accumulated_angle += dr
            prev_rz = rz
        self.get_logger().info(
            "Look around completed ({accumulated_angle:.2f}°) with {len(database)} samples"
        )
        # Find the best heading to go next
        best_heading, best_confidence = None, 0.0
        for heading, confidence in database:
            if confidence > best_confidence:
                best_heading, best_confidence = heading, confidence
        if best_heading is not None:
            return self.turn_to(best_heading)
        else:
            self.get_logger().warn("No plausible way found in look around database")
            # Try again
            return self.look_around(direction)

    def publish_motion(self, confidence: list[list[float]]):
        confidence = list(flatten(confidence))
        assert len(confidence) == 6
        EPS = 1e-2
        confidence = list(zip(confidence[:3], confidence[3:]))

        def fusion(f: float, n: float) -> float:
            wf, wn = 0.2, 0.8
            if n <= EPS:
                return n
            elif f >= n:
                return wf * f + wn * n
            else:
                return n

        l, c, r = (fusion(f, n) for f, n in confidence)

        if len(self.actions) > 0:
            # One or more action is currently in progress
            action = self.actions[-1]
            val, done = action(c)
            if done:
                self.actions.pop()
            if isinstance(val, Action):
                self.actions.append(val)
            elif isinstance(val, (list, tuple)):
                self(*val)
            elif val is not None:
                raise ValueError(f"Invalid action return value {val}")
        elif self.trapped_for() >= 5.0:
            # Continuous halt during normal operation
            self.actions.append(self.look_around(0.2 if l >= r else -0.2))
        else:
            # Normal operation
            # Range 0.0 ~ 1.0
            forward = vt_clamp(c * 1.2)
            # Range -1.0 ~ +1.0
            distraction = vr_clamp(l - r)
            if abs(distraction) > EPS:
                if forward > EPS:
                    # Turn down distraction term when moving forward
                    sweep, turn = [distraction / 4.0] * 2
                else:
                    # Turn around in the same spot
                    sweep, turn = 0.0, distraction / 4.0
            else:
                sweep, turn = 0.0, 0.0
            # Back off only when both turn and forward are zero
            if forward < EPS and abs(distraction) < EPS:
                forward, sweep, turn = -0.2, 0.0, 0.0
            # Publish motion
            self(forward, sweep, turn)
