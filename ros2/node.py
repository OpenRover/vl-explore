# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from rclpy.node import Node as ROS2Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from numpy import ndarray


def bgr8(image: ndarray) -> ndarray:
    if image.shape[2] == 1:
        image = image.repeat(3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.shape[2] != 3:
        raise ValueError("Invalid image format")
    return image


class Node(ROS2Node):

    def __init__(self):
        super().__init__("perception")
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, "image_in", self.handle_image_msg, 10
        )
        self.image_pub = self.create_publisher(Image, "image_out", 10)
        self.motion_pub = self.create_publisher(Twist, "motion", 10)

    image: ndarray = None
    stamp: Time = None

    def handle_image_msg(self, msg: Image):
        self.image = bgr8(self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))
        self.stamp = msg.header.stamp

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

    def publish_motion(self, confidence: list[float]):
        assert len(confidence) == 3
        EPS = 1e-2
        l, c, r = confidence
        # Range 0.0 ~ 1.0
        forward = max(0.0, min(1.0, c * 2))
        # Range -1.0 ~ +1.0
        distraction = max(-1.0, min(1.0, (l - r)))
        if abs(distraction) > EPS:
            if forward > EPS:
                sweep, turn = [distraction / 2.0] * 2
            else:
                # Turn around in the same spot
                sweep, turn = 0.0, distraction
        else:
            sweep, turn = 0.0, 0.0
        # Back off only when both turn and forward are zero
        if forward < EPS and abs(distraction) < EPS:
            forward, sweep, turn = -0.2, 0.0, 0.0
        msg = Twist()
        msg.linear.x = float(forward)
        msg.linear.y = float(sweep)
        msg.angular.z = float(turn)
        self.motion_pub.publish(msg)
