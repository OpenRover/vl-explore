# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from rclpy.node import Node as ROS2Node
from rclpy.subscription import Subscription
from rclpy.publisher import Publisher
from rclpy.time import Time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from numpy import ndarray


class Node(ROS2Node):
    image_sub: Subscription
    image_pub: Publisher

    def __init__(self):
        super().__init__("perception")
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, "image_in", self.handle_image_msg, 10
        )
        self.image_pub = self.create_publisher(Image, "perception", 10)

    image: ndarray = None
    stamp: Time = None

    def handle_image_msg(self, msg: Image):
        self.image = self.bridge.imgmsg_to_cv2(msg)
        self.stamp = msg.header.stamp

    def grab(self):
        image, stamp = self.image, self.stamp
        self.image = None
        self.stamp = None
        valid = image is not None and stamp is not None
        return image, stamp, valid

    def publish_image(self, image: ndarray, stamp: Time):
        msg = self.bridge.cv2_to_imgmsg(image)
        msg.header.stamp = stamp
        self.image_pub.publish(msg)
