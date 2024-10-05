# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from .node import Node
from rclpy import init, ok, spin_once
import models.clip as clip, cv2, numpy as np
from prompts import Prompt
import lib.navigation as Nav
from util.iter import flatten


def main():
    # Initialize ROS2 node
    init()
    nav: Nav.Navigation = None
    node = Node()
    try:
        while ok():
            spin_once(node)
            image, stamp, valid = node.grab()
            if not valid:
                continue
            if nav is None:
                # Load prompts
                prompts = [Prompt("navigation")]
                h, w = image.shape[:2]
                # Release CLIP text model from memory
                clip.text_model = None
                nav = Nav.Nav6T1P(prompts, (w, h), (1280, 720))
            pred, confidence, frame = nav(image)
            l, c, r = (f"{s:.4f} ({t})" for t, s in list(flatten(pred, 2))[3:])
            node.get_logger().info(
                "\n\t".join(
                    [
                        f"Navigation:",
                        f"L | {l}",
                        f"C | {c}",
                        f"R | {r}",
                    ]
                )
            )
            node.get_logger().info(f"Confidence: {confidence}")
            # Generate Motion Command based on Confidence
            node.publish_motion(confidence)
            # Publish the image
            nav.render(frame, pred, confidence)
            node.publish_image(frame, stamp)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
