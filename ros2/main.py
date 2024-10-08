# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from .node import Node
from rclpy import init, ok, spin_once

from prompts import Prompt
import models.clip as clip
import lib.navigation as Nav
from util.iter import flatten


def main():
    # Initialize ROS2 node
    init()
    nav: Nav.Navigation = None
    node = Node()
    node.actions.append(node.look_around(0.2))
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
            # Generate Motion Command based on Confidence
            node.publish_motion(confidence)
            # Publish the image
            nav.render(frame, pred, confidence)
            if len(node.actions) and "render" in node.actions[-1]:
                node.actions[-1]["render"](frame)
            node.publish_image(frame, stamp)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
