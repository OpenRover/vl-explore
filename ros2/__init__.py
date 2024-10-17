# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Callable
from pathlib import Path
from rclpy import init, spin_once, ok
from rclpy.node import Node as ROS2Node
from torch import Tensor

from lib.strategies import use

from . import __protocol__ as protocol

tmp = Path("/tmp")

socket_perception = tmp / "rover-master-perception.sock"
socket_correlator = tmp / "rover-master-correlator.sock"
socket_navigation = tmp / "rover-master-navigation.sock"

class Node(ROS2Node):
    def __init__(self, name: str):
        super().__init__(name)
        self.declare_parameter("strategy", "6T1P")
        self.strategy = use(str(self.get_parameter("strategy").value))
        self.mixer = self.strategy.MotionMixer()

def ros_entry(main: Callable[..., Node | list[Node] | None]):
    def wrapper(*args, **kwargs):
        init()
        ret = None
        try:
            ret = main(*args, **kwargs)
        except KeyboardInterrupt:
            pass
        finally:
            if not isinstance(ret, list):
                ret = [ret]
            for node in ret:
                if isinstance(node, Node):
                    node.destroy_node()
    return wrapper
