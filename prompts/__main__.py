# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from . import Prompt
from random import random
from util.env import Logger

log = Logger(__file__)

nav = Prompt("navigation")

for t, v in nav:
    if random() > 0.05:
        continue
    log.debug(t, "=>", *nav(v))
