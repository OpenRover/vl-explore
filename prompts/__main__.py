# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from . import Prompt
from random import random
from util.logger import Logger

log = Logger(__file__)

for _ in ["navigation", "nav-left", "nav-right", "nav-center", "target"]:
    log.info(f"Generating prompt: {_}")
    nav = Prompt(_)

    for t, v in nav:
        if random() > 0.05:
            continue
        log.debug(f"[{_}] {t} =>", *nav(v))
