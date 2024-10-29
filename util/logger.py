# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import sys
from pathlib import Path
from termcolor import colored
from typing import Callable, Any

from . import HOME


def format(*msgs, sep: str):
    return sep.join(map(str, msgs))

def print(msg: str, end="\n", **kwargs):
    sys.stderr.write(msg + end)
    sys.stderr.flush()


def create_logger(
    ID: str | None,
    level: str | None,
    level_color: str | None = None,
    msg_color: str | None = None,
    **kwargs,
):
    if level is None:
        level = ""
    else:
        level = f"[{level.upper().center(6)}]"
        if level_color is not None:
            level = colored(level, level_color)
    
    if ID is None:
        ID = ""
    else:
        ID = colored(f" {ID}:", "light_grey")

    def log(*msgs: str, print: Callable[[str], Any] = print, sep=" ", **_kwargs):
        if msg_color is not None:
            msgs = [colored(msg, msg_color) for msg in msgs]
        msg = format(level + ID, *msgs, sep=sep)
        print(msg, **kwargs, **_kwargs)

    return log


class Logger:
    kwargs = {}

    def __init__(self, src: str, **kwargs):
        """
        Usage: logger = Logger(__file__)
        """
        try:
            ID = str(Path(src).relative_to(HOME))
        except:
            ID = src
        self.debug = create_logger(ID, "DEBUG", "blue", "cyan", **kwargs)
        self.verbose = create_logger(ID, "VERBO", "light_grey", "light_grey", **kwargs)
        self.info = create_logger(ID, "INFO", "green", "white", **kwargs)
        self.warn = create_logger(ID, "WARN", "yellow", "light_yellow", **kwargs)
        self.error = create_logger(ID, "ERROR", "red", "light_red", **kwargs)

    def __call__(self, other_logger):
        """
        Use provided logger for levels defined by it.
        """
        for key in ["debug", "verbose", "info", "warn", "error"]:
            if hasattr(other_logger, key):
                other = getattr(other_logger, key)
                if callable(other):
                    setattr(self, key, other)
