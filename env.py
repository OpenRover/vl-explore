# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import argparse, torch, sys
from pathlib import Path
from termcolor import colored

HOME = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--dataset", type=str, default="nav")
parser.add_argument("--display", action="store_true")
parser.add_argument("--frame-skip", type=int, default=0)
args = parser.parse_args()


def logger(
    ID: str,
    level: str,
    level_color: str | None = None,
    msg_color: str | None = None,
    **kwargs,
):
    level = f"[{level.upper().center(6)}]"
    if level_color is not None:
        level = colored(level, level_color)

    def log(*msgs: str, **_kwargs):
        if msg_color is not None:
            msgs = [colored(msg, msg_color) for msg in msgs]
        msg, *_msgs = msgs
        print(f"{level} {ID}: {msg}", *_msgs, **kwargs, **_kwargs)

    return log


class Logger:
    kwargs = {}

    def __init__(self, src: str, file=sys.stderr, **kwargs):
        """
        Usage: logger = Logger(__file__)
        """
        ID = str(Path(src).relative_to(HOME))
        self.debug = logger(ID, "DEBUG", "blue", "cyan", file=file, **kwargs)
        self.verbose = logger(
            ID, "VERBO", "light_grey", "light_grey", file=file, **kwargs
        )
        self.info = logger(ID, "INFO", "green", "light_grey", file=file, **kwargs)
        self.warning = logger(ID, "WARN", "yellow", "light_yellow", file=file, **kwargs)
        self.error = logger(ID, "ERROR", "red", "light_red", file=file, **kwargs)


log = Logger(__file__)


def select_device(override: str | None = args.device) -> torch.device:
    kwargs = {}
    if override is not None:
        device = torch.device(override)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        kwargs["non_blocking"] = True
    elif torch.backends.mkl.is_available():
        device = torch.device("mkl")
    else:
        device = torch.device("cpu")
    return device, kwargs


device, to_device_args = select_device()


def to_device(item: torch.Tensor | torch.nn.Module, device=device, **kwargs):
    return item.to(device=device, **to_device_args, **kwargs)


log.info(f"using device: {device}")
