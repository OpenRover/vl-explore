# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Callable, Any, overload
import argparse, torch
from pathlib import Path
from os import getcwd

CWD = Path(getcwd())

from . import HOME
from .logger import Logger

log = Logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--dataset", type=str, default="nav")
parser.add_argument("--display", action="store_true")
parser.add_argument("--frame-skip", type=int, default=0)
args, unknown = parser.parse_known_args()

device: torch.device = None
to_device_args: dict[str, Any] = {}


def select_device(override: str | None = None):
    global device, to_device_args
    if override is None:
        if device is not None:
            return
        else:
            override = str(args.device)
    to_device_args = {}
    if override is not None:
        device = torch.device(override)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        to_device_args["non_blocking"] = True
    elif torch.backends.mkl.is_available():
        device = torch.device("mkl")
    else:
        device = torch.device("cpu")

    log.info(f"using device: {device}")


@overload
def to_device(
    item: torch.Tensor, device: str | torch.device, **kwargs
) -> torch.Tensor: ...


@overload
def to_device(
    item: torch.nn.Module, device: str | torch.device, **kwargs
) -> torch.nn.Module: ...


def to_device(item: torch.Tensor | torch.nn.Module, device=None, **kwargs):
    if device is None:
        device = globals()["device"]
    return item.to(device=device, **to_device_args, **kwargs)


def on_device(device=None, **kwargs):
    if device is None:
        device = globals()["device"]
    def decorator(fn: Callable[[Any], torch.Tensor | torch.nn.Module]):
        def wrapper(*args, **_kwargs):
            return to_device(fn(*args, **_kwargs), device=device, **kwargs)

        return wrapper

    return decorator
