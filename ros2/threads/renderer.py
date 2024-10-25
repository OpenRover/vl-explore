# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import os, sys, cv2, atexit
from threading import Thread
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from time import sleep
from subprocess import Popen, PIPE
from multiprocessing import Pool
from io import TextIOWrapper

from lib.strategies import use
from lib.slicer import shape
from lib.renderer import Renderer

from util.queue import Queue
from util.timer import Duration
from util.logger import Logger, create_logger
from util.exception import Expect
import util.JSON as JSON

from . import protocol

log = Logger(__file__)
parser = ArgumentParser()
parser.add_argument("--strategy", default="6T1P", help="Strategy to use")
parser.add_argument("--dir", help="Run directory")
parser.add_argument("--src", help="Dir name of raw images")
parser.add_argument("--dst", help="Output Directory", default="rendering")
args = parser.parse_args()
CWD = Path(os.path.realpath(str(args.dir)))
SRC = CWD / str(args.src)
DST = CWD / str(args.dst)


assert SRC.is_dir(), f"Invalid source directory: {SRC}"
DST.mkdir(exist_ok=True, parents=True)

FF_CONCAT = Path(f"{CWD}/list.txt")
VIDEO = Path(f"{CWD}.mp4")

if VIDEO.is_dir():
    raise IsADirectoryError(VIDEO)


def confirm(question: str) -> bool:
    while True:
        match input(f"{question} [Y/n] "):
            case "Y":
                return True
            case "n":
                return False
            case _:
                print(f"Invalid response")


if VIDEO.exists():
    if confirm(f"Overwrite {VIDEO}?"):
        VIDEO.unlink()
    else:
        raise FileExistsError(VIDEO)


class Matcher:
    t0: float = None
    d0: str = None
    t1: float = None
    d1: str = None

    decay: float = 0.5
    interval: float = None

    class Outdated(Exception):
        pass

    def next(self):
        line = self.src.readline().strip()
        t, _ = line.split(",", 1)
        return float(t), line

    def forward(self, init: bool = False):
        if init:
            self.t0, self.d0 = self.next()
            self.t1, self.d1 = self.next()
        else:
            self.t0, self.d0 = self.t1, self.d1
            self.t1, self.d1 = self.next()
        assert self.t1 >= self.t0, (self.t0, self.t1)
        if init:
            self.interval = self.t1 - self.t0
        else:
            interval = self.t1 - self.t0
            w1, w2 = self.decay, 1 - self.decay
            self.interval = w1 * self.interval + w2 * interval

    def __init__(self, src: TextIOWrapper):
        self.src = src
        self.forward(init=True)

    def __call__(self, ts: float) -> tuple[float, str]:
        while ts >= self.t1:
            self.forward()
        if ts < self.t0:
            raise self.Outdated()
        return self.t0, self.d0


def ms(t: float):
    n = f"{1000.0 * t:.2f}"
    return f"{n} ms"


def hz(t: float):
    f = f"{1.0 / t:.2f}"
    return f"{f} Hz"


def FFMPEG():
    command = [
        # fmt: off
        "ffmpeg",
        "-f",         "concat",   # Input mux: concat
        "-i",         FF_CONCAT,  # Input file: list of images
        "-vsync",     "vfr",      # Variable frame rate
        "-vcodec",    "libx264",  # Use x264 codec for encoding
        "-preset",    "medium",   # Medium encoding preset
        "-profile:v", "main",     # Main profile
        "-pix_fmt",   "yuv420p",  # QuickTime compatibility
        "-crf",       "10",       # Quality level (lower is higher quality)
        VIDEO,  # Output file
    ]
    return list(map(str, command))


renderer: Renderer = None


def render_frame(arguments):
    # src: str
    # dst: str
    # cor: str
    # nav: str
    # duration: float
    # interval: float
    try:
        src, dst, cor, nav, duration, interval = arguments
        if Path(dst).is_file():
            return
        global renderer
        (_, correlation), *_ = protocol.Correlation.decode(cor)
        (_, delay, motion, banner), *_ = protocol.Motion.decode(nav)
        frame = cv2.imread(src)
        if renderer is None:
            strategy = use(str(args.strategy))
            slicer = strategy.Slicer(shape(frame))
            renderer = strategy.Renderer(slicer)
        stats = [
            f"duration: {Duration.format(duration)}",
            f"delay: {ms(delay)}",
            f"frequency: {hz(interval)}",
        ]
        renderer(frame, correlation)
        blurred = renderer.blur(frame)
        renderer.stats(
            frame, " | ".join(f"{k}: {v}" for k, v in stats), blurred=blurred
        )
        if banner is None:
            motion = [f"{"+" if v > 0 else "-"}{abs(v):.2f}" for v in motion]
            motion = [f"{k} {v}" for k, v in zip("XYR", motion)]
            banner = f"Normal operation [ {" | ".join(motion)} ]"
            renderer.banner(frame, banner, color=(128, 128, 128), blurred=blurred)
        else:
            renderer.banner(frame, banner, blurred=blurred)
        cv2.imwrite(dst, frame)
    except Exception as e:
        return str(e)


def main():
    def parse(line) -> tuple[float, str]:
        return JSON.parse(f"[{line}]")

    with open(CWD / "images.list", "rt") as f:
        lst = f.readlines()
        total = len(lst)
        _, sample_path = parse(lst[0])
        h, w = cv2.imread(str(SRC / sample_path)).shape[:2]
        del lst

    progress = tqdm(
        total=total,
        desc="Rendering",
        unit="frames",
        file=sys.stderr,
        dynamic_ncols=True,
        unit_scale=False,
        leave=False,
    )
    progress.update(0)
    count: int = 0
    t0: float = None
    t1: float = None
    outputs = []

    def gen():
        with (
            Expect(Queue.Closed, KeyboardInterrupt, EOFError),
            open(CWD / "images.list", "rt") as img,
            open(CWD / "correlation.list") as cor,
            open(CWD / "navigation.list") as nav,
        ):
            nonlocal count, t0, t1
            skipped = 0
            first_skip: float = None

            M1 = Matcher(cor)
            M2 = Matcher(nav)

            for ts, filename in map(parse, img):
                if t0 is None:
                    t0 = ts
                t1 = ts
                count += 1
                try:
                    t1, d1 = M1(ts)  # Correlation
                    t2, d2 = M2(ts)  # Navigation
                except M1.Outdated:
                    if first_skip is None:
                        first_skip = ts
                    skipped += 1
                    continue
                if skipped > 0:
                    duration = ts - first_skip
                    msg = f"{skipped} frames skipped ({duration:.2f} seconds)"
                    log.warn(msg, print=progress.write)
                    skipped = 0
                    first_skip = None
                    progress.update(skipped)
                if t0 is None:
                    t0 = min(ts, t1, t2)
                duration = ts - t0
                outputs.append(filename)
                src = str(SRC / filename)
                dst = str(DST / filename)
                yield src, dst, d1, d2, duration, M1.interval

    with Pool(os.cpu_count()) as pool:
        for msg in pool.imap_unordered(render_frame, gen()):
            if type(msg) is str:
                log.error(msg, print=progress.write)
            progress.update()

    progress.close()

    assert count > 0, "No frames to render"
    assert t0 is not None
    assert t1 is not None
    assert t1 > t0
    fps = (t1 - t0) / (count - 1)
    return fps, outputs


if __name__ == "__main__":
    interval, outputs = main()
    ffmpeg = FFMPEG()

    with open(FF_CONCAT, "wt") as f:
        f.write("ffconcat version 1.0\n")
        cwd = FF_CONCAT.parent
        last_file = None
        for filename in outputs:
            path = Path(DST / filename)
            if not path.exists():
                log.warn(f"Missing frame: {path.relative_to(cwd)}")
                continue
            filename = path.relative_to(cwd)
            f.write(f"file '{filename}'\n")
            f.write(f"duration {interval}\n")
        # Write the last frame again due to a known bug in ffmpeg
        f.write(f"file '{filename}'\n")
    try:
        ff_log = create_logger("STDERR", "FFMPEG", "cyan", "light_grey")
        ff_log("=" * 60)
        ff_log(" ".join(ffmpeg))
        ff_log("=" * 60)

        proc = Popen(ffmpeg)

        exitcode = proc.wait()
        if exitcode != 0:
            log.warn(f"FFMPEG exited with code {exitcode}")
        else:
            log.info(f"FFMPEG exited with code {exitcode}")
    except KeyboardInterrupt:
        log.warn("User interrupted")

    if VIDEO.exists():
        print()
        msg = f"Video rendered: {VIDEO}"
        print("=" * len(msg))
        log.info(msg)
        print("=" * len(msg))
        print()
        if confirm(f"Remove rendering folder {DST}?"):
            for path in tqdm(
                list(DST.glob("*")),
                desc="Removing",
                unit="files",
                leave=False,
                dynamic_ncols=True,
            ):
                path.unlink()
            DST.rmdir()
    else:
        log.warn("No video rendered")
        log.info("Command: " + " ".join(ffmpeg))
