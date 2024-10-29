# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import os, sys, cv2
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from subprocess import Popen
from multiprocessing import Pool
from io import TextIOWrapper

from lib.strategies import use
from lib.slicer import shape
from lib.renderer import Renderer

from util.queue import Queue
from util.logger import Logger, create_logger
from util.exception import Expect
import util.JSON as JSON

from . import protocol

log = Logger(__file__)
parser = ArgumentParser()
parser.add_argument("--strategy", default="6T1P", help="Strategy to use")
parser.add_argument("--dir", help="Run directory")
parser.add_argument("--src", help="Dir name of raw images", default="recording")
parser.add_argument("--dst", help="Output Directory", default="rendering")
parser.add_argument("--resize", help="Resize Factor", type=float, default=1.0)
args = parser.parse_args()
CWD = Path(os.path.realpath(str(args.dir)))
SRC = CWD / str(args.src)
DST = CWD / str(args.dst)

size_factor = float(args.resize)
should_resize = size_factor != 1.0

assert SRC.is_dir(), f"Invalid source directory: {SRC}"
DST.mkdir(exist_ok=True, parents=True)

FF_CONCAT = Path(f"{CWD}/render.list")
VIDEO = Path(f"{CWD}.mp4")

if VIDEO.is_dir():
    raise IsADirectoryError(VIDEO)


def confirm(question: str, auto_rej: bool = False) -> bool:
    while True:
        dfl = "(n) " if auto_rej else ""
        match input(f"{question} [Y/n] {dfl}").lower():
            case "y":
                return True
            case "n":
                return False
            case "":
                if auto_rej:
                    return False
        print(f"Please respond with 'Y' or 'n'")


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
    intervals = list[float]()

    class Outdated(Exception):
        pass

    def forward(self, init: bool = False):
        if init:
            self.t0, self.d0 = next(self.src)
            self.t1, self.d1 = next(self.src)
        else:
            self.t0, self.d0 = self.t1, self.d1
            self.t1, self.d1 = next(self.src)
        assert self.t1 >= self.t0, (self.t0, self.t1)
        self.interval = self.t1 - self.t0

    def __init__(self, src: TextIOWrapper, protocol: type[protocol.JsonProtocol]):

        def gen():
            for line in src:
                for ts, *_ in self.protocol.decode(line):
                    self.intervals.append(ts)
                    yield ts, line.strip()

        self.src = gen()
        self.protocol = protocol
        self.forward(init=True)

    def __call__(self, ts: float) -> tuple[float, str]:
        while ts >= self.t1:
            self.forward()
        if ts < self.t0:
            raise self.Outdated()
        return self.t0, self.d0

    def freq(self, N: int = 5):
        self.intervals = self.intervals[-N:]
        if len(self.intervals) < 2:
            return 0
        dt = self.intervals[-1] - self.intervals[0]
        if dt <= 0:
            return 0
        n = len(self.intervals) - 1
        return n / dt


def fmt_time(t: float):
    m = t // 60
    s = t - m * 60
    h = m // 60
    m = m - h * 60
    return ":".join(str(int(t)).zfill(2) for t in (h, m, s))


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
        "-crf",       "23",       # Quality level (lower is higher quality)
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
        src, dst, cor, nav, duration, freq = arguments
        if Path(dst).is_file():
            return
        global renderer
        (_, correlation), *_ = protocol.Correlation.decode(cor)
        (_, delay, motion, banner), *_ = protocol.Motion.decode(nav)
        frame = cv2.imread(src)
        if should_resize:
            frame = cv2.resize(frame, None, fx=size_factor, fy=size_factor)
        assert type(delay) is float, delay
        if renderer is None:
            strategy = use(str(args.strategy))
            slicer = strategy.Slicer(shape(frame))
            renderer = strategy.Renderer(slicer)
        stats = [
            f"duration: {fmt_time(duration)}",
            f"frequency: {freq:.2f} Hz",
            f"delay: {1000.0 * delay:.2f} ms",
        ]
        renderer(frame, correlation)
        # blurred = renderer.blur(frame)
        blurred = None
        renderer.stats(frame, " | ".join(stats), blurred=blurred)
        if banner is None:
            sign = lambda x: "+" if x > 0 else "-"
            motion = [f"{sign(v)}{abs(v):.2f}" for v in motion]
            motion = [f"{k} {v}" for k, v in zip("XYR", motion)]
            banner = f"Normal operation [ {' | '.join(motion)} ]"
            renderer.banner(frame, banner, color=(192, 192, 192), blurred=blurred)
        else:
            renderer.banner(frame, banner, blurred=blurred)
        cv2.imwrite(dst, frame)
    except Exception as e:
        return e


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
    results, durations = [], []
    t0 = None
    prev_ts = None

    def gen():
        with (
            Expect(Queue.Closed, KeyboardInterrupt, StopIteration),
            open(CWD / "images.list", "rt") as img,
            open(CWD / "correlation.list") as cor,
            open(CWD / "navigation.list") as nav,
        ):
            nonlocal t0, prev_ts
            skipped = 0
            first_skip: float = None

            M1 = Matcher(cor, protocol.Correlation)
            M2 = Matcher(nav, protocol.Motion)

            for ts, filename in map(parse, img):
                if prev_ts is None:  # First frame
                    prev_ts = ts
                t1 = ts
                try:
                    t1, d1 = M1(ts)  # Correlation
                    t2, d2 = M2(ts)  # Navigation
                except Matcher.Outdated:
                    if first_skip is None:
                        first_skip = ts
                    skipped += 1
                    continue
                if skipped > 0:
                    skip_duration = ts - first_skip
                    msg = f"{skipped} frames skipped ({skip_duration:.2f} seconds)"
                    log.warn(msg, print=progress.write)
                    skipped = 0
                    first_skip = None
                    progress.update(skipped)
                if t0 is None:
                    t0 = min(ts, t1, t2)
                results.append(filename)
                durations.append(ts - prev_ts)
                src = str(SRC / filename)
                dst = str(DST / filename)
                duration = ts - t0
                yield src, dst, d1, d2, duration, M1.freq()
                prev_ts = ts

    with Pool(os.cpu_count()) as pool:
        for msg in pool.imap_unordered(render_frame, gen()):
            if msg is not None:
                log.error(str(msg), print=progress.write)
            progress.update()

    progress.close()
    return results, durations


if __name__ == "__main__":
    results, durations = main()
    ffmpeg = FFMPEG()

    durations = durations[1:] + [durations[-1]]

    with open(FF_CONCAT, "wt") as f:
        f.write("ffconcat version 1.0\n")
        cwd = FF_CONCAT.parent
        last_file = None
        for filename, duration in zip(results, durations):
            path = Path(DST / filename)
            if not path.exists():
                log.warn(f"Missing frame: {path.relative_to(cwd)}")
                continue
            filename = path.relative_to(cwd)
            f.write(f"file '{filename}'\n")
            f.write(f"duration {duration:.4f}\n")
    log.info(f"total {len(results)} frames to be encoded")
    try:
        ff_log = create_logger(None, "FFMPEG", "cyan", "light_grey")
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
