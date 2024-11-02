# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import sys, matplotlib.pyplot as plt, numpy as np, math
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.artist import Artist
from typing import Literal, Callable, Iterable
from json import dumps, loads
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager

from util.math import period_constraint, project
from util.colors import *
from util.logger import Logger
from util.str import center
from util.exception import Expect

log = Logger(__file__)

CANVAS_SIZE = 6  # inches
DEG2RAD = np.pi / 180
CENTER = dict(ha="center", va="center")


def colorize(x: float, C1: np.ndarray, C2: np.ndarray, CM=WHITE[::-1], POWER=0.5):
    x = min(1.0, max(-1.0, x))
    C = C2 if x < 0 else C1
    x = abs(x) ** POWER
    return tuple(C * x + CM * (1.0 - x))


def alpha(c: np.ndarray, a: float = 1.0):
    return np.append(c, a)


def fill_range(x: np.ndarray, a: float, b: float, neutral: float = None):
    a, b = min(a, b), max(a, b)
    l, r = np.min(x), np.max(x)
    if neutral is None:
        scale = (b - a) / (r - l)
        neutral = -l * scale + a
    else:
        assert a <= neutral <= b, f"Neutral value {neutral} is out of range [{a}, {b}]"
        a, b = a - neutral, b - neutral
        scales = list[float]()
        if l < 0 and a < 0:
            scales.append(a / l)
        if r > 0 and b > 0:
            scales.append(b / r)
        scale = min(scales, default=1.0)
        assert scale > 0, f"Bad scale: {scale}"
    return x * scale + neutral, neutral, scale


circular = period_constraint(-np.pi, np.pi)


def gaussian(
    X: np.ndarray,
    Y: np.ndarray,
    Xw: np.ndarray | None = None,  # optional: window size of each sample
    sigma=None,
    n_samples: int = 180,
    **unused,
):
    has_positive = np.max(Y) > 0.0
    # Gaussian sample x coordinates - evenly spaced (per 1 degree)
    gX = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
    gY = np.zeros_like(gX)
    if sigma is None:
        sigma = 6.0 * np.pi / 180.0  # default initial spread
        max_attempts = 10
    else:
        max_attempts = 1
    while True:
        for i, mu in enumerate(gX):
            x = circular(X - mu)
            g: np.ndarray = np.exp(-np.power((x / sigma), 2) / 2)
            if Xw is not None:
                g *= Xw
            # Gaussian smoothing on confidence score
            gY[i] = np.dot(g, Y) / g.sum()
        if gY.max() > 0.0 or not has_positive or max_attempts <= 0:
            break
        sigma /= 2.0  # decrease spread
        max_attempts -= 1
    return gX, gY, dict(sigma=sigma, n_samples=n_samples)


def window_scan(X: np.ndarray, Y: np.ndarray):
    # Find continuous positive regions
    candidates = list[tuple[float, float]]()
    y = np.array([Y[-1], *Y, Y[0]])
    edges_l = np.where((y[:-2] <= 0.0) & (y[1:-1] > 0.0))[0]
    edges_r = np.where((y[1:-1] > 0.0) & (y[2:] <= 0.0))[0]
    if edges_r[0] < edges_l[0]:
        # Shift around
        tmp = edges_r[0]
        edges_r[:-1] = edges_r[1:]
        edges_r[-1] = tmp + len(Y)
    assert np.all(edges_l <= edges_r), f"Bad edges: {edges_l}, {edges_r}"
    # Extend to double period (so 359deg -> 001deg will fit)
    x = np.concatenate([X, X + 2 * np.pi])
    y = np.concatenate([Y, Y], axis=0)
    for l, r in zip(edges_l, edges_r + 1):
        xw, yw = x[l:r], y[l:r]
        assert np.all(yw > 0.0), f"Bad yw: {yw}"
        area = np.trapz(yw, xw)
        # Weighted average
        pos = np.sum(xw * yw) / np.sum(yw)
        hdg = circular(pos) / DEG2RAD
        candidates.append((hdg, area))
    return candidates


@dataclass(frozen=False)
class RenderContext:
    title: str | None = None
    theme: Literal["light", "dark"] = "light"
    north: float = 0.0
    arrow: bool = False
    turn_dir: Literal["L", "R", None] = None

    y_range: tuple[float, float] = (-2.0, 1.0)
    _fig: plt.Figure | None = None
    _ax: Axes | None = None

    @property
    def fig(self) -> plt.Figure:
        if self._fig is None:
            with self.rc_context:
                self._fig = plt.figure()
        self._fig.set_size_inches(CANVAS_SIZE, CANVAS_SIZE)
        return self._fig

    @fig.setter
    def fig(self, fig: plt.Figure):
        self._fig = fig

    content_ratio: float = 0.7

    @property
    def ax(self) -> Axes:
        if self._ax is None:
            with self.rc_context:
                self.fig.clf()
                s = self.content_ratio
                d = (1 - s) / 2
                self._ax = self.fig.add_axes([d, d, s, s], polar=True)
                self.create_radar_plot(self._ax)
        return self._ax

    @ax.setter
    def ax(self, ax: Axes):
        raise RuntimeError("Cannot set ax directly")

    @property
    def fg(self) -> tuple[float, float, float]:
        return tuple((BLACK if self.theme == "light" else WHITE)[::-1])

    @property
    def bg(self) -> tuple[float, float, float]:
        return (WHITE if self.theme == "light" else BLACK)[::-1]

    @property
    def dim(self) -> float:
        return 0.8 if self.theme == "light" else 1.0

    @property
    def rc_context(self):
        c = "black" if self.theme == "light" else "white"
        K = (
            "text.color",
            "axes.edgecolor",
            "axes.labelcolor",
            "xtick.color",
        )
        return plt.rc_context({k: c for k in K})

    x_majors: np.ndarray = field(default_factory=lambda: np.arange(0, 360, 30))
    x_minors: np.ndarray = field(default_factory=lambda: np.arange(0, 360, 5))

    def create_radar_plot(self, ax: Axes):
        # Create a radar plot and apply styling on it, returns the figure and axis
        # Zero degrees at the top
        ax.set_theta_offset(np.pi / 2 - self.north * DEG2RAD)
        # Counter-clockwise
        ax.set_theta_direction(1)
        ax.set_ylim(*self.y_range)
        if self.title is not None:
            ax.set_title(self.title)
        # Remove grid lines, add clock-looking ticks
        ax.xaxis.grid(False)
        x_labels = list(map(lambda x: f"{int(x)}°", self.x_majors))
        ax.set_xticks(np.deg2rad(self.x_majors), x_labels)
        self.relax_labels()
        x_minors = np.setdiff1d(self.x_minors, self.x_majors)
        ax.set_xticks(np.deg2rad(x_minors), minor=True)
        ax.xaxis.set_tick_params("both", direction="out")
        ax.xaxis.set_tick_params("major", width=1.0, length=4)
        ax.xaxis.set_tick_params("minor", width=0.6, length=3)
        ax.xaxis.tick_top()
        ax.tick_params(labeltop=False, labelbottom=True)
        ax.set_yticks([])
        # Disable x axis line
        ax.spines["polar"].set_visible(False)
        # ax.spines["polar"].set_alpha(0.5)
        # ax.spines["polar"].set_linewidth(0.8)
        # ax.spines["polar"].set_color(self.fg)
        self.circle(
            1.0,
            self.fg,
            linestyle="solid",
            linewidth=0.8,
        )

    def relax_labels(self):
        for label, x in zip(self.ax.get_xticklabels(), np.deg2rad(self.x_majors)):
            label.set_ha("center")
            label.set_va("center")
            pos = x - self.north * DEG2RAD
            offset_y = (abs(math.sin(pos)) - 0.5) / 20.0
            label.set_position((0, -offset_y))
            label.set_fontsize(8)

    def circle(self, r: float, color: list[float], **kwargs):
        kw = dict(
            edgecolor=color,
            facecolor="none",
            linestyle="solid",
        )
        kw.update(kwargs)
        self.fig.add_artist(
            Circle(
                (0.5, 0.5),
                r * self.content_ratio / 2,
                # position and radius in figure coordinates
                transform=self.fig.transFigure,
                **kw,
            )
        )

    def __enter__(self):
        _ = self.fig
        return self

    def __exit__(self, *args):
        if self._fig is not None:
            plt.close(self._fig)

    tmp_renderers: list[Callable[["RenderContext"], list[Artist]]] = field(
        default_factory=list
    )

    @contextmanager
    def head_to(self, hdg: float):
        self.north = hdg
        self.ax.set_theta_offset(np.pi / 2 - self.north * DEG2RAD)
        self.relax_labels()
        elements = list[Artist]()
        for renderer in self.tmp_renderers:
            el = renderer(self)
            elements.extend(el)
        yield self
        for e in elements:
            try:
                e.remove()
            except:
                log.warn("Failed to remove", e.__class__.__name__)


class LookAroundDatabase:

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.data = list[list[float]]()
        self.attrs = dict()

    @property
    def filename(self):
        return self.name.strip().replace(" ", "_")

    @property
    def initial_rz(self):
        return self.get("initial_rz", self.data[0][0])

    def add(self, *records: str | list[float] | dict):
        for record in records:
            try:
                if isinstance(record, list):
                    self.data.append(record)
                elif isinstance(record, dict):
                    self.attrs.update(record)
                else:
                    assert type(record) is str, f"Bad record type: {type(record)}"
                    line = record.strip()
                    if line.startswith("@"):
                        key, value = line[1:].split(maxsplit=1)
                        assert len(key) > 0, f"Bad key: {key}"
                        if key in self.attrs:
                            log.warn(f"Duplicate attribute: {key}, overwriting.")
                        self.attrs[key] = loads(value)
                    else:
                        self.data.append(list(map(float, line.split(","))))
            except Exception as e:
                log.error(f"Bad line: {line} ({e})")
        return self

    def get(self, key: str, default=None):
        if key not in self.attrs:
            self.attrs[key] = default
        return self.attrs[key]

    def process(self):
        cube = np.array(self.data, dtype=np.float32)
        cube[:, 0] = cube[:, 0] % 360.0  # Normalize headings
        assert np.all(cube[:, 0] >= 0.0), f"Bad heading: {cube[:, 0]}"
        cube = cube[cube[:, 0].argsort(), :]
        hdg, nav, fam, ins, raw = cube.T[:5]
        ins = raw  # 4th col (`ins`) reserved for instruction compliance score
        # X variants
        X = circular(hdg * DEG2RAD)
        dX = circular(np.diff(X, prepend=X[-1], append=X[0]))
        assert np.all(dX >= 0.0), f"Bad heading order: {dX}"
        assert dX[0] == dX[-1], f"Bad continuity: {dX}"
        Xw = np.convolve(dX, [0.5, 0.5], "valid")  # window size of each sample
        Xc = X + np.diff(dX) / 4  # center of each sample in their range
        # Gaussian sample x coordinates - evenly spaced (per 1 degree)
        gX, gY, kws = gaussian(Xc, raw, Xw, **self.attrs)
        self.attrs.update(kws)
        # Selection of candidate headings
        if "candidates" not in self.attrs:
            candidates = window_scan(gX, gY)
            if "neg_window" in self.attrs:
                # Reconsider candidates given initial_rz and neg_window
                x0 = self.initial_rz * DEG2RAD
                x, y = np.array(candidates).T
                x *= DEG2RAD
                dx = self.attrs.get("neg_window", 90.0) / 2 * DEG2RAD

                def k_factors(x: np.ndarray, sigma: float) -> np.ndarray:
                    fx = np.power(x / sigma, 3.0)
                    return 1.0 - np.exp(-np.abs(fx) / 2)

                ky = k_factors(circular(x - x0), dx)
                candidates = list(zip(x / DEG2RAD, y * ky))
                if "debug_neg_window" in self.attrs:
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

                    for _x, _y in zip(gX, gY):
                        c = "r-" if _y < 0 else "b-"
                        ax1.plot([_x, _x], [0, _y], c, linewidth=0.5)
                    ax1.plot(gX, gY, "k-", linewidth=0.5)
                    ax1.plot(X, raw, "k.", markersize=4)
                    ax1.axhline(0, color="black", linestyle="--", linewidth=0.5)

                    _x = np.linspace(-np.pi, np.pi, 360)
                    _y = k_factors(circular(_x - x0), dx)
                    ax2.plot(_x, _y, "r-", linewidth=0.8)
                    ax2.plot(x, ky, "b.", markersize=4)

                    _w = 10 * np.pi / 180.0
                    ax3.bar(circular(x) - _w / 2, y, color="blue", width=_w)
                    ax3.bar(circular(x) + _w / 2, y * ky, color="orange", width=_w)
                    _w = circular(x0 - dx), circular(x0 + dx)
                    for ax in (ax1, ax2, ax3):
                        ax.set_xlim(-np.pi, np.pi)
                        ax.vlines(_w, *ax.get_ylim(), "black", "dashed", linewidth=0.75)
                        ax.vlines(x0, *ax.get_ylim(), "black", "dotted", linewidth=0.5)
                    fig.savefig(self.name.replace(" ", "_") + "_Debug.png", dpi=300)
            candidates.sort(key=lambda x: x[1], reverse=True)
            self.attrs["candidates"] = candidates
        # Return intermediate results
        return (X, Xw, Xc), (nav, fam, ins, raw), (gX, gY)

    def has_candidates(self):
        return "candidates" in self.attrs and len(self.attrs["candidates"]) > 0

    def next_candidate(self):
        assert self.has_candidates(), "No candidates available"
        return self.attrs["candidates"].pop(0)

    def __iter__(self):
        """
        Produces a CSV-like output with header, data and attribute rows.
        Can be used with file.writelines() to save to a file.
        """
        t_head = "#   hdg   |   nav   |   fam   |   raw   |   cnf   |"
        # Triple equals indicates the start of a new database
        banner = center(self.name, len(t_head), "=", " ")
        if not banner.startswith("=" * 3):
            # Ensure triple equals always present at the start
            banner = "=" * 3 + banner + "=" * 3
        yield banner + "\n"
        yield t_head + "\n"
        for row in self.data:
            yield " " + ",".join(f"{float(x):.3f} ".rjust(9) for x in row) + "\n"
        for k, v in self.attrs.items():
            yield f"@{k} {dumps(v)}" + "\n"

    _renderer: Callable[[RenderContext], RenderContext] | None = None

    def render(self, ctx: RenderContext = RenderContext()) -> RenderContext:
        if self._renderer is not None:
            return self._renderer(ctx)
        initial_rz = self.initial_rz * DEG2RAD
        # Well-known values, should not be modified or overridden
        (X, Xw, Xc), (nav, fam, ins, raw), (gX, gY) = self.process()
        # Prepare X ranges for radar plot
        S = 1.0 * np.pi / 180.0
        outer_ring_widths = np.maximum(Xw - S / 2, Xw * 0.4)
        outer_ring_bottoms = np.arange(-0.2, 1.0, 0.3)
        outer_ring_colors = [
            alpha(c[::-1]) for c in (GREEN * 0.8, BLUE, MAGENTA, ORANGE)
        ]
        outer_ring_vals = list[np.ndarray]()
        for arr in (nav, fam, ins, raw):
            if arr is fam:
                arr: np.ndarray = fam - np.mean(fam)
            else:
                arr: np.ndarray = arr.copy()
            if arr.max() > 0:
                arr[arr > 0] /= arr[arr > 0].max()
            if arr.min() < 0:
                arr[arr < 0] /= -arr[arr < 0].min()
            outer_ring_vals.append(arr)

        def plot_outer_rings(ctx: RenderContext):
            for i, (x, w) in enumerate(zip(Xc, outer_ring_widths)):
                cm = alpha(ctx.fg, 0.0)
                c2 = alpha(ctx.fg, 0.2)
                for y, bottom, c1 in zip(
                    outer_ring_vals,
                    outer_ring_bottoms,
                    outer_ring_colors,
                ):
                    bar = ctx.ax.bar(x, 0.2, w, bottom, align="center")[0]
                    *c, a = colorize(y[i], c1, c2, cm)
                    bar.set_facecolor(c)
                    bar.set_alpha(a)

        # Final score (with Gaussian smoothing)
        R = (-1.4, -0.3)
        x = list(X) + [X[0]]
        Y, neutral, _ = fill_range(raw, *R)
        y = list(Y) + [Y[0]]
        final_score_pts = (x, y)
        # Plot gX, gY
        gy, _, s = fill_range(gY, *R, neutral)
        # Values used for colorization
        gc = gY.copy()
        gc[gc > 0] /= gc[gc > 0].max()
        gc[gc < 0] /= -gc[gc < 0].min()
        gaussian_curve_bars = list[float](zip(gX, gy, gc))
        gaussian_curve_pts = list(gX) + [gX[0]], list(gy) + [gy[0]]

        def plot_core_gaussian(ctx: RenderContext):
            ctx.ax.plot(
                *final_score_pts, marker=".", markersize=2, linestyle="", color=ctx.fg
            )
            width = 0.6 * 2 * np.pi / len(gX)
            k = 1.0 if ctx.theme == "light" else 0.8
            C1 = alpha(BLUE[::-1] * k + WHITE * (1 - k))
            C2 = alpha(RED[::-1] * k + WHITE * (1 - k))
            CM = alpha(ctx.bg, 0.0)
            for x, y, c in gaussian_curve_bars:
                # Background - transparent black bar
                bg_bar = ctx.ax.bar(x, R[1] - R[0], width, R[0], align="center")[0]
                bg_bar.set_facecolor(ctx.fg)
                bg_bar.set_alpha(0.1)
                # Foreground - colorized bar
                bottom = min(y, neutral)
                height = abs(y - neutral)
                fb_bar = ctx.ax.bar(x, height, width, bottom, align="center")[0]
                *_c, a = colorize(c, C1, C2, CM)
                fb_bar.set_facecolor(_c)
            ctx.ax.plot(
                *gaussian_curve_pts, color=ctx.fg, linestyle="solid", linewidth=0.25
            )

            # Neutral indicator ring
            # ctx.ax.axhline(neutral, linestyle="dashed", color=ctx.fg, linewidth=0.75)
            ctx.circle(
                (neutral - ctx.y_range[0]) / (ctx.y_range[1] - ctx.y_range[0]),
                ctx.fg,
                linestyle="dashed",
                linewidth=0.75,
            )

        def plot_neutral_text(ctx: RenderContext):
            return [
                ctx.ax.text(
                    ctx.north * DEG2RAD,
                    neutral + 0.1,
                    "Neutral",
                    size=8,
                    color=ctx.fg,
                    **CENTER,
                )
            ]

        # Heading candidate indicators
        def plot_candidate_headings(ctx: RenderContext):
            if "candidates" in self.attrs:
                for i, (hdg, conf) in enumerate(self.attrs["candidates"]):
                    x = circular(hdg * DEG2RAD)
                    ctx.ax.plot(
                        (x, x),
                        (neutral, 1.5),
                        color=ctx.fg,
                        linestyle="dashed",
                        clip_on=False,
                        zorder=100,
                        linewidth=0.6 if i > 0 else 1.0,
                    )
                    ctx.ax.text(x, 1.7, f"C{i}", color=ctx.fg, **CENTER)
            # Draw initial heading as black line
            ctx.ax.plot(
                (initial_rz, initial_rz),
                (neutral, 1.5),
                linestyle="solid",
                color=ctx.fg,
                clip_on=False,
                zorder=100,
                linewidth=1.0,
                alpha=0.5,
            )

        def plot_start_text(ctx: RenderContext):
            pos = initial_rz - ctx.north * DEG2RAD
            offset = abs(math.sin(pos)) / 10.0
            t = ctx.ax.text(
                initial_rz, 1.7 + offset, "Start", color=ctx.fg, **CENTER, alpha=0.5
            )
            return [t]

        def plot_turn_progress(ctx: RenderContext):
            if ctx.turn_dir is None:
                return []
            r0 = initial_rz
            r1 = ctx.north * DEG2RAD
            if ctx.turn_dir == "R":
                r0, r1 = r1, r0
            dr = np.mod(r1 - r0, np.pi * 2)
            assert 0 <= dr <= np.pi * 2, f"Bad turn progress: {dr}"
            bars = ctx.ax.bar(
                (r0, r1),
                0.1,
                (dr, np.pi * 2 - dr),
                ctx.y_range[1],
                clip_on=False,
                align="edge",
                zorder=-100,
            )
            bars[0].set_facecolor(GREEN[::-1])
            bars[0].set_alpha(0.5)
            bars[1].set_facecolor(ctx.fg[::-1])
            bars[1].set_alpha(0.2)
            return bars

        def plot_center_arrow(ctx: RenderContext):
            """Add a heading arrow at the center of the radar plot"""
            if not ctx.arrow:
                return []
            x = np.array([0, 140, 180, -140]) + ctx.north
            y = np.array([0.6, 0.7, 0.25, 0.7])
            p = project((0, 1), (ctx.y_range[0], R[0]))

            def rep(arr: np.ndarray):
                return np.concatenate([arr, [arr[0]]])

            ply = ctx.ax.fill(x * DEG2RAD, p(y), color=ctx.fg, alpha=0.2)
            l1 = ctx.ax.plot(rep(x) * DEG2RAD, p(rep(y)), color=ctx.fg, linewidth=1.2)
            l2 = ctx.ax.plot(
                [x[0] * DEG2RAD] * 2,
                (p(0.8), ctx.y_range[1] + 0.1),
                linestyle="solid",
                color=ctx.fg,
                alpha=0.8,
                clip_on=False,
                linewidth=0.8,
            )
            l3 = ctx.ax.plot(
                [x[0] * DEG2RAD + np.pi] * 2,
                (p(0.45), ctx.y_range[1] + 0.1),
                linestyle="dashed",
                color=ctx.fg,
                alpha=0.5,
                clip_on=False,
                linewidth=0.8,
            )
            return *ply, *l1, *l2, *l3

        def renderer(ctx: RenderContext = RenderContext()):
            plot_outer_rings(ctx)
            plot_core_gaussian(ctx)
            plot_candidate_headings(ctx)
            ctx.tmp_renderers.append(plot_start_text)
            ctx.tmp_renderers.append(plot_neutral_text)
            ctx.tmp_renderers.append(plot_turn_progress)
            ctx.tmp_renderers.append(plot_center_arrow)
            return ctx

        self._renderer = renderer

        return renderer(ctx)


import multiprocessing as mp
from multiprocessing.pool import AsyncResult
from multiprocessing.managers import ValueProxy
from multiprocessing.synchronize import Lock
from queue import Queue


def render_worker(
    name: str,
    data: list[str],
    turn_dir: Literal["L", "R", None],
    img_files: list[tuple[float, float, str]],  # (ts, heading, filename)
    SRC: Path,
    DST: Path,
    counter: ValueProxy[int],
    counter_lock: Lock,
    log_queue: Queue[tuple[str, str, list[str]]],
):
    from util.math import ang_diff
    from util.geometry import Region
    from util.graphics import TextBox

    Logger.compose = lambda ID, level, msgs, *_, **__: log_queue.put((ID, level, msgs))
    log.info(f"Rendering {name}")
    db = LookAroundDatabase(name)
    db.add(*data).process()
    # Save PDF figure for paper writing
    with RenderContext(theme="light") as ctx:
        db.render(ctx)
        with ctx.head_to(0):
            ctx.fig.savefig(DST.parent / f"{db.filename}.pdf", transparent=True)
        with counter_lock:
            counter.value += 1

    if len(img_files) == 0:
        if turn_dir is not None:
            log.error(f"No image listed for {name} (turn dir = {turn_dir})")
        return
    elif turn_dir is None:
        log.error(f"Turn direction not specified for {name}")
        return

    with RenderContext(theme="dark", arrow=True, turn_dir=turn_dir) as ctx:
        db.render(ctx)
        last_hdg = db.initial_rz
        dpi: float | None = None
        t_box: TextBox | None = None
        for _, hdg, filename in img_files:
            dr = ang_diff(last_hdg, hdg)
            if (dr < 0 and turn_dir == "L") or (dr > 0 and turn_dir == "R"):
                # Prevent turning figure in the wrong direction
                hdg = last_hdg
            else:
                last_hdg = hdg
            video_frame = cv2.imread(str(SRC / filename), cv2.IMREAD_UNCHANGED)
            h, w, *_ = video_frame.shape
            if dpi is None:
                # Desired canvas output size
                size = min(h, w) * 0.875
                # Calculate DPI based on canvas size and output size
                dpi = int(round(size / CANVAS_SIZE))
            if t_box is None:
                t_box = TextBox(
                    Region(0, 0, w, int(round(h / 16))),
                    align="center",
                    vertical_align="middle",
                    scale=1.0,
                    thickness=1.0,
                )
            with ctx.head_to(hdg):
                ctx.fig.dpi = dpi
                ctx.fig.patch.set_alpha(0.0)
                ctx.ax.patch.set_alpha(0.0)
                ctx.fig.canvas.draw()
                frame = np.array(ctx.fig.canvas.buffer_rgba())
            frame = frame.astype(np.float32) / 255.0
            fig, mask = frame[..., :3], frame[..., 3:]
            video_frame = video_frame.astype(np.float32) / 255.0
            r = Region(w / 2, h / 2, *mask.shape[:2], anchor="center")
            video_frame[r.slice_y, r.slice_x] = r(video_frame) * (1 - mask) + fig * mask
            video_frame[t_box.box.slice_y, t_box.box.slice_x] = (
                t_box.box(video_frame) * 0.6
            )
            video_frame = (video_frame * 255.0).astype(np.uint8)
            t_box(video_frame, name, color=(255, 255, 255))
            cv2.imwrite(str(DST / filename), video_frame)
            with counter_lock:
                counter.value += 1


def debug_mp(*args):
    msg = " ".join(type(a).__name__ for a in args)
    return msg


if __name__ == "__main__":
    import cv2, matplotlib, os, time
    from subprocess import Popen
    from argparse import ArgumentParser
    from tqdm import tqdm
    from ctypes import c_int

    matplotlib.use("Agg")

    parser = ArgumentParser(description="Generate radar plots from look around data")
    parser.add_argument("cwd", type=str, help="Current working directory")
    parser.add_argument("--src", help="Dir name of raw images", default="recording")
    parser.add_argument("--dst", help="Output Directory", default="look_around")
    args = parser.parse_args()
    CWD = Path(os.path.realpath(str(args.cwd)))
    SRC = CWD / str(args.src)
    DST = CWD / str(args.dst)
    DST.mkdir(exist_ok=True, parents=True)
    images_list = CWD / "images.list"
    odometry_list = CWD / "odometry.list"
    look_around_list = CWD / "look_around.list"
    image_candidates = set[tuple[float, float, str]]()  # (ts, heading, filename)

    if odometry_list.exists() and images_list.exists():
        from .matcher import Matcher
        from ..threads import protocol

        M = Matcher(odometry_list.open("rt"), protocol=protocol.Odometry)
        with images_list.open() as f:
            for line in f:
                try:
                    ts, filename = line.split(maxsplit=1)
                    ts = float(ts)
                except:
                    continue
                _, line = M(ts)
                (_, (tx, ty, rz)), *_ = protocol.Odometry.decode(line)
                image_candidates.add((ts, rz, filename.strip()))

    if not look_around_list.exists():
        log.error(f"File not found: {look_around_list}")
        sys.exit(1)

    def process(lines: Iterable[str]):
        record: tuple[str, list[str]] | None = None
        for line in lines:
            if line.startswith("#"):
                pass
            elif line.startswith("=" * 3):
                # Start new record
                title = line.strip("= \n")
                if record is not None:
                    yield record
                record = (title, [])
            elif record is not None:
                record[1].append(line)
            else:
                log.warn("Unexpected line:", line)
        if record is not None:
            yield record

    tasks = list[
        tuple[str, list[str], Literal["L", "R", None], list[tuple[float, float, str]]]
    ]()

    image_list = list[tuple[float, Path, bool]]()

    with look_around_list.open() as f:
        for name, data in process(f):
            # Peek into the database to identify frames corresponding to it
            db = LookAroundDatabase(name).add(*data)
            turn_dir = db.attrs.get("direction", None)
            t0, t1 = db.attrs.get("time", (0.0, 0.0))
            imgs = set[tuple[float, float, str]]()
            if turn_dir is not None:
                for item in image_candidates:
                    ts, rz, filename = item
                    if t0 <= ts <= t1:
                        imgs.add(item)
                        image_list.append((ts, DST / filename, True))
                image_candidates.difference_update(imgs)
            tasks.append((name, data, turn_dir, list(imgs)))

    with mp.Manager() as manager:
        counter_lock = manager.Lock()
        counter = manager.Value(c_int, 0)
        log_queue = manager.Queue()
        with tqdm(
            total=sum(len(l) for n, d, t, l, *_ in tasks) + len(tasks),
            desc="Rendering look around",
            unit="frames",
            leave=False,
            dynamic_ncols=True,
        ) as progress, Expect(KeyboardInterrupt), Logger.use(progress.write), mp.Pool(
            os.cpu_count()
        ) as pool:
            results: list[AsyncResult] = [
                pool.apply_async(
                    func=render_worker,
                    args=(*t, SRC, DST, counter, counter_lock, log_queue),
                )
                for t in tasks
            ]
            while any(not r.ready() for r in results):
                while True:
                    try:
                        el = log_queue.get_nowait()
                    except:
                        break
                    if isinstance(el, tuple) and len(el) == 3:
                        ID, lv, msgs = el
                        lv = lv.lower()
                        if hasattr(log, lv):
                            getattr(log, lv.lower())(*msgs, ID=ID)
                        else:
                            Logger.create(ID, lv)(*msgs)
                    else:
                        log.warn("Bad log entry:", el)
                with counter_lock:
                    progress.n = counter.value
                progress.refresh()
                time.sleep(0.001)
            for (name, *_), r in zip(tasks, results):
                try:
                    r.get()
                except Exception as e:
                    log.error(f"Task {name} Error:", e)

    # Convert into video
    if len(image_list) == 0:
        log.warn("Nothing to render")
        sys.exit(0)
    image_list.sort(key=lambda x: x[0])
    black_frame = np.zeros_like(cv2.imread(str(DST / image_list[0][1])))
    blk = DST / "black.png"
    assert not blk.exists(), f"Black frame already exists: {blk}"
    cv2.imwrite(str(blk), black_frame)

    for ts, _, _ in image_candidates:
        # Invalid frames that are not rendered
        image_list.append((ts, blk, False))

    # Merge consecutive invalid frames
    ts, path, valid = image_list[0]
    lst: list[tuple[float, Path]] = [(ts, path)]
    for (_, _, v1), (ts, path, v2) in zip(image_list, image_list[1:]):
        if not v1 and not v2:
            continue
        lst.append((ts, path))

    ts, path, valid = image_list[-1]
    lst.append((ts, path))

    # Generate ff-concat file
    FF_CONCAT = DST / "concat.txt"
    with FF_CONCAT.open("wt") as f:
        for (t0, path), (t1, _) in zip(lst, lst[1:]):
            try:
                filename = path.relative_to(DST)
            except:
                filename = path
            duration = t1 - t0
            f.write(f"file '{filename}'\n")
            f.write(f"duration {duration}\n")

    from .ffmpeg import FFMPEG

    # Generate video
    try:
        ffmpeg = FFMPEG(FF_CONCAT, CWD / "look_around.mp4")
        ff_log = Logger.create(None, "FFMPEG", "cyan", "light_grey")
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

    # Cleanup
    from util.terminal import confirm
    if confirm("Remove intermediate files?", auto_acc=True):
        for path in tqdm(
            list(DST.glob("*")),
            desc="Removing",
            unit="files",
            leave=False,
            dynamic_ncols=True,
        ):
            path.unlink()
        DST.rmdir()
