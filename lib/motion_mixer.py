# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from numpy import (
    ndarray,
    array,
    ones_like,
    zeros_like,
    stack,
    cbrt as cubic_root,
    logical_and,
)
import numpy as np
from util.math import clamp, near_zero, project
from util import types

vt_clamp = clamp(0, 1.0)
vr_clamp = clamp(-1.0, 1.0)

# The faster the velocity forward, the slower the turning rate
k_r = project((0.2, 1.0), (0.8, 0.4), clamp=True)

Motion = tuple[float, float, float]


def lcr_mixer(l: float, c: float, r: float) -> Motion:
    # Range 0.0 ~ 1.0
    forward = vt_clamp(c * 1.2)
    # Range -1.0 ~ +1.0
    distraction = vr_clamp(l - r)
    if forward <= 0:
        # forward = 0.0
        # Special case: l and r might both be positive
        if near_zero(distraction, 0.1):
            if l <= 0 and r <= 0:
                # Nowhere to go, back-up slowly
                forward = min(forward, -0.2)
                # sweep, turn = 0.0, 0.0
            elif l > 0 and r > 0:
                # choose a better side to turn
                rate = max(l, r, 0.2)
                distraction = rate if l > r else -rate
        # Turn around in the same spot
        sweep, turn = 0.0, distraction * 0.4
    else:
        # Forward motion in progress
        forward = max(0.2, forward)
        if not near_zero(distraction, 0.1):
            k = k_r(forward)
            # Turn down distraction term when moving forward
            sweep = k * distraction * 0.1
            turn = k * distraction * 0.2
        else:
            sweep, turn = 0.0, 0.0
    # Publish motion
    return forward, sweep, turn


class MotionMixer:
    state: str | None = None

    def __call__(self, correlation: types.Correlation) -> Motion:
        """
        Map predictions to motion commands.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset internal state.
        """
        pass

    @staticmethod
    def to_numpy(correlation: types.Correlation) -> ndarray:
        """
        Convert correlation to numpy array converter.
        """
        c: list[list[float, float]] = []
        for p in correlation:
            _, *data = zip(*p)
            c.append(list(zip(*data)))
        return array(c)

    decay: float = 0.0
    rolling: ndarray = None

    def roll(self, m: ndarray) -> ndarray:
        """
        Perform one iteration of rolling average.
        """
        if self.rolling is None:
            self.rolling = zeros_like(m)
        else:
            d = self.decay
            self.rolling = self.rolling * d + m * (1 - d)
        return self.rolling


class MotionMixer2x3(MotionMixer):

    def reset(self):
        self.rolling = None

    def __call__(self, correlation: types.Correlation) -> Motion:
        self.state = None
        m = self.to_numpy(correlation[:-1]).squeeze()
        assert m.shape == (6, 3), f"Bad correlation: {m.shape}"
        t = self.to_numpy(correlation[-1:]).squeeze() * 2.0
        assert t.shape == (6, 3), f"Bad target: {t.shape}"
        # Special mode when target is identified
        t = t.reshape(2, 3, 3)[:, :, 0]
        if np.any(t > 0.1):
            self.state = "Target Identified"
            l, c, r = map(bool, np.max(t, axis=0) > 0)
            match int(l << 2 | c << 1 | r):
                case 0b010:
                    # Strong forward
                    return 0.2, 0.0, 0.0
                case 0b110:
                    # Weak turn left
                    return 0.1, 0.0, +0.1
                case 0b100:
                    # Strong turn left
                    return 0.0, 0.0, +0.2
                case 0b011:
                    # Weak turn right
                    return 0.1, 0.0, -0.1
                case 0b001:
                    # Strong turn right
                    return 0.0, 0.0, -0.2
                case _:
                    self.state = None
                    pass # Use normal mode
        # Normal navigation mode
        m = m.reshape(2, 3, 3)
        assert m.shape == (2, 3, 3), f"Bad shape: {m.shape}"
        FAR, NEAR = 0, 1
        nav, fam, std = m[:, :, 0], m[:, :, 1], m[:, :, 2]

        # No positive navigability where stddev is too low (no information)
        # update: this is now done in correlator
        # nav[logical_and(nav > 0, std < 0.1)] = 0.0

        # Consider FAR only when NEAR is clear
        selection = logical_and(nav[NEAR] > 0, nav[FAR] > 0)
        nav[NEAR][selection] += nav[FAR][selection]
        # Navigability score
        N = nav[NEAR]
        # Familiarity score (FAR matters more)
        F = fam[FAR] * 0.8 + fam[NEAR] * 0.2
        # Familiarity multiplier (0.5 ~ 1.5)
        K = np.max(F) - F
        if np.max(K) != 0:
            K /= np.max(K)
        K += 0.5
        # 3 elements float vector (l, c, r)
        l, c, r = N * K
        # Fusion
        return lcr_mixer(l, c, r)


class MotionMixer1x3(MotionMixer):
    def __call__(self, correlation: types.Correlation) -> Motion:
        raise NotImplementedError
