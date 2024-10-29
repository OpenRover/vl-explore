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
from util.math import clamp, near_zero, project
from util import types

vt_clamp = clamp(0, 1.0)
vr_clamp = clamp(-1.0, 1.0)

# The faster the velocity forward, the slower the turning rate
k_r = project((0.2, 1.0), (0.8, 0.4), clamp=True)

Motion = tuple[float, float, float]


class MotionMixer:
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
        m = self.roll(self.to_numpy(correlation).squeeze())
        assert len(m) == 6, f"Bad correlation: {m.shape}"
        m = stack([m[:3], m[3:]], axis=0)
        assert m.shape == (2, 3, 3), f"Bad shape: {m.shape}"
        FAR, NEAR = 0, 1
        nav, fam, std = m[:, :, 0], m[:, :, 1], m[:, :, 2]
        # No positive navigability where stddev is too low (no information)
        nav[logical_and(nav > 0, std < 0.1)] = 0.0
        # 3 elements float vector (l, c, r)
        # Navigability score
        N = nav[NEAR]
        # Consider FAR only when NEAR is clear
        selection = logical_and(N > 0, nav[FAR] > 0)
        N[selection] += nav[FAR][selection]
        # Familiarity score (FAR matters more)
        F = fam[FAR] * 0.8 + fam[NEAR] * 0.2
        # Familiarity multiplier (0.5 ~ 2.0)
        K = 2 * cubic_root(ones_like(F) - F)
        K[K < 0.5] = 0.5
        # Fusion
        l, c, r = N * K
        # Range 0.0 ~ 1.0
        forward = vt_clamp(c * 1.2)
        # Range -1.0 ~ +1.0
        distraction = vr_clamp(l - r)
        if near_zero(forward):
            forward = 0.0
            # Special case: l and r might both be positive
            if near_zero(distraction):
                if l > 0 and r > 0:
                    # choose a better side to turn
                    distraction = l if l >= r else -r
                else:
                    # Nowhere to go, back-up slowly
                    return -0.2, 0.0, 0.0
            # Turn around in the same spot
            sweep, turn = 0.0, distraction / 2.0
        else:
            forward = max(0.2, forward)
            # Forward motion in progress
            if not near_zero(distraction):
                k = k_r(forward)
                # Turn down distraction term when moving forward
                sweep = distraction / 3.0
                turn = k * distraction / 3.0
            else:
                sweep, turn = 0.0, 0.0
        # Publish motion
        return forward, sweep, turn


class MotionMixer1x3(MotionMixer):
    def __call__(self, correlation: types.Correlation) -> Motion:
        raise NotImplementedError
