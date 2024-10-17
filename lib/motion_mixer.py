# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from numpy import array, ones_like
from util.iter import flatten
from util.math import clamp, near_zero
from util import types

vt_clamp = clamp(0, 1.0)
vr_clamp = clamp(-1.0, 1.0)


class MotionMixer:
    def __call__(self, correlation: types.Correlation) -> types.Motion:
        """
        Map predictions to motion commands.
        """
        raise NotImplementedError


class MotionMixer2x3(MotionMixer):
    def __call__(self, correlation: types.Correlation) -> types.Motion:
        c: list[list[float, float]] = []
        for p in correlation:
            tmp: list[float, float] = []
            for _, n, f in p:
                tmp.append((n, f))
            c.append(tmp)
        # Shape: (1, 6, 2) or (6, 1, 2)
        m = array(c).reshape(2, 3, 2)
        assert m.shape == (2, 3, 2), m.shape
        FAR, NEAR = 0, 1
        nav, fam = m[:, :, 0], m[:, :, 1]
        # 3 elements float vector (l, c, r)
        # Navigability score
        N = 0.2 * nav[FAR] + 0.8 * nav[NEAR]
        # Familiarity score
        F = fam.max(axis=0, keepdims=False)
        F = ones_like(F) - F
        # Fusion
        l, c, r = N * (F**2)
        # Range 0.0 ~ 1.0
        forward = vt_clamp(c * 1.2)
        # Range -1.0 ~ +1.0
        distraction = vr_clamp(l - r)
        if not near_zero(distraction):
            if not near_zero(forward):
                # Turn down distraction term when moving forward
                sweep, turn = [distraction / 4.0] * 2
            else:
                # Turn around in the same spot
                sweep, turn = 0.0, distraction / 4.0
        else:
            sweep, turn = 0.0, 0.0
        # Back off only when both turn and forward are zero
        if near_zero(forward) and near_zero(distraction):
            forward, sweep, turn = -0.2, 0.0, 0.0
        # Publish motion
        return forward, sweep, turn


class MotionMixer1x3(MotionMixer):
    def __call__(self, correlation: types.Correlation) -> types.Motion:
        raise NotImplementedError
