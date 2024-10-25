# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from numpy import ndarray, array, ones_like, stack, cbrt as cubic_root
from util.math import clamp, near_zero
from util import types

vt_clamp = clamp(0, 1.0)
vr_clamp = clamp(-1.0, 1.0)


Motion = tuple[float, float, float]


class MotionMixer:
    def __call__(self, correlation: types.Correlation) -> Motion:
        """
        Map predictions to motion commands.
        """
        raise NotImplementedError

    @staticmethod
    def to_numpy(correlation: types.Correlation) -> ndarray:
        """
        Convert correlation to numpy array converter.
        """
        c: list[list[float, float]] = []
        for p in correlation:
            _, n, f = zip(*p)
            c.append(list(zip(n, f)))
        return array(c)


class MotionMixer2x3(MotionMixer):
    def __call__(self, correlation: types.Correlation) -> Motion:
        m = self.to_numpy(correlation).squeeze()
        assert len(m) == 6, f"Bad correlation: {m.shape}"
        m = stack([m[:3], m[3:]], axis=0)
        assert m.shape == (2, 3, 2), f"Bad shape: {m.shape}"
        FAR, NEAR = 0, 1
        nav, fam = m[:, :, 0], m[:, :, 1]
        # 3 elements float vector (l, c, r)
        # Navigability score
        N = 0.2 * nav[FAR] + 0.8 * nav[NEAR]
        # Familiarity score
        F = fam.max(axis=0, keepdims=False)
        # Familiarity multiplier (0.5 ~ 2.0)
        K = 2 * cubic_root(ones_like(F) - F)
        K[K < 0.5] = 0.5
        # Fusion
        l, c, r = N * K
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
    def __call__(self, correlation: types.Correlation) -> Motion:
        raise NotImplementedError
