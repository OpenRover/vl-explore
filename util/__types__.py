from torch import Tensor
from numpy import ndarray

PerceptionStamped = tuple[float, Tensor | ndarray]

# (label: str, navigability: float, familiarity: float)
Correlation = list[list[tuple[str, float, float]]]
CorrelationStamped = tuple[float, Correlation]

# (vx: float, vy: float, rz: float)
Motion = tuple[float, float, float]
# (t0, delay, motion, message)
MotionStamped = tuple[float, float, Motion, str | None]
