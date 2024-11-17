from torch import Tensor
from numpy import ndarray

# (timestamp: float, perception: N x 512 Tensor | ndarray)
PerceptionStamped = tuple[float, list[float], Tensor | ndarray | None]

# (label: str, navigability: float, familiarity: float, stddev: float)
# Correlation[prompt_id][tile_id]
Correlation = list[list[tuple[str, float, float, float]]]
CorrelationStamped = tuple[float, Correlation]

# (vx: float, vy: float, rz: float)
Motion = tuple[float, float, float]
# (t0, delay, motion, message)
MotionStamped = tuple[float, float, Motion, str | None]

# (tx: float, ty: float, rz: float)
Odometry = Motion
# (t0, odometry)
OdometryStamped = tuple[float, Odometry]
