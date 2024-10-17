from numpy import ndarray

PerceptionStamped = tuple[float, ndarray]

# (label: str, navigability: float, familiarity: float)
Correlation = list[list[tuple[str, float, float]]]
CorrelationStamped = tuple[float, Correlation]

# (vx: float, vy: float, rz: float)
Motion = tuple[float, float, float]
MotionStamped = tuple[float, Motion, str | None]
